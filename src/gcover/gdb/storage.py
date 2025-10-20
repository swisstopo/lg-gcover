#!/usr/bin/env python3

import base64
import hashlib
import hmac
import os
import re
import struct
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import duckdb
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from gcover.config.models import ProxyConfig

from .assets import (
    AssetType,
    BackupGDBAsset,
    GDBAsset,
    GDBAssetInfo,
    IncrementGDBAsset,
    VerificationGDBAsset,
)


@dataclass
class UploadResult:
    """Result of an upload operation"""

    success: bool
    status_code: int
    error_message: Optional[str] = None
    s3_key: Optional[str] = None
    method: Optional[str] = None  # 'presigned', 'direct', or 'fallback'

    @property
    def is_client_error(self) -> bool:
        """Check if error is client-side (4xx)"""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if error is server-side (5xx)"""
        return 500 <= self.status_code < 600

    @property
    def file_exists(self) -> bool:
        """Check if failure was due to file already existing"""
        return self.status_code == 409


class TOTPGenerator:
    """Generate TOTP tokens for authentication"""

    @staticmethod
    def generate_totp(secret: str, time_step: int = 30, digits: int = 6) -> str:
        """
        Generate TOTP token from secret

        Args:
            secret: Base32 encoded secret
            time_step: Time step in seconds (default 30)
            digits: Number of digits in token (default 6)

        Returns:
            TOTP token string
        """
        key = base64.b32decode(secret.upper())
        counter = int(time.time() // time_step)
        counter_bytes = struct.pack(">Q", counter)
        hmac_digest = hmac.new(key, counter_bytes, hashlib.sha1).digest()
        offset = hmac_digest[-1] & 0x0F
        truncated = struct.unpack(">I", hmac_digest[offset : offset + 4])[0]
        truncated &= 0x7FFFFFFF
        token = truncated % (10**digits)
        return str(token).zfill(digits)


class S3Uploader:
    """Enhanced S3 uploader with presigned URL support"""

    def __init__(
        self,
        bucket_name: str,
        aws_profile: Optional[str] = None,
        lambda_endpoint: Optional[str] = None,
        totp_secret: Optional[str] = None,
        totp_token: Optional[str] = None,
        proxy_config: Optional[ProxyConfig] = None,
        upload_method: str = "auto",
        show_progress: bool = True,
        progress_threshold: int = 10 * 1024 * 1024,  # 10 MB
    ):
        """
        Initialize S3 Uploader with multiple upload methods

        Args:
            bucket_name: S3 bucket name
            aws_profile: AWS profile name (for direct upload)
            lambda_endpoint: Lambda endpoint URL for presigned URLs
            totp_secret: TOTP secret for Lambda authentication (alternative to token)
            totp_token: Pre-generated TOTP token for Lambda authentication
            proxy_config: ProxyConfig instance for proxy settings (optional)
            upload_method: 'auto', 'direct', or 'presigned'
            show_progress: Whether to show progress bar for uploads
            progress_threshold: Minimum file size in bytes to show progress (default 10MB)
        """
        self.bucket_name = bucket_name
        self.profile_name = aws_profile
        self.lambda_endpoint = lambda_endpoint
        self.totp_secret = totp_secret
        self.totp_token = totp_token
        self.proxy_config = proxy_config
        self.upload_method = upload_method
        self.show_progress = show_progress
        self.progress_threshold = progress_threshold

        self.proxies = self._init_proxies()
        self.s3_client = None

        if not self.upload_method == "presigned":
            self._init_s3_client()

        self._determine_upload_strategy()

    def __repr__(self):
        method = "presigned" if self.use_presigned else "direct"
        proxy_info = "with_proxy" if self.proxies else "no_proxy"
        return (
            f"<gcover.gdb.storage.S3Uploader: bucket={self.bucket_name}, "
            f"profile={self.profile_name}, method={method}, {proxy_info}>"
        )

    def _init_proxies(self) -> Dict[str, str]:
        """Initialize proxy settings from ProxyConfig"""
        if self.proxy_config is None:
            logger.info("No proxy configuration provided")
            return {}

        proxies = self.proxy_config.to_requests_format()
        if proxies:
            logger.info(f"Using proxy configuration: {proxies}")
        else:
            logger.info("Proxy config provided but no proxies configured")

        return proxies

    def _init_s3_client(self):
        """Initialize boto3 S3 client with proxy support"""
        try:
            config = None

            if self.proxies:
                boto3_proxies = self.proxy_config.to_boto3_format()
                config = Config(proxies=boto3_proxies)
                logger.info(f"Boto3 using proxy configuration: {boto3_proxies}")

            if self.profile_name:
                session = boto3.Session(profile_name=self.profile_name)
                self.s3_client = session.client("s3", config=config)
            else:
                self.s3_client = boto3.client("s3", config=config)

            logger.debug("S3 client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize S3 client: {e}")
            self.s3_client = None

    def _determine_upload_strategy(self):
        """Determine which upload method to use"""
        if self.upload_method == "presigned":
            if not self.lambda_endpoint:
                raise ValueError("Lambda endpoint required for presigned upload method")
            self.use_presigned = True
        elif self.upload_method == "direct":
            if not self.s3_client:
                raise ValueError(
                    "S3 client initialization failed for direct upload method"
                )
            self.use_presigned = False
        else:  # auto
            self.use_presigned = bool(
                self.lambda_endpoint and (self.totp_secret or self.totp_token)
            )

        logger.info(
            f"Upload strategy: {'presigned URLs' if self.use_presigned else 'direct S3'}"
        )

    def _get_totp_token(self) -> Optional[str]:
        """Get TOTP token (either provided or generated)"""
        if self.totp_token:
            return self.totp_token
        elif self.totp_secret:
            return TOTPGenerator.generate_totp(self.totp_secret)
        else:
            return None

    def _get_presigned_url(
        self, s3_key: str, file_size: int, check_exists: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get presigned URL from Lambda endpoint

        Args:
            s3_key: S3 object key
            file_size: File size in bytes
            check_exists: Whether to check if file already exists

        Returns:
            Dict with presigned_url, headers, and status_code, or None if failed
        """
        if not self.lambda_endpoint:
            return None

        totp_token = self._get_totp_token()
        logger.debug(f"TOKEN: {totp_token}")

        if not totp_token:
            logger.error("No TOTP token available for Lambda authentication")
            return None

        try:
            payload = {
                "object_key": s3_key,
                "totp_code": totp_token,
                "expiration_hours": 24,
                "content_type": "application/octet-stream",
                "check_exists": check_exists,
            }

            headers = {
                "Authorization": f"TOTP {totp_token}",
                "Content-Type": "application/json",
            }

            logger.debug(f"Requesting presigned URL for {s3_key}")
            request_args = {
                "json": payload,
                "headers": headers,
                "timeout": 30,
                "verify": False,  # TODO: consider using a CA bundle instead
            }

            if self.proxies:
                request_args["proxies"] = self.proxies
                logger.debug(f"Using proxies for Lambda request: {self.proxies}")

            response = requests.post(self.lambda_endpoint, **request_args)

            # FIXED: Check status code properly
            if response.status_code in [200, 204, 409]:
                data = response.json()
                data["status_code"] = response.status_code
                logger.debug(
                    f"Presigned URL obtained with status {response.status_code}"
                )
                return data
            else:
                logger.error(
                    f"Lambda request failed: {response.status_code} - {response.text}"
                )
                return None

        except requests.RequestException as e:
            logger.error(f"Error requesting presigned URL: {e}")
            return None

    def _upload_with_presigned_url(self, file_path: Path, s3_key: str) -> UploadResult:
        """
        Upload file using presigned URL with progress bar

        Args:
            file_path: Local file path
            s3_key: S3 object key

        Returns:
            UploadResult with status and details
        """
        try:
            file_size = file_path.stat().st_size
            presigned_data = self._get_presigned_url(s3_key, file_size)

            if not presigned_data:
                return UploadResult(
                    success=False,
                    status_code=500,
                    error_message="Could not obtain presigned URL",
                    s3_key=s3_key,
                    method="presigned",
                )

            # Check if file already exists (status 409)
            status_code = presigned_data.get("status_code")
            if status_code == 409:
                logger.info(f"File already exists in s3://{self.bucket_name}/{s3_key}")
                return UploadResult(
                    success=True,
                    status_code=409,
                    error_message="File already exists (skipped)",
                    s3_key=s3_key,
                    method="presigned",
                )

            presigned_url = presigned_data.get("presigned_url")
            if not presigned_url:
                return UploadResult(
                    success=False,
                    status_code=500,
                    error_message="Presigned URL missing in response",
                    s3_key=s3_key,
                    method="presigned",
                )

            logger.debug(f"Uploading with presigned URL")

            # Determine if we should show progress bar
            show_bar = self.show_progress and file_size >= self.progress_threshold

            if show_bar:
                # Upload with progress bar
                response = self._upload_with_progress(
                    file_path,
                    presigned_url,
                    file_size,
                    presigned_data.get("headers", {}),
                )
            else:
                # Upload without progress bar (small files)
                with open(file_path, "rb") as file_obj:
                    request_args = {
                        "data": file_obj,
                        "headers": presigned_data.get("headers", {}),
                        "timeout": 300,
                        "verify": False,
                    }

                    if self.proxies:
                        request_args["proxies"] = self.proxies
                        logger.debug(
                            f"Using proxies for presigned upload: {self.proxies}"
                        )

                    response = requests.put(presigned_url, **request_args)

            success = response.status_code in [200, 204]

            if success:
                logger.info(
                    f"presigned URL - Successfully uploaded {file_path} to "
                    f"s3://{self.bucket_name}/{s3_key}"
                )
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")

            return UploadResult(
                success=success,
                status_code=response.status_code,
                error_message=None if success else response.text,
                s3_key=s3_key,
                method="presigned",
            )

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return UploadResult(
                success=False,
                status_code=404,
                error_message=str(e),
                s3_key=s3_key,
                method="presigned",
            )

        except requests.RequestException as e:
            logger.error(f"Network error during upload: {e}")
            return UploadResult(
                success=False,
                status_code=0,
                error_message=str(e),
                s3_key=s3_key,
                method="presigned",
            )

        except Exception as e:
            logger.error(f"Error uploading with presigned URL: {e}")
            return UploadResult(
                success=False,
                status_code=500,
                error_message=str(e),
                s3_key=s3_key,
                method="presigned",
            )

    def _upload_with_progress(
        self,
        file_path: Path,
        presigned_url: str,
        file_size: int,
        headers: Dict[str, str],
    ) -> requests.Response:
        """
        Upload file with rich progress bar

        Args:
            file_path: Path to file to upload
            presigned_url: Presigned S3 URL
            file_size: Total file size in bytes
            headers: HTTP headers for upload

        Returns:
            requests.Response object
        """

        # Custom file-like object that tracks progress
        class ProgressFileReader:
            def __init__(self, file_obj, progress_obj, task_id, total_size):
                self.file_obj = file_obj
                self.progress = progress_obj
                self.task_id = task_id
                self.total_size = total_size
                self.total_read = 0

            def read(self, size=-1):
                chunk = self.file_obj.read(size)
                if chunk:
                    self.total_read += len(chunk)
                    self.progress.update(self.task_id, completed=self.total_read)
                return chunk

            def __len__(self):
                """Return total size so requests can set Content-Length header"""
                return self.total_size

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.file_obj.close()

        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Uploading {file_path.name}", total=file_size
            )

            with open(file_path, "rb") as f:
                file_reader = ProgressFileReader(f, progress, task, file_size)

                # Add Content-Length header explicitly to prevent chunked encoding
                upload_headers = headers.copy()
                upload_headers["Content-Length"] = str(file_size)

                request_args = {
                    "data": file_reader,
                    "headers": upload_headers,
                    "timeout": 300,
                    "verify": False,
                }

                if self.proxies:
                    request_args["proxies"] = self.proxies

                response = requests.put(presigned_url, **request_args)

        return response

    def _upload_direct(self, file_path: Path, s3_key: str) -> UploadResult:
        """
        Upload file directly using boto3

        Args:
            file_path: Local file path
            s3_key: S3 object key

        Returns:
            UploadResult with status and details
        """
        if not self.s3_client:
            return UploadResult(
                success=False,
                status_code=500,
                error_message="S3 client not available for direct upload",
                s3_key=s3_key,
                method="direct",
            )

        try:
            self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
            logger.info(
                f"Direct upload (boto3) - Successfully uploaded {file_path} to "
                f"s3://{self.bucket_name}/{s3_key}"
            )
            return UploadResult(
                success=True, status_code=200, s3_key=s3_key, method="direct"
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            status_code = e.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode", 500
            )

            logger.error(f"Direct upload (boto3) failed: {error_code} - {e}")

            return UploadResult(
                success=False,
                status_code=status_code,
                error_message=f"{error_code}: {str(e)}",
                s3_key=s3_key,
                method="direct",
            )

    def upload_file(self, file_path: Path, s3_key: str) -> UploadResult:
        """
        Upload file to S3 using configured method

        Args:
            file_path: Local file path
            s3_key: S3 object key

        Returns:
            UploadResult with status and details
        """
        logger.info(f"Uploading {file_path} to s3://{self.bucket_name}/{s3_key}")

        if self.use_presigned:
            result = self._upload_with_presigned_url(file_path, s3_key)

            # Fallback to direct upload if presigned fails and s3_client is available
            if not result.success and self.s3_client:
                logger.warning(
                    "Presigned URL upload failed, falling back to direct upload"
                )
                result = self._upload_direct(file_path, s3_key)
                if result.success:
                    result.method = "fallback"  # Mark as fallback method

            return result
        else:
            return self._upload_direct(file_path, s3_key)

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3

        Args:
            s3_key: S3 object key

        Returns:
            True if file exists, False otherwise
        """
        if self.s3_client:
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                return True
            except ClientError:
                return False
        else:
            # If no S3 client, try using presigned URL check
            if self.lambda_endpoint:
                presigned_data = self._get_presigned_url(s3_key, 0, check_exists=True)
                if presigned_data:
                    return presigned_data.get("status_code") == 409

            logger.warning(
                "Cannot check file existence without S3 client or Lambda endpoint"
            )
            return False

    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download file from S3

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not available for download")
            return False

        try:
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False


def create_s3_uploader_with_proxy(
    bucket_name: str,
    proxy_http: Optional[str] = None,
    proxy_https: Optional[str] = None,
    **kwargs,
) -> S3Uploader:
    """
    Convenience function to create S3Uploader with proxy configuration

    Args:
        bucket_name: S3 bucket name
        proxy_http: HTTP proxy URL (optional)
        proxy_https: HTTPS proxy URL (optional)
        **kwargs: Additional arguments for S3Uploader

    Returns:
        Configured S3Uploader instance
    """
    proxy_config = None
    if proxy_http or proxy_https:
        proxy_config = ProxyConfig(http_proxy=proxy_http, https_proxy=proxy_https)

    return S3Uploader(bucket_name=bucket_name, proxy_config=proxy_config, **kwargs)


class MetadataDB:
    """Handle DuckDB metadata operations"""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.init_db()

    def __str__(self):
        return f"<MetadataDB: db_path: {self.db_path}>"

    def init_db(self):
        """Initialize database schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
            CREATE TABLE IF NOT EXISTS gdb_assets (
                id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
                path VARCHAR NOT NULL,
                asset_type VARCHAR NOT NULL,
                release_candidate VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                file_size BIGINT,
                hash_md5 VARCHAR,
                s3_key VARCHAR,
                uploaded BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            );
            """)

    def insert_asset(self, asset_info: GDBAssetInfo):
        """Insert asset information"""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO gdb_assets 
                (path, asset_type, release_candidate, timestamp, file_size, 
                 hash_md5, s3_key, uploaded, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
                [
                    str(asset_info.path),
                    asset_info.asset_type.value,
                    asset_info.release_candidate.value,
                    asset_info.timestamp,
                    asset_info.file_size,
                    asset_info.hash_md5,
                    asset_info.s3_key,
                    asset_info.uploaded,
                    asset_info.metadata,
                ],
            )

    def asset_exists(self, path: Path) -> bool:
        """Check if asset already exists in database"""
        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM gdb_assets WHERE path = ?", [str(path)]
            ).fetchone()
            return result[0] > 0
