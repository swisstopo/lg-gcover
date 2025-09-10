#!/usr/bin/env python3

import hashlib
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import duckdb
from botocore.exceptions import ClientError

# Configure logging
from loguru import logger

from .assets import (
    AssetType,
    BackupGDBAsset,
    GDBAsset,
    GDBAssetInfo,
    IncrementGDBAsset,
    VerificationGDBAsset,
)

from .storage import S3Uploader, MetadataDB

GBD_TO_EXCLUDE = ["progress.gdb", "temp.gdb"]


class GDBAssetManager:
    """Main manager for GDB assets with enhanced S3 upload capabilities"""

    def __init__(
        self,
        base_paths: Dict[str, Path],
        s3_config: Dict[str, Any],
        db_path: Union[str, Path],
        temp_dir: Union[str, Path] = "/tmp/gdb_zips",
    ):
        """
        Initialize GDB Asset Manager with enhanced configuration

        Args:
            base_paths: Dict with keys 'backup', 'verification', 'increment'
            s3_config: S3 configuration dictionary containing:
                - bucket: S3 bucket name
                - profile: AWS profile (optional)
                - lambda_endpoint: Lambda endpoint for presigned URLs (optional)
                - totp_secret: TOTP secret for Lambda auth (optional)
                - totp_token: Pre-generated TOTP token (optional)
                - proxy: Proxy settings (optional)
                - upload_method: 'auto', 'direct', or 'presigned'
            db_path: Path to DuckDB database
            temp_dir: Directory for temporary zip files
        """
        self.base_paths = {k: Path(v) for k, v in base_paths.items()}
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 uploader with enhanced configuration
        logger.debug(f"GDBAssetMananger: init()")
        self.s3_uploader = self._create_s3_uploader(s3_config)
        logger.debug(self.s3_uploader)
        self.metadata_db = MetadataDB(db_path)

        logger.info(f"Initialized {self.s3_uploader}")

    def _create_s3_uploader(self, s3_config: Dict[str, Any]) -> S3Uploader:
        """
        Create S3Uploader instance from configuration

        Args:
            s3_config: S3 configuration dictionary

        Returns:
            Configured S3Uploader instance
        """
        # Extract configuration values with defaults
        bucket_name = s3_config.bucket  # .get('bucket')
        if not bucket_name:
            raise ValueError("S3 bucket name is required")

        aws_profile = s3_config.profile  # .get('profile')
        lambda_endpoint = s3_config.lambda_endpoint  # .get('lambda_endpoint')
        totp_secret = s3_config.totp_secret  # .get('totp_secret')
        totp_token = s3_config.totp_token  # .get('totp_token')
        proxy_settings = s3_config.proxy  # .get('proxy', {})
        upload_method = s3_config.upload_method  # .get('upload_method', 'auto')

        return S3Uploader(
            bucket_name=bucket_name,
            aws_profile=aws_profile,
            lambda_endpoint=lambda_endpoint,
            totp_secret=totp_secret,
            totp_token=totp_token,
            proxy_settings=proxy_settings,
            upload_method=upload_method,
        )

    @classmethod
    def from_config(cls, config_obj, temp_dir: Optional[Union[str, Path]] = None):
        """
        Create GDBAssetManager from configuration object

        Args:
            config_obj: Configuration object with gdb and s3 sections
            temp_dir: Override temp directory

        Returns:
            Configured GDBAssetManager instance
        """
        # Extract base paths from config
        base_paths = {
            "backup": config_obj.gdb.base_paths.backup,
            "verification": config_obj.gdb.base_paths.verification,
            "increment": config_obj.gdb.base_paths.increment,
        }

        # Build S3 configuration
        s3_config = {
            "bucket": config_obj.global_config.s3.bucket,
            "profile": config_obj.global_config.s3.profile,
        }

        # Add optional Lambda/TOTP configuration if present
        if hasattr(config_obj.global_config.s3, "lambda_endpoint"):
            s3_config["lambda_endpoint"] = config_obj.global_config.s3.lambda_endpoint
        if hasattr(config_obj.global_config.s3, "totp_secret"):
            s3_config["totp_secret"] = config_obj.global_config.s3.totp_secret
        if hasattr(config_obj.global_config.s3, "totp_token"):
            s3_config["totp_token"] = config_obj.global_config.s3.totp_token
        if hasattr(config_obj.global_config.s3, "proxy"):
            s3_config["proxy"] = config_obj.global_config.s3.proxy
        if hasattr(config_obj.global_config.s3, "upload_method"):
            s3_config["upload_method"] = config_obj.global_config.s3.upload_method

        # Use temp_dir from parameter, config, or default
        if temp_dir is None:
            temp_dir = getattr(config_obj.gdb, "temp_dir", "/tmp/gcover/gdb")

        return cls(
            base_paths=base_paths,
            s3_config=s3_config,
            db_path=config_obj.gdb.db_path,
            temp_dir=temp_dir,
        )

    def create_asset(self, gdb_path: Path) -> GDBAsset:
        """Factory method to create appropriate asset type"""
        path_str = gdb_path.as_posix()

        if "/GCOVER/" in path_str and gdb_path.name.endswith(".gdb"):
            return BackupGDBAsset(gdb_path)
        elif "/Verifications/" in path_str and gdb_path.name.endswith(".gdb"):
            return VerificationGDBAsset(gdb_path)
        elif "/Increment/" in path_str and gdb_path.name.endswith(".gdb"):
            return IncrementGDBAsset(gdb_path)
        else:
            raise ValueError(f"Cannot determine asset type for: {gdb_path}")

    def scan_filesystem(self) -> List[GDBAsset]:
        """Scan filesystem for GDB assets"""
        assets = []

        for base_name, base_path in self.base_paths.items():
            if not base_path.exists():
                logger.warning(f"Base path does not exist: {base_path}")
                continue

            for gdb_path in base_path.rglob("*.gdb"):
                if gdb_path.is_dir():
                    # Filter out temporary files
                    if gdb_path.name.lower() in GBD_TO_EXCLUDE:
                        continue

                    try:
                        asset = self.create_asset(gdb_path)
                        assets.append(asset)
                        logger.debug(f"Found asset: {gdb_path}")
                    except ValueError as e:
                        logger.warning(f"Skipping {gdb_path}: {e}")

        return assets

    def process_asset(self, asset: GDBAsset) -> bool:
        """Process single asset: zip, hash, upload, update DB"""
        try:
            # Skip if already in database
            if self.metadata_db.asset_exists(asset.path):
                logger.info(f"Asset already processed: {asset.path}")
                return True

            # Create zip
            logger.info(f"Processing asset: {asset.path}")
            zip_path = asset.create_zip(self.temp_dir)

            # Compute hash
            hash_md5 = asset.compute_hash()

            # Generate S3 key
            s3_key = f"gdb-assets/{asset.info.release_candidate.short_name}/{asset.info.asset_type.value}/{zip_path.name}"
            asset.info.s3_key = s3_key

            # Upload to S3
            if not self.s3_uploader.file_exists(s3_key):
                uploaded = self.s3_uploader.upload_file(zip_path, s3_key)
                asset.info.uploaded = uploaded
            else:
                logger.info(f"File already exists in S3: {s3_key}")
                asset.info.uploaded = True

            # Update database
            self.metadata_db.insert_asset(asset.info)

            # Cleanup temp file
            zip_path.unlink()

            logger.info(f"Successfully processed: {asset.path}")
            return True

        except Exception as e:
            logger.error(f"Failed to process {asset.path}: {e}")
            return False

    def sync_all(self) -> Dict[str, int]:
        """Scan filesystem and sync all new assets"""
        logger.info("Starting filesystem scan...")
        assets = self.scan_filesystem()

        stats = {"found": len(assets), "processed": 0, "failed": 0, "skipped": 0}

        for asset in assets:
            if self.process_asset(asset):
                stats["processed"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"Sync complete: {stats}")
        return stats

    def download_asset(self, s3_key: str, local_path: Path) -> bool:
        """
        Download asset from S3

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            True if successful, False otherwise
        """
        return self.s3_uploader.download_file(s3_key, local_path)


# Backward compatibility function for existing code
def create_manager_from_legacy_params(
    base_paths: Dict[str, Path],
    s3_bucket: str,
    db_path: Union[str, Path],
    temp_dir: Union[str, Path] = "/tmp/gdb_zips",
    aws_profile: Optional[str] = None,
) -> GDBAssetManager:
    """
    Create GDBAssetManager using legacy parameter format

    This function provides backward compatibility for existing code.
    New code should use GDBAssetManager.from_config() instead.
    """
    s3_config = {
        "bucket": s3_bucket,
        "profile": aws_profile,
        "upload_method": "direct",  # Legacy behavior
    }

    return GDBAssetManager(
        base_paths=base_paths, s3_config=s3_config, db_path=db_path, temp_dir=temp_dir
    )


# Example usage
if __name__ == "__main__":
    # Modern usage with configuration
    from gcover.gdb.config import load_config

    config_obj = load_config(environment="development")
    manager = GDBAssetManager.from_config(config_obj)

    # Sync all assets
    stats = manager.sync_all()
    print(f"Sync completed: {stats}")
