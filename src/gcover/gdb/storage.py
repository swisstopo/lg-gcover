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
from botocore.config import Config


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


class S3Uploader:
    """Handle S3 operations"""

    def __init__(
        self,
        bucket_name: str,
        aws_profile: Optional[str] = None,
        proxy: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.profile_name = aws_profile

        # Build proxy config if provided
        config = None
        if proxy:
            config = Config(proxies={"http": proxy, "https": proxy})

        # Create session with or without profile
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client("s3", config=config)
        else:
            self.s3_client = boto3.client("s3", config=config)

    def upload_file(self, file_path: Path, s3_key: str) -> bool:
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
            logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False

    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def __repr__(self):
        return f"<gcover.gdb.storage.S3Uploader: bucket: {self.bucket_name}, profile: {self.profile_name}>"


class MetadataDB:
    """Handle DuckDB metadata operations"""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.init_db()

    def init_db(self):
        """Initialize database schema"""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS  id_sequence START 1;
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
