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

from .assets import (AssetType, BackupGDBAsset, GDBAsset, GDBAssetInfo,
                     IncrementGDBAsset, VerificationGDBAsset)

from .storage import (S3Uploader, MetadataDB)


class GDBAssetManager:
    """Main manager for GDB assets"""

    def __init__(self,
                 base_paths: Dict[str, Path],
                 s3_bucket: str,
                 db_path: Union[str, Path],
                 temp_dir: Union[str, Path] = "/tmp/gdb_zips",
                 aws_profile: Optional[str] = None):
        """
        Initialize GDB Asset Manager

        Args:
            base_paths: Dict with keys 'backup', 'verification', 'increment'
            s3_bucket: S3 bucket name
            db_path: Path to DuckDB database
            temp_dir: Directory for temporary zip files
            aws_profile: AWS profile name (optional)
        """
        self.base_paths = {k: Path(v) for k, v in base_paths.items()}
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.s3_uploader = S3Uploader(s3_bucket, aws_profile)
        self.metadata_db = MetadataDB(db_path)

    def create_asset(self, gdb_path: Path) -> GDBAsset:
        """Factory method to create appropriate asset type"""
        path_str = str(gdb_path)

        if "/GCOVER/" in path_str and gdb_path.name.endswith('.gdb'):
            return BackupGDBAsset(gdb_path)
        elif "/Verifications/" in path_str and gdb_path.name.endswith('.gdb'):
            return VerificationGDBAsset(gdb_path)
        elif "/Increments/" in path_str and gdb_path.name.endswith('.gdb'):
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
                if gdb_path.is_dir():  # GDB is a directory
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

        stats = {
            'found': len(assets),
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

        for asset in assets:
            if self.process_asset(asset):
                stats['processed'] += 1
            else:
                stats['failed'] += 1

        logger.info(f"Sync complete: {stats}")
        return stats


# Example usage
if __name__ == "__main__":
    # Configuration
    base_paths = {
        'backup': Path("/media/marco/SANDISK/GCOVER"),
        'verification': Path("/media/marco/SANDISK/Verifications"),
        'increment': Path("/media/marco/SANDISK/Increment")
    }

    manager = GDBAssetManager(
        base_paths=base_paths,
        s3_bucket="gcover-gdb-8552d86302f942779f83f7760a7b901b",
        db_path="gdb_metadata.duckdb",
        temp_dir="/tmp/gdb_zips",
        aws_profile="gcover_bucket"
    )

    # Sync all assets
    stats = manager.sync_all()
    print(f"Sync completed: {stats}")