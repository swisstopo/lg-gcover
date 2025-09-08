import hashlib
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

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
    ReleaseCandidate,
)

from .storage import S3Uploader, MetadataDB

GBD_TO_EXCLUDE = ["progress.gdb", "temp.gdb"]


class GDBAssetManager:
    """Main manager for GDB assets"""

    def __init__(
        self,
        base_paths: Dict[str, Path],
        s3_bucket: str,
        db_path: Union[str, Path],
        temp_dir: Union[str, Path] = "/tmp/gdb_zips",
        aws_profile: Optional[str] = None,
    ):
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

        logger.info(self.s3_uploader)

    def create_asset(self, gdb_path: Path) -> GDBAsset:
        """Factory method to create appropriate asset type"""
        path_str = str(gdb_path)

        if "/GCOVER/" in path_str and gdb_path.name.endswith(".gdb"):
            return BackupGDBAsset(gdb_path)
        elif "/Verifications/" in path_str and gdb_path.name.endswith(".gdb"):
            return VerificationGDBAsset(gdb_path)
        elif "/Increments/" in path_str and gdb_path.name.endswith(".gdb"):
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
                    # NOUVEAU : Filtrer les fichiers temporaires
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

    def get_latest_assets_by_rc(
        self, asset_type: Optional[str] = None, days_back: Optional[int] = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get the latest asset for each RC (RC1/RC2) for a given asset type.

        Args:
            asset_type: Filter by specific asset type (e.g., 'verification_topology')
            days_back: Only consider assets from the last N days (None for all)

        Returns:
            Dict with RC names as keys and asset info as values:
            {
                'RC1': {'timestamp': datetime, 'path': str, 'size': int, ...},
                'RC2': {'timestamp': datetime, 'path': str, 'size': int, ...}
            }
        """
        if asset_type and asset_type not in [t.value for t in AssetType]:
            valid_types = [t.value for t in AssetType]
            raise ValueError(
                f"Invalid asset_type: '{asset_type}'. Valid types: {valid_types}"
            )

        with duckdb.connect(str(self.metadata_db.db_path)) as conn:
            query = f"""
                WITH ranked_assets AS (
                    SELECT *,
                           CASE 
                               WHEN release_candidate = '{ReleaseCandidate.RC1.value}' THEN 'RC1'
                               WHEN release_candidate = '{ReleaseCandidate.RC2.value}' THEN 'RC2'
                               ELSE 'Unknown'
                           END as rc_name,
                           ROW_NUMBER() OVER (
                               PARTITION BY release_candidate 
                               ORDER BY timestamp DESC
                           ) as rn
                    FROM gdb_assets 
                    WHERE 1=1
                """

            # Add asset type filter
            if asset_type:
                query += f" AND asset_type = '{asset_type}'"

            # Add days filter
            if days_back is not None:
                query += f" AND timestamp >= CURRENT_DATE - INTERVAL {days_back} DAYS"

            query += """
                )
                SELECT rc_name, timestamp, path, asset_type, file_size, uploaded, s3_key
                FROM ranked_assets 
                WHERE rn = 1 AND rc_name IN ('RC1', 'RC2')
                ORDER BY rc_name
                """

            results = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description]

        latest_assets = {}
        for row in results:
            data = dict(zip(columns, row))
            rc_name = data.pop("rc_name")
            latest_assets[rc_name] = data

        return latest_assets

    def get_latest_release_couple(
        self, asset_type: Optional[str] = None, max_days_apart: int = 7
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the latest RC1/RC2 release couple (assets created close to each other).

        Args:
            asset_type: Filter by specific asset type
            max_days_apart: Maximum days between RC1 and RC2 releases to consider them a couple

        Returns:
            Tuple of (RC1_datetime, RC2_datetime) or None if no couple found
        """
        latest_assets = self.get_latest_assets_by_rc(asset_type=asset_type)

        if "RC1" not in latest_assets or "RC2" not in latest_assets:
            return None

        rc1_date = latest_assets["RC1"]["timestamp"]
        rc2_date = latest_assets["RC2"]["timestamp"]

        # Check if they're within the specified days apart
        days_diff = abs((rc1_date - rc2_date).days)

        if days_diff <= max_days_apart:
            return (rc1_date, rc2_date)
        else:
            logger.warning(
                f"Latest RC1 ({rc1_date.date()}) and RC2 ({rc2_date.date()}) "
                f"are {days_diff} days apart (max allowed: {max_days_apart})"
            )
            return None

    def get_latest_verification_runs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the latest verification runs grouped by verification type.

        Returns:
            Dict with verification types as keys and list of latest runs as values
        """
        with duckdb.connect(str(self.metadata_db.db_path)) as conn:
            # Use the actual RC values
            rc1_value = ReleaseCandidate.RC1.value  # "2016-12-31"
            rc2_value = ReleaseCandidate.RC2.value  # "2030-12-31"

            query = f"""
                WITH latest_per_type_rc AS (
                    SELECT *,
                           CASE 
                               WHEN release_candidate = '{rc1_value}' THEN 'RC1'
                               WHEN release_candidate = '{rc2_value}' THEN 'RC2'
                               ELSE 'Unknown'
                           END as rc_name,
                           ROW_NUMBER() OVER (
                               PARTITION BY asset_type, release_candidate 
                               ORDER BY timestamp DESC
                           ) as rn
                    FROM gdb_assets 
                    WHERE asset_type LIKE 'verification_%'
                )
                SELECT asset_type, rc_name, timestamp, path, file_size, uploaded
                FROM latest_per_type_rc 
                WHERE rn = 1 AND rc_name IN ('RC1', 'RC2')
                ORDER BY asset_type, rc_name
                """

            results = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description]

        verification_runs = {}
        for row in results:
            data = dict(zip(columns, row))
            asset_type = data["asset_type"]

            if asset_type not in verification_runs:
                verification_runs[asset_type] = []
            verification_runs[asset_type].append(data)

        return verification_runs


# Example usage
if __name__ == "__main__":
    # Configuration
    base_paths = {
        "backup": Path("/media/marco/SANDISK/GCOVER"),
        "verification": Path("/media/marco/SANDISK/Verifications"),
        "increment": Path("/media/marco/SANDISK/Increment"),
    }

    manager = GDBAssetManager(
        base_paths=base_paths,
        s3_bucket="gcover-gdb-8552d86302f942779f83f7760a7b901b",
        db_path="gdb_metadata.duckdb",
        temp_dir="/tmp/gdb_zips",
        aws_profile="gcover_bucket",
    )

    # Get latest topology verification for each RC
    latest_topo = manager.get_latest_assets_by_rc(asset_type="verification_topology")
    print("Latest topology verification:")
    for rc, info in latest_topo.items():
        print(f"  {rc}: {info['timestamp']} - {Path(info['path']).name}")

    # Get latest release couple
    couple = manager.get_latest_release_couple(asset_type="verification_topology")
    if couple:
        print(
            f"\nLatest release couple: RC1={couple[0].date()}, RC2={couple[1].date()}"
        )
    else:
        print("\nNo recent release couple found")
