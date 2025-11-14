#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb

# Configure logging
from loguru import logger

from gcover.gdb.assets import (
    AssetType,
    BackupGDBAsset,
    GDBAsset,
    IncrementGDBAsset,
    ReleaseCandidate,
    VerificationGDBAsset,
)
from gcover.gdb.storage import MetadataDB, S3Uploader

GBD_TO_EXCLUDE = ["progress.gdb", "temp.gdb"]


class GDBAssetManager:
    """Main manager for GDB assets with enhanced S3 upload capabilities"""

    def __init__(
        self,
        base_paths: Dict[str, Path],
        s3_config: Dict[str, Any],
        db_path: str | Path,
        temp_dir: str | Path = "/tmp/gdb_zips",
        upload_to_s3: Optional[bool] = True,
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
        self.bucket_name = s3_config.bucket
        self.base_paths = {k: Path(v) for k, v in base_paths.items()}
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.upload_to_s3 = upload_to_s3

        # Initialize S3 uploader with enhanced configuration
        logger.debug("GDBAssetMananger: init()")
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
        proxy_config = s3_config.proxy  # .get('proxy', {})
        upload_method = s3_config.upload_method  # .get('upload_method', 'auto')

        return S3Uploader(
            bucket_name=bucket_name,
            aws_profile=aws_profile,
            lambda_endpoint=lambda_endpoint,
            totp_secret=totp_secret,
            totp_token=totp_token,
            proxy_config=proxy_config,
            upload_method=upload_method,
        )

    @classmethod
    def from_config(cls, config_obj, temp_dir: Optional[str | Path] = None):
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
        parts = [part.lower() for part in gdb_path.parts]

        if "gcover" in parts and gdb_path.name.endswith(".gdb"):
            return BackupGDBAsset(gdb_path)
        elif "verifications" in parts and gdb_path.name.endswith(".gdb"):
            return VerificationGDBAsset(gdb_path)
        elif "increment" in parts and gdb_path.name.endswith(".gdb"):
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

            # Generate S3 key early (before zipping)
            zip_filename = f"{asset.path.name}.zip"
            s3_key = f"gdb-assets/{asset.info.release_candidate.short_name}/{asset.info.asset_type.value}/{zip_filename}"
            asset.info.s3_key = s3_key

            if self.upload_to_s3:
                if self.s3_uploader.file_exists(s3_key):
                    logger.info(
                        f"File {s3_key} already exists in S3: {self.bucket_name}"
                    )
                    # Still update database if not present
                    asset.info.s3_key = s3_key
                    asset.info.uploaded = True
                    asset.info.hash_md5 = None  # We don't have it without zipping
                    self.metadata_db.insert_asset(asset.info)
                    return True

                # Only now: create zip (expensive operation)
                logger.info(f"Zipping asset: {asset.path}")
                zip_path = asset.create_zip(self.temp_dir)

                # Compute hash
                hash_md5 = asset.compute_hash()
                asset.info.s3_key = s3_key

                # Upload to S3
                uploaded = self.s3_uploader.upload_file(zip_path, s3_key)
                asset.info.uploaded = uploaded.success
                logger.debug(f"Uploaded to s3://{self.bucket_name}/{s3_key}")

                # Update database
                self.metadata_db.insert_asset(asset.info)

                # Cleanup temp file
                zip_path.unlink()

                logger.info(f"Successfully processed: {asset.path}")
                return True

            else:
                logger.warning("Skipping upload (parameter `--no-upload` set)")

            # Update database
            logger.debug(f"Update database asset...")
            self.metadata_db.insert_asset(asset.info)

            # Cleanup temp file
            logger.debug(f"Cleanup...")
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
            data = dict(zip(columns, row, strict=False))
            rc_name = data.pop("rc_name")
            latest_assets[rc_name] = data

        return latest_assets

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
            data = dict(zip(columns, row, strict=False))
            asset_type = data["asset_type"]

            if asset_type not in verification_runs:
                verification_runs[asset_type] = []
            verification_runs[asset_type].append(data)

        return verification_runs


# Backward compatibility function for existing code
def create_manager_from_legacy_params(
    base_paths: Dict[str, Path],
    s3_bucket: str,
    db_path: str | Path,
    temp_dir: str | Path = "/tmp/gdb_zips",
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
