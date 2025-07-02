#!/usr/bin/env python3
"""
GDB Asset Management System for lg-gcover

Manages FileGDB assets: backups, verifications, and increments
Handles zipping, hashing, S3 upload, and metadata management
"""

import os
import zipfile
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError
import duckdb

# Configure logging
from loguru import logger


class AssetType(Enum):
    """Types of GDB assets"""
    BACKUP_DAILY = "backup_daily"
    BACKUP_WEEKLY = "backup_weekly"
    BACKUP_MONTHLY = "backup_monthly"
    VERIFICATION_TQA = "verification_tqa"  # TechnicalQualityAssurance
    VERIFICATION_TOPOLOGY = "verification_topology"
    INCREMENT = "increment"


class ReleaseCandidate(Enum):
    """Release candidates"""
    RC1 = "2016-12-31"
    RC2 = "2030-12-31"

    @classmethod
    def from_string(cls, date_str: str) -> Optional['ReleaseCandidate']:
        """Convert date string to RC enum"""
        for rc in cls:
            if rc.value == date_str:
                return rc
        return None

    @property
    def short_name(self) -> str:
        """Get short name (RC1, RC2)"""
        return "RC1" if self == ReleaseCandidate.RC1 else "RC2"


@dataclass
class GDBAssetInfo:
    """Information about a GDB asset"""
    path: Path
    asset_type: AssetType
    release_candidate: ReleaseCandidate
    timestamp: datetime
    file_size: int = 0
    hash_md5: Optional[str] = None
    zip_path: Optional[Path] = None
    s3_key: Optional[str] = None
    uploaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class GDBAsset:
    """Base class for GDB assets"""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.info = self._parse_path()

    def _parse_path(self) -> GDBAssetInfo:
        """Parse path to extract asset information - to be implemented by subclasses"""
        raise NotImplementedError

    def get_zip_target(self) -> Path:
        """Get the target path/directory to zip"""
        return self.path

    def create_zip(self, output_dir: Path) -> Path:
        """Create zip file of the asset"""
        zip_target = self.get_zip_target()
        zip_name = f"{self.info.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.info.release_candidate.short_name}_{self.info.asset_type.value}.zip"
        zip_path = output_dir / zip_name

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if zip_target.is_file():
                zipf.write(zip_target, zip_target.name)
            else:
                for file_path in zip_target.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(zip_target.parent)
                        zipf.write(file_path, arcname)

        self.info.zip_path = zip_path
        self.info.file_size = zip_path.stat().st_size
        return zip_path

    def compute_hash(self) -> str:
        """Compute MD5 hash of the zip file"""
        if not self.info.zip_path or not self.info.zip_path.exists():
            raise ValueError("Zip file must be created first")

        hash_md5 = hashlib.md5()
        with open(self.info.zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        self.info.hash_md5 = hash_md5.hexdigest()
        return self.info.hash_md5


class BackupGDBAsset(GDBAsset):
    """GDB asset for backups (daily, weekly, monthly)"""

    def _parse_path(self) -> GDBAssetInfo:
        """Parse backup path: YYYYMMDD_HHMM_YYYY-MM-DD.gdb"""
        gdb_name = self.path.name
        parent_name = self.path.parent.name

        # Pattern: 20221130_2200_2030-12-31.gdb
        pattern = r'(\d{8})_(\d{4})_(\d{4}-\d{2}-\d{2})\.gdb'
        match = re.match(pattern, gdb_name)

        if not match:
            raise ValueError(f"Invalid backup GDB name format: {gdb_name}")

        date_str, time_str, rc_str = match.groups()

        # Parse timestamp
        timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")

        # Parse release candidate
        rc = ReleaseCandidate.from_string(rc_str)
        if not rc:
            raise ValueError(f"Unknown release candidate: {rc_str}")

        # Determine asset type from parent directory
        asset_type_map = {
            "daily": AssetType.BACKUP_DAILY,
            "weekly": AssetType.BACKUP_WEEKLY,
            "monthly": AssetType.BACKUP_MONTHLY
        }
        asset_type = asset_type_map.get(parent_name)
        if not asset_type:
            raise ValueError(f"Unknown backup type: {parent_name}")

        return GDBAssetInfo(
            path=self.path,
            asset_type=asset_type,
            release_candidate=rc,
            timestamp=timestamp
        )


class VerificationGDBAsset(GDBAsset):
    """GDB asset for verifications (TQA, Topology)"""

    def _parse_path(self) -> GDBAssetInfo:
        """Parse verification path structure"""
        # Path structure: .../TechnicalQualityAssurance/RC_2016-12-31/20231203_22-00-09_2016-12-31/issue.gdb
        parts = self.path.parts

        # Find relevant parts
        test_type = None
        rc_str = None
        timestamp_dir = None

        for i, part in enumerate(parts):
            if part in ["TechnicalQualityAssurance", "Topology"]:
                test_type = part
            elif part.startswith("RC_"):
                rc_str = part[3:]  # Remove "RC_" prefix
            #elif re.match(r'\d{8}_\d{2}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}', part):
            elif re.match(r'\d{8}_\d{2}-\d{2}-\d{2}.*', part):
                timestamp_dir = part

        if not all([test_type, rc_str, timestamp_dir]):
            logger.error(f"{test_type}, {rc_str}, {timestamp_dir}")
            raise ValueError(f"Cannot parse verification path: {self.path}")

        # Parse timestamp from directory name
        # Pattern: 20231203_22-00-09_2016-12-31
        timestamp_pattern = r'(\d{8})_(\d{2})-(\d{2})-(\d{2})_\d{4}-\d{2}-\d{2}'
        # TODO
        timestamp_pattern = r'(\d{8})_(\d{2})-(\d{2})-(\d{2}).*'
        match = re.match(timestamp_pattern, timestamp_dir)
        if not match:
            raise ValueError(f"Cannot parse timestamp from: {timestamp_dir}")

        date_str, hour, minute, second = match.groups()
        timestamp = datetime.strptime(f"{date_str}_{hour}{minute}{second}", "%Y%m%d_%H%M%S")

        # Parse release candidate
        rc = ReleaseCandidate.from_string(rc_str)
        if not rc:
            raise ValueError(f"Unknown release candidate: {rc_str}")

        # Determine asset type
        asset_type = (AssetType.VERIFICATION_TQA if test_type == "TechnicalQualityAssurance"
                      else AssetType.VERIFICATION_TOPOLOGY)

        return GDBAssetInfo(
            path=self.path,
            asset_type=asset_type,
            release_candidate=rc,
            timestamp=timestamp,
            metadata={"gdb_name": self.path.name}
        )

    def get_zip_target(self) -> Path:
        """For verifications, zip the parent directory"""
        return self.path.parent


class IncrementGDBAsset(GDBAsset):
    """GDB asset for increments"""

    def _parse_path(self) -> GDBAssetInfo:
        """Parse increment path: YYYYMMDD_GCOVERP_YYYY-MM-DD.gdb"""
        gdb_name = self.path.name

        # Pattern: 20250224_GCOVERP_2016-12-31.gdb
        pattern = r'(\d{8})_GCOVERP_(\d{4}-\d{2}-\d{2})\.gdb'
        match = re.match(pattern, gdb_name)

        if not match:
            raise ValueError(f"Invalid increment GDB name format: {gdb_name}")

        date_str, rc_str = match.groups()

        # Parse timestamp (assume midnight for increments)
        timestamp = datetime.strptime(date_str, "%Y%m%d")

        # Parse release candidate
        rc = ReleaseCandidate.from_string(rc_str)
        if not rc:
            raise ValueError(f"Unknown release candidate: {rc_str}")

        return GDBAssetInfo(
            path=self.path,
            asset_type=AssetType.INCREMENT,
            release_candidate=rc,
            timestamp=timestamp
        )


class S3Uploader:
    """Handle S3 operations"""

    def __init__(self, bucket_name: str, aws_profile: Optional[str] = None):
        self.bucket_name = bucket_name
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client('s3')
        else:
            self.s3_client = boto3.client('s3')

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


class MetadataDB:
    """Handle DuckDB metadata operations"""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.init_db()

    def init_db(self):
        """Initialize database schema"""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gdb_assets (
                    id INTEGER PRIMARY KEY,
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
                )
            """)

    def insert_asset(self, asset_info: GDBAssetInfo):
        """Insert asset information"""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO gdb_assets 
                (path, asset_type, release_candidate, timestamp, file_size, 
                 hash_md5, s3_key, uploaded, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                str(asset_info.path),
                asset_info.asset_type.value,
                asset_info.release_candidate.value,
                asset_info.timestamp,
                asset_info.file_size,
                asset_info.hash_md5,
                asset_info.s3_key,
                asset_info.uploaded,
                asset_info.metadata
            ])

    def asset_exists(self, path: Path) -> bool:
        """Check if asset already exists in database"""
        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM gdb_assets WHERE path = ?",
                [str(path)]
            ).fetchone()
            return result[0] > 0


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
        elif "/Increment/" in path_str and gdb_path.name.endswith('.gdb'):
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