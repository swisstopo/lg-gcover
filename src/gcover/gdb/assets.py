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