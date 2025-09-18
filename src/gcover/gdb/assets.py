"""
GDB Asset Management System for lg-gcover

Manages FileGDB assets: backups, verifications, and increments
Handles zipping, hashing, S3 upload, and metadata management
"""

import hashlib
import os
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import boto3
import duckdb
from botocore.exceptions import ClientError
# Configure logging
from loguru import logger
from rich.console import Console

console = Console()


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
    def from_string(cls, date_str: str) -> Optional["ReleaseCandidate"]:
        """Convert date string to RC enum"""
        for rc in cls:
            if rc.value == date_str:
                return rc
        return None

    @property
    def long_name(self) -> str:
        """Get long name (2016-12-31, 2030-12-31)"""

        return self.value

    @property
    def short_name(self) -> str:
        """Get short name (RC1, RC2)"""

        return self.name


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

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if zip_target.is_file():
                zipf.write(zip_target, zip_target.name)
            else:
                for file_path in zip_target.rglob("*"):
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
        pattern = r"(\d{8})_(\d{4})_(\d{4}-\d{2}-\d{2})\.gdb"
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
            "monthly": AssetType.BACKUP_MONTHLY,
        }
        asset_type = asset_type_map.get(parent_name)
        if not asset_type:
            raise ValueError(f"Unknown backup type: {parent_name}")

        return GDBAssetInfo(
            path=self.path,
            asset_type=asset_type,
            release_candidate=rc,
            timestamp=timestamp,
        )


class VerificationGDBAsset(GDBAsset):
    """GDB asset for verifications (TQA, Topology)"""

    # TODO
    def __str__(self):
        return f"<VerificationGDBAsset: {self.path} >"

    def _parse_path(self) -> GDBAssetInfo:
        """Parse verification path structure"""
        if self.path.name.lower() == "progress.gdb":
            raise ValueError(f"Skipping temporary file: {self.path.name}")

        # Path structure: .../TechnicalQualityAssurance/RC_2016-12-31/20231203_22-00-09_2016-12-31/issue.gdb
        parts = self.path.parts

        # Find relevant parts
        test_type = None
        rc_str = None
        timestamp_dir = None

        logger.debug(f"Parts: {'|'.join(list(map(str, parts)))}")

        for i, part in enumerate(parts):
            if part in ["TechnicalQualityAssurance", "Topology"]:
                test_type = part
            elif part.startswith("RC_"):
                rc_str = part[3:]  # Remove "RC_" prefix
            # elif re.match(r'\d{8}_\d{2}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}', part):
            elif re.match(r"\d{8}_\d{2}-\d{2}-\d{2}.*", part):
                timestamp_dir = part

        if not all([test_type, rc_str, timestamp_dir]):
            logger.error(f"{test_type}, {rc_str}, {timestamp_dir}")
            raise ValueError(f"Cannot parse verification path: {self.path}")

        # Parse timestamp from directory name
        # Pattern: 20231203_22-00-09_2016-12-31
        timestamp_pattern = r"(\d{8})_(\d{2})-(\d{2})-(\d{2})_\d{4}-\d{2}-\d{2}"
        # TODO
        timestamp_pattern = r"(\d{8})_(\d{2})-(\d{2})-(\d{2}).*"
        match = re.match(timestamp_pattern, timestamp_dir)
        if not match:
            raise ValueError(f"Cannot parse timestamp from: {timestamp_dir}")

        date_str, hour, minute, second = match.groups()
        timestamp = datetime.strptime(
            f"{date_str}_{hour}{minute}{second}", "%Y%m%d_%H%M%S"
        )

        # Parse release candidate
        rc = ReleaseCandidate.from_string(rc_str)
        if not rc:
            raise ValueError(f"Unknown release candidate: {rc_str}")

        # Determine asset type
        asset_type = (
            AssetType.VERIFICATION_TQA
            if test_type == "TechnicalQualityAssurance"
            else AssetType.VERIFICATION_TOPOLOGY
        )

        return GDBAssetInfo(
            path=self.path,
            asset_type=asset_type,
            release_candidate=rc,
            timestamp=timestamp,
            metadata={"gdb_name": self.path.name},
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
        pattern = r"(\d{8})_GCOVERP_(\d{4}-\d{2}-\d{2})\.gdb"

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
            timestamp=timestamp,
        )


# Helper functions


def assets_are_equal(
    asset1: GDBAsset,
    asset2: GDBAsset,
    ignore_path: bool = True,
    ignore_file_attributes: bool = True,
) -> bool:
    """
    Compare two GDB assets to determine if they represent the same logical asset.

    Args:
        asset1: First asset to compare
        asset2: Second asset to compare
        ignore_path: If True, don't compare file paths (useful for moved files)
        ignore_file_attributes: If True, ignore file_size, hash_md5, zip_path, etc.

    Returns:
        True if assets are logically equivalent, False otherwise
    """
    # Different asset classes are never equal
    if type(asset1) != type(asset2):
        return False

    info1, info2 = asset1.info, asset2.info

    # Compare core logical attributes
    if (
        info1.asset_type != info2.asset_type
        or info1.release_candidate != info2.release_candidate
        or info1.timestamp != info2.timestamp
    ):
        return False

    # Compare paths if not ignored
    if not ignore_path and info1.path != info2.path:
        return False

    # Compare metadata (important for verification assets)
    if info1.metadata != info2.metadata:
        return False

    # Compare file attributes if not ignored
    if not ignore_file_attributes:
        if info1.file_size != info2.file_size or info1.hash_md5 != info2.hash_md5:
            return False

    return True


def get_asset_key(asset: GDBAsset, include_path: bool = False) -> Tuple[Any, ...]:
    """
    Generate a unique key for an asset based on its logical properties.

    Args:
        asset: The asset to generate a key for
        include_path: Whether to include the path in the key

    Returns:
        Tuple that can be used as a dictionary key or for grouping
    """
    info = asset.info

    # Base key components
    key_parts = [
        type(asset).__name__,  # Asset class type
        info.asset_type.value,
        info.release_candidate.value,
        info.timestamp,
    ]

    # Add path if requested
    if include_path:
        key_parts.append(str(info.path))

    # Add metadata as a sorted tuple (for consistent hashing)
    if info.metadata:
        metadata_items = tuple(sorted(info.metadata.items()))
        key_parts.append(metadata_items)

    return tuple(key_parts)


def remove_duplicate_assets(
    assets: List[GDBAsset], keep_strategy: str = "latest", ignore_path: bool = True
) -> List[GDBAsset]:
    """
    Remove duplicate assets from a list, keeping only unique logical assets.

    Args:
        assets: List of assets to deduplicate
        keep_strategy: Strategy for choosing which duplicate to keep:
            - "latest": Keep the asset with the latest timestamp
            - "first": Keep the first occurrence
            - "largest": Keep the asset with the largest file size
        ignore_path: Whether to ignore paths when comparing (useful for moved files)

    Returns:
        List with duplicates removed
    """
    if not assets:
        return []

    # Group assets by their logical key
    asset_groups = defaultdict(list)

    for asset in assets:
        key = get_asset_key(asset, include_path=not ignore_path)
        asset_groups[key].append(asset)

    # Select one asset from each group based on strategy
    unique_assets = []

    for group in asset_groups.values():
        if len(group) == 1:
            unique_assets.append(group[0])
        else:
            # Multiple assets with same logical key - choose based on strategy
            if keep_strategy == "latest":
                chosen = max(group, key=lambda a: a.info.timestamp)
            elif keep_strategy == "first":
                chosen = group[0]  # First in original order
            elif keep_strategy == "largest":
                chosen = max(group, key=lambda a: a.info.file_size or 0)
            else:
                raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

            unique_assets.append(chosen)

    return unique_assets


def find_duplicate_groups(
    assets: List[GDBAsset], ignore_path: bool = True
) -> Dict[Tuple[Any, ...], List[GDBAsset]]:
    """
    Find groups of duplicate assets without removing them.

    Args:
        assets: List of assets to analyze
        ignore_path: Whether to ignore paths when comparing

    Returns:
        Dictionary mapping asset keys to lists of duplicate assets.
        Only returns groups with more than one asset.
    """
    asset_groups = defaultdict(list)

    for asset in assets:
        key = get_asset_key(asset, include_path=not ignore_path)
        asset_groups[key].append(asset)

    # Return only groups with duplicates
    return {k: v for k, v in asset_groups.items() if len(v) > 1}


def print_duplicate_report(assets: List[GDBAsset], ignore_path: bool = True) -> None:
    """
    Print a report of duplicate assets found in the list.

    Args:
        assets: List of assets to analyze
        ignore_path: Whether to ignore paths when comparing
    """
    duplicate_groups = find_duplicate_groups(assets, ignore_path=ignore_path)

    if not duplicate_groups:
        console.print("No duplicate assets found.")
        return

    console.print(f"Found {len(duplicate_groups)} groups of duplicate assets:")
    console.print("=" * 60)

    for i, (key, group) in enumerate(duplicate_groups.items(), 1):
        console.print(f"\nGroup {i}: {len(group)} duplicates")
        console.print(f"Asset type: {group[0].__class__.__name__}")
        console.print(f"Logical key: {key}")

        for j, asset in enumerate(group, 1):
            info = asset.info
            console.print(f"  {j}. Path: {info.path}")
            console.print(f"     Timestamp: {info.timestamp}")
            if info.file_size:
                console.print(f"     Size: {info.file_size:,} bytes")
            if info.hash_md5:
                console.print(f"     Hash: {info.hash_md5[:12]}...")
