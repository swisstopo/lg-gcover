"""
GDB Asset Management Module for lg-gcover
"""

from .manager import GDBAssetManager
from .storage import S3Uploader, MetadataDB
from .assets import (
    GDBAsset,
    BackupGDBAsset,
    VerificationGDBAsset,
    IncrementGDBAsset,
    AssetType,
    ReleaseCandidate,
)

# Import utility functions
from gcover.gdb.utils import (
    get_directory_size,
    format_size,
    copy_gdb_asset,
    create_destination_path,
    filter_assets_by_criteria,
    check_disk_space,
    create_backup_manifest,
    verify_backup_integrity,
    quick_size_check,
    find_largest_assets,
    get_asset_age_distribution,
)

__all__ = [
    "GDBAssetManager",
    "AssetType",
    "ReleaseCandidate",
    "GDBAsset",
    "BackupGDBAsset",
    "VerificationGDBAsset",
    "IncrementGDBAsset",
    "S3Uploader",
    "MetadataDB",
    "get_directory_size",
    "format_size",
    "copy_gdb_asset",
    "create_destination_path",
    "filter_assets_by_criteria",
    "check_disk_space",
    "create_backup_manifest",
    "verify_backup_integrity",
    "quick_size_check",
    "find_largest_assets",
    "get_asset_age_distribution",
]
