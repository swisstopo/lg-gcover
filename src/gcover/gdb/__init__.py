"""
GDB Asset Management Module for lg-gcover
"""

# Import utility functions
from gcover.gdb.utils import (check_disk_space, copy_gdb_asset,
                              create_backup_manifest, create_destination_path,
                              filter_assets_by_criteria, find_largest_assets,
                              format_size, get_asset_age_distribution,
                              get_directory_size, quick_size_check,
                              verify_backup_integrity)

from gcover.gdb.assets import (AssetType, BackupGDBAsset, GDBAsset, IncrementGDBAsset,
                     ReleaseCandidate, VerificationGDBAsset, assets_are_equal,
                     find_duplicate_groups, get_asset_key,
                     print_duplicate_report, remove_duplicate_assets)
from gcover.gdb.manager import GDBAssetManager
from gcover.gdb.storage import MetadataDB, S3Uploader, create_s3_uploader_with_proxy

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
    "assets_are_equal",
    "get_asset_key",
    "remove_duplicate_assets",
    "find_duplicate_groups",
    "print_duplicate_report",
    "create_s3_uploader_with_proxy",
]
