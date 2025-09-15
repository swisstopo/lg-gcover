# !/usr/bin/env python3
"""
GDB Asset Management Utilities

This module provides utility functions for GDB asset management including
file operations, path mapping, and size calculations.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import click
import shutil
import click
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime

from rich.console import Console
from rich import print as rprint

from loguru import logger

try:
    from .assets import GDBAsset, AssetType, ReleaseCandidate
except ImportError:
    # Fallback for when used independently
    from gcover.gdb.assets import GDBAsset, AssetType, ReleaseCandidate

console = Console()


def check_gdb_integrity(gdb_path: Path, timeout: int = 300) -> Tuple[bool, List[str]]:
    """
    Check GDB integrity using GDAL/OGR tools

    Args:
        gdb_path: Path to GDB
        timeout: Timeout in seconds

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check if GDB exists and is accessible
    if not gdb_path.exists():
        errors.append(f"GDB does not exist: {gdb_path}")
        return False, errors

    # Basic info check
    try:
        result = subprocess.run(
            ["ogrinfo", "-al", "-so", str(gdb_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            errors.append(f"ogrinfo failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        errors.append("ogrinfo timed out")
    except FileNotFoundError:
        errors.append("ogrinfo not found - GDAL/OGR not installed")
    except Exception as e:
        errors.append(f"ogrinfo error: {e}")

    # Geometry validation (if basic check passed)
    if not errors:
        try:
            result = subprocess.run(
                ["ogrinfo", "-checkgeom", str(gdb_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                errors.append(f"Geometry validation failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            errors.append("Geometry validation timed out")
        except Exception as e:
            errors.append(f"Geometry validation error: {e}")

    return len(errors) == 0, errors


def get_gdb_info(gdb_path: Path) -> Optional[dict]:
    """
    Get basic information about a GDB

    Returns:
        Dictionary with GDB information or None if failed
    """
    try:
        result = subprocess.run(
            ["ogrinfo", "-json", "-so", str(gdb_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            import json

            return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"Failed to get GDB info for {gdb_path}: {e}")

    return None


def estimate_zip_size(directory: Path, compression_ratio: float = 0.3) -> int:
    """
    Estimate compressed zip size

    Args:
        directory: Directory to estimate
        compression_ratio: Expected compression ratio

    Returns:
        Estimated size in bytes
    """
    total_size = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return int(total_size * compression_ratio)


def create_secure_temp_dir() -> Path:
    """Create a secure temporary directory"""
    temp_dir = Path(tempfile.mkdtemp(prefix="gdb_", suffix="_temp"))
    # Set restrictive permissions
    os.chmod(temp_dir, 0o700)
    return temp_dir


def clean_temp_files(temp_dir: Path, max_age_hours: int = 24) -> int:
    """
    Clean old temporary files

    Args:
        temp_dir: Temporary directory to clean
        max_age_hours: Max age of files to keep

    Returns:
        Number of files cleaned
    """
    import time

    if not temp_dir.exists():
        return 0

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0

    for file_path in temp_dir.rglob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to clean {file_path}: {e}")

    return cleaned_count


def get_directory_size(path: Path) -> int:
    """
    Calculate total size of directory in bytes.

    Args:
        path: Directory path to calculate size for

    Returns:
        Total size in bytes

    Example:
        >>> size = get_directory_size(Path("/path/to/geodatabase.gdb"))
        >>> print(f"Size: {format_size(size)}")
    """
    total_size = 0
    if not path.exists():
        return 0

    if path.is_file():
        try:
            return path.stat().st_size
        except (OSError, FileNotFoundError):
            return 0

    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                continue
    return total_size


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string

    Example:
        >>> format_size(1536)
        '1.5 KB'
        >>> format_size(1073741824)
        '1.0 GB'
    """
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def copy_gdb_asset(
    source: Path,
    destination: Path,
    progress_callback: Optional[Callable] = None,
    overwrite: bool = False,
    verify: bool = False,
) -> bool:
    """
    Copy a GDB asset (directory) to destination with error handling.

    Args:
        source: Source GDB directory
        destination: Destination directory
        progress_callback: Optional callback for progress updates
        overwrite: Whether to overwrite existing destinations
        verify: Whether to verify copy integrity (size comparison)

    Returns:
        True if successful, False otherwise

    Example:
        >>> success = copy_gdb_asset(
        ...     Path("/source/data.gdb"),
        ...     Path("/dest/data.gdb"),
        ...     overwrite=True,
        ...     verify=True
        ... )
    """
    try:
        if not source.exists():
            rprint(f"[red]Source does not exist: {source}[/red]")
            return False

        if destination.exists():
            if not overwrite:
                if not click.confirm(f"Destination {destination} exists. Overwrite?"):
                    return False

            # Remove existing destination
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        # Ensure parent directories exist
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Copy the directory
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

        # Verify copy if requested
        if verify:
            source_size = get_directory_size(source)
            dest_size = get_directory_size(destination)

            if source_size != dest_size:
                rprint(f"[yellow]Warning: Size mismatch for {source.name}[/yellow]")
                rprint(f"  Source: {format_size(source_size)}")
                rprint(f"  Destination: {format_size(dest_size)}")
                # Don't return False for size mismatch, just warn

        if progress_callback:
            progress_callback()

        return True

    except PermissionError as e:
        rprint(f"[red]Permission denied copying {source.name}: {e}[/red]")
        return False
    except OSError as e:
        rprint(f"[red]OS error copying {source.name}: {e}[/red]")
        return False
    except Exception as e:
        rprint(f"[red]Unexpected error copying {source.name}: {e}[/red]")
        return False


def _map_asset_to_structure_windows(asset: GDBAsset, config_base_paths: Dict[str, str]) -> Path:
    """
    Internal function to map asset to directory structure for Windows.
    This is now a fallback function - the main logic is in _find_relative_path_from_common_dir_windows.

    Args:
        asset: GDB asset
        config_base_paths: Configuration base paths

    Returns:
        Relative path for asset structure
    """
    # Try to find relative path from common directory first
    relative_path = _find_relative_path_from_common_dir_windows(asset.path)

    if relative_path:
        return relative_path.parent  # Return parent because we'll add filename later

    # Fallback: use asset type to determine structure
    asset_type = asset.info.asset_type.value
    fallback_mapping = {
        'backup': Path("backup"),
        'verification': Path("QA"),
        'increment': Path("Increment"),
    }

    return fallback_mapping.get(asset_type, Path("other"))


def _map_asset_to_structure(asset: GDBAsset, config_base_paths: Dict[str, str]) -> Path:
    """
    Internal function to map asset to directory structure.
    This is now a fallback function - the main logic is in _find_relative_path_from_common_dir.

    Args:
        asset: GDB asset
        config_base_paths: Configuration base paths

    Returns:
        Relative path for asset structure
    """
    # Try to find relative path from common directory first
    relative_path = _find_relative_path_from_common_dir(asset.path)

    if relative_path:
        return relative_path.parent  # Return parent because we'll add filename later

    # Fallback: use asset type to determine structure
    asset_type = asset.info.asset_type.value
    fallback_mapping = {
        'backup': Path("backup"),
        'verification': Path("QA"),
        'increment': Path("Increment"),
    }

    return fallback_mapping.get(asset_type, Path("other"))

def filter_assets_by_criteria(
    assets: List[GDBAsset],
    asset_type: Optional[str] = None,
    rc: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    latest_only: bool = False,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> List[GDBAsset]:
    """
    Filter assets based on multiple criteria.

    Args:
        assets: List of GDB assets to filter
        asset_type: Filter by asset type (backup, verification, increment)
        rc: Filter by release candidate (RC1, RC2)
        since: Filter assets created since this date
        until: Filter assets created until this date
        latest_only: Get only the latest asset of each type/RC combination
        min_size: Minimum size in bytes
        max_size: Maximum size in bytes

    Returns:
        Filtered list of assets

    Example:
        >>> filtered = filter_assets_by_criteria(
        ...     all_assets,
        ...     asset_type='backup',
        ...     rc='RC2',
        ...     since=datetime(2025, 1, 1),
        ...     latest_only=True
        ... )
    """
    filtered = assets

    # Filter by asset type
    if asset_type:
        filtered = [a for a in filtered if a.info.asset_type.value == asset_type]

    # Filter by release candidate
    if rc:
        rc_value = ReleaseCandidate.RC1 if rc == "RC1" else ReleaseCandidate.RC2
        filtered = [a for a in filtered if a.info.release_candidate == rc_value]

    # Filter by date range
    if since:
        filtered = [a for a in filtered if a.info.timestamp >= since]
    if until:
        filtered = [a for a in filtered if a.info.timestamp <= until]

    # Filter by size
    if min_size is not None or max_size is not None:
        size_filtered = []
        for asset in filtered:
            if asset.path.exists():
                size = get_directory_size(asset.path)
                if min_size is not None and size < min_size:
                    continue
                if max_size is not None and size > max_size:
                    continue
                size_filtered.append(asset)
        filtered = size_filtered

    # Get only latest per type/RC combination
    if latest_only:
        grouped: Dict[tuple, List[GDBAsset]] = {}
        for asset in filtered:
            key = (asset.info.asset_type, asset.info.release_candidate)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(asset)

        # Take the latest from each group
        filtered = []
        for group in grouped.values():
            latest = max(group, key=lambda a: a.info.timestamp)
            filtered.append(latest)

    return filtered


def create_destination_path_windows(
        asset: GDBAsset,
        base_destination: Union[str, Path],
        config_base_paths: Dict[str, str],
        preserve_structure: bool = True,
        custom_structure: Optional[Dict[str, str]] = None
) -> Path:
    """
    Create appropriate destination path for Windows environments.

    When preserve_structure=True, preserves the directory structure starting
    from the common path element (QA, backup, Increment).

    Args:
        asset: The GDB asset to copy
        base_destination: Base destination (UNC or drive path)
        config_base_paths: Configuration paths
        preserve_structure: Whether to preserve directory structure
        custom_structure: Optional custom structure mapping

    Returns:
        Destination path optimized for Windows

    Example:
        Source: //server/random-path/QA/Verifications/Topology/RC_2030-12-31/20231216_03-00-12/issue.gdb
        Destination: /media/usb-stick/QA/Verifications/Topology/RC_2030-12-31/20231216_03-00-12/issue.gdb
    """
    # Normalize the base destination
    if isinstance(base_destination, str):
        base_destination = normalize_windows_path(base_destination)

    base_dest_path = Path(base_destination)

    if not preserve_structure:
        return base_dest_path / asset.path.name

    # Use custom structure if provided
    if custom_structure:
        asset_type = asset.info.asset_type.value
        if asset_type in custom_structure:
            subpath = Path(custom_structure[asset_type])
            return base_dest_path / subpath / asset.path.name

    # Find the relative path starting from common directory
    relative_path = _find_relative_path_from_common_dir_windows(asset.path)

    if relative_path:
        return base_dest_path / relative_path
    else:
        # Fallback to just the filename
        return base_dest_path / asset.path.name


def create_destination_path(
        asset: GDBAsset,
        base_destination: Path,
        config_base_paths: Dict[str, str],
        preserve_structure: bool = True,
        custom_structure: Optional[Dict[str, str]] = None
) -> Path:
    """
    Create appropriate destination path with structure preservation.

    When preserve_structure=True, preserves the directory structure starting
    from the common path element (QA, backup, Increment).

    Args:
        asset: The GDB asset to copy
        base_destination: Base destination directory
        config_base_paths: Configuration paths from config
        preserve_structure: Whether to preserve directory structure
        custom_structure: Optional custom structure mapping

    Returns:
        Destination path for the asset

    Example:
        Source: /server/random-path/backup/GCOVER/daily/20221130_2200_2030-12-31.gdb
        Destination: /media/usb-stick/backup/GCOVER/daily/20221130_2200_2030-12-31.gdb
    """
    if not preserve_structure:
        return base_destination / asset.path.name

    # Use custom structure if provided
    if custom_structure:
        asset_type = asset.info.asset_type.value
        if asset_type in custom_structure:
            subpath = Path(custom_structure[asset_type])
            return base_destination / subpath / asset.path.name

    # Find the relative path starting from common directory
    relative_path = _find_relative_path_from_common_dir(asset.path)

    if relative_path:
        return base_destination / relative_path
    else:
        # Fallback to just the filename
        return base_destination / asset.path.name


def _find_relative_path_from_common_dir_windows(asset_path: Path) -> Optional[Path]:
    """
    Find the relative path starting from the common directory (QA, backup, Increment).

    Args:
        asset_path: Full path to the asset

    Returns:
        Relative path starting from common directory, or None if not found

    Example:
        Input: //server/random-path/QA/Verifications/Topology/RC_2030-12-31/20231216_03-00-12/issue.gdb
        Output: QA/Verifications/Topology/RC_2030-12-31/20231216_03-00-12/issue.gdb
    """
    # Convert to string and normalize for Windows
    path_str = normalize_windows_path(str(asset_path)).lower()

    # Common directory patterns to look for (case insensitive)
    common_patterns = [
        'qa',  # For QA/Verifications
        'backup',  # For backup/GCOVER
        'increment'  # For Increment/GCOVERP
    ]

    # Split path into parts
    if path_str.startswith('\\\\'):
        # UNC path - split and handle appropriately
        parts = path_str.split('\\')
    else:
        # Regular path
        parts = path_str.replace('/', '\\').split('\\')

    # Find the first occurrence of any common pattern
    for i, part in enumerate(parts):
        if part.lower() in common_patterns:
            # Found a common directory, construct relative path from here
            relative_parts = parts[i:]

            # Reconstruct the original case from the original path
            original_parts = str(asset_path).replace('/', '\\').split('\\')

            # Find the corresponding parts in original path with correct case
            if len(original_parts) >= len(parts):
                start_idx = len(original_parts) - len(parts) + i
                if start_idx >= 0:
                    original_relative_parts = original_parts[start_idx:]
                    return Path('\\'.join(original_relative_parts)) if is_windows() else Path(
                        '/'.join(original_relative_parts))

            # Fallback: use lowercase parts
            return Path('\\'.join(relative_parts)) if is_windows() else Path('/'.join(relative_parts))

    return None


def _find_relative_path_from_common_dir(asset_path: Path) -> Optional[Path]:
    """
    Find the relative path starting from the common directory (QA, backup, Increment).

    Args:
        asset_path: Full path to the asset

    Returns:
        Relative path starting from common directory, or None if not found

    Example:
        Input: /server/random-path/backup/GCOVER/daily/20221130_2200_2030-12-31.gdb
        Output: backup/GCOVER/daily/20221130_2200_2030-12-31.gdb
    """
    # Convert to string for processing
    path_str = str(asset_path).lower()

    # Common directory patterns to look for (case insensitive)
    common_patterns = [
        'qa',  # For QA/Verifications
        'backup',  # For backup/GCOVER
        'increment'  # For Increment/GCOVERP
    ]

    # Split path into parts - handle both / and \ separators
    parts = path_str.replace('\\', '/').split('/')

    # Find the first occurrence of any common pattern
    for i, part in enumerate(parts):
        if part.lower() in common_patterns:
            # Found a common directory, construct relative path from here

            # Get the original parts with correct case
            original_parts = str(asset_path).replace('\\', '/').split('/')

            # Make sure we have enough parts
            if len(original_parts) >= len(parts) and i < len(original_parts):
                # Find the corresponding index in original parts
                start_idx = len(original_parts) - len(parts) + i
                if start_idx >= 0 and start_idx < len(original_parts):
                    relative_parts = original_parts[start_idx:]
                    return Path('/'.join(relative_parts))

            # Fallback: use the parts we found
            relative_parts = parts[i:]
            return Path('/'.join(relative_parts))

    return None


def check_disk_space(path: Path, required_space: int) -> Dict[str, Union[int, bool]]:
    """
    Check available disk space at given path.

    Args:
        path: Path to check disk space for
        required_space: Required space in bytes

    Returns:
        Dictionary with space information and availability check

    Example:
        >>> space_info = check_disk_space(Path("/media/usb"), 1024*1024*1024)  # 1GB
        >>> if space_info['sufficient']:
        ...     print("Enough space available")
    """
    try:
        stat = shutil.disk_usage(path)
        return {
            "total": stat.total,
            "used": stat.total - stat.free,
            "free": stat.free,
            "required": required_space,
            "sufficient": stat.free >= required_space,
            "utilization_percent": ((stat.total - stat.free) / stat.total) * 100,
        }
    except Exception as e:
        rprint(f"[yellow]Warning: Could not check disk space: {e}[/yellow]")
        return {
            "total": 0,
            "used": 0,
            "free": 0,
            "required": required_space,
            "sufficient": False,  # Conservative default
            "utilization_percent": 0,
            "error": str(e),
        }


def create_backup_manifest(
    assets: List[GDBAsset], destination: Path, metadata: Optional[Dict] = None
) -> Path:
    """
    Create a manifest file listing all backed up assets.

    Args:
        assets: List of assets that were backed up
        destination: Backup destination directory
        metadata: Optional metadata to include in manifest

    Returns:
        Path to created manifest file

    Example:
        >>> manifest_path = create_backup_manifest(
        ...     copied_assets,
        ...     Path("/media/usb"),
        ...     metadata={'backup_date': datetime.now(), 'operator': 'user'}
        ... )
    """
    import json
    from datetime import datetime

    manifest_path = destination / "GCOVER_backup_manifest.json"

    manifest_data = {
        "backup_info": {
            "creation_date": datetime.now().isoformat(),
            "destination": str(destination),
            "total_assets": len(assets),
            "total_size_bytes": sum(
                get_directory_size(asset.path)
                for asset in assets
                if asset.path.exists()
            ),
        },
        "metadata": metadata or {},
        "assets": [],
    }

    # Add asset information
    for asset in assets:
        asset_info = {
            "name": asset.path.name,
            "path": str(asset.path),
            "type": asset.info.asset_type.value,
            "release_candidate": asset.info.release_candidate.value,
            "timestamp": asset.info.timestamp.isoformat(),
            "size_bytes": get_directory_size(asset.path) if asset.path.exists() else 0,
        }
        manifest_data["assets"].append(asset_info)

    # Sort assets by timestamp (newest first)
    manifest_data["assets"].sort(key=lambda x: x["timestamp"], reverse=True)

    # Write manifest
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        rprint(f"[green]Created backup manifest: {manifest_path}[/green]")
        return manifest_path

    except Exception as e:
        rprint(f"[yellow]Warning: Could not create manifest: {e}[/yellow]")
        return manifest_path


def verify_backup_integrity(
    source_assets: List[GDBAsset], destination: Path
) -> Dict[str, List[str]]:
    """
    Verify integrity of backed up assets by comparing sizes and existence.

    Args:
        source_assets: Original assets that were backed up
        destination: Backup destination directory

    Returns:
        Dictionary with verification results

    Example:
        >>> results = verify_backup_integrity(original_assets, Path("/media/usb"))
        >>> if results['missing']:
        ...     print(f"Missing files: {results['missing']}")
    """
    results = {"verified": [], "missing": [], "size_mismatch": [], "errors": []}

    for asset in source_assets:
        try:
            # Find the backed up file (could be in various subdirectories)
            backup_files = list(destination.rglob(asset.path.name))

            if not backup_files:
                results["missing"].append(asset.path.name)
                continue

            # Take the first match (there should only be one)
            backup_file = backup_files[0]

            # Compare sizes
            source_size = get_directory_size(asset.path) if asset.path.exists() else 0
            backup_size = get_directory_size(backup_file)

            if source_size != backup_size:
                results["size_mismatch"].append(
                    {
                        "name": asset.path.name,
                        "source_size": source_size,
                        "backup_size": backup_size,
                    }
                )
            else:
                results["verified"].append(asset.path.name)

        except Exception as e:
            results["errors"].append(f"{asset.path.name}: {str(e)}")

    return results


# Convenience functions for common operations
def quick_size_check(paths: List[Path]) -> int:
    """Quick calculation of total size for multiple paths."""
    return sum(get_directory_size(path) for path in paths if path.exists())


def find_largest_assets(assets: List[GDBAsset], top_n: int = 5) -> List[tuple]:
    """Find the N largest assets by size."""
    asset_sizes = []
    for asset in assets:
        if asset.path.exists():
            size = get_directory_size(asset.path)
            asset_sizes.append((asset, size))

    # Sort by size (largest first)
    asset_sizes.sort(key=lambda x: x[1], reverse=True)
    return asset_sizes[:top_n]


def get_asset_age_distribution(assets: List[GDBAsset]) -> Dict[str, int]:
    """Get distribution of assets by age ranges."""
    from datetime import timedelta

    now = datetime.now()
    distribution = {"today": 0, "this_week": 0, "this_month": 0, "older": 0}

    for asset in assets:
        age = now - asset.info.timestamp

        if age <= timedelta(days=1):
            distribution["today"] += 1
        elif age <= timedelta(days=7):
            distribution["this_week"] += 1
        elif age <= timedelta(days=30):
            distribution["this_month"] += 1
        else:
            distribution["older"] += 1

    return distribution
