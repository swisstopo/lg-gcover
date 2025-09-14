# src/gcover/gdb/utils.py
"""
Utility functions for GDB Asset Management
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import click
import logging

logger = logging.getLogger(__name__)


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
