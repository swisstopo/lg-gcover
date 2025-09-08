#!/usr/bin/env python3
"""
Usage examples for the new latest RC functionality in gcover
"""

from datetime import datetime
from gcover.cli.gdb_cmd import get_latest_topology_dates

# ============================================================================
# CLI Usage Examples
# ============================================================================

# 1. Get latest topology verification tests (your specific use case)
# This will show the latest topology verification for RC1 and RC2
# gcover --env sandisk gdb latest-topology

# Example output:
"""
                    Latest Topology Verification Tests                     
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ RC     ┃ Test Date           ┃ File                                     ┃       Size ┃ Status ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│ RC1    │ 2025-08-23 03:00:10 │ issue.gdb                                │     4.7 MB │ ✅     │
│ RC2    │ 2025-08-22 07:00:20 │ issue.gdb                                │    15.7 MB │ ✅     │
└────────┴─────────────────────┴──────────────────────────────────────────┴────────────┴────────┘

✅ Latest Release Couple: 1 days apart
Latest tests: 2025-08-23 and 2025-08-22

Answer: The latest topology verification tests are:
  RC1: 2025-08-23
  RC2: 2025-08-22
"""

# 2. Get latest assets for any type with release couple check
# gcover --env sandisk gdb latest-by-rc --type verification_topology --show-couple

# 3. Get latest assets for backup type
# gcover --env sandisk gdb latest-by-rc --type backup

# 4. Show all latest verification runs
# gcover --env sandisk gdb latest-verifications

# 5. Get latest assets from last 60 days only
# gcover --env sandisk gdb latest-by-rc --days-back 60


# ============================================================================
# Python API Usage Examples
# ============================================================================

from pathlib import Path
from gcover.gdb.manager import GDBAssetManager


def example_usage():
    """Example of using the new methods programmatically"""

    # Setup (using your config)
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

    # Example 1: Get latest topology verification for each RC
    print("=== Latest Topology Verification ===")
    latest_topo = manager.get_latest_assets_by_rc(asset_type="verification_topology")

    for rc_name, asset_info in latest_topo.items():
        print(
            f"{rc_name}: {asset_info['timestamp'].strftime('%Y-%m-%d')} - {Path(asset_info['path']).name}"
        )

    # Output:
    # RC1: 2025-08-23 - issue.gdb
    # RC2: 2025-08-22 - issue.gdb

    # Example 2: Check if they form a release couple
    print("\n=== Release Couple Check ===")
    couple = manager.get_latest_release_couple(asset_type="verification_topology")

    if couple:
        rc1_date, rc2_date = couple
        print(f"Latest release couple found:")
        print(f"  RC1: {rc1_date.strftime('%Y-%m-%d')}")
        print(f"  RC2: {rc2_date.strftime('%Y-%m-%d')}")
        print(f"  Days apart: {abs((rc1_date - rc2_date).days)}")
    else:
        print("No recent release couple found")

    # Example 3: Get latest for any asset type
    print("\n=== Latest Backup Assets ===")
    latest_backups = manager.get_latest_assets_by_rc(asset_type="backup_daily")

    for rc_name, asset_info in latest_backups.items():
        print(
            f"{rc_name}: {asset_info['timestamp'].strftime('%Y-%m-%d %H:%M')} - {asset_info['file_size'] / (1024**2):.1f} MB"
        )

    # Example 4: Get all verification types
    print("\n=== All Latest Verifications ===")
    all_verifications = manager.get_latest_verification_runs()

    for verification_type, runs in all_verifications.items():
        print(f"\n{verification_type}:")
        for run in runs:
            print(f"  {run['rc_name']}: {run['timestamp'].strftime('%Y-%m-%d')}")

    # Example 5: Simple function to get just the dates
    print("\n=== Quick Date Check ===")

    dates = get_latest_topology_dates("gdb_metadata.duckdb")
    if dates:
        rc1_date, rc2_date = dates
        print(f"Latest topology tests: RC1={rc1_date}, RC2={rc2_date}")
    else:
        print("No topology verification dates found")


# ============================================================================
# Script Integration Examples
# ============================================================================


def check_daily_qa_runs():
    """Example script to check if daily QA runs are up to date"""
    from datetime import datetime, timedelta

    # Get latest dates
    dates = get_latest_topology_dates("gdb_metadata.duckdb")
    if not dates:
        print("ERROR: No topology verification data found")
        return False

    rc1_date_str, rc2_date_str = dates
    rc1_date = datetime.strptime(rc1_date_str, "%Y-%m-%d")
    rc2_date = datetime.strptime(rc2_date_str, "%Y-%m-%d")

    # Check if tests are recent (within last 2 days)
    now = datetime.now()
    rc1_age = (now - rc1_date).days
    rc2_age = (now - rc2_date).days

    print(f"RC1 test age: {rc1_age} days")
    print(f"RC2 test age: {rc2_age} days")

    if rc1_age <= 2 and rc2_age <= 2:
        print("✅ QA tests are up to date")
        return True
    else:
        print("⚠️  QA tests may be outdated")
        return False


def find_qa_test_gaps():
    """Example script to find gaps in QA testing"""
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

    # Get all verification types and their latest runs
    verifications = manager.get_latest_verification_runs()

    print("QA Test Status Summary:")
    print("=" * 50)

    for verification_type, runs in verifications.items():
        print(f"\n{verification_type.replace('verification_', '').upper()}:")

        runs_by_rc = {run["rc_name"]: run for run in runs}

        for rc in ["RC1", "RC2"]:
            if rc in runs_by_rc:
                run = runs_by_rc[rc]
                age_days = (datetime.now() - run["timestamp"]).days
                status = "✅" if age_days <= 3 else "⚠️" if age_days <= 7 else "❌"
                print(
                    f"  {rc}: {run['timestamp'].strftime('%Y-%m-%d')} ({age_days}d ago) {status}"
                )
            else:
                print(f"  {rc}: No data found ❌")


if __name__ == "__main__":
    # Run examples
    example_usage()
    print("\n" + "=" * 60 + "\n")
    check_daily_qa_runs()
    print("\n" + "=" * 60 + "\n")
    find_qa_test_gaps()
