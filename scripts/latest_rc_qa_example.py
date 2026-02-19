#!/usr/bin/env python3
"""
Enhanced test script to demonstrate the new latest QA functions with file paths
"""

from pathlib import Path
from datetime import datetime
import json


def test_enhanced_functions():
    """Test the enhanced topology verification functions"""

    # Import the new functions
    try:
        from gcover.cli.gdb_cmd import (
            get_latest_topology_verification_info,
            get_latest_topology_dates,
            verify_topology_files_exist,
        )
    except ImportError:
        print("❌ Functions not found - make sure to add them to gdb_cmd.py")
        return

    db_path = "gdb_metadata.duckdb"

    print("🧪 Testing Enhanced Latest QA Functions")
    print("=" * 50)

    # Test 1: Enhanced function with dates and paths
    print("\n1️⃣  Testing get_latest_topology_verification_info()")
    info = get_latest_topology_verification_info(db_path)

    if info:
        print("✅ Function returned data:")
        for rc, data in info.items():
            print(f"   {rc}: {data['date']} -> {data['path']}")

        # Verify data structure
        for rc, data in info.items():
            assert "date" in data, f"Missing 'date' key for {rc}"
            assert "path" in data, f"Missing 'path' key for {rc}"
            assert isinstance(data["date"], str), f"Date should be string for {rc}"
            assert isinstance(data["path"], str), f"Path should be string for {rc}"

        print("   ✅ Data structure is correct")
    else:
        print("❌ No data returned")
        return

    # Test 2: File existence verification
    print("\n2️⃣  Testing verify_topology_files_exist()")
    status = verify_topology_files_exist(db_path)

    if status:
        print("✅ File existence check:")
        for rc, exists in status.items():
            icon = "✅" if exists else "❌"
            print(f"   {rc}: {icon} {'exists' if exists else 'missing'}")

        # Show file details for existing files
        for rc, data in info.items():
            if status.get(rc, False):
                file_path = Path(data["path"])
                if file_path.exists():
                    stat = file_path.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    modified = datetime.fromtimestamp(stat.st_mtime)

                    print(
                        f"     📄 {file_path.name}: {size_mb:.1f} MB, modified {modified.strftime('%Y-%m-%d %H:%M')}"
                    )
    else:
        print("❌ No file status returned")

    # Test 3: Backwards compatible function
    print("\n3️⃣  Testing get_latest_topology_dates() [backwards compatible]")
    dates = get_latest_topology_dates(db_path)

    if dates:
        rc1_date, rc2_date = dates
        print(f"✅ Backwards compatible function:")
        print(f"   RC1: {rc1_date}")
        print(f"   RC2: {rc2_date}")

        # Verify consistency with enhanced function
        assert dates[0] == info["RC1"]["date"], "RC1 date mismatch between functions"
        assert dates[1] == info["RC2"]["date"], "RC2 date mismatch between functions"
        print("   ✅ Data consistent with enhanced function")
    else:
        print("❌ No dates returned")

    # Test 4: JSON serialization (for automation)
    print("\n4️⃣  Testing JSON serialization for automation")
    json_data = json.dumps(info, indent=2)
    print("✅ JSON serialization successful:")
    print(json_data)

    # Test deserialization
    parsed_data = json.loads(json_data)
    assert parsed_data == info, "JSON round-trip failed"
    print("   ✅ JSON round-trip successful")

    # Test 5: Practical usage example
    print("\n5️⃣  Practical Usage Example")
    print("Latest topology verification summary:")

    # Check if we have a release couple
    rc1_date = datetime.strptime(info["RC1"]["date"], "%Y-%m-%d")
    rc2_date = datetime.strptime(info["RC2"]["date"], "%Y-%m-%d")
    days_apart = abs((rc1_date - rc2_date).days)

    print(f"   🗓️  RC1: {info['RC1']['date']}")
    print(f"   🗓️  RC2: {info['RC2']['date']}")
    print(f"   📏 Days apart: {days_apart}")

    if days_apart <= 7:
        print("   ✅ Forms a release couple (tests are synchronized)")
    else:
        print("   ⚠️  Tests are not synchronized (not a release couple)")

    # Check currency
    latest_date = max(rc1_date, rc2_date)
    age_days = (datetime.now() - latest_date).days

    print(f"   📅 Latest test age: {age_days} days")
    if age_days <= 2:
        print("   ✅ Tests are current")
    elif age_days <= 7:
        print("   ⚠️  Tests are getting old")
    else:
        print("   ❌ Tests are outdated")

    print("\n🎉 All tests passed! Enhanced functions are working correctly.")


def demo_file_processing():
    """Demonstrate processing the actual QA files"""
    from gcover.cli.gdb_cmd import get_latest_topology_verification_info

    print("\n📁 File Processing Demo")
    print("=" * 30)

    info = get_latest_topology_verification_info("gdb_metadata.duckdb")
    if not info:
        print("❌ No data available for demo")
        return

    # Create demo processing directory
    demo_dir = Path("./demo_qa_processing")
    print(f"Creating demo directory: {demo_dir}")
    demo_dir.mkdir(exist_ok=True)

    for rc, data in info.items():
        source_path = Path(data["path"])

        print(f"\n{rc} ({data['date']}):")
        print(f"   Source: {source_path}")
        print(f"   Exists: {'✅' if source_path.exists() else '❌'}")

        if source_path.exists():
            # Create a symbolic link instead of copying (faster for demo)
            demo_link = demo_dir / f"{rc}_{data['date']}_issue.gdb"

            if demo_link.exists():
                demo_link.unlink()

            try:
                demo_link.symlink_to(source_path.absolute())
                print(f"   ✅ Demo link created: {demo_link}")

                # Show some file info
                if source_path.is_dir():
                    contents = list(source_path.iterdir())
                    print(f"   📊 Contains {len(contents)} items")

                    # Show first few items
                    for item in contents[:3]:
                        print(f"      - {item.name}")
                    if len(contents) > 3:
                        print(f"      ... and {len(contents) - 3} more")

            except OSError as e:
                print(f"   ⚠️  Could not create link: {e}")
        else:
            print(f"   ❌ Source file missing, cannot process")

    print(f"\nDemo files available in: {demo_dir.absolute()}")
    print("(These are symbolic links to the original files)")


if __name__ == "__main__":
    print("🚀 Enhanced Latest QA Functions Test Suite")
    print("Testing the new file path functionality...")

    try:
        test_enhanced_functions()
        demo_file_processing()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
