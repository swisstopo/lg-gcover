#!/usr/bin/env python
"""
Build script for gcover PyInstaller executable
Provides diagnostics and handles common build issues
"""
import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...\n")
    
    missing = []
    available = []
    
    # Core dependencies
    deps = [
        ('pyinstaller', 'PyInstaller'),
        ('geopandas', 'GeoPandas'),
        ('pandas', 'Pandas'),
        ('shapely', 'Shapely'),
        ('fiona', 'Fiona'),
        ('osgeo', 'GDAL/OGR'),
        ('loguru', 'Loguru'),
    ]
    
    # Add arcpy for Windows
    if platform.system() == 'Windows':
        deps.append(('arcpy', 'ArcPy'))
    
    for module, name in deps:
        try:
            __import__(module)
            available.append(f"  ✓ {name}")
        except ImportError:
            # Try alternative capitalization (conda vs pip)
            try:
                if module == 'pyinstaller':
                    __import__('PyInstaller')
                    available.append(f"  ✓ {name}")
                else:
                    raise
            except ImportError:
                missing.append(f"  ✗ {name} (module: {module})")
    
    print("\n".join(available))
    
    if missing:
        print("\n⚠️  Missing dependencies:")
        print("\n".join(missing))
        return False
    
    print("\n✅ All dependencies available\n")
    return True


def check_gdal_config():
    """Check GDAL configuration."""
    print("🌍 Checking GDAL configuration...\n")
    
    try:
        from osgeo import gdal, ogr, osr
        print(f"  ✓ GDAL version: {gdal.__version__}")
        
        # Check GDAL_DATA
        gdal_data = gdal.GetConfigOption('GDAL_DATA')
        if gdal_data:
            print(f"  ✓ GDAL_DATA: {gdal_data}")
        else:
            print(f"  ⚠️  GDAL_DATA not set")
        
        # Check PROJ_LIB
        proj_lib = os.environ.get('PROJ_LIB') or os.environ.get('PROJ_DATA')
        if proj_lib:
            print(f"  ✓ PROJ_LIB: {proj_lib}")
        else:
            print(f"  ⚠️  PROJ_LIB not set")
        
        # Test basic operations
        driver = ogr.GetDriverByName('GPKG')
        if driver:
            print(f"  ✓ GeoPackage driver available")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ GDAL check failed: {e}\n")
        return False


def clean_build_dirs():
    """Clean previous build directories."""
    print("🧹 Cleaning build directories...\n")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.pyc', '*.spec~']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed: {dir_name}/")
    
    print()


def run_pyinstaller(spec_file='geocover-qa-improved.spec', clean=False):
    """Run PyInstaller with the specified spec file."""
    print(f"🔨 Building with PyInstaller...\n")
    
    if not os.path.exists(spec_file):
        print(f"❌ Spec file not found: {spec_file}")
        return False
    
    cmd = ['pyinstaller']
    
    if clean:
        cmd.append('--clean')
    
    cmd.extend([
        '--noconfirm',  # Replace output directory without confirmation
        spec_file
    ])
    
    print(f"Running: {' '.join(cmd)}\n")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("="*80)
        print("\n✅ Build successful!\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print("="*80)
        print(f"\n❌ Build failed with error code {e.returncode}\n")
        return False


def test_executable():
    """Test the built executable."""
    print("🧪 Testing executable...\n")
    
    exe_path = None
    
    if platform.system() == 'Windows':
        exe_path = Path('dist/gcover.exe')
    else:
        exe_path = Path('dist/gcover')
    
    if not exe_path.exists():
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    print(f"  Found: {exe_path}")
    print(f"  Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Try to run with --help
    try:
        result = subprocess.run(
            [str(exe_path), '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"\n  ✅ Executable runs successfully!")
            print(f"\n  Output:\n{result.stdout[:500]}")
            return True
        else:
            print(f"\n  ❌ Executable failed with code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n  ⚠️  Executable timed out")
        return False
    except Exception as e:
        print(f"\n  ❌ Error running executable: {e}")
        return False


def main():
    """Main build workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build gcover with PyInstaller')
    parser.add_argument(
        '--spec',
        default='gcover.spec',
        help='Path to .spec file'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build before building'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency checks'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip executable test'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  GCover PyInstaller Build Script")
    print("="*80 + "\n")
    
    # Step 1: Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            print("❌ Missing dependencies. Install them first.")
            return 1
        
        if not check_gdal_config():
            print("⚠️  GDAL configuration issues detected.")
            print("    The build may work, but the executable might fail at runtime.")
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
    
    # Step 2: Clean build directories
    if args.clean:
        clean_build_dirs()
    
    # Step 3: Run PyInstaller
    if not run_pyinstaller(args.spec, args.clean):
        return 1
    
    # Step 4: Test executable
    if not args.skip_test:
        if not test_executable():
            print("\n⚠️  Executable test failed.")
            print("    The build completed, but the executable may not work correctly.")
            return 1
    
    print("\n" + "="*80)
    print("  ✅ Build completed successfully!")
    print("="*80)
    
    if platform.system() == 'Windows':
        print("\n📦 Executable: dist\\gcover.exe")
    else:
        print("\n📦 Executable: dist/gcover")
    
    print("\n💡 Next steps:")
    print("   1. Test the executable with your actual data")
    print("   2. Check that all GDAL/PROJ operations work correctly")
    print("   3. Test on a clean system without Python installed")
    
    if platform.system() == 'Windows':
        print("   4. Test on a system without ArcGIS Pro installed (if using arcpy)")
    
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
