"""
Runtime hook to create symlinks for GDAL libraries.

PyInstaller bundles the actual .so files but not the symlinks.
This hook creates the necessary symlinks at runtime.
"""
import os
import sys
from pathlib import Path

def create_gdal_symlinks():
    """Create symlinks for GDAL libraries in the bundle."""
    if sys.platform != 'linux':
        return
    
    # Get the bundle directory
    bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    
    print(f"[Symlink Hook] Bundle dir: {bundle_dir}")
    
    # Map of symlink -> target
    symlinks_needed = {}
    
    # Find all versioned GDAL libraries
    for lib_file in bundle_dir.glob('libgdal.so.*.*.*'):
        # libgdal.so.37.3.11.3 -> libgdal.so.37
        parts = lib_file.name.split('.')
        if len(parts) >= 4:  # libgdal.so.37.3.11.3
            major_version = f"{parts[0]}.{parts[1]}.{parts[2]}"  # libgdal.so.37
            symlinks_needed[major_version] = lib_file.name
    
    for lib_file in bundle_dir.glob('libproj.so.*.*.*'):
        parts = lib_file.name.split('.')
        if len(parts) >= 4:
            major_version = f"{parts[0]}.{parts[1]}.{parts[2]}"
            symlinks_needed[major_version] = lib_file.name
    
    for lib_file in bundle_dir.glob('libgeos.so.*.*.*'):
        parts = lib_file.name.split('.')
        if len(parts) >= 4:
            major_version = f"{parts[0]}.{parts[1]}.{parts[2]}"
            symlinks_needed[major_version] = lib_file.name
    
    for lib_file in bundle_dir.glob('libgeos_c.so.*.*.*'):
        parts = lib_file.name.split('.')
        if len(parts) >= 4:
            major_version = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"  # libgeos_c.so.1
            symlinks_needed[major_version] = lib_file.name
    
    # Create symlinks
    for symlink_name, target_name in symlinks_needed.items():
        symlink_path = bundle_dir / symlink_name
        target_path = bundle_dir / target_name
        
        if not symlink_path.exists() and target_path.exists():
            try:
                os.symlink(target_name, str(symlink_path))
                print(f"[Symlink Hook] Created: {symlink_name} -> {target_name}")
            except OSError as e:
                print(f"[Symlink Hook] Failed to create {symlink_name}: {e}")
        elif symlink_path.exists():
            print(f"[Symlink Hook] Already exists: {symlink_name}")

# Run at import time
try:
    create_gdal_symlinks()
except Exception as e:
    print(f"[Symlink Hook] Error: {e}")
    import traceback
    traceback.print_exc()
