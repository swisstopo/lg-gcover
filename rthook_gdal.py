"""
PyInstaller runtime hook for GDAL/PROJ configuration

This hook sets up the environment variables needed for GDAL and PROJ
to find their data files when running as a PyInstaller bundle.

Place this file in your project root and reference it in your .spec file:
    runtime_hooks=['rthook_gdal.py']
"""
import os
import sys
from pathlib import Path


def setup_gdal_environment():
    """Configure GDAL/PROJ environment for bundled executable."""
    
    # Check if running as a PyInstaller bundle
    if not getattr(sys, 'frozen', False):
        return  # Not running as bundled executable
    
    # Get the bundle directory
    bundle_dir = Path(sys._MEIPASS)
    
    print(f"[GDAL Hook] Running from bundle: {bundle_dir}")
    
    # Set GDAL_DATA path
    gdal_data = bundle_dir / 'gdal-data'
    if gdal_data.exists():
        os.environ['GDAL_DATA'] = str(gdal_data)
        print(f"[GDAL Hook] Set GDAL_DATA: {gdal_data}")
    else:
        print(f"[GDAL Hook] Warning: GDAL_DATA directory not found: {gdal_data}")
    
    # Set PROJ paths
    proj_data = bundle_dir / 'proj-data'
    if proj_data.exists():
        os.environ['PROJ_LIB'] = str(proj_data)
        os.environ['PROJ_DATA'] = str(proj_data)
        print(f"[GDAL Hook] Set PROJ_LIB: {proj_data}")
    else:
        print(f"[GDAL Hook] Warning: PROJ_DATA directory not found: {proj_data}")
    
    # Set GDAL_DRIVER_PATH for plugins (if needed)
    gdal_plugins = bundle_dir / 'gdalplugins'
    if gdal_plugins.exists():
        os.environ['GDAL_DRIVER_PATH'] = str(gdal_plugins)
        print(f"[GDAL Hook] Set GDAL_DRIVER_PATH: {gdal_plugins}")
    
    # Verify GDAL can be imported
    try:
        from osgeo import gdal
        print(f"[GDAL Hook] GDAL version: {gdal.__version__}")
        
        # Test GDAL configuration
        gdal.AllRegister()
        driver_count = gdal.GetDriverCount()
        print(f"[GDAL Hook] Available drivers: {driver_count}")
        
    except ImportError as e:
        print(f"[GDAL Hook] Error: Could not import GDAL: {e}")
    except Exception as e:
        print(f"[GDAL Hook] Error initializing GDAL: {e}")


# Execute setup when module is loaded
setup_gdal_environment()
