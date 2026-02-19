#!/usr/bin/env python
"""
Script de diagnostic pour comprendre l'ordre d'import et les conflits DLL.

Usage:
    python debug_import_order.py
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("DIAGNOSTIC IMPORT ORDER & DLL CONFLICTS")
print("=" * 80)

print("\n1. Python Environment Info:")
print(f"   Python: {sys.version}")
print(f"   Executable: {sys.executable}")
print(f"   Prefix: {sys.prefix}")

print("\n2. Critical Environment Variables:")
for var in ["PATH", "GDAL_DATA", "PROJ_LIB", "PYTHONPATH"]:
    value = os.environ.get(var, "NOT SET")
    if var == "PATH" and value != "NOT SET":
        # Show only ArcGIS-related paths
        paths = [p for p in value.split(";") if p and "ArcGIS" in p]
        print(f"   {var} (ArcGIS only):")
        for p in paths[:5]:  # First 5
            print(f"      {p}")
    else:
        print(f"   {var}: {value}")

print("\n3. Checking DLL files:")
dll_checks = [
    r"C:\Program Files\ArcGIS\Pro\bin\gdalplugins\ogr_Arrow.dll",
    r"C:\Program Files\ArcGIS\Pro\bin\gdalplugins\ogr_Parquet.dll",
]
for dll in dll_checks:
    exists = Path(dll).exists()
    print(f"   {'✓' if exists else '✗'} {dll}")

print("\n4. Import Order Test:")

# Test 1: Import arcpy directly (control)
print("\n   Test 1: Direct arcpy import")
try:
    import arcpy

    print("   ✓ arcpy imported successfully")
    info = arcpy.GetInstallInfo()
    print(f"     Version: {info.get('Version')}")
except Exception as e:
    print(f"   ✗ arcpy failed: {type(e).__name__}: {e}")

# Test 2: Check what's already imported
print("\n5. Already Imported Modules (relevant):")
relevant = ["arcpy", "arcgis", "geopandas", "pandas", "pyarrow", "fastparquet", "osgeo"]
for name in relevant:
    if name in sys.modules:
        module = sys.modules[name]
        location = getattr(module, "__file__", "built-in")
        print(f"   ✓ {name}: {location}")

# Test 3: Check for Arrow/Parquet conflicts
print("\n6. Arrow/Parquet Package Versions:")
packages = {
    "pyarrow": "pyarrow",
    "fastparquet": "fastparquet",
}
for display_name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        location = getattr(mod, "__file__", "unknown")
        print(f"   ✓ {display_name} {version}")
        print(f"     Location: {location}")
        # Check if it's from pip or conda
        if "site-packages" in location:
            print(f"     Source: pip")
        elif "ArcGIS" in location or "arcgispro" in location:
            print(f"     Source: ESRI")
        else:
            print(f"     Source: unknown")
    except ImportError:
        print(f"   ✗ {display_name}: not installed")

print("\n7. GDAL/OGR Info:")
try:
    from osgeo import gdal, ogr

    print(f"   ✓ GDAL version: {gdal.__version__}")
    print(f"   ✓ GDAL location: {gdal.__file__}")

    # Check for Arrow driver
    driver_count = ogr.GetDriverCount()
    arrow_found = False
    parquet_found = False
    for i in range(driver_count):
        driver = ogr.GetDriver(i)
        name = driver.GetName()
        if "Arrow" in name or "arrow" in name:
            arrow_found = True
            print(f"   ✓ Arrow driver: {name}")
        if "Parquet" in name or "parquet" in name:
            parquet_found = True
            print(f"   ✓ Parquet driver: {name}")

    if not arrow_found:
        print(f"   ✗ Arrow driver not found")
    if not parquet_found:
        print(f"   ✗ Parquet driver not found")

except Exception as e:
    print(f"   ✗ GDAL/OGR check failed: {e}")

print("\n8. Test Import Sequence (mimics your package):")
# Simulate what happens when gcover is imported
print("   a) Importing gcover.arcpy_compat...")
try:
    # Clear any previous import
    if "gcover.arcpy_compat" in sys.modules:
        del sys.modules["gcover.arcpy_compat"]

    from gcover.arcpy_compat import arcpy as arcpy_compat, HAS_ARCPY

    print(f"   ✓ arcpy_compat loaded, HAS_ARCPY={HAS_ARCPY}")
except Exception as e:
    print(f"   ✗ arcpy_compat failed: {type(e).__name__}: {e}")

print("\n   b) Importing other gcover modules...")
try:
    from gcover.publish import esri_classification_extractor

    print(f"   ✓ esri_classification_extractor loaded")
except Exception as e:
    print(f"   ✗ esri_classification_extractor failed: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

# Analyze and give recommendations
if "pyarrow" in sys.modules:
    pyarrow_location = sys.modules["pyarrow"].__file__
    if "site-packages" in pyarrow_location and "ArcGIS" not in pyarrow_location:
        print("⚠️  pyarrow is installed from pip, NOT from ESRI!")
        print("   This WILL cause conflicts with ArcGIS Pro's GDAL.")
        print("   Solution: Uninstall pip versions:")
        print("   pip uninstall pyarrow fastparquet")

if "fastparquet" in sys.modules:
    fp_location = sys.modules["fastparquet"].__file__
    if "site-packages" in fp_location and "ArcGIS" not in fp_location:
        print("⚠️  fastparquet is installed from pip, NOT from ESRI!")
        print("   Solution: Uninstall pip version:")
        print("   pip uninstall fastparquet")

print("\n✓ Run this after uninstalling conflicts:")
print("   python -c \"import arcpy; print('OK')\"")
print("   python -c \"from gcover.arcpy_compat import HAS_ARCPY; print(HAS_ARCPY)\"")