"""
PyInstaller hook for osgeo (GDAL/OGR)
Ensures all GDAL components, binaries, and data are included
"""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

# Collect all osgeo submodules
hiddenimports = collect_submodules('osgeo')

# Ensure critical GDAL modules are included
hiddenimports += [
    'osgeo.gdal',
    'osgeo.ogr',
    'osgeo.osr',
    'osgeo.gdal_array',
    'osgeo.gdalconst',
    'osgeo._gdal',
    'osgeo._ogr',
    'osgeo._osr',
    'osgeo._gdal_array',
    'osgeo._gdalconst',
]

# Collect data files (GDAL_DATA, etc.)
datas = collect_data_files('osgeo', include_py_files=False)

# Collect binary/dynamic libraries
binaries = collect_dynamic_libs('osgeo')
print(binaries)
