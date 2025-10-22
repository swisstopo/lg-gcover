# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for gcover CLI tool
Handles GDAL, arcpy, and geospatial dependencies across platforms
"""
import os
import sys
from pathlib import Path
import site
import platform
import glob

# =============================================================================
# PLATFORM DETECTION
# =============================================================================

IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_gdal_files():
    """Find GDAL-related DLL/SO files for bundling."""
    datas = []
    
    if IS_WINDOWS:
        # Common Windows locations for GDAL dependencies
        possible_paths = [
            os.path.join(sys.prefix, 'Library', 'bin'),  # Conda
            os.path.join(sys.prefix, 'DLLs'),             # Standard Python
            os.path.join(site.getsitepackages()[0], 'osgeo'),  # pip
            r'C:\OSGeo4W64\bin',  # OSGeo4W installation
        ]
        
        # Patterns for required libraries
        patterns = [
            'gdal*.dll',
            'proj*.dll', 
            'geos*.dll',
            'sqlite3.dll',
            'spatialite.dll',
            'hdf5.dll',
            'netcdf.dll',
            'openjp2.dll',
            'expat.dll',
            'libcurl.dll',
            'zlib*.dll',
        ]
        
    elif IS_LINUX:
        possible_paths = [
            '/usr/lib',
            '/usr/lib/x86_64-linux-gnu',
            '/usr/local/lib',
            os.path.join(sys.prefix, 'lib'),
        ]
        
        patterns = [
            'libgdal*.so*',
            'libproj*.so*',
            'libgeos*.so*',
            'libsqlite*.so*',
            'libspatialite*.so*',
        ]
        
    elif IS_MACOS:
        possible_paths = [
            '/usr/local/lib',
            '/opt/homebrew/lib',
            os.path.join(sys.prefix, 'lib'),
        ]
        
        patterns = [
            'libgdal*.dylib',
            'libproj*.dylib',
            'libgeos*.dylib',
            'libsqlite*.dylib',
            'libspatialite*.dylib',
        ]
    
    # Search for files
    for path in possible_paths:
        if not os.path.exists(path):
            continue
            
        for pattern in patterns:
            for file in Path(path).glob(pattern):
                # Avoid duplicates
                if not any(str(file) == item[0] for item in datas):
                    datas.append((str(file), '.'))
                    print(f"Found: {file}")
    
    return datas


def find_gdal_data():
    """Find GDAL data directory (contains projections, datum files, etc.)."""
    datas = []
    
    # Try to find GDAL_DATA environment variable first
    gdal_data = os.environ.get('GDAL_DATA')
    
    if not gdal_data or not os.path.exists(gdal_data):
        # Search common locations
        possible_paths = []
        
        if IS_WINDOWS:
            possible_paths = [
                os.path.join(sys.prefix, 'Library', 'share', 'gdal'),
                r'C:\OSGeo4W64\share\gdal',
                os.path.join(site.getsitepackages()[0], 'osgeo', 'data', 'gdal'),
            ]
        else:
            possible_paths = [
                '/usr/share/gdal',
                '/usr/local/share/gdal',
                os.path.join(sys.prefix, 'share', 'gdal'),
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                gdal_data = path
                break
    
    if gdal_data and os.path.exists(gdal_data):
        datas.append((gdal_data, 'gdal-data'))
        print(f"Found GDAL_DATA: {gdal_data}")
    
    return datas


def find_proj_data():
    """Find PROJ data directory (contains projection definitions)."""
    datas = []
    
    proj_data = os.environ.get('PROJ_LIB') or os.environ.get('PROJ_DATA')
    
    if not proj_data or not os.path.exists(proj_data):
        possible_paths = []
        
        if IS_WINDOWS:
            possible_paths = [
                os.path.join(sys.prefix, 'Library', 'share', 'proj'),
                r'C:\OSGeo4W64\share\proj',
            ]
        else:
            possible_paths = [
                '/usr/share/proj',
                '/usr/local/share/proj',
                os.path.join(sys.prefix, 'share', 'proj'),
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                proj_data = path
                break
    
    if proj_data and os.path.exists(proj_data):
        datas.append((proj_data, 'proj-data'))
        print(f"Found PROJ_DATA: {proj_data}")
    
    return datas


def find_dateparser_data():
    """Find dateparser data files (timezone cache, etc.)."""
    datas = []
    
    try:
        import dateparser
        dateparser_path = Path(dateparser.__file__).parent
        data_path = dateparser_path / 'data'
        
        if data_path.exists():
            # Include all data files
            for data_file in data_path.glob('*'):
                if data_file.is_file():
                    datas.append((str(data_file), 'dateparser/data'))
                    print(f"Found dateparser data: {data_file.name}")
        else:
            print("Warning: dateparser data directory not found")
    except ImportError:
        print("dateparser not installed, skipping")
    
    return datas


def find_gdal_binaries():
    """Find GDAL binary extensions (_gdal, _ogr, _osr, etc.)."""
    binaries = []
    
    try:
        from osgeo import gdal
        import osgeo
        import sys
        
        osgeo_path = Path(osgeo.__file__).parent
        
        # Get Python version for extension naming
        py_version = f"{sys.version_info.major}{sys.version_info.minor}"
        py_version_dot = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Patterns for binary files (Python 3.13+ uses different naming)
        if IS_WINDOWS:
            patterns = [
                '_gdal*.pyd',
                '_ogr*.pyd',
                '_osr*.pyd',
                '_gdal_array*.pyd',
                '_gdalconst*.pyd'
            ]
        else:
            # Python 3.13+ uses: _gdal.cpython-313-x86_64-linux-gnu.so
            # Python 3.8-3.12 uses: _gdal.cpython-38-x86_64-linux-gnu.so or _gdal.so
            patterns = [
                f'_gdal*.cpython-{py_version}*.so',
                f'_ogr*.cpython-{py_version}*.so',
                f'_osr*.cpython-{py_version}*.so',
                f'_gdal_array*.cpython-{py_version}*.so',
                f'_gdalconst*.cpython-{py_version}*.so',
                # Fallback to generic patterns
                '_gdal*.so',
                '_ogr*.so',
                '_osr*.so',
                '_gdal_array*.so',
                '_gdalconst*.so',
            ]
        
        # Search for binary files
        found_any = False
        for pattern in patterns:
            for binary_file in osgeo_path.glob(pattern):
                # Avoid duplicates
                if not any(str(binary_file) == item[0] for item in binaries):
                    binaries.append((str(binary_file), 'osgeo'))
                    print(f"Found GDAL binary: {binary_file.name}")
                    found_any = True
        
        if not found_any:
            print(f"Warning: No GDAL binary extensions found in {osgeo_path}")
            print(f"  Python version: {py_version_dot}")
            print(f"  Searched for patterns like: _gdal*.cpython-{py_version}*.so")
            
            # Try site-packages directly
            for site_pkg in site.getsitepackages():
                osgeo_alt = Path(site_pkg) / 'osgeo'
                if osgeo_alt.exists():
                    for pattern in patterns:
                        for binary_file in osgeo_alt.glob(pattern):
                            if not any(str(binary_file) == item[0] for item in binaries):
                                binaries.append((str(binary_file), 'osgeo'))
                                print(f"Found GDAL binary (alt): {binary_file.name}")
    
    except ImportError as e:
        print(f"Warning: GDAL not found ({e}), skipping binary search")
    
    if not binaries:
        print("ERROR: No GDAL binaries found! The executable will not work.")
        print("Please check that GDAL is properly installed.")
    
    return binaries


def get_arcpy_dependencies():
    """Get arcpy-related hidden imports (Windows only)."""
    if not IS_WINDOWS:
        return []
    
    return [
        'arcpy',
        'arcgis',
        # Add more if needed based on your actual usage
    ]


# =============================================================================
# COLLECT DATA FILES
# =============================================================================

# Platform-specific binaries
platform_binaries = find_gdal_files() + find_gdal_binaries()

# GDAL/PROJ data directories
platform_datas = find_gdal_data() + find_proj_data() + find_dateparser_data()

# Get dateparser data files
def get_dateparser_data():
    """Include dateparser data files."""
    try:
        import dateparser
        dateparser_path = Path(dateparser.__file__).parent
        data_dir = dateparser_path / 'data'
        if data_dir.exists():
            return [(str(data_dir), 'dateparser/data')]
    except ImportError:
        pass
    return []

# Your application data files
app_datas = [
    ('src/gcover/data/*.gpkg', 'gcover/data'),
    ('src/gcover/data/*.json', 'gcover/data'),  # If you have JSON configs
    ('src/gcover/data/*.xlsx', 'gcover/data'),  # Config files
    # Add other data files as needed
] + get_dateparser_data()

# Combine all data files
all_datas = app_datas + platform_datas

# Hidden imports (including arcpy if on Windows)
hidden_imports = [
    'geopandas',
    'pandas',
    'shapely',
    'shapely.geometry',
    'fiona',
    'fiona._shim',
    'fiona.schema',
    'pyproj',
    'rtree',
    'loguru',
    'click',  # Click CLI framework
    # GDAL/OGR modules (including C extensions)
    'osgeo',
    'osgeo.gdal',
    'osgeo.ogr',
    'osgeo.osr',
    'osgeo.gdal_array',
    'osgeo._gdal',      # C extension
    'osgeo._ogr',       # C extension
    'osgeo._osr',       # C extension
    'osgeo._gdal_array', # C extension
    'dateparser',
    'dateparser.timezone_parser',
    'dateparser.data',
    'regex',  # Required by dateparser
    'rich',   # Rich text formatting
    'rich.console',
    'rich.table',
    'rich.progress',
    'pydantic',
    'pydantic_settings',
]

# Add all gcover modules explicitly
gcover_modules = [
    # Core package
    'gcover',
    'gcover.__init__',
    # CLI
    'gcover.cli',
    'gcover.cli.__init__',
    'gcover.cli.main',
    'gcover.cli.schema_cmd',
    'gcover.cli.gdb_cmd',
    'gcover.cli.qa_cmd',
    'gcover.cli.publish_cmd',
    'gcover.cli.sde_cmd',
    # Config
    'gcover.config',
    'gcover.config.__init__',
    # Utils
    'gcover.utils',
    'gcover.utils.__init__',
    'gcover.utils.logging',
    # Core modules
    'gcover.schema',
    'gcover.qa',
    'gcover.gdb',
    'gcover.sde',
    'gcover.publish',
    # SDE submodules
    'gcover.sde.connection_manager',
    'gcover.sde.bridge',
    # GDB submodules  
    'gcover.gdb.manager',
    # QA submodules
    'gcover.qa.analyzer',
    'gcover.qa.rules',
    # Any other submodules
    'gcover.arcpy_compat',
]

hidden_imports += gcover_modules + get_arcpy_dependencies()

# =============================================================================
# ANALYSIS
# =============================================================================

a = Analysis(
    ['src/gcover/cli/main.py'],
    pathex=['src'],  # Add src to path so imports work
    binaries=platform_binaries,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=['.'],  # Look for hooks in current directory
    hooksconfig={},
    runtime_hooks=['rthook_gdal.py'],  # Explicitly list runtime hooks
    excludes=[
        'matplotlib',  # Exclude if not needed
        'IPython',
        'notebook',
        'pytest',
        'sphinx',
        'tkinter',
    ],
    noarchive=False,
    optimize=0,
)

# =============================================================================
# PYZ (Python ZIP archive)
# =============================================================================

pyz = PYZ(a.pure)

# =============================================================================
# EXE (Executable)
# =============================================================================

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='gcover',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',  # Don't compress MSVC runtime
        'python*.dll',        # Don't compress Python DLL
    ] if IS_WINDOWS else [],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='favicon.ico' if os.path.exists('favicon.ico') else None,
)

# =============================================================================
# ALTERNATIVE: COLLECT (for one-folder distribution)
# =============================================================================
# Uncomment this section if you prefer a one-folder distribution
# (easier for debugging and can help with GDAL/arcpy dependencies)

# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='gcover',
# )

print("\n" + "="*80)
print("PyInstaller Configuration Summary")
print("="*80)
print(f"Platform: {platform.system()}")
print(f"Python: {sys.version}")
print(f"Binary files collected: {len(platform_binaries)}")
print(f"Data files collected: {len(all_datas)}")
print(f"Hidden imports: {len(hidden_imports)}")
print("="*80 + "\n")
