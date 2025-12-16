"""
CLI module for gcover.
"""
import sys
if sys.platform == "win32":
    try:
        import arcpy.geoprocessing
        import arcpy as _arcpy_force_preload
        # Success - arcpy loaded first, its GDAL is now initialized
    except Exception:
        # Will be handled by arcpy_compat later
        pass

from gcover.cli.main import cli, info, logs, tail_logs, show_logs

from gcover.cli.gdb_cmd import gdb

__all__ = ["cli","gdb", "logs", "tail_logs", "show_logs"]
