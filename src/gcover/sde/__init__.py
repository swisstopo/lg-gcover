
from loguru import logger

from .bridge import GCoverSDEBridge, create_bridge, quick_export, quick_import
from .connection_manager import SDEConnectionManager
from .exceptions import SDEConnectionError, SDEVersionError

"""
SDE module - ESRI Enterprise Geodatabase connectivity.

This module uses lazy imports to avoid loading arcpy-dependent modules
unless they are actually used. This prevents import errors on systems
without ArcGIS Pro installed.
"""
from typing import TYPE_CHECKING

# Try to import arcpy - graceful degradation if not available
try:
    from gcover.arcpy_compat import HAS_ARCPY, arcpy

    HAS_ARCPY = True
    logger.info("arcpy is available - full functionality enabled")
except ImportError:
    HAS_ARCPY = False
    logger.warning("arcpy not available - using CIM JSON parsing only")


# Define __all__ for better IDE support
__all__ = [
    "GCoverSDEBridge",
    "SDEConnectionManager",
    "create_bridge",
    "quick_export",
    "quick_import",
    "HAS_ARCPY",
]
