
from loguru import logger

from gcover.sde.bridge import GCoverSDEBridge, create_bridge, quick_export, quick_import
from gcover.sde.connection_manager import SDEConnectionManager
from gcover.sde.exceptions import SDEConnectionError, SDEVersionError

"""
SDE module - ESRI Enterprise Geodatabase connectivity.

This module uses lazy imports to avoid loading arcpy-dependent modules
unless they are actually used. This prevents import errors on systems
without ArcGIS Pro installed.
"""
from typing import TYPE_CHECKING

# Try to import arcpy - graceful degradation if not available
from gcover.arcpy_compat import HAS_ARCPY, arcpy



# Define __all__ for better IDE support
__all__ = [
    "GCoverSDEBridge",
    "SDEConnectionManager",
    "create_bridge",
    "quick_export",
    "quick_import",
    "HAS_ARCPY",
]
