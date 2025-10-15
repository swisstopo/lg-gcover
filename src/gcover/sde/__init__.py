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

# Check if arcpy is available without importing the whole module
from gcover.arcpy_compat import HAS_ARCPY

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from .bridge import GCoverSDEBridge
    from .connection_manager import SDEConnectionManager

# Module-level docstring for discoverability
__doc__ = """
GCover SDE Bridge - Enterprise Geodatabase connectivity.

This module requires arcpy (ArcGIS Pro) to be installed.

Example usage:
    from gcover.sde import create_bridge, quick_export

    # Export data
    quick_export("TOPGIS_GC.GC_BEDROCK", "output.gpkg")

    # Custom bridge
    with create_bridge(instance="GCOVERP") as bridge:
        gdf = bridge.export_to_geodataframe("TOPGIS_GC.GC_BEDROCK")
"""


def __getattr__(name: str):
    """
    Lazy import for SDE modules.

    This prevents importing arcpy-dependent modules until they're actually used.
    Raises a helpful error if arcpy is not available.
    """
    if not HAS_ARCPY:
        raise ImportError(
            f"Cannot import {name} from gcover.sde: arcpy is not available.\n"
            "Please install ArcGIS Pro and activate its Python environment."
        )

    # Map of public names to their module locations
    _module_map = {
        "GCoverSDEBridge": "gcover.sde.bridge",
        "SDEConnectionManager": "gcover.sde.connection_manager",
        "create_bridge": "gcover.sde.bridge",
        "quick_export": "gcover.sde.bridge",
        "quick_import": "gcover.sde.bridge",
    }

    if name in _module_map:
        import importlib

        module_path = _module_map[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, name)

        # Cache the imported attribute
        globals()[name] = attr
        return attr

    raise AttributeError(f"module 'gcover.sde' has no attribute '{name}'")


# Define __all__ for better IDE support
__all__ = [
    "GCoverSDEBridge",
    "SDEConnectionManager",
    "create_bridge",
    "quick_export",
    "quick_import",
    "HAS_ARCPY",
]
