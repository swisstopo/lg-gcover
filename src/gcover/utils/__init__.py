"""
Gcover utilities module.

IMPORTANT: This module uses lazy imports to avoid loading heavy dependencies
(geopandas, pandas, etc.) until they are actually needed.
"""
from typing import TYPE_CHECKING

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from gcover.gpkg import (
        GPKGWriter,
        append_gdf_to_gpkg,
        estimate_gpkg_size,
        read_gpkg_metadata
    )
    from gcover.geometry import (
        buffer_geometry,
        simplify_geometry,
        validate_geometry
    )
    # Add other type hints as needed

# DO NOT import modules that use geopandas/pandas here!
# They will be imported lazily when accessed

# Lightweight imports only (no heavy dependencies)

from gcover.utils.logging import gcover_logger, logger, setup_logging

__all__ = [
    # Logging
    "gcover_logger",
    "setup_logging",

    # GPKG utilities (lazy loaded)
    "GPKGWriter",
    "append_gdf_to_gpkg",
    "estimate_gpkg_size",
    "read_gpkg_metadata",

    # Geometry utilities (lazy loaded)
    "buffer_geometry",
    "simplify_geometry",
    "validate_geometry",
]


def __getattr__(name: str):
    """
    Lazy import mechanism for heavy dependencies.

    This prevents loading geopandas/pandas until actually needed.
    """
    # Map of attribute names to their modules
    _lazy_imports = {
        # GPKG utilities
        "GPKGWriter": "gcover.utils.gpkg",
        "append_gdf_to_gpkg": "gcover.utils.gpkg",
        "estimate_gpkg_size": "gcover.utils.gpkg",
        "read_gpkg_metadata": "gcover.utils.gpkg",

        # Geometry utilities
        "buffer_geometry": "gcover.utils.geometry",
        "simplify_geometry": "gcover.utils.geometry",
        "validate_geometry": "gcover.utils.geometry",
    }

    if name in _lazy_imports:
        import importlib
        module_path = _lazy_imports[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, name)

        # Cache the imported attribute
        globals()[name] = attr
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Optional: Helper to check if heavy dependencies are available
def _check_geo_dependencies() -> bool:
    """Check if geopandas is available without importing it."""
    try:
        import importlib.util
        return importlib.util.find_spec("geopandas") is not None
    except (ImportError, ValueError):
        return False
