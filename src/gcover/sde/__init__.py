"""
lg-gcover: GeoCover geodatabase interface tool

A Python package for working with Swiss GeoCover geodatabase data,
providing efficient CRUD operations, bulk processing, and data synchronization
with comprehensive version management and conflict resolution.
"""

from .bridge import GeoCoverBridge
from .connection import SDEConnectionManager, ReadOnlyError
# from ..utils.gpkg import write_gdf_to_gpkg, append_gdf_to_gpkg, GPKGWriter
from .exceptions import SDEConnectionError, SDEVersionError
from .config import (
    DEFAULT_OPERATOR,
    DEFAULT_VERSION,
    FEAT_CLASSES_SHORTNAMES,
    DB_INSTANCES,
    SWISS_EPSG,
)


__all__ = [
    # Main classes
    "GeoCoverBridge",
    "SDEConnectionManager",
    #"GPKGWriter",

    # Utility functions
    #"write_gdf_to_gpkg",
    # "append_gdf_to_gpkg",

    # Exceptions
    "ReadOnlyError",

    # Configuration
    "DEFAULT_OPERATOR",
    "DEFAULT_VERSION",
    "FEAT_CLASSES_SHORTNAMES",
    "DB_INSTANCES",
    "SWISS_EPSG",
]
