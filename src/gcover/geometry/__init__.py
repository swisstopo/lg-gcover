"""
Geometry functions for gcover.
"""

from .utils import (
    GeometryCleanup,
    GeometryValidator,
    GeometryProcessor,
    GeometryCleanupError,
    read_filegdb_layers,
    read_all_filegdb_layers,
    debug_geodataframe,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_LENGTH,
    DEFAULT_SLIVER_RATIO,
    DEFAULT_SELF_INTERSECTION_TOLERANCE,
)

__all__ = [
    'GeometryCleanup',
    'GeometryValidator',
    'GeometryProcessor',
    'GeometryCleanupError',
    'read_filegdb_layers',
    'read_all_filegdb_layers',
    'write_cleaned_data',
    'debug_geodataframe',
    'DEFAULT_MIN_AREA',
    'DEFAULT_MIN_LENGTH',
    'DEFAULT_SLIVER_RATIO',
    'DEFAULT_SELF_INTERSECTION_TOLERANCE',
]