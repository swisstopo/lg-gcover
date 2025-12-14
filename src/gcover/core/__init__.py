"""Core utilities."""

from gcover.core.config import (
    RELEASE_CANDIDATES,
    convert_rc,
    get_all_rcs,
    long_to_short,
)
from gcover.core.geometry import (
    extract_valid_geometries,
    geometry_health_check,
    load_gpkg_with_validation,
    repair_self_intersections_with_quality_control,
    safe_read_filegdb,
    split_features_by_mapsheets,
    validate_and_repair_geometries,
)

__all__ = [
    "get_all_rcs",
    "convert_rc",
    "long_to_short",
    "RELEASE_CANDIDATES",
    "safe_read_filegdb",
    "repair_self_intersections_with_quality_control",
    "validate_and_repair_geometries",
    "geometry_health_check",
    "load_gpkg_with_validation",
    "extract_valid_geometries",
    "split_features_by_mapsheets",
]
