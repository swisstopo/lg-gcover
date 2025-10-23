"""Core utilities."""

from gcover.core.config import RELEASE_CANDIDATES, convert_rc, get_all_rcs, long_to_short
from gcover.core.geometry import (extract_valid_geometries, geometry_health_check,
                       load_gpkg_with_validation,
                       repair_self_intersections_with_quality_control,
                       split_features_by_mapsheets,
                       validate_and_repair_geometries)

__all__ = [
    "RELEASE_CANDIDATES",
    "long_to_short",
    "get_all_rcs",
    "convert_rc",
    "validate_and_repair_geometries",
    "load_gpkg_with_validation",
    "geometry_health_check",
    "repair_self_intersections_with_quality_control",
    "extract_valid_geometries",
    "split_features_by_mapsheets",
]
