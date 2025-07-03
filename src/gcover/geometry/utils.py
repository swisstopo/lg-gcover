#!/usr/bin/env python3
"""
Geometry Cleanup Library for lg-gcover
=====================================

This module provides tools for cleaning up geometry and topology issues
in geological vector data, particularly for FileGDB layers.

Main features:
- Geometry validation and repair
- Multi-geometry to single geometry conversion
- Small geometry removal
- Sliver polygon detection and removal
- Self-intersection correction
- Duplicate UUID handling
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import unary_union
from shapely import validation
import structlog

logger = structlog.get_logger(__name__)

# Constants for cleanup operations
DEFAULT_MIN_AREA = 1.0  # m²
DEFAULT_MIN_LENGTH = 0.5  # m
DEFAULT_SLIVER_RATIO = 0.1  # ratio of area to perimeter²
DEFAULT_SELF_INTERSECTION_TOLERANCE = 0.01  # m


class GeometryCleanupError(Exception):
    """Base exception for geometry cleanup operations."""
    pass


class GeometryValidator:
    """Validates and reports geometry issues."""
    
    def __init__(self):
        self.issues = []
    
    def validate_geometry(self, gdf: gpd.GeoDataFrame) -> Dict[str, List[int]]:
        """
        Validate geometries and return issues by type.
        
        Args:
            gdf: GeoDataFrame to validate
            
        Returns:
            Dictionary with issue types as keys and lists of indices as values
        """
        issues = {
            'null_geometry': [],
            'invalid_geometry': [],
            'empty_geometry': [],
            'self_intersecting': []
        }
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Check for null geometry
            if geom is None:
                issues['null_geometry'].append(idx)
                continue
            
            # Check for empty geometry
            if geom.is_empty:
                issues['empty_geometry'].append(idx)
                continue
            
            # Check for invalid geometry
            if not geom.is_valid:
                issues['invalid_geometry'].append(idx)
                
                # Check specifically for self-intersection
                reason = validation.explain_validity(geom)
                if 'self-intersection' in reason.lower():
                    issues['self_intersecting'].append(idx)
        
        self.issues = issues
        return issues
    
    def report_issues(self, issues: Dict[str, List[int]]) -> str:
        """Generate a report of geometry issues."""
        report = []
        total_issues = sum(len(indices) for indices in issues.values())
        
        if total_issues == 0:
            return "No geometry issues found."
        
        report.append(f"Found {total_issues} geometry issues:")
        
        for issue_type, indices in issues.items():
            if indices:
                report.append(f"  - {issue_type}: {len(indices)} features")
        
        return "\n".join(report)


class GeometryProcessor:
    """Processes and cleans geometries."""
    
    def __init__(self, 
                 min_area: float = DEFAULT_MIN_AREA,
                 min_length: float = DEFAULT_MIN_LENGTH,
                 sliver_ratio: float = DEFAULT_SLIVER_RATIO,
                 self_intersection_tolerance: float = DEFAULT_SELF_INTERSECTION_TOLERANCE):
        self.min_area = min_area
        self.min_length = min_length
        self.sliver_ratio = sliver_ratio
        self.self_intersection_tolerance = self_intersection_tolerance
        
    def explode_multi_geometries(self, gdf: gpd.GeoDataFrame, 
                               uuid_column: str = 'UUID') -> gpd.GeoDataFrame:
        """
        Convert multi-geometries to single geometries.
        
        Args:
            gdf: Input GeoDataFrame
            uuid_column: Name of UUID column
            
        Returns:
            GeoDataFrame with exploded geometries
        """
        logger.info("Exploding multi-geometries to single geometries")
        
        exploded_rows = []
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Check if geometry is multi-type
            if isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon)):
                # Get individual geometries
                individual_geoms = list(geom.geoms)
                
                if len(individual_geoms) > 1:
                    # Find the largest geometry (by area for polygons, length for lines)
                    if isinstance(geom, MultiPolygon):
                        areas = [g.area for g in individual_geoms]
                        largest_idx = areas.index(max(areas))
                    elif isinstance(geom, MultiLineString):
                        lengths = [g.length for g in individual_geoms]
                        largest_idx = lengths.index(max(lengths))
                    else:  # MultiPoint
                        largest_idx = 0  # For points, just pick the first one
                    
                    # Keep original UUID for largest geometry
                    for i, individual_geom in enumerate(individual_geoms):
                        new_row = row.copy()
                        new_row.geometry = individual_geom
                        
                        # Generate new UUID for exploded parts (except largest)
                        if i != largest_idx:
                            new_row[uuid_column] = str(uuid.uuid4())
                        
                        exploded_rows.append(new_row)
                else:
                    # Single geometry in multi-geometry wrapper
                    new_row = row.copy()
                    new_row.geometry = individual_geoms[0]
                    exploded_rows.append(new_row)
            else:
                # Single geometry, keep as is
                exploded_rows.append(row)
        
        result_gdf = gpd.GeoDataFrame(exploded_rows, crs=gdf.crs)
        logger.info(f"Exploded {len(gdf)} features to {len(result_gdf)} features")
        
        return result_gdf
    
    def remove_small_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove geometries smaller than minimum thresholds."""
        logger.info("Removing small geometries")
        
        mask = pd.Series(True, index=gdf.index)
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            if isinstance(geom, Polygon):
                if geom.area < self.min_area:
                    mask[idx] = False
            elif isinstance(geom, LineString):
                if geom.length < self.min_length:
                    mask[idx] = False
            # Points are kept regardless of size
        
        result_gdf = gdf[mask].copy()
        removed_count = len(gdf) - len(result_gdf)
        
        logger.info(f"Removed {removed_count} small geometries")
        return result_gdf
    
    def remove_sliver_polygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove sliver polygons based on area/perimeter ratio."""
        logger.info("Removing sliver polygons")
        
        mask = pd.Series(True, index=gdf.index)
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            if isinstance(geom, Polygon) and geom.area > 0:
                # Calculate area/perimeter² ratio
                ratio = geom.area / (geom.length ** 2)
                
                if ratio < self.sliver_ratio:
                    mask[idx] = False
        
        result_gdf = gdf[mask].copy()
        removed_count = len(gdf) - len(result_gdf)
        
        logger.info(f"Removed {removed_count} sliver polygons")
        return result_gdf
    
    def fix_self_intersections(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fix self-intersecting polygons if the intersecting part is small."""
        logger.info("Fixing self-intersecting polygons")
        
        fixed_count = 0
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            if isinstance(geom, Polygon) and not geom.is_valid:
                reason = validation.explain_validity(geom)
                
                if 'self-intersection' in reason.lower():
                    try:
                        # Try to fix using buffer(0) technique
                        fixed_geom = geom.buffer(0)
                        
                        # Check if the fix is reasonable (area change is small)
                        if fixed_geom.is_valid and abs(fixed_geom.area - geom.area) < self.self_intersection_tolerance:
                            gdf.loc[idx, 'geometry'] = fixed_geom
                            fixed_count += 1
                        else:
                            logger.warning(f"Could not fix self-intersection for feature {idx}")
                    except Exception as e:
                        logger.warning(f"Error fixing self-intersection for feature {idx}: {e}")
        
        logger.info(f"Fixed {fixed_count} self-intersecting polygons")
        return gdf
    
    def remove_duplicate_uuids(self, gdf: gpd.GeoDataFrame, 
                             uuid_column: str = 'UUID') -> gpd.GeoDataFrame:
        """Remove duplicate UUIDs, keeping the geometry with the largest area."""
        logger.info("Removing duplicate UUIDs")
        
        # Find duplicates
        duplicates = gdf[gdf.duplicated(subset=[uuid_column], keep=False)]
        
        if len(duplicates) == 0:
            logger.info("No duplicate UUIDs found")
            return gdf
        
        # Group by UUID and keep the largest geometry
        def keep_largest(group):
            if len(group) == 1:
                return group
            
            # Calculate area/length based on geometry type
            if isinstance(group.geometry.iloc[0], Polygon):
                areas = group.geometry.apply(lambda x: x.area if hasattr(x, 'area') else 0)
                return group.loc[areas.idxmax:areas.idxmax]
            elif isinstance(group.geometry.iloc[0], LineString):
                lengths = group.geometry.apply(lambda x: x.length if hasattr(x, 'length') else 0)
                return group.loc[lengths.idxmax:lengths.idxmax]
            else:
                # For points, just keep the first one
                return group.iloc[:1]
        
        # Apply the function to each UUID group
        result_gdf = gdf.groupby(uuid_column, group_keys=False).apply(keep_largest)
        
        removed_count = len(gdf) - len(result_gdf)
        logger.info(f"Removed {removed_count} duplicate UUIDs")
        
        return result_gdf


class GeometryCleanup:
    """Main class for geometry cleanup operations."""
    
    def __init__(self, **kwargs):
        self.validator = GeometryValidator()
        self.processor = GeometryProcessor(**kwargs)
        
    def cleanup_geodataframe(self, 
                           gdf: gpd.GeoDataFrame,
                           operations: Dict[str, bool] = None,
                           uuid_column: str = 'UUID') -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Perform cleanup operations on a GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame
            operations: Dictionary of operations to perform
            uuid_column: Name of UUID column
            
        Returns:
            Tuple of (cleaned GeoDataFrame, cleanup report)
        """
        if operations is None:
            operations = {
                'validate': True,
                'explode_multi': True,
                'remove_small': True,
                'remove_slivers': True,
                'fix_self_intersections': True,
                'remove_duplicate_uuids': True
            }
        
        logger.info(f"Starting cleanup of {len(gdf)} features")
        
        # Create copy to avoid modifying original
        cleaned_gdf = gdf.copy()
        report = {'original_count': len(gdf)}
        
        # Validate geometry
        if operations.get('validate', True):
            issues = self.validator.validate_geometry(cleaned_gdf)
            report['validation_issues'] = issues
            logger.info(self.validator.report_issues(issues))
        
        # Explode multi-geometries
        if operations.get('explode_multi', True):
            cleaned_gdf = self.processor.explode_multi_geometries(cleaned_gdf, uuid_column)
            report['after_explode'] = len(cleaned_gdf)
        
        # Remove small geometries
        if operations.get('remove_small', True):
            cleaned_gdf = self.processor.remove_small_geometries(cleaned_gdf)
            report['after_remove_small'] = len(cleaned_gdf)
        
        # Remove sliver polygons
        if operations.get('remove_slivers', True):
            cleaned_gdf = self.processor.remove_sliver_polygons(cleaned_gdf)
            report['after_remove_slivers'] = len(cleaned_gdf)
        
        # Fix self-intersections
        if operations.get('fix_self_intersections', True):
            cleaned_gdf = self.processor.fix_self_intersections(cleaned_gdf)
            report['after_fix_intersections'] = len(cleaned_gdf)
        
        # Remove duplicate UUIDs
        if operations.get('remove_duplicate_uuids', True):
            cleaned_gdf = self.processor.remove_duplicate_uuids(cleaned_gdf, uuid_column)
            report['after_remove_duplicates'] = len(cleaned_gdf)
        
        report['final_count'] = len(cleaned_gdf)
        report['features_removed'] = report['original_count'] - report['final_count']
        
        logger.info(f"Cleanup complete: {report['original_count']} -> {report['final_count']} features")
        
        return cleaned_gdf, report


def read_filegdb_layers(gdb_path: Union[str, Path]) -> Dict[str, gpd.GeoDataFrame]:
    """
    Read all layers from a FileGDB.
    
    Args:
        gdb_path: Path to FileGDB
        
    Returns:
        Dictionary mapping layer names to GeoDataFrames
    """
    import fiona
    
    gdb_path = Path(gdb_path)
    
    if not gdb_path.exists():
        raise FileNotFoundError(f"FileGDB not found: {gdb_path}")
    
    layers = {}
    
    try:
        # List all layers in the GDB
        layer_names = fiona.listlayers(str(gdb_path))
        
        for layer_name in layer_names:
            logger.info(f"Reading layer: {layer_name}")
            try:
                gdf = gpd.read_file(str(gdb_path), layer=layer_name)
                layers[layer_name] = gdf
                logger.info(f"Loaded {len(gdf)} features from {layer_name}")
            except Exception as e:
                logger.error(f"Error reading layer {layer_name}: {e}")
                
    except Exception as e:
        logger.error(f"Error accessing FileGDB {gdb_path}: {e}")
        raise GeometryCleanupError(f"Could not read FileGDB: {e}")
    
    return layers


def write_cleaned_data(layers: Dict[str, gpd.GeoDataFrame], 
                      output_path: Union[str, Path],
                      output_format: str = 'gpkg') -> None:
    """
    Write cleaned layers to output format.
    
    Args:
        layers: Dictionary of layer names to GeoDataFrames
        output_path: Output file path
        output_format: Output format ('gpkg' or 'filegdb')
    """
    output_path = Path(output_path)
    
    if output_format.lower() == 'gpkg':
        # Write all layers to a single GeoPackage
        for layer_name, gdf in layers.items():
            logger.info(f"Writing layer {layer_name} to {output_path}")
            gdf.to_file(str(output_path), layer=layer_name, driver='GPKG')
            
    elif output_format.lower() == 'filegdb':
        # Write to FileGDB (requires arcpy or similar)
        try:
            import arcpy
            
            # Create FileGDB
            gdb_dir = output_path.parent
            gdb_name = output_path.name
            
            if not arcpy.Exists(str(output_path)):
                arcpy.management.CreateFileGDB(str(gdb_dir), gdb_name)
            
            # Write each layer
            for layer_name, gdf in layers.items():
                logger.info(f"Writing layer {layer_name} to FileGDB")
                # Convert to feature class
                temp_path = output_path.parent / f"temp_{layer_name}.shp"
                gdf.to_file(str(temp_path))
                
                # Copy to FileGDB
                arcpy.conversion.FeatureClassToFeatureClass(
                    str(temp_path), 
                    str(output_path), 
                    layer_name
                )
                
                # Clean up temp file
                temp_path.unlink()
                
        except ImportError:
            logger.error("arcpy not available. Cannot write to FileGDB format.")
            raise GeometryCleanupError("arcpy required for FileGDB output")
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example cleanup
    cleanup = GeometryCleanup(
        min_area=1.0,
        min_length=0.5,
        sliver_ratio=0.1
    )
    
    # This would be used with real data
    print("Geometry cleanup library loaded successfully!")
