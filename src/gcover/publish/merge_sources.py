# src/gcover/publish/merge_sources.py
"""
Module for merging multiple FileGDB sources into a single publication GDB.

Supports clipping features from different source databases (RC1, RC2, custom GDBs)
based on mapsheet boundaries defined in administrative zones.

Optimized for performance:
- Dissolves mapsheets by source (4 clips instead of 221)
- Uses convex hull masks for fast spatial filtering
- Only clips features that actually cross boundaries
- Single Swiss border clip at the end
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from shapely import get_coordinates, set_coordinates, within, intersects, intersection
from shapely.geometry import (
    GeometryCollection, 
    LineString, 
    MultiLineString,
    MultiPoint,
    MultiPolygon, 
    Point, 
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union
from shapely import make_valid, simplify


from gcover.core.geometry import load_gpkg_with_validation, validate_and_repair_geometries

console = Console()

has_pyarrow = False
try:
    import pyarrow
    console.print("PyArrow version:", pyarrow.__version__)
    has_pyarrow = True
except ImportError:
    console.print("[red]PyArrow not installed[/red]")


console.print("[yellow]Suppressing some OGR warning (unclosed rings, only CCW, etc.[/yellow]")
# Suppress pandas fragmentation warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
os.environ['OGR_GEOMETRY_ACCEPT_UNCLOSED_RING'] = 'NO'
os.environ['METHOD'] = 'ONLY_CCW'


# =============================================================================
# FAST CLIP UTILITIES (OPTIMIZED)
# =============================================================================

def create_fast_mask(mapsheets: gpd.GeoDataFrame, source_rc: str) -> BaseGeometry:
    """
    Create a fast clip mask using convex hull.
    
    This dramatically reduces vertex count for complex boundaries (e.g., Swiss border)
    while still containing all features from the source.
    
    Args:
        mapsheets: GeoDataFrame with mapsheet polygons
        source_rc: Source identifier (e.g., 'RC1', 'RC2')
        
    Returns:
        Convex hull of dissolved mapsheets for this source
    """
    sheets = mapsheets[mapsheets['SOURCE_RC'] == source_rc]
    if sheets.empty:
        return None
    merged = unary_union(sheets.geometry.values)
    return merged.convex_hull


def create_exact_mask(mapsheets: gpd.GeoDataFrame, source_rc: str, tolerance: float = 0.1) -> BaseGeometry:
    """
    Create an exact (but simplified) clip mask.
    
    Use this when you need precise clipping at borders.
    
    Args:
        mapsheets: GeoDataFrame with mapsheet polygons
        source_rc: Source identifier
        tolerance: Simplification tolerance in CRS units (default 0.1m for LV95)
        
    Returns:
        Simplified dissolved geometry
    """
    sheets = mapsheets[mapsheets['SOURCE_RC'] == source_rc]
    if sheets.empty:
        return None
        
    # Dissolve to single geometry
    merged = unary_union(sheets.geometry.values)
    
    # Remove internal slivers/boundaries with buffer trick
    cleaned = merged.buffer(0.01).buffer(-0.01)
    
    # Simplify vertices
    simplified = simplify(cleaned, tolerance=tolerance, preserve_topology=True)
    
    return make_valid(simplified)


def fast_clip(gdf: gpd.GeoDataFrame, mask: BaseGeometry) -> gpd.GeoDataFrame:
    """
    Optimized clip: only compute intersection for features crossing the boundary.
    
    For typical geological data, 99%+ of features are fully inside the mask,
    so we can skip the expensive intersection computation for them.
    
    Performance: ~50x faster than gpd.clip() for large datasets with simple masks.
    
    Args:
        gdf: Input GeoDataFrame
        mask: Clip mask geometry
        
    Returns:
        Clipped GeoDataFrame
    """
    if gdf.empty:
        return gdf
        
    geoms = gdf.geometry.values
    
    # Vectorized predicates (shapely 2.0 - very fast)
    is_within = within(geoms, mask)
    does_intersect = intersects(geoms, mask)
    needs_clip = does_intersect & ~is_within
    
    # Log statistics
    n_total = len(gdf)
    n_inside = is_within.sum()
    n_clip = needs_clip.sum()
    n_outside = (~does_intersect).sum()
    
    logger.debug(f"Fast clip: {n_total:,} total, {n_inside:,} inside ({n_inside/n_total*100:.1f}%), "
                 f"{n_clip:,} need clipping, {n_outside:,} outside")
    
    # Start with all intersecting features
    result = gdf[does_intersect].copy()
    
    if result.empty:
        return result
    
    # Only clip the boundary-crossing features
    if needs_clip.any():
        # Reindex mask to result DataFrame
        clip_idx = needs_clip[does_intersect]
        result.loc[clip_idx, 'geometry'] = intersection(
            result.loc[clip_idx, 'geometry'].values,
            mask
        )
    
    return result


def fast_clip_with_stats(
    gdf: gpd.GeoDataFrame, 
    mask: BaseGeometry,
    label: str = ""
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Fast clip with detailed statistics for debugging/monitoring.
    
    Args:
        gdf: Input GeoDataFrame
        mask: Clip mask geometry
        label: Label for logging
        
    Returns:
        Tuple of (clipped GeoDataFrame, statistics dict)
    """
    if gdf.empty:
        return gdf, {"total": 0, "inside": 0, "clipped": 0, "outside": 0}
        
    geoms = gdf.geometry.values
    
    is_within = within(geoms, mask)
    does_intersect = intersects(geoms, mask)
    needs_clip = does_intersect & ~is_within
    
    stats = {
        "total": len(gdf),
        "inside": int(is_within.sum()),
        "clipped": int(needs_clip.sum()),
        "outside": int((~does_intersect).sum()),
    }
    
    if label:
        console.print(f"  {label}: {stats['inside']:,} inside, {stats['clipped']:,} clipped, "
                     f"{stats['outside']:,} outside")
    
    result = gdf[does_intersect].copy()
    
    if needs_clip.any():
        clip_idx = needs_clip[does_intersect]
        result.loc[clip_idx, 'geometry'] = intersection(
            result.loc[clip_idx, 'geometry'].values,
            mask
        )
    
    return result, stats


def get_mask_complexity(mask: BaseGeometry) -> Dict:
    """Get complexity metrics for a mask geometry."""
    if mask is None:
        return {"type": None, "polygons": 0, "vertices": 0}
        
    n_verts = sum(
        len(p.exterior.coords) 
        for p in getattr(mask, 'geoms', [mask])
    )
    
    return {
        "type": mask.geom_type,
        "polygons": len(mask.geoms) if hasattr(mask, 'geoms') else 1,
        "vertices": n_verts,
    }


# =============================================================================
# GEOMETRY NORMALIZATION UTILITIES
# =============================================================================

def get_expected_geometry_type(layer_name: str) -> str:
    """
    Get expected geometry type for a layer based on its name.
    
    Returns one of: 'MultiPolygon', 'MultiLineString', 'MultiPoint', 'Point'
    """
    layer_base = layer_name.split("/")[-1].upper()
    
    # Point layers
    if any(x in layer_base for x in ["POINT", "FOSSILS", "_PT"]):
        return "MultiPoint"
    
    # Line layers
    if any(x in layer_base for x in ["LINE", "LINEAR"]):
        return "MultiLineString"
    
    # Default to polygon
    return "MultiPolygon"


def extract_geometries_by_type(
    geometry, 
    target_type: str,
    min_area: float = 1e-10
) -> List:
    """
    Extract geometries of specific type from any geometry (including GeometryCollection).
    
    Args:
        geometry: Input shapely geometry
        target_type: One of 'MultiPolygon', 'MultiLineString', 'MultiPoint', 'Point'
        min_area: Minimum area for polygons
        
    Returns:
        List of extracted geometries of the target type
    """
    if geometry is None or geometry.is_empty:
        return []
    
    results = []
    
    # Define compatible types for each target
    type_mapping = {
        "MultiPolygon": ["Polygon", "MultiPolygon"],
        "MultiLineString": ["LineString", "MultiLineString"],
        "MultiPoint": ["Point", "MultiPoint"],
        "Point": ["Point"],
    }
    
    compatible_types = type_mapping.get(target_type, [target_type])
    
    def process_geom(geom):
        """Recursively process geometry."""
        if geom is None or geom.is_empty:
            return
            
        geom_type = geom.geom_type
        
        if geom_type == "GeometryCollection":
            for sub_geom in geom.geoms:
                process_geom(sub_geom)
        elif geom_type in compatible_types:
            # For polygons, check minimum area
            if geom_type in ["Polygon", "MultiPolygon"]:
                if geom.area > min_area:
                    results.append(geom)
            else:
                results.append(geom)
        elif geom_type == "MultiPolygon" and "Polygon" in compatible_types:
            for poly in geom.geoms:
                if poly.area > min_area:
                    results.append(poly)
        elif geom_type == "MultiLineString" and "LineString" in compatible_types:
            for line in geom.geoms:
                if line.length > 0:
                    results.append(line)
        elif geom_type == "MultiPoint" and "Point" in compatible_types:
            for pt in geom.geoms:
                results.append(pt)
    
    process_geom(geometry)
    return results


def normalize_geometry(
    geometry,
    target_type: str,
    preserve_z: bool = True,
    min_area: float = 1e-10
):
    """
    Normalize a geometry to the target type, handling 3D coordinates.
    
    Args:
        geometry: Input shapely geometry
        target_type: Target geometry type ('MultiPolygon', 'MultiLineString', 'MultiPoint')
        preserve_z: Whether to preserve Z coordinates
        min_area: Minimum area for polygon geometries
        
    Returns:
        Normalized geometry of target type, or None if conversion fails
    """
    if geometry is None or geometry.is_empty:
        return None
    
    # Make valid first
    if not geometry.is_valid:
        geometry = make_valid(geometry)
        if geometry is None or geometry.is_empty:
            return None
    
    # Extract compatible geometries
    extracted = extract_geometries_by_type(geometry, target_type, min_area)
    
    if not extracted:
        return None
    
    # Combine into target Multi* type
    if target_type == "MultiPolygon":
        polygons = []
        for geom in extracted:
            if geom.geom_type == "Polygon":
                polygons.append(geom)
            elif geom.geom_type == "MultiPolygon":
                polygons.extend(geom.geoms)
        if not polygons:
            return None
        result = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
        
    elif target_type == "MultiLineString":
        lines = []
        for geom in extracted:
            if geom.geom_type == "LineString":
                lines.append(geom)
            elif geom.geom_type == "MultiLineString":
                lines.extend(geom.geoms)
        if not lines:
            return None
        result = MultiLineString(lines) if len(lines) > 1 else lines[0]
        
    elif target_type in ["MultiPoint", "Point"]:
        points = []
        for geom in extracted:
            if geom.geom_type == "Point":
                points.append(geom)
            elif geom.geom_type == "MultiPoint":
                points.extend(geom.geoms)
        if not points:
            return None
        if target_type == "Point" and len(points) == 1:
            result = points[0]
        else:
            result = MultiPoint(points) if len(points) > 1 else points[0]
    else:
        result = extracted[0] if len(extracted) == 1 else None
    
    return result


def normalize_geodataframe_geometries(
    gdf: gpd.GeoDataFrame,
    target_type: str,
    preserve_z: bool = True,
    min_area: float = 1e-10
) -> gpd.GeoDataFrame:
    """
    Normalize all geometries in a GeoDataFrame to a consistent type.
    
    Handles:
    - GeometryCollection results from clipping
    - Mixed geometry types
    - 3D coordinate preservation
    - Invalid geometries
    
    Args:
        gdf: Input GeoDataFrame
        target_type: Target geometry type
        preserve_z: Whether to preserve Z coordinates
        min_area: Minimum area for polygons
        
    Returns:
        GeoDataFrame with normalized geometries
    """
    if gdf.empty:
        return gdf
    
    # Store original CRS
    original_crs = gdf.crs
    
    # Process each geometry
    normalized_geoms = []
    valid_indices = []
    
    for idx, geom in gdf.geometry.items():
        norm_geom = normalize_geometry(geom, target_type, preserve_z, min_area)
        if norm_geom is not None and not norm_geom.is_empty:
            normalized_geoms.append(norm_geom)
            valid_indices.append(idx)
    
    if not normalized_geoms:
        logger.warning(f"No valid geometries after normalization to {target_type}")
        return gpd.GeoDataFrame(columns=gdf.columns, crs=original_crs)
    
    # Create result GeoDataFrame
    result = gdf.loc[valid_indices].copy()
    result.geometry = normalized_geoms
    
    # Ensure CRS is set
    if result.crs is None:
        result.set_crs(original_crs, inplace=True)
    
    removed_count = len(gdf) - len(result)
    if removed_count > 0:
        logger.debug(f"Removed {removed_count} geometries during normalization")
    
    return result


def force_2d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Force geometries to 2D by removing Z coordinates.
    
    Use this as a fallback if 3D writing fails.
    """
    if gdf.empty:
        return gdf
    
    def remove_z(geom):
        if geom is None or geom.is_empty:
            return geom
        coords = get_coordinates(geom, include_z=True)
        if coords.shape[1] > 2:
            coords_2d = coords[:, :2]
            return set_coordinates(geom, coords_2d)
        return geom
    
    result = gdf.copy()
    result.geometry = result.geometry.apply(remove_z)
    return result


# =============================================================================
# DATA CLASSES
# =============================================================================

class SourceType(Enum):
    """Type of source database."""
    RC1 = "RC1"
    RC2 = "RC2"
    CUSTOM = "CUSTOM"


@dataclass
class SourceConfig:
    """Configuration for a source database."""
    name: str
    path: Path
    source_type: SourceType
    
    def exists(self) -> bool:
        return self.path.exists()


@dataclass 
class MergeConfig:
    """Configuration for the merge operation."""
    
    # Input sources
    rc1_path: Optional[Path] = None
    rc2_path: Optional[Path] = None
    custom_sources_dir: Optional[Path] = None
    
    # Mapsheet configuration
    admin_zones_path: Path = None
    mapsheets_layer: str = "mapsheets_sources_only"
    source_column: str = "SOURCE_RC"  # or "SOURCE_QA"
    mapsheet_nbr_column: str = "MSH_MAP_NBR"
    
    # Output
    output_path: Path = None
    
    # Processing options
    spatial_layers: List[str] = field(default_factory=lambda: [
        "GC_ROCK_BODIES/GC_BEDROCK",
        "GC_ROCK_BODIES/GC_UNCO_DESPOSIT", 
        "GC_ROCK_BODIES/GC_SURFACES",
        "GC_ROCK_BODIES/GC_LINEAR_OBJECTS",
        "GC_ROCK_BODIES/GC_POINT_OBJECTS",
        "GC_ROCK_BODIES/GC_FOSSILS",
        "GC_ROCK_BODIES/GC_EXPLOIT_GEOMAT_PLG",
        "GC_ROCK_BODIES/GC_EXPLOIT_GEOMAT_PT",
        "GC_ROCK_BODIES/GC_MAPSHEET",
    ])
    
    non_spatial_tables: List[str] = field(default_factory=lambda: [
        "GC_GEOL_MAPPING_UNIT",
        "GC_GEOL_MAPPING_UNIT_ATT",
        "GC_SYSTEM",
        "GC_COMPOSIT",
        "GC_ADMIXTURE",
        "GC_CHARCAT",
        "GC_LITHO",
        "GC_CHRONO",
        "GC_TECTO",
        "GC_LITHO_UNCO",
        "GC_LITSTRAT_UNCO",
        "GC_CORRELATION",
        "GC_LITSTRAT_FORMATION_BANK",
    ])
    
    # Reference source for non-spatial tables (defaults to RC2 if available)
    reference_source: str = "RC2"
    
    # Mapsheet filter (process only specific mapsheets)
    mapsheet_numbers: Optional[List[int]] = None
    
    # Processing parameters
    clip_buffer: float = 0.0  # Small buffer for edge handling
    preserve_z: bool = True
    
    # Optimization options
    use_convex_hull_masks: bool = True  # Use fast convex hull masks
    clip_to_swiss_border: bool = True   # Apply final Swiss border clip
    validate_geometries: bool = True    # Validate geometries on read
    
    # Fields to exclude from output (metadata fields, etc.)
    exclude_fields: Optional[List[str]] = None
    

@dataclass
class MergeStats:
    """Statistics from merge operation."""
    layers_processed: int = 0
    tables_copied: int = 0
    features_per_layer: Dict[str, int] = field(default_factory=dict)
    features_per_source: Dict[str, int] = field(default_factory=dict)
    mapsheets_processed: int = 0
    clip_stats: Dict[str, Dict] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# MAIN MERGER CLASS
# =============================================================================

class GDBMerger:
    """
    Merges multiple FileGDB sources into a single output based on mapsheet boundaries.
    
    Optimizations:
    - Dissolves mapsheets by source (4 clips instead of 221)
    - Uses convex hull masks for fast spatial filtering
    - Only clips features that actually cross boundaries (~99% skip expensive intersection)
    - Single Swiss border clip at the end
    """
    
    def __init__(self, config: MergeConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.sources: Dict[str, SourceConfig] = {}
        self.mapsheets_gdf: Optional[gpd.GeoDataFrame] = None
        self.source_masks: Dict[str, BaseGeometry] = {}  # Dissolved masks per source
        self.swiss_border: Optional[BaseGeometry] = None
        self.stats = MergeStats()
        
        logger.debug("Initializing GDBMerger (geopandas, optimized)")
        self._setup_sources()
    
    def _setup_sources(self) -> None:
        """Configure available source databases."""
        
        logger.debug("Setting up sources...")
        
        # Standard sources
        if self.config.rc1_path and self.config.rc1_path.exists():
            self.sources["RC1"] = SourceConfig(
                name="RC1",
                path=self.config.rc1_path,
                source_type=SourceType.RC1
            )
            logger.debug(f"  Registered RC1 source: {self.config.rc1_path}")
            
        if self.config.rc2_path and self.config.rc2_path.exists():
            self.sources["RC2"] = SourceConfig(
                name="RC2", 
                path=self.config.rc2_path,
                source_type=SourceType.RC2
            )
            logger.debug(f"  Registered RC2 source: {self.config.rc2_path}")
            
        # Custom sources from directory
        if self.config.custom_sources_dir and self.config.custom_sources_dir.exists():
            for gdb_path in self.config.custom_sources_dir.glob("*.gdb"):
                source_name = gdb_path.name  # e.g., "Saas.gdb"
                self.sources[source_name] = SourceConfig(
                    name=source_name,
                    path=gdb_path,
                    source_type=SourceType.CUSTOM
                )
                logger.debug(f"  Registered custom source: {gdb_path}")
                
        logger.info(f"Configured {len(self.sources)} source(s)")
                
    def _load_mapsheets(self) -> gpd.GeoDataFrame:
        """Load mapsheets with source assignments."""
        
        logger.info(f"Loading mapsheets from {self.config.admin_zones_path}")
        logger.debug(f"  Layer: {self.config.mapsheets_layer}")
        logger.debug(f"  Source column: {self.config.source_column}")
        
        gdf = gpd.read_file(
            self.config.admin_zones_path,
            layer=self.config.mapsheets_layer
        )
        
        logger.debug(f"  Loaded {len(gdf)} total mapsheets")
        
        # Verify required columns
        required_cols = [self.config.source_column, self.config.mapsheet_nbr_column]
        missing = [col for col in required_cols if col not in gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns in mapsheets: {missing}")
        
        # Store original mapsheets for Swiss border
        self.swiss_border = unary_union(gdf.geometry.values)
        
        # Filter by mapsheet numbers if specified
        if self.config.mapsheet_numbers:
            gdf = gdf[gdf[self.config.mapsheet_nbr_column].isin(self.config.mapsheet_numbers)]
            logger.info(f"Filtered to {len(gdf)} mapsheets")
            # Update Swiss border to filtered area
            self.swiss_border = unary_union(gdf.geometry.values)
            
        self.mapsheets_gdf = gdf
        return gdf
    
    def _create_source_masks(self) -> Dict[str, BaseGeometry]:
        """
        Create optimized clip masks for each source.
        
        Uses convex hull for speed, or exact dissolved boundaries if configured.
        """
        if self.mapsheets_gdf is None:
            self._load_mapsheets()
        
        # Rename column for consistency
        source_col = self.config.source_column
        self.mapsheets_gdf = self.mapsheets_gdf.rename(columns={source_col: 'SOURCE_RC'})
        
        masks = {}
        
        for source_name in self.mapsheets_gdf['SOURCE_RC'].unique():
            if self.config.use_convex_hull_masks:
                mask = create_fast_mask(self.mapsheets_gdf, source_name)
            else:
                mask = create_exact_mask(self.mapsheets_gdf, source_name)
            
            if mask is not None:
                masks[source_name] = mask
                complexity = get_mask_complexity(mask)
                logger.debug(f"  {source_name}: {complexity['type']}, {complexity['vertices']:,} vertices")
        
        self.source_masks = masks
        return masks
    
    def _resolve_source_path(self, source_name: str) -> Optional[Path]:
        """Resolve source name to actual GDB path."""
        
        # Direct match
        if source_name in self.sources:
            return self.sources[source_name].path
            
        # Try with .gdb extension
        source_with_ext = f"{source_name}.gdb" if not source_name.endswith(".gdb") else source_name
        if source_with_ext in self.sources:
            return self.sources[source_with_ext].path
            
        # Check if it's a path in custom sources directory
        if self.config.custom_sources_dir:
            potential_path = self.config.custom_sources_dir / source_with_ext
            if potential_path.exists():
                return potential_path
                
        return None
    
    def _get_layer_path(self, gdb_path: Path, layer_name: str) -> str:
        """Get the full path to a layer in a GDB."""
        # Handle grouped layers (e.g., "GC_ROCK_BODIES/GC_BEDROCK")
        if "/" in layer_name:
            return layer_name.split("/")[-1]
        return layer_name
    
    def _read_layer_for_source(
        self, 
        gdb_path: Path, 
        layer_name: str,
        mask: BaseGeometry
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Read a layer from a FileGDB with spatial filtering.
        
        Uses mask bounds for efficient reading.
        """
        actual_layer = self._get_layer_path(gdb_path, layer_name)
        
        try:
            # Use mask geometry for spatial filtering during read
            gdf = gpd.read_file(
                gdb_path, 
                layer=actual_layer, 
                engine='pyogrio',
                mask=mask  # Spatial filter on read
            )
            
            if self.config.validate_geometries and not gdf.empty:
                gdf = validate_and_repair_geometries(gdf)
            
            return gdf
            
        except Exception as e:
            logger.warning(f"Could not read {actual_layer} from {gdb_path}: {e}")
            return None
    
    def _merge_spatial_layer(
        self,
        layer_name: str,
        progress: Optional[Progress] = None,
        task_id: Optional[int] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Merge a single spatial layer from all sources using optimized clip.
        
        Key optimizations:
        - One dissolved mask per source (not per mapsheet)
        - Fast clip that skips features fully inside mask
        """
        merged_parts = []
        layer_feature_count = 0
        layer_crs = None
        
        # Get expected geometry type for this layer
        expected_type = get_expected_geometry_type(layer_name)
        logger.debug(f"Processing layer {layer_name}, expected type: {expected_type}")
        
        for source_name, mask in self.source_masks.items():
            # Resolve source path
            source_path = self._resolve_source_path(source_name)
            if source_path is None:
                msg = f"Source not found: {source_name}"
                logger.warning(msg)
                self.stats.warnings.append(msg)
                continue
            
            layer_start = time.time()
            
            # Read layer with spatial filter
            gdf = self._read_layer_for_source(source_path, layer_name, mask)
            if gdf is None or gdf.empty:
                logger.debug(f"  {source_name}: no features in {layer_name}")
                continue
            
            read_time = time.time() - layer_start
            logger.debug(f"  {source_name}: read {len(gdf)} features ({read_time:.1f}s)")
            
            # Store CRS from first successful read
            if layer_crs is None and gdf.crs is not None:
                layer_crs = gdf.crs
            
            # Fast clip
            clip_start = time.time()
            
            if expected_type in ["MultiPoint", "Point"]:
                # For points, use spatial intersection (no geometry modification needed)
                clipped = gdf[intersects(gdf.geometry.values, mask)].copy()
            else:
                # For polygons and lines, use fast clip
                clipped = fast_clip(gdf, mask)
            
            clip_time = time.time() - clip_start
            
            if not clipped.empty:
                # Add source tracking column
                clipped["_MERGE_SOURCE"] = source_name
                merged_parts.append(clipped)
                
                feature_count = len(clipped)
                layer_feature_count += feature_count
                self.stats.features_per_source[source_name] = \
                    self.stats.features_per_source.get(source_name, 0) + feature_count
                    
                logger.debug(f"  {source_name}: {feature_count} features after clip ({clip_time:.1f}s)")
                
            if progress and task_id:
                progress.advance(task_id)
        
        # Merge all parts
        if not merged_parts:
            return None
            
        if len(merged_parts) == 1:
            result = merged_parts[0]
        else:
            result = pd.concat(merged_parts, ignore_index=True)
            if layer_crs is not None:
                result = gpd.GeoDataFrame(result, geometry="geometry", crs=layer_crs)
            else:
                result = gpd.GeoDataFrame(result, geometry="geometry")
        
        # Apply Swiss border clip if configured
        if self.config.clip_to_swiss_border and self.swiss_border is not None:
            border_start = time.time()
            
            if expected_type in ["MultiPoint", "Point"]:
                result = result[intersects(result.geometry.values, self.swiss_border)].copy()
            else:
                result = fast_clip(result, self.swiss_border)
            
            border_time = time.time() - border_start
            logger.debug(f"  Swiss border clip: {len(result)} features ({border_time:.1f}s)")
        
        # Final normalization
        result = normalize_geodataframe_geometries(
            result,
            target_type=expected_type,
            preserve_z=self.config.preserve_z
        )
        
        if result.empty:
            logger.warning(f"No valid features after normalization for {layer_name}")
            return None
            
        self.stats.features_per_layer[layer_name] = len(result)
        return result
    
    def _copy_non_spatial_table(
        self,
        table_name: str,
        reference_path: Path
    ) -> Optional[pd.DataFrame]:
        """Copy a non-spatial table from the reference source."""
        
        try:
            import pyogrio
            
            df = pyogrio.read_dataframe(
                reference_path,
                layer=table_name,
                read_geometry=False
            )
            return df
            
        except Exception as e:
            try:
                gdf = gpd.read_file(reference_path, layer=table_name)
                if 'geometry' in gdf.columns:
                    if gdf.geometry.isna().all() or gdf.geometry.is_empty.all():
                        return pd.DataFrame(gdf.drop(columns=['geometry']))
                return pd.DataFrame(gdf)
            except Exception as e2:
                logger.warning(f"Could not read table {table_name}: {e2}")
                return None
    
    def merge(self) -> MergeStats:
        """Execute the merge operation."""
        
        console.print("\n[bold blue]🔀 Starting GDB Merge Operation (Optimized)[/bold blue]\n")
        
        total_start_time = time.time()
        
        # Validate configuration
        self._validate_config()
        
        # Load mapsheets and create optimized masks
        self._load_mapsheets()
        self._create_source_masks()
        
        # Display source summary
        self._display_source_summary()
        
        self.stats.mapsheets_processed = len(self.mapsheets_gdf)
        
        # Process spatial layers
        merged_layers = {}
        
        console.print("\n[cyan]Processing spatial layers...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=not self.verbose
        ) as progress:
            
            layers_task = progress.add_task(
                "[cyan]Processing spatial layers...",
                total=len(self.config.spatial_layers)
            )

            for layer_name in self.config.spatial_layers:
                layer_start = time.time()
                actual_layer = layer_name.split("/")[-1]
                
                progress.update(
                    layers_task, 
                    description=f"[cyan]Processing {actual_layer}..."
                )
                
                merged = self._merge_spatial_layer(layer_name)
                
                layer_elapsed = time.time() - layer_start
                
                if merged is not None and not merged.empty:
                    merged_layers[layer_name] = merged
                    self.stats.layers_processed += 1
                    console.print(f"  [green]✓[/green] {actual_layer}: {len(merged):,} features ({layer_elapsed:.1f}s)")
                else:
                    console.print(f"  [yellow]○[/yellow] {actual_layer}: no features")
                    
                progress.advance(layers_task)
        
        # Save merged spatial layers
        console.print("\n[cyan]Saving merged spatial layers...[/cyan]")
        self._save_merged_layers(merged_layers)
        
        # Copy non-spatial tables
        if self.config.non_spatial_tables:
            console.print("\n[cyan]Copying reference tables...[/cyan]")
            self._copy_reference_tables()
        
        # Display results
        total_elapsed = time.time() - total_start_time
        self._display_results(total_elapsed)
        
        return self.stats
    
    def _validate_config(self) -> None:
        """Validate merge configuration."""
        
        if not self.config.admin_zones_path or not self.config.admin_zones_path.exists():
            raise ValueError(f"Admin zones file not found: {self.config.admin_zones_path}")
            
        if not self.config.output_path:
            raise ValueError("Output path not specified")
            
        if not self.sources:
            raise ValueError("No valid sources configured")
            
        has_standard = any(
            s.source_type in [SourceType.RC1, SourceType.RC2] 
            for s in self.sources.values()
        )
        if not has_standard:
            logger.warning("No standard RC1/RC2 sources found - only custom sources available")
    
    def _display_source_summary(self) -> None:
        """Display summary of sources and their masks."""
        
        table = Table(title="Source Configuration (Optimized)", show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Mapsheets", justify="right", style="yellow")
        table.add_column("Mask Vertices", justify="right", style="magenta")
        table.add_column("Status", style="white")
        
        for source_name, mask in self.source_masks.items():
            source_path = self._resolve_source_path(source_name)
            n_mapsheets = len(self.mapsheets_gdf[self.mapsheets_gdf['SOURCE_RC'] == source_name])
            complexity = get_mask_complexity(mask)
            
            if source_path:
                source_config = self.sources.get(source_name) or self.sources.get(f"{source_name}.gdb")
                source_type = source_config.source_type.value if source_config else "CUSTOM"
                status = "✓ Found"
            else:
                source_type = "?"
                status = "✗ Missing"
                
            table.add_row(
                source_name,
                source_type,
                str(n_mapsheets),
                f"{complexity['vertices']:,}",
                status
            )
            
        console.print(table)
        
        if self.config.use_convex_hull_masks:
            console.print("[dim]Using convex hull masks for fast clipping[/dim]")
        if self.config.clip_to_swiss_border:
            border_complexity = get_mask_complexity(self.swiss_border)
            console.print(f"[dim]Swiss border: {border_complexity['vertices']:,} vertices[/dim]")
        console.print()
    
    def _save_merged_layers(self, merged_layers: Dict[str, gpd.GeoDataFrame]) -> None:
        """Save merged layers to output GDB with proper geometry handling."""
        
        output_path = self.config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_ext = output_path.suffix.lower()
        
        if output_ext == ".gdb":
            driver = "OpenFileGDB"
        elif output_ext == ".gpkg":
            driver = "GPKG"
        else:
            driver = "OpenFileGDB"
        
        logger.info(f"Output format: {driver}")
        
        exclude_fields = self.config.exclude_fields or []
        if exclude_fields:
            logger.info(f"Excluding {len(exclude_fields)} metadata fields")
        
        first_layer = True
        
        for layer_name, gdf in merged_layers.items():
            actual_layer = self._get_layer_path(output_path, layer_name)
            
            if gdf.empty:
                logger.warning(f"Skipping empty layer: {actual_layer}")
                continue
            
            try:
                expected_type = get_expected_geometry_type(layer_name)
                
                gdf_to_save = normalize_geodataframe_geometries(
                    gdf, 
                    target_type=expected_type,
                    preserve_z=self.config.preserve_z
                )
                
                if gdf_to_save.empty:
                    logger.warning(f"No valid geometries to save for {actual_layer}")
                    continue
                
                if exclude_fields:
                    cols_to_drop = [col for col in exclude_fields if col in gdf_to_save.columns and col != "geometry"]
                    if cols_to_drop:
                        gdf_to_save = gdf_to_save.drop(columns=cols_to_drop)
                
                try:
                    import pyogrio
                    
                    if first_layer and output_ext == ".gdb":
                        if output_path.exists():
                            import shutil
                            shutil.rmtree(output_path)
                    
                    pyogrio.write_dataframe(
                        gdf_to_save,
                        output_path,
                        layer=actual_layer,
                        driver=driver,
                        promote_to_multi=True,
                    )
                    
                    first_layer = False
                    console.print(f"  [green]✓[/green] {actual_layer}: {len(gdf_to_save)} features")
                    
                except ImportError:
                    mode = "w" if first_layer else "a"
                    
                    try:
                        gdf_to_save.to_file(
                            output_path,
                            layer=actual_layer,
                            driver=driver,
                            mode=mode
                        )
                        first_layer = False
                        console.print(f"  [green]✓[/green] {actual_layer}: {len(gdf_to_save)} features")
                        
                    except Exception as e3d:
                        logger.warning(f"3D write failed for {actual_layer}, trying 2D: {e3d}")
                        
                        gdf_2d = force_2d(gdf_to_save)
                        gdf_2d.to_file(
                            output_path,
                            layer=actual_layer,
                            driver=driver,
                            mode=mode
                        )
                        first_layer = False
                        console.print(f"  [yellow]✓[/yellow] {actual_layer}: {len(gdf_2d)} features (2D)")
                
            except Exception as e:
                msg = f"Failed to save {actual_layer}: {e}"
                logger.error(msg)
                self.stats.errors.append(msg)
    
    def _copy_reference_tables(self) -> None:
        """Copy non-spatial tables from reference source."""
        
        ref_source_name = self.config.reference_source
        ref_path = self._resolve_source_path(ref_source_name)
        
        if ref_path is None:
            for name, config in self.sources.items():
                if config.source_type in [SourceType.RC1, SourceType.RC2]:
                    ref_path = config.path
                    ref_source_name = name
                    break
                    
        if ref_path is None:
            logger.warning("No reference source available for non-spatial tables")
            return
            
        console.print(f"  [dim]Using {ref_source_name} as reference for tables[/dim]")
        
        for table_name in self.config.non_spatial_tables:
            df = self._copy_non_spatial_table(table_name, ref_path)
            
            if df is not None:
                try:
                    import pyogrio
                    
                    pyogrio.write_dataframe(
                        df,
                        self.config.output_path,
                        layer=table_name,
                        driver="OpenFileGDB"
                    )
                    
                    self.stats.tables_copied += 1
                    console.print(f"  [green]✓[/green] {table_name}: {len(df)} rows")
                    
                except Exception as e:
                    logger.warning(f"Could not save table {table_name}: {e}")
                    self.stats.warnings.append(f"Table {table_name} not copied: {e}")
    
    def _display_results(self, elapsed_time: float = 0) -> None:
        """Display merge results summary."""
        
        console.print("\n[bold green]✅ Merge Complete![/bold green]\n")
        
        table = Table(title="Merge Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Mapsheets processed", str(self.stats.mapsheets_processed))
        table.add_row("Spatial layers merged", str(self.stats.layers_processed))
        table.add_row("Reference tables copied", str(self.stats.tables_copied))
        
        total_features = sum(self.stats.features_per_layer.values())
        table.add_row("Total features", f"{total_features:,}")
        
        if elapsed_time > 0:
            mins, secs = divmod(elapsed_time, 60)
            table.add_row("Processing time", f"{int(mins)}m {secs:.1f}s")
            
            # Performance comparison
            if total_features > 0:
                features_per_sec = total_features / elapsed_time
                table.add_row("Throughput", f"{features_per_sec:,.0f} features/sec")
        
        console.print(table)
        
        # Features by source
        if self.stats.features_per_source:
            console.print("\n[bold]Features by Source:[/bold]")
            for source, count in sorted(self.stats.features_per_source.items(), key=lambda x: -x[1]):
                console.print(f"  • {source}: {count:,}")
                
        if self.verbose and self.stats.features_per_layer:
            console.print("\n[bold]Features by Layer:[/bold]")
            for layer, count in sorted(self.stats.features_per_layer.items()):
                layer_short = layer.split("/")[-1]
                console.print(f"  • {layer_short}: {count:,}")
        
        if self.stats.warnings:
            console.print("\n[yellow]⚠ Warnings:[/yellow]")
            for warning in self.stats.warnings:
                console.print(f"  • {warning}")
                
        if self.stats.errors:
            console.print("\n[red]✗ Errors:[/red]")
            for error in self.stats.errors:
                console.print(f"  • {error}")
                
        console.print(f"\n[dim]Output: {self.config.output_path}[/dim]")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_merge_config(
    rc1_path: Optional[Path] = None,
    rc2_path: Optional[Path] = None,
    custom_sources_dir: Optional[Path] = None,
    admin_zones_path: Path = None,
    output_path: Path = None,
    source_column: str = "SOURCE_RC",
    mapsheet_numbers: Optional[List[int]] = None,
    reference_source: str = "RC2",
    use_convex_hull_masks: bool = True,
    clip_to_swiss_border: bool = True,
) -> MergeConfig:
    """Create a merge configuration with sensible defaults."""
    
    return MergeConfig(
        rc1_path=rc1_path,
        rc2_path=rc2_path,
        custom_sources_dir=custom_sources_dir,
        admin_zones_path=admin_zones_path,
        output_path=output_path,
        source_column=source_column,
        mapsheet_numbers=mapsheet_numbers,
        reference_source=reference_source,
        use_convex_hull_masks=use_convex_hull_masks,
        clip_to_swiss_border=clip_to_swiss_border,
    )
