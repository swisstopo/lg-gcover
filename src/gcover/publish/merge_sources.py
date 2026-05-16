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

import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict

from typing import Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import fiona
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from rich.table import Table
from shapely import (difference, get_coordinates, intersection, intersects,
                     make_valid, set_coordinates, simplify, within)
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

from gcover.publish.utils import new_uuid
from shapely import STRtree


from gcover.core.geometry import (load_gpkg_with_validation,
                                  validate_and_repair_geometries)

console = Console(stderr=True)

has_pyarrow = False
try:
    import pyarrow
    console.print("PyArrow version:", pyarrow.__version__)
    has_pyarrow = True
except ImportError:
    console.print("[red]PyArrow not installed[/red]")


console.print("[yellow]Suppressing some OGR warning (unclosed rings, only CCW, etc.)[/yellow]")
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


def _create_source_masks(self) -> Dict[str, BaseGeometry]:
    """
    Create EXCLUSIVE (non-overlapping) clip masks for each source.
    """
    if self.mapsheets_gdf is None:
        self._load_mapsheets()

    # Rename column for consistency
    source_col = self.config.source_column
    if source_col != 'SOURCE_RC':
        self.mapsheets_gdf = self.mapsheets_gdf.rename(columns={source_col: 'SOURCE_RC'})

    # Determine source priority (RC2 wins over RC1 by default)
    all_sources = self.mapsheets_gdf['SOURCE_RC'].unique().tolist()
    source_priority = []
    if 'RC2' in all_sources:
        source_priority.append('RC2')
    if 'RC1' in all_sources:
        source_priority.append('RC1')
    for src in sorted(all_sources):
        if src not in source_priority:
            source_priority.append(src)

    logger.info(f"Source priority for exclusive masks: {source_priority}")

    masks = {}
    claimed_area = None

    for source_name in source_priority:
        sheets = self.mapsheets_gdf[self.mapsheets_gdf['SOURCE_RC'] == source_name]
        if sheets.empty:
            continue

        # Get EXACT dissolved boundary (NOT convex hull!)
        source_area = unary_union(sheets.geometry.values)

        # Clean up potential topology issues
        if not source_area.is_valid:
            source_area = source_area.buffer(0)

        # Make EXCLUSIVE by subtracting already-claimed areas
        if claimed_area is not None:
            exclusive_area = difference(source_area, claimed_area)
            if exclusive_area is None or exclusive_area.is_empty:
                logger.warning(f"Source {source_name} fully covered by higher priority sources")
                continue
            mask = exclusive_area
        else:
            mask = source_area

        if not mask.is_valid:
            mask = mask.buffer(0)

        masks[source_name] = mask

        complexity = get_mask_complexity(mask)
        logger.debug(f"  {source_name}: {complexity['type']}, {complexity['vertices']:,} vertices")

        # Update claimed area for next iteration
        if claimed_area is None:
            claimed_area = source_area
        else:
            claimed_area = unary_union([claimed_area, source_area])

    self.source_masks = masks
    return masks


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
        # "GC_ROCK_BODIES/GC_MAPSHEET",
        # For Increment
        "D_GC_EXPLOIT_GEOMAT_PLG",
        "M_GC_EXPLOIT_GEOMAT_PLG",
        "MG_GC_EXPLOIT_GEOMAT_PLG",
        "D_GC_LINEAR_OBJECTS",
        "A_GC_LINEAR_OBJECTS",
        "M_GC_LINEAR_OBJECTS",
        "MG_GC_LINEAR_OBJECTS",
        "D_GC_POINT_OBJECTS",
        "A_GC_POINT_OBJECTS",
        "M_GC_POINT_OBJECTS",
        "MG_GC_POINT_OBJECTS",
        "M_GC_FOSSILS",
        "MG_GC_FOSSILS",
        "D_GC_UNCO_DESPOSIT",
        "A_GC_UNCO_DESPOSIT",
        "M_GC_UNCO_DESPOSIT",
        "MG_GC_UNCO_DESPOSIT",
        "D_GC_BEDROCK",
        "A_GC_BEDROCK",
        "M_GC_BEDROCK",
        "MG_GC_BEDROCK",
        "D_GC_SURFACES",
        "A_GC_SURFACES",
        "M_GC_SURFACES",
        "MG_GC_SURFACES",

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

    # Post-processing options
    split_by_mapsheet: bool = False  # Split polygon/line features at mapsheet boundaries
    enrich_mapsheet_links: bool = False  # Add erl_link/ber_link per feature (requires split_by_mapsheet)
    mapsheet_transfer_cols: List[str] = field(default_factory=lambda: [
        "MSH_MAP_NBR", "MSH_MAP_TITLE", "SOURCE_RC"
    ])
    

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
# FileGDB discovery
# =============================================================================

def _discover_filegdbs(directory: Path) -> List[Tuple[Path, str, str]]:
    """
    Yield (path, stem, full_name) for every FileGDB found in *directory*.

    Detection uses the presence of a 'timestamps' file, which every ESRI
    FileGDB contains regardless of whether the directory ends with '.gdb'.

    Examples
    --------
    directory/BCK_2016/         → ("BCK_2016", "BCK_2016")
    directory/Saas.gdb/         → ("Saas",     "Saas.gdb")
    directory/20300501_Saas.gdb → ("20300501_Saas", "20300501_Saas.gdb")
    """
    results = []
    for candidate in sorted(directory.iterdir()):
        if not candidate.is_dir():
            continue
        # 'timestamps' is the canonical FileGDB marker (always present)
        if not (candidate / "timestamps").exists():
            continue
        full_name = candidate.name                          # e.g. "Saas.gdb" or "BCK_2016"
        stem      = candidate.stem                          # strips trailing .gdb if present
        results.append((candidate, stem, full_name))
    return results
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
        # Note, if FileGDB dir doesn't end with `.gdb` `geopandas` won't be able to open it
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

    def _build_enrichment_index(self) -> None:
        """
        Build a reusable STRtree on the 221 mapsheet polygons.
        Called once; reused by every _enrich_with_mapsheet_links() call.
        """
        link_cols = [c for c in ("erl_link", "ber_link")
                     if c in self.mapsheets_gdf.columns]
        if not link_cols:
            self._enrichment_tree = None
            self._enrichment_links = {}
            return

        # Arrays parallel to the tree — index i → mapsheet i's link values
        self._enrichment_tree = STRtree(self.mapsheets_gdf.geometry.values)
        self._enrichment_links = {
            col: self.mapsheets_gdf[col].to_numpy()
            for col in link_cols
        }
        logger.info(f"Enrichment index built: {len(self.mapsheets_gdf)} mapsheets, "
                    f"cols={link_cols}")

    def _layer_exists(self, gdb_path: Path, layer_name: str) -> bool:
        actual_name = self._get_layer_path(gdb_path, layer_name)
        if not hasattr(self, '_layer_cache'):
            self._layer_cache: Dict[Path, Set[str]] = {}
        if gdb_path not in self._layer_cache:
            self._layer_cache[gdb_path] = set(fiona.listlayers(gdb_path))
        return actual_name in self._layer_cache[gdb_path]

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
    
    # Fixed v2
    def _create_source_masks(self) -> Dict[str, BaseGeometry]:
            """
            Create EXCLUSIVE (non-overlapping) clip masks for each source.

            V2 improvements:
            - Better logging to diagnose issues
            - Verify masks don't overlap
            - Handle edge cases
            """
            if self.mapsheets_gdf is None:
                self._load_mapsheets()

            # Rename column for consistency
            source_col = self.config.source_column
            if source_col in self.mapsheets_gdf.columns and source_col != 'SOURCE_RC':
                self.mapsheets_gdf = self.mapsheets_gdf.rename(columns={source_col: 'SOURCE_RC'})

            # Get unique sources and their mapsheet counts
            source_counts = self.mapsheets_gdf['SOURCE_RC'].value_counts()
            logger.info(f"Sources in mapsheets: {dict(source_counts)}")

            # Determine priority (RC2 > RC1 > others)
            all_sources = list(source_counts.index)
            source_priority = []
            if 'RC2' in all_sources:
                source_priority.append('RC2')
            if 'RC1' in all_sources:
                source_priority.append('RC1')
            for src in sorted(all_sources):
                if src not in source_priority:
                    source_priority.append(src)

            logger.info(f"Source priority (higher priority = gets area first): {source_priority}")

            masks = {}
            claimed_area = None

            for source_name in source_priority:
                sheets = self.mapsheets_gdf[self.mapsheets_gdf['SOURCE_RC'] == source_name]
                if sheets.empty:
                    logger.warning(f"No mapsheets for source {source_name}")
                    continue

                # Dissolve mapsheets for this source
                source_area = unary_union(sheets.geometry.values)

                # Clean topology
                if not source_area.is_valid:
                    source_area = source_area.buffer(0)

                source_area_km2 = source_area.area / 1e6
                logger.debug(f"  {source_name}: raw area = {source_area_km2:.1f} km²")

                # Make EXCLUSIVE
                if claimed_area is not None:
                    exclusive_area = difference(source_area, claimed_area)

                    if exclusive_area is None or exclusive_area.is_empty:
                        logger.warning(f"  {source_name}: NO exclusive area (fully covered by higher priority)")
                        continue

                    exclusive_area_km2 = exclusive_area.area / 1e6
                    reduction = (1 - exclusive_area.area / source_area.area) * 100
                    logger.debug(f"  {source_name}: exclusive area = {exclusive_area_km2:.1f} km² "
                                 f"({reduction:.1f}% removed by higher priority sources)")

                    mask = exclusive_area
                else:
                    mask = source_area

                # Clean final mask
                if not mask.is_valid:
                    mask = mask.buffer(0)

                masks[source_name] = mask

                # Update claimed area
                if claimed_area is None:
                    claimed_area = source_area
                else:
                    claimed_area = unary_union([claimed_area, source_area])

            # VERIFY: Check that masks don't overlap
            logger.info("Verifying mask exclusivity...")
            mask_names = list(masks.keys())
            for i, name1 in enumerate(mask_names):
                for name2 in mask_names[i + 1:]:
                    overlap = masks[name1].intersection(masks[name2])
                    if overlap and not overlap.is_empty and overlap.area > 1:  # > 1 m²
                        logger.error(f"  OVERLAP between {name1} and {name2}: {overlap.area:.1f} m²!")
                    else:
                        logger.debug(f"  {name1} ∩ {name2} = OK (no significant overlap)")

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
    

    # FIXED
    def _read_layer_for_source(
            self,
            gdb_path: Path,
            layer_name: str,
            mask: BaseGeometry
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Read a layer from a FileGDB with STRICT spatial filtering.

        The original uses mask= for bbox filtering during read, but this can
        return features outside the mask (anything touching the bbox).

        This version does an additional intersection check after reading.
        """
        actual_layer = self._get_layer_path(gdb_path, layer_name)

        try:
            # Read with bbox filter (fast)
            gdf = gpd.read_file(
                gdb_path,
                layer=actual_layer,
                engine='pyogrio',
                mask=mask  # This does bbox filter, not exact intersection!
            )

            if gdf.empty:
                return gdf

            # STRICT filter: only keep features that actually intersect the mask
            # (not just its bounding box)
            gdf = gdf[intersects(gdf.geometry.values, mask)].copy()

            if self.config.validate_geometries and not gdf.empty:
                from gcover.core.geometry import validate_and_repair_geometries
                gdf = validate_and_repair_geometries(gdf)

            return gdf

        except Exception as e:
            logger.warning(f"Could not read {actual_layer} from {gdb_path}: {e}")
            return None
    # Fixed
    def _merge_spatial_layer(
            self,
            layer_name: str,
            progress: Optional[Progress] = None,
            task_id: Optional[int] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Merge a single spatial layer from all sources using optimized clip.

        Key changes:
        1. Create exclusive masks (no geographic overlap)
        2. For each exclusive mask, read ONLY from its authoritative source
        3. Track UUIDs to catch any remaining duplicates
        """
        from gcover.publish.merge_sources import (
            fast_clip, get_expected_geometry_type,
            normalize_geodataframe_geometries)

        merged_parts = []
        layer_crs = None
        seen_uuids: Set[str] = set()  # Track UUIDs to prevent ANY duplicates

        expected_type = get_expected_geometry_type(layer_name)
        logger.debug(f"Processing layer {layer_name}, expected type: {expected_type}")

        # Process each source with its EXCLUSIVE mask
        for source_name, clip_mask in self.source_masks.items():
            source_path = self._resolve_source_path(source_name)
            if source_path is None:
                msg = f"Source not found: {source_name}"
                logger.warning(msg)
                self.stats.warnings.append(msg)
                continue

            logger.debug(f"  Reading {source_name} from {source_path}")
            console.print(f"[dim]  Reading {source_name} from {source_path}[/dim]")

            layer_start = time.time()

            # Read from THIS source's GDB, filtered by the exclusive mask
            gdf = self._read_layer_for_source(source_path, layer_name, clip_mask)

            if gdf is None or gdf.empty:
                logger.debug(f"  {source_name}: no features in {layer_name}")
                continue

            read_time = time.time() - layer_start
            read_count = len(gdf)
            logger.debug(f"  {source_name}: read {read_count} features ({read_time:.1f}s)")

            if layer_crs is None and gdf.crs is not None:
                layer_crs = gdf.crs

            # Clip to the EXCLUSIVE mask (precise clipping)
            clip_start = time.time()

            if expected_type in ["MultiPoint", "Point"]:
                clipped = gdf[intersects(gdf.geometry.values, clip_mask)].copy()
            else:
                clipped = fast_clip(gdf, clip_mask)

            clip_time = time.time() - clip_start

            if clipped.empty:
                logger.debug(f"  {source_name}: no features after clip")
                continue

            # === UUID DEDUPLICATION — regenerate instead of dropping ===
            # Exclusive masks prevent real duplicates. The only remaining case is a
            # polygon that straddles an RC1/RC2 boundary: both GDBs contain it, each
            # clipped to its own half. Dropping one half silently loses area; instead
            # we give the "already seen" copy a fresh UUID so both halves survive.
            if 'UUID' in clipped.columns:
                collision_mask = (
                        clipped['UUID'].notna() & clipped['UUID'].isin(seen_uuids)
                )
                n_collisions = collision_mask.sum()
                if n_collisions > 0:
                    new_ids = [new_uuid(esri_style=True) for _ in range(n_collisions)]
                    clipped.loc[collision_mask, 'UUID'] = new_ids
                    logger.warning(
                        f"  {source_name}: regenerated {n_collisions} duplicate UUIDs "
                        f"(boundary features present in both sources)"
                    )
                # Register all UUIDs from this source
                seen_uuids.update(clipped['UUID'].dropna().unique())

            # Add source tracking
            clipped["_MERGE_SOURCE"] = source_name
            merged_parts.append(clipped)

            feature_count = len(clipped)
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

        # Final Swiss border clip
        if self.config.clip_to_swiss_border and self.swiss_border is not None:
            border_start = time.time()
            if expected_type in ["MultiPoint", "Point"]:
                result = result[intersects(result.geometry.values, self.swiss_border)].copy()
            else:
                result = fast_clip(result, self.swiss_border)
            logger.debug(f"  Swiss border clip: {len(result)} features ({time.time() - border_start:.1f}s)")

        # Normalize geometries
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
        self._build_enrichment_index()  # for links enrichment

        # Display source summary
        self._display_source_summary()
        
        self.stats.mapsheets_processed = len(self.mapsheets_gdf)
        
        # Process spatial layers
        merged_layers = {}
        
        console.print("\n[bold chartreuse2]===== Processing spatial layers... =====[/bold chartreuse2]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=not self.verbose
        ) as progress:

            actual_spatial_layers = [
              layer_name
              for layer_name in self.config.spatial_layers
              if any(
                self._layer_exists(src.path, layer_name)
                for src in self.sources.values()
              )
            ]

            _spatial_layers_nb = len(actual_spatial_layers)
            logger.info(
              f"{_spatial_layers_nb}/{len(self.config.spatial_layers)} configured layers "
              f"found across {len(self.sources)} source(s)"
            )

            layers_task = progress.add_task(
               f"[cyan]Processing {_spatial_layers_nb} spatial layers...",   # ← proper f-string
               total=_spatial_layers_nb
            )

            for layer_name in actual_spatial_layers:
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

        # Optional post-processing: split at mapsheet boundaries / enrich links
        console.print("\n[bold chartreuse2]===== Post processing =====[/bold chartreuse2]")

        if self.config.split_by_mapsheet:
            merged_layers = self._post_process_layers(merged_layers)

        if self.config.enrich_mapsheet_links:
          with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=not self.verbose
        ) as progress:
            layers_task = progress.add_task(
                "Enriching features with mapsheet PDF links...",
                total=len(merged_layers)
            )

            for layer_name in list(merged_layers.keys()):
                layer_start = time.time()
                progress.update(
                    layers_task,
                    description=f"[cyan]Enriching {layer_name} with links..."
                )
                merged_layers[layer_name] = self._enrich_with_mapsheet_links(
                    merged_layers[layer_name], layer_name
                )
                layer_elapsed = time.time() - layer_start
                console.print(f"  [green]✓[/green] {layer_name} ({layer_elapsed:.1f}s)")
                progress.advance(layers_task)


        # Save merged spatial layers
        console.print("\n[bold chartreuse2]===== Saving merged spatial layers... =====[/bold chartreuse2]")
        self._save_merged_layers(merged_layers)
        
        # Copy non-spatial tables
        if self.config.non_spatial_tables:
            console.print("\n[cyan]Copying reference tables...[/cyan]")
            self._copy_reference_tables()
        
        # Display results
        total_elapsed = time.time() - total_start_time
        self._display_results(total_elapsed)
        
        return self.stats

    def _enrich_with_mapsheet_links(
            self,
            gdf: gpd.GeoDataFrame,
            layer_name: str,
    ) -> gpd.GeoDataFrame:
        """
        Enrich features with erl_link / ber_link from mapsheets.

        - After split: each feature belongs to exactly one mapsheet → direct sjoin.
        - Without split: a feature may span many mapsheets → links are pipe-joined,
          e.g. "https://example.com/a.pdf|https://example.com/b.pdf".
        """

        if self._enrichment_tree is None:
            logger.warning("No enrichment index — skipping link enrichment")
            return gdf

        link_cols = list(self._enrichment_links.keys())
        if not link_cols:
            logger.warning("erl_link / ber_link not found in mapsheets layer — skipping enrichment")
            return gdf
        actual_layer = layer_name.split("/")[-1]
        expected_type = get_expected_geometry_type(layer_name)
        is_point = expected_type in ("MultiPoint", "Point")

        # ── 1. Representative points (one per feature, vectorized) ──────────
        rep_pts = (
            gdf.geometry.values  # already points
            if is_point
            else gdf.geometry.representative_point().values  # centroid-like, always inside
        )

        # ── 2. STRtree bulk query — returns (mapsheet_idx[], feature_idx[]) ─
        #    predicate='within': point is within mapsheet polygon
        # The STRtree is built from mapsheets (221).
        # Queried with rep_pts (295188 features).
        # Shapely 2.0 returns (input_indices, tree_indices) = (feat_idx, mapsheet_idx)
        feat_idx, mapsheet_idx = self._enrichment_tree.query(rep_pts, predicate="within")
        #          ^^^^^^^^^^ 0..295187                  ^^^^^^^^^^ 0..220
        logger.debug(f"feat_idx  range: {feat_idx.min()}–{feat_idx.max()}, mapsheets size: {len(self.mapsheets_gdf)}")
        logger.debug(f"mapsheet_idx range: {mapsheet_idx.min()}–{mapsheet_idx.max()}, features size: {len(gdf)}")

        # tree_idx  → indices into self.mapsheets_gdf  (0..220)
        # query_idx → indices into rep_pts / gdf       (0..n_features-1)
        # Both arrays have the same length; entry k means:
        #   rep_pts[feat_idx[k]] is within mapsheets[mapsheet_idx[k]]

        # ── 3. Assign links ─────────────────────────────────────────────────
        if self.config.split_by_mapsheet:
            # 1:1 — take first match only (features were already split)
            for col in link_cols:
                vals = np.full(len(gdf), None, dtype=object)
                _, first = np.unique(feat_idx, return_index=True)
                vals[feat_idx[first]] = self._enrichment_links[col][mapsheet_idx[first]]
                gdf[col] = vals
        else:
            # 1:N — use full geometry + intersects so features crossing
            # mapsheet boundaries collect links from all touched mapsheets.
            feat_idx, mapsheet_idx = self._enrichment_tree.query(
                gdf.geometry.values, predicate="intersects"
            )
            for col in link_cols:
                link_arr = self._enrichment_links[col]
                bucket: dict[int, list] = defaultdict(list)
                for fi, mi in zip(feat_idx, mapsheet_idx):
                    v = link_arr[mi]
                    if v and str(v).strip():
                        bucket[fi].append(str(v))

                vals = np.full(len(gdf), None, dtype=object)
                for fi, links in bucket.items():
                    seen = dict.fromkeys(links)  # deduplicate, preserve order
                    vals[fi] = "|".join(seen) if seen else None
                gdf[col] = vals

        # ── 4. Report ────────────────────────────────────────────────────────
        table = Table(title=f"Link statistics for layer: {actual_layer}")

        table.add_column("Column", style="cyan", no_wrap=True)
        table.add_column("Matched", style="green")
        table.add_column("Total", style="white")
        table.add_column("Multi‑mapsheet", style="magenta")

        for col in link_cols:
            n_matched = gdf[col].notna().sum()

            if not self.config.split_by_mapsheet:
                n_multi = gdf[col].str.contains("|", regex=False).sum()
            else:
                n_multi = 0

            table.add_row(
                col,
                f"{n_matched:,}",
                f"{len(gdf):,}",
                f"{n_multi:,}" if n_multi else "-"
            )

        console.print(table)

        return gdf

    def _split_layer_by_mapsheets(
            self,
            gdf: gpd.GeoDataFrame,
            layer_name: str,
    ) -> gpd.GeoDataFrame:
        """
        Split polygon/line features along mapsheet boundaries via overlay,
        or assign mapsheet attributes to point features via sjoin.

        After this step every feature belongs to exactly one mapsheet,
        making erl_link/ber_link enrichment unambiguous.
        """
        expected_type = get_expected_geometry_type(layer_name)
        is_point = expected_type in ("MultiPoint", "Point")

        # Build the slim mapsheets GDF to join/overlay against
        transfer_cols = list(self.config.mapsheet_transfer_cols)
        if self.config.enrich_mapsheet_links:
            for col in ("erl_link", "ber_link"):
                if col in self.mapsheets_gdf.columns and col not in transfer_cols:
                    transfer_cols.append(col)
                elif col not in self.mapsheets_gdf.columns:
                    logger.warning(f"Column '{col}' not found in mapsheets layer — skipping")

        mapsheets_slim = self.mapsheets_gdf[["geometry", *transfer_cols]].copy()

        before = len(gdf)

        if is_point:
            # Points: simple within-join, no geometry splitting needed
            result = gpd.sjoin(gdf, mapsheets_slim, how="left", predicate="within")
            result = result.drop(columns=["index_right"], errors="ignore")
            # Guard: point exactly on a shared boundary → keep first match only
            result = result[~result.index.duplicated(keep="first")]
        else:
            # Polygon / line: overlay splits features at mapsheet edges
            # keep_geom_type=True drops degenerate points/lines that appear
            # when two polygons share only an edge after intersection
            result = gpd.overlay(
                gdf,
                mapsheets_slim,
                how="intersection",
                keep_geom_type=True,
            )

        after = len(result)
        n_unmatched = result[transfer_cols[0]].isna().sum() if transfer_cols else 0


        #for stale_id_col in ("UUID", "OBJECTID", "OID", "FID"):
        #    if stale_id_col in result.columns:
        #        result = result.drop(columns=[stale_id_col])
        #        logger.info(f"  Dropped stale ID column '{stale_id_col}' after split")

        logger.info(
            f"  {layer_name.split('/')[-1]}: "
            f"{'split' if not is_point else 'joined'} "
            f"{before:,} → {after:,} features"
            + (f" ({n_unmatched} unmatched)" if n_unmatched else "")
        )

        return result

    def _post_process_layers(
            self,
            merged_layers: Dict[str, gpd.GeoDataFrame],
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Optional post-merge step: split features at mapsheet boundaries
        and/or enrich with mapsheet PDF links (erl_link, ber_link).
        """
        action = "Splitting + enriching" if self.config.enrich_mapsheet_links else "Splitting"
        console.print(f"\n[cyan]{action} features along mapsheet boundaries...[/cyan]")

        result = {}
        for layer_name, gdf in merged_layers.items():
            actual = layer_name.split("/")[-1]
            try:
                processed = self._split_layer_by_mapsheets(gdf, layer_name)
                result[layer_name] = processed
                console.print(f"  [green]✓[/green] {actual}: {len(processed):,} features")
            except Exception as e:
                logger.error(f"Post-processing failed for {actual}: {e}")
                self.stats.errors.append(f"Post-process {actual}: {e}")
                result[layer_name] = gdf  # fall back to unsplit layer

        return result
    
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
# SINGLE-GDB TRANSFORM
# =============================================================================

@dataclass
class TransformConfig:
    """Configuration for transforming a single FileGDB to the custom format."""
    input_path: Path
    output_path: Path
    preserve_z: bool = True
    validate_geometries: bool = True
    exclude_fields: Optional[List[str]] = None
    # None = auto-discover all layers from the input GDB
    spatial_layers: Optional[List[str]] = None
    skip_tables: bool = False


class GDBTransformer:
    """
    Normalizes a single FileGDB to the custom publication format.

    Applies the same geometry normalization pipeline as GDBMerger
    (type normalization, 2D/3D handling, field exclusion, validation)
    without the multi-source merging step.
    """

    def __init__(self, config: TransformConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.stats = MergeStats()

    def _discover_layers(self) -> Tuple[List[str], List[str]]:
        """Return (spatial_layers, non_spatial_tables) from the input GDB."""
        all_layers = fiona.listlayers(self.config.input_path)
        spatial: List[str] = []
        tables: List[str] = []
        for name in all_layers:
            try:
                with fiona.open(self.config.input_path, layer=name) as src:
                    geom_type = src.schema.get("geometry")
                    if geom_type and geom_type.lower() != "none":
                        spatial.append(name)
                    else:
                        tables.append(name)
            except Exception:
                tables.append(name)
        return spatial, tables

    def _read_and_normalize_layer(self, layer_name: str) -> Optional[gpd.GeoDataFrame]:
        """Read, validate, and normalize one spatial layer."""
        actual = layer_name.split("/")[-1]
        try:
            gdf = gpd.read_file(
                self.config.input_path,
                layer=actual,
                engine="pyogrio",
            )
        except Exception as e:
            logger.warning(f"Could not read {actual}: {e}")
            self.stats.errors.append(f"Read {actual}: {e}")
            return None

        if gdf.empty:
            return None

        if self.config.validate_geometries:
            from gcover.core.geometry import validate_and_repair_geometries
            gdf = validate_and_repair_geometries(gdf)

        expected_type = get_expected_geometry_type(layer_name)
        gdf = normalize_geodataframe_geometries(
            gdf,
            target_type=expected_type,
            preserve_z=self.config.preserve_z,
        )
        return gdf if not gdf.empty else None

    def _copy_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Read a non-spatial table from the input GDB."""
        try:
            import pyogrio
            return pyogrio.read_dataframe(
                self.config.input_path, layer=table_name, read_geometry=False
            )
        except Exception as e:
            logger.warning(f"Could not read table {table_name}: {e}")
            return None

    def _save_layer(
        self,
        gdf: gpd.GeoDataFrame,
        layer_name: str,
        first_layer: bool,
        driver: str,
    ) -> bool:
        """Write one spatial layer to the output. Returns True on success."""
        actual = layer_name.split("/")[-1]
        output_ext = self.config.output_path.suffix.lower()

        exclude = self.config.exclude_fields or []
        if exclude:
            cols_to_drop = [c for c in exclude if c in gdf.columns and c != "geometry"]
            if cols_to_drop:
                gdf = gdf.drop(columns=cols_to_drop)

        try:
            import pyogrio

            if first_layer and output_ext == ".gdb" and self.config.output_path.exists():
                import shutil
                shutil.rmtree(self.config.output_path)

            pyogrio.write_dataframe(
                gdf,
                self.config.output_path,
                layer=actual,
                driver=driver,
                promote_to_multi=True,
            )
            console.print(f"  [green]✓[/green] {actual}: {len(gdf):,} features")
            self.stats.features_per_layer[layer_name] = len(gdf)
            self.stats.layers_processed += 1
            return True

        except Exception as e:
            logger.error(f"Failed to save {actual}: {e}")
            self.stats.errors.append(f"Save {actual}: {e}")
            return False

    def _save_table(self, df: pd.DataFrame, table_name: str, driver: str) -> bool:
        """Write one non-spatial table to the output."""
        try:
            import pyogrio
            pyogrio.write_dataframe(
                df,
                self.config.output_path,
                layer=table_name,
                driver=driver,
            )
            self.stats.tables_copied += 1
            console.print(f"  [green]✓[/green] {table_name}: {len(df):,} rows")
            return True
        except Exception as e:
            logger.warning(f"Could not save table {table_name}: {e}")
            self.stats.warnings.append(f"Table {table_name}: {e}")
            return False

    def transform(self) -> MergeStats:
        """Execute the transform operation."""
        console.print("\n[bold blue]🔄 Starting GDB Transform[/bold blue]\n")
        total_start = time.time()

        discovered_spatial, discovered_tables = self._discover_layers()

        if self.config.spatial_layers is not None:
            spatial_layers = [
                ll for ll in self.config.spatial_layers
                if ll.split("/")[-1] in discovered_spatial
            ]
            missing = [
                ll for ll in self.config.spatial_layers
                if ll.split("/")[-1] not in discovered_spatial
            ]
            if missing:
                logger.warning(f"Layers not found in input: {missing}")
        else:
            spatial_layers = discovered_spatial

        tables = [] if self.config.skip_tables else discovered_tables

        output_ext = self.config.output_path.suffix.lower()
        driver = "OpenFileGDB" if output_ext == ".gdb" else "GPKG"

        console.print(f"[dim]Input:  {self.config.input_path}[/dim]")
        console.print(f"[dim]Output: {self.config.output_path} ({driver})[/dim]")
        console.print(f"[dim]{len(spatial_layers)} spatial layers, {len(tables)} tables[/dim]\n")

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        first_layer = True

        console.print("[bold chartreuse2]===== Transforming spatial layers... =====[/bold chartreuse2]")
        for layer_name in spatial_layers:
            gdf = self._read_and_normalize_layer(layer_name)
            if gdf is None:
                console.print(f"  [yellow]○[/yellow] {layer_name.split('/')[-1]}: no features")
                continue
            self._save_layer(gdf, layer_name, first_layer, driver)
            first_layer = False

        if tables:
            console.print(
                "\n[bold chartreuse2]===== Copying reference tables... =====[/bold chartreuse2]"
            )
            for table_name in tables:
                df = self._copy_table(table_name)
                if df is not None:
                    self._save_table(df, table_name, driver)

        total_elapsed = time.time() - total_start
        total_features = sum(self.stats.features_per_layer.values())
        mins, secs = divmod(total_elapsed, 60)

        console.print("\n[bold green]✅ Transform Complete![/bold green]")
        console.print(
            f"[dim]{self.stats.layers_processed} layers, {total_features:,} features, "
            f"{self.stats.tables_copied} tables — {int(mins)}m {secs:.1f}s[/dim]"
        )
        console.print(f"[dim]Output: {self.config.output_path}[/dim]")

        if self.stats.warnings:
            console.print("\n[yellow]⚠ Warnings:[/yellow]")
            for w in self.stats.warnings:
                console.print(f"  • {w}")
        if self.stats.errors:
            console.print("\n[red]✗ Errors:[/red]")
            for err in self.stats.errors:
                console.print(f"  • {err}")

        return self.stats


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
    split_by_mapsheet: bool = False,
    enrich_mapsheet_link: bool = False,
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
        split_by_mapsheet=split_by_mapsheet,
        enrich_mapsheet_link=enrich_mapsheet_link,
    )
