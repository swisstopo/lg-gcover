# src/gcover/publish/merge_sources.py
"""
Module for merging multiple FileGDB sources into a single publication GDB.

Supports clipping features from different source databases (RC1, RC2, custom GDBs)
based on mapsheet boundaries defined in administrative zones.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from shapely import get_coordinates, set_coordinates
from shapely.geometry import (
    GeometryCollection, 
    LineString, 
    MultiLineString,
    MultiPoint,
    MultiPolygon, 
    Point, 
    Polygon,
)
from shapely.ops import linemerge
from shapely import make_valid

from loguru import logger

# Suppress pandas fragmentation warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

console = Console()


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
    
    # Fields to exclude from output (metadata fields, etc.)
    # Set to None to keep all fields, or provide a list like DEFAULT_EXCLUDED_FIELDS
    exclude_fields: Optional[List[str]] = None
    

@dataclass
class MergeStats:
    """Statistics from merge operation."""
    layers_processed: int = 0
    tables_copied: int = 0
    features_per_layer: Dict[str, int] = field(default_factory=dict)
    features_per_source: Dict[str, int] = field(default_factory=dict)
    mapsheets_processed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class GDBMerger:
    """
    Merges multiple FileGDB sources into a single output based on mapsheet boundaries.
    """
    
    def __init__(self, config: MergeConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.sources: Dict[str, SourceConfig] = {}
        self.mapsheets_gdf: Optional[gpd.GeoDataFrame] = None
        self.stats = MergeStats()
        
        logger.debug("Initializing GDBMerger (geopandas)")
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
        
        # Filter by mapsheet numbers if specified
        if self.config.mapsheet_numbers:
            gdf = gdf[gdf[self.config.mapsheet_nbr_column].isin(self.config.mapsheet_numbers)]
            logger.info(f"Filtered to {len(gdf)} mapsheets")
            
        self.mapsheets_gdf = gdf
        return gdf
    
    def _get_mapsheets_by_source(self) -> Dict[str, gpd.GeoDataFrame]:
        """Group mapsheets by their source assignment."""
        
        if self.mapsheets_gdf is None:
            self._load_mapsheets()
            
        grouped = {}
        for source_name in self.mapsheets_gdf[self.config.source_column].unique():
            mask = self.mapsheets_gdf[self.config.source_column] == source_name
            grouped[source_name] = self.mapsheets_gdf[mask].copy()
            
        return grouped
    
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
            # For geopandas, we need just the layer name without the group
            return layer_name.split("/")[-1]
        return layer_name
    
    def _read_layer(
        self, 
        gdb_path: Path, 
        layer_name: str,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """Read a layer from a FileGDB."""
        
        actual_layer = self._get_layer_path(gdb_path, layer_name)
        
        try:
            read_kwargs = {"layer": actual_layer}
            if bbox:
                read_kwargs["bbox"] = bbox
                
            gdf = gpd.read_file(gdb_path, **read_kwargs)
            return gdf
            
        except Exception as e:
            logger.warning(f"Could not read {actual_layer} from {gdb_path}: {e}")
            return None
    
    def _clip_layer(
        self,
        gdf: gpd.GeoDataFrame,
        clip_geometry: gpd.GeoDataFrame,
        layer_name: str
    ) -> gpd.GeoDataFrame:
        """Clip a layer to the clip geometry (union of mapsheets) and normalize result."""
        
        if gdf.empty:
            return gdf
            
        # Ensure same CRS
        if gdf.crs != clip_geometry.crs:
            clip_geometry = clip_geometry.to_crs(gdf.crs)
            
        # Create union of clip polygons
        clip_union = clip_geometry.union_all()
        
        # Apply buffer if configured
        if self.config.clip_buffer > 0:
            clip_union = clip_union.buffer(self.config.clip_buffer)
        
        # Determine expected geometry type for this layer
        expected_type = get_expected_geometry_type(layer_name)
        
        # Get original geometry type info
        original_geom_type = gdf.geometry.geom_type.iloc[0] if not gdf.empty else None
        logger.debug(f"Layer {layer_name}: original type={original_geom_type}, expected={expected_type}")
        
        # Perform clip based on geometry type
        if expected_type in ["MultiPoint", "Point"]:
            # For points, use spatial intersection (no clipping needed)
            clipped = gdf[gdf.geometry.intersects(clip_union)].copy()
        else:
            # For polygons and lines, use clip
            try:
                clipped = gpd.clip(gdf, clip_union)
            except Exception as e:
                logger.warning(f"Clip failed for {layer_name}, using intersection: {e}")
                clipped = gdf[gdf.geometry.intersects(clip_union)].copy()
        
        if clipped.empty:
            return clipped
        
        # Normalize geometries to expected type (handles GeometryCollection from clip)
        clipped = normalize_geodataframe_geometries(
            clipped,
            target_type=expected_type,
            preserve_z=self.config.preserve_z,
            min_area=1e-10
        )
        
        return clipped
    
    def _merge_spatial_layer(
        self,
        layer_name: str,
        mapsheets_by_source: Dict[str, gpd.GeoDataFrame],
        progress: Optional[Progress] = None,
        task_id: Optional[int] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """Merge a single spatial layer from all sources with geometry normalization."""
        
        merged_parts = []
        layer_feature_count = 0
        layer_crs = None
        
        # Get expected geometry type for this layer
        expected_type = get_expected_geometry_type(layer_name)
        logger.debug(f"Processing layer {layer_name}, expected type: {expected_type}")
        
        for source_name, source_mapsheets in mapsheets_by_source.items():
            if source_mapsheets.empty:
                continue
                
            # Resolve source path
            source_path = self._resolve_source_path(source_name)
            if source_path is None:
                msg = f"Source not found: {source_name}"
                logger.warning(msg)
                self.stats.warnings.append(msg)
                continue
                
            # Get bounding box for efficient reading
            bounds = source_mapsheets.total_bounds
            bbox = tuple(bounds)
            logger.debug(f"  {source_name}: bbox={bbox}")
            
            # Read layer from source
            gdf = self._read_layer(source_path, layer_name, bbox=bbox)
            if gdf is None or gdf.empty:
                logger.debug(f"  {source_name}: no features in {layer_name}")
                continue
            
            logger.debug(f"  {source_name}: read {len(gdf)} features")
            
            # Store CRS from first successful read
            if layer_crs is None and gdf.crs is not None:
                layer_crs = gdf.crs
                
            # Clip to mapsheet boundaries
            clipped = self._clip_layer(gdf, source_mapsheets, layer_name)
            
            if not clipped.empty:
                # Add source tracking column
                clipped["_MERGE_SOURCE"] = source_name
                merged_parts.append(clipped)
                
                feature_count = len(clipped)
                layer_feature_count += feature_count
                self.stats.features_per_source[source_name] = \
                    self.stats.features_per_source.get(source_name, 0) + feature_count
                    
                logger.debug(f"  {source_name}: {feature_count} features after clip")
                
            if progress and task_id:
                progress.advance(task_id)
                
        # Merge all parts
        if not merged_parts:
            return None
            
        if len(merged_parts) == 1:
            result = merged_parts[0]
        else:
            # Concatenate all parts
            result = pd.concat(merged_parts, ignore_index=True)
            
            # Convert back to GeoDataFrame with proper CRS
            if layer_crs is not None:
                result = gpd.GeoDataFrame(result, geometry="geometry", crs=layer_crs)
            else:
                result = gpd.GeoDataFrame(result, geometry="geometry")
        
        # Final normalization to ensure consistent geometry types
        logger.debug(f"  Normalizing {len(result)} features to {expected_type}")
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
            # Use pyogrio for non-spatial tables
            import pyogrio
            
            df = pyogrio.read_dataframe(
                reference_path,
                layer=table_name,
                read_geometry=False
            )
            return df
            
        except Exception as e:
            # Fallback: try with geopandas (may have geometry)
            try:
                gdf = gpd.read_file(reference_path, layer=table_name)
                # Drop geometry if it exists and is empty/null
                if 'geometry' in gdf.columns:
                    if gdf.geometry.isna().all() or gdf.geometry.is_empty.all():
                        return pd.DataFrame(gdf.drop(columns=['geometry']))
                return pd.DataFrame(gdf)
            except Exception as e2:
                logger.warning(f"Could not read table {table_name}: {e2}")
                return None
    
    def merge(self) -> MergeStats:
        """Execute the merge operation."""
        
        import time
        
        console.print("\n[bold blue]🔀 Starting GDB Merge Operation[/bold blue]\n")
        
        total_start_time = time.time()
        
        # Validate configuration
        self._validate_config()
        
        # Load mapsheets
        self._load_mapsheets()
        mapsheets_by_source = self._get_mapsheets_by_source()
        
        # Display source summary
        self._display_source_summary(mapsheets_by_source)
        
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
            
            # Main task for layers
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
                
                merged = self._merge_spatial_layer(layer_name, mapsheets_by_source)
                
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
            
        # Check at least one standard source exists
        has_standard = any(
            s.source_type in [SourceType.RC1, SourceType.RC2] 
            for s in self.sources.values()
        )
        if not has_standard:
            logger.warning("No standard RC1/RC2 sources found - only custom sources available")
    
    def _display_source_summary(self, mapsheets_by_source: Dict[str, gpd.GeoDataFrame]) -> None:
        """Display summary of sources and mapsheet assignments."""
        
        table = Table(title="Source Assignments", show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Mapsheets", justify="right", style="yellow")
        table.add_column("Status", style="magenta")
        
        for source_name, mapsheets in mapsheets_by_source.items():
            source_path = self._resolve_source_path(source_name)
            
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
                str(len(mapsheets)),
                status
            )
            
        console.print(table)
        console.print()
    
    def _save_merged_layers(self, merged_layers: Dict[str, gpd.GeoDataFrame]) -> None:
        """Save merged layers to output GDB with proper geometry handling."""
        
        output_path = self.config.output_path
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine output format from extension
        output_ext = output_path.suffix.lower()
        
        if output_ext == ".gdb":
            driver = "OpenFileGDB"
        elif output_ext == ".gpkg":
            driver = "GPKG"
        else:
            driver = "OpenFileGDB"  # Default to FileGDB
        
        logger.info(f"Output format: {driver}")
        
        # Get fields to exclude
        exclude_fields = self.config.exclude_fields or []
        if exclude_fields:
            logger.info(f"Excluding {len(exclude_fields)} metadata fields")
            logger.debug(f"  Fields to exclude: {exclude_fields}")
        
        first_layer = True
        
        for layer_name, gdf in merged_layers.items():
            actual_layer = self._get_layer_path(output_path, layer_name)
            
            if gdf.empty:
                logger.warning(f"Skipping empty layer: {actual_layer}")
                continue
            
            try:
                # Get expected geometry type
                expected_type = get_expected_geometry_type(layer_name)
                
                logger.debug(f"Saving {actual_layer}: {len(gdf)} features, type={expected_type}")
                
                # Ensure geometries are normalized
                gdf_to_save = normalize_geodataframe_geometries(
                    gdf, 
                    target_type=expected_type,
                    preserve_z=self.config.preserve_z
                )
                
                if gdf_to_save.empty:
                    logger.warning(f"No valid geometries to save for {actual_layer}")
                    continue
                
                # Remove excluded fields (but keep geometry column)
                if exclude_fields:
                    cols_to_drop = [col for col in exclude_fields if col in gdf_to_save.columns and col != "geometry"]
                    if cols_to_drop:
                        logger.debug(f"  Dropping {len(cols_to_drop)} excluded fields: {cols_to_drop}")
                        gdf_to_save = gdf_to_save.drop(columns=cols_to_drop)
                
                # Map to GDAL geometry type names for OpenFileGDB
                gdal_geom_types = {
                    "MultiPolygon": "MultiPolygon",
                    "Polygon": "MultiPolygon",  # Promote to Multi
                    "MultiLineString": "MultiLineString", 
                    "LineString": "MultiLineString",  # Promote to Multi
                    "MultiPoint": "MultiPoint",
                    "Point": "Point",
                }
                
                # Determine actual geometry type in data
                actual_types = gdf_to_save.geometry.geom_type.unique()
                logger.debug(f"  Geometry types in data: {list(actual_types)}")
                
                # Try to save with pyogrio for better control
                try:
                    import pyogrio
                    
                    # Determine mode
                    if first_layer and output_ext == ".gdb":
                        # For FileGDB, we need to delete existing if present
                        if output_path.exists():
                            import shutil
                            logger.debug(f"  Removing existing: {output_path}")
                            shutil.rmtree(output_path)
                    
                    logger.debug(f"  Writing with pyogrio (promote_to_multi=True)")
                    
                    # Write with pyogrio
                    pyogrio.write_dataframe(
                        gdf_to_save,
                        output_path,
                        layer=actual_layer,
                        driver=driver,
                        promote_to_multi=True,  # Important for consistent types
                    )
                    
                    first_layer = False
                    console.print(f"  [green]✓[/green] {actual_layer}: {len(gdf_to_save)} features")
                    
                except ImportError:
                    # Fallback to geopandas
                    logger.debug("  pyogrio not available, using geopandas")
                    
                    # For geopandas, handle mode differently
                    mode = "w" if first_layer else "a"
                    
                    # Try with 3D first
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
                        # Fallback: try forcing 2D
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
                
                # Try GPKG as ultimate fallback
                if driver == "OpenFileGDB":
                    try:
                        fallback_path = output_path.with_suffix(".gpkg")
                        logger.info(f"Trying GPKG fallback: {fallback_path}")
                        
                        gdf_2d = force_2d(gdf)
                        gdf_2d.to_file(
                            fallback_path,
                            layer=actual_layer,
                            driver="GPKG"
                        )
                        console.print(f"  [yellow]⚠[/yellow] {actual_layer}: saved to GPKG fallback")
                        self.stats.warnings.append(f"{actual_layer} saved to GPKG fallback")
                        
                    except Exception as e_fallback:
                        logger.error(f"GPKG fallback also failed: {e_fallback}")
    
    def _copy_reference_tables(self) -> None:
        """Copy non-spatial tables from reference source."""
        
        # Determine reference source
        ref_source_name = self.config.reference_source
        ref_path = self._resolve_source_path(ref_source_name)
        
        if ref_path is None:
            # Fallback to first available standard source
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
                    # Save as GPKG since OpenFileGDB doesn't support non-spatial well
                    # Alternative: save to output GDB with dummy geometry
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
        
        # Summary table
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
        
        console.print(table)
        
        # Features by source
        if self.stats.features_per_source:
            console.print("\n[bold]Features by Source:[/bold]")
            for source, count in sorted(self.stats.features_per_source.items(), key=lambda x: -x[1]):
                console.print(f"  • {source}: {count:,}")
                
        # Features by layer (if verbose)
        if self.verbose and self.stats.features_per_layer:
            console.print("\n[bold]Features by Layer:[/bold]")
            for layer, count in sorted(self.stats.features_per_layer.items()):
                layer_short = layer.split("/")[-1]
                console.print(f"  • {layer_short}: {count:,}")
        
        # Warnings and errors
        if self.stats.warnings:
            console.print("\n[yellow]⚠ Warnings:[/yellow]")
            for warning in self.stats.warnings:
                console.print(f"  • {warning}")
                
        if self.stats.errors:
            console.print("\n[red]✗ Errors:[/red]")
            for error in self.stats.errors:
                console.print(f"  • {error}")
                
        console.print(f"\n[dim]Output: {self.config.output_path}[/dim]")


def create_merge_config(
    rc1_path: Optional[Path] = None,
    rc2_path: Optional[Path] = None,
    custom_sources_dir: Optional[Path] = None,
    admin_zones_path: Path = None,
    output_path: Path = None,
    source_column: str = "SOURCE_RC",
    mapsheet_numbers: Optional[List[int]] = None,
    reference_source: str = "RC2",
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
    )
