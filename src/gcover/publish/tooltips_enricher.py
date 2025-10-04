# src/gcover/publish/tooltips_enricher.py
"""
Enhanced GeoCover Tooltips Enrichment Module

Supports multiple tooltip layers, flexible source mappings, and configurable workflows.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Set
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import traceback
from dataclasses import dataclass
from enum import Enum

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pyogrio.errors import DataLayerError

try:
    import arcpy

    HAS_ARCPY = True
except ImportError:
    HAS_ARCPY = False
    logger.warning("arcpy not available - using pure geopandas approach")

from ..sde.bridge import GCoverSDEBridge, create_bridge
from gcover.core.geometry import load_gpkg_with_validation, safe_read_filegdb


console = Console()


class LayerType(Enum):
    """Tooltip layer types with different processing requirements."""

    POLYGON = "polygon"
    LINE = "line"
    POINT = "point"


@dataclass
class LayerMapping:
    """Configuration for a tooltip layer and its source mappings."""

    tooltip_layer: str
    source_layers: List[str]
    layer_type: LayerType
    transfer_fields: List[str]
    # Spatial matching parameters specific to layer type
    area_threshold: float = 0.7
    buffer_distance: float = 0.5
    point_tolerance: float = 1.0  # For point layers


@dataclass
class EnrichmentConfig:
    """Complete configuration for enrichment process."""

    # Input paths
    tooltip_db_path: Path
    admin_zones_path: Path
    source_paths: Dict[str, Path]  # e.g., {'rc1': Path, 'rc2': Path, 'saas': Path}

    # Output configuration
    output_path: Optional[Path] = None
    debug_output_dir: Optional[Path] = None
    save_intermediate: bool = False

    # Processing parameters
    mapsheet_numbers: Optional[List[int]] = None
    clip_tolerance: float = 0.0

    # Layer configurations
    layer_mappings: Optional[Dict[str, LayerMapping]] = None


class EnhancedTooltipsEnricher:
    """
    Enhanced enricher supporting multiple layers, flexible source mappings,
    and configurable workflows.
    """

    # Default layer mappings
    DEFAULT_LAYER_MAPPINGS = {
        "POLYGON_MAIN": LayerMapping(
            tooltip_layer="POLYGON_MAIN",
            source_layers=[
                "GC_ROCK_BODIES/GC_BEDROCK",
                "GC_ROCK_BODIES/GC_UNCO_DESPOSIT",
            ],
            layer_type=LayerType.POLYGON,
            transfer_fields=[
                "UUID",
                "GEOLCODE",
                "GLAC_TYP",
                "CHRONO_T",
                "CHRONO_B",
                "gmu_code",
                "tecto",
                "tecto_code",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            area_threshold=0.7,
            buffer_distance=0.5,
        ),
        "POLYGON_AUX_1": LayerMapping(
            tooltip_layer="POLYGON_AUX_1",
            source_layers=[
                "GC_ROCK_BODIES/GC_SURFACES",
                "GC_ROCK_BODIES/GC_EXPLOIT_GEOMAT_PLG",
            ],
            layer_type=LayerType.POLYGON,
            transfer_fields=[
                "UUID",
                "SURFACE_TYPE",
                "SURFACE_QUALITY",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            area_threshold=0.6,
            buffer_distance=1.0,
        ),
        "POLYGON_AUX_2": LayerMapping(
            tooltip_layer="POLYGON_AUX_2",
            source_layers=[
                "GC_ROCK_BODIES/GC_SURFACES",
                "GC_ROCK_BODIES/GC_EXPLOIT_GEOMAT_PLG",
            ],
            layer_type=LayerType.POLYGON,
            transfer_fields=[
                "UUID",
                "SURFACE_TYPE",
                "SURFACE_QUALITY",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            area_threshold=0.6,
            buffer_distance=1.0,
        ),
        "LINE_AUX": LayerMapping(
            tooltip_layer="LINE_AUX",
            source_layers=["GC_ROCK_BODIES/GC_LINEAR_OBJECTS"],
            layer_type=LayerType.LINE,
            transfer_fields=[
                "UUID",
                "LINEAR_TYPE",
                "LINEAR_CATEGORY",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            buffer_distance=2.0,
        ),
        "POINT_GEOL": LayerMapping(
            tooltip_layer="POINT_GEOL",
            source_layers=[
                "GC_ROCK_BODIES/GC_POINT_OBJECTS",
                "GC_ROCK_BODIES/GC_FOSSILS",
            ],
            layer_type=LayerType.POINT,
            transfer_fields=[
                "UUID",
                "POINT_TYPE",
                "POINT_CATEGORY",
                "FOSSIL_TYPE",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            point_tolerance=5.0,
        ),
        "POINT_HYDRO": LayerMapping(
            tooltip_layer="POINT_HYDRO",
            source_layers=["GC_ROCK_BODIES/GC_POINT_OBJECTS"],
            layer_type=LayerType.POINT,
            transfer_fields=[
                "UUID",
                "POINT_TYPE",
                "HYDRO_TYPE",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            point_tolerance=5.0,
        ),
        "POINT_STRUCT": LayerMapping(
            tooltip_layer="POINT_STRUCT",
            source_layers=["GC_ROCK_BODIES/GC_POINT_OBJECTS"],
            layer_type=LayerType.POINT,
            transfer_fields=[
                "UUID",
                "POINT_TYPE",
                "STRUCT_TYPE",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            point_tolerance=5.0,
        ),
        "POINT_DRILL": LayerMapping(
            tooltip_layer="POINT_DRILL",
            source_layers=["GC_ROCK_BODIES/GC_POINT_OBJECTS"],
            layer_type=LayerType.POINT,
            transfer_fields=[
                "UUID",
                "POINT_TYPE",
                "DRILL_TYPE",
                "DEPTH",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            point_tolerance=2.0,
        ),
        "POINT_INFO": LayerMapping(
            tooltip_layer="POINT_INFO",
            source_layers=["GC_ROCK_BODIES/GC_POINT_OBJECTS"],
            layer_type=LayerType.POINT,
            transfer_fields=[
                "UUID",
                "POINT_TYPE",
                "INFO_TYPE",
                "OPERATOR",
                "DATEOFCHANGE",
            ],
            point_tolerance=5.0,
        ),
    }

    def __init__(self, config: EnrichmentConfig):
        """
        Initialize with comprehensive configuration.

        Args:
            config: EnrichmentConfig with all necessary settings
        """
        self.config = config

        # Setup paths
        self.tooltip_db_path = config.tooltip_db_path
        self.admin_zones_path = config.admin_zones_path
        self.source_paths = config.source_paths

        # Setup layer mappings (use defaults if not specified)
        self.layer_mappings = config.layer_mappings or self.DEFAULT_LAYER_MAPPINGS

        # Setup temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gcover_enrich_"))
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Setup debug output directory
        self.debug_output_dir = config.debug_output_dir
        if self.debug_output_dir:
            self.debug_output_dir.mkdir(exist_ok=True, parents=True)

        # Validate inputs
        self._validate_configuration()

        logger.info(f"Enhanced TooltipsEnricher initialized")
        logger.info(f"Tooltip layers: {list(self.layer_mappings.keys())}")
        logger.info(f"Data sources: {list(self.source_paths.keys())}")

    def _validate_configuration(self) -> None:
        """Validate configuration and inputs."""
        # Check essential paths
        if not self.tooltip_db_path.exists():
            raise FileNotFoundError(
                f"Tooltip database not found: {self.tooltip_db_path}"
            )
        if not self.admin_zones_path.exists():
            raise FileNotFoundError(
                f"Admin zones file not found: {self.admin_zones_path}"
            )

        # Check source paths
        for source_name, source_path in self.source_paths.items():
            if not source_path.exists():
                logger.warning(f"Source {source_name} not found: {source_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup temporary files."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean temp directory: {e}")

    def load_mapsheets_info(self) -> gpd.GeoDataFrame:
        """Load mapsheet boundaries and their source information."""
        try:
            mapsheets = gpd.read_file(
                self.admin_zones_path, layer="mapsheets_sources_only"
            )
            logger.info(f"Loaded {len(mapsheets)} mapsheets from admin zones")

            # Get unique sources to validate against our source_paths
            available_sources = set(mapsheets["SOURCE_RC"].unique())
            configured_sources = set(self.source_paths.keys())

            logger.info(f"Available sources in mapsheets: {available_sources}")
            logger.info(f"Configured sources: {configured_sources}")

            missing_sources = available_sources - configured_sources
            if missing_sources:
                logger.warning(f"Missing source configurations: {missing_sources}")

            return mapsheets
        except Exception as e:
            raise RuntimeError(f"Error loading mapsheets: {e}")

    def load_tooltips_layer(self, layer_name: str) -> gpd.GeoDataFrame:
        """Load a specific layer from the tooltips database."""
        try:
            tooltips = gpd.read_file(self.tooltip_db_path, layer=layer_name)
            # Ensure we have a unique identifier
            if "OBJECTID" not in tooltips.columns:
                tooltips["OBJECTID"] = tooltips.index

            logger.info(f"Loaded {len(tooltips)} features from {layer_name}")
            return tooltips
        except Exception as e:
            logger.error(f"Error loading {layer_name}: {e}")
            return gpd.GeoDataFrame()

    def get_source_for_mapsheet(
        self, mapsheet_nbr: int, mapsheets: gpd.GeoDataFrame
    ) -> str:
        """Get source identifier for a given mapsheet number."""
        match = mapsheets[mapsheets["MSH_MAP_NBR"] == mapsheet_nbr]
        if match.empty:
            raise ValueError(f"Mapsheet {mapsheet_nbr} not found")

        source_rc = match.iloc[0]["SOURCE_RC"]

        # Map SOURCE_RC to our source_paths keys (case-insensitive)
        for source_key in self.source_paths.keys():
            if source_key.lower() == source_rc.lower():
                return source_key

        # If no exact match, try common mappings
        source_mapping = {
            "rc1": ["rc1", "2016-12-31"],
            "rc2": ["rc2", "2030-12-31"],
            "saas": ["saas"],
            "bkp_2016": ["bkp_2016", "bkp_2026"],  # Handle naming variations
        }

        for source_key, alternatives in source_mapping.items():
            if source_key in self.source_paths and source_rc.lower() in [
                alt.lower() for alt in alternatives
            ]:
                return source_key

        raise ValueError(
            f"No configured source found for mapsheet {mapsheet_nbr} with SOURCE_RC '{source_rc}'"
        )

    def clip_source_data_by_mapsheet(
        self, mapsheet_nbr: int, source_key: str, layer_mapping: LayerMapping
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Clip source data by mapsheet boundary for a specific layer mapping.

        Args:
            mapsheet_nbr: Mapsheet number
            source_key: Source identifier (e.g., 'rc1', 'rc2', 'saas')
            layer_mapping: Layer mapping configuration

        Returns:
            Dictionary of clipped GeoDataFrames
        """
        # Get mapsheet geometry
        mapsheets = self.load_mapsheets_info()
        mapsheet_geom = mapsheets[mapsheets["MSH_MAP_NBR"] == mapsheet_nbr]

        if mapsheet_geom.empty:
            raise ValueError(f"Mapsheet {mapsheet_nbr} not found")

        clip_boundary = mapsheet_geom.geometry.iloc[0]
        clip_bounds = clip_boundary.bounds

        # Apply clip tolerance if specified
        if self.config.clip_tolerance > 0.0:
            minx, miny, maxx, maxy = clip_bounds
            bbox_geom = box(minx, miny, maxx, maxy)
            clip_bounds = bbox_geom.buffer(self.config.clip_tolerance).bounds
            logger.debug(
                f"Applied clip tolerance of {self.config.clip_tolerance}m to mapsheet {mapsheet_nbr}"
            )

        mapsheet_geom = mapsheet_geom.set_crs("EPSG:2056", allow_override=True)

        logger.info(f"Clipping {source_key} data for mapsheet {mapsheet_nbr}")
        logger.debug(f"Processing layers: {layer_mapping.source_layers}")

        clipped_data = {}
        source_path = self.source_paths[source_key]

        for source_layer in layer_mapping.source_layers:
            try:
                # Extract layer name (handle group prefixes)
                layer_name = source_layer.split("/")[-1]
                logger.debug(f"Reading layer {layer_name} from {source_path}")

                # Load data with spatial filtering
                gdf = load_gpkg_with_validation(
                    source_path,
                    repair_geometries=True,
                    layer=layer_name,
                    bbox=clip_bounds,
                )

                if gdf.empty:
                    logger.warning(
                        f"No data found in {layer_name} for mapsheet {mapsheet_nbr}"
                    )
                    clipped_data[source_layer] = gpd.GeoDataFrame()
                    continue

                gdf = gdf.set_crs("EPSG:2056", allow_override=True)

                # Save intermediate clipped data if debug enabled
                if self.debug_output_dir:
                    debug_file = (
                        self.debug_output_dir
                        / f"clipped_{source_key}_{layer_name}_{mapsheet_nbr}.gpkg"
                    )
                    gdf.to_file(debug_file, layer=layer_name, driver="GPKG")
                    logger.debug(f"Saved debug file: {debug_file}")

                clipped_data[source_layer] = gdf
                logger.info(f"Clipped {len(gdf)} features from {source_layer}")

            except DataLayerError as de:
                logger.error(f"Error clipping {source_layer}: {de}")
                raise DataLayerError(f"Cannot load {source_layer}: {de}")

            except Exception as e:
                logger.error(f"Error clipping {source_layer}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                clipped_data[source_layer] = gpd.GeoDataFrame()

        return clipped_data

    def spatial_match_features(
        self,
        tooltip_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        layer_mapping: LayerMapping,
    ) -> pd.DataFrame:
        """
        Perform spatial matching optimized for different geometry types.

        Args:
            tooltip_features: Target features to enrich
            source_features: Source features to match against
            layer_mapping: Layer configuration with matching parameters

        Returns:
            DataFrame with OBJECTID-UUID matches
        """
        if source_features.empty:
            return pd.DataFrame(
                columns=["OBJECTID", "UUID", "match_method", "confidence"]
            )

        logger.info(
            f"Spatial matching {len(tooltip_features)} tooltip features against {len(source_features)} source features"
        )
        logger.debug(f"Layer type: {layer_mapping.layer_type}")

        # Use appropriate matching strategy based on geometry type
        if layer_mapping.layer_type == LayerType.POLYGON:
            return self._match_polygons(
                tooltip_features, source_features, layer_mapping
            )
        elif layer_mapping.layer_type == LayerType.LINE:
            return self._match_lines(tooltip_features, source_features, layer_mapping)
        elif layer_mapping.layer_type == LayerType.POINT:
            return self._match_points(tooltip_features, source_features, layer_mapping)
        else:
            raise ValueError(f"Unsupported layer type: {layer_mapping.layer_type}")

    def _match_polygons(
        self,
        tooltip_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        layer_mapping: LayerMapping,
    ) -> pd.DataFrame:
            """Fast polygon matching for nearly-identical geometries (Shapely 2.x compatible)."""

            # Ensure CRS compatibility
            if tooltip_features.crs != source_features.crs:
                source_features = source_features.to_crs(tooltip_features.crs)

            matches = []

            # Build spatial index
            source_sindex = source_features.sindex

            # Create lookup dict for faster access
            source_dict = {idx: row for idx, row in source_features.iterrows()}

            # Helper function to compare geometries with tolerance (Shapely 2.x compatible)
            def geometries_almost_equal(geom1, geom2, tolerance=1e-6):
                """Check if two geometries are almost equal within tolerance."""
                try:
                    # Method 1: Use equals_exact (Shapely 2.x)
                    if hasattr(geom1, 'equals_exact'):
                        return geom1.equals_exact(geom2, tolerance)
                    # Method 2: Fallback for Shapely 1.x
                    elif hasattr(geom1, 'almost_equals'):
                        return geom1.almost_equals(geom2, decimal=6)
                    # Method 3: Manual check using symmetric difference
                    else:
                        sym_diff = geom1.symmetric_difference(geom2)
                        return sym_diff.area < tolerance
                except Exception:
                    return False

            with Progress() as progress:
                task = progress.add_task(
                    f"Matching {layer_mapping.tooltip_layer} polygons...",
                    total=len(tooltip_features)
                )

                for idx, tooltip_feat in tooltip_features.iterrows():
                    tooltip_geom = tooltip_feat.geometry
                    progress.update(task, advance=1)

                    # Validate geometry
                    if tooltip_geom is None or tooltip_geom.is_empty:
                        logger.warning(f"Skipping empty geometry at index {idx}")
                        continue

                    # Quick spatial index query
                    possible_matches_idx = list(source_sindex.intersection(tooltip_geom.bounds))

                    if not possible_matches_idx:
                        continue

                    best_match = None
                    best_score = 0
                    match_method = "none"

                    for src_idx in possible_matches_idx:
                        src_feat = source_dict[src_idx]
                        src_geom = src_feat.geometry

                        # Validate source geometry
                        if src_geom is None or src_geom.is_empty:
                            continue

                        # Test 1: Exact equality (fastest)
                        if tooltip_geom.equals(src_geom):
                            best_match = src_feat["UUID"]
                            best_score = 1.0
                            match_method = "exact"
                            break

                        # Test 2: Almost equals with tolerance (handles rounding)
                        if geometries_almost_equal(tooltip_geom, src_geom, tolerance=1e-6):
                            best_match = src_feat["UUID"]
                            best_score = 0.95
                            match_method = "almost_exact"
                            break

                        # Test 3: High overlap for slightly different geometries
                        if tooltip_geom.intersects(src_geom):
                            try:
                                intersection = tooltip_geom.intersection(src_geom)
                                # Use Jaccard index (IoU)
                                union = tooltip_geom.union(src_geom)
                                iou = intersection.area / union.area if union.area > 0 else 0

                                if iou > best_score and iou > layer_mapping.area_threshold:
                                    best_match = src_feat["UUID"]
                                    best_score = iou
                                    match_method = "iou"
                            except Exception as e:
                                logger.debug(f"Error calculating overlap for idx {idx}: {e}")
                                continue

                    if best_match:
                        matches.append({
                            "OBJECTID": tooltip_feat["OBJECTID"],
                            "UUID": best_match,
                            "match_method": match_method,
                            "confidence": round(best_score, 3),
                        })
                    else:
                        logger.debug(f"No match found for OBJECTID {tooltip_feat.get('OBJECTID')}")

            matches_df = pd.DataFrame(matches)
            logger.info(f"Found {len(matches_df)} polygon matches out of {len(tooltip_features)} features")
            return matches_df

    def _match_polygons_ori(
        self,
        tooltip_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        layer_mapping: LayerMapping,
    ) -> pd.DataFrame:
        """Polygon-specific spatial matching using IoU and boundary similarity."""
        matches = []

        logger.add(self.debug_output_dir / "_match_polygons.log")

        # Ensure CRS compatibility
        if tooltip_features.crs != source_features.crs:
            source_features = source_features.to_crs(tooltip_features.crs)

        # Sort tooltips ascending
        tooltip_features = tooltip_features.assign(_area=tooltip_features.geometry.area)
        tooltip_features = tooltip_features.sort_values(
            by="_area", ascending=True
        ).drop(columns="_area")

        source_features = source_features.assign(_area=source_features.geometry.area)
        source_features = source_features.sort_values(by="_area", ascending=True).drop(
            columns="_area"
        )

        # Pre-calculate spatial index for performance
        source_sindex = (
            source_features.sindex if hasattr(source_features, "sindex") else None
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Matching {layer_mapping.tooltip_layer} polygons...",
                total=len(tooltip_features),
            )

            for idx, tooltip_feat in tooltip_features.iterrows():
                tooltip_geom = tooltip_feat.geometry
                progress.update(task, advance=1)

                logger.info(
                    f"Feat: {idx}, ID={tooltip_feat.get('OBJECTID')}, DESC={tooltip_feat.get('DESCRIPT_D')}"
                )

                # Use spatial index for faster querying
                if source_sindex:
                    possible_matches_idx = list(
                        source_sindex.intersection(tooltip_geom.bounds)
                    )
                    intersecting = source_features.iloc[possible_matches_idx]
                    intersecting = intersecting[
                        intersecting.geometry.intersects(tooltip_geom)
                    ]
                else:
                    intersecting = source_features[
                        source_features.geometry.intersects(tooltip_geom)
                    ]

                if intersecting.empty:
                    continue

                logger.info(f"Found {len(intersecting)} intersections")

                best_match = None
                best_confidence = 0
                match_method = "none"

                for src_idx, src_feat in intersecting.iterrows():
                    src_geom = src_feat.geometry

                    logger.info(
                        f"   Trying match on  {src_idx} {src_feat.get('UUID')} KIND:{src_feat.get('KIND')}"
                    )

                    # Method 1: Intersection over Union (IoU)
                    try:
                        intersection = tooltip_geom.intersection(src_geom)
                        if not intersection.is_empty:
                            union = tooltip_geom.union(src_geom)
                            iou = intersection.area / union.area

                            if (
                                iou > layer_mapping.area_threshold
                                and iou > best_confidence
                            ):
                                best_match = src_feat["UUID"]
                                best_confidence = iou
                                match_method = "iou"
                                logger.info(f"   Matching")
                                continue
                    except Exception as e:
                        logger.debug(f"Error calculating IoU: {e}")

                    # Method 2: Simple area overlap
                    try:
                        intersection = tooltip_geom.intersection(src_geom)
                        if not intersection.is_empty:
                            overlap_ratio = intersection.area / tooltip_geom.area

                            if (
                                overlap_ratio > layer_mapping.area_threshold
                                and overlap_ratio > best_confidence
                            ):
                                best_match = src_feat["UUID"]
                                best_confidence = overlap_ratio
                                match_method = "area_overlap"
                    except Exception as e:
                        logger.debug(f"Error calculating area overlap: {e}")

                # Method 3: Buffered matching for edge cases
                if best_match is None and layer_mapping.buffer_distance > 0:
                    try:
                        buffered_tooltip = tooltip_geom.buffer(
                            layer_mapping.buffer_distance
                        )
                        buffered_intersecting = source_features[
                            source_features.geometry.intersects(buffered_tooltip)
                        ]

                        for src_idx, src_feat in buffered_intersecting.iterrows():
                            intersection = buffered_tooltip.intersection(
                                src_feat.geometry
                            )
                            if not intersection.is_empty:
                                confidence = intersection.area / tooltip_geom.area
                                if confidence > best_confidence:
                                    best_match = src_feat["UUID"]
                                    best_confidence = confidence
                                    match_method = "buffered"
                    except Exception as e:
                        logger.debug(f"Error in buffered matching: {e}")

                if best_match:
                    matches.append(
                        {
                            "OBJECTID": tooltip_feat["OBJECTID"],
                            "UUID": best_match,
                            "match_method": match_method,
                            "confidence": round(best_confidence, 3),
                        }
                    )

        matches_df = pd.DataFrame(matches)
        logger.info(f"Found {len(matches_df)} polygon matches")
        return matches_df

    def _match_lines(
        self,
        tooltip_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        layer_mapping: LayerMapping,
    ) -> pd.DataFrame:
        """Line-specific spatial matching using buffer zones and line overlap."""
        matches = []

        # Ensure CRS compatibility
        if tooltip_features.crs != source_features.crs:
            source_features = source_features.to_crs(tooltip_features.crs)

        buffer_distance = layer_mapping.buffer_distance

        for idx, tooltip_feat in tooltip_features.iterrows():
            tooltip_geom = tooltip_feat.geometry

            # Buffer the tooltip line for intersection testing
            buffered_tooltip = tooltip_geom.buffer(buffer_distance)

            # Find intersecting source lines
            intersecting = source_features[
                source_features.geometry.intersects(buffered_tooltip)
            ]

            if intersecting.empty:
                continue

            best_match = None
            best_confidence = 0
            match_method = "none"

            for src_idx, src_feat in intersecting.iterrows():
                src_geom = src_feat.geometry

                try:
                    # Calculate line overlap using buffer intersection
                    src_buffered = src_geom.buffer(buffer_distance)
                    intersection = buffered_tooltip.intersection(src_buffered)

                    if not intersection.is_empty:
                        # Confidence based on intersection length vs tooltip length
                        confidence = intersection.length / tooltip_geom.length

                        if confidence > best_confidence:
                            best_match = src_feat["UUID"]
                            best_confidence = confidence
                            match_method = "line_buffer_overlap"

                except Exception as e:
                    logger.debug(f"Error in line matching: {e}")

            if best_match:
                matches.append(
                    {
                        "OBJECTID": tooltip_feat["OBJECTID"],
                        "UUID": best_match,
                        "match_method": match_method,
                        "confidence": round(best_confidence, 3),
                    }
                )

        matches_df = pd.DataFrame(matches)
        logger.info(f"Found {len(matches_df)} line matches")
        return matches_df

    def _match_points(
        self,
        tooltip_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        layer_mapping: LayerMapping,
    ) -> pd.DataFrame:
        """Point-specific spatial matching using distance tolerance."""
        matches = []

        # Ensure CRS compatibility
        if tooltip_features.crs != source_features.crs:
            source_features = source_features.to_crs(tooltip_features.crs)

        tolerance = layer_mapping.point_tolerance

        for idx, tooltip_feat in tooltip_features.iterrows():
            tooltip_geom = tooltip_feat.geometry

            # Find source points within tolerance
            distances = source_features.geometry.distance(tooltip_geom)
            within_tolerance = distances <= tolerance

            if not within_tolerance.any():
                continue

            # Get the closest point within tolerance
            closest_idx = distances[within_tolerance].idxmin()
            closest_distance = distances.loc[closest_idx]
            closest_feature = source_features.loc[closest_idx]

            # Confidence based on inverse distance (closer = higher confidence)
            confidence = 1.0 - (closest_distance / tolerance)

            matches.append(
                {
                    "OBJECTID": tooltip_feat["OBJECTID"],
                    "UUID": closest_feature["UUID"],
                    "match_method": "point_distance",
                    "confidence": round(confidence, 3),
                }
            )

        matches_df = pd.DataFrame(matches)
        logger.info(f"Found {len(matches_df)} point matches")
        return matches_df

    def transfer_attributes(
        self,
        tooltip_features: gpd.GeoDataFrame,
        source_features: Dict[str, gpd.GeoDataFrame],
        matches: pd.DataFrame,
        layer_mapping: LayerMapping,
    ) -> gpd.GeoDataFrame:
        """
        Transfer attributes from source features to tooltip features.

        Args:
            tooltip_features: Target GeoDataFrame to enrich
            source_features: Dict of source GeoDataFrames
            matches: DataFrame with OBJECTID-UUID matches
            layer_mapping: Layer configuration with transfer fields

        Returns:
            Enriched tooltip GeoDataFrame
        """
        # Create enriched copy
        enriched = tooltip_features.copy()

        # Initialize new columns from layer mapping
        for field in layer_mapping.transfer_fields:
            if field not in enriched.columns:
                enriched[field] = None

        # Add tracking fields
        enriched["SOURCE_UUID"] = None
        enriched["MATCH_METHOD"] = None
        enriched["MATCH_CONFIDENCE"] = 0.0
        enriched["MATCH_LAYER"] = None

        # Combine all source features
        all_sources = []
        for source_layer, source_gdf in source_features.items():
            if not source_gdf.empty:
                source_copy = source_gdf.copy()
                source_copy["_source_layer"] = source_layer
                all_sources.append(source_copy)

        if not all_sources:
            logger.warning("No source features available for attribute transfer")
            return enriched

        combined_sources = pd.concat(all_sources, ignore_index=True)

        # Transfer attributes for each match
        transferred_count = 0

        for _, match_row in matches.iterrows():
            objectid = match_row["OBJECTID"]
            uuid = match_row["UUID"]

            # Find tooltip feature
            tooltip_idx = enriched[enriched["OBJECTID"] == objectid].index
            if tooltip_idx.empty:
                continue
            tooltip_idx = tooltip_idx[0]

            # Find source feature
            source_feat = combined_sources[combined_sources["UUID"] == uuid]
            if source_feat.empty:
                continue
            source_feat = source_feat.iloc[0]

            # Transfer attributes
            for field in layer_mapping.transfer_fields:
                if field in source_feat.index and pd.notna(source_feat[field]):
                    enriched.loc[tooltip_idx, field] = source_feat[field]

            # Set tracking fields
            enriched.loc[tooltip_idx, "SOURCE_UUID"] = uuid
            enriched.loc[tooltip_idx, "MATCH_METHOD"] = match_row["match_method"]
            enriched.loc[tooltip_idx, "MATCH_CONFIDENCE"] = match_row["confidence"]
            enriched.loc[tooltip_idx, "MATCH_LAYER"] = source_feat[
                "_source_layer"
            ].split("/")[-1]

            transferred_count += 1

        logger.info(
            f"Transferred attributes to {transferred_count}/{len(tooltip_features)} features"
        )
        return enriched

    def enrich_layer(
        self, layer_name: str, mapsheet_numbers: Optional[List[int]] = None
    ) -> gpd.GeoDataFrame:
        """
        Enrich a specific tooltip layer.

        Args:
            layer_name: Name of the tooltip layer to enrich
            mapsheet_numbers: Specific mapsheets to process (None = all)

        Returns:
            Enriched GeoDataFrame
        """
        if layer_name not in self.layer_mappings:
            raise ValueError(f"No configuration found for layer {layer_name}")

        layer_mapping = self.layer_mappings[layer_name]
        logger.info(f"Starting enrichment for {layer_name}")

        # Load tooltip data
        tooltip_features = self.load_tooltips_layer(layer_name)
        if tooltip_features.empty:
            logger.warning(f"No features found in {layer_name}")
            return tooltip_features

        # Load mapsheet info
        mapsheets = self.load_mapsheets_info()

        # Determine mapsheets to process
        if mapsheet_numbers is None:
            unique_mapsheets = mapsheets["MSH_MAP_NBR"].unique()
        else:
            unique_mapsheets = mapsheet_numbers

        logger.info(f"Processing {len(unique_mapsheets)} mapsheets for {layer_name}")

        all_enriched = []

        for mapsheet_nbr in unique_mapsheets:
            logger.info(f"Processing mapsheet {mapsheet_nbr} for {layer_name}")
            console.print(
                f"[blue]Processing mapsheet {mapsheet_nbr} for {layer_name}[/blue]"
            )

            try:
                # Get source for this mapsheet
                source_key = self.get_source_for_mapsheet(mapsheet_nbr, mapsheets)

                # Skip if source not configured
                if source_key not in self.source_paths:
                    logger.warning(
                        f"Source {source_key} not configured for mapsheet {mapsheet_nbr}"
                    )
                    continue

                # Clip source data
                clipped_sources = self.clip_source_data_by_mapsheet(
                    mapsheet_nbr, source_key, layer_mapping
                )

                # Skip if no source data
                if all(gdf.empty for gdf in clipped_sources.values()):
                    logger.warning(f"No source data for mapsheet {mapsheet_nbr}")
                    continue

                # TODO logger.debug(f"Source columns: {clipped_sources['bedrock'].columns}")

                # Get mapsheet boundary for filtering tooltips
                mapsheet_boundary = mapsheets[
                    mapsheets["MSH_MAP_NBR"] == mapsheet_nbr
                ].geometry.iloc[0]

                # Filter tooltip features for this mapsheet
                tooltip_subset = tooltip_features[
                    tooltip_features.geometry.intersects(mapsheet_boundary)
                ].copy()

                if tooltip_subset.empty:
                    logger.warning(f"No tooltip features for mapsheet {mapsheet_nbr}")
                    continue

                logger.info(f"Processing {len(tooltip_subset)} tooltip features")

                # Spatial matching for each source layer
                all_matches = []

                for source_layer, source_gdf in clipped_sources.items():
                    if source_gdf.empty:
                        continue
                    logger.info(
                        f"[blue]==== Matching against {source_layer} ====[/blue]"
                    )
                    matches = self.spatial_match_features(
                        tooltip_subset, source_gdf, layer_mapping
                    )

                    if not matches.empty:
                        matches["source_fc"] = source_layer
                        all_matches.append(matches)

                if not all_matches:
                    logger.warning(f"No spatial matches for mapsheet {mapsheet_nbr}")
                    # Still add the tooltip subset with empty enrichment
                    tooltip_subset["MAPSHEET_NBR"] = mapsheet_nbr
                    tooltip_subset["SOURCE_KEY"] = source_key
                    all_enriched.append(tooltip_subset)
                    continue

                # Combine matches and remove duplicates (keep best confidence)
                combined_matches = pd.concat(all_matches, ignore_index=True)
                combined_matches = combined_matches.sort_values(
                    "confidence", ascending=False
                )
                combined_matches = combined_matches.drop_duplicates(
                    "OBJECTID", keep="first"
                )

                # Transfer attributes
                enriched_subset = self.transfer_attributes(
                    tooltip_subset, clipped_sources, combined_matches, layer_mapping
                )

                # Add tracking information
                enriched_subset["MAPSHEET_NBR"] = mapsheet_nbr
                enriched_subset["SOURCE_KEY"] = source_key

                all_enriched.append(enriched_subset)

                # TODO logger.debug(f"Enriched columns: {enriched_subset.columns}")

                # Save intermediate results if requested
                if self.config.save_intermediate and self.debug_output_dir:
                    temp_path = (
                        self.debug_output_dir
                        / f"{layer_name}_mapsheet_{mapsheet_nbr}_enriched.gpkg"
                    )
                    enriched_subset.to_file(temp_path, driver="GPKG")
                    logger.debug(f"Saved intermediate result: {temp_path}")

                logger.info(
                    f"Completed mapsheet {mapsheet_nbr}: {len(enriched_subset)} features processed"
                )

            except Exception as e:
                logger.error(
                    f"Error processing mapsheet {mapsheet_nbr} for {layer_name}: {e}"
                )
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        if not all_enriched:
            logger.error(f"No mapsheets were successfully processed for {layer_name}")
            return gpd.GeoDataFrame()

        # Combine all results
        final_result = pd.concat(all_enriched, ignore_index=True)
        logger.info(
            f"Enrichment complete for {layer_name}: {len(final_result)} total features"
        )

        return final_result

    def enrich_all_layers(
        self,
        layer_names: Optional[List[str]] = None,
        mapsheet_numbers: Optional[List[int]] = None,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Enrich multiple tooltip layers.

        Args:
            layer_names: Specific layers to process (None = all configured layers)
            mapsheet_numbers: Specific mapsheets to process (None = all)

        Returns:
            Dictionary mapping layer names to enriched GeoDataFrames
        """
        if layer_names is None:
            layer_names = list(self.layer_mappings.keys())

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Enriching layers...", total=len(layer_names))

            for layer_name in layer_names:
                progress.update(task, description=f"Enriching {layer_name}...")

                try:
                    enriched = self.enrich_layer(layer_name, mapsheet_numbers)
                    results[layer_name] = enriched

                    progress.update(task, advance=1)

                except Exception as e:
                    logger.error(f"Failed to enrich {layer_name}: {e}")
                    results[layer_name] = gpd.GeoDataFrame()
                    progress.update(task, advance=1)

        return results

    def save_enriched_data(
        self,
        enriched_data: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save enriched data to output file."""
        if output_path is None:
            output_path = self.config.output_path

        if output_path is None:
            raise ValueError("No output path specified")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(enriched_data, dict):
            # Save multiple layers to GPKG
            for layer_name, gdf in enriched_data.items():
                if not gdf.empty:
                    layer_output_name = f"{layer_name}_ENRICHED"
                    gdf.to_file(output_path, layer=layer_output_name, driver="GPKG")
                    logger.info(
                        f"Saved {len(gdf)} features to layer {layer_output_name}"
                    )
        else:
            # Save single layer
            enriched_data.to_file(output_path, driver="GPKG")
            logger.info(f"Saved {len(enriched_data)} features to {output_path}")

        return output_path


def create_enrichment_config(
    tooltip_db_path: Union[str, Path],
    admin_zones_path: Union[str, Path],
    source_paths: Dict[str, Union[str, Path]],
    **kwargs,
) -> EnrichmentConfig:
    """Helper function to create enrichment configuration."""
    return EnrichmentConfig(
        tooltip_db_path=Path(tooltip_db_path),
        admin_zones_path=Path(admin_zones_path),
        source_paths={k: Path(v) for k, v in source_paths.items()},
        **kwargs,
    )
