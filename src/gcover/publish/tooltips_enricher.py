# src/gcover/publish/tooltips_enricher.py
"""
GeoCover Tooltips Enrichment Module

Enriches lightweight geocover_tooltips with data from original RC databases
through intelligent spatial matching and attribute transfer.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import traceback

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
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel

try:
    import arcpy

    HAS_ARCPY = True
except ImportError:
    HAS_ARCPY = False
    logger.warning("arcpy not available - using pure geopandas approach")

from ..sde.bridge import GCoverSDEBridge, create_bridge
from gcover.core.geometry import load_gpkg_with_validation, safe_read_filegdb


class TooltipsEnricher:
    """
    Enriches geocover_tooltips with information from original RC databases
    through spatial matching and intelligent attribute transfer.
    """

    # Default field mappings for different feature types
    BEDROCK_FIELDS = [
        "UUID",
        "GEOL_MAPPING_UNIT_ATT_UUID",
        "TECTO",
        "OPERATOR",
        "DATEOFCHANGE",
        "tecto_code",
        "gmu_code",
    ]

    UNCO_FIELDS = [
        "UUID",
        "GEOLCODE",
        "GLAC_TYP",
        "CHRONO_T",
        "CHRONO_B",
        "GMU_CODE",
        "COMPOSIT",
        "ADMIXTURE",
        "OPERATOR",
        "DATEOFCHANGE",
    ]

    # Spatial matching parameters
    DEFAULT_AREA_THRESHOLD = 0.7  # Minimum overlap ratio for matching
    DEFAULT_BUFFER_DISTANCE = 0.5  # Buffer for edge matching (meters)

    def __init__(
        self,
        tooltips_gdb: Union[str, Path],
        admin_zones_gpkg: Union[str, Path],
        rc_data_sources: Optional[Dict[str, Union[str, Path]]] = None,
        sde_bridge: Optional[GCoverSDEBridge] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the enricher.

        Args:
            tooltips_gdb: Path to geocover_tooltips.gdb
            admin_zones_gpkg: Path to administrative_zones.gpkg
            rc_data_sources: Dict mapping RC versions to FileGDB paths
                           {'RC1': '/path/to/2016-12-31.gdb', 'RC2': '/path/to/2030-12-31.gdb'}
            sde_bridge: Optional SDE bridge for live data access
            temp_dir: Temporary directory for intermediate files
        """
        self.tooltips_gdb = Path(tooltips_gdb)
        self.admin_zones_gpkg = Path(admin_zones_gpkg)
        self.rc_data_sources = rc_data_sources or {}
        self.sde_bridge = sde_bridge
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Validate inputs
        if not self.tooltips_gdb.exists():
            raise FileNotFoundError(f"Tooltips GDB not found: {self.tooltips_gdb}")
        if not self.admin_zones_gpkg.exists():
            raise FileNotFoundError(
                f"Admin zones GPKG not found: {self.admin_zones_gpkg}"
            )

        logger.info(f"Initialized TooltipsEnricher with temp dir: {self.temp_dir}")

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
        """Load mapsheet boundaries and their RC source information."""
        try:
            mapsheets = gpd.read_file(
                self.admin_zones_gpkg, layer="mapsheets_sources_only"
            )
            logger.info(f"Loaded {len(mapsheets)} mapsheets from admin zones")
            return mapsheets
        except Exception as e:
            raise RuntimeError(f"Error loading mapsheets: {e}")

    def load_tooltips_layer(self, layer_name: str = "POLYGON_MAIN") -> gpd.GeoDataFrame:
        """Load a layer from the tooltips GDB."""
        try:
            tooltips = gpd.read_file(self.tooltips_gdb, layer=layer_name)
            # tooltips = load_gpkg_with_validation(self.tooltips_gdb, layer=layer_name), repair_geometries=True)
            tooltips["OBJECTID"] = tooltips.index  # TODO
            logger.info(f"Loaded {len(tooltips)} features from {layer_name}")
            return tooltips
        except Exception as e:
            raise RuntimeError(f"Error loading {layer_name}: {e}")

    def get_rc_source_for_mapsheet(
        self, mapsheet_nbr: int, mapsheets: gpd.GeoDataFrame
    ) -> str:
        """Get RC source (RC1/RC2) for a given mapsheet number."""
        match = mapsheets[mapsheets["MSH_MAP_NBR"] == mapsheet_nbr]
        if match.empty:
            raise ValueError(f"Mapsheet {mapsheet_nbr} not found")
        return match.iloc[0]["SOURCE_RC"]

    def clip_rc_data_by_mapsheet(
        self,
        mapsheet_nbr: int,
        rc_source: str,
        feature_classes: List[str] = None,
        clip_tolerance: float = 0.0,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Clip RC data by mapsheet boundary.

        Args:
            mapsheet_nbr: Mapsheet number to clip by
            rc_source: RC source ('RC1' or 'RC2')
            feature_classes: List of feature classes to clip

        Returns:
            Dictionary of clipped GeoDataFrames
        """
        if feature_classes is None:
            feature_classes = [
                "GC_ROCK_BODIES/GC_BEDROCK",
                "GC_ROCK_BODIES/GC_UNCO_DESPOSIT",
            ]

        # Get mapsheet geometry
        mapsheets = self.load_mapsheets_info()
        mapsheet_geom = mapsheets[mapsheets["MSH_MAP_NBR"] == mapsheet_nbr]

        if mapsheet_geom.empty:
            raise ValueError(f"Mapsheet {mapsheet_nbr} not found")

        clip_boundary = mapsheet_geom.geometry.iloc[0]
        clip_bounds = clip_boundary.bounds
        if clip_tolerance > 0.0:
            minx, miny, maxx, maxy = clip_bounds
            bbox_geom = box(minx, miny, maxx, maxy)
            clip_bounds = bbox_geom.buffer(clip_tolerance)
            logger.warning(
                f"Enlarging mapsheet {mapsheet_nbr} bound by {clip_tolerance} meters: {clip_bounds.area}"
            )

        mapsheet_geom = mapsheet_geom.set_crs("EPSG:2056", allow_override=True)

        logger.info(f"Clipping RC data for mapsheet {mapsheet_nbr} ({rc_source})")

        clipped_data = {}

        # Determine data source
        if self.sde_bridge and self.sde_bridge.is_writable:
            # Use SDE bridge for live data
            for fc in feature_classes:
                try:
                    gdf = self.sde_bridge.export_to_geodataframe(
                        fc, bbox=clip_bounds, fields=self._get_fields_for_fc(fc)
                    )
                    # Precise clipping with geometry
                    if not gdf.empty:
                        gdf = gdf[gdf.geometry.intersects(clip_boundary)]
                        gdf = gpd.clip(gdf, mapsheet_geom)

                    clipped_data[fc] = gdf
                    logger.info(f"Clipped {len(gdf)} features from {fc}")

                except Exception as e:
                    logger.error(f"Error clipping {fc}: {e}")
                    clipped_data[fc] = gpd.GeoDataFrame()

        elif rc_source in self.rc_data_sources:
            # Use FileGDB source
            rc_path = self.rc_data_sources[rc_source]
            for fc in feature_classes:
                try:
                    # Load with bbox filter for efficiency
                    layername = fc.split("/")[-1]  # Remove group prefix

                    """gdf = safe_read_filegdb(
                        rc_path,
                        layer_name=layername,
                        bbox=clip_bounds,
                    )
                    logger.warning("after loading...")"""

                    gdf = load_gpkg_with_validation(
                        rc_path,
                        repair_geometries=True,
                        layer=layername,
                        bbox=clip_bounds,
                    )
                    gdf = gdf.set_crs("EPSG:2056", allow_override=True)

                    # TODO
                    gdf.to_file(
                        Path(
                            f"/home/marco/DATA/Derivations/output/R14/source_rc_{mapsheet_nbr}.gpkg"
                        ),
                        layer=layername,
                        driver="GPKG",
                    )

                    # Precise clipping
                    if not gdf.empty:
                        # TODO  if already split not needed
                        #  gdf = gdf[gdf.geometry.intersects(clip_boundary)]
                        # gdf = gpd.clip(gdf, mapsheet_geom)
                        pass

                    clipped_data[fc] = gdf
                    logger.info(f"Clipped {len(gdf)} features from {fc}")

                except Exception as e:
                    logger.error(f"Error clipping {fc}: {e}")
                    clipped_data[fc] = gpd.GeoDataFrame()
        else:
            raise ValueError(f"No data source available for {rc_source}")

        return clipped_data

    def _get_fields_for_fc(self, feature_class: str) -> List[str]:
        """Get appropriate fields list for feature class."""
        if "BEDROCK" in feature_class.upper():
            return self.BEDROCK_FIELDS
        elif "UNCO" in feature_class.upper():
            return self.UNCO_FIELDS
        else:
            return ["UUID", "GEOLCODE"]  # Minimal fallback

    def spatial_match_features(
        self,
        tooltips_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        area_threshold: float = 0.9,  # IoU threshold
        boundary_threshold: float = 0.9,  # Boundary similarity
        centroid_tolerance: float = 1.0,  # Meters
        buffer_distance: float = 0.1,
    ) -> pd.DataFrame:
        # Pre-calculate spatial index for performance
        source_sindex = (
            source_features.sindex if hasattr(source_features, "sindex") else None
        )

        matches = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Spatial matching feature...", total=len(tooltips_features)
            )

            for idx, tooltip_feat in tooltips_features.iterrows():
                tooltip_geom = tooltip_feat.geometry
                tooltip_area = tooltip_geom.area
                progress.update(task, advance=1)

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

                best_match = None
                best_confidence = 0
                match_method = "none"

                for src_idx, src_feat in intersecting.iterrows():
                    src_geom = src_feat.geometry

                    # Method 1: Intersection over Union (more robust than simple overlap)
                    try:
                        intersection = tooltip_geom.intersection(src_geom)
                        if intersection.is_empty:
                            continue

                        union = tooltip_geom.union(src_geom)
                        iou = intersection.area / union.area

                        if iou > area_threshold and iou > best_confidence:
                            best_match = src_feat["UUID"]
                            best_confidence = iou
                            match_method = "iou"
                            continue  # High confidence match found

                    except Exception as e:
                        logger.debug(f"Error calculating IoU: {e}")

                    # Method 2: Boundary similarity (your original requirement)
                    try:
                        if tooltip_geom.geom_type in [
                            "Polygon",
                            "MultiPolygon",
                        ] and src_geom.geom_type in ["Polygon", "MultiPolygon"]:
                            # Compare boundaries with buffer tolerance
                            tooltip_boundary = tooltip_geom.boundary
                            src_boundary = src_geom.boundary

                            # Buffer boundaries for tolerance
                            buffered_tooltip_boundary = tooltip_boundary.buffer(
                                buffer_distance
                            )
                            buffered_src_boundary = src_boundary.buffer(buffer_distance)

                            # Calculate boundary overlap
                            boundary_intersection = (
                                buffered_tooltip_boundary.intersection(
                                    buffered_src_boundary
                                )
                            )
                            boundary_union = buffered_tooltip_boundary.union(
                                buffered_src_boundary
                            )

                            if boundary_union.length > 0:
                                boundary_similarity = (
                                    boundary_intersection.length / boundary_union.length
                                )

                                if (
                                    boundary_similarity > boundary_threshold
                                    and boundary_similarity > best_confidence
                                ):
                                    best_match = src_feat["UUID"]
                                    best_confidence = boundary_similarity
                                    match_method = "boundary_similarity"
                                    continue

                    except Exception as e:
                        logger.debug(f"Error calculating boundary similarity: {e}")

                    # Method 3: Hausdorff distance (shape similarity)
                    try:
                        hausdorff_dist = tooltip_geom.hausdorff_distance(src_geom)
                        # Normalize by feature size
                        max_dimension = max(tooltip_geom.length, src_geom.length)
                        if max_dimension > 0:
                            hausdorff_similarity = 1 - (hausdorff_dist / max_dimension)

                            if (
                                hausdorff_similarity > 0.8
                                and hausdorff_similarity > best_confidence
                            ):
                                best_match = src_feat["UUID"]
                                best_confidence = hausdorff_similarity
                                match_method = "hausdorff"

                    except Exception as e:
                        logger.debug(f"Error calculating Hausdorff distance: {e}")

                # Method 4: Centroid proximity (fallback)
                if best_match is None:
                    tooltip_centroid = tooltip_geom.centroid
                    for src_idx, src_feat in intersecting.iterrows():
                        src_centroid = src_feat.geometry.centroid
                        distance = tooltip_centroid.distance(src_centroid)

                        if distance <= centroid_tolerance:
                            confidence = 1 - (distance / centroid_tolerance)
                            if confidence > best_confidence:
                                best_match = src_feat["UUID"]
                                best_confidence = confidence
                                match_method = "centroid_proximity"

                if best_match:
                    matches.append(
                        {
                            "OBJECTID": tooltip_feat["OBJECTID"],
                            "UUID": best_match,
                            "match_method": match_method,
                            "confidence": round(best_confidence, 3),
                        }
                    )

        return pd.DataFrame(matches)

    def spatial_match_features_ori(
        self,
        tooltips_features: gpd.GeoDataFrame,
        source_features: gpd.GeoDataFrame,
        area_threshold: float = None,
        buffer_distance: float = None,
    ) -> pd.DataFrame:
        """
        Perform spatial matching between tooltip and source features.

        Uses a multi-step approach:
        1. Large area overlap matching
        2. Centroid-based matching for smaller features
        3. Buffered edge matching for boundary cases

        Returns:
            DataFrame with tooltip OBJECTID and matched source UUID pairs
        """
        if area_threshold is None:
            area_threshold = self.DEFAULT_AREA_THRESHOLD
        if buffer_distance is None:
            buffer_distance = self.DEFAULT_BUFFER_DISTANCE

        logger.info(
            f"Spatial matching {len(tooltips_features)} tooltip features against {len(source_features)} source features"
        )

        if source_features.empty:
            return pd.DataFrame(
                columns=["OBJECTID", "UUID", "match_method", "confidence"]
            )

        matches = []

        logger.warning(source_features.dtypes)

        # Ensure CRS compatibility
        if tooltips_features.crs != source_features.crs:
            source_features = source_features.to_crs(tooltips_features.crs)

        for idx, tooltip_feat in tooltips_features.iterrows():
            tooltip_geom = tooltip_feat.geometry
            tooltip_area = tooltip_geom.area

            # Find intersecting source features
            intersecting = source_features[
                source_features.geometry.intersects(tooltip_geom)
            ]

            if intersecting.empty:
                continue

            best_match = None
            best_confidence = 0
            match_method = "none"

            # Method 1: Area overlap ratio
            for src_idx, src_feat in intersecting.iterrows():
                try:
                    intersection = tooltip_geom.intersection(src_feat.geometry)
                    if intersection.is_empty:
                        continue

                    overlap_area = intersection.area
                    overlap_ratio = overlap_area / tooltip_area

                    if (
                        overlap_ratio > area_threshold
                        and overlap_ratio > best_confidence
                    ):
                        best_match = src_feat["UUID"]
                        best_confidence = overlap_ratio
                        match_method = "area_overlap"

                except Exception as e:
                    logger.debug(f"Error calculating overlap: {e}")
                    continue

            # Method 2: Centroid-based (if no good area match)
            if best_match is None:
                tooltip_centroid = tooltip_geom.centroid
                for src_idx, src_feat in intersecting.iterrows():
                    if src_feat.geometry.contains(tooltip_centroid):
                        # Calculate confidence based on relative areas
                        confidence = min(tooltip_area / src_feat.geometry.area, 1.0)
                        if confidence > best_confidence:
                            best_match = src_feat["UUID"]
                            best_confidence = confidence
                            match_method = "centroid_contains"

            # Method 3: Buffered matching for edge cases
            if best_match is None and buffer_distance > 0:
                buffered_tooltip = tooltip_geom.buffer(buffer_distance)
                buffered_intersecting = source_features[
                    source_features.geometry.intersects(buffered_tooltip)
                ]

                for src_idx, src_feat in buffered_intersecting.iterrows():
                    try:
                        intersection = buffered_tooltip.intersection(src_feat.geometry)
                        if not intersection.is_empty:
                            confidence = intersection.area / tooltip_area
                            if confidence > best_confidence:
                                best_match = src_feat["UUID"]
                                best_confidence = confidence
                                match_method = "buffered"
                    except Exception as e:
                        continue

            # Store match if found

            if best_match:
                matches.append(
                    {
                        "OBJECTID": tooltip_feat["OBJECTID"],  # TODO
                        "UUID": best_match,
                        "match_method": match_method,
                        "confidence": round(best_confidence, 3),
                    }
                )

        matches_df = pd.DataFrame(matches)
        logger.info(f"Found {len(matches_df)} spatial matches")

        if not matches_df.empty:
            method_counts = matches_df["match_method"].value_counts()
            logger.info(f"Match methods: {dict(method_counts)}")

        return matches_df

    def transfer_attributes(
        self,
        tooltips_features: gpd.GeoDataFrame,
        source_features: Dict[str, gpd.GeoDataFrame],
        matches: pd.DataFrame,
        target_fields: List[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Transfer attributes from source features to tooltip features.

        Args:
            tooltips_features: Target GeoDataFrame to enrich
            source_features: Dict of source GeoDataFrames (bedrock, unco)
            matches: DataFrame with OBJECTID-UUID matches
            target_fields: List of fields to transfer

        Returns:
            Enriched tooltips GeoDataFrame
        """
        if target_fields is None:
            target_fields = [
                "GEOLCODE",
                "GLAC_TYP",
                "CHRONO_T",
                "CHRONO_B",
                "gmu_code",
                "tecto",
                "tecto_code",
            ]

        # Create enriched copy
        enriched = tooltips_features.copy()

        # Initialize new columns
        for field in target_fields:
            enriched[field] = None

        # Add source tracking fields
        enriched["SOURCE_UUID"] = None
        enriched["MATCH_METHOD"] = None
        enriched["MATCH_CONFIDENCE"] = 0.0
        enriched["MATCH_LAYER"] = None

        # Combine all source features
        all_sources = pd.concat(
            [df for df in source_features.values() if not df.empty], ignore_index=True
        )

        if all_sources.empty:
            logger.warning("No source features available for attribute transfer")
            return enriched

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
            source_feat = all_sources[all_sources["UUID"] == uuid]
            if source_feat.empty:
                continue
            source_feat = source_feat.iloc[0]

            # Transfer attributes
            for field in target_fields:
                if field in source_feat.index and pd.notna(source_feat[field]):
                    enriched.loc[tooltip_idx, field] = source_feat[field]

            # Set tracking fields
            enriched.loc[tooltip_idx, "SOURCE_UUID"] = uuid
            enriched.loc[tooltip_idx, "MATCH_METHOD"] = match_row["match_method"]
            enriched.loc[tooltip_idx, "MATCH_CONFIDENCE"] = match_row["confidence"]
            layer = match_row["source_fc"].split("/")[-1]
            enriched.loc[tooltip_idx, "MATCH_LAYER"] = layer

            transferred_count += 1

        logger.info(
            f"Transferred attributes to {transferred_count}/{len(tooltips_features)} features"
        )
        # logger.debug(source_feat.index)
        return enriched

    def compute_symbol_classification(
        self,
        enriched_features: gpd.GeoDataFrame,
        classification_rules: Optional[Dict[str, Any]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Compute SYMBOL field based on classification rules.

        Args:
            enriched_features: GeoDataFrame with transferred attributes
            classification_rules: Custom classification rules (optional)

        Returns:
            GeoDataFrame with computed SYMBOL field
        """
        result = enriched_features.copy()

        # Default classification: use GEOLCODE as base
        if classification_rules is None:
            if "GEOLCODE" in result.columns:
                result["SYMBOL"] = result["GEOLCODE"].fillna("UNKNOWN")
            else:
                logger.warning(f"No column `GEOLCODE` in result")
        else:
            # Apply custom rules (implement based on your specific needs)
            result["SYMBOL"] = self._apply_classification_rules(
                result, classification_rules
            )

        # Log classification stats
        if "SYMBOL" in result.columns:
            symbol_counts = result["SYMBOL"].value_counts()
            logger.info(f"Symbol classification: {len(symbol_counts)} unique symbols")
            logger.debug(f"Top symbols: {dict(symbol_counts.head())}")

        return result

    def _apply_classification_rules(
        self, features: gpd.GeoDataFrame, rules: Dict[str, Any]
    ) -> pd.Series:
        """Apply custom classification rules to compute symbols."""
        # Placeholder for complex classification logic
        # Implement based on your specific requirements
        # print(features.columns)
        return features["GEOLCODE"].fillna("UNKNOWN")

    def enrich_polygon_main(
        self,
        mapsheet_numbers: Optional[List[int]] = None,
        target_fields: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        area_threshold: float = None,
        save_intermediate: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Complete publish pipeline for POLYGON_MAIN layer.

        Args:
            mapsheet_numbers: Specific mapsheets to process (None = all)
            area_threshold: Spatial matching threshold
            save_intermediate: Save intermediate results for debugging

        Returns:
            Enriched POLYGON_MAIN GeoDataFrame
        """
        logger.info("Starting POLYGON_MAIN publish pipeline")

        # Load base data
        tooltips_main = self.load_tooltips_layer("POLYGON_MAIN")
        mapsheets = self.load_mapsheets_info()

        # Determine mapsheets to process
        if mapsheet_numbers is None:
            unique_mapsheets = mapsheets["MSH_MAP_NBR"].unique()
        else:
            unique_mapsheets = mapsheet_numbers

        logger.info(f"Processing {len(unique_mapsheets)} mapsheets")

        all_enriched = []

        for mapsheet_nbr in unique_mapsheets:
            logger.info(f"Processing mapsheet {mapsheet_nbr}")

            try:
                # Get RC source for this mapsheet
                rc_source = self.get_rc_source_for_mapsheet(mapsheet_nbr, mapsheets)

                # Get tooltip features for this mapsheet (if you have mapsheet info)
                # For now, process all features - you might need to filter by geometry

                # Clip RC data
                clipped_rc = self.clip_rc_data_by_mapsheet(
                    mapsheet_nbr,
                    rc_source,
                    clip_tolerance=0.0,  # meters, feature_classes=layers
                )

                # Skip if no RC data
                if all(df.empty for df in clipped_rc.values()):
                    logger.warning(f"No RC data for mapsheet {mapsheet_nbr}")
                    continue

                # TODO
                if save_intermediate:
                    for layername, gdf in clipped_rc.items():
                        layer = layername.split("/")[-1].lower()
                        gdf.to_file(
                            Path(
                                f"/home/marco/DATA/Derivations/output/R14/clipped_rc_{mapsheet_nbr}.gpkg"
                            ),
                            layer=layer,
                            driver="GPKG",
                        )
                        gdf.to_file(
                            Path(
                                f"/home/marco/DATA/Derivations/output/R14/clipped_rc_{mapsheet_nbr}.gpkg"
                            ),
                            layer=layer,
                            driver="GPKG",
                        )
                    """for layername, gdf in rc_source.items():
                        layer = layername.split('/')[-1].lower()
                        gdf.to_file(
                            Path(f"/home/marco/DATA/Derivations/output/R14/source_rc_{mapsheet_nbr}.gpkg"),
                            layer=layer, driver='GPKG')
                        gdf.to_file(
                            Path(f"/home/marco/DATA/Derivations/output/R14/source_rc_{mapsheet_nbr}.gpkg"),
                            layer=layer,
                            driver='GPKG')"""

                # Get mapsheet boundary for filtering tooltips
                logger.info("Get mapsheet boundary for filtering tooltips...")
                mapsheet_boundary = mapsheets[
                    mapsheets["MSH_MAP_NBR"] == mapsheet_nbr
                ].geometry.iloc[0]

                # Filter tooltip features that intersect this mapsheet
                logger.info("Filter tooltip features that intersect this mapsheet...")
                tooltips_subset = tooltips_main[
                    tooltips_main.geometry.intersects(mapsheet_boundary)
                ].copy()

                if tooltips_subset.empty:
                    logger.warning(f"No tooltip features for mapsheet {mapsheet_nbr}")
                    continue

                logger.warning(
                    f"Processing {len(tooltips_subset)} tooltip features"
                )  # OK

                # Spatial matching
                all_matches = []

                for fc_name, source_gdf in clipped_rc.items():
                    logger.info(f"Layer: {fc_name}")
                    if source_gdf.empty:
                        continue

                    matches = self.spatial_match_features(  # TODO: bogus
                        tooltips_subset,
                        source_gdf,
                        area_threshold=area_threshold,
                    )

                    logger.info(f"Got {len(matches)}")

                    if not matches.empty:
                        matches["source_fc"] = fc_name
                        all_matches.append(matches)

                if not all_matches:
                    logger.warning(f"No spatial matches for mapsheet {mapsheet_nbr}")
                    continue

                logger.info("Concatening...")
                combined_matches = pd.concat(all_matches, ignore_index=True)

                # Remove duplicate matches (keep best confidence)
                combined_matches = combined_matches.sort_values(
                    "confidence", ascending=False
                )
                combined_matches = combined_matches.drop_duplicates(
                    "OBJECTID", keep="first"
                )

                # Transfer attributes
                enriched_subset = self.transfer_attributes(
                    tooltips_subset,
                    clipped_rc,
                    combined_matches,
                    target_fields,
                )

                # Compute symbols
                enriched_subset = self.compute_symbol_classification(enriched_subset)

                # Add mapsheet tracking
                enriched_subset["MAPSHEET_NBR"] = mapsheet_nbr
                enriched_subset["RC_SOURCE"] = rc_source

                all_enriched.append(enriched_subset)

                # Save intermediate results if requested
                if save_intermediate:
                    temp_path = self.temp_dir / f"mapsheet_{mapsheet_nbr}_enriched.gpkg"
                    enriched_subset.to_file(temp_path, driver="GPKG")
                    logger.info(f"Saved intermediate result: {temp_path}")

                logger.info(
                    f"Completed mapsheet {mapsheet_nbr}: {len(enriched_subset)} features processed"
                )

            except Exception as e:
                logger.error(f"Error processing mapsheet {mapsheet_nbr}: {e}")
                logger.error(f"Full error details: {traceback.format_exc()}")

                continue

        if not all_enriched:
            raise RuntimeError("No mapsheets were successfully processed")

        # Combine all results
        final_result = pd.concat(all_enriched, ignore_index=True)
        logger.info(
            f"Enrichment complete: {len(final_result)} total features processed"
        )

        return final_result

    def save_enriched_data(
        self,
        enriched_gdf: gpd.GeoDataFrame,
        output_path: Union[str, Path],
        layer_name: str = "POLYGON_MAIN_ENRICHED",
    ) -> Path:
        """Save enriched data to file."""
        output_path = Path(output_path)

        # Determine format from extension
        if output_path.suffix.lower() == ".gpkg":
            enriched_gdf.to_file(output_path, layer=layer_name, driver="GPKG")
        else:
            enriched_gdf.to_file(output_path)

        logger.info(f"Saved enriched data to {output_path}")
        return output_path


def enrich_tooltips_from_config(config_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Enrich tooltips using configuration file.

    Config format (YAML):
    ```yaml
    tooltips_gdb: "/path/to/geocover_tooltips.gdb"
    admin_zones_gpkg: "/path/to/administrative_zones.gpkg"
    rc_sources:
      RC1: "/path/to/2016-12-31.gdb"
      RC2: "/path/to/2030-12-31.gdb"
    mapsheets: [55, 25, 173]  # optional, all if not specified
    area_threshold: 0.7
    output_path: "/path/to/enriched_output.gpkg"
    ```
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with TooltipsEnricher(
        tooltips_gdb=config["tooltips_gdb"],
        admin_zones_gpkg=config["admin_zones_gpkg"],
        rc_data_sources=config.get("rc_sources", {}),
    ) as enricher:
        enriched = enricher.enrich_polygon_main(
            mapsheet_numbers=config.get("mapsheets"),
            area_threshold=config.get("area_threshold", 0.7),
        )

        if "output_path" in config:
            enricher.save_enriched_data(enriched, config["output_path"])

        return enriched
