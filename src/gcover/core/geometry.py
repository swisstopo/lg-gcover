import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from shapely import is_empty, is_valid, make_valid
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.validation import explain_validity


def safe_read_filegdb(
    gdb_path: Union[str, Path],
    layer_name: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    chunk_size: int = 5000,
) -> gpd.GeoDataFrame:
    """
    Safely read FileGDB with multiple fallback strategies to avoid segmentation faults.

    Args:
        gdb_path: Path to FileGDB
        layer_name: Layer name to read
        bbox: Bounding box filter (minx, miny, maxx, maxy)
        chunk_size: Chunk size for chunked reading fallback

    Returns:
        GeoDataFrame with loaded data
    """
    gdb_path = Path(gdb_path)

    # Strategy 1: Direct read with bbox (fastest, but might cause segfault)
    if bbox:
        try:
            logger.debug(f"Attempting direct bbox read for {layer_name}")
            gdf = gpd.read_file(str(gdb_path), layer=layer_name, bbox=bbox)
            logger.debug(f"Direct bbox read successful: {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.warning(f"Direct bbox read failed: {type(e).__name__}: {e}")

    # Strategy 2: Read without bbox, then filter (safer)
    try:
        logger.debug(f"Attempting read without bbox for {layer_name}")
        gdf = gpd.read_file(str(gdb_path), layer=layer_name)
        logger.debug(f"Read {len(gdf)} features without bbox")

        # Apply bbox filter after reading
        if bbox and not gdf.empty:
            minx, miny, maxx, maxy = bbox
            gdf = gdf.cx[minx:maxx, miny:maxy]
            logger.debug(f"Filtered to {len(gdf)} features with bbox")

        return gdf

    except Exception as e:
        logger.warning(f"Standard read failed: {type(e).__name__}: {e}")

    # Strategy 3: Chunked reading using fiona (slowest, but most reliable)
    try:
        logger.debug(f"Attempting chunked read for {layer_name}")
        import fiona

        chunks = []
        with fiona.open(str(gdb_path), layer=layer_name) as src:
            total_features = len(src)
            logger.debug(f"Total features in {layer_name}: {total_features}")

            # Read in chunks
            for i in range(0, total_features, chunk_size):
                chunk_features = []
                for j, feature in enumerate(src):
                    if j < i:
                        continue
                    if j >= i + chunk_size:
                        break

                    # Basic bbox filtering at feature level
                    if bbox:
                        geom = feature["geometry"]
                        if geom and "coordinates" in geom:
                            # Simple bounds check (not perfect but fast)
                            coords = geom["coordinates"]
                            if isinstance(coords, list) and len(coords) > 0:
                                # This is a simplified check - just include feature if unsure
                                pass

                    chunk_features.append(feature)

                if chunk_features:
                    try:
                        chunk_gdf = gpd.GeoDataFrame.from_features(
                            chunk_features, crs=src.crs
                        )

                        # Apply bbox filter to chunk
                        if bbox and not chunk_gdf.empty:
                            minx, miny, maxx, maxy = bbox
                            chunk_gdf = chunk_gdf.cx[minx:maxx, miny:maxy]

                        if not chunk_gdf.empty:
                            chunks.append(chunk_gdf)
                            logger.debug(
                                f"Processed chunk {i // chunk_size + 1}: {len(chunk_gdf)} features"
                            )
                    except Exception as chunk_e:
                        logger.warning(
                            f"Error processing chunk {i // chunk_size + 1}: {chunk_e}"
                        )
                        continue

        if chunks:
            result = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True))
            logger.debug(f"Chunked read successful: {len(result)} total features")
            return result
        else:
            logger.warning("No chunks successfully processed")
            return gpd.GeoDataFrame()

    except Exception as e:
        logger.error(f"Chunked read failed: {type(e).__name__}: {e}")

    # Strategy 4: Return empty GeoDataFrame as last resort
    logger.error(f"All read strategies failed for {layer_name} in {gdb_path}")
    return gpd.GeoDataFrame()


def repair_self_intersections_with_quality_control(
    gdf: gpd.GeoDataFrame,
    max_area_change: float = 0.1,  # 10% max area change
    max_iou_threshold: float = 0.8,  # Minimum IoU after repair
    buffer_tolerance: float = 0.001,  # Buffer tolerance for initial repair attempt
    discard_unreparable: bool = True,
    log_details: bool = True,
) -> gpd.GeoDataFrame:
    """
    Repair self-intersecting rings with quality control to discard overly changed geometries.

    Args:
        gdf: Input GeoDataFrame.
        max_area_change: Maximum allowed area change ratio (0.1 = 10%). Defaults to 0.1.
        max_iou_threshold: Minimum Intersection over Union (IoU) after repair. Defaults to 0.8.
        buffer_tolerance: Buffer distance for initial repair attempt (often fixes self-intersections).
            Defaults to 0.001.
        discard_unreparable: Whether to discard geometries that fail quality checks. Defaults to True.
        log_details: Whether to log detailed repair information. Defaults to True.

    Returns:
        A gpd.GeoDataFrame with repaired geometries (or discarded if quality check fails).
    """

    def calculate_geometry_similarity(
        geom_original, geom_repaired
    ) -> Tuple[float, float]:
        """Calculate area change ratio and IoU between original and repaired geometry."""
        try:
            if geom_original.is_empty or geom_repaired.is_empty:
                return 0.0, 0.0

            area_original = geom_original.area
            area_repaired = geom_repaired.area

            # Area change ratio (absolute value)
            if area_original > 0:
                area_change = abs(area_repaired - area_original) / area_original
            else:
                area_change = 1.0  # Original was empty or zero-area

            # Intersection over Union
            intersection = geom_original.intersection(geom_repaired)
            union = geom_original.union(geom_repaired)

            if union.area > 0:
                iou = intersection.area / union.area
            else:
                iou = 0.0

            return area_change, iou

        except Exception as e:
            logger.debug(f"Error calculating geometry similarity: {e}")
            return 1.0, 0.0  # Worst case values

    def extract_main_polygon(geometry):
        """Extract the main polygon from GeometryCollection or MultiPolygon."""
        if geometry.is_empty:
            return None

        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry

        if isinstance(geometry, GeometryCollection):
            # Find the largest polygon in the collection
            polygons = []
            for geom in geometry.geoms:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    polygons.append(geom)

            if polygons:
                # Return the largest polygon by area
                return max(polygons, key=lambda x: x.area)
            else:
                # No polygons found, try to get any valid geometry
                valid_geoms = [geom for geom in geometry.geoms if not geom.is_empty]
                if valid_geoms:
                    return max(valid_geoms, key=lambda x: x.area)

        return None

    def repair_single_geometry(geom_original):
        """Repair a single geometry with quality control."""
        if geom_original.is_empty or geom_original.is_valid:
            return geom_original, "valid", 1.0, 1.0  # No repair needed

        validity_info = explain_validity(geom_original)

        # Only handle self-intersections specifically
        if "Self-intersection" not in validity_info:
            return geom_original, "not_self_intersection", 1.0, 1.0

        repair_attempts = []

        # Strategy 1: Try buffer(0) method (often fixes self-intersections)
        try:
            geom_repaired = geom_original.buffer(0, resolution=0)
            if is_valid(geom_repaired) and not is_empty(geom_repaired):
                area_change, iou = calculate_geometry_similarity(
                    geom_original, geom_repaired
                )
                repair_attempts.append(("buffer", geom_repaired, area_change, iou))
        except Exception as e:
            logger.debug(f"Buffer repair failed: {e}")

        # Strategy 2: Try make_valid()
        try:
            geom_repaired = make_valid(geom_original)
            if is_valid(geom_repaired) and not is_empty(geom_repaired):
                area_change, iou = calculate_geometry_similarity(
                    geom_original, geom_repaired
                )
                repair_attempts.append(("make_valid", geom_repaired, area_change, iou))
        except Exception as e:
            logger.debug(f"Make_valid repair failed: {e}")

        # Strategy 3: Try buffer with small tolerance
        try:
            geom_repaired = geom_original.buffer(buffer_tolerance, resolution=1).buffer(
                -buffer_tolerance, resolution=1
            )
            if is_valid(geom_repaired) and not is_empty(geom_repaired):
                area_change, iou = calculate_geometry_similarity(
                    geom_original, geom_repaired
                )
                repair_attempts.append(
                    ("buffer_tolerance", geom_repaired, area_change, iou)
                )
        except Exception as e:
            logger.debug(f"Buffer tolerance repair failed: {e}")

        # Evaluate repair attempts
        if repair_attempts:
            # Filter attempts that pass quality thresholds
            valid_attempts = [
                (method, geom, area_change, iou)
                for method, geom, area_change, iou in repair_attempts
                if area_change <= max_area_change and iou >= max_iou_threshold
            ]

            if valid_attempts:
                # Choose the best attempt (highest IoU)
                best_attempt = max(valid_attempts, key=lambda x: x[3])  # x[3] is IoU
                method, geom_repaired, area_change, iou = best_attempt

                # Extract main polygon if it's a GeometryCollection
                if isinstance(geom_repaired, GeometryCollection):
                    main_geom = extract_main_polygon(geom_repaired)
                    if main_geom and is_valid(main_geom) and not main_geom.is_empty:
                        geom_repaired = main_geom
                        # Recalculate similarity with extracted geometry
                        area_change, iou = calculate_geometry_similarity(
                            geom_original, geom_repaired
                        )

                return geom_repaired, f"repaired_{method}", area_change, iou

            # If no attempts pass quality threshold but we have repair attempts
            best_attempt = max(
                repair_attempts, key=lambda x: x[3]
            )  # Best IoU regardless
            method, geom_repaired, area_change, iou = best_attempt

            if not discard_unreparable:
                # Extract main polygon if needed
                if isinstance(geom_repaired, GeometryCollection):
                    main_geom = extract_main_polygon(geom_repaired)
                    if main_geom and is_valid(main_geom) and not main_geom.is_empty:
                        geom_repaired = main_geom

                return geom_repaired, f"repaired_{method}_low_quality", area_change, iou

        # All repair attempts failed or no valid repairs
        return geom_original, "unreparable", 0.0, 0.0

    if gdf.empty:
        logger.warning("Empty GeoDataFrame provided")
        return gdf

    gdf_clean = gdf.copy()
    repair_stats = {
        "total_processed": 0,
        "repaired": 0,
        "discarded": 0,
        "unchanged": 0,
        "repair_methods": {},
    }

    indices_to_drop = []

    for idx, geom_original in gdf_clean.geometry.items():
        repair_stats["total_processed"] += 1

        if geom_original.is_empty:
            indices_to_drop.append(idx)
            repair_stats["discarded"] += 1
            continue

        if is_valid(geom_original):
            repair_stats["unchanged"] += 1
            continue

        validity_info = explain_validity(geom_original)

        # Only process self-intersections
        if "Self-intersection" in validity_info:
            geom_repaired, repair_status, area_change, iou = repair_single_geometry(
                geom_original
            )

            if repair_status.startswith("repaired"):
                gdf_clean.at[idx, "geometry"] = geom_repaired
                repair_stats["repaired"] += 1
                repair_stats["repair_methods"][repair_status] = (
                    repair_stats["repair_methods"].get(repair_status, 0) + 1
                )

                if log_details:
                    logger.info(
                        f"Repaired self-intersection at index {idx}: "
                        f"method={repair_status}, area_change={area_change:.3f}, IoU={iou:.3f}"
                    )

            elif repair_status == "unreparable" and discard_unreparable:
                indices_to_drop.append(idx)
                repair_stats["discarded"] += 1
                if log_details:
                    logger.warning(
                        f"Discarded unreparable self-intersection at index {idx}"
                    )
            else:
                repair_stats["unchanged"] += 1
        else:
            repair_stats["unchanged"] += 1

    # Remove discarded geometries
    if indices_to_drop:
        gdf_clean = gdf_clean.drop(indices_to_drop)

    # Log summary
    logger.info("Self-intersection repair summary:")
    logger.info(f"  - Total geometries processed: {repair_stats['total_processed']}")
    logger.info(f"  - Successfully repaired: {repair_stats['repaired']}")
    logger.info(f"  - Discarded: {repair_stats['discarded']}")
    logger.info(f"  - Unchanged: {repair_stats['unchanged']}")

    if repair_stats["repair_methods"]:
        logger.info("  - Repair methods:")
        for method, count in repair_stats["repair_methods"].items():
            logger.info(f"    - {method}: {count}")

    return gdf_clean


def validate_and_repair_geometries(
    gdf: gpd.GeoDataFrame,
    repair: bool = True,
    remove_empty: bool = True,
    log_details: bool = True,
) -> gpd.GeoDataFrame:
    """
    Validate geometries in a GeoDataFrame and optionally repair invalid ones.

    Parameters:

      gdf : gpd.GeoDataFrame
        Input GeoDataFrame
      repair : bool, default True
        Whether to attempt to repair invalid geometries
      remove_empty : bool, default True
        Whether to remove empty geometries
      log_details : bool, default True
        Whether to log details about invalid geometries

    Returns:
      gpd.GeoDataFrame with validated (and optionally repaired) geometries
    """

    if gdf.empty:
        logger.warning("Empty GeoDataFrame provided")
        return gdf

    original_count = len(gdf)
    invalid_indices = []
    empty_indices = []

    # Check each geometry
    for idx, geom in gdf.geometry.items():
        if is_empty(geom):
            empty_indices.append(idx)
            if log_details:
                logger.warning(f"Empty geometry at index {idx}")
            continue

        if not is_valid(geom):
            invalid_indices.append(idx)
            if log_details:
                validity_info = explain_validity(geom)
                logger.warning(f"Invalid geometry at index {idx}: {validity_info}")

    # Log summary
    logger.info(f"Geometry validation summary:")
    logger.info(f"  - Total geometries: {original_count}")
    logger.info(f"  - Invalid geometries: {len(invalid_indices)}")
    logger.info(f"  - Empty geometries: {len(empty_indices)}")

    if repair:
        # Create a copy to avoid modifying original
        gdf_clean = gdf.copy()

        # Repair invalid geometries
        if invalid_indices:
            logger.info("Attempting to repair invalid geometries...")
            for idx in invalid_indices:
                original_geom = gdf_clean.loc[idx, "geometry"]
                try:
                    repaired_geom = make_valid(original_geom)
                    if is_valid(repaired_geom) and not is_empty(repaired_geom):
                        gdf_clean.loc[idx, "geometry"] = repaired_geom
                        logger.info(f"Successfully repaired geometry at index {idx}")
                    else:
                        logger.warning(f"Repair failed for geometry at index {idx}")
                except Exception as e:
                    logger.error(f"Error repairing geometry at index {idx}: {e}")

        # Remove empty geometries if requested
        if remove_empty and empty_indices:
            gdf_clean = gdf_clean.drop(empty_indices)
            logger.info(f"Removed {len(empty_indices)} empty geometries")

        # Final validation
        final_invalid = [
            idx
            for idx, geom in gdf_clean.geometry.items()
            if not is_valid(geom) or is_empty(geom)
        ]

        if final_invalid:
            logger.warning(
                f"Still {len(final_invalid)} invalid/empty geometries after repair"
            )
        else:
            logger.info("All geometries are now valid")

        return gdf_clean

    else:
        # Just remove problematic geometries if not repairing
        indices_to_remove = (
            invalid_indices + empty_indices if remove_empty else invalid_indices
        )
        if indices_to_remove:
            gdf_clean = gdf.drop(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} problematic geometries")
            return gdf_clean
        else:
            return gdf


def geometry_health_check(gdf: gpd.GeoDataFrame) -> dict:
    """
    Perform a quick health check on geometries and return statistics.

    Returns:
        dict with geometry health statistics
    """
    if gdf.empty:
        return {"status": "empty", "message": "GeoDataFrame is empty"}

    stats = {
        "total_geometries": len(gdf),
        "valid_geometries": 0,
        "invalid_geometries": 0,
        "empty_geometries": 0,
        "geometry_types": {},
        "invalid_indices": [],
        "empty_indices": [],
    }

    for idx, geom in gdf.geometry.items():
        geom_type = geom.geom_type if not geom.is_empty else "Empty"
        stats["geometry_types"][geom_type] = (
            stats["geometry_types"].get(geom_type, 0) + 1
        )

        if geom.is_empty:
            stats["empty_geometries"] += 1
            stats["empty_indices"].append(idx)
        elif geom.is_valid:
            stats["valid_geometries"] += 1
        else:
            stats["invalid_geometries"] += 1
            stats["invalid_indices"].append(idx)

    stats["valid_percentage"] = (
        stats["valid_geometries"] / stats["total_geometries"]
    ) * 100

    return stats


def load_gpkg_with_validation(
    gpkg_path: str,
    layer: str = None,
    repair_geometries: bool = True,
    remove_invalid: bool = True,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Load a GeoPackage with automatic geometry validation and cleaning.

    Parameters:
        gpkg_path : str
          Path to the GeoPackage file
        layer : str, optional
           Layer name to load (if None, loads first layer)
        repair_geometries : bool, default True
          Whether to attempt to repair invalid geometries
        remove_invalid : bool, default True
            Whether to remove geometries that cannot be repaired
        **kwargs : additional arguments to pass to gpd.read_file()

    Returns:
        gpd.GeoDataFrame with validated geometries
    """

    try:
        # Load the GeoPackage
        if layer:
            gdf = gpd.read_file(gpkg_path, layer=layer, **kwargs)
        else:
            gdf = gpd.read_file(gpkg_path, **kwargs)

        logger.info(f"Loaded GeoPackage: {gpkg_path}")
        logger.info(f"Initial feature count: {len(gdf)}")

        if gdf.empty:
            loggerwarning("Loaded an empty GeoDataFrame")
            return gdf

        # Validate and repair geometries
        gdf_clean = validate_and_repair_geometries(
            gdf, repair=repair_geometries, remove_empty=remove_invalid, log_details=True
        )

        final_count = len(gdf_clean)
        removed_count = len(gdf) - final_count

        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} invalid geometries. Final count: {final_count}"
            )
        else:
            logger.info("No geometries removed during validation")

        return gdf_clean

    except Exception as e:
        logger.error(f"Error loading GeoPackage {gpkg_path}: {e}")
        raise


def extract_valid_geometries(geometry, target_types=None, min_area=1e-10):
    """
    Extract valid geometries from any geometry type including GeometryCollection.

    Parameters
    ----------
    geometry : shapely.geometry
        Input geometry (any type)
    target_types : list, optional
        List of geometry types to extract
    min_area : float
        Minimum area for polygon geometries

    Returns
    -------
    list
        List of extracted geometries
    """
    if target_types is None:
        target_types = ["Polygon", "MultiPolygon"]

    valid_geoms = []

    if geometry.is_empty:
        return valid_geoms

    # Handle GeometryCollection
    if geometry.geom_type == "GeometryCollection":
        for geom in geometry.geoms:
            if geom.is_empty:
                continue
            if geom.geom_type in target_types:
                if geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        if not poly.is_empty and poly.area > min_area:
                            valid_geoms.append(poly)
                elif geom.geom_type == "Polygon":
                    if geom.area > min_area:
                        valid_geoms.append(geom)
                else:
                    valid_geoms.append(geom)

    # Handle MultiPolygon
    elif geometry.geom_type == "MultiPolygon":
        for poly in geometry.geoms:
            if not poly.is_empty and poly.area > min_area:
                valid_geoms.append(poly)

    # Handle simple Polygon
    elif geometry.geom_type == "Polygon":
        if geometry.area > min_area:
            valid_geoms.append(geometry)

    # Handle other geometry types if included in target_types
    elif geometry.geom_type in target_types:
        valid_geoms.append(geometry)

    return valid_geoms


def split_features_by_mapsheets(
    input_gdf: gpd.GeoDataFrame,
    mapsheets_gdf: gpd.GeoDataFrame,
    output_crs: Optional[str] = None,
    keep_attributes: bool = True,
    progress_bar: bool = True,
    area_threshold: float = 1e-10,
) -> gpd.GeoDataFrame:
    """
    Split features from a GeoDataFrame along mapsheet boundaries.

    Parameters
    ----------
    input_gdf : gpd.GeoDataFrame
        Input features to split (points, lines, polygons)
    mapsheets_gdf : gpd.GeoDataFrame
        Mapsheets GeoDataFrame
    output_crs : str, optional
        CRS to use for output (will reproject if different from input)
    keep_attributes : bool
        Whether to keep original attributes on split features
    progress_bar : bool
        Whether to show progress bar
    area_threshold : float
        Minimum area for polygon features to be kept

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with features split along mapsheet boundaries
    """

    console = Console()

    # Suppress warnings during processing
    warnings.filterwarnings("ignore", category=UserWarning)

    logger.debug("Starting feature splitting process...")
    logger.debug(f"Input features: {len(input_gdf)}")
    logger.debug(f"Mapsheets: {len(mapsheets_gdf)}")

    # Ensure CRS compatibility
    if input_gdf.crs != mapsheets_gdf.crs:
        logger.warning("Reprojecting input features to match mapsheets CRS")
        input_gdf = input_gdf.to_crs(mapsheets_gdf.crs)

    if output_crs and output_crs != mapsheets_gdf.crs:
        logger.warning(f"Output will be reprojected to: {output_crs}")

    # Prepare result container
    result_features = []
    result_attributes = []

    # Get geometry type information
    if len(input_gdf) > 0:
        input_type = input_gdf.geometry.type.iloc[0]
        logger.debug(f"Input geometry type: {input_type}")

    if len(mapsheets_gdf) > 0:
        mapsheet_type = mapsheets_gdf.geometry.type.iloc[0]
        logger.debug(f"Mapsheet geometry type: {mapsheet_type}")

    # Create progress bar
    total_features = len(input_gdf)

    def process_feature(feature_idx, feature_row):
        """Process a single feature and split it across mapsheets"""
        geometry = feature_row.geometry
        attributes = feature_row.drop("geometry").to_dict() if keep_attributes else {}

        # Skip null geometries
        if geometry is None or geometry.is_empty:
            return

        # Find mapsheets that intersect with this feature
        intersecting_mapsheets = mapsheets_gdf[mapsheets_gdf.intersects(geometry)]

        if len(intersecting_mapsheets) == 0:
            # Feature doesn't intersect any mapsheet - keep original
            result_features.append(geometry)
            if keep_attributes:
                result_attributes.append(attributes)
            return

        # Check if feature is entirely within one mapsheet
        entirely_contained = False

        for _, mapsheet_row in intersecting_mapsheets.iterrows():
            if mapsheet_row.geometry.contains(geometry):
                entirely_contained = True
                break

        if entirely_contained:
            # Add the entire feature once
            result_features.append(geometry)
            if keep_attributes:
                result_attributes.append(attributes)
        else:
            # Feature spans multiple mapsheets - split it
            for _, mapsheet_row in intersecting_mapsheets.iterrows():
                mapsheet_geom = mapsheet_row.geometry

                try:
                    intersection = geometry.intersection(mapsheet_geom)

                    if not intersection.is_empty:
                        # Extract valid geometries based on input type
                        if geometry.geom_type in ["Point", "MultiPoint"]:
                            valid_parts = extract_valid_geometries(
                                intersection, target_types=["Point", "MultiPoint"]
                            )
                        elif geometry.geom_type in ["LineString", "MultiLineString"]:
                            valid_parts = extract_valid_geometries(
                                intersection,
                                target_types=["LineString", "MultiLineString"],
                            )
                        else:  # Polygons
                            valid_parts = extract_valid_geometries(
                                intersection,
                                target_types=["Polygon", "MultiPolygon"],
                                min_area=area_threshold,
                            )

                        # Add the valid parts
                        for part in valid_parts:
                            result_features.append(part)
                            if keep_attributes:
                                result_attributes.append(attributes.copy())

                except Exception as e:
                    logger.warning(f"Error processing feature {feature_idx}: {e}")
                    continue

    # Process features with progress bar
    if progress_bar and total_features > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Splitting features...", total=total_features)

            for idx, row in input_gdf.iterrows():
                process_feature(idx, row)
                progress.update(task, advance=1)
    else:
        # Process without progress bar
        for idx, row in input_gdf.iterrows():
            process_feature(idx, row)

    # Create result GeoDataFrame
    logger.debug(f"Processed {len(result_features)} output features")

    if len(result_features) == 0:
        logger.warning("No features were generated")
        return gpd.GeoDataFrame(geometry=[], crs=mapsheets_gdf.crs)

    # Create result DataFrame
    if keep_attributes and result_attributes:
        result_gdf = gpd.GeoDataFrame(
            result_attributes, geometry=result_features, crs=mapsheets_gdf.crs
        )
    else:
        result_gdf = gpd.GeoDataFrame(geometry=result_features, crs=mapsheets_gdf.crs)

    # Reproject if requested
    if output_crs and output_crs != result_gdf.crs:
        logger.debug("Reprojecting result to output CRS")
        result_gdf = result_gdf.to_crs(output_crs)

    # Reset index
    result_gdf = result_gdf.reset_index(drop=True)

    logger.debug("Feature splitting completed successfully!")

    return result_gdf
