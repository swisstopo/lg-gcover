import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import split, unary_union
import uuid
import numpy as np
from pathlib import Path
from loguru import logger


def load_geodatabase_layers(
    gdb_path, linear_layer="GC_LINEAR_OBJECTS", unco_layer="GC_UNCO_DESPOSIT", bbox=None
):
    """
    Load the required layers from the FileGDB

    Parameters:
    - gdb_path: Path to the geodatabase
    - linear_layer: Name of the linear objects layer
    - unco_layer: Name of the unco deposit layer
    - bbox: Bounding box as (minx, miny, maxx, maxy) in LV95 coordinates
    """
    try:
        # Load tectonic lines
        logger.info(f"Loading {linear_layer} from {gdb_path}")
        if bbox:
            logger.info(f"Applying BBOX filter: {bbox}")
            tectonic_lines = gpd.read_file(gdb_path, layer=linear_layer, bbox=bbox)
        else:
            tectonic_lines = gpd.read_file(gdb_path, layer=linear_layer)

        # Load unco deposit polygons
        logger.info(f"Loading {unco_layer} from {gdb_path}")
        if bbox:
            unco_polygons = gpd.read_file(gdb_path, layer=unco_layer, bbox=bbox)
        else:
            unco_polygons = gpd.read_file(gdb_path, layer=unco_layer)

        logger.success(
            f"Loaded {len(tectonic_lines)} tectonic lines and {len(unco_polygons)} unco polygons"
        )

        return tectonic_lines, unco_polygons

    except Exception as e:
        logger.error(f"Error loading layers: {e}")
        return None, None


def create_test_bbox_alps(size_km=10):
    """
    Create a test bounding box in the Swiss Alps (LV95 coordinates)

    Parameters:
    - size_km: Size of the square bbox in kilometers (default 10km x 10km = 100 sq km)

    Returns:
    - bbox: (minx, miny, maxx, maxy) tuple
    """
    # Example coordinates for a location in the Swiss Alps
    # These are approximate coordinates near Zermatt/Matterhorn area
    center_x = 2_621_000  # LV95 Easting
    center_y = 1_099_000  # LV95 Northing

    # Convert km to meters (LV95 uses meters)
    half_size = (size_km * 1000) / 2

    bbox = (
        center_x - half_size,  # minx
        center_y - half_size,  # miny
        center_x + half_size,  # maxx
        center_y + half_size,  # maxy
    )

    logger.info(f"Test BBOX ({size_km}km x {size_km}km): {bbox}")
    logger.info(f"Area: {size_km * size_km} sq km")

    return bbox


def create_custom_bbox(center_x, center_y, size_km=10):
    """
    Create a custom bounding box around specific LV95 coordinates

    Parameters:
    - center_x: LV95 Easting coordinate of center
    - center_y: LV95 Northing coordinate of center
    - size_km: Size of the square bbox in kilometers

    Returns:
    - bbox: (minx, miny, maxx, maxy) tuple
    """
    half_size = (size_km * 1000) / 2

    bbox = (
        center_x - half_size,  # minx
        center_y - half_size,  # miny
        center_x + half_size,  # maxx
        center_y + half_size,  # maxy
    )

    logger.info(
        f"Custom BBOX around ({center_x}, {center_y}) - {size_km}km x {size_km}km: {bbox}"
    )

    return bbox


def filter_tectonic_lines(tectonic_lines):
    """
    Filter tectonic lines by KIND between 14901001 and 14901009
    """
    # Filter by KIND range
    mask = (tectonic_lines["KIND"] >= 14901001) & (tectonic_lines["KIND"] <= 14901009)
    filtered_lines = tectonic_lines[mask].copy()

    logger.info(
        f"Found {len(filtered_lines)} tectonic lines with KIND between 14901001-14901009"
    )
    return filtered_lines


def split_line_by_polygon_boundary(line_geom, polygon_geom):
    """
    Split a line at the polygon boundary and return inside/outside parts
    Returns None if splitting fails (e.g., geometry segment overlaps with splitter)
    """
    try:
        # Get the polygon boundary
        boundary = polygon_geom.boundary

        # Split the line by the boundary
        split_result = split(line_geom, boundary)

        if hasattr(split_result, "geoms"):
            line_parts = list(split_result.geoms)
        else:
            line_parts = [split_result]  # Single geometry

        inside_parts = []
        outside_parts = []

        for part in line_parts:
            if isinstance(part, LineString):
                # Check if the centroid is inside the polygon
                if polygon_geom.contains(part.centroid):
                    inside_parts.append(part)
                else:
                    outside_parts.append(part)

        return inside_parts, outside_parts

    except ValueError as e:
        if "overlaps with the splitter" in str(e):
            logger.warning(
                f"Line geometry overlaps with splitter - skipping split: {e}"
            )
            return None, None  # Signal to keep original geometry unchanged
        else:
            logger.error(f"ValueError in line splitting: {e}")
            return [], []

    except Exception as e:
        logger.error(f"Error splitting line: {e}")
        return [], []


def get_line_length(geom):
    """
    Get the length of a line geometry
    """
    return geom.length


def generate_new_uuid():
    """
    Generate a new UUID string
    """
    return str(uuid.uuid4())


def find_intersecting_unco_polygons(line_geom, unco_polygons):
    """
    Find all UNCO polygons that intersect with a line geometry
    """
    intersecting_polys = []
    for idx, poly_row in unco_polygons.iterrows():
        poly_geom = poly_row.geometry
        if line_geom.intersects(poly_geom) and not line_geom.within(poly_geom):
            intersecting_polys.append(poly_geom)
    return intersecting_polys


def create_union_geometry(polygons, line_geom, max_polygons=20):
    """
    Create union of polygons, with fallback for large numbers

    Parameters:
    - polygons: List of polygon geometries
    - line_geom: Line geometry for buffered approach fallback
    - max_polygons: Maximum number of polygons to union directly
    """
    if len(polygons) == 0:
        return None

    if len(polygons) == 1:
        return polygons[0]

    try:
        if len(polygons) <= max_polygons:
            # Direct union for reasonable number of polygons
            logger.debug(f"Computing union of {len(polygons)} polygons")
            return unary_union(polygons)
        else:
            # For many polygons, use buffer approach around line
            logger.debug(f"Too many polygons ({len(polygons)}), using buffer approach")
            # Create small buffer around line to capture relevant polygons
            line_buffer = line_geom.buffer(100)  # 100m buffer, adjust as needed
            relevant_polys = [p for p in polygons if line_buffer.intersects(p)]
            logger.debug(f"Reduced to {len(relevant_polys)} relevant polygons")

            if len(relevant_polys) <= max_polygons:
                return unary_union(relevant_polys)
            else:
                # If still too many, return largest polygon as approximation
                logger.warning(
                    f"Still too many polygons, using largest as approximation"
                )
                return max(relevant_polys, key=lambda p: p.area)

    except Exception as e:
        logger.warning(f"Union computation failed: {e}, using largest polygon")
        return max(polygons, key=lambda p: p.area)


def process_intersecting_lines(tectonic_lines, unco_polygons, use_union=True):
    """
    Main processing function to handle line-polygon intersections

    Parameters:
    - use_union: If True, union intersecting UNCO polygons before splitting
    """
    # Ensure same CRS
    if tectonic_lines.crs != unco_polygons.crs:
        logger.info("Converting CRS to match...")
        unco_polygons = unco_polygons.to_crs(tectonic_lines.crs)

    # Find intersecting lines
    logger.info("Finding intersections...")
    intersecting_data = []

    for idx, line_row in tectonic_lines.iterrows():
        line_geom = line_row.geometry

        # Find all intersecting UNCO polygons for this line
        intersecting_polys = find_intersecting_unco_polygons(line_geom, unco_polygons)

        if intersecting_polys:
            intersecting_data.append((idx, line_row, intersecting_polys))

    logger.info(f"Found {len(intersecting_data)} lines that intersect unco polygons")

    # Log statistics about intersecting polygons
    poly_counts = [len(polys) for _, _, polys in intersecting_data]
    if poly_counts:
        logger.info(
            f"Polygon intersections per line: min={min(poly_counts)}, max={max(poly_counts)}, avg={np.mean(poly_counts):.1f}"
        )

    # Process each intersecting line
    new_features = []
    updated_features = []
    unchanged_indices = []  # Track lines that couldn't be split
    processed_indices = []

    for line_idx, line_row, intersecting_polys in intersecting_data:
        line_geom = line_row.geometry

        # Create combined geometry for splitting
        if use_union and len(intersecting_polys) > 1:
            logger.debug(
                f"Line {line_idx}: Processing {len(intersecting_polys)} intersecting UNCO polygons"
            )
            combined_geom = create_union_geometry(intersecting_polys, line_geom)
        else:
            # Use first intersecting polygon (original behavior for single polygons)
            combined_geom = intersecting_polys[0]

        if combined_geom is None:
            logger.warning(f"Line {line_idx}: No valid combined geometry")
            continue

        # Split the line
        inside_parts, outside_parts = split_line_by_polygon_boundary(
            line_geom, combined_geom
        )

        # Handle case where splitting failed due to overlap
        if inside_parts is None and outside_parts is None:
            logger.debug(f"Line {line_idx} kept unchanged due to geometry overlap")
            unchanged_indices.append(line_idx)
            continue

        if not inside_parts and not outside_parts:
            # No split occurred, skip
            logger.debug(f"Line {line_idx} - no split occurred")
            continue

        # Create new features
        original_attrs = line_row.drop("geometry").to_dict()

        # Process inside parts (each becomes a separate LineString feature)
        # This ensures we only have LineString geometries, not MultiLineString
        inside_features = []
        if inside_parts:
            for i, inside_part in enumerate(inside_parts):
                if isinstance(inside_part, LineString):
                    inside_feature = original_attrs.copy()
                    inside_feature["geometry"] = inside_part
                    inside_feature["TTEC_STATUS"] = (
                        14906003  # Set as 'not certain' inside UNCO
                    )
                    inside_feature["_change"] = "update"  # Mark as updated
                    inside_features.append(inside_part)
                    updated_features.append(inside_feature)

        # Process outside parts (each becomes a separate LineString feature)
        outside_features = []
        if outside_parts:
            for i, outside_part in enumerate(outside_parts):
                if isinstance(outside_part, LineString):
                    outside_feature = original_attrs.copy()
                    outside_feature["geometry"] = outside_part
                    # Keep original TTEC_STATUS (no change for outside parts)
                    outside_feature["_change"] = "new"  # Mark as new feature
                    outside_features.append(outside_part)
                    new_features.append(outside_feature)

        # Handle UUID assignment - give new UUIDs to all but the longest part
        # The longest part keeps the original UUID to maintain primary reference
        all_parts = list(
            zip(inside_features, ["inside"] * len(inside_features))
        ) + list(zip(outside_features, ["outside"] * len(outside_features)))

        if len(all_parts) > 1:
            # Find the longest part (keeps original UUID)
            longest_part, longest_type = max(all_parts, key=lambda x: x[0].length)

            # Assign new UUIDs to all other parts
            for part, part_type in all_parts:
                if part != longest_part:  # Not the longest part
                    # Find the corresponding feature and update its UUID
                    if part_type == "inside":
                        # Find in updated_features
                        for feat in reversed(updated_features):
                            if feat["geometry"] == part:
                                feat["UUID"] = generate_new_uuid()
                                break
                    else:  # outside
                        # Find in new_features
                        for feat in reversed(new_features):
                            if feat["geometry"] == part:
                                feat["UUID"] = generate_new_uuid()
                                break

        # Track this as processed (using dummy polygon for compatibility)
        processed_indices.append((line_idx, {"dummy": "processed"}))

    logger.info(f"Successfully processed {len(processed_indices)} intersecting lines")
    logger.info(f"Created {len(new_features)} new LineString features (outside UNCO)")
    logger.info(
        f"Created {len(updated_features)} updated LineString features (inside UNCO)"
    )
    logger.info(f"Kept {len(unchanged_indices)} lines unchanged due to geometry issues")

    return new_features, updated_features, processed_indices


def save_results(
    tectonic_lines,
    new_features,
    updated_features,
    intersecting_indices,
    output_path,
    save_deleted=False,
):
    """
    Save the corrected results to a new geodatabase or file

    Parameters:
    - save_deleted: If True, includes original intersecting lines marked as 'delete'
    """
    # Get original intersecting lines for deletion tracking
    indices_to_remove = [idx for idx, _ in intersecting_indices]
    remaining_lines = tectonic_lines.drop(indices_to_remove)
    deleted_lines = tectonic_lines.loc[indices_to_remove].copy()

    # Mark remaining lines that weren't changed
    remaining_lines = remaining_lines.copy()
    remaining_lines["_change"] = "unchanged"

    # Combine all features
    all_new_features = new_features + updated_features

    if save_deleted and len(deleted_lines) > 0:
        # Mark deleted lines
        deleted_lines["_change"] = "delete"
        all_new_features.extend(deleted_lines.to_dict("records"))

    if all_new_features:
        # Create GeoDataFrame from new features
        new_gdf = gpd.GeoDataFrame(all_new_features, crs=tectonic_lines.crs)

        # Combine with remaining lines
        final_gdf = pd.concat([remaining_lines, new_gdf], ignore_index=True)
    else:
        final_gdf = remaining_lines

    # Save to file
    logger.info(f"Saving corrected data to {output_path}")

    # Determine output format based on extension
    if output_path.suffix.lower() == ".gdb":
        # Save to FileGDB
        final_gdf.to_file(
            output_path, layer="GC_LINEAR_OBJECTS_CORRECTED", driver="FileGDB"
        )
    else:
        # Save to other format (shapefile, geopackage, etc.)
        final_gdf.to_file(output_path)

    # Summary statistics
    change_counts = final_gdf["_change"].value_counts()
    logger.success(f"Saved {len(final_gdf)} features total")
    logger.info("Change summary:")
    for change_type, count in change_counts.items():
        logger.info(f"  {change_type}: {count} features")


def main():
    """
    Main execution function
    """
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),  # Print to console
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    # Input parameters - modify these as needed
    gdb_path = "path/to/your/geodatabase.gdb"  # Update this path
    output_path = Path(
        "path/to/output/corrected_tectonic_lines.gpkg"
    )  # Update this path

    # BBOX options for testing (uncomment one of these):

    # Option 1: Use predefined Alps test area (10km x 10km = 100 sq km)
    test_bbox = create_test_bbox_alps(size_km=10)

    # Option 2: Use smaller test area (3km x 3km = 9 sq km)
    # test_bbox = create_test_bbox_alps(size_km=3)

    # Option 3: Use custom coordinates (example for different Alps location)
    # test_bbox = create_custom_bbox(center_x=2_650_000, center_y=1_150_000, size_km=5)

    # Option 4: No BBOX (process entire dataset) - comment out test_bbox line
    # test_bbox = None

    logger.info("Starting tectonic lines correction process...")

    # Load data with BBOX
    tectonic_lines, unco_polygons = load_geodatabase_layers(gdb_path, bbox=test_bbox)

    if tectonic_lines is None or unco_polygons is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Filter tectonic lines
    filtered_lines = filter_tectonic_lines(tectonic_lines)

    if len(filtered_lines) == 0:
        logger.warning("No tectonic lines found with specified KIND range. Exiting.")
        return

    # Process intersections
    new_features, updated_features, intersecting_indices = process_intersecting_lines(
        filtered_lines,
        unco_polygons,
        use_union=True,  # Default to union in standalone script
    )

    if not new_features and not updated_features:
        logger.info("No intersections found or processed. No changes needed.")
        return

    # Save results
    save_deleted_features = (
        True  # Set to False if you don't want to track deleted features
    )
    save_results(
        filtered_lines,
        new_features,
        updated_features,
        intersecting_indices,
        output_path,
        save_deleted=save_deleted_features,
    )

    logger.success("Tectonic lines correction completed successfully!")


def main():
    """
    Main execution function
    """
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),  # Print to console
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    # Input parameters - modify these as needed
    gdb_path = (
        "/home/marco/DATA/Backups/20250708_0300_2030-12-31.gdb"  # Update this path
    )
    output_path = Path(
        "output/20250708_rc2_corrected_tectonic_lines.gpkg"
    )  # Update this path

    # BBOX options for testing (uncomment one of these):

    # Option 1: Use predefined Alps test area (10km x 10km = 100 sq km)
    test_bbox = create_test_bbox_alps(size_km=10)

    # Option 2: Use smaller test area (3km x 3km = 9 sq km)
    # test_bbox = create_test_bbox_alps(size_km=3)

    # Option 3: Use custom coordinates (example for different Alps location)
    # test_bbox = create_custom_bbox(center_x=2_650_000, center_y=1_150_000, size_km=5)

    # Option 4: No BBOX (process entire dataset) - comment out test_bbox line
    # test_bbox = None

    logger.info("Starting tectonic lines correction process...")

    # Load data with BBOX
    tectonic_lines, unco_polygons = load_geodatabase_layers(gdb_path, bbox=test_bbox)

    if tectonic_lines is None or unco_polygons is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Filter tectonic lines
    filtered_lines = filter_tectonic_lines(tectonic_lines)

    if len(filtered_lines) == 0:
        logger.warning("No tectonic lines found with specified KIND range. Exiting.")
        return

    # Process intersections
    new_features, updated_features, intersecting_indices = process_intersecting_lines(
        filtered_lines, unco_polygons
    )

    if not new_features and not updated_features:
        logger.info("No intersections found or processed. No changes needed.")
        return

    # Save results
    save_deleted_features = (
        True  # Set to False if you don't want to track deleted features
    )
    save_results(
        filtered_lines,
        new_features,
        updated_features,
        intersecting_indices,
        output_path,
        save_deleted=save_deleted_features,
    )

    logger.success("Tectonic lines correction completed successfully!")


if __name__ == "__main__":
    main()
