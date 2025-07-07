import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import split
import uuid
import numpy as np
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import split
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


def process_intersecting_lines(tectonic_lines, unco_polygons):
    """
    Main processing function to handle line-polygon intersections
    """
    # Ensure same CRS
    if tectonic_lines.crs != unco_polygons.crs:
        logger.info("Converting CRS to match...")
        unco_polygons = unco_polygons.to_crs(tectonic_lines.crs)

    # Find intersecting lines
    logger.info("Finding intersections...")
    intersecting_indices = []

    for idx, line_row in tectonic_lines.iterrows():
        line_geom = line_row.geometry

        # Check if line intersects any unco polygon
        for _, poly_row in unco_polygons.iterrows():
            poly_geom = poly_row.geometry

            if line_geom.intersects(poly_geom) and not line_geom.within(poly_geom):
                intersecting_indices.append((idx, poly_row))
                break  # Found intersection, move to next line

    logger.info(f"Found {len(intersecting_indices)} lines that intersect unco polygons")

    # Process each intersecting line
    new_features = []
    updated_features = []
    unchanged_indices = []  # Track lines that couldn't be split

    for line_idx, poly_row in intersecting_indices:
        line_row = tectonic_lines.loc[line_idx]
        line_geom = line_row.geometry
        poly_geom = poly_row.geometry

        # Split the line
        inside_parts, outside_parts = split_line_by_polygon_boundary(
            line_geom, poly_geom
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

        # Combine parts into MultiLineString if multiple parts
        if len(inside_parts) > 1:
            inside_geom = MultiLineString(inside_parts)
        elif len(inside_parts) == 1:
            inside_geom = inside_parts[0]
        else:
            inside_geom = None

        if len(outside_parts) > 1:
            outside_geom = MultiLineString(outside_parts)
        elif len(outside_parts) == 1:
            outside_geom = outside_parts[0]
        else:
            outside_geom = None

        # Calculate lengths to determine smaller part
        inside_length = inside_geom.length if inside_geom else 0
        outside_length = outside_geom.length if outside_geom else 0

        # Create new features
        original_attrs = line_row.drop("geometry").to_dict()

        # Create feature for inside part with TTEC_STATUS = 14906003 (not certain)
        if inside_geom:
            inside_feature = original_attrs.copy()
            inside_feature["geometry"] = inside_geom
            inside_feature["TTEC_STATUS"] = 14906003  # Set as 'not certain' inside UNCO
            inside_feature["_change"] = "update"  # Mark as updated
            updated_features.append(inside_feature)

        # Create feature for outside part (keep original status)
        if outside_geom:
            outside_feature = original_attrs.copy()
            outside_feature["geometry"] = outside_geom
            # Keep original TTEC_STATUS (no change for outside parts)
            outside_feature["_change"] = "new"  # Mark as new feature
            new_features.append(outside_feature)

        # Give new UUID to the smaller part
        if inside_geom and outside_geom:
            if inside_length <= outside_length:
                # Inside part is smaller, give it new UUID
                updated_features[-1]["UUID"] = generate_new_uuid()
            else:
                # Outside part is smaller, give it new UUID
                new_features[-1]["UUID"] = generate_new_uuid()

    # Update intersecting_indices to remove unchanged lines
    processed_indices = [
        (idx, poly)
        for idx, poly in intersecting_indices
        if idx not in unchanged_indices
    ]

    logger.info(f"Successfully processed {len(processed_indices)} intersecting lines")
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
    gdb_path = (
        "/home/marco/DATA/Backups/20250703_0300_2030-12-31.gdb"  # Update this path
    )
    output_path = Path(
        "output/20250703_rc2_corrected_tectonic_lines.gpkg"
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
