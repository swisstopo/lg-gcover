#!/usr/bin/env python3
"""
GeoCover Denormalization Tool
A professional CLI tool to denormalize GeoCover geodatabase tables with their lookup relationships.
"""

import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
import fiona
import geopandas as gpd
import pandas as pd
from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from rich.table import Table
from rich.tree import Tree
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import split as shapely_split
from shapely.ops import unary_union
from shapely.prepared import prep

try:
    from osgeo import ogr
except ImportError:
    logger.warning("GDAL/OGR not available - coded domain extraction disabled")
    ogr = None

try:
    from gcover.core.geometry import (load_gpkg_with_validation,
                                      split_features_by_mapsheets)
except ImportError:
    logger.warning("Not validation function")
    load_gpkg_with_validation = None


# Configure rich console
console = Console()

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Field domain mapping for coded domain decoding
FIELD_DOMAIN_MAPPING = {
    "RUNC_MORPHOLO": ("GC_UN_DEP_RUNC_MORPHOLO_CD", "morpholo_desc"),
    "RUNC_GLAC_TYP": ("GC_UN_DEP_RUNC_GLAC_TYP_CD", "glac_typ_desc"),
    "RUNC_THIN_COV": ("GC_UN_DEP_RUNC_THIN_COV_CD", "thin_cov_desc"),
    "RUNC_BURIED_OUT": ("GC_BOOLEAN_CD", "buried_out_desc"),
}

# Lookup table mappings for description fields
LOOKUP_TABLE_MAPPING = {
    "CHRONO_TOP": ("GC_CHRONO", "chrono_top_code", "chrono_top_desc"),
    "CHRONO_BASE": ("GC_CHRONO", "chrono_base_code", "chrono_base_desc"),
    "LITHO_MAIN": ("GC_LITHO", "litho_main_code", "litho_main_desc"),
    "LITHO_SEC": ("GC_LITHO", "litho_sec_code", "litho_sec_desc"),
    "LITHO_TER": ("GC_LITHO", "litho_ter_code", "litho_ter_desc"),
    "LITSTRAT_FORMATION_BANK": (
        "GC_LITSTRAT_FORMATION_BANK",
        "litstrat_formation_bank_code",
        "litstrat_formation_bank_desc",
    ),
    "CORRELATION": ("GC_CORRELATION", "correlation_code", "correlation_desc"),
    "TECTO": ("GC_TECTO", "tecto_code", "tecto_desc"),
}


def extend_line_to_cross_polygon(line, polygon, extension_factor=1.5):
    """
    Extend a line segment to ensure it fully crosses a polygon.
    This helps shapely's split function work properly.
    """
    if line.geom_type == "MultiLineString":
        # For MultiLineString, process each part
        extended_parts = []
        for part in line.geoms:
            extended_parts.append(
                extend_line_to_cross_polygon(part, polygon, extension_factor)
            )
        return unary_union(extended_parts)

    if line.geom_type != "LineString":
        return line

    # Get the line's bounding box
    line_bounds = line.bounds
    poly_bounds = polygon.bounds

    # Calculate extension distance based on polygon size
    poly_width = poly_bounds[2] - poly_bounds[0]
    poly_height = poly_bounds[3] - poly_bounds[1]
    extension_dist = max(poly_width, poly_height) * extension_factor

    # Get line coordinates
    coords = list(line.coords)
    if len(coords) < 2:
        return line

    # Extend first point
    start = coords[0]
    second = coords[1]
    dx = start[0] - second[0]
    dy = start[1] - second[1]
    length = (dx**2 + dy**2) ** 0.5
    if length > 0:
        dx, dy = dx / length, dy / length
        new_start = (start[0] + dx * extension_dist, start[1] + dy * extension_dist)
    else:
        new_start = start

    # Extend last point
    end = coords[-1]
    second_last = coords[-2]
    dx = end[0] - second_last[0]
    dy = end[1] - second_last[1]
    length = (dx**2 + dy**2) ** 0.5
    if length > 0:
        dx, dy = dx / length, dy / length
        new_end = (end[0] + dx * extension_dist, end[1] + dy * extension_dist)
    else:
        new_end = end

    # Create extended line
    extended_coords = [new_start] + coords[1:-1] + [new_end]
    return LineString(extended_coords)


def split_polygon_with_line(polygon, line):
    """
    Split a polygon using a line, handling edge cases properly.
    Returns a list of polygon parts.
    """
    if not polygon.intersects(line):
        return [polygon]

    # Try direct split first
    try:
        result = shapely_split(polygon, line)

        # Extract polygon parts only
        parts = []
        if hasattr(result, "geoms"):
            for geom in result.geoms:
                if geom.geom_type == "Polygon":
                    parts.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    parts.extend(list(geom.geoms))
        elif result.geom_type == "Polygon":
            parts = [result]
        elif result.geom_type == "MultiPolygon":
            parts = list(result.geoms)
        else:
            parts = [polygon]  # Fallback

        # If we got more than one part, split succeeded
        if len(parts) > 1:
            return parts
    except Exception as e:
        pass

    # If direct split didn't work, try with extended line
    try:
        extended_line = extend_line_to_cross_polygon(line, polygon)
        result = shapely_split(polygon, extended_line)

        parts = []
        if hasattr(result, "geoms"):
            for geom in result.geoms:
                if geom.geom_type == "Polygon":
                    # Check if this part actually intersects the original polygon
                    if polygon.intersects(geom) and geom.area > 1e-6:
                        parts.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    for subgeom in geom.geoms:
                        if polygon.intersects(subgeom) and subgeom.area > 1e-6:
                            parts.append(subgeom)
        elif result.geom_type == "Polygon":
            parts = [result]
        elif result.geom_type == "MultiPolygon":
            parts = [p for p in result.geoms if p.area > 1e-6]

        if len(parts) > 1:
            return parts
    except Exception as e:
        pass

    # Last resort: use difference with buffered line
    try:
        buffered_line = line.buffer(0.001)
        diff_result = polygon.difference(buffered_line)

        parts = []
        if diff_result.geom_type == "Polygon":
            parts = [diff_result]
        elif diff_result.geom_type == "MultiPolygon":
            parts = [p for p in diff_result.geoms if p.area > 1e-6]
        elif diff_result.geom_type == "GeometryCollection":
            for geom in diff_result.geoms:
                if geom.geom_type == "Polygon" and geom.area > 1e-6:
                    parts.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    parts.extend([p for p in geom.geoms if p.area > 1e-6])

        if len(parts) > 1:
            return parts
    except Exception:
        pass

    # No split occurred, return original
    return [polygon]


def split_polygons_with_layer(
    gdf, split_layer_path, split_layer_name=None, split_tolerance=0.001
):
    """
    Split polygons using a line or polygon layer.

    Parameters:
    - gdf: GeoDataFrame with polygons to split
    - split_layer_path: Path to FileGDB containing split layer
    - split_layer_name: Name of the layer to use for splitting
    - split_tolerance: Tolerance for geometric operations

    Returns:
    - GeoDataFrame with split polygons, all attributes retained
    """
    split_layer_path = Path(split_layer_path)

    # Load split layer
    layers = fiona.listlayers(str(split_layer_path))
    if split_layer_name is None:
        split_layer_name = layers[0]

    splitter = gpd.read_file(split_layer_path, layer=split_layer_name)

    # CRS alignment
    if gdf.crs != splitter.crs:
        splitter = splitter.to_crs(gdf.crs)

    # Prepare splitter geometry
    splitter_geom = unary_union(splitter.geometry)

    console.print(splitter.sample(5))

    # Convert to lines if needed
    if splitter_geom.geom_type in ("Polygon", "MultiPolygon"):
        splitter_geom = splitter_geom.boundary

    if splitter_geom is None or splitter_geom.is_empty:
        return gdf.copy().reset_index(drop=True)

    # Merge line segments for better splitting
    if splitter_geom.geom_type == "MultiLineString":
        try:
            splitter_geom = linemerge(splitter_geom)
        except:
            pass

    # Prepare spatial index
    splitter_prep = prep(splitter_geom)

    out_records = []
    split_count = 0
    no_split_count = 0
    total_parts = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Splitting polygons...", total=len(gdf))

        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Handle trivial cases
            if geom is None or geom.is_empty:
                out_records.append(row)
                progress.update(task, advance=1)
                continue

            # Fix invalid geometries
            if not geom.is_valid:
                geom = geom.buffer(0)

            # Skip if no intersection
            if not splitter_prep.intersects(geom):
                out_records.append(row)
                no_split_count += 1
                progress.update(task, advance=1)
                continue

            # Get the part of the splitter that intersects this polygon
            try:
                local_splitter = splitter_geom.intersection(
                    geom.buffer(split_tolerance)
                )

                # Convert to line if needed
                if local_splitter.geom_type in ("Polygon", "MultiPolygon"):
                    local_splitter = local_splitter.boundary
                elif local_splitter.geom_type == "GeometryCollection":
                    # Extract lines from collection
                    lines = []
                    for g in local_splitter.geoms:
                        if g.geom_type in ("LineString", "MultiLineString"):
                            lines.append(g)
                        elif g.geom_type in ("Polygon", "MultiPolygon"):
                            lines.append(g.boundary)
                    if lines:
                        local_splitter = unary_union(lines)
                    else:
                        local_splitter = None

                if local_splitter is None or local_splitter.is_empty:
                    out_records.append(row)
                    no_split_count += 1
                    progress.update(task, advance=1)
                    continue

                # Perform the split
                parts = split_polygon_with_line(geom, local_splitter)

                if len(parts) > 1:
                    split_count += 1
                    total_parts += len(parts)
                    # Add all parts with original attributes
                    for part in parts:
                        if part.area > 1e-6:  # Skip tiny slivers
                            new_row = row.copy()
                            new_row["geometry"] = part
                            out_records.append(new_row)
                else:
                    out_records.append(row)
                    no_split_count += 1

            except Exception as e:
                print(f"Warning: Failed to split geometry at index {idx}: {e}")
                out_records.append(row)
                no_split_count += 1

            progress.update(task, advance=1)

    print(f"\nResults:")
    print(f"  - {split_count} polygons split into {total_parts} parts")
    print(f"  - {no_split_count} polygons unchanged")
    print(f"  - Total output features: {len(out_records)}")

    result = gpd.GeoDataFrame(out_records, crs=gdf.crs)
    return result.reset_index(drop=True)


def extract_coded_domains(gdb_path: Union[str, Path]) -> Dict[str, Dict]:
    """Extract coded domain definitions from FileGDB metadata"""
    if ogr is None:
        logger.warning("GDAL/OGR not available - skipping coded domain extraction")
        return {}

    coded_domains = {}
    gdb_path = str(gdb_path)

    try:
        ds = ogr.Open(gdb_path)
        if ds is None:
            logger.warning(f"Could not open {gdb_path} with OGR")
            return {}

        res = ds.ExecuteSQL("SELECT * FROM GDB_Items")
        if res is None:
            logger.warning("Could not execute SQL query on GDB_Items")
            return {}

        logger.debug("Extracting coded domains from GDB_Items...")

        for i in range(res.GetFeatureCount()):
            feature = res.GetNextFeature()
            if feature:
                definition = feature.GetField("Definition")
                if definition:
                    try:
                        xml = ET.fromstring(definition)
                        if xml.tag == "GPCodedValueDomain2":
                            domain_name = xml.find("DomainName").text
                            description = (
                                xml.find("Description").text
                                if xml.find("Description") is not None
                                else ""
                            )
                            field_type = (
                                xml.find("FieldType").text
                                if xml.find("FieldType") is not None
                                else ""
                            )

                            # Extract code-value pairs
                            code_mapping = {}
                            for table in xml.iter("CodedValues"):
                                for child in table:
                                    code_elem = child.find("Code")
                                    name_elem = child.find("Name")
                                    if code_elem is not None and name_elem is not None:
                                        code = code_elem.text
                                        name = name_elem.text
                                        if code and name:
                                            try:
                                                code_mapping[int(code)] = name
                                            except ValueError:
                                                code_mapping[code] = name

                            if code_mapping:
                                coded_domains[domain_name] = {
                                    "description": description,
                                    "field_type": field_type,
                                    "codes": code_mapping,
                                }
                                logger.debug(
                                    f"Found domain {domain_name}: {len(code_mapping)} codes"
                                )
                    except ET.ParseError as e:
                        logger.warning(f"Could not parse XML definition: {e}")
                        continue

        res = None
        ds = None

    except Exception as e:
        logger.warning(f"Could not extract coded domains: {e}")

    logger.info(f"Extracted {len(coded_domains)} coded domains")
    return coded_domains


def decode_coded_domains(
    df: pd.DataFrame, coded_domains: Dict[str, Dict]
) -> pd.DataFrame:
    """Decode coded domain fields to human-readable descriptions"""
    result_df = df.copy()

    for field_name, (domain_name, output_col) in FIELD_DOMAIN_MAPPING.items():
        if field_name in result_df.columns and domain_name in coded_domains:
            logger.debug(f"Decoding {field_name} using domain {domain_name}")

            # Create mapping function
            code_mapping = coded_domains[domain_name]["codes"]

            # Apply mapping
            result_df[output_col] = result_df[field_name].map(code_mapping)

            # Log statistics
            decoded_count = result_df[output_col].dropna().count()
            total_count = result_df[field_name].dropna().count()
            logger.debug(
                f"Decoded {decoded_count}/{total_count} values for {field_name}"
            )

        elif field_name in result_df.columns:
            logger.warning(f"Domain {domain_name} not found for field {field_name}")
        else:
            logger.debug(f"Field {field_name} not found in dataset")

    return result_df


def add_lookup_descriptions(
    df: pd.DataFrame, gdb_path: Union[str, Path]
) -> pd.DataFrame:
    """Add description fields by joining with lookup tables"""
    result_df = df.copy()

    for field_name, (
        lookup_table,
        code_field,
        desc_field,
    ) in LOOKUP_TABLE_MAPPING.items():
        if field_name in result_df.columns:
            try:
                # Read lookup table
                lookup_df = gpd.read_file(gdb_path, layer=lookup_table)
                logger.debug(f"Read {len(lookup_df)} records from {lookup_table}")

                # Rename original field to code field
                if field_name in result_df.columns:
                    result_df = result_df.rename(columns={field_name: code_field})

                # Join with lookup table to get descriptions
                lookup_dict = dict(zip(lookup_df["GEOLCODE"], lookup_df["DESCRIPTION"]))
                result_df[desc_field] = result_df[code_field].map(lookup_dict)

                # Log statistics
                decoded_count = result_df[desc_field].dropna().count()
                total_count = result_df[code_field].dropna().count()
                logger.debug(
                    f"Added descriptions for {decoded_count}/{total_count} {field_name} values"
                )

            except Exception as e:
                logger.warning(f"Could not add descriptions for {field_name}: {e}")

    return result_df


class GeoCoverDenormalizer:
    """Main class for GeoCover denormalization operations."""

    def __init__(self, gdb_path: Union[str, Path], verbose: bool = True):
        self.gdb_path = Path(gdb_path)
        self.verbose = verbose
        self.console = console
        self.coded_domains = None

        if not self.gdb_path.exists():
            raise FileNotFoundError(f"FileGDB not found: {self.gdb_path}")

        logger.info(f"Initialized GeoCover denormalizer for: {self.gdb_path}")

        # Extract coded domains
        logger.info("Extracting coded domains...")
        self.coded_domains = extract_coded_domains(self.gdb_path)

    def _read_layer_safe(
        self, layer_name: str, group: Optional[str] = "GC_ROCK_BODIES"
    ) -> gpd.GeoDataFrame:
        """Safely read a layer, trying with and without group prefix."""
        try:
            if group:
                layer_path = f"{group}/{layer_name}"
                if load_gpkg_with_validation:
                    gdf = load_gpkg_with_validation(self.gdb_path, layer=layer_path)
                else:
                    gdf = gpd.read_file(self.gdb_path, layer=layer_path)
                logger.debug(f"Successfully read {layer_path}")
                return gdf
        except Exception:
            logger.debug(
                f"Failed to read with group prefix, trying {layer_name} directly"
            )

        try:
            if load_gpkg_with_validation:
                gdf = load_gpkg_with_validation(self.gdb_path, layer=layer_name)
            else:
                gdf = gpd.read_file(self.gdb_path, layer=layer_name)
            logger.debug(f"Successfully read {layer_name}")
            return gdf
        except Exception as e:
            logger.error(f"Failed to read layer {layer_name}: {e}")
            raise

    def denormalize_simple_relationship(
        self,
        main_table: str,
        lookup_table: str,
        relationship_table: str,
        description_alias: str,
        task_id=None,
        progress=None,
    ) -> gpd.GeoDataFrame:
        """Denormalize tables with simple junction table relationships."""

        if progress and task_id:
            progress.update(task_id, description=f"Reading {main_table}...")

        # Read tables
        main_gdf = self._read_layer_safe(main_table)
        lookup_df = gpd.read_file(self.gdb_path, layer=lookup_table)
        relationship_df = gpd.read_file(self.gdb_path, layer=relationship_table)

        logger.info(f"Loaded {len(main_gdf)} features from {main_table}")
        logger.info(f"Loaded {len(lookup_df)} records from {lookup_table}")
        logger.info(
            f"Loaded {len(relationship_df)} relationships from {relationship_table}"
        )

        if progress and task_id:
            progress.update(
                task_id, description=f"Joining {main_table} with {lookup_table}..."
            )

        # Auto-detect foreign key columns
        rel_columns = [col for col in relationship_df.columns if col != "geometry"]
        main_fk_col, lookup_fk_col = rel_columns[0], rel_columns[1]

        logger.debug(
            f"Using foreign keys: {main_fk_col} -> {main_table}, {lookup_fk_col} -> {lookup_table}"
        )

        # Perform joins
        main_with_relation = main_gdf.merge(
            relationship_df[[main_fk_col, lookup_fk_col]],
            left_on="UUID",
            right_on=main_fk_col,
            how="left",
        )

        lookup_columns = ["UUID"] + [
            col
            for col in lookup_df.columns
            if col not in ["UUID", "geometry"]
            and col in ["DESCRIPTION", "GEOLCODE", "TREE_LEVEL"]
        ]

        denormalized_gdf = main_with_relation.merge(
            lookup_df[lookup_columns],
            left_on=lookup_fk_col,
            right_on="UUID",
            how="left",
            suffixes=("", "_LOOKUP"),
        )

        # Clean up and rename
        columns_to_drop = [
            col
            for col in denormalized_gdf.columns
            if col.endswith("_LOOKUP") or col in [main_fk_col, lookup_fk_col]
        ]
        denormalized_gdf = denormalized_gdf.drop(
            columns=columns_to_drop, errors="ignore"
        )

        if "DESCRIPTION" in denormalized_gdf.columns:
            denormalized_gdf = denormalized_gdf.rename(
                columns={"DESCRIPTION": description_alias}
            )
            logger.debug(f"Renamed DESCRIPTION to {description_alias}")

        null_count = (
            denormalized_gdf[description_alias].isna().sum()
            if description_alias in denormalized_gdf.columns
            else 0
        )
        logger.info(
            f"Final result: {len(denormalized_gdf)} records, {null_count} without lookup data"
        )

        return denormalized_gdf

    def denormalize_bedrock(self, task_id=None, progress=None) -> gpd.GeoDataFrame:
        """Denormalize GC_BEDROCK with geological mapping unit attributes."""

        if progress and task_id:
            progress.update(task_id, description="Reading bedrock tables...")

        # Read tables
        bedrock_gdf = self._read_layer_safe("GC_BEDROCK")
        mapping_unit_att_df = gpd.read_file(
            self.gdb_path, layer="GC_GEOL_MAPPING_UNIT_ATT"
        )
        mapping_unit_df = gpd.read_file(self.gdb_path, layer="GC_GEOL_MAPPING_UNIT")

        logger.info(f"Loaded {len(bedrock_gdf)} bedrock features")
        logger.info(f"Loaded {len(mapping_unit_att_df)} mapping unit attributes")
        logger.info(f"Loaded {len(mapping_unit_df)} mapping units")

        if progress and task_id:
            progress.update(task_id, description="Joining bedrock with attributes...")

        # Try direct foreign key first
        if "GEOL_MAPPING_UNIT_ATT_UUID" in bedrock_gdf.columns:
            logger.debug("Using direct foreign key GEOL_MAPPING_UNIT_ATT_UUID")

            bedrock_with_att = bedrock_gdf.merge(
                mapping_unit_att_df,
                left_on="GEOL_MAPPING_UNIT_ATT_UUID",
                right_on="UUID",
                how="left",
                suffixes=("", "_ATT"),
            )

            if "GEOL_MAPPING_UNIT" in bedrock_with_att.columns:
                denormalized_gdf = bedrock_with_att.merge(
                    mapping_unit_df[["GEOLCODE", "UUID", "DESCRIPTION", "TREE_LEVEL"]],
                    left_on="GEOL_MAPPING_UNIT",
                    right_on="GEOLCODE",
                    how="left",
                    suffixes=("", "_MAPPING_UNIT"),
                )
            else:
                denormalized_gdf = bedrock_with_att
        else:
            # Use relationship table
            logger.debug("Using relationship table GC_BEDR_GEOL_MAPPING_UNIT_ATT")
            try:
                relationship_df = gpd.read_file(
                    self.gdb_path, layer="GC_BEDR_GEOL_MAPPING_UNIT_ATT"
                )
                rel_columns = [
                    col for col in relationship_df.columns if col != "geometry"
                ]

                bedrock_with_relation = bedrock_gdf.merge(
                    relationship_df, left_on="UUID", right_on=rel_columns[1], how="left"
                )

                denormalized_gdf = bedrock_with_relation.merge(
                    mapping_unit_att_df,
                    left_on=rel_columns[0],
                    right_on="UUID",
                    how="left",
                    suffixes=("", "_ATT"),
                )
            except Exception as e:
                logger.warning(f"Relationship table method failed: {e}")
                denormalized_gdf = bedrock_gdf

        # Clean up
        columns_to_drop = [
            col
            for col in denormalized_gdf.columns
            if col.endswith("_ATT") or col.endswith("_MAPPING_UNIT")
        ]
        denormalized_gdf = denormalized_gdf.drop(
            columns=columns_to_drop, errors="ignore"
        )

        if (
            "DESCRIPTION" in denormalized_gdf.columns
            and "DESCRIPTION" not in bedrock_gdf.columns
        ):
            denormalized_gdf = denormalized_gdf.rename(
                columns={"DESCRIPTION": "MAPPING_UNIT_DESC"}
            )

        logger.info(
            f"Bedrock denormalization complete: {len(denormalized_gdf)} records"
        )
        return denormalized_gdf

    def denormalize_unco_deposits(
        self, task_id=None, progress=None
    ) -> gpd.GeoDataFrame:
        """Denormalize GC_UNCO_DESPOSIT with multiple lookup tables."""

        if progress and task_id:
            progress.update(
                task_id, description="Reading unconsolidated deposit tables..."
            )

        # Read main table
        unco_gdf = self._read_layer_safe("GC_UNCO_DESPOSIT")
        logger.info(f"Loaded {len(unco_gdf)} unconsolidated deposit features")

        # Configuration for relationships
        relationships_config = {
            "admixture": {
                "lookup_table": "GC_ADMIXTURE",
                "relationship_table": "GC_UN_DEP_ADMIXTUR_GC_ADMIXT",
                "field_prefix": "ADMIX",
            },
            "characteristics": {
                "lookup_table": "GC_CHARCAT",
                "relationship_table": "GC_UN_DEP_CHARACT_GC_CHARCAT",
                "field_prefix": "CHARCAT",
            },
            "composition": {
                "lookup_table": "GC_COMPOSIT",
                "relationship_table": "GC_UN_DEP_COMPOSIT_GC_COMPOS",
                "field_prefix": "COMPOSIT",
            },
            "lithology": {
                "lookup_table": "GC_LITHO",
                "relationship_table": "GC_UN_DEP_MAT_TYPE_GC_LITHO",
                "field_prefix": "LITHO",
            },
        }

        result_gdf = unco_gdf.copy()

        for rel_name, config in relationships_config.items():
            if progress and task_id:
                progress.update(
                    task_id, description=f"Processing {rel_name} relationship..."
                )

            try:
                lookup_df = gpd.read_file(self.gdb_path, layer=config["lookup_table"])
                relationship_df = gpd.read_file(
                    self.gdb_path, layer=config["relationship_table"]
                )

                logger.debug(
                    f"{rel_name}: {len(lookup_df)} lookup records, {len(relationship_df)} relationships"
                )

                rel_columns = [
                    col for col in relationship_df.columns if col != "geometry"
                ]
                unco_fk_col, lookup_fk_col = rel_columns[0], rel_columns[1]

                rel_with_lookup = relationship_df.merge(
                    lookup_df[["UUID", "DESCRIPTION", "GEOLCODE"]],
                    left_on=lookup_fk_col,
                    right_on="UUID",
                    how="left",
                )

                # Aggregate descriptions
                aggregated = (
                    rel_with_lookup.groupby(unco_fk_col)
                    .agg(
                        {
                            "DESCRIPTION": lambda x: " | ".join(x.dropna().astype(str)),
                            "GEOLCODE": lambda x: " | ".join(x.dropna().astype(str)),
                        }
                    )
                    .reset_index()
                )

                field_mapping = {
                    "DESCRIPTION": f"{config['field_prefix']}_DESC",
                    "GEOLCODE": f"{config['field_prefix']}_CODES",
                }
                aggregated = aggregated.rename(columns=field_mapping)

                result_gdf = result_gdf.merge(
                    aggregated, left_on="UUID", right_on=unco_fk_col, how="left"
                )
                result_gdf = result_gdf.drop(columns=[unco_fk_col], errors="ignore")

                logger.debug(f"Added {config['field_prefix']} fields")

            except Exception as e:
                logger.warning(f"Error processing {rel_name}: {e}")
                continue

        logger.info(
            f"Unco deposits denormalization complete: {len(result_gdf)} records"
        )
        return result_gdf

    def copy_table_as_is(
        self, table_name: str, task_id=None, progress=None
    ) -> gpd.GeoDataFrame:
        """Copy a table as-is without any denormalization."""

        if progress and task_id:
            progress.update(task_id, description=f"Reading {table_name}...")

        logger.debug(f"Attempting to read table: {table_name}")
        gdf = self._read_layer_safe(table_name)

        logger.info(f"Successfully copied {len(gdf)} features from {table_name}")
        logger.debug(f"{table_name} columns: {list(gdf.columns)}")

        # Verify we got the right table by checking some expected characteristics
        if hasattr(gdf, "geometry") and not gdf.empty:
            geom_types = gdf.geometry.geom_type.value_counts()
            logger.debug(f"{table_name} geometry types: {dict(geom_types)}")

        return gdf

    def clean_metadata_columns(
        self, gdf: gpd.GeoDataFrame, remove_metadata: bool = False
    ) -> gpd.GeoDataFrame:
        """Remove metadata columns, perform data type conversions, and enhance fields."""

        metadata_columns = [
            "INTEGRATION_OBJECT_UUID",
            "DATEOFCHANGE",
            "DATEOFCREATION",
            "ORIGINAL_ORIGIN",
            "REASONFORCHANGE",
            "REVISION_QUALITY",
            "OBJECTORIGIN_MONTH",
            "OBJECTORIGIN_YEAR",
            "CREATION_YEAR",
            "CREATION_MONTH",
            "RC_ID_CREATION",
            "RC_ID",
            "REVISION_MONTH",
            "REVISION_YEAR",
            "WU_ID_CREATION",
            "WU_ID",
            "TREE_LEVEL",  # User requested to omit this
        ]

        result_gdf = gdf.copy()

        # Remove metadata columns if requested
        if remove_metadata:
            columns_to_remove = [
                col for col in metadata_columns if col in result_gdf.columns
            ]
            if columns_to_remove:
                result_gdf = result_gdf.drop(columns=columns_to_remove)
                logger.debug(
                    f"Removed {len(columns_to_remove)} metadata columns: {columns_to_remove}"
                )

        # Rename GEOLCODE and DESCRIPTION for geological mapping units
        if "GEOLCODE" in result_gdf.columns:
            result_gdf = result_gdf.rename(columns={"GEOLCODE": "GMU_CODE"})
            logger.debug("Renamed GEOLCODE to GMU_CODE")

        if "DESCRIPTION" in result_gdf.columns and "DESCRIPTION" in gdf.columns:
            # Only rename if it's from geological mapping unit context
            result_gdf = result_gdf.rename(columns={"DESCRIPTION": "GMU_DESC"})
            logger.debug("Renamed DESCRIPTION to GMU_DESC")

        # Add lookup descriptions for coded fields
        result_gdf = add_lookup_descriptions(result_gdf, self.gdb_path)

        # Decode coded domains for unco deposits
        if any(field in result_gdf.columns for field in FIELD_DOMAIN_MAPPING.keys()):
            result_gdf = decode_coded_domains(result_gdf, self.coded_domains)

        # Force integer conversion for all code fields
        integer_fields = [
            "GMU_CODE",
            "GEOLCODE",  # Main codes
            "chrono_top_code",
            "chrono_base_code",  # Chronology codes
            "litho_main_code",
            "litho_sec_code",
            "litho_ter_code",  # Lithology codes
            "litstrat_formation_bank_code",
            "correlation_code",
            "tecto_code",  # Other codes
            "TECTO",  # Legacy fields that should be integers
            "LITSTRAT_FORMATION_BANK",
            "CHRONO_TOP",
            "CHRONO_BASE",
            "LITHO_MAIN",
            "LITHO_SEC",
            "LITHO_TER",
            "CORRELATION",
        ]

        for field in integer_fields:
            if field in result_gdf.columns:
                try:
                    # Convert to numeric, handling NaN values
                    result_gdf[field] = pd.to_numeric(
                        result_gdf[field], errors="coerce"
                    ).astype("Int64")
                    logger.debug(f"Converted {field} to integer type")
                except Exception as e:
                    logger.warning(f"Could not convert {field} to integer: {e}")

        # Also handle any other code fields that end with _CODES
        code_fields = [col for col in result_gdf.columns if col.endswith("_CODES")]
        for field in code_fields:
            try:
                result_gdf[field] = pd.to_numeric(
                    result_gdf[field], errors="coerce"
                ).astype("Int64")
                logger.debug(f"Converted {field} to integer type")
            except Exception as e:
                logger.warning(f"Could not convert {field} to integer: {e}")

        return result_gdf

    def denormalize_all(
        self,
        remove_metadata: bool = False,
        split_layer: bool = True,
        tables: List[str] = [],
    ) -> Dict[str, gpd.GeoDataFrame]:
        """Denormalize all GeoCover tables with progress tracking."""

        tables_config = {
            "fossils": {
                "main_table": "GC_FOSSILS",
                "lookup_table": "GC_SYSTEM",
                "relationship_table": "GC_FOSS_SYSTEM_GC_SYSTEM",
                "description_alias": "SYSTEM_DESC",
                "method": "simple",
            },
            "exploit_points": {
                "main_table": "GC_EXPLOIT_GEOMAT_PT",
                "lookup_table": "GC_GEOL_MAPPING_UNIT",
                "relationship_table": "GC_EX_GEO_PNT_EXP_UNIT_GC_GMU",
                "description_alias": "MAPPING_UNIT_DESC",
                "method": "simple",
            },
            "exploit_polygons": {
                "main_table": "GC_EXPLOIT_GEOMAT_PLG",
                "lookup_table": "GC_GEOL_MAPPING_UNIT",
                "relationship_table": "GC_EX_GEO_PLG_EXP_UNIT_GC_GMU",
                "description_alias": "MAPPING_UNIT_DESC",
                "method": "simple",
            },
            "bedrock": {"method": "special"},
            "unco_deposits": {"method": "special"},
            "surfaces": {"main_table": "GC_SURFACES", "method": "copy"},
            "linear_objects": {"main_table": "GC_LINEAR_OBJECTS", "method": "copy"},
            "point_objects": {"main_table": "GC_POINT_OBJECTS", "method": "copy"},
        }

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            main_task = progress.add_task(
                "Denormalizing GeoCover tables...", total=len(tables_config)
            )

            filtered_config = {
                k: v
                for k, v in tables_config.items()
                if tables is None or not tables or k in tables
            }

            for table_name, config in filtered_config.items():
                task_id = progress.add_task(f"Processing {table_name}...", total=100)

                try:
                    gdf = None  # Initialize to None to catch issues

                    if config["method"] == "simple":
                        gdf = self.denormalize_simple_relationship(
                            config["main_table"],
                            config["lookup_table"],
                            config["relationship_table"],
                            config["description_alias"],
                            task_id,
                            progress,
                        )
                    elif table_name == "bedrock":
                        gdf = self.denormalize_bedrock(task_id, progress)
                    elif table_name == "unco_deposits":
                        gdf = self.denormalize_unco_deposits(task_id, progress)
                    elif config["method"] == "copy":
                        gdf = self.copy_table_as_is(
                            config["main_table"], task_id, progress
                        )
                    else:
                        raise ValueError(f"Unknown processing method for {table_name}")

                    if gdf is None:
                        raise ValueError(f"No data returned for {table_name}")

                    # Apply metadata cleaning and data type conversions
                    if progress and task_id:
                        progress.update(
                            task_id, description=f"Cleaning {table_name}..."
                        )

                    cleaned_gdf = self.clean_metadata_columns(gdf, remove_metadata)
                    results[table_name] = cleaned_gdf

                    # Log some basic info to verify correctness
                    logger.debug(
                        f"{table_name}: {len(cleaned_gdf)} features, {len(cleaned_gdf.columns)} columns"
                    )
                    if hasattr(cleaned_gdf, "geometry") and not cleaned_gdf.empty:
                        geom_types = cleaned_gdf.geometry.geom_type.value_counts()
                        logger.debug(f"{table_name} geometry types: {dict(geom_types)}")

                    progress.update(
                        task_id, completed=100, description=f"✅ {table_name} complete"
                    )

                except Exception as e:
                    logger.error(f"Failed to process {table_name}: {e}")
                    progress.update(
                        task_id, completed=100, description=f"❌ {table_name} failed"
                    )
                    # Don't add failed tables to results

                progress.advance(main_task)

        return results


def create_summary_table(results: Dict[str, gpd.GeoDataFrame]) -> Table:
    """Create a rich table summarizing the denormalization results."""

    table = Table(title="GeoCover Processing Summary")
    table.add_column("Table", style="cyan", no_wrap=True)
    table.add_column("Features", justify="right", style="green")
    table.add_column("Columns", justify="right", style="blue")
    table.add_column("Geometry Type", style="magenta")
    table.add_column("Processing", style="yellow")
    table.add_column("New Fields", style="dim")

    # Define which tables are denormalized vs copied
    denormalized_tables = {
        "fossils",
        "exploit_points",
        "exploit_polygons",
        "bedrock",
        "unco_deposits",
    }

    for table_name, gdf in results.items():
        # Debug: Log what we're actually processing
        logger.debug(
            f"Summary for {table_name}: {len(gdf)} features, {len(gdf.columns)} columns"
        )

        # Count features and columns
        feature_count = f"{len(gdf):,}"
        column_count = str(len(gdf.columns))

        # Get geometry type - be more specific about geometry types
        if hasattr(gdf, "geometry") and not gdf.empty and not gdf.geometry.isna().all():
            # Get the most common geometry type
            geom_types = gdf.geometry.geom_type.value_counts()
            geom_type = geom_types.index[0] if len(geom_types) > 0 else "Unknown"
            # If multiple types, show the dominant one
            if len(geom_types) > 1:
                geom_type += f" (+{len(geom_types) - 1} others)"
        else:
            geom_type = "None"

        # Determine processing type and new fields
        if table_name in denormalized_tables:
            processing_type = "Denormalized"
            # Find new description fields - be more specific
            desc_fields = [
                col
                for col in gdf.columns
                if ("DESC" in col and col != "DESCRIPTION")
                or ("CODES" in col)
                or (col.endswith("_DESC"))
                or (col.endswith("_CODES"))
            ]

            if desc_fields:
                new_fields = ", ".join(desc_fields[:2])  # Show first 2
                if len(desc_fields) > 2:
                    new_fields += f" (+{len(desc_fields) - 2})"
            else:
                new_fields = "None added"
        else:
            processing_type = "Copied as-is"
            new_fields = "N/A"

        table.add_row(
            table_name.replace("_", " ").title(),
            feature_count,
            column_count,
            geom_type,
            processing_type,
            new_fields,
        )

    return table


@click.command()
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output GPKG file path (default: auto-generated name)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--tables",
    "-t",
    multiple=True,
    type=click.Choice(
        [
            "fossils",
            "exploit_points",
            "exploit_polygons",
            "bedrock",
            "unco_deposits",
            "surfaces",
            "linear_objects",
            "point_objects",
        ]
    ),
    help="Process only specific tables (default: all)",
)
@click.option(
    "--remove-metadata",
    is_flag=True,
    help="Remove metadata columns (dates, origins, revisions, etc.)",
)
@click.option(
    "--split-layer",
    is_flag=True,
    help="Split layers on mapsheet border",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without writing output",
)
def denormalize_geocover(
    gdb_path: Path,
    output: Optional[Path],
    verbose: bool,
    tables: Tuple[str],
    remove_metadata: bool,
    split_layer: bool,
    overwrite: bool,
    dry_run: bool,
):
    """
    Denormalize GeoCover geodatabase tables with their lookup relationships.

    This tool reads a GeoCover FileGDB and creates denormalized versions of the main
    feature classes by joining them with their related lookup tables through
    junction/relationship tables. It also copies 3 additional tables as-is:
    GC_SURFACES, GC_LINEAR_OBJECTS, and GC_POINT_OBJECTS.

    Total output: 8 layers in a single GPKG file.

    GDB_PATH: Path to the GeoCover FileGDB (.gdb directory)
    """

    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Welcome message
    console.print(
        Panel.fit(
            "[bold blue]GeoCover Denormalization Tool[/bold blue]\n"
            "[dim]Professional geodatabase denormalization with Rich & Loguru[/dim]",
            border_style="blue",
        )
    )

    # Validate input
    if not gdb_path.exists():
        console.print(f"[red]Error: FileGDB not found: {gdb_path}[/red]")
        raise click.Abort()

    console.print(f"[green]📂 Processing:[/green] {gdb_path}")

    # Initialize denormalizer
    try:
        denormalizer = GeoCoverDenormalizer(gdb_path, verbose=verbose)
    except Exception as e:
        console.print(f"[red]Error initializing denormalizer: {e}[/red]")
        raise click.Abort()

    # Process tables
    try:
        if tables:
            console.print(
                f"[yellow]Processing selected tables:[/yellow] {', '.join(tables)}"
            )
            results = denormalizer.denormalize_all(
                remove_metadata=remove_metadata, tables=tables
            )
            # Filter results to selected tables
            results = {k: v for k, v in results.items() if k in tables}
        else:
            console.print("[yellow]Processing all 8 tables...[/yellow]")
            results = denormalizer.denormalize_all(remove_metadata=remove_metadata)

        if remove_metadata:
            console.print("[blue]🧹 Metadata columns will be removed[/blue]")

        if not results:
            console.print("[red]No tables were successfully processed![/red]")
            raise click.Abort()

        if split_layer:
            split_layer_path = files("gcover.data").joinpath(
                "administrative_zones.gpkg"
            )
            if not split_layer_path or not Path(split_layer_path).exists():
                raise FileNotFoundError(
                    f"[ERROR] GeoPackage not found: {split_layer_path}"
                )

            split_layer_name = "mapsheets_sources_only"

            try:
                mapsheets_gdf = gpd.read_file(split_layer_path, layer=split_layer_name)
            except ValueError as ve:
                raise ValueError(
                    f"[ERROR] Layer '{split_layer_name}' not found in {split_layer_path}: {ve}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[ERROR] Failed to read GeoPackage '{split_layer_path}': {e}"
                )

            for name, input_gdf in results.items():
                console.print(
                    f"[blue]🧹 Table {name} will be split along mapsheets[/blue]"
                )
                original_feat_nb = len(input_gdf)

                console.print(f"   Input : {original_feat_nb} features")
                console.print(f"   Mapsheets: {len(mapsheets_gdf)}")

                try:
                    if not (input_gdf.geom_type == "Point").all():
                        splitted_gdf = split_features_by_mapsheets(
                            input_gdf=input_gdf,
                            mapsheets_gdf=mapsheets_gdf,
                            keep_attributes=True,
                            progress_bar=True,
                            area_threshold=0.1,
                        )
                        splitted_gdf.to_file("splitted.gpkg")
                        results[name] = splitted_gdf
                        splitted_feat_nb = len(splitted_gdf)

                        diff = splitted_feat_nb - original_feat_nb
                        sign = "+" if diff > 0 else ""
                        console.print(
                            f"[dim]  After split: {splitted_feat_nb} features ([bold]{sign}{diff}[/bold])[/dim]"
                        )

                    else:
                        console.print("    Ignoring POINT geometries")

                except Exception as e:
                    logger.error(e)

        # Debug: Show what we actually got
        if verbose:
            console.print("\n[dim]Debug: Results summary[/dim]")
            for name, gdf in results.items():
                console.print(
                    f"[dim]  {name}: {len(gdf)} features, {len(gdf.columns)} columns[/dim]"
                )

        # Show summary
        summary_table = create_summary_table(results)
        console.print("\n")
        console.print(summary_table)

        if dry_run:
            console.print("\n[yellow]🔍 Dry run complete - no files written[/yellow]")
            return

        # Generate output filename if not provided
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rc_version = "RC2" if "2030-12-31" in str(gdb_path) else "RC1"
            output = Path(f"geocover_denormalized_{rc_version}_{timestamp}.gpkg")

        # Write results to single GPKG
        console.print(f"\n[green]💾 Writing results to:[/green] {output}")
        kwargs = {}
        mode = "a"
        if overwrite:
            kwargs["OVERWRITE"] = "YES"
            mode = "w"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            write_task = progress.add_task("Writing GPKG layers...", total=len(results))

            for table_name, gdf in results.items():
                progress.update(write_task, description=f"Writing {table_name}...")

                try:
                    # Use table_name as layer name in GPKG
                    if output.exists():  # and table_name != list(results.keys())[0]:
                        # Append to existing GPKG
                        # gdf.to_file(output, layer=table_name, driver="GPKG", mode="a")
                        gdf.to_file(
                            output,
                            layer=table_name,
                            driver="GPKG",
                            engine="fiona",
                            mode=mode,
                            **kwargs,
                        )
                    else:
                        # Create new GPKG or write first layer
                        gdf.to_file(output, layer=table_name, driver="GPKG")

                    logger.info(f"Wrote layer '{table_name}' with {len(gdf)} features")

                except Exception as e:
                    logger.error(f"Failed to write {table_name}: {e}")
                    console.print(f"[red]❌ Failed to write {table_name}: {e}[/red]")

                progress.advance(write_task)

        # Final success message
        total_features = sum(len(gdf) for gdf in results.values())
        console.print(
            Panel.fit(
                f"[bold green]✅ Success![/bold green]\n\n"
                f"📊 Processed {len(results)} tables with {total_features:,} total features\n"
                f"📁 Output: {output}\n"
                f"💾 Size: {output.stat().st_size / (1024 * 1024):.1f} MB\n"
                f"🏷️ Layers: {', '.join(results.keys())}",
                border_style="green",
                title="Denormalization Complete",
            )
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    denormalize_geocover()
