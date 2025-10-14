#!/usr/bin/env python

"""
Create administrative_zones.gpkg from 4 standardized source files.

Simplified script that handles the real data sources with their actual attributes.

2025-09-05  Added `SOURCE_RC` sources for BKP (data for publication)
2025-10-14  Added sources `SOURCE_QA`  for QA (before publication)
"""

import os
import warnings
from datetime import datetime as dt
from importlib.resources import files
from pathlib import Path

import click
import fiona
import geopandas as gpd
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CRS = "EPSG:2056"


def load_lots(lots_path: Path) -> gpd.GeoDataFrame:
    """Load lots from either shapefile or geojson."""
    logger.info(f"Loading lots from {lots_path}")

    try:
        lots_gdf = gpd.read_file(lots_path)
        lots_gdf = lots_gdf.to_crs(DEFAULT_CRS)

        # Ensure we have the essential columns
        excluded = {"Id", "LOT_NR"}

        if not excluded.intersection(lots_gdf.columns):
            raise ValueError("Lots file missing '{excluded}}' column")

        # Keep only essential columns + any extra that exist
        keep_cols = ["Id", "geometry"]
        for col in ["Resp", "Status"]:
            if col in lots_gdf.columns:
                keep_cols.append(col)

        lots_clean = lots_gdf[keep_cols].copy()
        lots_clean = lots_clean.rename(columns={"Id": "LOT_NR"})

        logger.info(
            f"Loaded {len(lots_clean)} lots with columns: {list(lots_clean.columns)}"
        )
        return lots_clean

    except Exception as e:
        logger.error(f"Failed to load lots: {e}")
        raise


def load_work_units(wu_path: Path) -> gpd.GeoDataFrame:
    """Load work units from geojson."""
    logger.info(f"Loading work units from {wu_path}")

    try:
        wu_gdf = gpd.read_file(wu_path)
        wu_gdf = wu_gdf.to_crs(DEFAULT_CRS)

        # Ensure we have the essential columns
        if "NAME" not in wu_gdf.columns:
            raise ValueError("WU file missing 'NAME' column")

        # Keep only essential columns + any extra that exist
        keep_cols = ["NAME", "geometry"]
        for col in [
            "WU_ID",
            "START_DATE",
            "LAST_OPENED_BY",
            "LAST_OPENED",
            "PERIMETER_OID",
        ]:
            if col in wu_gdf.columns:
                keep_cols.append(col)

        wu_clean = wu_gdf[keep_cols].copy()

        logger.info(
            f"Loaded {len(wu_clean)} work units with columns: {list(wu_clean.columns)}"
        )
        return wu_clean

    except Exception as e:
        logger.error(f"Failed to load work units: {e}")
        raise


def load_mapsheets(mapsheets_path: Path) -> gpd.GeoDataFrame:
    """Load mapsheets from geojson."""
    logger.info(f"Loading mapsheets from {mapsheets_path}")

    try:
        mapsheets_gdf = gpd.read_file(mapsheets_path)
        mapsheets_gdf = mapsheets_gdf.to_crs(DEFAULT_CRS)

        # Check for essential columns
        required_cols = ["MSH_MAP_NBR", "MSH_TOPO_NR"]
        missing_cols = [
            col for col in required_cols if col not in mapsheets_gdf.columns
        ]
        if missing_cols:
            raise ValueError(f"Mapsheets file missing required columns: {missing_cols}")

        # Keep all MSH_ columns that exist
        keep_cols = ["geometry"]
        for col in mapsheets_gdf.columns:
            if col.startswith("MSH_"):
                keep_cols.append(col)

        mapsheets_clean = mapsheets_gdf[keep_cols].copy()

        logger.info(
            f"Loaded {len(mapsheets_clean)} mapsheets with columns: {list(mapsheets_clean.columns)}"
        )
        return mapsheets_clean

    except Exception as e:
        logger.error(f"Failed to load mapsheets: {e}")
        raise


def load_sources(sources_path: Path) -> pd.DataFrame:
    """Load sources from Excel file."""
    logger.info(f"Loading sources from {sources_path}")

    try:
        # Try different sheet names
        sheet_names = ["Feuil1", "Sheet1", "Feuille1", 0]  # Common variations
        sources_df = None

        for sheet in sheet_names:
            try:
                sources_df = pd.read_excel(sources_path, sheet_name=sheet)
                logger.info(f"Successfully read sheet: {sheet}")
                break
            except:
                continue

        if sources_df is None:
            # Fallback: read first sheet
            sources_df = pd.read_excel(sources_path)
            logger.info("Used default sheet")

        # Check for essential columns
        if "MSH_MAP_NBR" not in sources_df.columns:
            raise ValueError("Sources file missing 'MSH_MAP_NBR' column")
        if "BKP" not in sources_df.columns:
            raise ValueError("Sources file missing 'BKP' column")
        else:
            sources_df = sources_df.rename(
                columns={"BKP": "SOURCE_RC", "QA": "SOURCE_QA"}
            )

        # Keep only essential columns + any extra that exist
        keep_cols = ["MSH_MAP_NBR"]
        for col in [
            "MSH_MAP_TITLE",
            "MSH_TOPO_NR",
            "SOURCE_RC",
            "SOURCE_QA",
            "Version",
            "Notice",
        ]:
            if col in sources_df.columns:
                keep_cols.append(col)

        sources_clean = sources_df[keep_cols].copy()

        logger.info(
            f"Loaded {len(sources_clean)} source records with columns: {list(sources_clean.columns)}"
        )
        return sources_clean

    except Exception as e:
        logger.error(f"Failed to load sources: {e}")
        raise


def merge_sources(
    lot_gdf: gpd.GeoDataFrame, sources_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Merges a GeoDataFrame of land lots with a DataFrame of source information
    based on a common map number identifier.

    The function drops 'MSH_MAP_TITLE' and 'MSH_TOPO_NR' columns from the
    sources DataFrame before performing a left merge. It ensures that the
    necessary columns exist in both input DataFrames and handles potential
    issues gracefully using logging.

    Args:
        lot_gdf (gpd.GeoDataFrame):
            A GeoDataFrame containing land lot geometries and attributes,
            expected to have an 'MSH_MAP_NBR' column.
        sources_df (pd.DataFrame):
            A DataFrame containing additional source information, expected
            to have 'MSH_MAP_NBR', 'MSH_MAP_TITLE', and 'MSH_TOPO_NR' columns.

    Returns:
        gpd.GeoDataFrame:
            A new GeoDataFrame resulting from the left merge. It will contain
            all rows from `lot_gdf` and matched columns from `sources_df`.
            Returns an empty GeoDataFrame if inputs are invalid or merge key is missing.

    Raises:
        ValueError:
            If essential columns are missing in the input DataFrames.
    """
    required_lot_cols = ["MSH_MAP_NBR"]
    required_source_cols = ["MSH_MAP_TITLE", "MSH_TOPO_NR"]

    # --- Input Validation ---
    if not isinstance(lot_gdf, gpd.GeoDataFrame):
        logger.error(
            f"Invalid input: 'lot_gdf' must be a GeoDataFrame, got {type(lot_gdf)}."
        )
        return gpd.GeoDataFrame()
    if not isinstance(sources_df, pd.DataFrame):
        logger.error(
            f"Invalid input: 'sources_df' must be a pandas DataFrame, got {type(sources_df)}."
        )
        return gpd.GeoDataFrame()

    # Check for required columns in lot_gdf
    missing_lot_cols = [col for col in required_lot_cols if col not in lot_gdf.columns]
    if missing_lot_cols:
        logger.error(
            f"Missing required columns in 'lot_gdf': {missing_lot_cols}. Cannot perform merge."
        )
        return gpd.GeoDataFrame(geometry=[])
    # Check for required columns in sources_df
    missing_source_cols = [
        col for col in required_source_cols if col not in sources_df.columns
    ]
    if missing_source_cols:
        logger.warning(
            f"Warning: Missing expected columns in 'sources_df': {missing_source_cols}. "
            f"These columns were expected for dropping/merging. "
            f"The merge will proceed but might not be as intended if 'MSH_MAP_NBR' is missing."
        )
        if "MSH_MAP_NBR" not in sources_df.columns:
            logger.error(
                "Critical error: 'MSH_MAP_NBR' is missing in 'sources_df'. Merge cannot proceed."
            )
            return gpd.GeoDataFrame(geometry=[])

    # Rn
    """if 'MSH_TOPO_NR' in sources_df.columns:
      sources_df = sources_df.rename(columns={"MSH_TOPO_NR": "MSH_TOPO_NBR"})
      print(sources_df.head())
    """

    # --- Column Dropping ---
    columns_to_drop = ["MSH_MAP_TITLE", "MSH_TOPO_NR"]
    columns_to_drop = ["MSH_MAP_TITLE", "MSH_MAP_NBR"]
    existing_cols_to_drop = [
        col for col in columns_to_drop if col in sources_df.columns
    ]
    if existing_cols_to_drop:
        sources_df_processed = sources_df.drop(columns=existing_cols_to_drop)
        logger.info(f"Dropped columns {existing_cols_to_drop} from 'sources_df'.")
    else:
        sources_df_processed = (
            sources_df.copy()
        )  # Ensure we work on a copy if no columns are dropped
        logger.info("No specified columns to drop were found in 'sources_df'.")

    # --- Merge Operation ---
    merge_key = "MSH_TOPO_NR"  # was MSH_MAP_NBR
    if (
        merge_key not in lot_gdf.columns
        or merge_key not in sources_df_processed.columns
    ):
        logger.error(
            f"Merge key '{merge_key}' is missing in one or both DataFrames. Cannot perform merge."
        )
        return gpd.GeoDataFrame(geometry=[])

    try:
        merged_gdf = lot_gdf.merge(
            sources_df_processed,
            left_on=merge_key,
            right_on=merge_key,
            how="left",  # Keeps all rows from the left GeoDataFrame (lot_gdf)
        )
        logger.info(f"Successfully merged GeoDataFrames on '{merge_key}'.")
        logger.info(f"Resulting GeoDataFrame shape: {merged_gdf.shape}")
        return merged_gdf
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during the merge: {e}",
            exc_info=True,
        )
        return gpd.GeoDataFrame(
            geometry=[]
        )  # Return an empty GeoDataFrame on critical failure


def spatial_join_safe(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    right_cols: list,
    operation_name: str,
) -> gpd.GeoDataFrame:
    """
    Perform safe spatial join with error handling.

    Args:
        left_gdf: Left GeoDataFrame (target)
        right_gdf: Right GeoDataFrame (source for join)
        right_cols: Columns to keep from right GDF
        operation_name: Name for logging
    """
    logger.info(f"Performing spatial join: {operation_name}")

    if left_gdf.empty or right_gdf.empty:
        logger.warning(f"Empty GeoDataFrame for {operation_name}, skipping")
        return left_gdf.copy()

    # Ensure same CRS
    if left_gdf.crs != right_gdf.crs:
        right_gdf = right_gdf.to_crs(left_gdf.crs)

    # Create small inner buffer on right geometries to avoid edge effects
    right_buffered = right_gdf.copy()
    try:
        right_buffered["geometry"] = right_gdf.geometry.buffer(-10)  # 10m inner buffer

        # Remove invalid/empty geometries
        right_buffered = right_buffered[right_buffered.geometry.is_valid]
        right_buffered = right_buffered[~right_buffered.geometry.is_empty]

        if right_buffered.empty:
            logger.warning(
                f"All geometries became empty after buffering for {operation_name}"
            )
            return left_gdf.copy()

    except Exception as e:
        logger.warning(
            f"Buffering failed for {operation_name}, using original geometries: {e}"
        )
        right_buffered = right_gdf.copy()

    # Filter columns that actually exist
    available_cols = ["geometry"] + [
        col for col in right_cols if col in right_buffered.columns
    ]

    try:
        # Spatial join
        result = gpd.sjoin(
            left_gdf, right_buffered[available_cols], how="left", predicate="intersects"
        )

        # Clean up
        result = result.drop(columns=["index_right"], errors="ignore")

        # Log results
        before_count = len(left_gdf)
        after_count = len(result)
        joined_count = result[right_cols[0]].notna().sum() if right_cols else 0

        logger.info(
            f"{operation_name}: {before_count} → {after_count} features, {joined_count} successfully joined"
        )

        return result

    except Exception as e:
        logger.error(f"Spatial join failed for {operation_name}: {e}")
        return left_gdf.copy()


def clean_wu(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Cleans the WU (Work Unit) GeoDataFrame by removing a predefined list of columns.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame to be cleaned.

    Returns:
        gpd.GeoDataFrame: The cleaned GeoDataFrame with specified columns removed.
    """
    logger.info("Running clean_wu function.")

    # List of columns to remove
    columns_to_remove = [
        "START_DATE",
        "LAST_OPENED_BY",
        "LAST_OPENED",
        "PERIMETER_OID",
        "WU_ID",
        "WU_STATE",
        "RC_ID",
        "RC_NAME",
        "RC_STATE",
        "RC_PLANNED_RELEASE_DATE",
        "PROJ_ID",
        "REVISION_MONTH",
        "CREATION_DAY",
        "REVISION_DAY",
        "CREATION_MONTH",
        "REVISION_YEAR",
        "CREATION_YEAR",
        "REVISION_DATE",
        "CREATION_DATE",
        "LAST_UPDATE",
        "CREATED_USER",
        "LAST_USER",
        "OPERATOR",
        "DATEOFCREATION",
        "DATEOFCHANGE",
        "CREATION_YEAR",
        "CREATION_MONTH",
        "REVISION_YEAR",
        "REVISION_MONTH",
        "REASONFORCHANGE",
        "OBJECTORIGIN",
        "OBJECTORIGIN_YEAR",
        "OBJECTORIGIN_MONTH",
        "KIND",
        "RC_ID",  # Duplicate, but kept as in original list
        "WU_ID",  # Duplicate, but kept as in original list
        "RC_ID_CREATION",
        "WU_ID_CREATION",
        "REVISION_QUALITY",
        "ORIGINAL_ORIGIN",
        "INTEGRATION_OBJECT_UUID",
        "SHAPE.AREA",
        "SHAPE.LEN",
        "MORE_INFO",
    ]

    # Remove columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_remove if col in gdf.columns]
    if columns_to_drop:
        gdf = gdf.drop(columns=columns_to_drop)
        logger.info(f"Columns removed: {columns_to_drop}")
    else:
        logger.info("No specified columns to remove were found in the GeoDataFrame.")

    logger.info(f"Remaining columns after clean_wu: {list(gdf.columns)}")

    return gdf


@click.command(context_settings={"show_default": True})
@click.option(
    "--output",
    "-o",
    "output_path",
    default=str(files("gcover.data").joinpath("administrative_zones.gpkg")),
    type=click.Path(path_type=Path),
    help="Output GPKG file path",
)
@click.option(
    "--lots-file",
    required=True,
    default=str(files("gcover.data").joinpath("lots.geojson")),
    type=click.Path(exists=True, path_type=Path),
    help="Path to lots file (shapefile or geojson)",
)
@click.option(
    "--wu-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to work units geojson file",
    default=str(files("gcover.data").joinpath("WU.json")),
)
@click.option(
    "--mapsheets-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to mapsheets geojson file",
    default=str(files("gcover.data").joinpath("mapsheets.geojson")),
)
@click.option(
    "--sources-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to sources Excel file",
    default=str(files("gcover.data").joinpath("GC_Sources_QA.xlsx")),
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def create_administrative_zones(
    output_path: Path,
    lots_file: Path,
    wu_file: Path,
    mapsheets_file: Path,
    sources_file: Path,
    overwrite: bool,
    verbose: bool,
):
    """
    Create administrative_zones.gpkg for QA analysis.
    
    This script processes the 4 standardized source files into a single GPKG
    with separate layers for QA analysis. It handles the real data structure
    and attributes without assuming perfect standardization.
    
    Example:
        python scripts/create_administrative_zones.py \\
            --lots-file data/lots.geojson \\
            --wu-file data/WU.geojson \\
            --mapsheets-file data/mapsheets.geojson \\
            --sources-file data/GC_Sources_PA.xlsx \\
            --output gcover/data/administrative_zones.gpkg \\
            --overwrite
    """
    if verbose:
        logger.add(lambda msg: click.echo(msg, err=True), level="DEBUG")

    # Check if output exists

    if output_path.exists() and not overwrite:
        click.echo(f"❌ Output file exists: {output_path}")
        click.echo("   Use --overwrite to replace it")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.exists():
        output_path.unlink()
        logger.info(f"Removed existing file: {output_path}")

    try:
        # 1. Load all source data
        click.echo("📁 Loading source data...")

        lots_gdf = load_lots(lots_file)
        wu_gdf = load_work_units(wu_file)
        mapsheets_gdf = load_mapsheets(mapsheets_file)
        sources_df = load_sources(sources_file)

        # 2. Join mapsheets with sources (attribute join on MSH_MAP_NBR)
        click.echo("🔗 Joining mapsheets with sources...")

        mapsheets_with_sources = mapsheets_gdf.merge(
            sources_df, on="MSH_MAP_NBR", how="left"
        )

        # TODO alternate

        mapsheets_with_sources = clean_wu(
            merge_sources(mapsheets_gdf, sources_df.copy())
        )  # Pass copy to avoid modifying original
        if mapsheets_with_sources.empty:
            logger.error(
                "merge_sources returned an empty GeoDataFrame. Skipping subsequent operations."
            )
            return

        # Log join results
        before_join = len(mapsheets_gdf)
        after_join = len(mapsheets_with_sources)
        with_source = (
            mapsheets_with_sources["SOURCE_RC"].notna().sum()
            if "SOURCE_RC" in mapsheets_with_sources.columns
            else 0
        )

        logger.info(
            f"Mapsheets-Sources join: {before_join} → {after_join} features, {with_source} with sources"
        )

        # 3. Spatial joins with lots and work units
        click.echo("🗺️  Performing spatial joins...")

        # Join with lots
        mapsheets_with_lots = spatial_join_safe(
            mapsheets_with_sources,
            lots_gdf,
            ["LOT_NR", "Resp", "Status"],
            "mapsheets-lots",
        )

        # Join with work units
        mapsheets_complete = spatial_join_safe(
            mapsheets_with_lots, wu_gdf, ["NAME", "WU_ID"], "mapsheets-work_units"
        )

        # 4. Write to GPKG
        click.echo(f"💾 Writing to {output_path}")

        # Main layer: mapsheets with all attributes
        # TODO: only source is OK?
        mapsheets_with_sources.to_file(
            output_path, layer="mapsheets_sources_only", driver="GPKG"
        )
        mapsheets_complete.to_file(
            output_path, layer="mapsheets_with_sources", driver="GPKG"
        )
        logger.info(
            f"✓ Written layer: mapsheets_with_sources ({len(mapsheets_complete)} features)"
        )

        # Individual zone layers
        lots_gdf.to_file(output_path, layer="lots", driver="GPKG", mode="a")
        logger.info(f"✓ Written layer: lots ({len(lots_gdf)} features)")

        wu_gdf.to_file(output_path, layer="work_units", driver="GPKG", mode="a")
        logger.info(f"✓ Written layer: work_units ({len(wu_gdf)} features)")

        # Base mapsheets layer (for reference)
        mapsheets_gdf.to_file(output_path, layer="mapsheets", driver="GPKG", mode="a")
        logger.info(f"✓ Written layer: mapsheets ({len(mapsheets_gdf)} features)")

        # 5. Summary and validation
        click.echo(f"✅ Administrative zones created successfully!")
        click.echo(f"   📁 File: {output_path}")
        click.echo(f"   📊 Layers: mapsheets_with_sources, lots, work_units, mapsheets")

        # Write the docstring to a file
        now = dt.now().strftime("%Y-%m-%d %H:%M:%S")

        layers = fiona.listlayers(output_path)
        layer_string = "\n * ".join(layers)
        docstring = (
            f"""{__doc__ or ""}\nLayer list:\n * {layer_string}\n\nGenerated on {now}"""
        )
        output_without_ext = output_path.with_suffix("")
        with open(output_without_ext.with_suffix(".README"), "w") as f:
            f.write(docstring)

        # Validation summary
        if "SOURCE_RC" in mapsheets_complete.columns:
            rc1_count = (mapsheets_complete["SOURCE_RC"] == "RC1").sum()
            rc2_count = (mapsheets_complete["SOURCE_RC"] == "RC2").sum()
            no_source_count = mapsheets_complete["SOURCE_RC"].isna().sum()

            click.echo(f"   🔵 RC1 mapsheets: {rc1_count}")
            click.echo(f"   🟢 RC2 mapsheets: {rc2_count}")
            click.echo(f"   ⚪ No source: {no_source_count}")

            if no_source_count > 0:
                click.echo(
                    f"   ⚠️  {no_source_count} mapsheets have no source assignment"
                )

        if "LOT_NR" in mapsheets_complete.columns:
            with_lot = mapsheets_complete["LOT_NR"].notna().sum()
            click.echo(f"   📦 Mapsheets with lot assignment: {with_lot}")

        if "NAME" in mapsheets_complete.columns:
            with_wu = mapsheets_complete["NAME"].notna().sum()
            click.echo(f"   👥 Mapsheets with work unit assignment: {with_wu}")

        # Show available columns
        click.echo(
            f"   📋 Available attributes: {', '.join([col for col in mapsheets_complete.columns if col != 'geometry'])}"
        )

    except Exception as e:
        logger.error(f"Failed to create administrative zones: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    create_administrative_zones()
