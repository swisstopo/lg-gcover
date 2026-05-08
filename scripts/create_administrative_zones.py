#!/usr/bin/env python

"""
Create administrative zones from 4 standardized source files.

Inputs
------
- lots.geojson         — lot polygons with LOT_NR, Resp, Status
- WU.json              — work-unit polygons with NAME (→ WU_NAME)
- mapsheets.geojson    — 1:25 000 mapsheet polygons with MSH_* attributes
- GC_Sources_PA.xlsx   — per-mapsheet source assignments (BKP → SOURCE_RC)
                         and document availability flags (BER, ERL)

Output formats (--format, repeatable)
--------------------------------------
gpkg        Multi-layer GeoPackage at the path given by --output (default).
filegdb     ESRI FileGDB at the same base path with a .gdb extension.
geojson     One GeoJSON file per layer in {output_stem}/ (EPSG:2056).
parquet     One GeoParquet file per layer in {output_stem}/ (EPSG:2056).
flatgeobuf  One FlatGeobuffer file per layer {output_stem}/ (EPSG:4326).

All formats can be produced in a single run:
  create-administrative-zones -f gpkg -f filegdb -f geojson -f parquet --overwrite

Output layers
-------------
mapsheets_sources_only  Base mapsheets joined with SOURCE_RC, Version,
                        and document links (ber_link, erl_link).
mapsheets_with_sources  Same, further enriched with LOT_NR, WU_NAME (spatial
                        join; values from multiple lots/WUs are pipe-separated).
borders_50m             Single polygon: full mapped area minus a 50 m buffer
                        around all internal mapsheet borders.  Used to identify
                        features well away from any join line.
border_segments         Classified border lines between adjacent mapsheets:
                          RC1-RC1  both neighbours from RC1 (tolerant)
                          RC1-RC2  one RC1, one RC2 (tolerant)
                          RC2-RC2  both from RC2 (strict)
                          land     external perimeter of the mapped area (strict)
tolerance_zones_50m     Single polygon: 50 m buffer around tolerant borders
                        (RC1-RC1 and RC1-RC2).  QA errors intersecting this zone
                        can be ignored.
strict_zones_50m        Full mapped area minus tolerance_zones_50m.  Only QA
                        errors inside this zone need investigation.
lots                    Raw lot polygons.
work_units              Raw work-unit polygons.
mapsheets               Raw mapsheet polygons (no source join).
qa_rand_gc              Raw QA_Rand_GC.gdb layer (first layer), reprojected to
                        EPSG:2056; all column names lowercased.  Only present
                        when --qa-rand-gc is supplied.
qa_rand_gc_buffer_50m   Single polygon: full mapped area minus a 50 m buffer
                        around active border features (rand != '1'; rand = '1'
                        means terminated and is excluded).  Used to
                        select features that lie well away from active borders.
                        Only present when --qa-rand-gc is supplied.

Changelog
---------
2025-09-05  Added SOURCE_RC (BKP) per mapsheet for publication tracking
2025-10-13  Added SOURCE_QA per mapsheet for pre-publication QA
2025-10-14  Added borders_100m layer
2026-01-28  Downgraded missing SOURCE_QA to warning only
2026-03-13  Added BER/ERL flags and ber_link/erl_link URL columns
2026-03-15  Removed SOURCE_QA column entirely
2026-03-24  New WU.json format; removed Campodolcino sheet (merged into Val Bregaglia); provisional PA for R17
2026-04-25  Added border_segments, tolerance_zones_Xm, strict_zones_Xm for QA error border filtering
2026-04-26  Improved docstring; added HTTP check for ber_link/erl_link URLs
2026-05-05  Added GeoJSON and GeoParquet output formats via --format option
2026-05-07  Added qa_rand_gc and qa_rand_gc_buffer_50m layers via --qa-rand-gc

"""

import os
import urllib.error
import urllib.request
import warnings
from datetime import datetime as dt
from importlib.resources import files
from pathlib import Path

import click
import fiona
import geopandas as gpd
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from shapely import STRtree
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CRS = "EPSG:2056"

ERLAUETERUNG_LINK = "https://data.geo.admin.ch/ch.swisstopo.geologie-geologischer_atlas/erlaeuterungen/GA25-ERL-"
BERICHT_LINK =      "https://data.geo.admin.ch/ch.swisstopo.geologie-geocover/berichte/BER_"

console = Console()


def _check_pdf_urls(df: pd.DataFrame, url_cols: list[str], timeout: int = 5) -> None:
    """Issue a HEAD request for every non-empty URL and warn on HTTP errors."""
    for col in url_cols:
        if col not in df.columns:
            continue
        urls = df[col].dropna()
        urls = urls[urls != ""]
        for url in urls:
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=timeout):
                    pass
            except urllib.error.HTTPError as exc:
                logger.warning(f"PDF not found ({exc.code}): {url}")
            except Exception as exc:
                logger.warning(f"PDF unreachable ({exc}): {url}")


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

        wu_gdf = wu_gdf.rename(columns={"NAME": "WU_NAME"})

        # Keep only essential columns + any extra that exist
        keep_cols = ["WU_NAME", "geometry"]
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
            "Remark",
            "BER",
            "ERL"
        ]:
            if col in sources_df.columns:
                keep_cols.append(col)

        sources_clean = sources_df[keep_cols].copy()

        # Add links
        sources_clean["erl_link"] = sources_clean.apply(
          lambda row: f"{ERLAUETERUNG_LINK}{row['MSH_MAP_NBR']}.pdf" if row["ERL"] == "y" else "",
          axis = 1
        )
        sources_clean["ber_link"] = sources_clean.apply(
          lambda row: f"{BERICHT_LINK}{row['MSH_MAP_NBR']}.pdf" if row["BER"] == "y" else "",
          axis = 1
        )

        _check_pdf_urls(sources_clean, ["erl_link", "ber_link"])

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


def border_mapsheet(
    mapsheets: gpd.GeoDataFrame, buffer_distance: float = 100
) -> gpd.GeoDataFrame:
    """
    Retourne l'aire totale des mapsheets MOINS un buffer autour de toutes les bordures.

    Cela permet d'identifier les zones "intérieures" éloignées des bordures/joints
    entre mapsheets.

    Parameters
    ----------
    mapsheets : gpd.GeoDataFrame
        GeoDataFrame contenant les polygones des mapsheets
    buffer_distance : float, default 100
        Distance du buffer en unités du CRS (généralement mètres)

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame avec une seule géométrie représentant la zone intérieure
    """
    # 1. Dissoudre tous les polygons en une seule géométrie (l'aire totale)
    total_area = mapsheets.unary_union

    # 2. Extraire les bordures de CHAQUE mapsheet (pas de total_area!)
    # Cela inclut les bordures internes (joints entre mapsheets)
    borders = mapsheets.boundary.unary_union

    # 3. Créer un buffer autour des bordures
    border_buffer = borders.buffer(buffer_distance)

    # 4. Soustraire le buffer de l'aire totale
    inner_area = total_area.difference(border_buffer)

    # 5. Créer un GDF avec cette géométrie
    inner_area_gdf = gpd.GeoDataFrame(geometry=[inner_area], crs=mapsheets.crs)

    return inner_area_gdf


def classify_border_segments(
    mapsheets_with_sources: gpd.GeoDataFrame,
    buffer_distance: float = 100,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Find adjacent mapsheet pairs, classify shared borders by RC type.

    Returns (border_segments_gdf, tolerance_zones_gdf).
    Tolerant borders (RC1-RC1, RC1-RC2) are buffered to form the tolerance zones.
    Land border = external perimeter; classified strict (not buffered).
    """
    msh = mapsheets_with_sources.reset_index(drop=True)
    geoms = list(msh.geometry)
    tree = STRtree(geoms)

    border_rows = []
    seen_pairs: set[tuple[int, int]] = set()

    for i, geom_i in enumerate(geoms):
        candidates = tree.query(geom_i)
        for j in candidates:
            if j <= i:
                continue
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))

            try:
                shared = geom_i.intersection(geoms[j])
            except Exception:
                continue

            if shared.is_empty:
                continue
            if shared.geom_type in ("Point", "MultiPoint"):
                continue
            if shared.geom_type == "GeometryCollection":
                lines = [g for g in shared.geoms if "Line" in g.geom_type]
                if not lines:
                    continue
                shared = unary_union(lines)
            if "Line" not in shared.geom_type:
                continue

            left_rc = msh.iloc[i].get("SOURCE_RC") if "SOURCE_RC" in msh.columns else None
            right_rc = msh.iloc[j].get("SOURCE_RC") if "SOURCE_RC" in msh.columns else None

            left_rc_str = str(left_rc) if pd.notna(left_rc) else "unknown"
            right_rc_str = str(right_rc) if pd.notna(right_rc) else "unknown"
            pair_rcs = tuple(sorted([left_rc_str, right_rc_str]))

            if pair_rcs == ("RC1", "RC1"):
                border_type = "RC1-RC1"
            elif pair_rcs == ("RC2", "RC2"):
                border_type = "RC2-RC2"
            elif set(pair_rcs) == {"RC1", "RC2"}:
                border_type = "RC1-RC2"
            else:
                border_type = "-".join(pair_rcs)

            border_rows.append({
                "geometry": shared,
                "border_type": border_type,
                "left_msh": msh.iloc[i].get("MSH_MAP_NBR"),
                "right_msh": msh.iloc[j].get("MSH_MAP_NBR"),
                "left_rc": left_rc_str,
                "right_rc": right_rc_str,
            })

    if border_rows:
        border_segments_gdf = gpd.GeoDataFrame(border_rows, crs=msh.crs)
    else:
        border_segments_gdf = gpd.GeoDataFrame(
            columns=["geometry", "border_type", "left_msh", "right_msh", "left_rc", "right_rc"],
            crs=msh.crs,
        )

    # Land border = outer perimeter of the union of all mapsheets.
    # Internal shared segments lie in the interior of the union and are absent here.
    land_geom = unary_union(geoms).boundary
    if not land_geom.is_empty:
        land_row = gpd.GeoDataFrame([{
            "geometry": land_geom,
            "border_type": "land",
            "left_msh": None,
            "right_msh": None,
            "left_rc": None,
            "right_rc": None,
        }], crs=msh.crs)
        border_segments_gdf = pd.concat([border_segments_gdf, land_row], ignore_index=True)

    # Tolerance zones: buffer RC1-RC1 and RC1-RC2 borders only
    total_area = unary_union(geoms)
    tolerant = border_segments_gdf[border_segments_gdf["border_type"].isin(["RC1-RC1", "RC1-RC2"])]
    if not tolerant.empty:
        tolerant_union = unary_union(tolerant.geometry.values)
        tolerance_poly = tolerant_union.buffer(buffer_distance)
        tolerance_zones_gdf = gpd.GeoDataFrame(
            [{
                "geometry": tolerance_poly,
                "buffer_m": buffer_distance,
                "border_types": "RC1-RC1,RC1-RC2",
                "description": f"{buffer_distance}m tolerance zone around RC1 borders",
            }],
            crs=msh.crs,
        )
        strict_poly = total_area.difference(tolerance_poly)
    else:
        tolerance_zones_gdf = gpd.GeoDataFrame(
            columns=["geometry", "buffer_m", "border_types", "description"],
            crs=msh.crs,
        )
        strict_poly = total_area

    strict_zones_gdf = gpd.GeoDataFrame(
        [{
            "geometry": strict_poly,
            "buffer_m": buffer_distance,
            "description": f"Mapped area minus {buffer_distance}m RC1 border tolerance zones",
        }],
        crs=msh.crs,
    )

    return border_segments_gdf, tolerance_zones_gdf, strict_zones_gdf


def load_qa_rand_gc(gdb_path: Path) -> gpd.GeoDataFrame:
    """Load QA_Rand_GC GDB; lowercase all column names (geometry kept as-is)."""
    logger.info(f"Loading QA Rand GC from {gdb_path}")
    layers = fiona.listlayers(str(gdb_path))
    if not layers:
        raise ValueError(f"No layers found in {gdb_path}")
    layer_name = layers[0]
    if len(layers) > 1:
        logger.warning(f"Multiple layers in {gdb_path}, using first: {layer_name}")
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    gdf = gdf.to_crs(DEFAULT_CRS)
    gdf.columns = [c.lower() if c != "geometry" else c for c in gdf.columns]
    logger.info(f"Loaded {len(gdf)} features, columns: {list(gdf.columns)}")
    return gdf


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


def _promote_to_multi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Promote single geometries to Multi* and LinearRing to LineString.

    FileGDB requires homogeneous geometry types with no LinearRing.
    """
    from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon

    _promote_map = {
        "Polygon":    lambda g: MultiPolygon([g]),
        "LineString": lambda g: MultiLineString([g]),
        "LinearRing": lambda g: MultiLineString([LineString(g)]),
        "Point":      lambda g: MultiPoint([g]),
    }

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: _promote_map[g.geom_type](g) if g is not None and g.geom_type in _promote_map else g
    )
    return gdf


def _write_layers(
    layers: dict[str, gpd.GeoDataFrame],
    output_path: Path,
    formats: tuple[str, ...],
) -> None:
    """Write all layers in each requested format."""
    if "gpkg" in formats:
        first = True
        for name, gdf in layers.items():
            mode = "w" if first else "a"
            gdf.to_file(output_path, layer=name, driver="GPKG", mode=mode)
            logger.info(f"✓ Written GPKG layer: {name} ({len(gdf)} features)")
            first = False

    if "filegdb" in formats:
        import shutil
        gdb_path = output_path.with_suffix(".gdb")
        if gdb_path.exists():
            shutil.rmtree(gdb_path)
        first = True
        for name, gdf in layers.items():
            mode = "w" if first else "a"
            _promote_to_multi(gdf).to_file(gdb_path, layer=name, driver="OpenFileGDB", mode=mode)
            logger.info(f"✓ Written FileGDB layer: {name} ({len(gdf)} features)")
            first = False

    web_formats = [f for f in ("geojson", "parquet", "flatgeobuf") if f in formats]
    if web_formats:
        layers_dir = output_path.parent / output_path.stem
        layers_dir.mkdir(parents=True, exist_ok=True)
        for name, gdf in layers.items():
            if "geojson" in web_formats:
                out = layers_dir / f"{name}.geojson"
                gdf.to_file(out, driver="GeoJSON")
                logger.info(f"✓ Written GeoJSON: {out}")
            if "parquet" in web_formats:
                out = layers_dir / f"{name}.parquet"
                gdf.to_parquet(out)
                logger.info(f"✓ Written GeoParquet: {out}")
            if "flatgeobuf" in web_formats:
                out = layers_dir / f"{name}.fgb"
                # Convert to WGS84 for web compatibility
                gdf_web = gdf.to_crs("EPSG:4326")
                gdf_web.to_file(out, driver="FlatGeobuf")
                logger.info(f"Converted to FlatGeoBuffer (EPSG:4326): {out}")



@click.command(context_settings={"show_default": True})
@click.option(
    "--output",
    "-o",
    "output_path",
    default=str(files("gcover.data").joinpath("administrative_zones.gpkg")),
    type=click.Path(path_type=Path),
    help="Output base path (GPKG); layer files go into a sibling directory with the same stem",
)
@click.option(
    "--format",
    "-f",
    "formats",
    multiple=True,
    type=click.Choice(["gpkg", "filegdb", "geojson", "parquet","flatgeobuf"], case_sensitive=False),
    default=("gpkg",),
    show_default=True,
    help="Output format(s). Repeat to enable multiple: -f gpkg -f filegdb",
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
@click.option(
    "--qa-rand-gc",
    "qa_rand_gc_file",
    required=False,
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to QA_Rand_GC.gdb; adds raw layer and a 50 m buffer of rand<>1 features",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def create_administrative_zones(
    output_path: Path,
    lots_file: Path,
    wu_file: Path,
    mapsheets_file: Path,
    sources_file: Path,
    formats: tuple[str, ...],
    qa_rand_gc_file: Path | None,
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
    layers_dir = output_path.parent / output_path.stem
    if not overwrite:
        if "gpkg" in formats and output_path.exists():
            click.echo(f"❌ Output file exists: {output_path}")
            click.echo("   Use --overwrite to replace it")
            return
        if "filegdb" in formats and output_path.with_suffix(".gdb").exists():
            click.echo(f"❌ Output FileGDB exists: {output_path.with_suffix('.gdb')}")
            click.echo("   Use --overwrite to replace it")
            return
        if any(f in formats for f in ("geojson", "parquet")) and layers_dir.exists():
            click.echo(f"❌ Output directory exists: {layers_dir}")
            click.echo("   Use --overwrite to replace it")
            return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if "gpkg" in formats and output_path.exists():
        output_path.unlink()
        logger.info(f"Removed existing file: {output_path}")

    try:
        # 1. Load all source data
        input_files = {
            "lots":       lots_file,
            "work units": wu_file,
            "mapsheets":  mapsheets_file,
            "sources":    sources_file,
        }
        if qa_rand_gc_file is not None:
            input_files["qa rand gc"] = qa_rand_gc_file

        table = Table(title="Input files", show_header=True, header_style="bold cyan")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Path")
        table.add_column("Modified", style="green", no_wrap=True)
        for label, path in input_files.items():
            mtime = dt.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(label, str(path.resolve()), mtime)
        console.print(table)

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
        # TODO: what to do with this?
        mapsheets_complete = spatial_join_safe(
            mapsheets_with_lots, wu_gdf, ["WU_NAME", "WU_ID"], "mapsheets-work_units"
        )

        # 4. Aggregate mapsheets_complete

        # Define the grouping columns (mapsheet columns)
        # "geometry" is always kept — it is the per-mapsheet deduplication key.
        # All other columns are optional: missing ones are silently dropped with a warning.
        _wanted_mapsheet_cols = [
            "MSH_MAP_TITLE",
            "MSH_MAP_NBR",
            "MSH_MAP_SCALE",
            "MSH_BASIS_TOPO",
            "MSH_AUTHOR",
            "MSH_OWNER",
            "MSH_MAPPING_PERIOD",
            "MSH_PUBL_YEAR",
            "MSH_BASIS_VECT",
            "MSH_MORE_INFO",
            "MSH_TOPO_NR",
            "SOURCE_RC",
            "Version",
        ]

        missing_cols = sorted(c for c in _wanted_mapsheet_cols if c not in mapsheets_complete.columns)
        if missing_cols:
            click.secho("Warning: grouping columns not found and will be skipped:", fg="yellow")
            for col in missing_cols:
                click.secho(f"  • {col}", fg="yellow")

        if "geometry" not in mapsheets_complete.columns:
            raise ValueError("mapsheets_complete has no geometry column — cannot aggregate")

        mapsheet_cols = ["geometry"] + [c for c in _wanted_mapsheet_cols if c in mapsheets_complete.columns]

        SOURCES_COLUMNS = ("SOURCE_RC",)




        # Define the columns to concatenate
        concat_cols = ["LOT_NR", "Status", "WU_NAME", "WU_ID"]

        aggregation_dict = {
            "LOT_NR": lambda x: "|".join(x.dropna().astype(int).astype(str).unique()),
            "Status": lambda x: "|".join(x.dropna().unique()),
            "WU_NAME": lambda x: "|".join(x.dropna().unique()),
            "WU_ID": lambda x: "|".join(x.dropna().astype(int).astype(str).unique()),
        }

        gdf_aggregated = (
            mapsheets_complete.groupby(mapsheet_cols, dropna=False)
            .agg(aggregation_dict)
            .reset_index()
        )

        mapsheets_complete = gpd.GeoDataFrame(
            gdf_aggregated, geometry="geometry", crs=mapsheets_gdf.crs
        )

        buffer_distance = 50
        border_gdf = border_mapsheet(mapsheets_gdf, buffer_distance=buffer_distance)

        # Classified border segments and tolerance zones
        click.echo("🗺️  Classifying border segments...")
        border_segments_gdf, tolerance_zones_gdf, strict_zones_gdf = classify_border_segments(
            mapsheets_with_sources, buffer_distance=buffer_distance
        )

        # 5. Collect all layers and write in requested format(s)
        layers: dict[str, gpd.GeoDataFrame] = {
            "mapsheets_sources_only": mapsheets_with_sources,
            "mapsheets_with_sources": mapsheets_complete,
            f"borders_{buffer_distance}m": border_gdf,
            "border_segments": border_segments_gdf,
            f"tolerance_zones_{buffer_distance}m": tolerance_zones_gdf,
            f"strict_zones_{buffer_distance}m": strict_zones_gdf,
            "lots": lots_gdf,
            "work_units": wu_gdf,
            "mapsheets": mapsheets_gdf,
        }

        if qa_rand_gc_file is not None:
            click.echo("📁 Loading QA_Rand_GC...")
            qa_rand_gc_gdf = load_qa_rand_gc(qa_rand_gc_file)
            layers["qa_rand_gc"] = qa_rand_gc_gdf

            if "rand" not in qa_rand_gc_gdf.columns:
                logger.warning("'rand' column not found in QA_Rand_GC; skipping buffer layer")
            else:
                rand_borders = qa_rand_gc_gdf[qa_rand_gc_gdf["rand"].astype(str) != "1"]
                total_area = unary_union(mapsheets_gdf.geometry)
                rand_buffer = unary_union(rand_borders.geometry).buffer(buffer_distance)
                inner_area = total_area.difference(rand_buffer)
                layers[f"qa_rand_gc_buffer_{buffer_distance}m"] = gpd.GeoDataFrame(
                    [{"geometry": inner_area, "buffer_m": buffer_distance}],
                    crs=qa_rand_gc_gdf.crs,
                )
                logger.info(f"Buffer layer: mapsheets minus {buffer_distance} m zone around {len(rand_borders)} active (rand != '1') features")

        click.echo(f"💾 Writing to {output_path} (formats: {', '.join(formats)})")
        _write_layers(layers, output_path, formats)

        # 5. Summary and validation
        click.echo(f"✅ Administrative zones created successfully!")
        click.echo(f"   📁 File: {output_path}")
        click.echo(f"   📊 Layers: mapsheets_with_sources, lots, work_units, mapsheets")

        def sources_diff(gdf):
            console = Console(record=True)
            diff_rows = gdf[gdf["SOURCE_RC"] != gdf["SOURCE_QA"]]
            table = Table(title="Mapsheets with differing SOURCE_RC and SOURCE_QA")

            table.add_column("MSH_MAP_TITLE", style="cyan")
            table.add_column("SOURCE_RC", style="magenta")
            table.add_column("SOURCE_QA", style="green")

            for _, row in diff_rows.iterrows():
                table.add_row(
                    str(row["MSH_MAP_TITLE"]),
                    str(row["SOURCE_RC"]),
                    str(row["SOURCE_QA"]),
                )

            console.print(table)
            return console.export_text()

        def validate_column(gdf, source_col):
            console = Console(record=True)
            # Count non-null unique values
            counts = gdf[source_col].dropna().value_counts()
            total = counts.sum()

            # Create Rich table
            table = Table(title=f"Unique {source_col} values")
            table.add_column("Source", style="cyan", no_wrap=True)
            table.add_column("Count", style="magenta", justify="right")

            for value, count in counts.items():
                table.add_row(str(value), str(count))
            table.add_row("[bold]Total[/bold]", f"[bold magenta]{total}[/bold magenta]")

            # Display

            console.print(table)
            return console.export_text()

            """rc1_count = (mapsheets_complete[source_col] == "RC1").sum()
            rc2_count = (mapsheets_complete[source_col] == "RC2").sum()
            other_count =
            no_source_count = mapsheets_complete[source_col].isna().sum()

            click.echo(f"==Column: {source_col}==")
            click.echo(f"   🔵 RC1 mapsheets: {rc1_count}")
            click.echo(f"   🟢 RC2 mapsheets: {rc2_count}")
            click.echo(f"   ⚪ No source: {no_source_count}")

            if no_source_count > 0:
                click.echo(
                    f"   ⚠️  {no_source_count} mapsheets have no source assignment"
                )"""

        # Validation summary
        validation_str = ""
        for source_col in SOURCES_COLUMNS:
            if source_col in mapsheets_with_sources.columns:
                validation_str += validate_column(mapsheets_with_sources, source_col)


        # Write the docstring to a file
        now = dt.now().strftime("%Y-%m-%d %H:%M:%S")

        layer_string = "\n * ".join(layers.keys())
        if "gpkg" in formats and output_path.exists():
            gpkg_layers = fiona.listlayers(output_path)
            layer_string = "\n * ".join(gpkg_layers)
        docstring = f"""{__doc__ or ""}\n\nXLSX file:{str(sources_file)}\n\nLayer list:\n * {layer_string}\n\n-- 'mapsheets_sources_only' --\n\n{validation_str}\n\n\nGenerated on {now}"""
        output_without_ext = output_path.with_suffix("")
        with open(output_without_ext.with_suffix(".README"), "w", encoding="utf-8") as f:
            f.write(docstring)

        if "LOT_NR" in mapsheets_complete.columns:
            with_lot = mapsheets_complete["LOT_NR"].notna().sum()
            click.echo(f"   📦 Mapsheets with lot assignment: {with_lot}")

        if "WU_NAME" in mapsheets_complete.columns:
            with_wu = mapsheets_complete["WU_NAME"].notna().sum()
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
