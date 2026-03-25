"""
assign_aspect.py
================
Assign DEM-derived slope aspect to hexagonal-grid MultiPoint features.

Two methods are available as subcommands:

  simple  – Circular mean over all valid samples per feature.
             Fast, one output row per input feature.

  gmm     – Gaussian Mixture Model on (sin θ, cos θ) unit vectors.
             Can split a feature into multiple aspect groups when
             a polygon spans more than one coherent slope direction.
             One or more output rows per input feature.

Common workflow
---------------
Both subcommands share the same input arguments and most options:

    python assign_aspect.py [COMMON OPTIONS] simple GPKG RASTER
    python assign_aspect.py [COMMON OPTIONS] gmm    GPKG RASTER [GMM OPTIONS]

Example — simple circular mean:
    python assign_aspect.py \\
        --points-layer surfaces_aux_points \\
        --polygons-layer surfaces_filtered \\
        --output-layer surfaces_with_aspect \\
        --join-key UUID \\
        simple surfaces_aux.gpkg swissALTI3DRegio_aspect.tif

Example — GMM with spatial test-region filter:
    python assign_aspect.py \\
        --points-layer surfaces_aux_points \\
        --polygons-layer surfaces_filtered \\
        --output-layer surfaces_gmm_aspect \\
        --join-key UUID \\
        gmm surfaces_aux.gpkg swissALTI3DRegio_aspect.tif \\
        --max-components 3 --bbox 2554000 1145000 2575000 1164000

Output columns (both methods)
------------------------------
  azimuth            Circular mean, 0 = N, clockwise (upslope; flipped 180°
                     from raw aspect by default in the gmm subcommand).
  azimuth_std        Circular standard deviation — quality flag; > 45° is noisy.
  mapserver_angle    (360 − azimuth) % 360, ready for MapServer ANGLE directive.
  n_aspect_samples   Valid samples contributing to this row.
  n_aspect_total     Total grid points for this row (includes nodata).

Additional columns from the gmm subcommand
-------------------------------------------
  _gmm_k             Number of GMM components chosen for the parent feature.
  _gmm_component     0-based component index within the parent feature.
  _original_fid      Positional index of the parent feature in the input layer.
  _was_split         True when the parent was split into more than one group.

Circular data note
------------------
  Aspect is circular (0° = 360°). The GMM projects each angle onto the unit
  circle as (sin θ, cos θ) before fitting, so angular wrap-around is handled
  correctly. The simple subcommand uses the same projection for computing the
  circular mean.
"""

from __future__ import annotations

import sys
from typing import Optional

import click
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import MultiPoint, box
from sklearn.mixture import GaussianMixture
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TextColumn
from rich.table import Table
from rich import box as rbox

console = Console()

# ---------------------------------------------------------------------------
# Logging setup – loguru → stderr, no ANSI conflicts with rich on stdout
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular mean of an angle array (degrees). Returns NaN for empty/all-NaN."""
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return np.nan
    rad = np.deg2rad(angles_deg)
    return float(
        np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360
    )


def circular_std_deg(angles_deg: np.ndarray) -> float:
    """Circular standard deviation (Mardia & Jupp, 2000). Returns NaN for empty/all-NaN."""
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return np.nan
    rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.sin(rad)) ** 2 + np.mean(np.cos(rad)) ** 2)
    # R ≈ 1 → tightly clustered; R ≈ 0 → uniformly distributed
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(np.clip(R, 1e-10, 1.0)))))


def azimuth_to_mapserver(az: float) -> float:
    """
    Geographic azimuth (0 = N, clockwise) → MapServer symbol angle.

    MapServer rotates counter-clockwise from the x-axis (east), so:
        mapserver_angle = (360 - azimuth) % 360
    This mirrors the convention already used for planar / oriented point layers.
    """
    return (360.0 - az) % 360.0


def sample_raster_at_points(
    raster_path: str,
    coords: list[tuple[float, float]],
    nodata_value: Optional[float] = None,
) -> np.ndarray:
    """
    Sample a single-band raster at (x, y) coordinate pairs.
    Nodata and out-of-range (not in 0–360) values become NaN.
    """
    with rasterio.open(raster_path) as src:
        nd = nodata_value if nodata_value is not None else src.nodata
        values = np.array(
            [v[0] for v in sample_gen(src, coords)], dtype=np.float32
        )
        if nd is not None:
            values[values == nd] = np.nan
        values[(values < 0) | (values > 360)] = np.nan
    return values


def _load_and_filter(
    gpkg_path: str,
    points_layer: str,
    polygons_layer: str,
    bbox: Optional[tuple],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load point and polygon layers, optionally clip to a bounding box."""
    pts_gdf  = gpd.read_file(gpkg_path, layer=points_layer)
    poly_gdf = gpd.read_file(gpkg_path, layer=polygons_layer)
    logger.info(
        f"Loaded {len(pts_gdf)} multipoint features, {len(poly_gdf)} polygons"
    )

    if bbox:
        clip_box = box(*bbox)
        pts_gdf  = pts_gdf[pts_gdf.intersects(clip_box)].copy().reset_index(drop=True)
        poly_gdf = poly_gdf[poly_gdf.intersects(clip_box)].copy().reset_index(drop=True)
        logger.info(
            f"Bbox filter → {len(pts_gdf)} multipoint features, {len(poly_gdf)} polygons"
        )

    return pts_gdf, poly_gdf


def _join_polygon_attrs(
    pts_gdf: gpd.GeoDataFrame,
    poly_gdf: gpd.GeoDataFrame,
    join_key: Optional[str],
) -> gpd.GeoDataFrame:
    """Left-join polygon attributes onto the points GeoDataFrame by join_key."""
    if not join_key:
        return pts_gdf

    if join_key in pts_gdf.columns and join_key in poly_gdf.columns:
        poly_attrs = poly_gdf.drop(columns=["geometry"])
        existing   = set(pts_gdf.columns)
        new_cols   = [c for c in poly_attrs.columns if c not in existing or c == join_key]
        pts_gdf    = pts_gdf.join(
            poly_attrs[new_cols].set_index(join_key), on=join_key, how="left"
        )
        joined = [c for c in new_cols if c != join_key]
        logger.info(f"Joined polygon attributes on '{join_key}': {joined}")
    else:
        logger.warning(
            f"Join key '{join_key}' not present in both layers — skipping attribute join.\n"
            f"  Points  columns: {list(pts_gdf.columns)}\n"
            f"  Polygon columns: {list(poly_gdf.columns)}"
        )

    return pts_gdf


def _explode_and_sample(
    pts_gdf: gpd.GeoDataFrame,
    raster_path: str,
    nodata_value: Optional[float],
) -> gpd.GeoDataFrame:
    """
    Explode MultiPoints to individual Points, sample the raster at each,
    and return the exploded GeoDataFrame with a '_aspect' column.
    """
    logger.info("Exploding multipoints for bulk raster sampling…")
    exploded = pts_gdf.copy().explode(index_parts=True)
    exploded = exploded.reset_index(level=1, drop=True)
    exploded.index.name = "_feat_idx"
    exploded = exploded.reset_index()

    coords  = list(zip(exploded.geometry.x, exploded.geometry.y))
    logger.info(f"Sampling {len(coords):,} points from raster…")
    sampled = sample_raster_at_points(raster_path, coords, nodata_value)
    exploded["_aspect"] = sampled

    valid_pct = np.sum(np.isfinite(sampled)) / max(len(sampled), 1) * 100
    logger.info(f"  {valid_pct:.1f}% valid samples (non-nodata, within 0–360°)")

    return exploded


def _write_output(
    result: gpd.GeoDataFrame,
    gpkg_path: str,
    output_layer: str,
    pts_crs,
) -> gpd.GeoDataFrame:
    result = gpd.GeoDataFrame(result, crs=pts_crs)
    result = result[result.geometry.notna()].reset_index(drop=True)
    logger.info(f"Writing '{output_layer}' → {gpkg_path}  ({len(result):,} features)")
    result.to_file(gpkg_path, layer=output_layer, driver="GPKG")
    logger.success("Write complete.")
    return result


def _print_summary_table(result: gpd.GeoDataFrame, method: str) -> None:
    """Print a Rich summary table to stdout."""
    table = Table(
        title=f"Aspect assignment summary  [{method}]",
        box=rbox.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Output features", f"{len(result):,}")

    if "azimuth" in result.columns:
        n_az = result["azimuth"].notna().sum()
        table.add_row("Features with azimuth", f"{n_az:,}")
        table.add_row(
            "Mean azimuth",
            f"{result['azimuth'].mean():.1f}°" if n_az else "—",
        )

    if "azimuth_std" in result.columns:
        noisy = (result["azimuth_std"] > 45).sum()
        table.add_row(
            "Mean circular std",
            f"{result['azimuth_std'].mean():.1f}°" if result["azimuth_std"].notna().any() else "—",
        )
        table.add_row("Noisy features (std > 45°)", f"{noisy:,}")

    if "mapserver_angle" in result.columns:
        table.add_row(
            "Mean MapServer angle",
            f"{result['mapserver_angle'].mean():.1f}°"
            if result["mapserver_angle"].notna().any() else "—",
        )

    # GMM-specific rows
    if "_was_split" in result.columns:
        n_split = result.groupby("_original_fid")["_was_split"].first().sum()
        table.add_row("Parent features split", f"{int(n_split):,}")

    if "_gmm_k" in result.columns:
        dist = result.groupby("_gmm_k").size().to_dict()
        for k, cnt in sorted(dist.items()):
            table.add_row(f"  k={k} component(s)", f"{cnt:,} features")

    console.print(table)


# ===========================================================================
# Simple method
# ===========================================================================

def run_simple(
    gpkg_path: str,
    raster_path: str,
    points_layer: str,
    polygons_layer: str,
    output_layer: str,
    join_key: Optional[str],
    nodata_value: Optional[float],
    bbox: Optional[tuple],
    print_sample: bool,
) -> gpd.GeoDataFrame:
    """Circular mean per feature — one output row per input MultiPoint."""

    pts_gdf, poly_gdf = _load_and_filter(
        gpkg_path, points_layer, polygons_layer, bbox
    )
    pts_gdf = _join_polygon_attrs(pts_gdf, poly_gdf, join_key)
    exploded = _explode_and_sample(pts_gdf, raster_path, nodata_value)

    logger.info(f"Using aspect GeoTiff: {raster_path}")
    logger.info("Computing circular mean per feature…")
    all_rows: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        groups = list(exploded.groupby("_feat_idx"))
        task = progress.add_task("Processing features…", total=len(groups))

        for feat_idx, group in groups:
            source_row = pts_gdf.iloc[feat_idx]
            attrs = {
                k: v for k, v in source_row.items()
                if k != "geometry" and not k.startswith("_")
            }

            angles = group["_aspect"].values
            valid  = angles[np.isfinite(angles)]

            az  = circular_mean_deg(valid)
            std = circular_std_deg(valid)
            ms  = azimuth_to_mapserver(az) if np.isfinite(az) else np.nan

            all_rows.append(
                {
                    **attrs,
                    "geometry":          source_row.geometry,   # original MultiPoint
                    "azimuth":           round(az,  1) if np.isfinite(az)  else None,
                    "azimuth_std":       round(std, 1) if np.isfinite(std) else None,
                    "mapserver_angle":   round(ms,  1) if np.isfinite(ms)  else None,
                    "n_aspect_samples":  int(np.sum(np.isfinite(angles))),
                    "n_aspect_total":    len(angles),
                }
            )
            progress.advance(task)

    result = _write_output(all_rows, gpkg_path, output_layer, pts_gdf.crs)

    if print_sample:
        cols = [
            c for c in
            ["azimuth", "azimuth_std", "mapserver_angle", "n_aspect_samples", "n_aspect_total"]
            if c in result.columns
        ]
        console.print(result[cols].head(10).to_string())

    _print_summary_table(result, "simple")
    return result


# ===========================================================================
# GMM method
# ===========================================================================

def _angles_to_unit_vectors(angles_deg: np.ndarray) -> np.ndarray:
    """Degrees → (sin θ, cos θ) unit vectors for GMM fitting in ℝ²."""
    rad = np.deg2rad(angles_deg)
    return np.column_stack([np.sin(rad), np.cos(rad)])


def _select_gmm_components(
    X: np.ndarray,
    max_components: int,
    n_init: int = 5,
    random_state: int = 42,
) -> tuple[GaussianMixture, int]:
    """
    Fit GMM for k = 1 … max_components, return the model with lowest BIC
    and the chosen k.  n_init > 1 is important because the likelihood surface
    over (sin θ, cos θ) can be multimodal.
    """
    best_bic   = np.inf
    best_model: Optional[GaussianMixture] = None
    best_k     = 1

    for k in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                random_state=random_state,
            ).fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_model, best_k = bic, gmm, k
        except Exception as exc:
            logger.debug(f"GMM k={k} failed: {exc}")
            break

    return best_model, best_k


def _merge_small_components(
    labels: np.ndarray,
    X: np.ndarray,
    min_pts: int,
) -> np.ndarray:
    """
    Merge any component with fewer than min_pts points into its nearest
    surviving neighbour (Euclidean distance between component means in
    (sin θ, cos θ) space — equivalent to angular proximity).
    Returns relabelled array numbered 0..M-1.
    """
    labels = labels.copy()

    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = unique[counts < min_pts]
        if len(small) == 0:
            break

        target    = small[np.argmin(counts[counts < min_pts])]
        remaining = unique[counts >= min_pts]

        if len(remaining) == 0:
            break   # All components are small — give up merging

        target_mean     = X[labels == target].mean(axis=0)
        surviving_means = np.array([X[labels == r].mean(axis=0) for r in remaining])
        nearest         = remaining[np.argmin(np.linalg.norm(surviving_means - target_mean, axis=1))]
        labels[labels == target] = nearest

    # Consecutive relabelling 0..M-1
    final = np.full_like(labels, -1)
    for new_idx, old_label in enumerate(np.unique(labels)):
        final[labels == old_label] = new_idx
    return final


def _process_feature_gmm(
    points: list,
    angles: np.ndarray,
    attrs: dict,
    max_components: int,
    min_pts_per_group: int,
    min_pts_total: int,
    flip_180: bool,
    feat_idx: int,
) -> list[dict]:
    """
    Process one MultiPoint feature with GMM.  Returns one dict per
    surviving aspect group (may be > 1 if the feature was split).
    """
    valid_mask   = np.isfinite(angles)
    valid_pts    = [p for p, v in zip(points, valid_mask) if v]
    valid_angles = angles[valid_mask]

    def _make_row(pts, angs, gmm_k, gmm_comp, was_split):
        az  = circular_mean_deg(angs) if len(angs) > 0 else np.nan
        if flip_180 and np.isfinite(az):
            az = (az + 180.0) % 360.0
        std = circular_std_deg(angs) if len(angs) > 0 else np.nan
        ms  = azimuth_to_mapserver(az) if np.isfinite(az) else np.nan
        return {
            **attrs,
            "geometry":          MultiPoint(pts) if pts else None,
            "azimuth":           round(az,  1) if np.isfinite(az)  else None,
            "azimuth_std":       round(std, 1) if np.isfinite(std) else None,
            "mapserver_angle":   round(ms,  1) if np.isfinite(ms)  else None,
            "n_aspect_samples":  len(angs),
            "n_aspect_total":    len(pts),
            "_gmm_k":            gmm_k,
            "_gmm_component":    gmm_comp,
            "_original_fid":     feat_idx,
            "_was_split":        was_split,
        }

    if len(valid_pts) < min_pts_total:
        # Too few valid samples — skip GMM, keep as single feature
        return [_make_row(points, valid_angles, 1, 0, False)]

    X     = _angles_to_unit_vectors(valid_angles)
    k_max = max(1, min(max_components, len(valid_pts) // max(min_pts_per_group, 1)))

    gmm, k_chosen = _select_gmm_components(X, k_max)
    labels        = gmm.predict(X)
    labels        = _merge_small_components(labels, X, min_pts_per_group)

    n_groups  = len(np.unique(labels))
    was_split = n_groups > 1

    rows = []
    for comp in np.unique(labels):
        mask       = labels == comp
        group_pts  = [p for p, m in zip(valid_pts, mask) if m]
        group_angs = valid_angles[mask]
        rows.append(_make_row(group_pts, group_angs, k_chosen, int(comp), was_split))

    return rows


def run_gmm(
    gpkg_path: str,
    raster_path: str,
    points_layer: str,
    polygons_layer: str,
    output_layer: str,
    join_key: Optional[str],
    keep_join_key: bool,
    nodata_value: Optional[float],
    flip_180: bool,
    max_components: int,
    min_pts_per_group: int,
    min_pts_total: int,
    bbox: Optional[tuple],
    print_sample: bool,
) -> gpd.GeoDataFrame:
    """GMM clustering on circular aspect data — may produce multiple rows per feature."""

    pts_gdf, poly_gdf = _load_and_filter(
        gpkg_path, points_layer, polygons_layer, bbox
    )
    logger.info(f"gpkg_path: {gpkg_path}, points_layer: {points_layer}, polygon_layers: {polygons_layer}")
    pts_gdf  = _join_polygon_attrs(pts_gdf, poly_gdf, join_key)
    exploded = _explode_and_sample(pts_gdf, raster_path, nodata_value)

    logger.info("Running GMM clustering per feature…")
    logger.info(f"Using aspect GeoTiff: {raster_path}")
    all_rows: list[dict] = []
    n_split   = 0
    k_counts: dict[int, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        groups = list(exploded.groupby("_feat_idx"))
        task = progress.add_task("GMM clustering…", total=len(groups))

        for feat_idx, group in groups:
            source_row = pts_gdf.iloc[feat_idx]
            attrs = {
                k: v for k, v in source_row.items()
                if k != "geometry" and not k.startswith("_")
            }

            rows = _process_feature_gmm(
                points=list(group.geometry),
                angles=group["_aspect"].values,
                attrs=attrs,
                max_components=max_components,
                min_pts_per_group=min_pts_per_group,
                min_pts_total=min_pts_total,
                flip_180=flip_180,
                feat_idx=feat_idx,
            )
            all_rows.extend(rows)

            if any(r["_was_split"] for r in rows):
                n_split += 1
            k = rows[0]["_gmm_k"]
            k_counts[k] = k_counts.get(k, 0) + 1

            progress.advance(task)

    logger.info(
        f"Features split by GMM: {n_split} / {len(pts_gdf)}  |  "
        f"k distribution: { {k: v for k, v in sorted(k_counts.items())} }"
    )

    result = _write_output(all_rows, gpkg_path, output_layer, pts_gdf.crs)

    # Drop join key — it's meaningless when a parent feature is split into groups
    if join_key and join_key in result.columns and not keep_join_key:
        result = result.drop(columns=[join_key])
        logger.info(f"Dropped '{join_key}' (ambiguous after GMM split; use --keep-join-key to retain)")

    if print_sample:
        cols = [
            c for c in
            ["azimuth", "azimuth_std", "mapserver_angle",
             "n_aspect_samples", "_gmm_k", "_gmm_component", "_was_split"]
            if c in result.columns
        ]
        console.print(result[cols].head(10).to_string())

    _print_summary_table(result, "gmm")
    return result


# ===========================================================================
# Click CLI
# ===========================================================================

@click.group()
@click.argument("gpkg_path",   type=click.Path(exists=True))
@click.argument("raster_path", type=click.Path(exists=True))
@click.option("--points-layer",   default="surfaces_aux_points", show_default=True,
              help="MultiPoint layer name in the GPKG.")
@click.option("--polygons-layer", default="surfaces_filtered",   show_default=True,
              help="Source polygon layer name in the GPKG.")
@click.option("--output-layer",   default="surfaces_with_aspect", show_default=True,
              help="Name of the output layer written to GPKG.")
@click.option("--join-key",       default="UUID", show_default=True,
              help="Column linking MultiPoint features to their source polygon for "
                   "attribute transfer.  Pass an empty string to skip the join.")
@click.option("--nodata",         default=None, type=float,
              help="Override raster nodata value (default: read from raster metadata).")
@click.option("--bbox",           default=None, type=float, nargs=4,
              metavar="XMIN YMIN XMAX YMAX",
              help="Spatial filter in map CRS (EPSG:2056).  Useful for test runs.  "
                   "Example (Vevey–Bulle): --bbox 2554000 1145000 2575000 1164000")
@click.option("--print-sample",   is_flag=True,
              help="Print first 10 rows of the result to stdout.")
@click.pass_context
def cli(ctx, gpkg_path, raster_path, points_layer, polygons_layer,
        output_layer, join_key, nodata, bbox, print_sample):
    """
    Assign DEM-derived slope aspect to MultiPoint geological features.

    \b
    GPKG_PATH    GeoPackage containing both the MultiPoint and polygon layers.
    RASTER_PATH  Single-band aspect GeoTIFF (values 0–360°, 0 = North, clockwise).

    Choose a method:

    \b
      simple  Fast circular mean — one output row per input feature.
      gmm     Gaussian Mixture Model — can split features by aspect group.

    \b
    Examples:
        assign_aspect.py --join-key UUID data.gpkg aspect.tif simple
        assign_aspect.py --join-key UUID data.gpkg aspect.tif gmm --max-components 3

    Run  'assign_aspect.py GPKG RASTER METHOD --help'  for method-specific options.
    """
    ctx.ensure_object(dict)
    ctx.obj.update(
        gpkg_path=gpkg_path,
        raster_path=raster_path,
        points_layer=points_layer,
        polygons_layer=polygons_layer,
        output_layer=output_layer,
        join_key=join_key or None,
        nodata_value=nodata,
        bbox=tuple(bbox) if bbox else None,
        print_sample=print_sample,
    )


@cli.command("simple")
@click.pass_context
def cmd_simple(ctx):
    """
    Assign aspect via circular mean.  One output row per input feature.

    \b
    Example:
        python assign_aspect.py --output-layer surfaces_with_aspect \\
            data.gpkg aspect.tif simple
    """
    run_simple(**ctx.obj)


@cli.command("gmm")
@click.option("--keep-join-key", is_flag=True, default=False,
              help="Retain the join-key column in the output.  "
                   "Dropped by default because it becomes ambiguous after a feature split.")
@click.option("--no-flip", is_flag=True, default=False,
              help="Keep downslope direction (do not flip aspect by 180°).  "
                   "By default the script flips to upslope for cartographic conventions.")
@click.option("--max-components",    default=3, show_default=True, type=int,
              help="Maximum number of GMM components (aspect groups) tested per feature.  "
                   "The optimal k is chosen by BIC.")
@click.option("--min-pts-per-group", default=4, show_default=True, type=int,
              help="Minimum grid points per GMM component.  "
                   "Smaller groups are merged into the nearest surviving component.")
@click.option("--min-pts-total",     default=6, show_default=True, type=int,
              help="Skip GMM for features with fewer valid samples than this threshold "
                   "and return a single circular-mean row instead.")
@click.pass_context
def cmd_gmm(ctx, keep_join_key, no_flip, max_components, min_pts_per_group, min_pts_total):
    """
    Assign aspect via Gaussian Mixture Model on circular data.

    Each MultiPoint feature is clustered in (sin θ, cos θ) space.  If BIC
    favours k > 1 components the feature is split into separate groups —
    one output row per group, each with its own circular mean azimuth.

    \b
    Example (full dataset):
        python assign_aspect.py --output-layer surfaces_gmm_aspect \\
            data.gpkg aspect.tif gmm --max-components 3

    \b
    Example (test region, Vevey–Bulle):
        python assign_aspect.py data.gpkg aspect.tif \\
            --bbox 2554000 1145000 2575000 1164000 gmm --print-sample
    """
    run_gmm(
        **ctx.obj,
        keep_join_key=keep_join_key,
        flip_180=not no_flip,
        max_components=max_components,
        min_pts_per_group=min_pts_per_group,
        min_pts_total=min_pts_total,
    )


if __name__ == "__main__":
    cli()
