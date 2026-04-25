#!/usr/bin/env python3
"""
Check geometry validity and bedrock/unco_deposits coverage per mapsheet.

For each of the 220 mapsheets, reports:
  - geometry validity of GC_BEDROCK and GC_UNCO_DESPOSIT features
  - combined coverage (gaps)
  - whether the mapsheet uses approach 1 (bedrock fills, unco overlaps)
    or approach 2 (contiguous tiling)
"""

import importlib.resources as ir
import sys
import warnings
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from shapely import STRtree, make_valid, unary_union
from shapely.validation import explain_validity

from gcover.core.geometry import geometry_health_check, safe_read_filegdb

warnings.filterwarnings("ignore", category=UserWarning)

console = Console(record=True)

BEDROCK_LAYER = "GC_BEDROCK"
UNCO_LAYER = "GC_UNCO_DESPOSIT"

# Approach classification thresholds
APPROACH1_BEDROCK_MIN = 0.95  # bedrock covers ≥95% of mapsheet
APPROACH1_OVERLAP_MIN = 0.30  # ≥30% of unco sits on top of bedrock
APPROACH2_OVERLAP_MAX = 0.05  # <5% overlap → contiguous


def _load_mapsheets() -> gpd.GeoDataFrame:
    with ir.path("gcover.data", "administrative_zones.gpkg") as p:
        return gpd.read_file(str(p), layer="mapsheets_sources_only")


def _build_index(gdf: gpd.GeoDataFrame) -> tuple[STRtree, gpd.GeoDataFrame]:
    """Drop null/empty geometries and build STRtree."""
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return STRtree(gdf.geometry.values), gdf


def _clip_to_mapsheet(gdf: gpd.GeoDataFrame, tree: STRtree, ms_geom) -> gpd.GeoDataFrame | None:
    """Return features intersecting ms_geom, clipped to it."""
    idxs = tree.query(ms_geom, predicate="intersects")
    if len(idxs) == 0:
        return None
    clipped = gdf.iloc[idxs].clip(ms_geom)
    clipped = clipped[~clipped.geometry.is_empty]
    return clipped if not clipped.empty else None


def _safe_union(gdf: gpd.GeoDataFrame | None):
    """Unary union of valid geometries, returns None if empty."""
    if gdf is None or gdf.empty:
        return None
    geoms = [make_valid(g) for g in gdf.geometry if g is not None and not g.is_empty]
    return unary_union(geoms) if geoms else None


def _classify_approach(bedrock_cov: float, unco_area: float, overlap_ratio: float) -> str:
    if unco_area == 0:
        return "bedrock only"
    if bedrock_cov >= APPROACH1_BEDROCK_MIN and overlap_ratio >= APPROACH1_OVERLAP_MIN:
        return "1"
    if overlap_ratio <= APPROACH2_OVERLAP_MAX:
        return "2"
    return "mixed"


def _validate_layer(gdf: gpd.GeoDataFrame, layer_name: str) -> list[dict]:
    """Run geometry health check, return list of dicts for invalid/empty features."""
    stats = geometry_health_check(gdf)
    console.print(
        f"  {layer_name}: {stats['valid_geometries']:,} valid  "
        f"[yellow]{stats['invalid_geometries']:,} invalid[/]  "
        f"[red]{stats['empty_geometries']:,} empty[/]"
    )
    uuid_col = "UUID" if "UUID" in gdf.columns else None
    rows = []
    for label, indices, reason_fn in [
        ("invalid", stats["invalid_indices"], lambda g: explain_validity(g)),
        ("empty",   stats["empty_indices"],   lambda g: "empty geometry"),
    ]:
        for idx in indices:
            row = gdf.loc[idx]
            geom = row.geometry
            rows.append({
                "layer":   layer_name,
                "uuid":    row[uuid_col] if uuid_col else None,
                "issue":   label,
                "reason":  reason_fn(geom),
                "geometry": geom.centroid if geom and not geom.is_empty else None,
            })
    return rows


@click.command()
@click.argument("master_gdb", type=click.Path(exists=True))
@click.option("--output-gpkg", required=True, type=click.Path(),
              help="Output GPKG (three layers: mapsheet_metrics, coverage_gaps, invalid_geometries)")
@click.option("--min-gap-area", default=100.0, show_default=True,
              help="Minimum gap area in m² to record")
@click.option("--report", type=click.Path(), default=None,
              help="Write plain-text summary to this file")
def main(master_gdb, output_gpkg, min_gap_area, report):
    """Check geometry validity and bedrock/unco_deposits coverage per mapsheet."""
    gdb = Path(master_gdb)
    out = Path(output_gpkg)
    if out.exists():
        out.unlink()

    # ── Load ────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Loading layers from GDB...[/]")
    bedrock = safe_read_filegdb(gdb, BEDROCK_LAYER)
    console.print(f"  {BEDROCK_LAYER}: {len(bedrock):,} features")
    unco = safe_read_filegdb(gdb, UNCO_LAYER)
    console.print(f"  {UNCO_LAYER}: {len(unco):,} features")
    mapsheets = _load_mapsheets()
    console.print(f"  Mapsheets: {len(mapsheets)}")

    # Align CRS
    if unco.crs != bedrock.crs:
        unco = unco.to_crs(bedrock.crs)
    if mapsheets.crs != bedrock.crs:
        mapsheets = mapsheets.to_crs(bedrock.crs)

    # ── Geometry validation ──────────────────────────────────────────────────
    console.print("\n[bold cyan]Geometry validation...[/]")
    invalid_rows = _validate_layer(bedrock, BEDROCK_LAYER)
    invalid_rows += _validate_layer(unco, UNCO_LAYER)

    if invalid_rows:
        inv_gdf = gpd.GeoDataFrame(invalid_rows, geometry="geometry", crs=bedrock.crs)
        inv_gdf.to_file(str(out), layer="invalid_geometries", driver="GPKG")
        console.print(f"  [yellow]⚠[/]  {len(invalid_rows)} issues written to 'invalid_geometries'")
    else:
        console.print("  [green]✓[/] No invalid or empty geometries found")

    # ── Spatial indexes ──────────────────────────────────────────────────────
    console.print("\n[bold cyan]Building spatial indexes...[/]")
    bedrock_tree, bedrock_v = _build_index(bedrock)
    unco_tree,    unco_v    = _build_index(unco)

    # ── Per-mapsheet analysis ────────────────────────────────────────────────
    console.print("\n[bold cyan]Analysing mapsheets...[/]")
    metrics_rows: list[dict] = []
    gap_rows:     list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(mapsheets))

        for _, ms in mapsheets.iterrows():
            ms_geom  = make_valid(ms.geometry)
            ms_area  = ms_geom.area
            ms_nbr   = ms["MSH_MAP_NBR"]
            ms_title = ms["MSH_MAP_TITLE"]
            ms_rc    = ms.get("SOURCE_RC", "?")

            progress.update(task, description=f"[{ms_nbr}] {ms_title[:28]}")

            bed_clip  = _clip_to_mapsheet(bedrock_v, bedrock_tree, ms_geom)
            unco_clip = _clip_to_mapsheet(unco_v,    unco_tree,    ms_geom)

            bed_union  = _safe_union(bed_clip)
            unco_union = _safe_union(unco_clip)

            bed_area  = bed_union.area  if bed_union  else 0.0
            unco_area = unco_union.area if unco_union else 0.0

            if bed_union and unco_union:
                combined      = bed_union.union(unco_union)
                intersect_area = bed_union.intersection(unco_union).area
            elif bed_union:
                combined, intersect_area = bed_union, 0.0
            elif unco_union:
                combined, intersect_area = unco_union, 0.0
            else:
                combined, intersect_area = None, 0.0

            combined_area  = combined.area if combined else 0.0
            bedrock_cov    = bed_area  / ms_area if ms_area else 0.0
            unco_cov       = unco_area / ms_area if ms_area else 0.0
            combined_cov   = combined_area / ms_area if ms_area else 0.0
            overlap_ratio  = intersect_area / unco_area if unco_area else 0.0
            approach       = _classify_approach(bedrock_cov, unco_area, overlap_ratio)

            # Gap
            if combined:
                gap_geom = ms_geom.difference(combined)
                gap_area = gap_geom.area if gap_geom and not gap_geom.is_empty else 0.0
            else:
                gap_geom, gap_area = ms_geom, ms_area

            if gap_area > min_gap_area and gap_geom and not gap_geom.is_empty:
                gap_rows.append({
                    "MSH_MAP_NBR":   ms_nbr,
                    "MSH_MAP_TITLE": ms_title,
                    "SOURCE_RC":     ms_rc,
                    "approach":      approach,
                    "gap_area_m2":   round(gap_area, 1),
                    "gap_pct":       round(gap_area / ms_area * 100, 3),
                    "geometry":      gap_geom,
                })

            metrics_rows.append({
                "MSH_MAP_NBR":          ms_nbr,
                "MSH_MAP_TITLE":        ms_title,
                "SOURCE_RC":            ms_rc,
                "approach":             approach,
                "ms_area_m2":           round(ms_area, 1),
                "bedrock_cov_pct":      round(bedrock_cov   * 100, 2),
                "unco_cov_pct":         round(unco_cov      * 100, 2),
                "combined_cov_pct":     round(combined_cov  * 100, 2),
                "overlap_ratio_pct":    round(overlap_ratio * 100, 2),
                "gap_area_m2":          round(gap_area, 1),
                "geometry":             ms.geometry,
            })

            progress.advance(task)

    # ── Write outputs ────────────────────────────────────────────────────────
    metrics_gdf = gpd.GeoDataFrame(metrics_rows, geometry="geometry", crs=mapsheets.crs)
    metrics_gdf.to_file(str(out), layer="mapsheet_metrics", driver="GPKG", append=out.exists())

    if gap_rows:
        gaps_gdf = gpd.GeoDataFrame(gap_rows, geometry="geometry", crs=mapsheets.crs)
        gaps_gdf.to_file(str(out), layer="coverage_gaps", driver="GPKG", append=out.exists())

    # ── Summary ──────────────────────────────────────────────────────────────
    approach_counts = pd.Series([r["approach"] for r in metrics_rows]).value_counts()

    table = Table(title="Coverage & Approach Summary", header_style="bold cyan", show_lines=False)
    table.add_column("", style="dim", width=40)
    table.add_column("Count", justify="right", width=8)
    table.add_column("RC1", justify="right", width=6)
    table.add_column("RC2", justify="right", width=6)

    df = pd.DataFrame(metrics_rows)
    for label, key in [
        ("Approach 1 (bedrock fills, unco overlaps)", "1"),
        ("Approach 2 (contiguous tiling)",            "2"),
        ("Mixed / ambiguous",                         "mixed"),
        ("Bedrock only (no unco_deposits)",            "bedrock only"),
    ]:
        sub = df[df["approach"] == key]
        rc1 = (sub["SOURCE_RC"] == "RC1").sum()
        rc2 = (sub["SOURCE_RC"] == "RC2").sum()
        style = "yellow" if key == "mixed" else None
        table.add_row(label, str(len(sub)), str(rc1), str(rc2), style=style)

    table.add_row("", "", "", "")
    n_gaps = len(gap_rows)
    gap_sub = df[df["gap_area_m2"] > min_gap_area]
    table.add_row(
        f"Mapsheets with gaps > {min_gap_area:.0f} m²",
        str(n_gaps),
        str((gap_sub["SOURCE_RC"] == "RC1").sum()),
        str((gap_sub["SOURCE_RC"] == "RC2").sum()),
        style="red" if n_gaps else None,
    )
    table.add_row("Invalid / empty geometries", str(len(invalid_rows)), "", "")

    console.print(table)
    console.print(f"\n[green]✓[/] Results written to [bold]{out}[/] "
                  f"({len(metrics_rows)} mapsheets, {n_gaps} gaps)")

    if report:
        Path(report).write_text(console.export_text())
        console.print(f"[green]✓[/] Report written to [bold]{report}[/]")

    if n_gaps > 0 or invalid_rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
