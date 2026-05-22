#!/usr/bin/env python3
"""
Check line topology for tectonic linear features against GC_BEDROCK polygon boundaries.

Three checks per mapsheet:

  thrust_deviations  — sections of thrust lines that deviate from any GC_BEDROCK
                       boundary edge by more than the snap tolerance.  Dangling
                       endpoints far from all boundaries are also reported so that
                       you can distinguish near-misses (small max_dist) from
                       intentionally unconnected ends (large max_dist).

  fault_gaps         — fault endpoints that are within gap_threshold of a boundary
                       but not snapped to it; typical digitising undershoot.

  fault_overshoots   — points where a fault crosses a GC_BEDROCK boundary at a
                       non-endpoint location; the fault was not split there.

The mapsheet approach (bedrock-only vs mixed) is detected per mapsheet: bedrock
covering ≥ BEDROCK_ONLY_THRESHOLD of the mapsheet area → bedrock-only.  The line
checks are identical in both cases; only GC_BEDROCK boundaries are used as the
topological reference for lines.

KIND codes (edit THRUST_KINDS / FAULT_KINDS to reclassify):
  14901001  Überschiebung         → thrust check
  14901002  Abschiebung           → fault check
  14901004  Bruch                 → fault check
  14901005  Aufschiebung          → fault check
  14901006  Blattverschiebung     → fault check
  14901007  komplexe Störung      → fault check
  14901008  Störung i. Allg.      → fault check
  14901009  neotektonischer Bruch → fault check
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
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points

from gcover.core.geometry import safe_read_filegdb

warnings.filterwarnings("ignore", category=UserWarning)

console = Console(record=True)

# ── Layer names ──────────────────────────────────────────────────────────────
BEDROCK_LAYER = "GC_BEDROCK"
LINEAR_LAYER  = "GC_LINEAR_OBJECTS"

# ── KIND classification — move codes between sets to reclassify ───────────────
THRUST_KINDS = frozenset([14901001])                                   # Überschiebung
FAULT_KINDS  = frozenset([14901002, 14901004, 14901005,
                           14901006, 14901007, 14901008, 14901009])

ALL_TECTO_KINDS = THRUST_KINDS | FAULT_KINDS

KIND_LABELS: dict[int, str] = {
    14901001: "Überschiebung",
    14901002: "Abschiebung",
    14901004: "Bruch",
    14901005: "Aufschiebung",
    14901006: "Blattverschiebung",
    14901007: "komplexe Störung",
    14901008: "Störung i. Allg.",
    14901009: "neotektonischer Bruch",
}

# ── Thresholds ────────────────────────────────────────────────────────────────
BEDROCK_ONLY_THRESHOLD = 0.99   # bedrock_area / mapsheet_area ≥ this → bedrock-only


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_mapsheets() -> gpd.GeoDataFrame:
    with ir.path("gcover.data", "administrative_zones.gpkg") as p:
        return gpd.read_file(str(p), layer="mapsheets_sources_only")


def _build_index(gdf: gpd.GeoDataFrame) -> tuple[STRtree, gpd.GeoDataFrame]:
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return STRtree(gdf.geometry.values), gdf


def _clip_to_mapsheet(gdf: gpd.GeoDataFrame, tree: STRtree, ms_geom) -> gpd.GeoDataFrame | None:
    idxs = tree.query(ms_geom, predicate="intersects")
    if len(idxs) == 0:
        return None
    clipped = gdf.iloc[idxs].clip(ms_geom)
    clipped = clipped[~clipped.geometry.is_empty]
    return clipped if not clipped.empty else None


def _bedrock_boundary(bedrock_clip: gpd.GeoDataFrame):
    """Union of ALL GC_BEDROCK polygon boundary rings (interior + exterior edges)."""
    geoms = [make_valid(g) for g in bedrock_clip.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    return unary_union([g.boundary for g in geoms])


def _get_endpoints(geom) -> list[Point]:
    """Return the start and end Point of every line component in geom."""
    if geom is None or geom.is_empty:
        return []
    lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
    pts: list[Point] = []
    for line in lines:
        coords = list(line.coords)
        if len(coords) >= 2:
            pts.append(Point(coords[0]))
            pts.append(Point(coords[-1]))
    return pts


def _uuid(row) -> str | None:
    for col in ("UUID", "uuid", "OBJECTID"):
        if col in row.index:
            return str(row[col])
    return None


# ── Per-check functions ───────────────────────────────────────────────────────

def _check_thrust_deviations(
    thrusts: gpd.GeoDataFrame,
    boundary,
    boundary_buffered,
    ms_nbr: str,
    min_deviation_length: float,
) -> list[dict]:
    rows = []
    for _, row in thrusts.iterrows():
        geom = make_valid(row.geometry)
        if geom is None or geom.is_empty:
            continue

        # Parts of the thrust NOT within tolerance of any BEDROCK boundary
        try:
            deviation = geom.difference(boundary_buffered)
        except Exception:
            continue

        if deviation is None or deviation.is_empty:
            continue

        # Decompose to individual linestring parts
        candidates = (
            list(deviation.geoms)
            if hasattr(deviation, "geoms")
            else [deviation]
        )
        for part in candidates:
            if part.geom_type not in ("LineString", "MultiLineString"):
                continue
            if part.length < min_deviation_length:
                continue
            try:
                max_dist = part.distance(boundary)
            except Exception:
                max_dist = float("nan")
            rows.append({
                "mapsheet_nbr":         ms_nbr,
                "uuid":                 _uuid(row),
                "kind":                 row.get("KIND"),
                "kind_label":           KIND_LABELS.get(int(row.get("KIND", 0)), ""),
                "deviation_length_m":   round(part.length, 2),
                "max_dist_to_bdry_m":   round(max_dist, 3),
                "check":                "thrust_deviation",
                "geometry":             part,
            })
    return rows


def _check_fault_gaps(
    faults: gpd.GeoDataFrame,
    boundary,
    ms_nbr: str,
    tolerance: float,
    gap_threshold: float,
) -> list[dict]:
    rows = []
    for _, row in faults.iterrows():
        geom = make_valid(row.geometry)
        if geom is None or geom.is_empty:
            continue
        for endpoint in _get_endpoints(geom):
            try:
                dist = endpoint.distance(boundary)
            except Exception:
                continue
            if tolerance < dist < gap_threshold:
                rows.append({
                    "mapsheet_nbr": ms_nbr,
                    "uuid":         _uuid(row),
                    "kind":         row.get("KIND"),
                    "kind_label":   KIND_LABELS.get(int(row.get("KIND", 0)), ""),
                    "gap_m":        round(dist, 3),
                    "check":        "fault_gap",
                    "geometry":     endpoint,
                })
    return rows


def _check_fault_overshoots(
    faults: gpd.GeoDataFrame,
    boundary,
    ms_nbr: str,
    tolerance: float,
) -> list[dict]:
    rows = []
    for _, row in faults.iterrows():
        geom = make_valid(row.geometry)
        if geom is None or geom.is_empty:
            continue
        try:
            crossing = geom.intersection(boundary)
        except Exception:
            continue
        if crossing is None or crossing.is_empty:
            continue

        # Collect only Point intersections — LineString means fault lies ON boundary (OK)
        pts: list[Point] = []
        if crossing.geom_type == "Point":
            pts = [crossing]
        elif crossing.geom_type == "MultiPoint":
            pts = list(crossing.geoms)
        elif crossing.geom_type == "GeometryCollection":
            pts = [g for g in crossing.geoms if g.geom_type == "Point"]

        if not pts:
            continue

        endpoints = _get_endpoints(geom)
        for pt in pts:
            min_ep_dist = min((pt.distance(ep) for ep in endpoints), default=float("inf"))
            if min_ep_dist > tolerance:
                rows.append({
                    "mapsheet_nbr":          ms_nbr,
                    "uuid":                  _uuid(row),
                    "kind":                  row.get("KIND"),
                    "kind_label":            KIND_LABELS.get(int(row.get("KIND", 0)), ""),
                    "dist_to_endpoint_m":    round(min_ep_dist, 3),
                    "check":                 "fault_overshoot",
                    "geometry":              pt,
                })
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("master_gdb", type=click.Path(exists=True))
@click.option("--output-gpkg", required=True, type=click.Path(),
              help="Output GeoPackage (layers: thrust_deviations, fault_gaps, fault_overshoots)")
@click.option("--mapsheet", default=None,
              help="Filter to a single mapsheet number (e.g. 1145) for targeted debugging")
@click.option("--tolerance", default=0.01, show_default=True,
              help="Snap tolerance in metres (matches SDE XY tolerance)")
@click.option("--gap-threshold", default=2.0, show_default=True,
              help="Max endpoint-to-boundary distance (m) still considered a gap. "
                   "Endpoints farther than this are assumed interior/intentional.")
@click.option("--min-deviation-length", default=0.5, show_default=True,
              help="Min length (m) of a thrust deviation section to report")
@click.option("--report", type=click.Path(), default=None,
              help="Write plain-text summary to this file")
def main(
    master_gdb,
    output_gpkg,
    mapsheet,
    tolerance,
    gap_threshold,
    min_deviation_length,
    report,
):
    """Check tectonic line topology against GC_BEDROCK boundaries per mapsheet."""
    gdb = Path(master_gdb)
    out = Path(output_gpkg)
    if out.exists():
        out.unlink()

    # ── Load ─────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Loading layers from GDB...[/]")
    bedrock = safe_read_filegdb(gdb, BEDROCK_LAYER)
    console.print(f"  {BEDROCK_LAYER}: {len(bedrock):,} features")

    linear_all = safe_read_filegdb(gdb, LINEAR_LAYER)
    console.print(f"  {LINEAR_LAYER}: {len(linear_all):,} features total")

    linear = linear_all[linear_all["KIND"].isin(ALL_TECTO_KINDS)].copy()
    console.print(f"  {LINEAR_LAYER} (tectonic kinds only): {len(linear):,} features")

    if linear.empty:
        console.print("[yellow]No tectonic linear features found — nothing to check.[/]")
        sys.exit(0)

    mapsheets = _load_mapsheets()
    if mapsheet:
        mapsheets = mapsheets[mapsheets["MSH_MAP_NBR"].astype(str) == str(mapsheet)]
        if mapsheets.empty:
            console.print(f"[red]Mapsheet {mapsheet!r} not found.[/]")
            sys.exit(1)
    console.print(f"  Mapsheets: {len(mapsheets)}")

    # Align CRS
    target_crs = bedrock.crs
    for gdf_name, gdf in [("linear", linear), ("mapsheets", mapsheets)]:
        if gdf.crs != target_crs:
            if gdf_name == "linear":
                linear = linear.to_crs(target_crs)
            else:
                mapsheets = mapsheets.to_crs(target_crs)

    # ── Spatial indexes ───────────────────────────────────────────────────────
    console.print("\n[bold cyan]Building spatial indexes...[/]")
    bedrock_tree, bedrock_v = _build_index(bedrock)
    linear_tree,  linear_v  = _build_index(linear)

    # ── Per-mapsheet loop ─────────────────────────────────────────────────────
    console.print("\n[bold cyan]Checking line topology per mapsheet...[/]")

    thrust_rows:    list[dict] = []
    gap_rows:       list[dict] = []
    overshoot_rows: list[dict] = []
    metrics_rows:   list[dict] = []

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
            ms_nbr   = str(ms["MSH_MAP_NBR"])
            ms_title = ms.get("MSH_MAP_TITLE", "")
            ms_rc    = ms.get("SOURCE_RC", "?")

            progress.update(task, description=f"[{ms_nbr}] {str(ms_title)[:28]}")

            # Clip layers to mapsheet
            bed_clip = _clip_to_mapsheet(bedrock_v, bedrock_tree, ms_geom)
            lin_clip = _clip_to_mapsheet(linear_v,  linear_tree,  ms_geom)

            n_thrusts    = 0
            n_faults     = 0
            n_thrust_dev = 0
            n_gaps       = 0
            n_overshoots = 0
            approach     = "unknown"

            if bed_clip is not None:
                bed_area = unary_union(
                    [make_valid(g) for g in bed_clip.geometry if g and not g.is_empty]
                ).area
                approach = "bedrock-only" if bed_area / ms_area >= BEDROCK_ONLY_THRESHOLD else "mixed"
            else:
                bed_area = 0.0

            if lin_clip is not None and bed_clip is not None:
                boundary = _bedrock_boundary(bed_clip)

                if boundary is not None and not boundary.is_empty:
                    boundary_buffered = boundary.buffer(tolerance)

                    thrusts = lin_clip[lin_clip["KIND"].isin(THRUST_KINDS)]
                    faults  = lin_clip[lin_clip["KIND"].isin(FAULT_KINDS)]
                    n_thrusts = len(thrusts)
                    n_faults  = len(faults)

                    if not thrusts.empty:
                        devs = _check_thrust_deviations(
                            thrusts, boundary, boundary_buffered,
                            ms_nbr, min_deviation_length,
                        )
                        thrust_rows.extend(devs)
                        n_thrust_dev = len(devs)

                    if not faults.empty:
                        gaps = _check_fault_gaps(
                            faults, boundary, ms_nbr, tolerance, gap_threshold,
                        )
                        gap_rows.extend(gaps)
                        n_gaps = len(gaps)

                        overshoots = _check_fault_overshoots(
                            faults, boundary, ms_nbr, tolerance,
                        )
                        overshoot_rows.extend(overshoots)
                        n_overshoots = len(overshoots)

            metrics_rows.append({
                "mapsheet_nbr":    ms_nbr,
                "mapsheet_title":  ms_title,
                "source_rc":       ms_rc,
                "approach":        approach,
                "bedrock_pct":     round(bed_area / ms_area * 100, 1) if ms_area else 0,
                "n_thrusts":       n_thrusts,
                "n_faults":        n_faults,
                "n_thrust_dev":    n_thrust_dev,
                "n_fault_gaps":    n_gaps,
                "n_overshoots":    n_overshoots,
            })

            progress.advance(task)

    # ── Write outputs ─────────────────────────────────────────────────────────
    crs = bedrock.crs

    def _write(rows, layer, geom_type_msg):
        if not rows:
            console.print(f"  [dim]{layer}: no issues[/]")
            return 0
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        gdf.to_file(str(out), layer=layer, driver="GPKG", append=out.exists())
        console.print(f"  [yellow]{layer}[/]: {len(gdf):,} {geom_type_msg}")
        return len(gdf)

    console.print("\n[bold cyan]Writing results...[/]")
    n_dev  = _write(thrust_rows,    "thrust_deviations", "line segments")
    n_gap  = _write(gap_rows,       "fault_gaps",        "endpoints")
    n_over = _write(overshoot_rows, "fault_overshoots",  "crossing points")

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(metrics_rows)

    table = Table(title="Line Topology Check Summary", header_style="bold cyan", show_lines=False)
    table.add_column("Mapsheet", style="dim", width=8)
    table.add_column("Title", width=28)
    table.add_column("RC", width=4)
    table.add_column("Approach", width=12)
    table.add_column("Thrust dev.", justify="right", width=10)
    table.add_column("Fault gaps", justify="right", width=10)
    table.add_column("Overshoots", justify="right", width=10)

    flagged = df[(df["n_thrust_dev"] > 0) | (df["n_fault_gaps"] > 0) | (df["n_overshoots"] > 0)]
    for _, r in flagged.sort_values("mapsheet_nbr").iterrows():
        style = "red" if (r["n_overshoots"] > 0 or r["n_thrust_dev"] > 0) else "yellow"
        table.add_row(
            str(r["mapsheet_nbr"]),
            str(r["mapsheet_title"])[:28],
            str(r["source_rc"]),
            str(r["approach"]),
            str(r["n_thrust_dev"]) if r["n_thrust_dev"] else "-",
            str(r["n_fault_gaps"]) if r["n_fault_gaps"] else "-",
            str(r["n_overshoots"]) if r["n_overshoots"] else "-",
            style=style,
        )

    table.add_row("", "", "", "", "", "", "")
    table.add_row(
        "TOTAL", f"{len(flagged)}/{len(df)} mapsheets flagged", "", "",
        str(n_dev), str(n_gap), str(n_over),
        style="bold",
    )
    console.print(table)

    has_issues = n_dev + n_gap + n_over > 0
    status = "[red]Issues found[/]" if has_issues else "[green]✓ No issues[/]"
    console.print(f"\n{status} — results in [bold]{out}[/]")
    if mapsheet:
        console.print(f"  (filtered to mapsheet {mapsheet})")

    if report:
        Path(report).write_text(console.export_text())
        console.print(f"[green]✓[/] Report written to [bold]{report}[/]")

    sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
