#!/usr/bin/env python3
"""
Generate auxiliary (labelling) points for surface polygon features.

Improvements over v1:
  • Fallback chain for features that produce no hex-grid points:
      1. Full buffer hex grid   (pt_method='grid')
      2. Half-buffer hex grid   (pt_method='grid_half')
      3. No-buffer hex grid     (pt_method='grid_none')
      4. representative_point() (pt_method='centroid')  ← always succeeds
  • Overlap detection: areas where different class features share space are
    saved as a separate GPKG layer (<layer>_overlaps).
  • Per-class MapServer STYLE OFFSET stored as ms_offset_x / ms_offset_y
    columns so MapServer can offset symbols to avoid piling up in overlap zones.
  • Per-feature ms_symbol_size: scales the nominal SIZE down for features
    smaller than the symbol's ground footprint, so symbols never overspill.

Glyph metrics (GeoFonts1.ttf, ink-bbox centering, 96 dpi, SYMBOLSCALEDENOM 12500):
  char 60 – sackungsgebiet (V)        4.75 × 5.09 px   safe radius 11.5 m
  char 65 – rutschgebiet  (arc `)`)   9.06 × 4.46 px   safe radius 16.7 m  ← widest
  char 67 – solifluktion  (flat arc)  6.49 × 1.62 px   safe radius 11.1 m
  char 68 – hakenwurf     (hook)      5.92 × 3.40 px   safe radius 11.3 m
  Recommended buffer: 18 m  (1.3 m margin above worst-case rutschgebiet radius)
  Overlap offset: rutschgebiet +7 px east (half-widths sum = 4.53 + 2.38 = 6.91 px)
"""

import math
import warnings

import click
import geopandas as gpd
import numpy as np
from rich import print as rprint
from rich.progress import Progress
from rich.table import Table
from shapely.geometry import MultiPoint, Point

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target symbols per input layer
SYMBOLS_BY_LAYER = {
    "surfaces": [
        "surfaces_gins_sackungsgebiet",
        "surfaces_gins_rutschgebiet",
        "surfaces_gins_gebiet_mit_solifluktion",
        "surfaces_gins_gebiet_mit_hakenwurf",
    ],
    "unco_deposits": [
        "unco_litho_rutschmasse",
        "unco_litho_zerruettete_sackungsmasse",
    ],
}

# Per-class safe symbol radius in ground metres (ink-bbox centering, 96 dpi,
# SYMBOLSCALEDENOM 12500, SIZE 12).  Used to scale ms_symbol_size for features
# smaller than the symbol footprint.
SYMBOL_SAFE_RADIUS_M: dict[str, float] = {
    "surfaces_gins_sackungsgebiet":          11.5,
    "surfaces_gins_rutschgebiet":            16.7,   # widest glyph – drives buffer choice
    "surfaces_gins_gebiet_mit_solifluktion": 11.1,
    "surfaces_gins_gebiet_mit_hakenwurf":    11.3,
    "unco_litho_rutschmasse":                16.7,   # geofonts1_65 – same glyph as rutschgebiet
    "unco_litho_zerruettete_sackungsmasse":  11.5,   # geofonts1_60 – same glyph as sackungsgebiet
}

# Per-class pixel offsets for MapServer STYLE OFFSET (screen pixels, not rotated
# with ANGLE).
# surfaces layer: only rutschgebiet (wide arc, 9.06 px) and sackungsgebiet (V,
#   4.75 px) overlap significantly (535 pairs).  Minimum clearance offset:
#   half-width rutschgebiet (4.53 px) + half-width sackungsgebiet (2.38 px) = 6.91 → 7 px
#   solifluktion and hakenwurf have ≤14 overlapping pairs → no offset needed.
# unco_deposits layer: no significant class overlaps → all offsets (0, 0).
CLASS_OFFSETS: dict[str, tuple[int, int]] = {
    "surfaces_gins_sackungsgebiet":          (0, 0),
    "surfaces_gins_rutschgebiet":            (7, 0),   # shift east by 7 px to clear V symbol
    "surfaces_gins_gebiet_mit_solifluktion": (0, 0),
    "surfaces_gins_gebiet_mit_hakenwurf":    (0, 0),
    "unco_litho_rutschmasse":                (0, 0),   # no class overlaps in unco layer
    "unco_litho_zerruettete_sackungsmasse":  (0, 0),
}


# ---------------------------------------------------------------------------
# Hex-grid helpers
# ---------------------------------------------------------------------------

def _try_hex_grid(inset_geom, dx: float, dy: float) -> list[Point]:
    """Place a globally-aligned hex grid inside *inset_geom*.

    Returns a (possibly empty) list of Points.
    """
    if inset_geom is None or inset_geom.is_empty:
        return []
    xmin, ymin, xmax, ymax = inset_geom.bounds
    start_y = math.floor(ymin / dy) * dy
    end_y   = math.ceil(ymax  / dy) * dy
    pts: list[Point] = []
    for y in np.arange(start_y, end_y + dy / 2, dy):
        row_idx = round(y / dy)
        x_off   = (dx / 2) if row_idx % 2 != 0 else 0.0
        start_x = math.floor((xmin - x_off) / dx) * dx + x_off
        for x in np.arange(start_x, xmax + dx / 2, dx):
            p = Point(x, y)
            if inset_geom.contains(p):
                pts.append(p)
    return pts


def _points_for_feature(
    geom, spacing: float, buffer: float
) -> tuple[list[Point], str]:
    """Generate labelling points for one feature with a 4-tier fallback.

    Returns
    -------
    (points, method)
      method ∈ {'grid', 'grid_half', 'grid_none', 'centroid'}
    """
    dx = spacing
    dy = spacing * math.sqrt(3) / 2

    # Tier 1 – full inset (original behaviour)
    inset = geom.buffer(-abs(buffer))
    pts = _try_hex_grid(inset, dx, dy)
    if pts:
        return pts, "grid"

    # Tier 2 – half inset (helps narrow but non-trivial features)
    if buffer > 0:
        inset_half = geom.buffer(-abs(buffer) / 2)
        pts = _try_hex_grid(inset_half, dx, dy)
        if pts:
            return pts, "grid_half"

    # Tier 3 – no inset (the grid is clipped to the raw polygon boundary)
    pts = _try_hex_grid(geom, dx, dy)
    if pts:
        return pts, "grid_none"

    # Tier 4 – guaranteed fallback: a single representative point
    rep = geom.representative_point()
    return [rep], "centroid"


# ---------------------------------------------------------------------------
# Symbol size scaling
# ---------------------------------------------------------------------------

def _symbol_size_for_feature(
    geom,
    map_symbol: str,
    nominal_size: int,
    min_size: int,
) -> float:
    """Compute a scaled-down MapServer SIZE for features smaller than the symbol.

    Strategy
    --------
    Approximate the feature's "effective radius" as the radius of a circle with
    the same 2-D area (i.e. r = sqrt(area / π)).  This is a proxy for the
    largest inscribed circle radius and is fast to compute.

    Compare against the per-class safe symbol radius (ground metres at the
    reference scale/dpi, from SYMBOL_SAFE_RADIUS_M).  If the feature is
    smaller, scale the SIZE proportionally, clamped to min_size.

    Returns
    -------
    float – the SIZE value to store as ms_symbol_size.  Equal to nominal_size
    for features large enough that the symbol fits comfortably.
    """
    safe_r_m = SYMBOL_SAFE_RADIUS_M.get(map_symbol, max(SYMBOL_SAFE_RADIUS_M.values()))
    area_2d  = geom.area          # shapely .area ignores Z, so this is the 2-D area
    r_feature = math.sqrt(area_2d / math.pi)           # equivalent-circle radius
    scale     = min(1.0, r_feature / safe_r_m)
    return max(float(min_size), round(nominal_size * scale, 1))


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def _detect_overlaps(
    gdf: gpd.GeoDataFrame,
    target_symbols: list[str],
    min_area_m2: float = 100.0,
) -> gpd.GeoDataFrame | None:
    """Find pairwise overlaps (shared interior area) between target features.

    Parameters
    ----------
    gdf : GeoDataFrame
        Full feature layer (polygon, any CRS in metres).
    target_symbols : list[str]
        Symbols (map_symbol values) to check.
    min_area_m2 : float
        Intersection area threshold to exclude boundary-only touches.

    Returns
    -------
    GeoDataFrame with intersection polygons + metadata, or None if empty.
    """
    rprint("[blue]→ Detecting overlaps between surface classes…[/blue]")

    # Caller has already lowercased all column names
    sel_cols = [c for c in ["uuid", "map_symbol", "geometry"] if c in gdf.columns]
    work = gdf[gdf["map_symbol"].isin(target_symbols)][sel_cols].copy()
    work = work.rename(columns={"uuid": "UUID"})
    # Force 2-D so sjoin / intersection work reliably on 3-D input
    work.geometry = work.geometry.force_2d()
    work = work.reset_index(drop=True)

    # Spatial join: all pairs whose interiors *intersect* (not just touch)
    left  = work.rename(columns={"UUID": "UUID_a", "map_symbol": "symbol_a"})
    right = work.rename(columns={"UUID": "UUID_b", "map_symbol": "symbol_b"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joined = gpd.sjoin(left, right, how="inner", predicate="intersects")

    # Drop self-matches
    joined = joined[joined["UUID_a"] != joined["UUID_b"]].copy()

    # Deduplicate A-B / B-A pairs
    joined["_pair_key"] = joined.apply(
        lambda r: tuple(sorted([r["UUID_a"], r["UUID_b"]])), axis=1
    )
    joined = joined.drop_duplicates("_pair_key").reset_index(drop=True)

    if joined.empty:
        rprint("[yellow]  No overlapping pairs found.[/yellow]")
        return None

    rprint(
        f"[blue]  Found {len(joined)} candidate pairs – computing intersection geometries…[/blue]"
    )

    uuid_to_geom: dict[str, object] = dict(zip(work["UUID"], work.geometry))

    records = []
    for _, pair in joined.iterrows():
        geom_a = uuid_to_geom.get(pair["UUID_a"])
        geom_b = uuid_to_geom.get(pair["UUID_b"])
        if geom_a is None or geom_b is None:
            continue
        inter = geom_a.intersection(geom_b)
        if inter.is_empty or inter.area < min_area_m2:
            continue
        sym_a, sym_b = sorted([pair["symbol_a"], pair["symbol_b"]])
        records.append(
            {
                "UUID_a":          pair["UUID_a"],
                "UUID_b":          pair["UUID_b"],
                "symbol_a":        sym_a,
                "symbol_b":        sym_b,
                "class_pair":      f"{sym_a} ∩ {sym_b}",
                "overlap_area_m2": round(inter.area, 1),
                "geometry":        inter,
            }
        )

    if not records:
        rprint("[yellow]  All candidate pairs are boundary-touches only (area < threshold).[/yellow]")
        return None

    out = gpd.GeoDataFrame(records, crs=work.crs)
    rprint(f"[green]  → {len(out)} real overlaps (area ≥ {min_area_m2} m²)[/green]")

    # Summary table
    tbl = Table(title="Overlap class pairs")
    tbl.add_column("Class pair", style="cyan")
    tbl.add_column("Count", justify="right", style="magenta")
    tbl.add_column("Total area (m²)", justify="right", style="yellow")
    for cp, grp in out.groupby("class_pair"):
        tbl.add_row(cp, str(len(grp)), f"{grp['overlap_area_m2'].sum():,.0f}")
    rprint(tbl)

    return out


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

@click.command()
@click.option("--input",  "-i", type=click.Path(exists=True), required=True,
              help="Path to input GPKG.")
@click.option("--layer",  "-l", required=True, default="surfaces",
              help="Layer name inside the GPKG.")
@click.option("--output", "-o", default="hex_multipoints.gpkg",
              help="Output GPKG filename.")
@click.option("--spacing", "-s", default=80.0,
              help="Hex grid spacing in metres (default 80).")
@click.option("--buffer", "-b", default=18.0,
              help="Inset distance from polygon edge in metres (default 18, "
                   "derived from worst-case symbol safe radius of 16.7 m for rutschgebiet).")
@click.option("--copy-polygons/--no-copy-polygons", "-d", is_flag=True,
              default=True, help="Copy the filtered polygons into the output GPKG.")
@click.option("--detect-overlaps/--no-detect-overlaps", default=True,
              help="Detect class overlaps and save as a separate layer (default on).")
@click.option("--class-offsets/--no-class-offsets", default=True,
              help="Add ms_offset_x/ms_offset_y columns for MapServer STYLE OFFSET.")
@click.option("--min-overlap-area", default=100.0, show_default=True,
              help="Minimum intersection area in m² to count as a real overlap.")
@click.option("--symbol-size", default=12, show_default=True,
              help="Nominal MapServer SIZE (pixels at SYMBOLSCALEDENOM). "
                   "Used as the upper bound for ms_symbol_size.")
@click.option("--min-symbol-size", default=4, show_default=True,
              help="Minimum ms_symbol_size in pixels (below this the symbol is unreadable).")
def generate_grid(
    input, layer, output, spacing, buffer, copy_polygons,
    detect_overlaps: bool, class_offsets: bool, min_overlap_area: float,
    symbol_size: int, min_symbol_size: int,
):
    """Generate a hexagonal MultiPoint grid for geologic surface features.

    For features too small or narrow to receive any grid point, a fallback
    strategy ensures every feature ends up with at least one representative
    point.  The *pt_method* attribute records which tier was used.
    """
    rprint(f"[bold blue]→ Reading layer:[/bold blue] {layer}")
    gdf = gpd.read_file(input, layer=layer)
    # Normalise to lowercase once so the rest of the script is uniform
    # (final denormalized_classified_translated.gpkg is all-lowercase;
    # other sources keep mixed case like UUID, KIND, …)
    gdf.columns = [c.lower() for c in gdf.columns]

    # ── Symbol selection ──────────────────────────────────────────────────
    target_symbols = SYMBOLS_BY_LAYER.get(layer)
    if target_symbols is None:
        rprint(
            f"[bold red]Error:[/bold red] No target symbols defined for layer '{layer}'. "
            f"Known layers: {', '.join(SYMBOLS_BY_LAYER)}"
        )
        return

    keep_cols = ["kind", "uuid", "label", "map_symbol", "geometry"]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    filtered_gdf = gdf[gdf["map_symbol"].isin(target_symbols)][existing_cols].copy()

    if filtered_gdf.empty:
        rprint("[bold red]Error:[/bold red] No matching features found for the specified symbols.")
        return

    rprint(f"  {len(filtered_gdf)} features selected across {filtered_gdf['map_symbol'].nunique()} classes.")

    # ── Overlap detection (before grid generation) ────────────────────────
    if detect_overlaps:
        overlaps_gdf = _detect_overlaps(gdf, target_symbols, min_area_m2=min_overlap_area)

    # ── Grid generation ───────────────────────────────────────────────────
    method_counts: dict[str, int] = {"grid": 0, "grid_half": 0, "grid_none": 0, "centroid": 0}
    results = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating points…", total=len(filtered_gdf))

        for _, row in filtered_gdf.iterrows():
            geom = row.geometry
            pts, method = _points_for_feature(geom, spacing, buffer)
            method_counts[method] += 1

            new_row = row.copy()
            new_row.geometry = MultiPoint(pts)
            new_row["pt_count"]    = len(pts)
            new_row["pt_method"]   = method
            new_row["ms_symbol_size"] = _symbol_size_for_feature(
                geom, row["map_symbol"], symbol_size, min_symbol_size
            )

            if class_offsets:
                ox, oy = CLASS_OFFSETS.get(row["map_symbol"], (0, 0))
                new_row["ms_offset_x"] = ox
                new_row["ms_offset_y"] = oy

            results.append(new_row)
            progress.update(task, advance=1)

    # ── Write outputs ─────────────────────────────────────────────────────
    if copy_polygons:
        filtered_gdf.to_file(output, layer=f"{layer}_filtered", driver="GPKG")
        rprint(f"  Polygons → layer '{layer}_filtered'")

    if results:
        out_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)
        out_gdf.to_file(output, layer=f"{layer}_aux_points", driver="GPKG")
        rprint(f"  Points   → layer '{layer}_aux_points'")

    if detect_overlaps and overlaps_gdf is not None:
        overlaps_gdf.to_file(output, layer=f"{layer}_overlaps", driver="GPKG")
        rprint(f"  Overlaps → layer '{layer}_overlaps'")

    # ── Summary tables ────────────────────────────────────────────────────
    if results:
        out_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)

        # Points per class
        tbl = Table(title="Points per class")
        tbl.add_column("Symbol",       style="cyan")
        tbl.add_column("Features",     justify="right")
        tbl.add_column("Points",       justify="right", style="magenta")
        tbl.add_column("Avg pts",      justify="right")
        tbl.add_column("Full size",    justify="right")
        tbl.add_column("Scaled down",  justify="right", style="yellow")
        tbl.add_column("Min size",     justify="right")
        for sym in target_symbols:
            sub = out_gdf[out_gdf["map_symbol"] == sym]
            if sub.empty:
                continue
            n_feat      = len(sub)
            n_pts       = int(sub["pt_count"].sum())
            avg_pts     = f"{n_pts / n_feat:.1f}" if n_feat else "—"
            n_full      = (sub["ms_symbol_size"] == symbol_size).sum()
            n_scaled    = (sub["ms_symbol_size"] < symbol_size).sum()
            min_sz      = f"{sub['ms_symbol_size'].min():.1f}"
            tbl.add_row(sym, str(n_feat), str(n_pts), avg_pts,
                        str(n_full), str(n_scaled), min_sz)
        rprint(tbl)

        # Fallback method breakdown
        tbl2 = Table(title="Point generation method (fallback chain)")
        tbl2.add_column("Method",      style="cyan")
        tbl2.add_column("Features",    justify="right", style="magenta")
        tbl2.add_column("%",           justify="right")
        total = sum(method_counts.values())
        labels = {
            "grid":       "grid (full buffer)",
            "grid_half":  "grid (½ buffer)",
            "grid_none":  "grid (no buffer)",
            "centroid":   "representative_point() fallback",
        }
        for m, label in labels.items():
            cnt = method_counts[m]
            if cnt:
                tbl2.add_row(label, str(cnt), f"{100 * cnt / total:.1f}%")
        rprint(tbl2)

        rprint(f"[bold green]✓ Done.[/bold green] Output: {output}")
    else:
        rprint("[bold red]No points generated. Check your buffer/spacing settings.[/bold red]")


if __name__ == "__main__":
    generate_grid()
