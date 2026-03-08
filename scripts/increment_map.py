#!/usr/bin/env python3
"""
increment_map.py  –  Visualise GeoCover Increment FileGDB changes.

Usage examples:
    # Dark theme, full Switzerland, no background
    python increment_map.py 20260302_GCOVERP_2030-12-31.gdb

    # Light theme (print-ready), mapsheets background
    python increment_map.py ...gdb --theme light --bg data/administrative_zones.gpkg --bg-layer mapsheets

    # Zoom to a specific mapsheet (by MSH_MAP_NBR), work-units background
    python increment_map.py ...gdb --bg data/administrative_zones.gpkg --bg-layer wu --zoom-mapsheet 55

    # Lots background, light theme, 300 dpi
    python increment_map.py ...gdb --theme light --bg data/administrative_zones.gpkg --bg-layer lots --dpi 300

Change-type prefixes:
    A_   Addition              → green
    D_   Deletion              → red
    M_   Attr. modification    → orange
    MG_  Geometry modif.       → blue
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from importlib.resources import files


import click
import fiona
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rich import box as rbox
from rich.console import Console
from rich.table import Table

DEFAULT_ZONES_PATH = files("gcover.data").joinpath("administrative_zones.gpkg")



console = Console()

# ── Change-type configuration ─────────────────────────────────────────────────

CHANGE_TYPES: dict[str, dict] = {
    "A":  {"label": "Addition",           "color_dark": "#2ecc71", "color_light": "#1a7f3c", "zorder": 4},
    "D":  {"label": "Deletion",           "color_dark": "#e74c3c", "color_light": "#c0392b", "zorder": 3},
    "M":  {"label": "Attr. modification", "color_dark": "#f39c12", "color_light": "#d35400", "zorder": 5},
    "MG": {"label": "Geom. modification", "color_dark": "#3498db", "color_light": "#1a5a99", "zorder": 6},
}

GEOM_STYLE = {
    "polygon": {"alpha_dark": 0.55, "alpha_light": 0.45, "linewidth": 0.4},
    "line":    {"alpha_dark": 0.85, "alpha_light": 0.80, "linewidth": 1.4},
    "point":   {"alpha_dark": 0.90, "alpha_light": 0.90, "markersize": 5},
}

# Switzerland bounding box (LV95 / EPSG:2056) — fallback when no background
CH_BOUNDS_LV95 = (2484000, 1074000, 2834000, 1296000)
ZOOM_BUFFER_M  = 8_000   # metres padding around a zoomed mapsheet

# ── Background layer registry ─────────────────────────────────────────────────
# key: CLI name  →  (gpkg layer name, id field)

BG_LAYERS: dict[str, tuple[str, str]] = {
    "mapsheets": ("mapsheets_with_sources", "MSH_MAP_NBR"),
    "lots":      ("lots",                   "LOT_NR"),
    "wu":        ("work_units",             "WU_NAME"),
    "sources":   ("mapsheets_with_sources", "SOURCE_RC"),   # dissolved by RC1/RC2
}

# ── Themes ────────────────────────────────────────────────────────────────────

THEMES: dict[str, dict] = {
    "dark": {
        "fig_bg":       "#1a1a2e",
        "ax_bg":        "#16213e",
        "bg_face":      "#0f3460",
        "bg_edge":      "#22304a",
        "bg_lw":        0.3,
        "spine_color":  "#3d5a80",
        "tick_color":   "#7a8aaa",
        "title_color":  "white",
        "label_color":  "white",
        "legend_face":  "#0f3460",
        "legend_edge":  "#3d5a80",
        "legend_alpha": 0.35,
    },
    "light": {
        "fig_bg":       "white",
        "ax_bg":        "#f0f4f8",
        "bg_face":      "#dce8f5",
        "bg_edge":      "#7a9bbf",
        "bg_lw":        0.5,
        "spine_color":  "#4a6a8a",
        "tick_color":   "#333333",
        "title_color":  "#1a2a3a",
        "label_color":  "#1a2a3a",
        "legend_face":  "white",
        "legend_edge":  "#4a6a8a",
        "legend_alpha": 0.90,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_gdb_meta(gdb_path: Path) -> dict:
    stem = gdb_path.stem                    # 20260302_GCOVERP_2030-12-31
    parts = stem.split("_")
    date_str = parts[0] if parts else "unknown"
    return {"date": date_str, "stem": stem}


def format_date(raw: str) -> str:
    if re.match(r"^\d{8}$", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw


def classify_layer(layer_name: str) -> tuple[str | None, str]:
    """Return (change_key, base_name); MG must be tested before M."""
    for prefix in ("MG", "A", "D", "M"):
        if layer_name.startswith(f"{prefix}_"):
            return prefix, layer_name[len(prefix) + 1:]
    return None, layer_name


def geom_family(geom_type: str) -> str:
    gt = geom_type.lower()
    if "polygon" in gt:
        return "polygon"
    if "line" in gt or "string" in gt:
        return "line"
    return "point"


def reproject(gdf: gpd.GeoDataFrame, epsg: int = 2056) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg=epsg)
    if gdf.crs.to_epsg() != epsg:
        return gdf.to_crs(epsg=epsg)
    return gdf


# ── Data loading ──────────────────────────────────────────────────────────────

def load_increment_layers(gdb_path: Path) -> dict[str, gpd.GeoDataFrame]:
    try:
        available = fiona.listlayers(str(gdb_path))
    except Exception as exc:
        console.print(f"[red]Cannot open GDB:[/red] {exc}")
        sys.exit(1)

    result: dict[str, gpd.GeoDataFrame] = {}
    with console.status("[cyan]Reading increment layers…"):
        for lyr in available:
            change_key, _ = classify_layer(lyr)
            if change_key is None:
                continue
            try:
                gdf = gpd.read_file(str(gdb_path), layer=lyr)
                if gdf.empty:
                    continue
                gdf["geometry"] = gdf.geometry.force_2d()
                result[lyr] = reproject(gdf)
            except Exception as exc:
                console.print(f"[yellow]  skip {lyr}:[/yellow] {exc}")
    return result


def load_background(
    gpkg_path: Path | None,
    layer_key: str,
) -> gpd.GeoDataFrame | None:
    """Load named layer from administrative_zones.gpkg."""
    if gpkg_path is None:
        return None
    gpkg_layer, id_field = BG_LAYERS.get(layer_key, BG_LAYERS["mapsheets"])
    try:
        gdf = gpd.read_file(str(gpkg_path), layer=gpkg_layer)
        gdf = reproject(gdf)

        if layer_key == "sources":
            gdf = gdf.dissolve(by="SOURCE_RC", as_index=False)[["SOURCE_RC", "geometry"]]
            console.print(
                f"[dim]Background: layer=[bold]{gpkg_layer}[/bold] dissolved by SOURCE_RC"
                f"  ({len(gdf)} regions: {', '.join(gdf['SOURCE_RC'].tolist())})[/dim]"
            )
        else:
            console.print(
                f"[dim]Background: layer=[bold]{gpkg_layer}[/bold]  "
                f"({len(gdf)} features, id={id_field})[/dim]"
            )
        return gdf
    except Exception as exc:
        console.print(f"[yellow]Could not load background layer '{gpkg_layer}': {exc}[/yellow]")
        return None


def zoom_bounds_for_mapsheet(
    gpkg_path: Path,
    msh_nbr: int,
    buffer: int = ZOOM_BUFFER_M,
) -> tuple[float, float, float, float] | None:
    """Return (xmin, ymin, xmax, ymax) in LV95 for a given MSH_MAP_NBR + buffer."""
    gpkg_layer, id_field = BG_LAYERS["mapsheets"]
    try:
        gdf = gpd.read_file(str(gpkg_path), layer=gpkg_layer)
        gdf = reproject(gdf)
        row = gdf[gdf[id_field] == msh_nbr]
        if row.empty:
            console.print(
                f"[yellow]MSH_MAP_NBR {msh_nbr} not found in {gpkg_layer}; "
                f"using full extent.[/yellow]"
            )
            return None
        b = row.total_bounds           # (xmin, ymin, xmax, ymax)
        return (b[0] - buffer, b[1] - buffer, b[2] + buffer, b[3] + buffer)
    except Exception as exc:
        console.print(f"[yellow]zoom-mapsheet failed: {exc}[/yellow]")
        return None


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(layers: dict[str, gpd.GeoDataFrame]) -> None:
    table = Table(
        title="Increment layer summary",
        box=rbox.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Layer", style="dim")
    table.add_column("Type", justify="center")
    table.add_column("Features", justify="right")
    table.add_column("Geometry")

    totals: dict[str, int] = {k: 0 for k in CHANGE_TYPES}

    for lyr_name, gdf in sorted(layers.items()):
        change_key, base = classify_layer(lyr_name)
        cfg   = CHANGE_TYPES[change_key]
        color = cfg["color_dark"].lstrip("#")
        n     = len(gdf)
        totals[change_key] += n
        gt    = gdf.geometry.geom_type.iloc[0] if not gdf.empty else "?"
        table.add_row(base, f"[#{color}]{cfg['label']}[/#{color}]", str(n), gt)

    console.print(table)
    console.print()
    for key, cfg in CHANGE_TYPES.items():
        c = cfg["color_dark"].lstrip("#")
        console.print(f"  [#{c}]■[/#{c}] {cfg['label']:25s} {totals[key]:>6} features")
    console.print()


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(
    layers: dict[str, gpd.GeoDataFrame],
    meta: dict,
    background: gpd.GeoDataFrame | None,
    zoom: tuple[float, float, float, float] | None,
    theme_name: str,
    output_path: Path,
    dpi: int,
    zoom_label: str | None,
) -> None:
    th = THEMES[theme_name]

    # ── Determine the viewport ONCE; never derived from layer content ──────────
    # Bogus features at (0, 0) must never influence the view extent.
    if zoom is not None:
        view = zoom                     # (xmin, ymin, xmax, ymax)
    else:
        bx, by, bX, bY = CH_BOUNDS_LV95
        view = (bx, by, bX, bY)        # fixed Swiss extent

    fig, ax = plt.subplots(figsize=(16, 10), facecolor=th["fig_bg"])
    ax.set_facecolor(th["ax_bg"])

    # Pin the view before any plotting so gdf.plot() cannot auto-expand it
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # ── background ──
    # RC1/RC2 always get fixed colours; any other SOURCE_RC value (e.g. "Saas.gdb",
    # "BKP.gdb") is assigned a colour sequentially from the extras palette.
    RC_FIXED = {
        "dark":  {"RC1": "#1a3a5c", "RC2": "#2d1b4e"},
        "light": {"RC1": "#c8dff5", "RC2": "#e8d5f0"},
    }

    if background is not None:
        if "SOURCE_RC" in background.columns:
            # Dissolved sources layer — fixed colours for RC1/RC2, sequential for others
            rc_color_map: dict[str, str] = {}
            extras_cycle = iter(RC_EXTRAS[theme_name] * 10)   # repeat if > 4 extras
            for rc_val in sorted(background["SOURCE_RC"].unique()):
                if rc_val in RC_FIXED[theme_name]:
                    rc_color_map[rc_val] = RC_FIXED[theme_name][rc_val]
                else:
                    rc_color_map[rc_val] = next(extras_cycle)
            for rc_val, grp in background.groupby("SOURCE_RC"):
                grp.plot(
                    ax=ax,
                    color=rc_color_map[rc_val],
                    edgecolor=th["bg_edge"],
                    linewidth=th["bg_lw"],
                    zorder=1,
                 
                )
        else:
            background.plot(
                ax=ax,
                color=th["bg_face"],
                edgecolor=th["bg_edge"],
                linewidth=th["bg_lw"],
                zorder=1,
            )
    else:
        bx, by, bX, bY = CH_BOUNDS_LV95
        ax.add_patch(Rectangle(
            (bx, by), bX - bx, bY - by,
            linewidth=0.5,
            edgecolor=th["bg_edge"],
            facecolor=th["bg_face"],
            zorder=1,
        ))

    # ── change layers ──
    plotted: set[str] = set()
    color_key = f"color_{theme_name}"
    alpha_key = f"alpha_{theme_name}"

    for lyr_name, gdf in layers.items():
        change_key, _ = classify_layer(lyr_name)
        cfg    = CHANGE_TYPES[change_key]
        color  = cfg[color_key]
        zorder = cfg["zorder"]
        family = geom_family(gdf.geometry.geom_type.iloc[0])
        style  = GEOM_STYLE[family]
        alpha  = style[alpha_key]

        if family == "polygon":
            gdf.plot(ax=ax, color=color, edgecolor=color,
                     alpha=alpha, linewidth=style["linewidth"], zorder=zorder)
        elif family == "line":
            gdf.plot(ax=ax, color=color,
                     alpha=alpha, linewidth=style["linewidth"], zorder=zorder)
        else:
            gdf.plot(ax=ax, color=color, alpha=alpha,
                     markersize=style["markersize"], marker="o", zorder=zorder)
        plotted.add(change_key)

    # Re-enforce viewport — gdf.plot() may have silently expanded the axes
    # if any feature has a bogus geometry (e.g. snapped to 0, 0).
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # ── legend ──
    handles = [
        mpatches.Patch(
            facecolor=CHANGE_TYPES[k][color_key],
            label=CHANGE_TYPES[k]["label"],
            alpha=0.85,
        )
        for k in ("A", "D", "M", "MG") if k in plotted
    ]
    if background is not None and "SOURCE_RC" in background.columns:
        handles += [
            mpatches.Patch(facecolor=rc_color_map[rc], label=rc, alpha=0.9)
            for rc in sorted(rc_color_map)
        ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        framealpha=th["legend_alpha"],
        facecolor=th["legend_face"],
        edgecolor=th["legend_edge"],
        labelcolor=th["label_color"],
        fontsize=10,
        title="Change type",
        title_fontsize=11,
    )
    legend.get_title().set_color(th["title_color"])

    # ── axes styling ──
    ax.tick_params(colors=th["tick_color"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(th["spine_color"])

    # ── title ──
    date_fmt    = format_date(meta["date"])
    total_feat  = sum(len(g) for g in layers.values())
    zoom_suffix = f"  •  mapsheet {zoom_label}" if zoom_label else ""
    ax.set_title(
        f"GeoCover Increment  •  {date_fmt}  •  {total_feat:,} changed features{zoom_suffix}",
        color=th["title_color"],
        fontsize=14,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("E  (LV95)", color=th["tick_color"], fontsize=8)
    ax.set_ylabel("N  (LV95)", color=th["tick_color"], fontsize=8)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor=th["fig_bg"])
    plt.close(fig)
    console.print(f"[green]✓[/green] Map saved → [bold]{output_path}[/bold]")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("gdb_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output",  "-o", default=None,
              help="Output PNG. Default: <gdb_stem>_increment_map[_msNNN]_<theme>.png.")
@click.option("--dpi", default=150, show_default=True,
              help="Output resolution in DPI.")
@click.option("--theme", "-t",
              type=click.Choice(["dark", "light"], case_sensitive=False),
              default="dark", show_default=True,
              help="'dark' for screen, 'light' for print.")
@click.option("--bg", "-b", "bg_path",
              default=None, type=click.Path(exists=True),
              help="Path to administrative_zones.gpkg.")
@click.option("--bg-layer", "bg_layer",
              type=click.Choice(list(BG_LAYERS.keys()), case_sensitive=False),
              default="mapsheets", show_default=True,
              help=(
                  "Background layer to draw:\n\n"
                  "  mapsheets → mapsheets_with_sources (MSH_MAP_NBR)\n"
                  "  lots      → lots                   (LOT_NR)\n"
                  "  wu        → work_units             (WU_NAME)\n"
                  "  sources   → mapsheets dissolved by SOURCE_RC (RC1/RC2)"
              ))
@click.option("--zoom-mapsheet", "zoom_mapsheet",
              default=None, type=int,
              help="Zoom view to one mapsheet by MSH_MAP_NBR (requires --bg).")
def main(
    gdb_path: str,
    output: str | None,
    dpi: int,
    theme: str,
    bg_path: str | None,
    bg_layer: str,
    zoom_mapsheet: int | None,
) -> None:
    """Generate a change map from a GeoCover Increment FileGDB.

    Layers with prefixes A_ / D_ / M_ / MG_ are colour-coded by change type.
    An optional background layer from administrative_zones.gpkg can be added,
    and the view can be restricted to a single mapsheet with --zoom-mapsheet.
    """
    gdb  = Path(gdb_path)
    meta = parse_gdb_meta(gdb)

    if output is None:
        extent_tag = f"ms{zoom_mapsheet}" if zoom_mapsheet else "ch"
        bg_tag     = f"_{bg_layer}" if bg_path else ""
        out_path   = gdb.parent / f"{meta['stem']}_increment_map_{extent_tag}{bg_tag}_{theme}.png"
    else:
        out_path = Path(output)

    console.rule("[bold cyan]GeoCover Increment Map[/bold cyan]")
    console.print(f"  GDB    : [bold]{gdb}[/bold]")
    console.print(f"  Date   : {format_date(meta['date'])}")
    console.print(f"  Theme  : {theme}")
    console.print(f"  BG     : {bg_layer if bg_path else '—'}")
    if zoom_mapsheet:
        console.print(f"  Zoom   : mapsheet {zoom_mapsheet}")
    console.print(f"  Output : {out_path}")
    console.print()

    if zoom_mapsheet and not bg_path:
        console.print("[red]--zoom-mapsheet requires --bg (path to administrative_zones.gpkg).[/red]")
        sys.exit(1)

    layers = load_increment_layers(gdb)
    if not layers:
        console.print("[red]No increment layers found (A_/D_/M_/MG_ prefixes).[/red]")
        sys.exit(1)

    print_summary(layers)

    bg   = load_background(Path(bg_path) if bg_path else None, bg_layer)
    zoom = zoom_bounds_for_mapsheet(Path(bg_path), zoom_mapsheet) if zoom_mapsheet and bg_path else None

    build_map(
        layers=layers,
        meta=meta,
        background=bg,
        zoom=zoom,
        theme_name=theme.lower(),
        output_path=out_path,
        dpi=dpi,
        zoom_label=str(zoom_mapsheet) if zoom_mapsheet else None,
    )


if __name__ == "__main__":
    main()
