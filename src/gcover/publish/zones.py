# src/gcover/publish/zones.py
"""Source-assignment join utilities and overview map generation.

Used by `gcover publish merge` (runtime join) and `gcover publish build-zones`
(to produce durable by-products alongside GC_Sources_PA.xlsx).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from loguru import logger

from gcover.publish.administrative_zones import load_sources

# Columns forwarded from the XLSX to the joined GeoDataFrame
_SOURCE_COLS = ("SOURCE_RC", "Version", "Notice", "BER", "ERL", "ber_link", "erl_link")

_SOURCE_COLOURS = {
    "RC2": "#2ecc71",
    "RC1": "#3498db",
}
_FALLBACK_COLOUR = "#cccccc"


def join_mapsheets_sources(
    admin_zones: Path,
    sources_xlsx: Path,
    geometry_layer: str = "mapsheets",
) -> gpd.GeoDataFrame:
    """Join static mapsheet geometry with source assignments from an XLSX file.

    Reads *geometry_layer* from *admin_zones* (geometry + MSH_* attributes),
    then left-joins selected columns from *sources_xlsx* on ``MSH_MAP_NBR``.

    Args:
        admin_zones: Path to the administrative zones GPKG.
        sources_xlsx: Path to GC_Sources_PA.xlsx (or equivalent).
        geometry_layer: Layer name in the GPKG (must contain ``MSH_MAP_NBR``).

    Returns:
        GeoDataFrame with geometry + MSH_* + source assignment columns.
    """
    mapsheets = gpd.read_file(admin_zones, layer=geometry_layer)
    sources_df = load_sources(sources_xlsx)

    keep = ["MSH_MAP_NBR"] + [c for c in _SOURCE_COLS if c in sources_df.columns]
    joined = mapsheets.merge(sources_df[keep], on="MSH_MAP_NBR", how="left")

    # Fallback: geometry rows still unmatched after the primary join get a second
    # chance via MSH_TOPO_NR (geometry) ↔ MSH_MAP_NBR (XLSX).  This covers sheets
    # that received a new publication number in the XLSX but whose geometry export
    # still carries the old topo number as MSH_MAP_NBR.
    unmatched_mask = joined["SOURCE_RC"].isna() if "SOURCE_RC" in joined.columns else joined.index.isin([])
    if unmatched_mask.any() and "MSH_TOPO_NR" in mapsheets.columns:
        topo_keep = ["MSH_TOPO_NR"] + [c for c in _SOURCE_COLS if c in sources_df.columns]
        topo_lookup = sources_df[topo_keep].dropna(subset=["MSH_TOPO_NR"])
        # Join geometry.MSH_TOPO_NR ↔ XLSX.MSH_TOPO_NR; don't overwrite already-resolved rows
        fallback = mapsheets.loc[unmatched_mask, ["MSH_MAP_NBR", "MSH_TOPO_NR"]].merge(
            topo_lookup, on="MSH_TOPO_NR", how="left"
        ).set_index(mapsheets.index[unmatched_mask])
        src_cols = [c for c in _SOURCE_COLS if c in fallback.columns]
        if fallback["SOURCE_RC"].notna().any():
            resolved = fallback.loc[fallback["SOURCE_RC"].notna()]
            for nbr, topo, title in zip(
                resolved["MSH_MAP_NBR"],
                fallback.loc[resolved.index, "MSH_TOPO_NR"],
                mapsheets.loc[resolved.index, "MSH_MAP_TITLE"],
            ):
                logger.warning(
                    f"join_mapsheets_sources: {title} matched via MSH_TOPO_NR={topo} "
                    f"(geometry MSH_MAP_NBR={nbr}) — update mapsheets.geojson export"
                )
            joined.loc[unmatched_mask, src_cols] = fallback[src_cols]

    assigned = joined["SOURCE_RC"].notna().sum() if "SOURCE_RC" in joined.columns else 0
    still_unmatched = joined["SOURCE_RC"].isna().sum() if "SOURCE_RC" in joined.columns else 0
    logger.info(
        f"join_mapsheets_sources: {len(mapsheets)} mapsheets, "
        f"{assigned} with SOURCE_RC, {still_unmatched} unmatched, from {sources_xlsx.name}"
    )
    return joined


def _gdb_date(gdb_path: Path) -> str | None:
    """Return YYYY-MM-DD parsed from the real GDB directory name (YYYYMMDD_…).

    Resolves symlinks so RC2.gdb → 20260518_0330_….gdb gives '2026-05-18'.
    Returns None if the path is absent or the name doesn't start with 8 digits.
    """
    if gdb_path is None or not gdb_path.exists():
        return None
    real_name = gdb_path.resolve().name
    digits = real_name[:8]
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return None


def write_sources_overview_png(
    mapsheets_gdf: gpd.GeoDataFrame,
    output_path: Path,
    source_column: str = "SOURCE_RC",
    title: str | None = None,
    sources_path: Path | None = None,
    source_dates: dict[str, str] | None = None,
) -> None:
    """Write a choropleth PNG of mapsheets coloured by source assignment.

    Skips with a warning if matplotlib is unavailable.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping overview PNG")
        return

    title = title or f"Source assignments ({source_column})"

    subtitle = None
    if sources_path is not None and sources_path.exists():
        from datetime import datetime
        mtime = datetime.fromtimestamp(sources_path.stat().st_mtime)
        subtitle = f"{sources_path.name}  ·  {mtime:%Y-%m-%d %H:%M}"

    src_values = sorted(str(s) for s in mapsheets_gdf[source_column].dropna().unique())

    fig, ax = plt.subplots(figsize=(14, 10))

    for src in src_values:
        subset = mapsheets_gdf[mapsheets_gdf[source_column] == src]
        subset.plot(ax=ax, color=_SOURCE_COLOURS.get(src, _FALLBACK_COLOUR),
                    edgecolor="white", linewidth=0.3)

    unassigned = mapsheets_gdf[mapsheets_gdf[source_column].isna()]
    if not unassigned.empty:
        unassigned.plot(ax=ax, color=_FALLBACK_COLOUR, edgecolor="white", linewidth=0.3)
        src_values = src_values + ["(unassigned)"]

    def _legend_label(s: str) -> str:
        if source_dates and s in source_dates:
            return f"{s}  ({source_dates[s]})"
        return s

    patches = [
        mpatches.Patch(color=_SOURCE_COLOURS.get(s, _FALLBACK_COLOUR), label=_legend_label(s))
        for s in src_values
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, -0.02), xycoords="axes fraction",
                    ha="center", va="top", fontsize=8, color="#666666")
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Written overview PNG: {output_path}")
