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

    assigned = joined["SOURCE_RC"].notna().sum() if "SOURCE_RC" in joined.columns else 0
    logger.info(
        f"join_mapsheets_sources: {len(mapsheets)} mapsheets, "
        f"{assigned} with SOURCE_RC, from {sources_xlsx.name}"
    )
    return joined


def write_sources_overview_png(
    mapsheets_gdf: gpd.GeoDataFrame,
    output_path: Path,
    source_column: str = "SOURCE_RC",
    title: str | None = None,
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
    sources = sorted(str(s) for s in mapsheets_gdf[source_column].dropna().unique())

    fig, ax = plt.subplots(figsize=(14, 10))

    for src in sources:
        subset = mapsheets_gdf[mapsheets_gdf[source_column] == src]
        subset.plot(ax=ax, color=_SOURCE_COLOURS.get(src, _FALLBACK_COLOUR),
                    edgecolor="white", linewidth=0.3)

    unassigned = mapsheets_gdf[mapsheets_gdf[source_column].isna()]
    if not unassigned.empty:
        unassigned.plot(ax=ax, color=_FALLBACK_COLOUR, edgecolor="white", linewidth=0.3)
        sources = sources + ["(unassigned)"]

    patches = [
        mpatches.Patch(color=_SOURCE_COLOURS.get(s, _FALLBACK_COLOUR), label=s)
        for s in sources
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Written overview PNG: {output_path}")
