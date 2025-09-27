# gcover/cli/sde_cmd.py
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
import geopandas as gpd
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tabulate import tabulate

from gcover.config import SDE_INSTANCES, AppConfig, load_config  # TODO
from gcover.sde import SDEConnectionManager, create_bridge

console = Console()


@click.group(name="publish")
def publish_commands():
    """Commands for preparing data for publication."""
    pass


@publish_commands.command()
@click.argument("tooltip_path", type=click.Path(exists=True, path_type=Path))
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), help="Output file path")
@click.option(
    "--target-fields", multiple=True, help="Fields to transfer from source to tooltip"
)
def enrich(tooltip_path, source_path, output, target_fields, verbose):
    """Enrich tooltip GeoPackage with attributes from source GeoPackage."""

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()  # Remove all handlers
        logger.add(sys.stderr, level="DEBUG")  # Add debug handler
        console.print("[dim]Verbose logging enabled[/dim]")

    if verbose:
        logger.remove()
        logger.add(lambda msg: console.print(msg, markup=True), level="DEBUG")

    logger.info(f"🔍 Loading tooltip DB: {tooltip_path}")
    enriched = gpd.read_file(tooltip_path)

    logger.info(f"🔍 Loading source DB: {source_path}")
    all_sources = gpd.read_file(source_path)

    # Ensure CRS alignment
    enriched = enriched.set_crs("EPSG:2056", allow_override=True)
    all_sources = all_sources.set_crs("EPSG:2056", allow_override=True)

    # Check and filter target fields
    if not target_fields:
        target_fields = ["GEOLCODE", "GLAC_TYP", "CHRONO_T", "CHRONO_B", "GMU_CODE"]
    target_fields = [f for f in target_fields if f in all_sources.columns]

    logger.debug(f"✅ Fields to transfer: {target_fields}")

    # Match by UUID
    matches = enriched[["OBJECTID", "UUID"]].dropna()

    enriched["MATCH_CONFIDENCE"] = pd.Series([None] * len(enriched), dtype=float)
    enriched["MATCH_LAYER"] = None

    for _, match_row in matches.iterrows():
        objectid = match_row["OBJECTID"]
        uuid = match_row["UUID"]

        tooltip_idx = enriched[enriched["OBJECTID"] == objectid].index
        if tooltip_idx.empty:
            continue
        tooltip_idx = tooltip_idx[0]

        source_feat = all_sources[all_sources["UUID"] == uuid]
        if source_feat.empty:
            continue
        source_feat = source_feat.iloc[0]

        for field in target_fields:
            if field in source_feat.index and pd.notna(source_feat[field]):
                enriched.loc[tooltip_idx, field] = source_feat[field]

        enriched.loc[tooltip_idx, "MATCH_CONFIDENCE"] = 1.0
        enriched.loc[tooltip_idx, "MATCH_LAYER"] = source_feat.get(
            "source_fc", ""
        ).split("/")[-1]

    # Show sample enriched rows
    table = Table(
        title="🔎 Sample Enriched Data", show_header=True, header_style="bold magenta"
    )
    for col in ["OBJECTID", "UUID", "MATCH_LAYER", "MATCH_CONFIDENCE"] + list(
        target_fields
    ):
        if col in enriched.columns:
            table.add_column(col, style="cyan", max_width=30, overflow="fold")

    for _, row in enriched.sample(n=min(5, len(enriched)), random_state=42).iterrows():
        table.add_row(*[str(row.get(col, "")) for col in table.columns])

    console.print(table)

    # Save output
    if output:
        logger.info(f"💾 Saving enriched data to: {output}")
        enriched.to_file(output, driver="GPKG")

    logger.success("🎉 Enrichment complete.")
