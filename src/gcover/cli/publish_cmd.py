# gcover/cli/sde_cmd.py
import os
import sys
from importlib.resources import files
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
from gcover.publish.tooltips_enricher import TooltipsEnricher

console = Console()


@click.group(name="publish")
@click.pass_context
def publish_commands(ctx):
    """Commands for preparing data for publication."""
    pass


@publish_commands.command()
@click.pass_context
@click.argument("tooltip_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--rc1-gdb",
    type=click.Path(exists=True, path_type=Path),
    help="Path to RC1 QA FileGDB (for two-RC mode)",
)
@click.option(
    "--rc2-gdb",
    type=click.Path(exists=True, path_type=Path),
    help="Path to RC2 QA FileGDB (for two-RC mode)",
)
@click.option("--output", type=click.Path(path_type=Path), help="Output file path")
@click.option(
    "--target-fields", multiple=True, help="Fields to transfer from source to tooltip"
)
def enrich(ctx, tooltip_path, rc1_gdb, rc2_gdb, output, target_fields):
    """Enrich tooltip GeoPackage with attributes from source GeoPackage."""

    verbose = ctx.obj.get("verbose", False)

    enricher = None

    if verbose:
        logger.remove()  # Remove all handlers
        logger.add(sys.stderr, level="DEBUG")  # Add debug handler
        console.print("[dim]Verbose logging enabled[/dim]")

    data_sources = {"RC1": rc1_gdb, "RC2": rc2_gdb}

    # Run publish
    with TooltipsEnricher(
        tooltips_gdb=tooltip_path,
        admin_zones_gpkg=str(
            files("gcover.data").joinpath("administrative_zones.gpkg")
        ),
        rc_data_sources=data_sources,
    ) as enricher:
        # Enrich specific mapsheets
        enriched_data = enricher.enrich_polygon_main(
            mapsheet_numbers=[55, 25, 48, 27],
            area_threshold=0.7,
            save_intermediate=True,
            target_fields=[
                "KIND",
                "gmu_code",
                "tecto",
                "tecto_code",
                "lithology_main",
                "lithology_unco",
                "age_top",
                "age_base",
                "glacier_type_desc",
            ],
        )

        console.print(f"Enrichment complete: {len(enriched_data)} features processed")

    enriched_data = gpd.read_file(
        "/home/marco/DATA/Derivations/output/R14/enriched_polygon_main.gpkg"
    )

    # Define columns to show
    columns_to_show = [
        "DESCRIPT_D",
        "gmu_code",
        "tecto",
        "lithology_main",
        "SOURCE_UUID",
        "MATCH_METHOD",
        "MATCH_CONFIDENCE",
        "MATCH_LAYER",
        "MAPSHEET_NBR",
        "RC_SOURCE",
    ]

    # Create table
    table = Table(
        title="🔎 Sample Enriched Data", show_header=True, header_style="bold magenta"
    )

    # Add columns to table
    for col in columns_to_show:
        if col in enriched_data.columns:
            table.add_column(col, style="cyan", max_width=40, overflow="fold")

    for _, row in enriched_data.sample(
        n=min(5, len(enriched_data)), random_state=42
    ).iterrows():
        table.add_row(
            *[
                str(row[col.header]) if col.header in row else ""
                for col in table.columns
            ]
        )

    # Print table
    console.print(table)

    # Save output
    if output and enricher:
        output_path = Path(output) / "enriched_polygon_main.gpkg"
        logger.info(f"💾 Saving enriched data to: {output_path}")

        # Save results
        output_path = enricher.save_enriched_data(enriched_data, output_path)

    logger.success("🎉 Enrichment complete.")
