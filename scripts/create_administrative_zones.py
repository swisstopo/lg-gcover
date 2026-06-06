#!/usr/bin/env python
"""CLI entry point for building administrative_zones.gpkg.

All logic lives in gcover.publish.administrative_zones.
Run with --help for full option documentation.
"""

import click
from importlib.resources import files
from pathlib import Path

from gcover.publish.administrative_zones import create_administrative_zones


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
    type=click.Choice(["gpkg", "filegdb", "geojson", "parquet", "flatgeobuf"], case_sensitive=False),
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
    help="Path to sources Excel file (GC_Sources_PA.xlsx)",
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
@click.option(
    "--border-zones",
    "border_zones",
    is_flag=True,
    default=False,
    help="Also create border_segments, tolerance_zones and strict_zones layers (off by default).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(
    output_path: Path,
    lots_file: Path,
    wu_file: Path,
    mapsheets_file: Path,
    sources_file: Path,
    formats: tuple,
    qa_rand_gc_file,
    border_zones: bool,
    overwrite: bool,
    verbose: bool,
):
    """Create administrative zones GPKG from 4 standardized source files."""
    create_administrative_zones(
        output_path=output_path,
        lots_file=lots_file,
        wu_file=wu_file,
        mapsheets_file=mapsheets_file,
        sources_file=sources_file,
        formats=formats,
        qa_rand_gc_file=qa_rand_gc_file,
        border_zones=border_zones,
        overwrite=overwrite,
        verbose=verbose,
    )


if __name__ == "__main__":
    cli()
