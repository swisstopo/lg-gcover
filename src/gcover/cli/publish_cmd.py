# src/gcover/cli/publish_cmd.py
"""
Enhanced CLI commands for preparing GeoCover data for publication.
Supports multiple tooltip layers, flexible source mappings, and comprehensive configuration.
"""

import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

import click
import geopandas as gpd
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from gcover.config import SDE_INSTANCES, AppConfig, load_config
from gcover.publish.tooltips_enricher import (
    EnhancedTooltipsEnricher,
    EnrichmentConfig,
    LayerMapping,
    LayerType,
    create_enrichment_config,
)

console = Console()


def get_publish_config(ctx):
    """Get publish configuration from context."""
    try:
        app_config: AppConfig = load_config(environment=ctx.obj["environment"])
        return app_config.publish, app_config.global_
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print(
            "Make sure your configuration includes publish and global settings"
        )
        raise click.Abort()


@click.group(name="publish")
@click.pass_context
def publish_commands(ctx):
    """Commands for preparing GeoCover data for publication."""
    pass


@publish_commands.command()
@click.pass_context
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="YAML configuration file for enrichment",
)
@click.option(
    "--tooltip-db",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    help="Path to geocover_tooltips.gdb",
)
@click.option(
    "--admin-zones",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    help="Path to administrative_zones.gpkg",
)
@click.option(
    "--source",
    "-s",
    multiple=True,
    help="Source data in format 'name:path' (e.g., 'rc1:/path/to/RC1.gdb')",
)
@click.option(
    "--layers",
    "-l",
    multiple=True,
    help="Specific tooltip layers to process (default: all configured)",
)
@click.option(
    "--mapsheets", "-m", help="Comma-separated mapsheet numbers (default: all)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for enriched data (.gpkg)",
)
@click.option(
    "--debug-dir",
    type=click.Path(path_type=Path),
    help="Directory to save debug/intermediate files",
)
@click.option(
    "--save-intermediate", is_flag=True, help="Save intermediate results per mapsheet"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without running enrichment",
)
def enrich(
    ctx,
    config_file: Optional[Path],
    tooltip_db: Optional[Path],
    admin_zones: Optional[Path],
    source: tuple,
    layers: tuple,
    mapsheets: Optional[str],
    output: Optional[Path],
    debug_dir: Optional[Path],
    save_intermediate: bool,
    dry_run: bool,
):
    """
    Enrich tooltip layers with attributes from source databases.
    
    Can be used with a configuration file or command-line options.
    
    Examples:
    
    \b
    # Using configuration file
    gcover publish enrich --config-file enrichment_config.yaml
    
    \b
    # Using command-line options
    gcover publish enrich \\
        --tooltip-db /path/to/geocover_tooltips.gdb \\
        --admin-zones /path/to/administrative_zones.gpkg \\
        --source rc1:/path/to/RC1.gdb \\
        --source rc2:/path/to/RC2.gdb \\
        --source saas:/path/to/Saas.gdb \\
        --layers POLYGON_MAIN \\
        --layers POINT_GEOL \\
        --mapsheets "55,25,48" \\
        --output enriched_tooltips.gpkg \\
        --debug-dir debug_output \\
        --save-intermediate
    """

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        console.print("[dim]Verbose logging enabled[/dim]")

    # Load configuration
    config = None

    if config_file:
        console.print(f"📄 Loading configuration from {config_file}")
        config = load_enrichment_config_from_file(config_file)
    else:
        # Build configuration from command-line options
        if not tooltip_db or not admin_zones:
            console.print(
                "[red]Error: --tooltip-db and --admin-zones are required when not using --config-file[/red]"
            )
            raise click.Abort()

        # Parse source arguments
        source_paths = {}
        for src in source:
            if ":" not in src:
                console.print(
                    f"[red]Error: Invalid source format '{src}'. Use 'name:path'[/red]"
                )
                raise click.Abort()
            name, path = src.split(":", 1)
            source_paths[name] = Path(path)

        if not source_paths:
            console.print("[red]Error: At least one source must be specified[/red]")
            raise click.Abort()

        # Parse mapsheets
        mapsheet_numbers = None
        if mapsheets:
            try:
                mapsheet_numbers = [int(x.strip()) for x in mapsheets.split(",")]
            except ValueError:
                console.print(
                    f"[red]Error: Invalid mapsheet format '{mapsheets}'[/red]"
                )
                raise click.Abort()

        # Create configuration
        config = create_enrichment_config(
            tooltip_db_path=tooltip_db,
            admin_zones_path=admin_zones,
            source_paths=source_paths,
            output_path=output,
            debug_output_dir=debug_dir,
            save_intermediate=save_intermediate,
            mapsheet_numbers=mapsheet_numbers,
        )

    # Override config with command-line options if provided
    if output:
        config.output_path = output
    if debug_dir:
        config.debug_output_dir = debug_dir
        config.save_intermediate = save_intermediate
    if mapsheets and not config_file:  # Don't override config file mapsheets
        try:
            config.mapsheet_numbers = [int(x.strip()) for x in mapsheets.split(",")]
        except ValueError:
            console.print(f"[red]Error: Invalid mapsheet format '{mapsheets}'[/red]")
            raise click.Abort()

    # Validate configuration
    try:
        validate_enrichment_config(config)
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()

    # Determine layers to process
    layers_to_process = list(layers) if layers else None

    # Show configuration summary
    show_config_summary(config, layers_to_process, dry_run)

    if dry_run:
        console.print(
            "\n[yellow]Dry run completed. Use --no-dry-run to execute enrichment.[/yellow]"
        )
        return

    # Confirm before processing
    if not click.confirm("\nProceed with enrichment?"):
        console.print("Enrichment cancelled.")
        return

    # Run enrichment
    try:
        with EnhancedTooltipsEnricher(config) as enricher:
            console.print("\n🚀 Starting enrichment process...")

            if layers_to_process and len(layers_to_process) == 1:
                # Single layer enrichment
                layer_name = layers_to_process[0]
                enriched_data = enricher.enrich_layer(
                    layer_name=layer_name, mapsheet_numbers=config.mapsheet_numbers
                )

                # Save results
                if config.output_path:
                    output_path = enricher.save_enriched_data(enriched_data)
                    console.print(f"💾 Saved enriched {layer_name} to: {output_path}")

                # Show results summary
                show_enrichment_summary({layer_name: enriched_data})

            else:
                # Multi-layer enrichment
                enriched_data = enricher.enrich_all_layers(
                    layer_names=layers_to_process,
                    mapsheet_numbers=config.mapsheet_numbers,
                )

                # Save results
                if config.output_path:
                    output_path = enricher.save_enriched_data(enriched_data)
                    console.print(f"💾 Saved enriched layers to: {output_path}")

                # Show results summary
                show_enrichment_summary(enriched_data)

    except Exception as e:
        console.print(f"[red]Enrichment failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()

    console.print("\n🎉 Enrichment completed successfully!")


@publish_commands.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="enrichment_config.yaml",
    help="Output path for configuration template",
)
@click.option(
    "--example-sources", is_flag=True, help="Include example source configurations"
)
def create_config(output: Path, example_sources: bool):
    """Create a configuration template file for enrichment."""

    # Base configuration template
    config_template = {
        "tooltip_db_path": "/path/to/geocover_tooltips.gdb",
        "admin_zones_path": "/path/to/administrative_zones.gpkg",
        "source_paths": {
            "rc1": "/path/to/RC1.gdb",
            "rc2": "/path/to/RC2.gdb",
        },
        "output_path": "/path/to/enriched_tooltips.gpkg",
        "debug_output_dir": "/path/to/debug_output",
        "save_intermediate": True,
        "mapsheet_numbers": None,  # Process all mapsheets
        "clip_tolerance": 0.0,
        # Layer-specific configurations (optional - uses defaults if not specified)
        "layer_mappings": {
            "POLYGON_MAIN": {
                "source_layers": [
                    "GC_ROCK_BODIES/GC_BEDROCK",
                    "GC_ROCK_BODIES/GC_UNCO_DESPOSIT",
                ],
                "transfer_fields": [
                    "UUID",
                    "GEOLCODE",
                    "GLAC_TYP",
                    "CHRONO_T",
                    "CHRONO_B",
                    "gmu_code",
                    "tecto",
                    "tecto_code",
                    "OPERATOR",
                    "DATEOFCHANGE",
                ],
                "area_threshold": 0.7,
                "buffer_distance": 0.5,
            },
            "POINT_GEOL": {
                "source_layers": [
                    "GC_ROCK_BODIES/GC_POINT_OBJECTS",
                    "GC_ROCK_BODIES/GC_FOSSILS",
                ],
                "transfer_fields": [
                    "UUID",
                    "POINT_TYPE",
                    "POINT_CATEGORY",
                    "FOSSIL_TYPE",
                    "OPERATOR",
                    "DATEOFCHANGE",
                ],
                "point_tolerance": 5.0,
            },
        },
    }

    # Add example sources if requested
    if example_sources:
        config_template["source_paths"].update(
            {"saas": "/path/to/Saas.gdb", "bkp_2016": "/path/to/BKP_2016.gdb"}
        )

    # Save configuration
    with open(output, "w") as f:
        yaml.dump(
            config_template, f, default_flow_style=False, sort_keys=False, indent=2
        )

    console.print(
        f"[green]✓ Created configuration template: [cyan]{output}[/cyan][/green]"
    )
    console.print("\nEdit the configuration file and run:")
    console.print(f"[bold]gcover publish enrich --config-file {output}[/bold]")


@publish_commands.command()
@click.pass_context
@click.option(
    "--admin-zones",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    help="Path to administrative_zones.gpkg (uses default if not specified)",
)
def list_mapsheets(ctx, admin_zones: Optional[Path]):
    """List available mapsheets and their source assignments."""

    if not admin_zones:
        try:
            admin_zones = files("gcover.data").joinpath("administrative_zones.gpkg")
        except:
            console.print(
                "[red]Error: Could not find default administrative_zones.gpkg[/red]"
            )
            console.print("Please specify path with --admin-zones")
            raise click.Abort()

    try:
        mapsheets = gpd.read_file(admin_zones, layer="mapsheets_sources_only")

        console.print(f"[bold]Available Mapsheets from {admin_zones}:[/bold]")
        console.print()

        table = Table()
        table.add_column("Map Number", justify="right", style="cyan")
        table.add_column("Map Title", style="green")
        table.add_column("Source", style="yellow")

        for _, row in mapsheets.sort_values("MSH_MAP_NBR").iterrows():
            table.add_row(
                str(row["MSH_MAP_NBR"]), row["MSH_MAP_TITLE"], row["SOURCE_RC"]
            )

        console.print(table)

        # Summary
        source_counts = mapsheets["SOURCE_RC"].value_counts()
        console.print()
        console.print("[bold]Summary:[/bold]")
        for source, count in source_counts.items():
            console.print(f"  {source}: {count} mapsheets")

        console.print(f"\nTotal: {len(mapsheets)} mapsheets")

    except Exception as e:
        console.print(f"[red]Error reading mapsheets: {e}[/red]")
        raise click.Abort()


@publish_commands.command()
@click.argument("tooltip_db", type=click.Path(exists=True, path_type=Path))
def list_layers(tooltip_db: Path):
    """List available layers in tooltip database."""

    try:
        import fiona

        layers = fiona.listlayers(str(tooltip_db))

        console.print(f"[bold]Available layers in {tooltip_db.name}:[/bold]")
        console.print()

        # Try to get feature counts and geometry types
        table = Table()
        table.add_column("Layer Name", style="cyan")
        table.add_column("Geometry Type", style="green")
        table.add_column("Feature Count", justify="right", style="yellow")

        for layer in layers:
            try:
                gdf = gpd.read_file(tooltip_db, layer=layer, rows=1)
                geom_type = (
                    gdf.geometry.geom_type.iloc[0] if not gdf.empty else "Unknown"
                )

                # Get full count
                full_gdf = gpd.read_file(tooltip_db, layer=layer)
                feature_count = len(full_gdf)

                table.add_row(layer, geom_type, str(feature_count))

            except Exception as e:
                table.add_row(layer, "Error", f"Error: {e}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error reading layers: {e}[/red]")
        raise click.Abort()


def load_enrichment_config_from_file(config_path: Path) -> EnrichmentConfig:
    """Load enrichment configuration from YAML file."""

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Convert paths
    source_paths = {k: Path(v) for k, v in config_data["source_paths"].items()}

    # Load layer mappings if specified
    layer_mappings = None
    if "layer_mappings" in config_data:
        layer_mappings = {}
        for layer_name, layer_config in config_data["layer_mappings"].items():
            # Determine layer type from geometry
            if "POLYGON" in layer_name.upper():
                layer_type = LayerType.POLYGON
            elif "LINE" in layer_name.upper():
                layer_type = LayerType.LINE
            elif "POINT" in layer_name.upper():
                layer_type = LayerType.POINT
            else:
                layer_type = LayerType.POLYGON  # Default

            layer_mappings[layer_name] = LayerMapping(
                tooltip_layer=layer_name,
                source_layers=layer_config["source_layers"],
                layer_type=layer_type,
                transfer_fields=layer_config["transfer_fields"],
                area_threshold=layer_config.get("area_threshold", 0.7),
                buffer_distance=layer_config.get("buffer_distance", 0.5),
                point_tolerance=layer_config.get("point_tolerance", 5.0),
            )

    return EnrichmentConfig(
        tooltip_db_path=Path(config_data["tooltip_db_path"]),
        admin_zones_path=Path(config_data["admin_zones_path"]),
        source_paths=source_paths,
        output_path=Path(config_data["output_path"])
        if config_data.get("output_path")
        else None,
        debug_output_dir=Path(config_data["debug_output_dir"])
        if config_data.get("debug_output_dir")
        else None,
        save_intermediate=config_data.get("save_intermediate", False),
        mapsheet_numbers=config_data.get("mapsheet_numbers"),
        clip_tolerance=config_data.get("clip_tolerance", 0.0),
        layer_mappings=layer_mappings,
    )


def validate_enrichment_config(config: EnrichmentConfig) -> None:
    """Validate enrichment configuration."""

    # Check required paths exist
    if not config.tooltip_db_path.exists():
        raise ValueError(f"Tooltip database not found: {config.tooltip_db_path}")

    if not config.admin_zones_path.exists():
        raise ValueError(f"Admin zones file not found: {config.admin_zones_path}")

    # Check at least one source exists
    existing_sources = [
        name for name, path in config.source_paths.items() if path.exists()
    ]
    if not existing_sources:
        raise ValueError("No valid source paths found")

    # Warn about missing sources
    missing_sources = [
        name for name, path in config.source_paths.items() if not path.exists()
    ]
    if missing_sources:
        logger.warning(f"Missing source paths: {missing_sources}")


def show_config_summary(
    config: EnrichmentConfig, layers_to_process: Optional[List[str]], dry_run: bool
):
    """Display configuration summary."""

    title = (
        "🔍 Enrichment Configuration (Dry Run)"
        if dry_run
        else "⚙️  Enrichment Configuration"
    )

    # Create summary table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Tooltip Database", str(config.tooltip_db_path))
    table.add_row("Admin Zones", str(config.admin_zones_path))

    # Source paths
    for source_name, source_path in config.source_paths.items():
        status = "✓" if source_path.exists() else "✗"
        table.add_row(f"Source: {source_name}", f"{status} {source_path}")

    # Processing parameters
    if config.mapsheet_numbers:
        table.add_row(
            "Mapsheets",
            f"{len(config.mapsheet_numbers)} specified: {config.mapsheet_numbers[:5]}{'...' if len(config.mapsheet_numbers) > 5 else ''}",
        )
    else:
        table.add_row("Mapsheets", "All available")

    if layers_to_process:
        table.add_row(
            "Layers", f"{len(layers_to_process)} specified: {layers_to_process}"
        )
    else:
        table.add_row("Layers", "All configured layers")

    # Output settings
    if config.output_path:
        table.add_row("Output Path", str(config.output_path))

    if config.debug_output_dir:
        table.add_row("Debug Output", str(config.debug_output_dir))
        table.add_row("Save Intermediate", "Yes" if config.save_intermediate else "No")

    console.print(table)


def show_enrichment_summary(results: Dict[str, gpd.GeoDataFrame]):
    """Display enrichment results summary."""

    console.print("\n📊 Enrichment Results Summary")

    # Results table
    table = Table()
    table.add_column("Layer", style="cyan")
    table.add_column("Total Features", justify="right", style="green")
    table.add_column("Enriched", justify="right", style="yellow")
    table.add_column("Success Rate", justify="right", style="magenta")
    table.add_column("Mapsheets", justify="right", style="blue")

    total_features = 0
    total_enriched = 0

    for layer_name, gdf in results.items():
        if gdf.empty:
            table.add_row(layer_name, "0", "0", "0%", "0")
            continue

        feature_count = len(gdf)
        enriched_count = (
            len(gdf[gdf["SOURCE_UUID"].notna()]) if "SOURCE_UUID" in gdf.columns else 0
        )
        success_rate = (
            (enriched_count / feature_count * 100) if feature_count > 0 else 0
        )
        mapsheet_count = (
            gdf["MAPSHEET_NBR"].nunique() if "MAPSHEET_NBR" in gdf.columns else 0
        )

        table.add_row(
            layer_name,
            str(feature_count),
            str(enriched_count),
            f"{success_rate:.1f}%",
            str(mapsheet_count),
        )

        total_features += feature_count
        total_enriched += enriched_count

    # Add total row
    if len(results) > 1:
        overall_success_rate = (
            (total_enriched / total_features * 100) if total_features > 0 else 0
        )
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_features}[/bold]",
            f"[bold]{total_enriched}[/bold]",
            f"[bold]{overall_success_rate:.1f}%[/bold]",
            "-",
        )

    console.print(table)

    # Show sample of enriched data for first non-empty result
    for layer_name, gdf in results.items():
        if not gdf.empty and "SOURCE_UUID" in gdf.columns:
            enriched_sample = gdf[gdf["SOURCE_UUID"].notna()]
            if not enriched_sample.empty:
                show_sample_data(layer_name, enriched_sample.head(3))
                break


def show_sample_data(layer_name: str, sample_gdf: gpd.GeoDataFrame):
    """Show sample of enriched data."""

    console.print(f"\n🔍 Sample Enriched Data ({layer_name})")

    # Select interesting columns to display
    display_columns = []
    for col in [
        "OBJECTID",
        "GEOLCODE",
        "gmu_code",
        "tecto",
        "SOURCE_UUID",
        "MATCH_METHOD",
        "MATCH_CONFIDENCE",
        "MAPSHEET_NBR",
    ]:
        if col in sample_gdf.columns:
            display_columns.append(col)

    if not display_columns:
        console.print("No relevant columns to display")
        return

    table = Table()
    for col in display_columns:
        table.add_column(col, style="dim", max_width=20, overflow="fold")

    for _, row in sample_gdf.iterrows():
        values = []
        for col in display_columns:
            val = row.get(col, "")
            if pd.isna(val):
                val = "NULL"
            values.append(str(val)[:18])  # Truncate long values
        table.add_row(*values)

    console.print(table)
