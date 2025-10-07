# src/gcover/cli/publish_cmd.py
"""
Enhanced CLI commands for preparing GeoCover data for publication.
Supports multiple tooltip layers, flexible source mappings, and comprehensive configuration.
"""

import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import geopandas as gpd
import pandas as pd
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gcover.config import SDE_INSTANCES, AppConfig, load_config
from gcover.publish.esri_classification_applicator import \
    ClassificationApplicator
from gcover.publish.style_config import (BatchClassificationConfig,
                                         apply_batch_from_config)
from gcover.publish.tooltips_enricher import (EnhancedTooltipsEnricher,
                                              EnrichmentConfig, LayerMapping,
                                              LayerType,
                                              create_enrichment_config)

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


'''
@publish_commands.command()
@click.pass_context
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
def cli(verbose: bool, quiet: bool):
    """🎨 Classification Symbol Applicator

    Apply ESRI classification rules to GeoDataFrames/GPKG files.
    Adds a SYMBOL field with generated class identifiers based on classification rules.
    """
    if quiet:
        logger.remove()
        logger.add(sys.stdout, level="ERROR", format="<red>{level}</red>: {message}")
    elif verbose:
        logger.remove()
        logger.add(
            sys.stdout,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )'''


@publish_commands.command()
@click.pass_context
@click.argument("gpkg_file", type=click.Path(exists=True, path_type=Path))
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--layer", "-l", help="Specific layer to process (default: all layers in config)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output GPKG path (default: input_classified.gpkg)",
)
@click.option(
    "--styles-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Base directory for resolving relative style paths (default: config file directory)",
)
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option(
    "--dry-run", is_flag=True, help="Parse config without applying classifications"
)
def apply_config(
    ctx,
    gpkg_file: Path,
    config_file: Path,
    layer: Optional[str],
    output: Optional[Path],
    styles_dir: Optional[Path],
    debug: bool,
    dry_run: bool,
):
    """Apply multiple classifications from YAML configuration file.

    This command processes a GPKG using a YAML configuration that specifies
    which style files to apply to which layers, with field mappings and filters.

    \b
    Example config structure:
      global:
        treat_zero_as_null: true
        symbol_field: SYMBOL
        label_field: LABEL
      layers:
        - gpkg_layer: GC_POINT_OBJECTS
          classifications:
            - style_file: styles/springs.lyrx
              classification_name: Quelle
              filter: KIND == 12501001
              symbol_prefix: spring
            - style_file: styles/boreholes.lyrx
              filter: KIND == 12501002
              symbol_prefix: borehole

    \b
    Examples:
      # Apply all classifications from config
      classifier apply-config geocover.gpkg config.yaml

      # Process only specific layer
      classifier apply-config geocover.gpkg config.yaml -l GC_POINT_OBJECTS

      # Specify styles directory
      classifier apply-config data.gpkg config.yaml --styles-dir /path/to/styles

      # Dry run to validate config
      classifier apply-config geocover.gpkg config.yaml --dry-run
    """
    try:
        console.print(f"\n[bold blue]📋 Batch Classification from Config[/bold blue]\n")

        # Load configuration
        with console.status("[cyan]Loading configuration...", spinner="dots"):
            config = BatchClassificationConfig(config_file, styles_dir)

        console.print(f"[green]✓[/green] Loaded configuration:")
        console.print(f"  • Layers: {len(config.layers)}")
        console.print(f"  • Symbol field: {config.symbol_field}")
        console.print(f"  • Label field: {config.label_field}")
        console.print(f"  • Treat 0 as NULL: {config.treat_zero_as_null}")

        # Display layer summary
        table = Table(title="Configuration Summary", show_header=True)
        table.add_column("GPKG Layer", style="cyan")
        table.add_column("Classifications", style="yellow", justify="right")
        table.add_column("Style Files", style="dim")

        for layer_config in config.layers:
            style_files = [c.style_file.name for c in layer_config.classifications]
            table.add_row(
                layer_config.gpkg_layer,
                str(len(layer_config.classifications)),
                ", ".join(style_files[:3]) + ("..." if len(style_files) > 3 else ""),
            )

        console.print(table)

        if dry_run:
            console.print("\n[yellow]🔍 Dry run - no changes will be made[/yellow]")

            # Validate that style files exist
            console.print("\nValidating style files...")
            all_valid = True
            for layer_config in config.layers:
                for class_config in layer_config.classifications:
                    if not class_config.style_file.exists():
                        console.print(
                            f"  [red]✗ Missing: {class_config.style_file}[/red]"
                        )
                        all_valid = False
                    else:
                        console.print(
                            f"  [green]✓ Found: {class_config.style_file.name}[/green]"
                        )

            if all_valid:
                console.print("\n[green]✓ Configuration is valid![/green]")
            else:
                console.print("\n[red]✗ Configuration has errors[/red]")
            return

        # Apply classifications
        stats = apply_batch_from_config(
            gpkg_path=gpkg_file,
            config=config,
            layer_name=layer,
            output_path=output,
            debug=debug,
        )

        # Display final statistics
        console.print("\n[bold green]✅ Batch processing complete![/bold green]\n")

        summary_table = Table(title="Processing Statistics", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("Layers processed", str(stats["layers_processed"]))
        summary_table.add_row(
            "Classifications applied", str(stats["classifications_applied"])
        )
        summary_table.add_row("Features classified", str(stats["features_classified"]))
        summary_table.add_row("Total features", str(stats["features_total"]))

        if stats["features_total"] > 0:
            pct = stats["features_classified"] / stats["features_total"] * 100
            summary_table.add_row("Coverage", f"{pct:.1f}%")

        console.print(summary_table)

        output_file = output or gpkg_file.parent / f"{gpkg_file.stem}_classified.gpkg"
        console.print(f"\n[dim]Output: {output_file}[/dim]")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if debug:
            import traceback

            logger.debug(traceback.format_exc())
        raise


@publish_commands.command()
@click.pass_context
@click.argument(
    "output_path", type=click.Path(path_type=Path), default="classification_config.yaml"
)
@click.option(
    "--example",
    type=click.Choice(["simple", "complex"]),
    default="complex",
    help="Type of example to generate",
)
def create_config(ctx, output_path: Path, example: str):
    """Create an example YAML configuration file for batch processing.

    \b
    Examples:
      # Create simple example
      classifier create-config config.yaml --example simple

      # Create complex example with multiple layers
      classifier create-config config.yaml --example complex
    """
    if example == "simple":
        config = {
            "global": {
                "treat_zero_as_null": True,
                "symbol_field": "SYMBOL",
                "label_field": "LABEL",
            },
            "layers": [
                {
                    "gpkg_layer": "GC_FOSSILS",
                    "classifications": [
                        {
                            "style_file": "styles/Fossils.lyrx",
                            "classification_name": "Fossils",
                            "filter": "KIND == 14601006",
                            "symbol_prefix": "fossil",
                            "fields": {
                                "KIND": "KIND",
                                "LFOS_DIVISION": "LFOS_DIVISION",
                                "LFOS_STATUS": "LFOS_STATUS",
                            },
                        }
                    ],
                }
            ],
        }
    else:  # complex
        config = {
            "global": {
                "treat_zero_as_null": True,
                "symbol_field": "SYMBOL",
                "label_field": "LABEL",
                "overwrite": False,
            },
            "layers": [
                {
                    "gpkg_layer": "GC_POINT_OBJECTS",
                    "classifications": [
                        {
                            "style_file": "styles/Point_Objects_Quelle.lyrx",
                            "classification_name": "Quelle",
                            "filter": "KIND == 12501001",
                            "symbol_prefix": "spring",
                            "fields": {
                                "KIND": "KIND",
                                "HSUR_TYPE": "HSUR_TYPE",
                                "HSUR_STATUS": "HSUR_STATUS",
                            },
                        },
                        {
                            "style_file": "styles/Point_Objects_Bohrung_Fels_erreicht.lyrx",
                            "classification_name": "Bohrung Fels erreicht",
                            "filter": "KIND == 12501002 and LBOR_ROCK_REACHED == 1",
                            "symbol_prefix": "borehole_rock",
                        },
                        {
                            "style_file": "styles/Point_Objects_Erraticker.lyrx",
                            "classification_name": "Erraticker",
                            "filter": "KIND == 14601008",
                            "symbol_prefix": "erratic",
                        },
                    ],
                },
                {
                    "gpkg_layer": "GC_FOSSILS",
                    "classifications": [
                        {
                            "style_file": "styles/Fossils.lyrx",
                            "filter": "KIND == 14601006",
                            "symbol_prefix": "fossil",
                        }
                    ],
                },
            ],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    console.print(f"[green]✓ Created example configuration: {output_path}[/green]")
    console.print(f"\nEdit this file to match your layers and style files, then run:")
    console.print(
        f"[cyan]  classifier apply-config your_data.gpkg {output_path}[/cyan]"
    )


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
                result = {layer_name: enriched_data}
                # Save results
                if config.output_path:
                    output_path = enricher.save_enriched_data(result)
                    console.print(f"💾 Saved enriched {layer_name} to: {output_path}")

                # Show results summary
                show_enrichment_summary(result)

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
@click.pass_context
@click.argument("style_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output mapfile path",
)
@click.option("--layer-name", "-l", required=True, help="MapServer layer name")
@click.option(
    "--data-path", "-d", required=True, help="Data source path (OGR connection string)"
)
@click.option(
    "--layer-type",
    "-t",
    type=click.Choice(["POLYGON", "LINE", "POINT"]),
    default="POLYGON",
    help="Geometry type",
)
@click.option("--classification-name", "-c", help="Specific classification to use")
@click.option(
    "--symbol-prefix",
    "-p",
    default="class",
    help='Prefix for symbol IDs (e.g., "bedrock", "unco")',
)
@click.option(
    "--use-symbol-field",
    is_flag=True,
    help="Use SYMBOL field instead of complex expressions",
)
@click.option(
    "--symbol-field", default="SYMBOL", help="Name of symbol field (default: SYMBOL)"
)
def mapserver(
    ctx,
    style_file,
    output,
    layer_name,
    data_path,
    layer_type,
    classification_name,
    symbol_prefix,
    use_symbol_field,
    symbol_field,
):
    """Generate MapServer mapfile CLASS sections.

    \b
    Two modes:
    1. Field-based (default): Complex expressions using classification fields
    2. Symbol-based (--use-symbol-field): Simple expressions using SYMBOL field

    \b
    Examples:
      # Field-based (complex expressions)
      style-gen mapserver Bedrock.lyrx -o bedrock.map \\
        -l gc_bedrock -d "data/geocover.gpkg,layer=GC_BEDROCK" -t POLYGON

      # Symbol-based (simple expressions, requires pre-classified data)
      style-gen mapserver Bedrock.lyrx -o bedrock.map \\
        -l gc_bedrock -d "data/geocover.gpkg,layer=GC_BEDROCK" -t POLYGON \\
        --use-symbol-field --symbol-prefix bedrock
    """
    # Load classification
    console.print(f"Loading classification from {style_file}...")
    classifications = extract_lyrx(style_file, display=False)

    # Select classification
    if classification_name:
        classification = next(
            (c for c in classifications if c.layer_name == classification_name), None
        )
        if not classification:
            console.print(
                f"[red]Classification '{classification_name}' not found[/red]"
            )
            return
    elif len(classifications) == 1:
        classification = classifications[0]
    else:
        console.print(
            "[red]Multiple classifications found, specify --classification-name[/red]"
        )
        return

    # Generate mapfile
    generator = MapServerGenerator(
        layer_type=layer_type,
        use_symbol_field=use_symbol_field,
        symbol_field=symbol_field,
    )

    mapfile_content = generator.generate_layer(
        classification=classification,
        layer_name=layer_name,
        data_path=data_path,
        symbol_prefix=symbol_prefix,
    )

    # Save
    output.write_text(mapfile_content)
    console.print(f"[green]✓ Generated MapServer layer: {output}[/green]")
    console.print(f"  Classes: {len([c for c in classification.classes if c.visible])}")
    console.print(
        f"  Mode: {'SYMBOL field' if use_symbol_field else 'Field expressions'}"
    )

    # Generate symbol file
    symbol_file = output.parent / "symbols.sym"
    if not symbol_file.exists():
        symbol_content = generator.generate_symbol_file()
        symbol_file.write_text(symbol_content)
        console.print(f"[green]✓ Generated symbol file: {symbol_file}[/green]")


@publish_commands.command()
@click.pass_context
@click.argument("style_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output QML file path",
)
@click.option(
    "--geometry-type",
    "-t",
    type=click.Choice(["Polygon", "Line", "Point"]),
    default="Polygon",
    help="Geometry type",
)
@click.option("--classification-name", "-c", help="Specific classification to use")
@click.option(
    "--symbol-prefix",
    "-p",
    default="class",
    help='Prefix for symbol IDs (e.g., "bedrock", "unco")',
)
@click.option(
    "--use-symbol-field",
    is_flag=True,
    help="Use SYMBOL field instead of complex expressions",
)
@click.option(
    "--symbol-field", default="SYMBOL", help="Name of symbol field (default: SYMBOL)"
)
def qgis(
    ctx,
    style_file,
    output,
    geometry_type,
    classification_name,
    symbol_prefix,
    use_symbol_field,
    symbol_field,
):
    """Generate QGIS QML style file.

    \b
    Examples:
      # Field-based rules
      style-gen qgis Bedrock.lyrx -o bedrock.qml -t Polygon

      # Symbol-based rules (simpler, faster)
      style-gen qgis Bedrock.lyrx -o bedrock.qml -t Polygon \\
        --use-symbol-field --symbol-prefix bedrock
    """
    # Load classification
    console.print(f"Loading classification from {style_file}...")
    classifications = extract_lyrx(style_file, display=False)

    # Select classification
    if classification_name:
        classification = next(
            (c for c in classifications if c.layer_name == classification_name), None
        )
        if not classification:
            console.print(
                f"[red]Classification '{classification_name}' not found[/red]"
            )
            return
    elif len(classifications) == 1:
        classification = classifications[0]
    else:
        console.print(
            "[red]Multiple classifications found, specify --classification-name[/red]"
        )
        return

    # Generate QML
    generator = QGISGenerator(
        geometry_type=geometry_type,
        use_symbol_field=use_symbol_field,
        symbol_field=symbol_field,
    )

    qml_content = generator.generate_qml(classification, symbol_prefix)

    # Save
    output.write_text(qml_content)
    console.print(f"[green]✓ Generated QGIS style: {output}[/green]")
    console.print(f"  Rules: {len([c for c in classification.classes if c.visible])}")
    console.print(
        f"  Mode: {'SYMBOL field' if use_symbol_field else 'Field expressions'}"
    )


@publish_commands.command()
@click.pass_context
@click.argument("tooltip_db", type=click.Path(exists=True, path_type=Path))
def list_layers(ctx, tooltip_db: Path):
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
                show_sample_data(layer_name, enriched_sample.head(10))
                break


def show_sample_data(layer_name: str, sample_gdf: gpd.GeoDataFrame):
    """Show sample of enriched data."""

    console.print(f"\n🔍 Sample Enriched Data ({layer_name})")

    # Select interesting columns to display
    display_columns = []
    for col in [
        "OBJECTID",
        "DESCRIPT_DE",
        "gmu_code",
        "tecto",
        "SOURCE_UUID",
        "MATCH_METHOD",
        "MATCH_LAYER",
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
