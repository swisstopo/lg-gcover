# src/gcover/cli/publish_cmd.py
"""
Enhanced CLI commands for preparing GeoCover data for publication.
Supports multiple tooltip layers, flexible source mappings, and comprehensive configuration.
"""

import json
import os
import sys
import time
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

from gcover.cli.main import _split_bbox
from gcover.config import (DEFAULT_EXCLUDED_FIELDS, SDE_INSTANCES, AppConfig,
                           load_config)
from gcover.publish.esri_classification_applicator import \
    ClassificationApplicator
from gcover.publish.esri_classification_extractor import (
    ClassificationJSONEncoder, explore_layer_structure,
    export_classifications_to_csv, extract_lyrx_complete, to_serializable_dict)
from gcover.publish.generator import MapServerGenerator
from gcover.publish.merge_sources import (GDBMerger, MergeConfig,
                                          create_merge_config)
from gcover.publish.qgis_generator import QGISGenerator
from gcover.publish.style_config import (BatchClassificationConfig,
                                         apply_batch_from_config)
from gcover.publish.tooltips_enricher import (EnhancedTooltipsEnricher,
                                              EnrichmentConfig, LayerMapping,
                                              LayerType,
                                              create_enrichment_config)

DEFAULT_ZONES_PATH = files("gcover.data").joinpath("administrative_zones.gpkg")

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
    # Ensure context object exists and has required keys
    if ctx.obj is None:
        ctx.ensure_object(dict)

    # Set defaults if not provided by parent gcover command
    ctx.obj.setdefault("environment", "development")
    ctx.obj.setdefault("verbose", False)
    ctx.obj.setdefault("config_path", None)


@publish_commands.command()
@click.pass_context
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.option("--quiet", is_flag=True, help="Suppress rich display")
@click.option(
    "--explore",
    is_flag=True,
    help="Explore layer structure (show all layers and groups)",
)
@click.option(
    "--export", type=click.Choice(["json", "csv"]), help="Export results to file"
)
@click.option(
    "--max-label-length",
    type=int,
    default=130,
    help="Maximum label length (default: 40)",
)
@click.option(
    "--head",
    type=int,
    default=None,
    help="Display only the first n rows",
)
def extract_classification(
    ctx,
    input,
    quiet,
    explore,
    export,
    max_label_length,
    head,
):
    """Extract ESRI layer classification information from .lyrx files."""
    logger.info(f"COMMAND START: apply-config")
    verbose = ctx.obj.get("verbose", False)
    environnement = ctx.obj.get('environment')
    use_arcpy = False

    if verbose and quiet:
        console.print(
            "[red]Error: options `--verbose` and `--quiet`are mutually exclusive[/red]"
        )
        raise click.Abort()

    input_path = Path(input)

    if explore:
        if input_path.suffix.lower() != ".lyrx":
            console.print(
                "[red]Error: --explore only supports .lyrx files currently[/red]"
            )
            raise click.Abort()

        structure = explore_layer_structure(input_path)
        # You can optionally display or return structure here

    elif input_path.suffix.lower() == ".lyrx":
        results = extract_lyrx_complete(  # TODO switched from extract_lyrx
            input_path,
            use_arcpy=use_arcpy,
            display=not quiet,
            head=head,
            #  max_label_length=max_label_length,
        )

        if quiet:
            console.print(f"Extracted {len(results)} layer classifications")

        if export:
            export_path = input_path.with_suffix(f".classifications.{export}")

            if export == "json":
                export_data = [to_serializable_dict(c) for c in results]
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(
                        export_data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        cls=ClassificationJSONEncoder,
                    )
                console.print(f"[green]Exported to {export_path}[/green]")

            elif export == "csv":
                export_classifications_to_csv(results, export_path)
                console.print(f"[green]Exported to {export_path}[/green]")

    else:
        console.print("[red]Error: Input must be .lyrx[/red]")
        raise click.Abort()


@publish_commands.command()
@click.pass_context
@click.argument("gpkg_file", type=click.Path(exists=True, path_type=Path))
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--layer", "-l", help="Specific layer to process (default: all layers in config)"
)
@click.option(
    "-b",
    "--bbox",
    required=False,
    nargs=1,
    type=click.STRING,
    callback=_split_bbox,
    default=None,
    help="Filter with a bbox",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output GPKG path (default: input_classified.gpkg)",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue processing other assets if one fails",
)
@click.option(
    "--styles-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Base directory for resolving relative style paths (default: config file directory)",
)
@click.option(
    "--dry-run", is_flag=True, help="Parse config without applying classifications"
)
@click.option("--overwrite", is_flag=True, help="Overwrite the classification field")
def apply_config(
    ctx,
    gpkg_file: Path,
    config_file: Path,
    layer: Optional[str],
    output: Optional[Path],
    styles_dir: Optional[Path],
    dry_run: bool,
    bbox: Optional[tuple],
    continue_on_error: bool,
    overwrite: bool,
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
      gcover publish apply-config geocover.gpkg config.yaml

      # Process only specific layer
      gcover publish apply-config geocover.gpkg config.yaml -l GC_POINT_OBJECTS

      # Specify styles directory
      gcover publish apply-config data.gpkg config.yaml --styles-dir /path/to/styles

      # Dry run to validate config
      gcover publish apply-config geocover.gpkg config.yaml --dry-run
    """
    verbose = ctx.obj.get("verbose", False)
    environnement = ctx.obj.get('environment')

    logger.info(f"COMMAND START: apply-config")

    if verbose:
        console.print("[dim]Verbose logging enabled[/dim]")
    try:
        console.print(f"\n[bold blue]📋 Batch Classification from Config[/bold blue]\n")

        # Load configuration
        with console.status("[cyan]Loading configuration...", spinner="dots"):
            config = BatchClassificationConfig(config_file, styles_dir, environnement)



        # TODO
        from pprint import pprint
        pprint(config.raw_config.get('layers')[0])
        print(yaml.safe_dump(config.raw_config.get('layers')[1], sort_keys=False))

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

        start_time = time.time()

        # Apply classifications
        stats = apply_batch_from_config(
            gpkg_path=gpkg_file,
            config=config,
            layer_name=layer,
            output_path=output,
            debug=verbose,
            bbox=bbox,
            continue_on_error=continue_on_error,
            preserve_existing=True,
            overwrite=overwrite,
        )

        end_time = time.time()
        elapsed = end_time - start_time
        mins, secs = divmod(elapsed, 60)

        # Check if stats is empty (indicating an error occurred)
        if not stats:
            error_message = (
                "No processing statistics available. This usually indicates:\n"
                "• An error occurred during processing\n"
                "• No features were found matching your criteria\n"
                "• There was an issue with the input data or configuration"
            )

            console.print("\n[bold red]❌ Batch processing failed![/bold red]")
            console.print(
                Panel(
                    error_message,
                    title="Error Details",
                    title_align="left",
                    border_style="red",
                )
            )

            # Rest of troubleshooting table...
            troubleshooting_table = Table(
                title="Troubleshooting Tips", show_header=True
            )
            troubleshooting_table.add_column("Check", style="cyan")
            troubleshooting_table.add_column("Action", style="white")

            troubleshooting_table.add_row(
                "Input file", "Verify the GeoPackage exists and is accessible"
            )
            troubleshooting_table.add_row(
                "Layer name", f"Confirm layer '{layer}' exists in the file"
            )
            troubleshooting_table.add_row(
                "Configuration", "Check your classification rules are valid"
            )
            troubleshooting_table.add_row(
                "Bounding box", "Verify bbox coordinates if used"
            )
            troubleshooting_table.add_row(
                "Debug mode", "Run with --debug for detailed error information"
            )

            console.print(troubleshooting_table)
            console.print(
                f"\n[yellow]Processing time: {int(mins)}m {secs:.1f}s[/yellow]"
            )

            sys.exit(1)

        # Display final statistics
        # vectorized stats are print in this file
        if stats.get("features_newly_classified"):
            console.print("\n[bold green]✅ Batch processing complete![/bold green]\n")

            summary_table = Table(title="Processing Statistics", show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green", justify="right")

            summary_table.add_row("Layers processed", str(stats["layers_processed"]))
            summary_table.add_row(
                "Classifications applied", str(stats["classifications_applied"])
            )
            summary_table.add_row(
                "Features newly classified", str(stats["features_newly_classified"])
            )
            summary_table.add_row(
                "Features classified", str(stats["features_classified"])
            )
            summary_table.add_row("Total features", str(stats["features_total"]))

            if stats["features_total"] > 0:
                pct = stats["features_classified"] / stats["features_total"] * 100
                summary_table.add_row("Coverage", f"{pct:.1f}%")
            summary_table.add_row("Processing time", f"{int(mins)}m {secs:.1f}s")

            console.print(summary_table)

            logger.info(
                "SUMMARY: Layers processed {layers_processed} | "
                "Classifications applied {classifications_applied} | "
                "Features classified {features_classified} |"
                "Total features {features_total} |".format(**stats)
            )
            output_file = (
                output or gpkg_file.parent / f"{gpkg_file.stem}_classified.gpkg"
            )
            console.print(f"\n[dim]Output: {output_file}[/dim]")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        raise
    finally:
        logger.info(f"COMMAND END: apply-config")


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
def create_classification_config(ctx, output_path: Path, example: str):
    """Create an example YAML configuration file for batch classification.

    \b
    Examples:
      # Create simple example
      gcover publish create-classification-config config.yaml --example simple

      # Create complex example with multiple layers
      gcover publish create-classification-config config.yaml --example complex
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
        f"[cyan]  gcover publish apply-config your_data.gpkg {output_path}[/cyan]"
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
    default=DEFAULT_ZONES_PATH,
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
    default=DEFAULT_ZONES_PATH,
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
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for mapfiles",
)
# YAML configuration file (extracts prefixes and layer names automatically",
@click.argument(
    "config-file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--connection",
    "connection_name",
    type=str,
    default="postgis_wms",
    help="Database connection to use (as defined in the main YAML config)",
)
@click.option(
    "--use-symbol-field",
    is_flag=True,
    help="Use SYMBOL field instead of complex expressions",
)
@click.option(
    "--symbol-field", default="SYMBOL", help="Name of symbol field (default: SYMBOL)"
)
@click.option(
    "--styles-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Base directory for resolving relative style paths (default: config file directory)",
)
@click.option(
    "--prefixes",
    "-p",
    help="Layer name to prefix mapping as JSON (overrides config file)",
)
@click.option(
    "--generate-combined",
    is_flag=True,
    help="Generate combined symbol file and fontset for all layers",
)
def mapserver(
    ctx,
    output_dir: Path,
    config_file: Optional[Path],
    styles_dir: Optional[Path],
    use_symbol_field: bool,
    symbol_field: str,
    prefixes: Optional[str],
    generate_combined: bool,
    connection_name: Optional[str] = None,
):
    """Generate MapServer mapfiles from ESRI style files.

    Processes multiple style files and generates individual mapfiles plus
    optional combined symbol file and fontset with proper symbol tracking.

    \b
    Examples:
      # Using configuration file (recommended)
      gcover publish mapserver -o output/ --generate-combined \\
          config/esri_classifier.yaml

      # Manual with JSON prefixes
      gcover publish mapserver --style-dir styles/ -o output/ -t Polygon \\
        --use-symbol-field --generate-combined \\
        --prefixes '{"Bedrock":"bedr","Surfaces":"surf","Fossils":"foss"}' \\
        config/esri_classifier.yaml


    """
    layer_type = None

    publish_config, global_config = get_publish_config(ctx)

    try:
        connections = global_config.mapserver.connections
    except:
        raise click.BadParameter(
            f"Cannot find mapserver connections for environment: {ctx.obj['environment']}"
        )

    if connection_name and connection_name in global_config.mapserver.connections:
        connections = global_config.mapserver.connections
        connection = connections[connection_name]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration if provided
    prefix_map = {}
    mapfile_names = {}
    mapfile_groups = {}
    mapfile_labels = {}
    active_classes = {}
    mapfiles = []
    config = None
    symbol_field = None

    if config_file:
        console.print(f"[cyan]Loading configuration from {config_file}[/cyan]")
        config = BatchClassificationConfig(config_file)

        identifier_fields = {}

        symbol_field = config.symbol_field.lower()
        label_field = config.label_field.lower()

        # Extract prefixes and mapfile names from config
        for layer_config in config.layers:
            for class_config in sorted(
                layer_config.classifications, key=lambda x: x.index
            ):
                if class_config.classification_name:
                    # Use classification name as key
                    key = class_config.classification_name
                    mapfile_labels[key] = class_config.map_label
                    active_classes[key] = class_config.active
                    if class_config.symbol_prefix:
                        prefix_map[key] = class_config.symbol_prefix
                    if class_config.mapfile_group:
                        mapfile_groups[key] = class_config.mapfile_group
                    if class_config.mapfile_name:
                        mapfile_names[key] = class_config.mapfile_name

                    if class_config.identifier_field:
                        field_name = class_config.identifier_field
                        identifier_fields[key] = field_name
                        logger.debug(
                            f"Layer '{key}' will use identifier_field: {field_name}"
                        )

        console.print(
            f"  [green]✓[/green] Extracted prefixes for {len(prefix_map)} classifications"
        )

        # If no style files specified, use all from config
        style_files = []
        for layer_config in config.layers:
            layer_type = layer_config.layer_type
            gpkg_layer = layer_config.gpkg_layer
            connection_ref = layer_config.connection_ref
            for class_config in layer_config.classifications:
                if class_config.style_file not in style_files:
                    params = (
                        class_config.style_file,
                        layer_type,
                        gpkg_layer,
                        connection_ref,
                        class_config.data,
                        layer_config.template,
                        layer_config.max_scale,
                    )
                    logger.debug(params)
                    style_files.append(params)

        console.print(
            f"  [green]✓[/green] Found {len(style_files)} style files in config"
        )

    # Override with manual prefixes if provided
    if prefixes:
        import json

        try:
            manual_prefixes = json.loads(prefixes)
            prefix_map.update(manual_prefixes)
            console.print(f"[yellow]Applied manual prefix overrides[/yellow]")
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing prefixes JSON: {e}[/red]")
            raise click.Abort()

    if not style_files:
        console.print(
            "[red]Error: No style files specified. Use style_files argument or --config-file[/red]"
        )
        raise click.Abort()

    # Store all classifications for combined symbol file
    all_classifications = []
    generated_files = []

    console.print(f"\n[bold blue]🗺️  Generating MapServer Mapfiles[/bold blue]\n")

    # Single generator for all layers (to track symbols across layers)
    generator = MapServerGenerator(
        layer_type=layer_type.name,  # TODO remove
        use_symbol_field=use_symbol_field,
        symbol_field=symbol_field,
    )

    for (
        style_file,
        layer_type,
        gpkg_layer,
        connection_ref,
        mapserver_data,
        template,
        layer_max_scale,
    ) in style_files:
        console.print(f"Processing {style_file.name} [{layer_type}]...")

        if connection_ref and connections.get(connection_ref):
            connection = connections.get(connection_ref)

        if not connection:
            raise click.Abort("Not connection found. Aborting")

        lyrx_path = styles_dir / style_file.name

        generator.layer_type = layer_type.name
        # Load classifications
        classifications = extract_lyrx_complete(
            lyrx_path,
            display=False,
            identifier_fields=identifier_fields,  # ← NEW
        )  # TODO switched from extract_lyrx

        if not classifications:
            console.print(
                f"[yellow]⚠ No classifications found in {style_file.name}[/yellow]"
            )
            continue

        # Process each classification

        for classification in classifications:
            layer_name = classification.layer_name or style_file.stem

            # Get prefix and mapfile name from config if available
            symbol_prefix = prefix_map.get(layer_name, layer_name.lower())
            mapfile_layer_name = mapfile_names.get(layer_name, layer_name)
            mapfile_layer_group = mapfile_groups.get(layer_name, None)
            mapfile_label = mapfile_labels.get(layer_name, None)
            is_active = active_classes.get(layer_name)

            if not is_active:
                console.print(
                    f"  [bold orange1]Skipping {layer_name} (inactive)[/bold orange1]"
                )
                continue

            if not mapserver_data:
                mapserver_data = f"geom from (SELECT * from {gpkg_layer} ) as blabla using unique gid using srid=2056"
                mapserver_data = f"geom FROM  geol.geocover_{gpkg_layer}  USING UNIQUE gid USING SRID=2056"

            # Generate mapfile
            mapfile_content = generator.generate_layer(
                classification=classification,
                layer_name=mapfile_layer_name,
                layer_group=mapfile_layer_group,
                symbol_prefix=symbol_prefix,
                data=mapserver_data,
                layer_type=layer_type,
                connection=connection,
                symbol_field=symbol_field,
                template=template,
                map_label=mapfile_label,
                layer_max_scale=layer_max_scale,
            )

            # Save mapfile
            mapfile = f"{mapfile_layer_name}.map"
            output_file = output_dir / mapfile
            output_file.write_text(mapfile_content)
            generated_files.append(output_file)
            mapfiles.append(mapfile)

            console.print(
                f"  [green]✓[/green] Generated: {output_file.name} "
                f"(prefix: {symbol_prefix}, "
                f"{len([c for c in classification.classes if c.visible])} classes)"
            )

            # Store classification for combined symbol file
            all_classifications.append(classification)

    # Generate combined symbol file if requested
    if generate_combined and all_classifications:
        # Generate fontset
        fontset_content = generator.generate_fontset()
        fontset_file = output_dir / "fonts.txt"
        fontset_file.write_text(fontset_content)
        console.print(f"  [green]✓[/green] Fontset: {fontset_file}")

        # Check for PDF
        pdf_file = output_dir / "font_characters.pdf"
        if pdf_file.exists():
            console.print(f"  [green]✓[/green] Font symbols PDF: {pdf_file}")
        console.print("\n[cyan]Generating combined symbol file...[/cyan]")

        # Generate symbol file with all collected symbols
        symbol_content = generator.generate_symbol_file(
            classification_list=all_classifications, prefixes=prefix_map
        )

        symbol_file = output_dir / "symbols.sym"
        symbol_file.write_text(symbol_content)
        console.print(f"  [green]✓[/green] Symbol file: {symbol_file}")

        console.print(
            f"\n[dim]Tracked {len(generator.symbol_registry)} unique font symbols[/dim]"
        )
        console.print(f"[dim]Used {len(generator.fonts_used)} fonts[/dim]")

        console.print(f"[cyan]Includes for the main mapfile[/cyan]")
        for mapfile in reversed(mapfiles):  # TOP: drawn first...
            console.print(f'INCLUDE "layers/{mapfile}"')

    # Summary
    console.print(
        f"\n[bold green]✅ Generated {len(generated_files)} mapfile(s)[/bold green]"
    )
    console.print(f"[dim]Output directory: {output_dir}[/dim]")


@publish_commands.command()
@click.pass_context
@click.argument("style_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for QML files",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="YAML configuration file (extracts prefixes automatically)",
)
@click.option(
    "--geometry-type",
    "-t",
    type=click.Choice(["Polygon", "Line", "Point"]),
    default="Polygon",
    help="Geometry type",
)
@click.option(
    "--use-symbol-field",
    is_flag=True,
    help="Use SYMBOL field instead of complex expressions",
)
@click.option(
    "--symbol-field", default="SYMBOL", help="Name of symbol field (default: SYMBOL)"
)
@click.option(
    "--prefixes",
    "-p",
    help="Layer name to prefix mapping as JSON (overrides config file)",
)
def qgis(
    ctx,
    style_files: tuple,
    output_dir: Path,
    config_file: Optional[Path],
    geometry_type: str,
    use_symbol_field: bool,
    symbol_field: str,
    prefixes: Optional[str],
):
    """Generate QGIS QML style files from ESRI style files.

    \b
    Examples:
      # Using configuration file (recommended)
      gcover publish qgis -o output/ -c config/esri_classifier.yaml

      # Manual with JSON prefixes
      gcover publish qgis styles/*.lyrx -o output/ -t Polygon \\
        --use-symbol-field \\
        --prefixes '{"Bedrock":"bedr","Surfaces":"surf"}'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration if provided
    prefix_map = {}
    config = None

    if config_file:
        console.print(f"[cyan]Loading configuration from {config_file}[/cyan]")
        config = BatchClassificationConfig(config_file)

        # Extract prefixes from config
        for layer_config in config.layers:
            for class_config in layer_config.classifications:
                if class_config.classification_name and class_config.symbol_prefix:
                    prefix_map[class_config.classification_name] = (
                        class_config.symbol_prefix
                    )

        console.print(
            f"  [green]✓[/green] Extracted prefixes for {len(prefix_map)} classifications"
        )

        # If no style files specified, use all from config
        if not style_files:
            style_files = []
            for layer_config in config.layers:
                for class_config in layer_config.classifications:
                    if class_config.style_file not in style_files:
                        style_files.append(class_config.style_file)
            console.print(
                f"  [green]✓[/green] Found {len(style_files)} style files in config"
            )

    # Override with manual prefixes if provided
    if prefixes:
        import json

        try:
            manual_prefixes = json.loads(prefixes)
            prefix_map.update(manual_prefixes)
            console.print(f"[yellow]Applied manual prefix overrides[/yellow]")
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing prefixes JSON: {e}[/red]")
            raise click.Abort()

    if not style_files:
        console.print(
            "[red]Error: No style files specified. Use style_files argument or --config-file[/red]"
        )
        raise click.Abort()

    # Generator
    generator = QGISGenerator(
        geometry_type=geometry_type,
        use_symbol_field=use_symbol_field,
        symbol_field=symbol_field,
    )

    generated_files = []

    console.print(f"\n[bold blue]🗺️  Generating QGIS QML Files[/bold blue]\n")

    for style_file in style_files:
        console.print(f"Processing {style_file.name}...")

        # Load classifications
        classifications = extract_lyrx_complete(
            style_file, display=False
        )  # TODO switched from extract_lyrx

        if not classifications:
            console.print(
                f"[yellow]⚠ No classifications found in {style_file.name}[/yellow]"
            )
            continue

        # Process each classification
        for classification in classifications:
            layer_name = classification.layer_name or style_file.stem
            symbol_prefix = prefix_map.get(layer_name, layer_name.lower())

            # Generate QML
            qml_content = generator.generate_qml(classification, symbol_prefix)

            # Save QML file
            output_file = output_dir / f"{layer_name}.qml"
            output_file.write_text(qml_content)
            generated_files.append(output_file)

            console.print(
                f"  [green]✓[/green] Generated: {output_file.name} "
                f"({len([c for c in classification.classes if c.visible])} rules)"
            )

    # Summary
    console.print(
        f"\n[bold green]✅ Generated {len(generated_files)} QML file(s)[/bold green]"
    )
    console.print(f"[dim]Output directory: {output_dir}[/dim]")


from gcover.publish.console_generator import inspect_styles_main


@publish_commands.command(name="inspect")
@click.pass_context
@click.argument("style_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed symbol information including layers",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Load style files from YAML configuration",
)
def inspect_styles_cmd(
    ctx, style_files: tuple, detailed: bool, config_file: Optional[Path]
):
    """Inspect and display ESRI style file contents."""
    inspect_styles_main(style_files, detailed, config_file)


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


@publish_commands.command()
@click.pass_context
@click.option(
    "--rc1",
    type=click.Path(exists=True, path_type=Path),
    help="Path to RC1 (legacy complete) FileGDB",
)
@click.option(
    "--rc2",
    type=click.Path(exists=True, path_type=Path),
    help="Path to RC2 (work-in-progress) FileGDB",
)
@click.option(
    "--custom-sources-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing custom source GDBs (Saas.gdb, etc.)",
)
@click.option(
    "--admin-zones",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_ZONES_PATH,
    required=True,
    help="Path to administrative_zones.gpkg with mapsheet boundaries",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path (.gdb for FileGDB, .gpkg for GeoPackage)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["auto", "gdb", "gpkg"]),
    default="auto",
    help="Output format (auto=detect from extension, gdb=FileGDB, gpkg=GeoPackage)",
)
@click.option(
    "--source-column",
    "-s",
    type=click.Choice(["SOURCE_RC", "SOURCE_QA"]),
    default="SOURCE_RC",
    help="Column indicating source assignment (default: SOURCE_RC for publication)",
)
@click.option(
    "--mapsheets-layer",
    default="mapsheets_sources_only",
    help="Layer name in admin_zones containing mapsheet boundaries",
)
@click.option(
    "--mapsheets",
    "-m",
    help="Comma-separated mapsheet numbers to process (default: all)",
)
@click.option(
    "--reference-source",
    "-r",
    type=click.Choice(["RC1", "RC2"]),
    default="RC2",
    help="Source for reference tables (default: RC2)",
)
@click.option(
    "--layers",
    "-l",
    multiple=True,
    help="Specific spatial layers to process (default: all standard layers)",
)
@click.option(
    "--skip-tables",
    is_flag=True,
    help="Skip copying non-spatial reference tables",
)
@click.option(
    "--force-2d",
    is_flag=True,
    help="Force 2D geometries (drop Z coordinates) - useful if 3D causes issues",
)
@click.option('--clip-to-swiss-border/--no-clip-to-swiss-border', help="Clip data to Swiss border (mapsheet)", default=True)
@click.option('--validate-geometries/--no-validate-geometries', help="Validate and fix geometries", default=True)

@click.option(
    "--exclude-metadata",
    is_flag=True,
    help="Exclude metadata fields (CREATED_USER, LAST_EDITED_DATE, GlobalID, etc.)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without executing",
)
def merge(
        ctx,
        rc1: Optional[Path],
        rc2: Optional[Path],
        custom_sources_dir: Optional[Path],
        admin_zones: Path,
        output: Path,
        output_format: str,
        source_column: str,
        mapsheets_layer: str,
        mapsheets: Optional[str],
        reference_source: str,
        layers: tuple,
        skip_tables: bool,
        force_2d: bool,
        clip_to_swiss_border: bool,
        validate_geometries: bool,
        exclude_metadata: bool,
        dry_run: bool,
):
    """
    Merge multiple FileGDB sources into a single publication GDB.

    Clips features from different source databases (RC1, RC2, custom GDBs)
    based on mapsheet boundaries defined in administrative zones, then
    merges them into a single output suitable for publication.

    Source assignments are read from the admin_zones file, where each mapsheet
    has a SOURCE_RC (or SOURCE_QA) column indicating which GDB to use:

    \b
    - RC1: Legacy complete dataset (before a certain date)
    - RC2: Work-in-progress dataset (current editing)
    - Custom names (e.g., Saas.gdb): Specific overrides from custom sources

    \b
    Output formats:
    - .gdb: ESRI FileGDB (may have issues with 3D geometries)
    - .gpkg: GeoPackage (recommended for better compatibility)

    \b
    Examples:
      # Basic merge with RC1 and RC2 to GPKG (recommended)
      gcover publish merge \\
        --rc1 /path/to/RC1_2016.gdb \\
        --rc2 /path/to/RC2_2030.gdb \\
        --admin-zones /path/to/administrative_zones.gpkg \\
        --output /path/to/merged_geocover.gpkg

      # Force FileGDB output with 2D geometries
      gcover publish merge \\
        --rc1 /path/to/RC1.gdb \\
        --rc2 /path/to/RC2.gdb \\
        --admin-zones admin.gpkg \\
        --output merged.gdb \\
        --force-2d

      # Include custom source overrides
      gcover publish merge \\
        --rc1 /path/to/RC1.gdb \\
        --rc2 /path/to/RC2.gdb \\
        --custom-sources-dir /path/to/custom_gdbs/ \\
        --admin-zones admin.gpkg \\
        --output merged.gpkg

      # For QA dataset (using SOURCE_QA column)
      gcover publish merge \\
        --rc1 /path/to/RC1.gdb \\
        --rc2 /path/to/RC2.gdb \\
        --admin-zones admin.gpkg \\
        --source-column SOURCE_QA \\
        --output qa_merged.gpkg

      # Process specific mapsheets only
      gcover publish merge \\
        --rc1 /path/to/RC1.gdb \\
        --rc2 /path/to/RC2.gdb \\
        --admin-zones admin.gpkg \\
        --mapsheets "55,25,173" \\
        --output partial_merge.gpkg

      # Dry run to preview sources
      gcover publish merge \\
        --rc1 /path/to/RC1.gdb \\
        --rc2 /path/to/RC2.gdb \\
        --admin-zones admin.gpkg \\
        --output test.gpkg \\
        --dry-run
    """

    verbose = ctx.obj.get("verbose", False)

    # Validate at least one source is provided
    if not rc1 and not rc2 and not custom_sources_dir:
        console.print("[red]Error: At least one source must be specified (--rc1, --rc2, or --custom-sources-dir)[/red]")
        raise click.Abort()

    # Handle output format
    if output_format == "auto":
        # Detect from extension
        ext = output.suffix.lower()
        if ext == ".gpkg":
            output_format = "gpkg"
        elif ext == ".gdb":
            output_format = "gdb"
        else:
            console.print(f"[yellow]Unknown extension '{ext}', defaulting to GPKG[/yellow]")
            output = output.with_suffix(".gpkg")
            output_format = "gpkg"
    elif output_format == "gpkg" and not output.suffix.lower() == ".gpkg":
        output = output.with_suffix(".gpkg")
    elif output_format == "gdb" and not output.suffix.lower() == ".gdb":
        output = output.with_suffix(".gdb")

    # Parse mapsheet numbers
    mapsheet_numbers = None
    if mapsheets:
        try:
            mapsheet_numbers = [int(x.strip()) for x in mapsheets.split(",")]
        except ValueError:
            console.print(f"[red]Error: Invalid mapsheet format '{mapsheets}'. Use comma-separated numbers.[/red]")
            raise click.Abort()

    # Build configuration
    config = MergeConfig(
        rc1_path=rc1,
        rc2_path=rc2,
        custom_sources_dir=custom_sources_dir,
        admin_zones_path=admin_zones,
        mapsheets_layer=mapsheets_layer,
        source_column=source_column,
        output_path=output,
        reference_source=reference_source,
        mapsheet_numbers=mapsheet_numbers,
        preserve_z=not force_2d,  # If force_2d, don't preserve Z
        exclude_fields=DEFAULT_EXCLUDED_FIELDS if exclude_metadata else None,
        use_convex_hull_masks=True,
        clip_to_swiss_border=clip_to_swiss_border,
        validate_geometries=validate_geometries,

    )


    # Override layers if specified
    if layers:
        config.spatial_layers = list(layers)

    # Skip tables if requested
    if skip_tables:
        config.non_spatial_tables = []

    verbose = ctx.obj.get("verbose", False)

    console.print(f"\n[bold blue]🔀 GeoCover Source Merger[/bold blue]\n")

    if exclude_metadata:
        console.print(f"[dim]Excluding metadata fields: {', '.join(DEFAULT_EXCLUDED_FIELDS)}[/dim]")

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")

    # Display configuration
    _display_merge_config(config, dry_run)

    if dry_run:
        # Show mapsheet assignments preview
        _preview_merge(config)
        console.print("\n[yellow]Dry run completed. Remove --dry-run to execute merge.[/yellow]")
        return

    # Confirm before processing
    if not click.confirm("\nProceed with merge?"):
        console.print("Merge cancelled.")
        return

    # Execute merge - try arcpy first if available
    try:
        from gcover.arcpy_compat import HAS_ARCPY

        if HAS_ARCPY and output.suffix.lower() == ".gdb":
            console.print("[cyan]Using arcpy-based merger (optimal for FileGDB)[/cyan]")
            from gcover.publish.merge_sources_arcpy import GDBMergerArcPy
            merger = GDBMergerArcPy(config, verbose=verbose)
        else:
            if output.suffix.lower() == ".gdb":
                console.print("[yellow]arcpy not available, using geopandas-based merger[/yellow]")
            merger = GDBMerger(config, verbose=verbose)

        stats = merger.merge()

        if stats.errors:
            console.print(f"\n[yellow]⚠ Merge completed with {len(stats.errors)} error(s)[/yellow]")
        else:
            console.print("\n[bold green]🎉 Merge completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]Merge failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _display_merge_config(config: MergeConfig, dry_run: bool) -> None:
    """Display merge configuration summary."""

    title = "🔍 Merge Configuration (Dry Run)" if dry_run else "⚙️  Merge Configuration"

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Sources
    if config.rc1_path:
        status = "✓" if config.rc1_path.exists() else "✗"
        table.add_row("RC1 Source", f"{status} {config.rc1_path}")
    else:
        table.add_row("RC1 Source", "[dim]Not specified[/dim]")

    if config.rc2_path:
        status = "✓" if config.rc2_path.exists() else "✗"
        table.add_row("RC2 Source", f"{status} {config.rc2_path}")
    else:
        table.add_row("RC2 Source", "[dim]Not specified[/dim]")

    if config.custom_sources_dir:
        if config.custom_sources_dir.exists():
            gdb_count = len(list(config.custom_sources_dir.glob("*.gdb")))
            table.add_row("Custom Sources", f"✓ {config.custom_sources_dir} ({gdb_count} GDBs)")
        else:
            table.add_row("Custom Sources", f"✗ {config.custom_sources_dir}")

    # Admin zones
    table.add_row("Admin Zones", str(config.admin_zones_path))
    table.add_row("Mapsheets Layer", config.mapsheets_layer)
    table.add_row("Source Column", config.source_column)

    # Filtering
    if config.mapsheet_numbers:
        table.add_row("Mapsheet Filter", f"{len(config.mapsheet_numbers)} mapsheets specified")
    else:
        table.add_row("Mapsheet Filter", "All mapsheets")

    # Processing
    table.add_row("Spatial Layers", str(len(config.spatial_layers)))
    table.add_row("Reference Tables", str(len(config.non_spatial_tables)))
    table.add_row("Reference Source", config.reference_source)

    # Geometry handling
    z_mode = "Preserve Z (3D)" if config.preserve_z else "Force 2D"
    table.add_row("Z Coordinates", z_mode)

    # Output
    output_format = config.output_path.suffix.upper().replace(".", "")
    table.add_row("Output", f"{config.output_path} ({output_format})")

    console.print(table)


def _preview_merge(config: MergeConfig) -> None:
    """Preview merge operation without executing."""

    console.print("\n[cyan]Loading mapsheet assignments...[/cyan]")

    try:
        # Load mapsheets
        gdf = gpd.read_file(
            config.admin_zones_path,
            layer=config.mapsheets_layer
        )

        if config.mapsheet_numbers:
            gdf = gdf[gdf[config.mapsheet_nbr_column].isin(config.mapsheet_numbers)]

        # Group by source
        source_counts = gdf[config.source_column].value_counts()

        console.print("\n[bold]Source Assignments:[/bold]")

        table = Table(show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Mapsheets", justify="right", style="yellow")
        table.add_column("Status", style="green")

        for source_name, count in source_counts.items():
            # Check if source exists
            status = "?"
            if source_name == "RC1" and config.rc1_path:
                status = "✓ Available" if config.rc1_path.exists() else "✗ Missing"
            elif source_name == "RC2" and config.rc2_path:
                status = "✓ Available" if config.rc2_path.exists() else "✗ Missing"
            elif config.custom_sources_dir:
                custom_path = config.custom_sources_dir / f"{source_name}"
                if not custom_path.suffix:
                    custom_path = config.custom_sources_dir / f"{source_name}.gdb"
                status = "✓ Available" if custom_path.exists() else "✗ Missing"
            else:
                status = "✗ No source configured"

            table.add_row(source_name, str(count), status)

        table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(gdf)}[/bold]", "")

        console.print(table)

        # Show sample mapsheets per source
        console.print("\n[bold]Sample Mapsheets by Source:[/bold]")
        for source_name in source_counts.index[:5]:  # First 5 sources
            source_mapsheets = gdf[gdf[config.source_column] == source_name]
            sample = source_mapsheets[config.mapsheet_nbr_column].head(5).tolist()
            sample_str = ", ".join(map(str, sample))
            if len(source_mapsheets) > 5:
                sample_str += f" ... (+{len(source_mapsheets) - 5} more)"
            console.print(f"  • {source_name}: {sample_str}")

    except Exception as e:
        console.print(f"[red]Error loading preview: {e}[/red]")


@publish_commands.command()
@click.pass_context
@click.option(
    "--admin-zones",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    help="Path to administrative_zones.gpkg (uses default if not specified)",
)
@click.option(
    "--source-column",
    "-s",
    type=click.Choice(["SOURCE_RC", "SOURCE_QA"]),
    default="SOURCE_RC",
    help="Column to display (default: SOURCE_RC)",
)
@click.option(
    "--layer",
    "-l",
    default="mapsheets_sources_only",
    help="Layer name containing mapsheets",
)
def list_sources(
        ctx,
        admin_zones: Optional[Path],
        source_column: str,
        layer: str,
):
    """
    List mapsheets and their source assignments.

    Shows which source (RC1, RC2, or custom GDB) is assigned to each mapsheet
    for the merge operation.

    \b
    Examples:
      # List SOURCE_RC assignments (for publication)
      gcover publish list-sources --admin-zones admin.gpkg

      # List SOURCE_QA assignments (for QA dataset)
      gcover publish list-sources --admin-zones admin.gpkg -s SOURCE_QA
    """

    if not admin_zones:
        try:
            from importlib.resources import files
            admin_zones = files("gcover.data").joinpath("administrative_zones.gpkg")
        except:
            console.print("[red]Error: Could not find default administrative_zones.gpkg[/red]")
            console.print("Please specify path with --admin-zones")
            raise click.Abort()

    try:
        gdf = gpd.read_file(admin_zones, layer=layer)

        if source_column not in gdf.columns:
            console.print(f"[red]Error: Column '{source_column}' not found in layer '{layer}'[/red]")
            console.print(f"Available columns: {list(gdf.columns)}")
            raise click.Abort()

        console.print(f"\n[bold]Mapsheet Source Assignments ({source_column})[/bold]")
        console.print(f"[dim]Layer: {layer} in {admin_zones}[/dim]\n")

        # Summary by source
        source_counts = gdf[source_column].value_counts()

        summary_table = Table(title="Summary by Source", show_header=True)
        summary_table.add_column("Source", style="cyan")
        summary_table.add_column("Count", justify="right", style="yellow")
        summary_table.add_column("Percentage", justify="right", style="green")

        total = len(gdf)
        for source, count in source_counts.items():
            pct = count / total * 100
            summary_table.add_row(source, str(count), f"{pct:.1f}%")

        console.print(summary_table)

        # Detailed list
        console.print("\n[bold]Detailed Mapsheet List:[/bold]\n")

        detail_table = Table(show_header=True)
        detail_table.add_column("Map #", justify="right", style="cyan")
        detail_table.add_column("Title", style="green")
        detail_table.add_column("Source", style="yellow")

        for _, row in gdf.sort_values("MSH_MAP_NBR").iterrows():
            detail_table.add_row(
                str(row.get("MSH_MAP_NBR", "?")),
                str(row.get("MSH_MAP_TITLE", "?"))[:30],
                str(row[source_column])
            )

        console.print(detail_table)
        console.print(f"\n[dim]Total: {total} mapsheets[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@publish_commands.command()
@click.pass_context
@click.argument("filegdb", type=click.Path(exists=True, path_type=Path))
@click.argument("classification_db", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-a", "--attribute",
    "attributes",
    multiple=True,
    required=True,
    help="Attribute(s) to copy (can be specified multiple times)"
)
@click.option(
    "-c", "--config-file",
    type=click.Path(exists=True, path_type=Path),
    help="YAML config file with layer mapping (same format as apply-config)"
)
@click.option(
    "--uuid-field",
    default="UUID",
    show_default=True,
    help="UUID field name for joining"
)
@click.option(
    "--feature-dataset",
    default="GC_ROCK_BODIES",
    show_default=True,
    help="Feature dataset in FileGDB containing target layers"
)
@click.option(
    "--layer",
    "layers",
    multiple=True,
    help="Specific layer(s) to process (default: all from config or auto-detect)"
)
@click.option(
    "--dryrun", "-n",
    is_flag=True,
    help="Show what would be updated without making changes"
)
def writeback(
        ctx,
        filegdb: Path,
        classification_db: Path,
        attributes: tuple[str, ...],
        config_file: Optional[Path],
        uuid_field: str,
        feature_dataset: str,
        layers: tuple[str, ...],
        dryrun: bool,
):
    """
    Write back classification attributes from CLASSIFICATION_DB to FILEGDB.

    Uses layer mapping from YAML config file (same format as apply-config),
    or auto-detects matching layers by name.

    \b
    Examples:
        # Write back using config file for layer mapping
        python writeback_classification.py data.gdb classified.gpkg \\
            -a SYMBOL -a LABEL \\
            -c config/esri_classifier.yaml

        # Auto-detect layers (no config)
        python writeback_classification.py data.gdb classified.gpkg \\
            -a class_id -a class_label

        # Dryrun on specific layer
        python writeback_classification.py data.gdb classified.gpkg \\
            -a SYMBOL --layer GC_BEDROCK -n
    """
    verbose = ctx.obj.get("verbose", False)


    attributes = list(attributes)

    click.echo(f"Source FileGDB: {filegdb}")
    click.echo(f"Classification DB: {classification_db}")
    click.echo(f"Attributes: {', '.join(attributes)}")

    from gcover.publish.writeback import (build_uuid_lookup,
                                          extract_layer_name,
                                          get_matching_layers_auto,
                                          get_matching_layers_from_config,
                                          list_filegdb_layers,
                                          list_gpkg_layers,
                                          load_layer_mapping_from_config,
                                          update_filegdb_layer)

    if dryrun:
        click.secho("=== DRYRUN MODE ===", fg="yellow", bold=True)

    # Load layer mapping from config if provided
    config_mapping = {}
    if config_file:
        click.echo(f"Config file: {config_file}")
        config_mapping = load_layer_mapping_from_config(config_file)
        click.echo(f"  Loaded {len(config_mapping)} layer mappings from config")

    # List available layers
    gpkg_layers = list_gpkg_layers(classification_db)
    click.echo(f"\nGPKG layers: {len(gpkg_layers)}")
    gdb_layers = list_filegdb_layers(filegdb, feature_dataset)


    click.echo(f"FileGDB layers in {feature_dataset}: {len(gdb_layers)}")

    # Find matching layers
    if layers:
        # Use specified layers - try to find matching GDB layer for each
        matches = []
        gdb_lookup = {lyr.upper(): lyr for lyr in gdb_layers}
        for lyr in layers:
            gpkg_lyr = lyr
            # Check if it's in config mapping
            if config_mapping and lyr in config_mapping:
                gdb_lyr = extract_layer_name(config_mapping[lyr])
            else:
                gdb_lyr = extract_layer_name(lyr)

            # Find in GDB
            if gdb_lyr in gdb_layers:
                matches.append((gpkg_lyr, gdb_lyr))
            elif gdb_lyr.upper() in gdb_lookup:
                matches.append((gpkg_lyr, gdb_lookup[gdb_lyr.upper()]))
            else:
                click.secho(f"  Warning: Layer {lyr} not found in FileGDB", fg="yellow")
    elif config_mapping:
        # Use config mapping
        matches = get_matching_layers_from_config(config_mapping, gpkg_layers, gdb_layers)
    else:
        # Auto-detect
        matches = get_matching_layers_auto(gpkg_layers, gdb_layers)

    if not matches:
        click.secho("No matching layers found!", fg="red")
        raise SystemExit(1)

    click.echo(f"Matching layers: {len(matches)}")
    for gpkg_lyr, gdb_lyr in matches:
        if gpkg_lyr != gdb_lyr:
            click.echo(f"  • {gpkg_lyr} → {gdb_lyr}")
        else:
            click.echo(f"  • {gpkg_lyr}")

    # Process each matching layer
    total_stats = {"matched": 0, "updated": 0, "skipped": 0, "not_found": 0}

    for gpkg_layer, gdb_layer in matches:
        click.echo(f"\n{'─' * 50}")
        click.echo(f"Processing: {gpkg_layer} → {gdb_layer}")

        # Build lookup from classification DB
        try:
            uuid_lookup = build_uuid_lookup(
                classification_db, gpkg_layer, uuid_field, attributes
            )
        except Exception as e:
            click.secho(f"  Error reading {gpkg_layer}: {e}", fg="red")
            continue

        click.echo(f"  Classification records: {len(uuid_lookup)}")

        if not uuid_lookup:
            click.secho("  No records with UUID, skipping", fg="yellow")
            continue

        # Update FileGDB
        try:
            stats = update_filegdb_layer(
                filegdb,
                feature_dataset,
                gdb_layer,
                uuid_lookup,
                uuid_field,
                attributes,
                dryrun=dryrun
            )
        except Exception as e:
            click.secho(f"  Error updating {gdb_layer}: {e}", fg="red")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

        # Report
        click.echo(f"  Matched: {stats['matched']}")
        if stats["updated"]:
            click.secho(f"  Updated: {stats['updated']}", fg="green")
        if stats["skipped"]:
            click.echo(f"  Skipped (no change): {stats['skipped']}")
        if stats["not_found"]:
            click.secho(f"  Not in classification: {stats['not_found']}", fg="yellow")

        # Accumulate
        for k in total_stats:
            total_stats[k] += stats[k]

    # Summary
    click.echo(f"\n{'═' * 50}")
    click.secho("SUMMARY", bold=True)
    click.echo(f"Total matched: {total_stats['matched']}")
    click.echo(f"Total updated: {total_stats['updated']}")
    click.echo(f"Total skipped: {total_stats['skipped']}")
    click.echo(f"Total not found: {total_stats['not_found']}")

    if dryrun:
        click.secho("\nNo changes made (dryrun mode)", fg="yellow")