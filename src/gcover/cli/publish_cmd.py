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
import time
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gcover.config import SDE_INSTANCES, AppConfig, load_config
from gcover.publish.esri_classification_applicator import ClassificationApplicator
from gcover.publish.esri_classification_extractor import extract_lyrx
from gcover.publish.generator import MapServerGenerator
from gcover.publish.qgis_generator import QGISGenerator
from gcover.publish.style_config import (
    BatchClassificationConfig,
    apply_batch_from_config,
)
from gcover.publish.tooltips_enricher import (
    EnhancedTooltipsEnricher,
    EnrichmentConfig,
    LayerMapping,
    LayerType,
    create_enrichment_config,
)

from .main import _split_bbox

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
    bbox: Optional[tuple],
    continue_on_error: bool,
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

        start_time = time.time()

        # Apply classifications
        stats = apply_batch_from_config(
            gpkg_path=gpkg_file,
            config=config,
            layer_name=layer,
            output_path=output,
            debug=debug,
            bbox=bbox,
            continue_on_error=continue_on_error,
        )

        end_time = time.time()
        elapsed = end_time - start_time
        mins, secs = divmod(elapsed, 60)

        # Display final statistics
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
        summary_table.add_row("Features classified", str(stats["features_classified"]))
        summary_table.add_row("Total features", str(stats["features_total"]))

        if stats["features_total"] > 0:
            pct = stats["features_classified"] / stats["features_total"] * 100
            summary_table.add_row("Coverage", f"{pct:.1f}%")
        summary_table.add_row("Processing time", f"{int(mins)}m {secs:.1f}s")

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
@click.argument(
    "style_files", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for mapfiles",
)
@click.option(
    "--layer-type",
    "-t",
    type=click.Choice(["Polygon", "Line", "Point"]),
    default="Polygon",
    help="Geometry type",
)
@click.option(
    "--data-path", "-d", help="Data source path template (use {layer} placeholder)"
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
    help='Layer name to prefix mapping as JSON (e.g., \'{"Bedrock":"bedr","Surfaces":"surf"}\')',
)
@click.option(
    "--generate-combined",
    is_flag=True,
    help="Generate combined symbol file and fontset for all layers",
)
def mapserver(
    ctx,
    style_files: tuple,
    output_dir: Path,
    layer_type: str,
    data_path: Optional[str],
    use_symbol_field: bool,
    symbol_field: str,
    prefixes: Optional[str],
    generate_combined: bool,
):
    """Generate MapServer mapfiles from ESRI style files.

    Processes multiple style files and generates individual mapfiles plus
    optional combined symbol file and fontset with proper symbol tracking.

    \b
    Examples:
      # Generate single layer
      gcover publish mapserver styles/Bedrock.lyrx -o output/ -t Polygon \\
        --data-path "geocover.gpkg,layer=GC_BEDROCK"

      # Generate multiple layers with combined symbols
      gcover publish mapserver styles/*.lyrx -o output/ -t Polygon \\
        --use-symbol-field --generate-combined \\
        --prefixes '{"Bedrock":"bedr","Surfaces":"surf","Fossils":"foss"}'

      # With data path template
      gcover publish mapserver styles/*.lyrx -o output/ \\
        --data-path "geocover.gpkg,layer={layer}" \\
        --generate-combined
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse prefix mappings
    prefix_map = {}
    if prefixes:
        import json

        try:
            prefix_map = json.loads(prefixes)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing prefixes JSON: {e}[/red]")
            raise click.Abort()

    # Single generator for all layers (to track symbols across layers)
    generator = MapServerGenerator(
        layer_type=layer_type,
        use_symbol_field=use_symbol_field,
        symbol_field=symbol_field,
    )

    # Store all classifications for combined symbol file
    all_classifications = []
    generated_files = []

    console.print(f"\n[bold blue]🗺️  Generating MapServer Mapfiles[/bold blue]\n")

    for style_file in style_files:
        console.print(f"Processing {style_file.name}...")

        # Load classifications
        classifications = extract_lyrx(style_file, display=False)

        if not classifications:
            console.print(
                f"[yellow]⚠ No classifications found in {style_file.name}[/yellow]"
            )
            continue

        # Process each classification
        for classification in classifications:
            layer_name = classification.layer_name or style_file.stem
            symbol_prefix = prefix_map.get(layer_name, layer_name.lower())

            # Determine data path
            layer_data_path = None
            if data_path:
                layer_data_path = data_path.replace("{layer}", layer_name.upper())

            # Generate mapfile
            mapfile_content = generator.generate_layer(
                classification=classification,
                layer_name=layer_name,
                data_path=layer_data_path,
                symbol_prefix=symbol_prefix,
            )

            # Save mapfile
            output_file = output_dir / f"{layer_name}.map"
            output_file.write_text(mapfile_content)
            generated_files.append(output_file)

            console.print(
                f"  [green]✓[/green] Generated: {output_file.name} "
                f"({len([c for c in classification.classes if c.visible])} classes)"
            )

            # Store classification for combined symbol file
            all_classifications.append(classification)

    # Generate combined symbol file if requested
    if generate_combined and all_classifications:
        console.print("\n[cyan]Generating combined symbol file...[/cyan]")

        # Generate symbol file with all collected symbols
        symbol_content = generator.generate_symbol_file(
            classification_list=all_classifications, prefixes=prefix_map
        )

        symbol_file = output_dir / "symbols.sym"
        symbol_file.write_text(symbol_content)
        console.print(f"  [green]✓[/green] Symbol file: {symbol_file}")

        # Generate fontset
        fontset_content = generator.generate_fontset()
        fontset_file = output_dir / "fonts.txt"
        fontset_file.write_text(fontset_content)
        console.print(f"  [green]✓[/green] Fontset: {fontset_file}")

        # Check for PDF
        pdf_file = output_dir / "font_characters.pdf"
        if pdf_file.exists():
            console.print(f"  [green]✓[/green] Font symbols PDF: {pdf_file}")

        console.print(
            f"\n[dim]Tracked {len(generator.symbol_registry)} unique font symbols[/dim]"
        )
        console.print(f"[dim]Used {len(generator.fonts_used)} fonts[/dim]")

    # Summary
    console.print(
        f"\n[bold green]✅ Generated {len(generated_files)} mapfile(s)[/bold green]"
    )
    console.print(f"[dim]Output directory: {output_dir}[/dim]")


@publish_commands.command()
@click.pass_context
@click.argument(
    "style_files", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for QML files",
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
    help="Layer name to prefix mapping as JSON",
)
def qgis(
    ctx,
    style_files: tuple,
    output_dir: Path,
    geometry_type: str,
    use_symbol_field: bool,
    symbol_field: str,
    prefixes: Optional[str],
):
    """Generate QGIS QML style files from ESRI style files.

    \b
    Examples:
      # Generate single layer
      gcover publish qgis styles/Bedrock.lyrx -o output/ -t Polygon

      # Generate multiple layers with symbol field
      gcover publish qgis styles/*.lyrx -o output/ -t Polygon \\
        --use-symbol-field \\
        --prefixes '{"Bedrock":"bedr","Surfaces":"surf"}'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse prefix mappings
    prefix_map = {}
    if prefixes:
        import json

        try:
            prefix_map = json.loads(prefixes)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing prefixes JSON: {e}[/red]")
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
        classifications = extract_lyrx(style_file, display=False)

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


# ... (rest of the enrichment commands remain the same)
# I'm keeping only the modified commands for brevity - the enrichment commands
# don't need changes since they don't use the generator classes
