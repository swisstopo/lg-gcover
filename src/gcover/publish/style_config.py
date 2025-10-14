#!/usr/bin/env python
#
#
"""
Extended classification applicator with YAML-based batch processing
"""

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import fiona
import geopandas as gpd
import pandas as pd
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from .esri_classification_applicator import ClassificationApplicator
from .esri_classification_extractor import extract_lyrx
from .tooltips_enricher import LayerType

from gcover.utils.console import console


@dataclass
class ClassificationConfig:
    """Configuration for a single classification application."""

    style_file: Path
    mapfile_name: Optional[str] = None
    classification_name: Optional[str] = None
    fields: Optional[Dict[str, str]] = None
    filter: Optional[str] = None
    symbol_prefix: Optional[str] = None



@dataclass
class LayerConfig:
    """Configuration for a GPKG layer with multiple classifications."""

    gpkg_layer: str
    classifications: List[ClassificationConfig]
    field_types: Optional[Dict[str, str]] = None
    layer_type: Optional[LayerType] = None


class BatchClassificationConfig:
    """Parse and manage batch classification configuration."""

    def __init__(self, config_path: Path, styles_base_path: Optional[Path] = None):
        """
        Load batch configuration from YAML.

        Args:
            config_path: Path to YAML config file
            styles_base_path: Base directory for resolving relative style paths
        """
        self.config_path = config_path
        self.styles_base_path = styles_base_path or config_path.parent

        with open(config_path, "r", encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)

        # Parse global settings
        self.global_settings = self.raw_config.get("global", {})
        self.treat_zero_as_null = self.global_settings.get("treat_zero_as_null", False)
        self.symbol_field = self.global_settings.get("symbol_field", "SYMBOL")
        self.label_field = self.global_settings.get("label_field", "LABEL")
        self.overwrite = self.global_settings.get("overwrite", False)

        # Parse layer configurations
        self.layers: List[LayerConfig] = []
        for layer_config in self.raw_config.get("layers", []):
            self.layers.append(self._parse_layer_config(layer_config))

        logger.info(f"Loaded config with {len(self.layers)} layers")

    def _parse_layer_config(self, layer_dict: dict) -> LayerConfig:
        """Parse a single layer configuration."""
        gpkg_layer = layer_dict["gpkg_layer"]
        layer_type_str = layer_dict.get("layer_type")
        layer_type = LayerType(layer_type_str) if layer_type_str else None
        classifications = []
        field_types = layer_dict.get("field_types", {})

        for class_dict in layer_dict.get("classifications", []):
            # Resolve style file path
            style_file = Path(class_dict["style_file"])
            if not style_file.is_absolute():
                style_file = self.styles_base_path / style_file

            classifications.append(
                ClassificationConfig(
                    style_file=style_file,
                    classification_name=class_dict.get("classification_name"),
                    fields=class_dict.get("fields"),
                    filter=class_dict.get("filter"),
                    symbol_prefix=class_dict.get("symbol_prefix"),
                )
            )

        return LayerConfig(
            gpkg_layer=gpkg_layer,
            classifications=classifications,
            field_types=field_types,
            layer_type=layer_type,
        )

    def get_layer_config(self, gpkg_layer: str) -> Optional[LayerConfig]:
        """Get configuration for a specific GPKG layer."""
        for layer in self.layers:
            if layer.gpkg_layer == gpkg_layer:
                return layer
        return None


def apply_batch_from_config(
    gpkg_path: Path,
    config: BatchClassificationConfig,
    layer_name: Optional[str] = None,
    output_path: Optional[Path] = None,
    debug: bool = False,
    bbox: Optional[tuple] = None,
) -> Dict[str, any]:
    """
    Apply all classifications from config to a GPKG.

    Args:
        gpkg_path: Input GPKG file
        config: Batch configuration
        layer_name: Specific layer to process (None = process all layers in config)
        output_path: Output GPKG path
        debug: Enable debug output

    Returns:
        Dictionary with processing statistics
    """

    # Determine output path
    if output_path is None:
        output_path = gpkg_path.parent / f"{gpkg_path.stem}_classified.gpkg"

    # Get available layers
    available_layers = fiona.listlayers(str(gpkg_path))

    # Determine which layers to process
    if layer_name:
        if layer_name not in available_layers:
            raise ValueError(f"Layer '{layer_name}' not found in GPKG")
        layers_to_process = [layer_name]
    else:
        # Process all layers that have config
        layers_to_process = [
            layer.gpkg_layer
            for layer in config.layers
            if layer.gpkg_layer in available_layers
        ]

    if not layers_to_process:
        logger.warning("No layers to process!")
        return {}

    logger.info(f"Processing {len(layers_to_process)} layers: {layers_to_process}")

    # Statistics
    stats = {
        "layers_processed": 0,
        "classifications_applied": 0,
        "features_classified": 0,
        "features_total": 0,
    }

    # Process each layer
    for layer in layers_to_process:
        layer_config = config.get_layer_config(layer)
        if not layer_config:
            logger.warning(f"No configuration for layer '{layer}', skipping")
            continue

        console.print(f"\n[bold blue]Processing layer: {layer}[/bold blue]")
        console.print(f"Applying {len(layer_config.classifications)} classifications")

        kwargs = {"layer": layer}
        if bbox:
            kwargs["bbox"] = bbox

        # Load layer
        gdf = gpd.read_file(gpkg_path, **kwargs)

        stats["features_total"] += len(gdf)

        # Initialize symbol/label fields
        if config.symbol_field not in gdf.columns:
            gdf[config.symbol_field] = None
        if config.label_field not in gdf.columns:
            gdf[config.label_field] = None

        # Cast fields
        field_types = layer_config.field_types

        console.print(field_types)
        console.print(gdf.columns)

        for field, dtype in field_types.items():
            if field in gdf.columns:
                try:
                    gdf[field] = (
                        pd.to_numeric(
                            gdf[field], errors="coerce"
                        )  # convert safely, invalids become NaN
                        .dropna()  # optional: drop rows with NaN
                        .astype(dtype)
                    )
                    console.print(
                        f"Casted [green]{field}[/green] to [green]{dtype}[/green]"
                    )
                except Exception as e:
                    console.print(f"[red]Failed to cast {field} to {dtype}: {e}[/red]")

        # Apply each classification
        for i, class_config in enumerate(layer_config.classifications, 1):
            console.print(
                f"\n  [{i}/{len(layer_config.classifications)}] {class_config.style_file.name}"
            )

            # Load classification from style file
            classifications = extract_lyrx(class_config.style_file, display=False)

            # Find the right classification
            classification = None
            if class_config.classification_name:
                for c in classifications:
                    if c.layer_name == class_config.classification_name:
                        classification = c
                        break
                if not classification:
                    logger.error(
                        f"Classification '{class_config.classification_name}' not found in {class_config.style_file.name}"
                    )
                    continue
            elif len(classifications) == 1:
                classification = classifications[0]
            else:
                logger.error(
                    f"Multiple classifications in {class_config.style_file.name}, specify classification_name"
                )
                continue

            # Apply classification
            try:
                applicator = ClassificationApplicator(
                    classification=classification,
                    symbol_field=config.symbol_field,
                    label_field=config.label_field,
                    symbol_prefix=class_config.symbol_prefix,
                    field_mapping=class_config.fields,
                    treat_zero_as_null=config.treat_zero_as_null,
                    debug=debug,
                )

                gdf_result = applicator.apply_v2(
                    gdf,
                    additional_filter=class_config.filter,
                    overwrite=False,  # Don't overwrite the field itself
                    preserve_existing=True,  # But preserve existing non-NULL values
                )
                # TODO After bedrock classification
                console.print(
                    f"After {class_config.classification_name}: {gdf_result[config.symbol_field].notna().sum()} symbols"
                )
                console.print(
                    f"{class_config.classification_name} symbols: {gdf_result[config.symbol_field].value_counts()}"
                )

                # Clean up string "None" values
                gdf_result[config.symbol_field] = gdf_result[
                    config.symbol_field
                ].replace(["None", ""], None)
                if config.label_field:
                    gdf_result[config.label_field] = gdf_result[
                        config.label_field
                    ].replace(["None", ""], None)

                # Update statistics
                newly_classified = (
                    gdf_result[config.symbol_field].notna()
                    & (gdf[config.symbol_field].isna())
                ).sum()
                stats["features_classified"] += newly_classified
                stats["classifications_applied"] += 1

                # Update gdf for next classification
                gdf = gdf_result

                console.print(
                    f"    [green]✓ Classified {newly_classified} features[/green]"
                )

            except Exception as e:
                logger.error(f"Failed to apply classification: {e}")
                debug = True
                if debug:
                    import traceback

                    logger.debug(traceback.format_exc())
                continue

        # Save layer to output
        console.print(f"\n  [cyan]Saving layer to {output_path}...[/cyan]")

        console.print(f"Before save: {gdf[config.symbol_field].notna().sum()} symbols")

        # Check if we need to append or create new
        if output_path.exists() and layer != layers_to_process[0]:
            # Append to existing GPKG
            gdf.to_file(output_path, layer=layer, driver="GPKG", mode="a")
        else:
            # Create new GPKG or overwrite
            gdf.to_file(output_path, layer=layer, driver="GPKG")

        gdf_check = gpd.read_file(output_path, layer=layer)
        console.print(
            f"After save: {gdf_check[config.symbol_field].notna().sum()} symbols"
        )
        console.print(f"Saved symbols: {gdf_check[config.symbol_field].value_counts()}")

        stats["layers_processed"] += 1

        # Display layer summary
        _display_layer_summary(gdf, layer, config.symbol_field, config.label_field)

    return stats


def _display_layer_summary(
    gdf: gpd.GeoDataFrame, layer_name: str, symbol_field: str, label_field: str
):
    """Display summary statistics for a processed layer."""
    total = len(gdf)
    classified = gdf[symbol_field].notna().sum()
    unclassified = total - classified

    table = Table(title=f"Layer: {layer_name}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="blue", justify="right")

    table.add_row("Total features", str(total), "100%")
    table.add_row("Classified", str(classified), f"{classified / total * 100:.1f}%")
    table.add_row(
        "Unclassified", str(unclassified), f"{unclassified / total * 100:.1f}%"
    )

    console.print(table)

    # Show top symbol types
    if classified > 0:
        symbol_counts = gdf[symbol_field].value_counts().head(10)
        console.print(
            f"\n  [dim]Top symbols: {', '.join([f'{s} ({c})' for s, c in symbol_counts.items()])}[/dim]"
        )


# =============================================================================
# CLI COMMAND
# =============================================================================


@click.group(name="classifier")
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
        )


@cli.command()
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


@cli.command()
@click.argument(
    "output_path", type=click.Path(path_type=Path), default="classification_config.yaml"
)
@click.option(
    "--example",
    type=click.Choice(["simple", "complex"]),
    default="complex",
    help="Type of example to generate",
)
def create_config(output_path: Path, example: str):
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


if __name__ == "__main__":
    cli()
