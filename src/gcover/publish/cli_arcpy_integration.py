#!/usr/bin/env python
"""
CLI integration for ArcPy-based in-place SYMBOL field updater

Add this to your gcover/publish/cli.py or create as standalone command
"""

import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


@click.command(name="apply-config-filegdb")
@click.argument("gdb_file", type=click.Path(exists=True, path_type=Path))
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--feature-class",
    "-fc",
    multiple=True,
    help="Specific feature class(es) to process (can specify multiple times). Default: all in config",
)
@click.option(
    "--styles-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Base directory for resolving relative style paths (default: config file directory)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Parse config and validate without making changes",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def apply_config_filegdb(
        gdb_file: Path,
        config_file: Path,
        feature_class: tuple,
        styles_dir: Optional[Path],
        dry_run: bool,
        debug: bool,
):
    """
    Apply classification symbols to FileGDB feature classes using ArcPy.

    This command updates SYMBOL fields in-place using native ArcPy cursors,
    avoiding potential FileGDB corruption from geopandas/pyogrio.

    \b
    Example YAML config structure:
      global:
        symbol_field: SYMBOL
        label_field: LABEL
        treat_zero_as_null: true
        overwrite: false
      layers:
        - feature_class: GC_ROCK_BODIES/GC_POINT_OBJECTS
          classifications:
            - style_file: styles/springs.lyrx
              classification_name: Quelle
              filter: KIND == 12501001
              symbol_prefix: spring
              fields:
                KIND: KIND
                HSUR_TYPE: HSUR_TYPE

    \b
    Examples:
      # Apply all classifications from config to FileGDB
      gcover publish apply-config-filegdb data.gdb config.yaml

      # Process only specific feature class
      gcover publish apply-config-filegdb data.gdb config.yaml --feature-class GC_POINT_OBJECTS

      # Process multiple feature classes
      gcover publish apply-config-filegdb data.gdb config.yaml -fc GC_BEDROCK -fc GC_POINT_OBJECTS

      # Specify styles directory
      gcover publish apply-config-filegdb data.gdb config.yaml --styles-dir /path/to/styles

      # Dry run to validate config
      gcover publish apply-config-filegdb data.gdb config.yaml --dry-run
    """
    try:
        # Check arcpy availability
        try:
            import arcpy
        except ImportError:
            console.print("[red]Error: arcpy not available. This command requires ArcGIS Pro.[/red]")
            sys.exit(1)

        # Import the updater module
        from gcover.publish.arcpy_symbol_updater import (
            ArcPyClassificationConfig,
            ArcPySymbolUpdater,
            apply_config_to_filegdb,
        )

        console.print(f"\n[bold blue]📋 FileGDB Classification with ArcPy[/bold blue]\n")

        # Load configuration
        with console.status("[cyan]Loading configuration...", spinner="dots"):
            config = ArcPyClassificationConfig(config_file, styles_dir)

        console.print(f"[green]✓[/green] Loaded configuration:")
        console.print(f"  • Feature classes: {len(config.layers)}")
        console.print(f"  • Symbol field: {config.symbol_field}")
        console.print(f"  • Label field: {config.label_field}")
        console.print(f"  • Treat 0 as NULL: {config.treat_zero_as_null}")
        console.print(f"  • Overwrite existing: {config.overwrite}")

        # Display feature class summary
        table = Table(title="Configuration Summary", show_header=True)
        table.add_column("GPKG Layer", style="dim")
        table.add_column("FileGDB Feature Class", style="cyan")
        table.add_column("Classifications", style="yellow", justify="right")
        table.add_column("Style Files", style="dim")

        for layer_config in config.layers:
            style_files = [c.style_file.name for c in layer_config.classifications]
            table.add_row(
                layer_config.gpkg_layer or "-",
                layer_config.feature_class or layer_config.gpkg_layer or "-",
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

            # Validate that FileGDB exists
            if not arcpy.Exists(str(gdb_file)):
                console.print(f"\n  [red]✗ FileGDB not found: {gdb_file}[/red]")
                all_valid = False
            else:
                console.print(f"\n  [green]✓ FileGDB exists: {gdb_file}[/green]")

            # Validate feature classes exist
            arcpy.env.workspace = str(gdb_file)
            console.print("\nValidating feature classes in FileGDB...")
            for layer_config in config.layers:
                fc_name = layer_config.get_target_name("filegdb")
                if not fc_name:
                    console.print(
                        f"  [yellow]⚠ Layer has no feature_class defined: {layer_config.gpkg_layer}[/yellow]"
                    )
                    continue

                fc_path = str(gdb_file / fc_name)
                if arcpy.Exists(fc_path):
                    count = arcpy.management.GetCount(fc_path)[0]
                    console.print(
                        f"  [green]✓ {fc_name} ({count} features)[/green]"
                    )
                else:
                    console.print(
                        f"  [red]✗ Feature class not found: {fc_name}[/red]"
                    )
                    all_valid = False

            if all_valid:
                console.print("\n[green]✓ Configuration is valid![/green]")
            else:
                console.print("\n[red]✗ Configuration has errors[/red]")
            return

        # Apply classifications
        feature_classes_list = list(feature_class) if feature_class else None

        stats = apply_config_to_filegdb(
            gdb_path=gdb_file,
            config_path=config_file,
            feature_classes=feature_classes_list,
            styles_dir=styles_dir,
        )

        # Display final statistics
        console.print("\n[bold green]✅ Processing complete![/bold green]\n")

        summary_table = Table(title="Processing Statistics", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row(
            "Feature classes processed", str(stats["feature_classes_processed"])
        )
        summary_table.add_row(
            "Classifications applied", str(stats["classifications_applied"])
        )
        summary_table.add_row("Features updated", str(stats["features_updated"]))
        summary_table.add_row("Features skipped", str(stats["features_skipped"]))

        if stats["errors"] > 0:
            summary_table.add_row("Errors", str(stats["errors"]), style="red")

        console.print(summary_table)
        console.print(f"\n[dim]FileGDB: {gdb_file}[/dim]")

        if stats["errors"] > 0:
            console.print(
                "\n[yellow]⚠ Some errors occurred. Check logs for details.[/yellow]"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if debug:
            import traceback
            logger.debug(traceback.format_exc())
        raise


# Alternative: Add this to your existing CLI group
def register_arcpy_commands(cli_group):
    """
    Register ArcPy commands to an existing Click group.

    Usage in your gcover/publish/cli.py:
        from .cli_arcpy_integration import register_arcpy_commands

        @click.group()
        def publish():
            '''Publishing commands'''
            pass

        register_arcpy_commands(publish)
    """
    cli_group.add_command(apply_config_filegdb)


if __name__ == "__main__":
    apply_config_filegdb()