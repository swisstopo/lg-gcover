#!/usr/bin/env python3
"""
CLI for Geometry Cleanup
========================

Command-line interface for cleaning up geometry and topology issues
in geological vector data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import structlog

from gcover.geometry.utils import (
    GeometryCleanup,
    read_filegdb_layers,
    write_cleaned_data,
    GeometryCleanupError,
)

# Initialize console for rich output
console = Console()
logger = structlog.get_logger(__name__)


def setup_logging(verbose: bool = False):
    """Setup structured logging."""
    level = "DEBUG" if verbose else "INFO"

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def geometry(ctx, verbose):
    """Geometry cleanup tools for geological vector data."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@geometry.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output file path (default: input_cleaned.gpkg)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["gpkg", "filegdb"], case_sensitive=False),
    default="gpkg",
    help="Output format",
)
@click.option("--uuid-column", default="UUID", help="Name of UUID column")
@click.option(
    "--min-area", type=float, default=1.0, help="Minimum area for polygons (m²)"
)
@click.option(
    "--min-length", type=float, default=0.5, help="Minimum length for lines (m)"
)
@click.option(
    "--sliver-ratio",
    type=float,
    default=0.1,
    help="Sliver ratio threshold (area/perimeter²)",
)
@click.option(
    "--self-intersection-tolerance",
    type=float,
    default=0.01,
    help="Tolerance for self-intersection fixes (m)",
)
@click.option("--skip-validation", is_flag=True, help="Skip geometry validation")
@click.option("--skip-explode", is_flag=True, help="Skip multi-geometry explosion")
@click.option("--skip-small", is_flag=True, help="Skip small geometry removal")
@click.option("--skip-slivers", is_flag=True, help="Skip sliver polygon removal")
@click.option("--skip-intersections", is_flag=True, help="Skip self-intersection fixes")
@click.option("--skip-duplicates", is_flag=True, help="Skip duplicate UUID removal")
@click.option(
    "--layers", "-l", multiple=True, help="Specific layers to process (default: all)"
)
@click.option(
    "--report",
    "-r",
    type=click.Path(path_type=Path),
    help="Save cleanup report to JSON file",
)
@click.pass_context
def cleanup(
    ctx,
    input_path,
    output_path,
    output_format,
    uuid_column,
    min_area,
    min_length,
    sliver_ratio,
    self_intersection_tolerance,
    skip_validation,
    skip_explode,
    skip_small,
    skip_slivers,
    skip_intersections,
    skip_duplicates,
    layers,
    report,
):
    """Clean up geometry and topology issues in FileGDB layers."""

    verbose = ctx.obj.get("verbose", False)

    # Set default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.gpkg"

    # Configure cleanup operations
    operations = {
        "validate": not skip_validation,
        "explode_multi": not skip_explode,
        "remove_small": not skip_small,
        "remove_slivers": not skip_slivers,
        "fix_self_intersections": not skip_intersections,
        "remove_duplicate_uuids": not skip_duplicates,
    }

    console.print(
        Panel.fit(
            f"🧹 Geometry Cleanup Tool\n"
            f"Input: {input_path}\n"
            f"Output: {output_path} ({output_format.upper()})\n"
            f"UUID Column: {uuid_column}",
            title="Configuration",
        )
    )

    try:
        # Read input layers
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Reading FileGDB layers...", total=None)

            all_layers = read_filegdb_layers(input_path)

            # Filter layers if specified
            if layers:
                filtered_layers = {
                    name: gdf for name, gdf in all_layers.items() if name in layers
                }
                if not filtered_layers:
                    console.print(
                        f"[red]No matching layers found: {', '.join(layers)}[/red]"
                    )
                    console.print(f"Available layers: {', '.join(all_layers.keys())}")
                    sys.exit(1)
                all_layers = filtered_layers

            progress.update(task, description=f"Found {len(all_layers)} layers")

        # Initialize cleanup
        cleanup_tool = GeometryCleanup(
            min_area=min_area,
            min_length=min_length,
            sliver_ratio=sliver_ratio,
            self_intersection_tolerance=self_intersection_tolerance,
        )

        # Process each layer
        cleaned_layers = {}
        total_report = {}

        for layer_name, gdf in all_layers.items():
            console.print(f"\n[bold blue]Processing layer: {layer_name}[/bold blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Cleaning {layer_name}...", total=None)

                cleaned_gdf, layer_report = cleanup_tool.cleanup_geodataframe(
                    gdf, operations, uuid_column
                )

                cleaned_layers[layer_name] = cleaned_gdf
                total_report[layer_name] = layer_report

                progress.update(task, description=f"Cleaned {layer_name}")

            # Display layer summary
            _display_layer_summary(layer_name, layer_report)

        # Write output
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Writing cleaned data...", total=None)

            write_cleaned_data(cleaned_layers, output_path, output_format)

            progress.update(task, description="Output written")

        # Save report if requested
        if report:
            with open(report, "w") as f:
                json.dump(total_report, f, indent=2)
            console.print(f"[green]Report saved to: {report}[/green]")

        # Display final summary
        _display_final_summary(total_report, output_path)

    except GeometryCleanupError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@geometry.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--layers", "-l", multiple=True, help="Specific layers to validate (default: all)"
)
@click.option(
    "--report",
    "-r",
    type=click.Path(path_type=Path),
    help="Save validation report to JSON file",
)
@click.pass_context
def validate(ctx, input_path, layers, report):
    """Validate geometry in FileGDB layers without making changes."""

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            f"🔍 Geometry Validation Tool\nInput: {input_path}", title="Validation"
        )
    )

    try:
        # Read input layers
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Reading FileGDB layers...", total=None)

            all_layers = read_filegdb_layers(input_path)

            # Filter layers if specified
            if layers:
                filtered_layers = {
                    name: gdf for name, gdf in all_layers.items() if name in layers
                }
                if not filtered_layers:
                    console.print(
                        f"[red]No matching layers found: {', '.join(layers)}[/red]"
                    )
                    console.print(f"Available layers: {', '.join(all_layers.keys())}")
                    sys.exit(1)
                all_layers = filtered_layers

            progress.update(task, description=f"Found {len(all_layers)} layers")

        # Validate each layer
        from gcover.geometry.utils import GeometryValidator

        validator = GeometryValidator()

        validation_report = {}

        for layer_name, gdf in all_layers.items():
            console.print(f"\n[bold blue]Validating layer: {layer_name}[/bold blue]")

            issues = validator.validate_geometry(gdf)
            validation_report[layer_name] = {
                "total_features": len(gdf),
                "issues": issues,
            }

            # Display validation results
            _display_validation_results(layer_name, issues, len(gdf))

        # Save report if requested
        if report:
            with open(report, "w") as f:
                json.dump(validation_report, f, indent=2)
            console.print(f"[green]Validation report saved to: {report}[/green]")

        # Display overall summary
        _display_validation_summary(validation_report)

    except GeometryCleanupError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@geometry.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
def info(input_path):
    """Display information about FileGDB layers."""

    console.print(
        Panel.fit(
            f"📊 FileGDB Information\nPath: {input_path}", title="Layer Information"
        )
    )

    try:
        # Read layer information
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Reading layer information...", total=None)

            layers = read_filegdb_layers(input_path)

            progress.update(task, description=f"Found {len(layers)} layers")

        # Create summary table
        table = Table(title="Layer Summary")
        table.add_column("Layer Name", style="cyan")
        table.add_column("Feature Count", justify="right")
        table.add_column("Geometry Type", style="green")
        table.add_column("CRS", style="yellow")

        for layer_name, gdf in layers.items():
            geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else "Unknown"
            crs = str(gdf.crs) if gdf.crs else "Unknown"

            table.add_row(layer_name, str(len(gdf)), geom_type, crs)

        console.print(table)

    except GeometryCleanupError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print_exception()
        sys.exit(1)


def _display_layer_summary(layer_name: str, report: Dict):
    """Display summary for a single layer."""
    table = Table(title=f"Cleanup Summary - {layer_name}")
    table.add_column("Operation", style="cyan")
    table.add_column("Result", justify="right")

    table.add_row("Original Features", str(report["original_count"]))

    if "after_explode" in report:
        table.add_row("After Explode", str(report["after_explode"]))

    if "after_remove_small" in report:
        table.add_row("After Remove Small", str(report["after_remove_small"]))

    if "after_remove_slivers" in report:
        table.add_row("After Remove Slivers", str(report["after_remove_slivers"]))

    if "after_fix_intersections" in report:
        table.add_row("After Fix Intersections", str(report["after_fix_intersections"]))

    if "after_remove_duplicates" in report:
        table.add_row("After Remove Duplicates", str(report["after_remove_duplicates"]))

    table.add_row("Final Features", str(report["final_count"]), style="bold green")
    table.add_row("Features Removed", str(report["features_removed"]), style="bold red")

    console.print(table)


def _display_validation_results(layer_name: str, issues: Dict, total_features: int):
    """Display validation results for a layer."""
    table = Table(title=f"Validation Results - {layer_name}")
    table.add_column("Issue Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total_issues = sum(len(indices) for indices in issues.values())

    for issue_type, indices in issues.items():
        count = len(indices)
        percentage = (count / total_features) * 100 if total_features > 0 else 0

        style = "red" if count > 0 else "green"
        table.add_row(
            issue_type.replace("_", " ").title(),
            str(count),
            f"{percentage:.1f}%",
            style=style,
        )

    table.add_row(
        "Total Issues",
        str(total_issues),
        f"{(total_issues / total_features * 100):.1f}%" if total_features > 0 else "0%",
        style="bold",
    )

    console.print(table)


def _display_final_summary(report: Dict, output_path: Path):
    """Display final summary of all operations."""
    console.print(f"\n[bold green]✅ Cleanup Complete![/bold green]")
    console.print(f"Output saved to: {output_path}")

    # Calculate totals
    total_original = sum(
        layer_report["original_count"] for layer_report in report.values()
    )
    total_final = sum(layer_report["final_count"] for layer_report in report.values())
    total_removed = total_original - total_final

    console.print(f"Total features: {total_original} → {total_final}")
    console.print(f"Features removed: {total_removed}")

    if total_removed > 0:
        percentage = (total_removed / total_original) * 100
        console.print(f"Reduction: {percentage:.1f}%")


def _display_validation_summary(report: Dict):
    """Display validation summary for all layers."""
    console.print(f"\n[bold blue]📋 Validation Summary[/bold blue]")

    total_features = sum(
        layer_report["total_features"] for layer_report in report.values()
    )

    # Count total issues by type
    issue_totals = {}
    for layer_report in report.values():
        for issue_type, indices in layer_report["issues"].items():
            issue_totals[issue_type] = issue_totals.get(issue_type, 0) + len(indices)

    table = Table(title="Overall Validation Summary")
    table.add_column("Issue Type", style="cyan")
    table.add_column("Total Count", justify="right")
    table.add_column("Percentage", justify="right")

    total_issues = sum(issue_totals.values())

    for issue_type, count in issue_totals.items():
        percentage = (count / total_features) * 100 if total_features > 0 else 0
        style = "red" if count > 0 else "green"

        table.add_row(
            issue_type.replace("_", " ").title(),
            str(count),
            f"{percentage:.1f}%",
            style=style,
        )

    table.add_row(
        "Total Issues",
        str(total_issues),
        f"{(total_issues / total_features * 100):.1f}%" if total_features > 0 else "0%",
        style="bold",
    )

    console.print(table)


if __name__ == "__main__":
    geometry()
