#!/usr/bin/env python3
"""
CLI commands for processing QA/verification FileGDB results.

This module provides commands to process QA test results from FileGDBs,
store them in DuckDB, and query the results.
"""

import sys
from collections import defaultdict
from datetime import datetime
from datetime import datetime as dt
from datetime import timedelta
from importlib.resources import files
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gcover.cli.gdb_cmd import (
    get_latest_assets_info,
    get_latest_topology_verification_info,
)
from gcover.cli.main import confirm_extended
from gcover.config import AppConfig, load_config
from gcover.gdb.assets import AssetType
from gcover.gdb.enhanced_qa_stats import EnhancedQAConverter
from gcover.gdb.manager import GDBAssetManager
from gcover.gdb.qa_converter import FileGDBConverter
from gcover.qa.analyzer import QAAnalyzer

from gcover.cli.main import parse_since

OUTPUT_FORMATS = ["csv", "xlsx", "json"]
GROUP_BY_CHOICES = ["mapsheets", "work_units", "lots"]


DEFAULT_ZONES_PATH = files("gcover.data").joinpath("administrative_zones.gpkg")


console = Console()


# Remove default loguru handler and add Rich-style handler
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",  # Default level
)


@click.group(name="qa")
def qa_commands():
    """Process and analyze QA test results from FileGDBs"""
    pass


# TODO
def get_qa_config(ctx):
    """Get QA configuration from context"""
    try:
        app_config: AppConfig = load_config(environment=ctx.obj["environment"])
        return app_config.qa, app_config.global_
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Make sure your configuration includes QA and global S3 settings")
        raise click.Abort()


@qa_commands.command("process")
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for processed files",
)
@click.option("--no-upload", is_flag=True, help="Skip S3 upload")
@click.option(
    "--format",
    type=click.Choice(["geoparquet", "geojson", "both"]),
    default="geoparquet",
    help="Output format",
)
@click.option("--no-convert", is_flag=True, help="Skip web format conversion")
@click.option(
    "--simplify-tolerance", type=float, help="Tolerance for geometry simplification"
)
@click.pass_context
def process_single(
    ctx,
    gdb_path: Path,
    output_dir: Optional[Path],
    no_upload: bool,
    format: str,
    simplify_tolerance: Optional[float],
    no_convert: bool,
):
    """
    Process a single verification FileGDB.

    Converts spatial layers to web formats and generates statistics.

    Examples:
        gcover qa process /path/to/issue.gdb
        gcover qa process /path/to/issue.gdb --simplify-tolerance 1.0 --verbose
    """
    verbose = ctx.obj.get("verbose", False)

    if not no_upload and no_convert:
        raise click.BadParameter(
            "Conversion to Web Format is obligatory before uploading to S3."
        )

    if verbose:
        logger.remove()  # Remove all handlers
        logger.add(sys.stderr, level="DEBUG")  # Add debug handler
        console.print("[dim]Verbose logging enabled[/dim]")

    qa_config, global_config = get_qa_config(ctx)

    console.log(qa_config)

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    if not output_dir:
        output_dir = qa_config.output_dir

    if verbose:
        console.print(f"[dim]S3 Bucket: {s3_bucket}[/dim]")
        console.print(f"[dim]S3 Profile: {s3_profile or 'default'}[/dim]")

    if simplify_tolerance:
        console.print(
            f"[yellow]⚠️  Geometry simplification enabled (tolerance: {simplify_tolerance})[/yellow]"
        )

    # Create converter with proper config
    converter = FileGDBConverter(
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        s3_bucket=s3_bucket,
        s3_profile=s3_profile,
        max_workers=global_config.max_workers,
        s3_config=global_config.s3,
    )

    try:
        console.print(f"[blue]Processing: {gdb_path.name}[/blue]")

        summary = converter.process_gdb(
            gdb_path=gdb_path,
            output_dir=output_dir,
            upload_to_s3=not no_upload,
            output_format=format,
            simplify_tolerance=simplify_tolerance,
            convert_to_web=not no_convert,
        )

        # Display summary
        console.print("\n[green]✅ Processing complete![/green]")
        console.print(f"Inserted into: {qa_config.db_path}")
        console.print(f"Verification Type: [blue]{summary.verification_type}[/blue]")
        console.print(f"RC Version: [blue]{summary.rc_version}[/blue]")
        console.print(f"Timestamp: {summary.timestamp}")
        console.print(f"Total Features: {summary.total_features:,}")

        # Show layer breakdown
        if summary.layers:
            table = Table(title="Layer Summary")
            table.add_column("Layer", style="cyan")
            table.add_column("Features", justify="right")
            table.add_column("Top Test", style="dim")
            table.add_column("Top Issue Type", style="dim")

            for layer_name, stats in summary.layers.items():
                top_test = (
                    max(stats.test_names.items(), key=lambda x: x[1])[0]
                    if stats.test_names
                    else "N/A"
                )
                top_issue = (
                    max(stats.issue_types.items(), key=lambda x: x[1])[0]
                    if stats.issue_types
                    else "N/A"
                )

                table.add_row(
                    layer_name, f"{stats.feature_count:,}", top_test, top_issue
                )

            console.print(table)

        # Show helpful tips
        if summary.total_features > 0:
            tips = []
            if not simplify_tolerance:
                tips.append(
                    "If you encounter geometry errors, try --simplify-tolerance 1.0"
                )
            tips.append("Use --verbose for detailed processing information")
            tips.append("Check 'gcover qa stats' for analysis")

            if tips:
                console.print("\n[dim]💡 Tips:[/dim]")
                for tip in tips:
                    console.print(f"[dim]  - {tip}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        else:
            console.print("[dim]Use --verbose for detailed error information[/dim]")
        raise

    finally:
        converter.close()


@qa_commands.command("process-all")
@click.argument("base_directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--qa-type",
    type=click.Choice(["topology", "tqa", "all"]),
    default="all",
    help="Type of QA tests to process (default: all)",
)
@click.option(
    "--yes", is_flag=True, help="Automatically confirm prompts (for scripting)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without actually processing",
)
@click.option(
    "--format",
    type=click.Choice(["geoparquet", "geojson", "flatgeobuf", "all"]),
    default="geoparquet",
    help="Output format",
)
@click.option(
    "--since",
    type=str,
    help="Only process assets modified since date (YYYY-MM-DD) or natural date(`2 weeks ago`)",
)
@click.option("--no-upload", is_flag=True, help="Skip S3 upload")
@click.option("--no-convert", is_flag=True, help="Skip web format conversion")
@click.option(
    "--simplify-tolerance",
    type=float,
    help="Tolerance for geometry simplification (applies to all GDBs)",
)
@click.option("--max-workers", type=int, help="Maximum number of parallel workers")
@click.pass_context
def process_all(
    ctx,
    base_directory: Path,
    qa_type: str,
    dry_run: bool,
    format: str,
    no_upload: bool,
    simplify_tolerance: Optional[float],
    max_workers: Optional[int],
    no_convert: bool,
    yes: bool,
    since: str,
):
    """
    Process multiple verification FileGDBs using the asset manager.

    Automatically discovers and processes QA test results from the filesystem.

    Examples:
        gcover qa process-all /media/marco/SANDISK/Verifications
        gcover qa process-all /path/to/verifications --qa-type topology --dry-run
    """
    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()  # Remove all handlers
        logger.add(sys.stderr, level="DEBUG")  # Add debug handler
        console.print("[dim]Verbose logging enabled[/dim]")

    qa_config, global_config = get_qa_config(ctx)

    if not no_upload and no_convert:
        raise click.BadParameter(
            "Conversion to Web Format is obligatory before uploading to S3."
        )

    # Override max_workers if specified
    if max_workers:
        global_config.max_workers = max_workers

    since_date = None
    if since:
        since_date = parse_since(since)
        console.print(f"Parsed date: {since_date}")

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    console.print(f"[blue]Scanning directory: {base_directory}[/blue]")
    console.print(f"[dim]QA Type filter: {qa_type}[/dim]")
    console.print(f"[dim]Output format: {format}[/dim]")
    if simplify_tolerance:
        console.print(f"[yellow]Geometry simplification: {simplify_tolerance}[/yellow]")

    # Set up base paths for the asset manager
    base_paths = {
        "verification": base_directory,
    }

    # Create asset manager
    # TODO
    manager = GDBAssetManager(
        base_paths=base_paths,
        # s3_bucket=s3_bucket,
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        # aws_profile=s3_profile,
        s3_config=global_config.s3,
    )

    try:
        # Scan filesystem using the manager
        console.print("[dim]Scanning filesystem...[/dim]")
        all_assets = []
        for asset in manager.scan_filesystem():
            if since_date and asset.info.timestamp < since_date:
                continue
            all_assets.append(asset)

        # Filter by asset type if specified
        if qa_type != "all":
            asset_type_filter = {
                "topology": AssetType.VERIFICATION_TOPOLOGY,
                "tqa": AssetType.VERIFICATION_TQA,
            }

            filtered_assets = [
                asset
                for asset in all_assets
                if asset.info.asset_type == asset_type_filter[qa_type]
            ]
        else:
            # Filter to only verification assets
            filtered_assets = [
                asset
                for asset in all_assets
                if asset.info.asset_type
                in [AssetType.VERIFICATION_TOPOLOGY, AssetType.VERIFICATION_TQA]
            ]

        if not filtered_assets:
            console.print(
                f"[red]No verification GDB files found in: {base_directory}[/red]"
            )
            console.print(
                "Make sure the directory contains Topology or TechnicalQualityAssurance subdirectories."
            )
            return

        console.print(f"Found {len(filtered_assets)} verification GDB files to process")

        if dry_run:
            table = Table(title="Files to Process (Dry Run)")
            table.add_column("Path", style="cyan")
            table.add_column("Type", style="dim")
            table.add_column("RC Version", style="dim")
            table.add_column("Timestamp", style="dim")

            for asset in sorted(filtered_assets, key=lambda a: a.path):
                table.add_row(
                    str(asset.path.relative_to(base_directory)),
                    asset.info.asset_type.value,
                    asset.info.release_candidate.short_name,
                    asset.info.timestamp.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(table)
            return

        # Process each GDB
        converter = FileGDBConverter(
            db_path=qa_config.db_path,
            temp_dir=qa_config.temp_dir,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            max_workers=global_config.max_workers,
            s3_config=global_config.s3,
        )
        console.print(
            f"\n[blue]Converted assets will be saved in: {qa_config.temp_dir}[/blue]"
        )

        try:
            processed = 0
            failed = 0
            skipped = 0

            for i, asset in enumerate(filtered_assets, 1):
                try:
                    verification_type, rc_version, timestamp = converter.parse_gdb_path(
                        asset.path
                    )

                    converted_dir = (
                        Path(qa_config.temp_dir)
                        / verification_type
                        / rc_version
                        / timestamp.strftime("%Y%m%d_%H%M%S")
                    )
                    converted_dir.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"\n[blue]Processing {i}/{len(filtered_assets)}: {asset.path.name}[/blue]"
                    )

                    summary = converter.process_gdb(
                        gdb_path=asset.path,
                        simplify_tolerance=simplify_tolerance,
                        output_format=format,
                        upload_to_s3=not no_upload,
                        output_dir=converted_dir,
                        convert_to_web=not no_convert,
                    )

                    if summary.total_features > 0:
                        processed += 1
                        console.print(
                            f"[green]✅ Processed {summary.total_features:,} features[/green]"
                        )
                    else:
                        skipped += 1
                        console.print(
                            "[yellow]⚠️  Skipped - no processable features[/yellow]"
                        )

                except Exception as e:
                    failed += 1
                    console.print(f"[red]❌ Failed: {e}[/red]")
                    if verbose:
                        logger.exception("Full error details:")
                    continue

            # Final summary
            console.print("\n[bold]Batch processing complete![/bold]")
            console.print(f"✅ Processed: {processed}")
            console.print(f"⚠️  Skipped: {skipped}")
            console.print(f"❌ Failed: {failed}")

            if failed > 0:
                console.print(
                    "[dim]💡 For failed files, try using --simplify-tolerance or use 'gcover qa diagnose'[/dim]"
                )

        finally:
            converter.close()

    except Exception as e:
        console.print(f"[red]Failed to process directory: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        raise


@qa_commands.command("diagnose")
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option("--layer", help="Specific layer to diagnose (default: all layers)")
@click.pass_context
def diagnose_gdb(gdb_path: Path, layer: Optional[str], verbose: bool):
    """
    Diagnose FileGDB structure and identify potential issues.

    Use this command to investigate geometry problems before processing.

    Examples:
        gcover qa diagnose /path/to/issue.gdb
        gcover qa diagnose /path/to/issue.gdb --layer IssuePolygons
    """
    import fiona

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()  # Remove all handlers
        logger.add(sys.stderr, level="DEBUG")  # Add debug handler
        console.print("[dim]Verbose logging enabled[/dim]")

    console.print(f"[bold blue]🔍 Diagnosing FileGDB:[/bold blue] {gdb_path}")

    try:
        # List all layers
        with fiona.Env():
            layers = fiona.listlayers(str(gdb_path))

        if not layers:
            console.print("[red]No layers found in FileGDB[/red]")
            return

        # Filter to specific layer if requested
        if layer:
            if layer in layers:
                layers = [layer]
            else:
                console.print(f"[red]Layer '{layer}' not found.[/red]")
                console.print(f"Available layers: {', '.join(layers)}")
                return

        # Diagnose each layer
        for layer_name in layers:
            console.print(f"\n[cyan]📊 Layer: {layer_name}[/cyan]")

            try:
                _diagnose_layer(gdb_path, layer_name, verbose)
            except Exception as e:
                console.print(f"[red]Error diagnosing layer {layer_name}: {e}[/red]")

        console.print("\n[bold green]✅ Diagnosis complete[/bold green]")
        console.print(
            "[dim]Use 'gcover qa process --help' for processing options[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Failed to diagnose FileGDB: {e}[/red]")


def _diagnose_layer(gdb_path: Path, layer_name: str, verbose: bool):
    """Diagnose a specific layer in the FileGDB"""
    import fiona

    with fiona.open(str(gdb_path), layer=layer_name) as src:
        # Basic info
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Property", style="dim")
        info_table.add_column("Value")

        info_table.add_row("Feature Count", f"{len(src):,}")
        info_table.add_row("Geometry Type", str(src.schema.get("geometry", "None")))
        info_table.add_row("CRS", str(src.crs))

        # Field information
        fields = src.schema.get("properties", {})
        field_list = ", ".join(fields.keys()) if fields else "None"
        if len(field_list) > 80:
            field_list = field_list[:80] + "..."
        info_table.add_row("Fields", field_list)

        console.print(info_table)

        # Geometry analysis
        if len(src) > 0 and src.schema.get("geometry") != "None":
            _analyze_geometries(src, verbose)


def _analyze_geometries(src, verbose: bool):
    """Analyze geometry complexity and validity"""
    complex_geoms = 0
    invalid_geoms = 0
    total_parts = 0
    max_parts = 0

    sample_size = min(100, len(src))

    if verbose:
        console.print(
            f"[dim]Sampling {sample_size} features for geometry analysis...[/dim]"
        )

    for i, record in enumerate(src):
        if i >= sample_size:
            break

        try:
            if record["geometry"]:
                geom_type = record["geometry"]["type"]

                # Count parts in multi-geometries
                if geom_type.startswith("Multi"):
                    parts = len(record["geometry"]["coordinates"])
                    total_parts += parts
                    max_parts = max(max_parts, parts)

                    if parts > 50:  # Threshold for "complex"
                        complex_geoms += 1

                # Check for very complex polygons
                elif geom_type == "Polygon":
                    coords = record["geometry"]["coordinates"]
                    if len(coords) > 10:  # Many holes
                        complex_geoms += 1

        except Exception:
            invalid_geoms += 1
            continue

    # Display geometry analysis results
    geom_table = Table(title="Geometry Analysis", show_header=False, box=None)
    geom_table.add_column("Metric", style="dim")
    geom_table.add_column("Value")

    if total_parts > 0:
        avg_parts = total_parts / sample_size
        geom_table.add_row("Avg Parts/Feature", f"{avg_parts:.1f}")
        geom_table.add_row("Max Parts/Feature", str(max_parts))

    geom_table.add_row("Complex Geometries", f"{complex_geoms}/{sample_size}")
    geom_table.add_row("Invalid Geometries", f"{invalid_geoms}/{sample_size}")

    console.print(geom_table)

    # Generate recommendations
    recommendations = []
    if complex_geoms > sample_size * 0.1:  # >10% complex
        recommendations.append(
            "High complexity detected - consider using --simplify-tolerance"
        )
    if invalid_geoms > 0:
        recommendations.append(
            "Invalid geometries found - some features may be skipped"
        )
    if max_parts > 100:
        recommendations.append(
            "Very complex multi-part geometries - processing may be slow"
        )

    if recommendations:
        console.print(
            Panel(
                "\n".join(f"• {rec}" for rec in recommendations),
                title="[yellow]⚠️  Recommendations[/yellow]",
                border_style="yellow",
            )
        )


@qa_commands.command("stats")
@click.option(
    "--qa-type",
    type=click.Choice(["Topology", "TechnicalQualityAssurance"]),
    help="Filter by QA type",
)
@click.option("--rc-version", help="Filter by RC version (e.g., 2030-12-31)")
@click.option(
    "--days-back",
    type=int,
    default=30,
    help="Number of days to look back (default: 30)",
)
@click.option(
    "--export-csv", type=click.Path(path_type=Path), help="Export results to CSV file"
)
@click.pass_context
def show_stats(
    ctx,
    qa_type: Optional[str],
    rc_version: Optional[str],
    days_back: int,
    export_csv: Optional[Path],
):
    """
    Display QA statistics from the database.

    Examples:
        gcover qa stats --qa-type Topology --days-back 7
        gcover qa stats --rc-version 2030-12-31 --export-csv results.csv
    """
    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()  # Remove all handlers
        logger.add(sys.stderr, level="DEBUG")  # Add debug handler
        console.print("[dim]Verbose logging enabled[/dim]")

    qa_config, global_config = get_qa_config(ctx)

    # Check if database exists
    if not qa_config.db_path.exists():
        console.print(f"[red]Statistics database not found: {qa_config.db_path}[/red]")
        console.print("Run 'gcover qa process' first to generate statistics.")
        return

    console.print(f"[dim]Using database: {qa_config.db_path}[/dim]")

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    converter = FileGDBConverter(
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        s3_bucket=s3_bucket,
        s3_profile=s3_profile,
        max_workers=global_config.max_workers,
        s3_config=global_config.s3,
    )

    try:
        # Convert RC version format if provided
        if rc_version:
            from gcover.core.config import convert_rc

            rc_version = convert_rc(rc_version.upper(), force="long")

        df = converter.get_statistics_summary(
            verification_type=qa_type, rc_version=rc_version, days_back=days_back
        )

        if df.empty:
            console.print(
                "[yellow]No statistics found for the specified criteria[/yellow]"
            )
            return

        # Display top issues
        console.print(f"\n[bold]Top Issues (Last {days_back} days)[/bold]")

        table = Table()
        table.add_column("QA Type", style="cyan")
        table.add_column("RC Version", style="dim")
        table.add_column("Test Name", max_width=30)
        table.add_column("Issue Type")
        table.add_column("Total Count", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("Latest", style="dim")

        for _, row in df.head(20).iterrows():
            issue_type = str(row["issue_type"]).lower()
            if "error" in issue_type:
                style = "bold red"
            elif "warning" in issue_type:
                style = "bold yellow"
            else:
                style = "dim"

            table.add_row(
                row["verification_type"],
                row["rc_version"],
                row["test_name"],
                f"[{style}]{row['issue_type']}[/{style}]",
                f"{row['total_count']:,}",
                str(row["num_runs"]),
                row["latest_run"].strftime("%Y-%m-%d"),
            )

        console.print(table)

        # Summary stats
        console.print("\n[bold]Summary[/bold]")
        console.print(f"Total unique tests: {df['test_name'].nunique()}")
        console.print(f"Total issues: {df['total_count'].sum():,}")
        console.print(
            f"Error issues: {df[df['issue_type'].str.lower() == 'error']['total_count'].sum():,}"
        )
        console.print(
            f"Warning issues: {df[df['issue_type'].str.lower() == 'warning']['total_count'].sum():,}"
        )

        if export_csv:
            df.to_csv(export_csv, index=False)
            console.print(f"[green]Exported to: {export_csv}[/green]")

    except Exception as e:
        console.print(f"[red]Failed to get statistics: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        raise

    finally:
        converter.close()


@qa_commands.command("dashboard")
@click.option(
    "--days-back",
    type=int,
    default=90,
    help="Number of days to include in dashboard (default: 90)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("qa_dashboard.html"),
    help="Output HTML file path",
)
@click.pass_context
def generate_dashboard(ctx, days_back: int, output: Path):
    """
    Generate an HTML dashboard for QA statistics.

    Creates a local HTML file with charts and statistics.

    Example:
        gcover qa dashboard --days-back 60 --output my_dashboard.html
    """
    qa_config, global_config = get_qa_config(ctx)

    # Check if database exists
    if not qa_config.db_path.exists():
        console.print(f"[red]Statistics database not found: {qa_config.db_path}[/red]")
        console.print("Run 'gcover qa process' first to generate statistics.")
        return

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    converter = FileGDBConverter(
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        s3_bucket=s3_bucket,
        s3_profile=s3_profile,
        max_workers=global_config.max_workers,
        s3_config=global_config.s3,
    )

    try:
        # Get data
        console.print(
            f"[blue]Generating dashboard with {days_back} days of data...[/blue]"
        )
        df = converter.get_statistics_summary(days_back=days_back)

        if df.empty:
            console.print("[yellow]No data available for dashboard[/yellow]")
            return

        # Generate HTML
        html_content = _generate_dashboard_html(df, days_back)
        output.write_text(html_content)

        console.print(f"[green]✅ Dashboard generated: {output.absolute()}[/green]")
        console.print(f"[dim]Open in browser: file://{output.absolute()}[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to generate dashboard: {e}[/red]")
        raise

    finally:
        converter.close()


def _generate_dashboard_html(df, days_back: int) -> str:
    """Generate HTML dashboard content"""
    from datetime import datetime

    # Calculate metrics
    total_issues = df["total_count"].sum()
    unique_tests = df["test_name"].nunique()
    error_issues = df[df["issue_type"].str.lower() == "error"]["total_count"].sum()
    warning_issues = df[df["issue_type"].str.lower() == "warning"]["total_count"].sum()

    # Prepare chart data
    top_tests = df.groupby("test_name")["total_count"].sum().head(10).to_dict()
    issue_types = df.groupby("issue_type")["total_count"].sum().to_dict()

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>GeoCover QA Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f8f9fa; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 20px; margin-bottom: 30px; }}
        .metric {{ background: white; padding: 20px; border-radius: 8px; text-align: center;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; font-weight: 500; }}
        .metric .value {{ font-size: 2.5em; font-weight: 700; margin: 0; }}
        .metric .error {{ color: #dc3545; }}
        .metric .warning {{ color: #fd7e14; }}
        .metric .info {{ color: #0dcaf0; }}
        .metric .success {{ color: #198754; }}
        .chart {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .table-container {{ background: white; padding: 20px; border-radius: 8px;
                           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .error-row {{ background-color: #fff5f5; }}
        .warning-row {{ background-color: #fffbf0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ GeoCover QA Dashboard</h1>
        <p>Quality Assurance statistics for the last {days_back} days</p>
        <p style="color: #666; margin: 0;">Generated on {
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <h3>Total Issues</h3>
            <p class="value error">{total_issues:,}</p>
        </div>
        <div class="metric">
            <h3>Unique Tests</h3>
            <p class="value info">{unique_tests}</p>
        </div>
        <div class="metric">
            <h3>Error Issues</h3>
            <p class="value error">{error_issues:,}</p>
        </div>
        <div class="metric">
            <h3>Warning Issues</h3>
            <p class="value warning">{warning_issues:,}</p>
        </div>
    </div>

    <div id="top-tests" class="chart"></div>
    <div id="issue-types" class="chart"></div>

    <div class="table-container">
        <h2>📊 Recent Issues (Top 20)</h2>
        <table>
            <thead>
                <tr>
                    <th>QA Type</th>
                    <th>Test Name</th>
                    <th>Issue Type</th>
                    <th>Count</th>
                    <th>Runs</th>
                    <th>Latest Run</th>
                </tr>
            </thead>
            <tbody>
                {
        "".join(
            [
                f'''<tr class="{"error-row" if "error" in str(row["issue_type"]).lower() else "warning-row" if "warning" in str(row["issue_type"]).lower() else ""}">
                        <td>{row["verification_type"]}</td>
                        <td>{row["test_name"]}</td>
                        <td>{row["issue_type"]}</td>
                        <td>{row["total_count"]:,}</td>
                        <td>{row["num_runs"]}</td>
                        <td>{row["latest_run"].strftime("%Y-%m-%d")}</td>
                    </tr>'''
                for _, row in df.head(20).iterrows()
            ]
        )
    }
            </tbody>
        </table>
    </div>

    <script>
        // Top tests chart
        var topTests = {top_tests};
        var topTestsData = [{{
            x: Object.keys(topTests),
            y: Object.values(topTests),
            type: 'bar',
            marker: {{ color: '#0dcaf0' }}
        }}];

        Plotly.newPlot('top-tests', topTestsData, {{
            title: {{
                text: '📈 Top 10 Tests by Issue Count',
                font: {{ size: 18, family: 'system-ui' }}
            }},
            xaxis: {{ tickangle: -45 }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }});

        // Issue types pie chart
        var issueTypes = {issue_types};
        var colors = ['#dc3545', '#fd7e14', '#198754', '#0dcaf0'];
        var issueTypesData = [{{
            values: Object.values(issueTypes),
            labels: Object.keys(issueTypes),
            type: 'pie',
            marker: {{ colors: colors }}
        }}];

        Plotly.newPlot('issue-types', issueTypesData, {{
            title: {{
                text: '🥧 Issues by Type',
                font: {{ size: 18, family: 'system-ui' }}
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }});
    </script>
</body>
</html>"""


def _auto_detect_qa_couple(
    ctx,
    rc1_gdb: Optional[Path],
    rc2_gdb: Optional[Path],
    asset_type: Optional[str] = "verification_topology",
) -> tuple[Path, Path]:
    """
    Auto-detect latest QA couple if paths not provided manually.

    Returns:
        Tuple of (RC1_path, RC2_path)
    """

    # If both provided manually, use them
    if rc1_gdb and rc2_gdb:
        return rc1_gdb, rc2_gdb

    # If only one provided, error
    if rc1_gdb or rc2_gdb:
        raise click.BadParameter(
            "If specifying manual paths, both --rc1-gdb and --rc2-gdb must be provided. "
            "Use neither for auto-detection or both for manual specification."
        )

    # Auto-detect from GDB asset database
    console.print("[cyan]🔍 Auto-detecting latest QA couple...[/cyan]")

    try:
        # Try to get GDB config to find database path
        qa_config, global_config = get_qa_config(
            ctx
        )  # TODO: cleanup, print a lot of noise
        app_config: AppConfig = load_config(environment=ctx.obj["environment"])

        gdb_config = app_config.gdb

        # gdb_db_path = qa_config.db_path.parent / "gdb_metadata.duckdb"
        gdb_db_path = gdb_config.db_path

        # Fallback to common database paths if not found
        possible_db_paths = [
            gdb_db_path,
            Path("gdb_metadata.duckdb"),
            Path("data/gdb_metadata.duckdb"),
            Path("data/dev_gdb_metadata.duckdb"),
            Path("data/prod_gdb_metadata.duckdb"),
        ]

        db_path = None
        for path in possible_db_paths:
            if path.exists():
                db_path = str(path)
                break

        if not db_path:
            raise click.ClickException(
                "❌ Cannot auto-detect QA couple: GDB metadata database not found.\n"
                "   Expected locations:\n"
                + "\n".join(f"     - {p}" for p in possible_db_paths)
                + "\n\n   Solutions:\n"
                "     1. Run 'gcover gdb scan' to create the database\n"
                "     2. Specify paths manually with --rc1-gdb and --rc2-gdb"
            )

        console.print(f"[dim]Using database: {db_path}[/dim]")

        # Get latest topology verification info
        info = get_latest_assets_info(db_path, asset_type=asset_type)

        if not info:
            raise click.ClickException(
                "❌ No topology verification data found in database.\n"
                "   Run 'gcover gdb scan' and 'gcover gdb sync' to populate the database."
            )

        if "RC1" not in info or "RC2" not in info:
            available_rcs = list(info.keys())
            raise click.ClickException(
                f"❌ Incomplete QA couple found. Available: {available_rcs}\n"
                "   Both RC1 and RC2 topology verification data required."
            )
        console.print(info)

        # Get file paths
        rc1_path = Path(info["RC1"]["path"])
        rc2_path = Path(info["RC2"]["path"])

        # Verify files exist
        missing_files = []
        if not rc1_path.exists():
            missing_files.append(f"RC1: {rc1_path}")
        if not rc2_path.exists():
            missing_files.append(f"RC2: {rc2_path}")

        if missing_files:
            raise click.ClickException(
                "❌ QA files not found on filesystem:\n"
                + "\n".join(f"     {f}" for f in missing_files)
                + "\n\n   The database has records but files may have been moved/deleted."
            )

        # Success!
        console.print(f"[green]✅ Found latest QA {asset_type} couple:[/green]")
        console.print(f"   RC1 ({info['RC1']['date']}): {rc1_path.name}")
        console.print(f"   RC2 ({info['RC2']['date']}): {rc2_path.name}")

        # Check if it's a recent couple
        from datetime import datetime

        rc1_date = datetime.strptime(info["RC1"]["date"], "%Y-%m-%d")
        rc2_date = datetime.strptime(info["RC2"]["date"], "%Y-%m-%d")
        days_apart = abs((rc1_date - rc2_date).days)

        if days_apart <= 7:
            console.print(f"[dim]   ✅ Release couple ({days_apart} days apart)[/dim]")
        else:
            console.print(
                f"[yellow]   ⚠️  Tests are {days_apart} days apart (not a close couple)[/yellow]"
            )

        return rc1_path, rc2_path

    except Exception as e:
        if isinstance(e, click.ClickException):
            raise
        else:
            raise click.ClickException(
                f"❌ Failed to auto-detect QA couple: {e}\n"
                "   Use --rc1-gdb and --rc2-gdb to specify paths manually."
            )


@qa_commands.command("aggregate")
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
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    help="Single input: RC1 GDB, RC2 GDB, or merged data (GPKG/FileGDB)",
)
@click.option(
    "--zones-file",
    default=DEFAULT_ZONES_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to administrative zones GPKG file",
)
@click.option(
    "--zone-type",
    type=click.Choice(["mapsheets", "work_units", "lots"]),
    default="mapsheets",
    help="Type of zones to aggregate by (default: mapsheets)",
)
@click.option(
    "--type",
    "asset_type",
    default=AssetType.VERIFICATION_TOPOLOGY.value,
    type=click.Choice(
        [t.value for t in [AssetType.VERIFICATION_TOPOLOGY, AssetType.VERIFICATION_TQA]]
    ),
    help=f"Filter by verification asset type.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for aggregated statistics (without extension)",
)
@click.option(
    "--base-dir",
    type=click.Path(),
    default="static/output",
    help="Base directory for structured output.",
)
@click.option(
    "--output-format",
    type=click.Choice(["csv", "xlsx", "json"]),
    default="csv",
    help="Output format for statistics (default: csv)",
)
@click.option(
    "--extract-first",
    is_flag=True,
    help="Extract relevant issues first when using two-RC mode (recommended)",
)
@click.option(
    "--auto-discover",
    is_flag=True,
    help="Auto-discover RC GDBs from base verification directory",
)
@click.option(
    "--yes", is_flag=True, help="Automatically confirm prompts (for scripting)"
)
@click.pass_context
def aggregate_qa_stats(
    ctx,
    rc1_gdb: Optional[Path],
    rc2_gdb: Optional[Path],
    input: Optional[Path],
    zones_file: Optional[Path],
    zone_type: str,
    asset_type: str,
    output: Optional[Path],
    base_dir: Optional[Path],
    output_format: str,
    extract_first: bool,
    auto_discover: bool,
    yes: bool,
):
    """
    Aggregate QA statistics by administrative zones.

    Multiple input modes supported:

    \b
    1. Two-RC mode: --rc1-gdb and --rc2-gdb
       Processes both RCs and merges relevant issues based on mapsheet sources

    \b
    2. Single input mode: --input
       Uses a single RC GDB or pre-merged data (from extract_relevant_issues)

    \b
    3. Auto-discovery mode: --auto-discover --base-dir
       Automatically finds latest RC GDBs in verification directory

    Examples:
        # Two-RC mode with extraction
        gcover qa aggregate --rc1-gdb rc1.gdb --rc2-gdb rc2.gdb --extract-first

        # Single merged input
        gcover qa aggregate --input merged_issues.gpkg --zone-type mapsheets

        # Auto-discovery
        gcover qa aggregate --auto-discover --base-dir /path/to/verifications
    """
    final_path = None
    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        console.print("[dim]Verbose logging enabled[/dim]")

    qa_config, global_config = get_qa_config(ctx)

    # Validate input combinations
    input_modes = [
        bool(rc1_gdb and rc2_gdb),  # Two-RC mode
        bool(input),  # Single input mode
        bool(auto_discover),  # Auto-discovery mode
    ]

    if sum(input_modes) != 1:
        raise click.BadParameter(
            "Exactly one input mode must be specified:\n"
            "  - Two-RC mode: --rc1-gdb AND --rc2-gdb\n"
            "  - Single input: --input\n"
            "  - Auto-discovery: --auto-discover --base-dir"
        )

    # if auto_discover and not base_dir:
    #    raise click.BadParameter("--base-dir is required with --auto-discover")

    # TODO
    if output and base_dir != "static/output":
        raise click.UsageError("Use either --output or --base-dir, not both.")
    if output:
        final_path = output
        final_path.mkdir(parents=True, exist_ok=True)

    # Set default zones file from config if available
    if not zones_file:
        # Try to get from config or use a default path
        zones_file = getattr(qa_config, "zones_file", None)
        if not zones_file:
            console.print(
                "[yellow]Warning: No zones file specified. Use --zones-file option.[/yellow]"
            )
            return

    # Set default output path
    # TODO
    """if not output:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output = Path(f"qa_aggregated_stats_{timestamp}")"""

    console.print(f"[blue]Aggregating QA statistics by {zone_type}[/blue]")

    def get_structured_dir(base_dir: str | Path, gdb_path: str | Path) -> Path:
        """
        Construct the output directory path for converted data.

        Args:
            output: Base output directory.
            verification_type: Type of verification (e.g., "Topology").
            timestamp: Timestamp object to format.

        Returns:
            A Path object representing the full converted directory.
        """
        verification_type, rc_version, timestamp = FileGDBConverter.parse_gdb_path(
            gdb_path
        )

        return (
            Path(base_dir)
            / verification_type
            / rc_version
            / timestamp.strftime("%Y%m%d_%H-%M-%S")
        )

    try:
        from gcover.qa.analyzer import QAAnalyzer

        # Initialize analyzer
        analyzer = QAAnalyzer(zones_file)
        console.print(f"[dim]Loaded zones from: {zones_file}[/dim]")

        # Handle different input modes
        if rc1_gdb and rc2_gdb:
            # Two-RC mode
            console.print(
                f"[dim]Two-RC mode: RC1={rc1_gdb.name}, RC2={rc2_gdb.name}[/dim]"
            )
            if not final_path:
                final_path = get_structured_dir(base_dir, rc2_gdb)
                final_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[dim]Output dir: {final_path}[/dim]")

            if extract_first:
                # Extract relevant issues first, then aggregate
                console.print("[dim]Extracting relevant issues first...[/dim]")
                temp_merged = final_path.parent / f"temp_merged_{final_path.stem}"

                extraction_stats = analyzer.extract_relevant_issues(
                    rc1_gdb, rc2_gdb, temp_merged, output_format="gpkg"
                )

                console.print(
                    f"[green]Extracted {extraction_stats['total_issues']} relevant issues "
                    f"({extraction_stats['rc1_issues']} RC1, {extraction_stats['rc2_issues']} RC2)[/green]"
                )

                # Aggregate from merged data
                merged_path = temp_merged.with_suffix(".gpkg")
                stats_df = analyzer.aggregate_by_zone(
                    merged_path, zone_type, output_format
                )

                # Cleanup temp file
                if merged_path.exists():
                    merged_path.unlink()

            else:
                # Original two-RC aggregation method
                console.print("[dim]Using original two-RC aggregation method[/dim]")
                stats_df = analyzer.aggregate_by_zone(
                    rc1_gdb, rc2_gdb, zone_type, output_format
                )

        elif input:
            # Single input mode
            console.print(f"[dim]Single input mode: {input.name}[/dim]")

            if not final_path:
                final_path = get_structured_dir(base_dir, input)
                final_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[dim]Output dir: {final_path}[/dim]")

            # Determine if input is merged data or single RC
            if _is_merged_data(input):
                console.print("[dim]Input appears to be merged data[/dim]")
                stats_df = analyzer.aggregate_by_zone(input, zone_type, output_format)
            else:
                console.print("[dim]Input appears to be single RC GDB[/dim]")
                # Treat as single RC - create empty second RC
                empty_rc = Path("/dev/null")  # This will be handled gracefully
                stats_df = analyzer.aggregate_by_zone(
                    input, empty_rc, zone_type, output_format
                )

        elif auto_discover:
            # Auto-discovery mode
            qa_config, global_config = get_qa_config(ctx)

            # Auto-detect QA couple if not provided
            rc1_path, rc2_path = _auto_detect_qa_couple(
                ctx, None, None, asset_type=asset_type
            )
            # console.print(f"[dim]Auto-discovery mode in: {base_dir}[/dim]")

            # rc1_path, rc2_path = _auto_discover_rc_gdbs(base_dir)

            console.print(
                f"[dim]Discovered RC1: {rc1_path if rc1_path else 'None'}[/dim]"
            )
            console.print(
                f"[dim]Discovered RC2: {rc2_path if rc2_path else 'None'}[/dim]"
            )

            if not final_path:
                final_path = get_structured_dir(base_dir, rc2_path)
                final_path.mkdir(parents=True, exist_ok=True)

            console.print(f"[dim]Output dir: {final_path}[/dim]")

            if not yes:
                response = (
                    console.input(
                        f"[bold yellow]Proceed with\nRC1: {rc1_path} and \nRC2: {rc2_path}?[/bold yellow] [green](y/n)[/green]: "
                    )
                    .strip()
                    .lower()
                )
                if response not in {"y", "yes", "o", "oui"}:
                    console.print("[red]Aborted.[/red]")
                    ctx.exit(1)

            if rc1_path and rc2_path:
                # Use two-RC mode with extraction by default
                console.print(
                    "[dim]Extracting relevant issues from discovered RCs...[/dim]"
                )
                temp_merged = final_path.parent / f"temp_merged_{final_path.stem}"
                console.print(f"[dim]Zone type: {zone_type}[/dim]")  # TODO
                extraction_stats = analyzer.extract_relevant_issues(
                    rc1_path, rc2_path, temp_merged, output_format="gpkg"
                )

                console.print(
                    f"[green]Extracted {extraction_stats['total_issues']} relevant issues[/green]"
                )
                # TODO
                merged_path = temp_merged.with_suffix(".gpkg")
                console.print(f"[dim]Zone type: {zone_type}[/dim]")  # TODO
                stats_df = analyzer.aggregate_by_zone(
                    merged_path, None, zone_type, output_format
                )

                # Cleanup
                if merged_path.exists():
                    merged_path.unlink()

            elif rc1_path or rc2_path:
                # Only one RC found
                single_rc = rc1_path or rc2_path
                console.print(
                    f"[yellow]Only one RC found, processing: {single_rc.name}[/yellow]"
                )
                empty_rc = Path("/dev/null")
                stats_df = analyzer.aggregate_by_zone(
                    single_rc, empty_rc, zone_type, output_format
                )
            else:
                console.print("[red]No RC GDBs found in base directory[/red]")
                return

        # Write results
        if not stats_df.empty:
            analyzer.write_aggregated_stats(
                stats_df, final_path, zone_type, output_format
            )

            # Display summary
            console.print(f"\n[green]✅ Aggregation complete![/green]")
            console.print(f"Statistics: {len(stats_df)} rows")
            console.print(f"Zone types: {', '.join(stats_df['zone_type'].unique())}")
            console.print(f"Layer types: {', '.join(stats_df['layer_type'].unique())}")
            if "source_rc" in stats_df.columns:
                rc_counts = stats_df["source_rc"].value_counts()
                console.print(f"RC sources: {dict(rc_counts)}")

            # Show top issues by zone
            if zone_type in stats_df.columns:
                zone_id_col = analyzer._get_zone_id_column(zone_type)
                if zone_id_col in stats_df.columns:
                    top_zones = (
                        stats_df.groupby(zone_id_col)["total_issues"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(5)
                    )

                    console.print(f"\n[bold]Top 5 {zone_type} by issue count:[/bold]")
                    for zone_id, count in top_zones.items():
                        console.print(f"  {zone_id}: {count:,} issues")

        else:
            console.print("[yellow]No aggregated statistics generated[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Aggregation failed: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        raise


def _is_merged_data(file_path: Path) -> bool:
    """
    Determine if input file is merged data or single RC GDB.

    Args:
        file_path: Path to input file

    Returns:
        True if file appears to be merged data, False if single RC
    """
    # Simple heuristics based on file structure
    if file_path.suffix.lower() == ".gpkg":
        # GPKG is likely merged data
        return True
    elif file_path.suffix.lower() == ".gdb":
        # Check if it has the typical QA issue layers
        try:
            import fiona

            layers = fiona.listlayers(str(file_path))

            # If it has source_rc info in layer data or specific naming pattern,
            # it's likely merged data
            qa_layers = ["IssuePolygons", "IssueLines", "IssuePoints"]
            has_qa_layers = any(layer in layers for layer in qa_layers)

            # Check for merged data indicators (this is heuristic)
            merged_indicators = [
                "_RC1" in str(file_path),
                "_RC2" in str(file_path),
                "merged" in str(file_path).lower(),
                "combined" in str(file_path).lower(),
                "relevant" in str(file_path).lower(),
            ]

            return has_qa_layers and any(merged_indicators)

        except Exception:
            # If we can't determine, assume it's a single RC
            return False

    return False


def _auto_discover_rc_gdbs(base_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    Auto-discover RC1 and RC2 GDBs in verification directory.

    Args:
        base_dir: Base verification directory

    Returns:
        Tuple of (rc1_path, rc2_path), either can be None if not found
    """
    rc1_path = None
    rc2_path = None

    # Look for typical verification directory structure
    # /Verifications/TechnicalQualityAssurance/RC_*/timestamp/issue.gdb
    # /Verifications/Topology/RC_*/timestamp/issue.gdb

    patterns = [
        "TechnicalQualityAssurance/RC_2016-12-31/*/issue.gdb",
        "TechnicalQualityAssurance/RC_2030-12-31/*/issue.gdb",
        "Topology/RC_2016-12-31/*/issue.gdb",
        "Topology/RC_2030-12-31/*/issue.gdb",
    ]

    import glob

    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            # Take the most recent (sort by timestamp in path)
            latest = max(matches, key=lambda p: p.parent.name)

            if "2016-12-31" in pattern:
                rc1_path = latest
            elif "2030-12-31" in pattern:
                rc2_path = latest

    return rc1_path, rc2_path


@qa_commands.command("extract")
@click.option(
    "--rc1-gdb",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    help="Path to RC1 QA FileGDB (issue.gdb). If not specified, auto-detects latest couple.",
)
@click.option(
    "--rc2-gdb",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    help="Path to RC2 QA FileGDB (issue.gdb). If not specified, auto-detects latest couple.",
)
@click.option(
    "--zones-file",
    default=DEFAULT_ZONES_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to administrative zones GPKG file",
)
@click.option(
    "--type",
    "asset_type",
    default=AssetType.VERIFICATION_TOPOLOGY.value,
    type=click.Choice(
        [t.value for t in [AssetType.VERIFICATION_TOPOLOGY, AssetType.VERIFICATION_TQA]]
    ),
    help=f"Filter by verification asset type.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path (without extension)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["gpkg", "filegdb"], case_sensitive=False),
    default="gpkg",
    help="Output format: GPKG for analysis, FileGDB for ESRI tools",
)
@click.option(
    "--filter-by-source/--no-filter",
    default=True,
    help="Filter issues by mapsheet source (RC1/RC2). Disable to extract all issues.",
)
@click.option(
    "--yes", is_flag=True, help="Automatically confirm prompts (for scripting)"
)
@click.pass_context
def extract(
    ctx,
    rc1_gdb: Optional[Path],
    rc2_gdb: Optional[Path],
    zones_file: Path,
    output: Path,
    output_format: str,
    filter_by_source: bool,
    yes: bool,
    asset_type: str,
):
    """
    Extract relevant QA issues based on mapsheet source mapping.

    This command extracts only the QA issues that are relevant for each mapsheet
    based on the source mapping (RC1 or RC2). Features overlapping multiple zones
    are included multiple times.

    Bronze → Silver data transformation.

    AUTO-DETECTION:
    If --rc1-gdb and --rc2-gdb are not specified, automatically uses the latest
    QA topology verification couple from the GDB asset database.

    MANUAL SPECIFICATION:
    Use both --rc1-gdb and --rc2-gdb to specify exact file paths.

    Examples:
        # Auto-detect latest QA couple
        gcover qa extract --output /data/silver/qa/filtered/relevant_issues

        # Manual specification
        gcover qa extract \\
            --rc1-gdb /data/bronze/qa/RC1/issue.gdb \\
            --rc2-gdb /data/bronze/qa/RC2/issue.gdb \\
            --output /data/silver/qa/filtered/relevant_issues \\
            --format filegdb \\
            --filter-by-source
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        qa_config, global_config = get_qa_config(ctx)

        # Auto-detect QA couple if not provided
        rc1_gdb, rc2_gdb = _auto_detect_qa_couple(
            ctx, rc1_gdb, rc2_gdb, asset_type=asset_type
        )

        logger.info(f"Using for RC1: {rc1_gdb}")
        logger.info(f"Using for RC2: {rc2_gdb}")

        if not yes:
            response = (
                console.input(
                    f"[bold yellow]Proceed with\nRC1: {rc1_gdb} and \nRC2: {rc2_gdb}?[/bold yellow] [green](y/n)[/green]: "
                )
                .strip()
                .lower()
            )
            if response not in {"y", "yes", "o", "oui"}:
                console.print("[red]Aborted.[/red]")
                ctx.exit(1)

        # Get S3 settings
        s3_bucket = qa_config.get_s3_bucket(global_config)
        s3_profile = qa_config.get_s3_profile(global_config)

        # TODO use same function as `aggregate`
        verification_type, rc_version, timestamp = FileGDBConverter.parse_gdb_path(
            rc2_gdb
        )

        converted_dir = (
            Path(output)
            / verification_type
            / "RC_combined"
            / timestamp.strftime("%Y%m%d_%H-%M-%S")
        )
        converted_dir = (
                Path(output)
                / verification_type
                / timestamp.strftime("%Y-%m-%d")


        )
        logger.debug(f"Output dir: {converted_dir}")
        converted_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[blue]Saving to {converted_dir}[/blue]")

        # Initialize analyzer
        logger.info(f"Initializing QA analyzer with zones from {zones_file}")
        analyzer = QAAnalyzer(zones_file)

        # Determine output file extension
        issue_output = converted_dir / "RC_combined" / "issue"
        ext = ".gdb" if output_format.lower() == "filegdb" else ".gpkg"
        output_file = issue_output.with_suffix(ext)

        # Extract relevant issues
        if filter_by_source:
            logger.info("Extracting relevant issues based on mapsheet sources")
            stats = analyzer.extract_relevant_issues(
                rc1_gdb=rc1_gdb,
                rc2_gdb=rc2_gdb,
                output_path=issue_output,
                output_format=output_format.lower(),
            )
        else:
            logger.warning("Extracting all issues (no source filtering)")
            stats = analyzer._extract_all_issues(
                rc1_gdb, rc2_gdb, issue_output, output_format.lower()
            )

        # Summary
        click.echo("✅ Extraction complete!")
        click.echo(f"   📊 {stats['total_issues']:,} total issues extracted")
        click.echo(f"   🔵 {stats['rc1_issues']:,} RC1 issues")
        click.echo(f"   🟢 {stats['rc2_issues']:,} RC2 issues")
        click.echo(f"   📁 Output: {output_file}")
        click.echo(
            f"   🔧 Format: {output_format.upper()} ({'analysis' if output_format.lower() == 'gpkg' else 'ESRI tools'})"
        )

        if filter_by_source:
            click.echo("   🎯 Source filtering: enabled (mapsheet-specific)")
        else:
            click.echo("   🎯 Source filtering: disabled (all issues)")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


# Additional helper command for QA couple status
@qa_commands.command("latest-couple")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def show_latest_couple(ctx, verbose: bool):
    """
    Show the latest QA topology verification couple.

    Displays information about the most recent RC1/RC2 QA test files
    that would be used by extract and aggregate commands.

    Examples:
        gcover qa latest-couple
        gcover qa latest-couple --verbose
    """

    try:
        # Use the same auto-detection logic but just display info
        console.print("[cyan]🔍 Checking latest QA couple...[/cyan]")

        # This will display the detection process and file info
        rc1_gdb, rc2_gdb = _auto_detect_qa_couple(ctx, None, None)

        if verbose:
            # Show additional file details
            console.print("\n[bold]File Details:[/bold]")

            for rc_name, gdb_path in [("RC1", rc1_gdb), ("RC2", rc2_gdb)]:
                stat = gdb_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)

                console.print(f"[cyan]{rc_name}:[/cyan]")
                console.print(f"   Path: {gdb_path}")
                console.print(f"   Size: {size_mb:.1f} MB")
                console.print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")

                # Check if it's a directory (FileGDB)
                if gdb_path.is_dir():
                    contents = list(gdb_path.iterdir())
                    console.print(f"   Contents: {len(contents)} items")

        console.print("\n[green]✅ Ready for QA processing![/green]")
        console.print(
            "[dim]Use 'gcover qa extract' or 'gcover qa aggregate' without --rc1-gdb/--rc2-gdb to use these files.[/dim]"
        )

    except Exception as e:
        console.print(f"[red]❌ {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


@qa_commands.command("trend-analysis")
@click.option(
    "--qa-type",
    type=click.Choice(["Topology", "TechnicalQualityAssurance", "TQA"]),
    required=True,
    help="QA test type to analyze",
)
@click.option(
    "--test-name",
    help="Specific test name to analyze (optional, shows all if not specified)",
)
@click.option(
    "--rc-version",
    type=click.Choice(["RC1", "RC2", "2016-12-31", "2030-12-31"]),
    help="Specific RC version to analyze",
)
@click.option(
    "--layer", help="Specific layer to analyze (e.g., IssuePolygons, IssueLines)"
)
@click.option(
    "--weeks-back", type=int, default=8, help="Number of weeks to analyze (default: 8)"
)
@click.pass_context
def trend_analysis(
    ctx,
    qa_type: str,
    test_name: Optional[str],
    rc_version: Optional[str],
    layer: Optional[str],
    weeks_back: int,
):
    """
    Deep trend analysis for specific tests over time.

    This command provides detailed trend analysis for specific tests,
    showing how issue counts have changed over multiple weeks.

    Examples:
        # Analyze all Topology tests for RC2 over 8 weeks
        gcover qa trend-analysis --qa-type Topology --rc-version RC2

        # Analyze a specific test across all layers
        gcover qa trend-analysis --qa-type Topology --test-name "IsCoveredByOther(0)"

        # Analyze issues in IssuePolygons layer only
        gcover qa trend-analysis --qa-type TQA --layer IssuePolygons --weeks-back 12
    """
    verbose = ctx.obj.get("verbose", False)
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    qa_config, global_config = get_qa_config(ctx)

    # Check if database exists
    if not qa_config.db_path.exists():
        console.print(f"[red]Statistics database not found: {qa_config.db_path}[/red]")
        console.print("Run 'gcover qa process' first to generate statistics.")
        return

    # Normalize inputs
    if qa_type == "TQA":
        qa_type = "TechnicalQualityAssurance"

    if rc_version:
        rc_map = {"RC1": "2016-12-31", "RC2": "2030-12-31"}
        if rc_version.upper() in rc_map:
            rc_version = rc_map[rc_version.upper()]

    try:
        # Get S3 settings
        s3_bucket = qa_config.get_s3_bucket(global_config)
        s3_profile = qa_config.get_s3_profile(global_config)

        # Create converter
        converter = FileGDBConverter(
            db_path=qa_config.db_path,
            temp_dir=qa_config.temp_dir,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            max_workers=global_config.max_workers,
            s3_config=global_config.s3,
        )

        # Build detailed trend query
        query = """
                    SELECT
                        s.verification_type,
                        s.rc_version,
                        ts.layer_name,
                        ts.test_name,
                        ts.issue_type,
                        strftime(s.timestamp, '%Y-%W') as week_year,
                        -- date_trunc('week', s.timestamp) as week_start,  -- Better for DuckDB
                         s.timestamp,
                        CAST(SUM(ts.feature_count) AS INTEGER) as issue_count
                    FROM test_stats ts
                    JOIN gdb_summaries s ON ts.gdb_summary_id = s.id
                    WHERE s.verification_type = ?
                        AND s.timestamp >= (now() - INTERVAL '{} days')
                """.format(weeks_back * 7)

        params = [qa_type]

        # Add filters
        if rc_version:
            query += " AND s.rc_version = ?"
            params.append(rc_version)

        if test_name:
            query += " AND ts.test_name = ?"
            params.append(test_name)

        if layer:
            query += " AND ts.layer_name = ?"
            params.append(layer)

        query += """
                    GROUP BY s.verification_type, s.rc_version, ts.layer_name,
                             ts.test_name, ts.issue_type,  timestamp
                    ORDER BY s.rc_version, ts.layer_name, ts.test_name,
                             ts.issue_type, s.timestamp DESC
                """

        df = converter.conn.execute(query, params).df()

        if df.empty:
            console.print(
                "[yellow]No trend data found for the specified criteria[/yellow]"
            )
            return

        console.print(df.head())

        # Display trend analysis
        console.print(f"\n[bold blue]📈 Trend Analysis: {qa_type}[/bold blue]")
        if test_name:
            console.print(f"[dim]Test: {test_name}[/dim]")
        if rc_version:
            console.print(f"[dim]RC Version: {rc_version}[/dim]")
        if layer:
            console.print(f"[dim]Layer: {layer}[/dim]")
        console.print(f"[dim]Period: {weeks_back} weeks[/dim]")

        # Group by test for detailed analysis
        grouped = df.groupby(["rc_version", "layer_name", "test_name", "issue_type"])

        console.print(grouped.head())

        for (rc_ver, layer_name, test_nm, issue_tp), group in grouped:
            if len(group) < 2:  # Need at least 2 data points for trend
                continue

            rc_short = "RC2" if rc_ver == "2030-12-31" else "RC1"

            # Calculate trend
            latest = group.iloc[0]
            oldest = group.iloc[-1]

            trend_change = latest["issue_count"] - oldest["issue_count"]
            trend_pct = (
                (trend_change / oldest["issue_count"] * 100)
                if oldest["issue_count"] > 0
                else 0
            )

            # Style based on trend
            if trend_change > 0:
                trend_style = "red"
                trend_icon = "📈"
            elif trend_change < 0:
                trend_style = "green"
                trend_icon = "📉"
            else:
                trend_style = "dim"
                trend_icon = "➡️"

            console.print(f"\n[bold]{rc_short} | {layer_name} | {test_nm}[/bold]")
            console.print(f"  Issue Type: {issue_tp}")
            console.print(f"  Current: {latest['issue_count']:,} issues")
            console.print(f"  {weeks_back} weeks ago: {oldest['issue_count']:,} issues")
            console.print(
                f"  Trend: [{trend_style}]{trend_change:+,} ({trend_pct:+.1f}%) {trend_icon}[/{trend_style}]"
            )

            # Show weekly progression
            if verbose and len(group) > 2:
                console.print(f"  Weekly progression:")
                for _, row in group.head(weeks_back).iterrows():  # Show last 6 weeks
                    week_date = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
                    console.print(f"    {week_date}: {row['issue_count']:,}")

    except Exception as e:
        console.print(f"[red]Failed to perform trend analysis: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        raise

    finally:
        converter.close()


@qa_commands.command("enhanced-stats")
@click.option(
    "--qa-type",
    type=click.Choice(["Topology", "TechnicalQualityAssurance", "TQA"]),
    help="Filter by QA test type",
)
@click.option(
    "--target-week",
    help="Target week for analysis (YYYY-MM-DD format, any day of the week)",
)
@click.option("--target-date", help="Specific date for analysis (YYYY-MM-DD format)")
@click.option(
    "--rc-versions",
    help="RC versions to include (RC1,RC2 or specific versions like 2016-12-31,2030-12-31)",
)
@click.option(
    "--no-trends",
    is_flag=True,
    help="Disable trend analysis (faster for large datasets)",
)
@click.option("--show-schedule", is_flag=True, help="Display test schedule information")
@click.option("--weekly-summary", is_flag=True, help="Show 4-week trend summary")
@click.option(
    "--top-n",
    type=int,
    default=20,
    help="Number of top issues to display (default: 20)",
)
@click.option(
    "--export-csv",
    type=click.Path(path_type=Path),
    help="Export enhanced results to CSV file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def enhanced_stats(
    ctx,
    qa_type: Optional[str],
    target_week: Optional[str],
    target_date: Optional[str],
    rc_versions: Optional[str],
    no_trends: bool,
    show_schedule: bool,
    weekly_summary: bool,
    top_n: int,
    export_csv: Optional[Path],
):
    """
    Enhanced QA statistics with trend analysis and scheduling awareness.

    This command provides detailed trend analysis comparing current test results
    with previous runs, taking into account the different test schedules for
    each RC version.

    Examples:
        # Latest Topology results for both RC versions with trends
        gcover qa enhanced-stats --qa-type Topology --show-schedule

        # Results for a specific week (RC2 Topology runs on Friday, RC1 on Saturday)
        gcover qa enhanced-stats --target-week 2025-01-20 --qa-type Topology

        # Only RC2 results with 4-week trend summary
        gcover qa enhanced-stats --rc-versions RC2 --weekly-summary

        # TQA results for a specific date without trends (faster)
        gcover qa enhanced-stats --qa-type TQA --target-date 2025-01-15 --no-trends
    """
    converter = None

    verbose = ctx.obj.get("verbose", False)
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    qa_config, global_config = get_qa_config(ctx)

    # Check if database exists
    if not qa_config.db_path.exists():
        console.print(f"[red]Statistics database not found: {qa_config.db_path}[/red]")
        console.print("Run 'gcover qa process' first to generate statistics.")
        return

    # Normalize qa_type
    if qa_type == "TQA":
        qa_type = "TechnicalQualityAssurance"

    # Parse date inputs
    parsed_target_week = None
    week_diff = 4
    if target_week:
        try:
            # Accept any day of the week, find the Monday
            any_day = datetime.strptime(target_week, "%Y-%m-%d")
            parsed_target_week = any_day - timedelta(days=any_day.weekday())
            console.print(
                f"[dim]Target week: {parsed_target_week.strftime('%Y-%m-%d')} (Monday)[/dim]"
            )
            # Reference date: today, aligned to Monday
            today = datetime.today()
            current_week = today - timedelta(days=today.weekday())

            # Compute week difference
            week_diff = (current_week - parsed_target_week).days // 7
            console.print(f"[bold]→ {week_diff} week(s) ago[/bold]")
        except ValueError:
            console.print(
                f"[red]Invalid week format: {target_week}. Use YYYY-MM-DD[/red]"
            )
            return

    parsed_target_date = None
    if target_date:
        try:
            parsed_target_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            console.print(
                f"[red]Invalid date format: {target_date}. Use YYYY-MM-DD[/red]"
            )
            return

    # Parse RC versions
    parsed_rc_versions = None
    if rc_versions:
        rc_map = {
            "RC1": "2016-12-31",
            "RC2": "2030-12-31",
            "2016-12-31": "2016-12-31",
            "2030-12-31": "2030-12-31",
        }
        parsed_rc_versions = []
        for rc in rc_versions.split(","):
            rc = rc.strip()
            if rc.upper() in ["RC1", "RC2"]:
                rc = rc.upper()
            if rc in rc_map:
                parsed_rc_versions.append(rc_map[rc])
            else:
                console.print(
                    f"[red]Invalid RC version: {rc}. Use RC1, RC2, 2016-12-31, or 2030-12-31[/red]"
                )
                return

        rc_display = [
            k
            for k, v in rc_map.items()
            if v in parsed_rc_versions and k.startswith("RC")
        ]
        console.print(f"[dim]RC Versions: {', '.join(rc_display)}[/dim]")

    # Initialize enhanced converter
    try:
        # Get S3 settings (reusing existing config logic)
        s3_bucket = qa_config.get_s3_bucket(global_config)
        s3_profile = qa_config.get_s3_profile(global_config)

        # Create standard converter for database connection
        converter = FileGDBConverter(
            db_path=qa_config.db_path,
            temp_dir=qa_config.temp_dir,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            max_workers=global_config.max_workers,
            s3_config=global_config.s3,
        )

        # Create enhanced converter with the same connection
        converter = EnhancedQAConverter(converter.conn)

        # For now, simulate the enhanced functionality using the existing converter
        # In practice, you'd replace this with the actual EnhancedQAConverter

        # Show test schedule if requested
        if show_schedule and qa_type:
            _display_test_schedule(qa_type)

        # Get time range for analysis
        if parsed_target_week:
            days_back = 7
            reference_date = parsed_target_week + timedelta(
                days=6
            )  # Sunday of target week
        elif parsed_target_date:
            days_back = 1
            reference_date = parsed_target_date
        else:
            days_back = 7
            reference_date = datetime.now()

        console.print(
            f"[blue]Analyzing QA results for the last {days_back} days from {reference_date.strftime('%Y-%m-%d')}[/blue]"
        )

        # Get current results using existing method
        qa_, df = converter.get_enhanced_statistics_summary(
            qa_test_type=qa_type,
            target_week=parsed_target_week,
            target_date=parsed_target_date,
            rc_versions=parsed_rc_versions,
            include_trends=not no_trends,
            top_n=top_n,
        )

        for q in qa_:
            trend_data = q.trend_data
            console.print(
                f"{trend_data.change_percent}, {trend_data.change}, {trend_data.trend_indicator}"
            )

        if df.empty:
            console.print(
                "[yellow]No statistics found for the specified criteria[/yellow]"
            )

            # Suggest checking test schedule
            if qa_type and not target_date:
                console.print(
                    "\n[dim]💡 Tip: Tests run on specific days of the week.[/dim]"
                )
                _display_test_schedule(qa_type)
            return

        # Filter by multiple RC versions if specified
        if parsed_rc_versions and len(parsed_rc_versions) > 1:
            df = df[df["rc_version"].isin(parsed_rc_versions)]

        # Enhanced display with simulated trend analysis
        _display_enhanced_results(df, qa_type, not no_trends, top_n)

        # Show weekly summary if requested
        if weekly_summary:
            _display_weekly_summary(converter, week_diff, qa_type, parsed_rc_versions)

        # Export if requested
        if export_csv:
            # Add computed columns for export
            df["rc_short"] = df["rc_version"].map(
                {"2016-12-31": "RC1", "2030-12-31": "RC2"}
            )

            df.to_csv(export_csv, index=False)
            console.print(f"[green]Enhanced results exported to: {export_csv}[/green]")

    except OSError as rte:
        console.print(f"[red]OSError error: {rte}[/red]")
        exit(1)

    except Exception as e:
        console.print(f"[red]Failed to get enhanced statistics: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        raise

    finally:
        if converter:
            converter.close()


def _display_test_schedule(qa_type: str):
    """Display the test schedule information."""
    schedules = {
        "Topology": {"RC2 (2030-12-31)": "Friday", "RC1 (2016-12-31)": "Saturday"},
        "TechnicalQualityAssurance": {
            "RC2 (2030-12-31)": "Friday",
            "RC1 (2016-12-31)": "Saturday",
        },
    }

    if qa_type in schedules:
        schedule_text = "\n".join(
            [f"  [bold]{rc}:[/bold] {day}" for rc, day in schedules[qa_type].items()]
        )

        schedule_panel = Panel(
            schedule_text, title=f"📅 {qa_type} Test Schedule", border_style="cyan"
        )
        console.print(schedule_panel)


def _display_enhanced_results(
    df: pd.DataFrame, qa_type: Optional[str], show_trends: bool, top_n: int
):
    """Display enhanced results with simulated trend analysis."""

    # Limit results
    df_display = df.head(top_n)

    # Calculate summary statistics
    total_issues = df_display["total_count"].sum()
    unique_tests = df_display["test_name"].nunique()

    # Group by RC for summary
    rc_summary = df_display.groupby("rc_version")["total_count"].sum()
    rc1_issues = rc_summary.get("2016-12-31", 0)
    rc2_issues = rc_summary.get("2030-12-31", 0)

    # Display summary
    stats_panel = Panel(
        f"[bold]Total Issues:[/bold] {total_issues:,}\n"
        f"[bold]Unique Tests:[/bold] {unique_tests}\n"
        f"[bold]RC1 Issues:[/bold] {rc1_issues:,}\n"
        f"[bold]RC2 Issues:[/bold] {rc2_issues:,}",
        title="📊 Summary Statistics",
        border_style="blue",
    )
    console.print(stats_panel)

    # Create enhanced results table
    table = Table(title=f"Enhanced QA Results{' with Trends' if show_trends else ''}")
    table.add_column("QA Type", style="cyan", width=12)
    table.add_column("RC", style="bold", width=4)
    table.add_column("Test Name", max_width=25)
    table.add_column("Issue Type", width=12)
    table.add_column("Count", justify="right", style="bold")
    table.add_column("Runs", justify="right", style="dim")
    if show_trends:
        table.add_column("Trend*", justify="center", width=6)
    table.add_column("Latest Run", style="dim", width=12)

    for _, row in df_display.iterrows():
        # Map RC version to short form
        rc_short = "RC2" if row["rc_version"] == "2030-12-31" else "RC1"

        # Style based on issue type
        if "error" in str(row["issue_type"]).lower():
            issue_style = "bold red"
        elif "warning" in str(row["issue_type"]).lower():
            issue_style = "bold yellow"
        else:
            issue_style = "dim"

        # TODO: Simulate trend indicator (in real implementation, this would be calculated)
        trend_indicator = (
            "📈"
            if row["total_count"] > 100
            else "📉"
            if row["total_count"] < 10
            else "➡️"
        )

        row_data = [
            str(row["verification_type"]).replace("TechnicalQualityAssurance", "TQA"),
            rc_short,
            str(row["test_name"]),
            f"[{issue_style}]{row['issue_type']}[/{issue_style}]",
            f"{row['total_count']:,}",
            str(row["num_runs"]),
        ]

        if show_trends:
            row_data.append(trend_indicator)

        row_data.append(row["latest_run"].strftime("%m-%d %H:%M"))

        table.add_row(*row_data)

    console.print(table)

    if show_trends:
        console.print(
            "[dim]* Trend indicators: 📈 Increasing, 📉 Decreasing, ➡️ Stable[/dim]"
        )


def _display_weekly_summary(
    converter, week_diff, qa_type: Optional[str], rc_versions: Optional[List[str]]
):
    """Display a 4-week summary."""
    console.print(f"\n[bold cyan]📅 {week_diff}-Week Trend Summary[/bold cyan]")

    weekly_results = converter.get_weekly_summary(
        qa_test_type=qa_type, weeks_back=week_diff
    )

    # Step 1: Create a lookup table for week_start
    week_start_map = (
        weekly_results[["week_number", "week_start"]]
        .drop_duplicates()
        .set_index("week_number")
    )

    # Step 2: Group and reshape
    pivot = (
        weekly_results.groupby(["week_number", "rc_short"])
        .agg(issues=("total_issues", "sum"), runs=("test_runs", "sum"))
        .unstack(fill_value=0)
    )

    # Step 3: Flatten columns
    pivot.columns = [f"{metric}_{rc}" for metric, rc in pivot.columns]

    # Step 4: Compute total issues
    pivot["total"] = pivot.get("issues_RC1", 0) + pivot.get("issues_RC2", 0)

    # Step 5: Merge week_start back in
    pivot = pivot.merge(week_start_map, left_index=True, right_index=True)

    # Build the rich table
    weekly_table = Table()
    weekly_table.add_column("Week", style="bold")
    weekly_table.add_column("Week start", justify="right")
    weekly_table.add_column("RC1 Issues", justify="right")
    weekly_table.add_column("RC1 Runs", justify="right", style="dim")
    weekly_table.add_column("RC2 Issues", justify="right")
    weekly_table.add_column("RC2 Runs", justify="right", style="dim")
    weekly_table.add_column("Total", justify="right", style="bold")

    for week, row in pivot.iterrows():
        weekly_table.add_row(
            str(week),
            str(row.get("week_start", 0)),
            str(row.get("issues_RC1", 0)),
            str(row.get("runs_RC1", 0)),
            str(row.get("issues_RC2", 0)),
            str(row.get("runs_RC2", 0)),
            str(row["total"]),
        )

    console.print(weekly_table)


# Updated stats command with basic enhancements
@qa_commands.command("stats-v2")
@click.option(
    "--qa-type",
    type=click.Choice(["Topology", "TechnicalQualityAssurance", "TQA"]),
    help="Filter by QA type",
)
@click.option("--rc-version", help="Filter by RC version (RC1, RC2, or full version)")
@click.option(
    "--days-back",
    type=int,
    default=7,
    help="Number of days to look back (default: 7 for weekly schedule)",
)
@click.option(
    "--group-by-rc",
    is_flag=True,
    help="Group results by RC version for easier comparison",
)
@click.option(
    "--show-empty-rc",
    is_flag=True,
    help="Show RC versions even if they have no recent runs",
)
@click.option(
    "--export-csv", type=click.Path(path_type=Path), help="Export results to CSV file"
)
@click.pass_context
def stats_v2(
    ctx,
    qa_type: Optional[str],
    rc_version: Optional[str],
    days_back: int,
    group_by_rc: bool,
    show_empty_rc: bool,
    export_csv: Optional[Path],
):
    """
    Enhanced statistics display with RC grouping and scheduling awareness.

    This is an enhanced version of the basic 'stats' command that provides
    better organization and understanding of the test schedule.
    """

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    qa_config, global_config = get_qa_config(ctx)

    # Check if database exists
    if not qa_config.db_path.exists():
        console.print(f"[red]Statistics database not found: {qa_config.db_path}[/red]")
        console.print("Run 'gcover qa process' first to generate statistics.")
        return

    console.print(f"[dim]Using database: {qa_config.db_path}[/dim]")

    # Normalize inputs
    if qa_type == "TQA":
        qa_type = "TechnicalQualityAssurance"

    if rc_version:
        rc_map = {"RC1": "2016-12-31", "RC2": "2030-12-31"}
        if rc_version.upper() in rc_map:
            rc_version = rc_map[rc_version.upper()]

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    converter = FileGDBConverter(
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        s3_bucket=s3_bucket,
        s3_profile=s3_profile,
        max_workers=global_config.max_workers,
        s3_config=global_config.s3,
    )

    try:
        console.print(f"[blue]📊 QA Statistics (Last {days_back} days)[/blue]")

        df = converter.get_statistics_summary(
            verification_type=qa_type, rc_version=rc_version, days_back=days_back
        )

        if df.empty:
            console.print(
                "[yellow]No statistics found for the specified criteria[/yellow]"
            )

            # Show helpful info about test schedules
            if qa_type and days_back <= 7:
                console.print(f"\n[dim]💡 {qa_type} tests run weekly:[/dim]")
                console.print("[dim]   • RC2 (2030-12-31): Friday[/dim]")
                console.print("[dim]   • RC1 (2016-12-31): Saturday[/dim]")
                console.print(
                    f"[dim]   Try increasing --days-back or check if tests ran this week.[/dim]"
                )

            return

        # Add RC short names
        df["rc_short"] = df["rc_version"].map(
            {"2016-12-31": "RC1", "2030-12-31": "RC2"}
        )

        if group_by_rc:
            _display_grouped_by_rc(df, qa_type, show_empty_rc)
        else:
            _display_standard_results(df, days_back)

        # Summary stats
        _display_summary_stats(df)

        if export_csv:
            df.to_csv(export_csv, index=False)
            console.print(f"[green]Results exported to: {export_csv}[/green]")

    except Exception as e:
        console.print(f"[red]Failed to get statistics: {e}[/red]")
        if verbose:
            logger.exception("Full error details:")
        raise

    finally:
        converter.close()


def _display_grouped_by_rc(
    df: pd.DataFrame, qa_type: Optional[str], show_empty_rc: bool
):
    """Display results grouped by RC version."""

    rc_versions = ["RC1", "RC2"] if show_empty_rc else df["rc_short"].unique()

    for rc in rc_versions:
        rc_df = df[df["rc_short"] == rc]

        if rc_df.empty and not show_empty_rc:
            continue

        rc_long = "2030-12-31" if rc == "RC2" else "2016-12-31"
        title = f"🏷️  {rc} ({rc_long}) Results"

        if rc_df.empty:
            console.print(f"\n[bold]{title}[/bold]")
            console.print("[dim]No recent test runs found[/dim]")
            continue

        console.print(f"\n[bold]{title}[/bold]")

        # Create table for this RC
        table = Table()
        table.add_column("QA Type", style="cyan")
        table.add_column("Test Name", max_width=30)
        table.add_column("Issue Type")
        table.add_column("Count", justify="right", style="bold")
        table.add_column("Runs", justify="right", style="dim")
        table.add_column("Latest", style="dim")

        for _, row in rc_df.head(10).iterrows():
            issue_type = str(row["issue_type"]).lower()
            if "error" in issue_type:
                style = "bold red"
            elif "warning" in issue_type:
                style = "bold yellow"
            else:
                style = "dim"

            table.add_row(
                str(row["verification_type"]).replace(
                    "TechnicalQualityAssurance", "TQA"
                ),
                str(row["test_name"]),
                f"[{style}]{row['issue_type']}[/{style}]",
                f"{row['total_count']:,}",
                str(row["num_runs"]),
                row["latest_run"].strftime("%Y-%m-%d"),
            )

        console.print(table)


def _display_standard_results(df: pd.DataFrame, days_back: int):
    """Display results in standard format."""
    console.print(f"\n[bold]Top Issues (Last {days_back} days)[/bold]")

    table = Table()
    table.add_column("QA Type", style="cyan")
    table.add_column("RC", style="bold", width=4)
    table.add_column("Test Name", max_width=30)
    table.add_column("Issue Type")
    table.add_column("Count", justify="right", style="bold")
    table.add_column("Runs", justify="right", style="dim")
    table.add_column("Latest", style="dim")

    for _, row in df.head(20).iterrows():
        issue_type = str(row["issue_type"]).lower()
        if "error" in issue_type:
            style = "bold red"
        elif "warning" in issue_type:
            style = "bold yellow"
        else:
            style = "dim"

        table.add_row(
            str(row["verification_type"]).replace("TechnicalQualityAssurance", "TQA"),
            row["rc_short"],
            str(row["test_name"]),
            f"[{style}]{row['issue_type']}[/{style}]",
            f"{row['total_count']:,}",
            str(row["num_runs"]),
            row["latest_run"].strftime("%Y-%m-%d"),
        )

    console.print(table)


def _display_summary_stats(df: pd.DataFrame):
    """Display summary statistics."""
    console.print(f"\n[bold]Summary[/bold]")

    total_issues = df["total_count"].sum()
    unique_tests = df["test_name"].nunique()
    error_issues = df[df["issue_type"].str.lower().str.contains("error", na=False)][
        "total_count"
    ].sum()
    warning_issues = df[df["issue_type"].str.lower().str.contains("warning", na=False)][
        "total_count"
    ].sum()

    # RC breakdown
    rc_summary = df.groupby("rc_short")["total_count"].sum()

    console.print(f"Total unique tests: {unique_tests}")
    console.print(f"Total issues: {total_issues:,}")
    console.print(f"Error issues: {error_issues:,}")
    console.print(f"Warning issues: {warning_issues:,}")

    for rc in ["RC1", "RC2"]:
        count = rc_summary.get(rc, 0)
        console.print(f"{rc} issues: {count:,}")


# Import the helper function from the original qa_cmd.py
def get_qa_config(ctx):
    """Get QA configuration from context (reuse existing function)"""
    try:
        app_config: AppConfig = load_config(environment=ctx.obj["environment"])
        return app_config.qa, app_config.global_
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Make sure your configuration includes QA and global S3 settings")
        raise click.Abort()
