#!/usr/bin/env python3
"""
CLI commands for processing QA/verification FileGDB results.

This module provides commands to process QA test results from FileGDBs,
store them in DuckDB, and query the results.
"""

import sys
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from gcover.config import load_config, AppConfig
from gcover.gdb.manager import GDBAssetManager
from gcover.gdb.assets import AssetType
from gcover.gdb.qa_converter import FileGDBConverter

console = Console()


@click.group(name="qa")
def qa_commands():
    """Process and analyze QA test results from FileGDBs"""
    pass


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
@click.option(
    "--simplify-tolerance", type=float, help="Tolerance for geometry simplification"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def process_single(
    ctx,
    gdb_path: Path,
    output_dir: Optional[Path],
    no_upload: bool,
    format: str,
    simplify_tolerance: Optional[float],
    verbose: bool,
):
    """
    Process a single verification FileGDB.

    Converts spatial layers to web formats and generates statistics.

    Examples:
        gcover qa process /path/to/issue.gdb
        gcover qa process /path/to/issue.gdb --simplify-tolerance 1.0 --verbose
    """
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        console.print("[dim]Verbose logging enabled[/dim]")

    qa_config, global_config = get_qa_config(ctx)

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

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
    )

    try:
        console.print(f"[blue]Processing: {gdb_path.name}[/blue]")

        summary = converter.process_gdb(
            gdb_path=gdb_path,
            output_dir=output_dir,
            upload_to_s3=not no_upload,
            output_format=format,
            simplify_tolerance=simplify_tolerance,
        )

        # Display summary
        console.print(f"\n[green]✅ Processing complete![/green]")
        console.print(f"Verification Type: {summary.verification_type}")
        console.print(f"RC Version: {summary.rc_version}")
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
                console.print(f"\n[dim]💡 Tips:[/dim]")
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
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without actually processing",
)
@click.option(
    "--simplify-tolerance",
    type=float,
    help="Tolerance for geometry simplification (applies to all GDBs)",
)
@click.option("--max-workers", type=int, help="Maximum number of parallel workers")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def process_all(
    ctx,
    base_directory: Path,
    qa_type: str,
    dry_run: bool,
    simplify_tolerance: Optional[float],
    max_workers: Optional[int],
    verbose: bool,
):
    """
    Process multiple verification FileGDBs using the asset manager.

    Automatically discovers and processes QA test results from the filesystem.

    Examples:
        gcover qa process-all /media/marco/SANDISK/Verifications
        gcover qa process-all /path/to/verifications --qa-type topology --dry-run
    """
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        console.print("[dim]Verbose logging enabled[/dim]")

    qa_config, global_config = get_qa_config(ctx)

    # Override max_workers if specified
    if max_workers:
        global_config.max_workers = max_workers

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    console.print(f"[blue]Scanning directory: {base_directory}[/blue]")
    console.print(f"[dim]QA Type filter: {qa_type}[/dim]")
    if simplify_tolerance:
        console.print(f"[yellow]Geometry simplification: {simplify_tolerance}[/yellow]")

    # Set up base paths for the asset manager
    base_paths = {
        "verification": base_directory,
    }

    # Create asset manager
    manager = GDBAssetManager(
        base_paths=base_paths,
        s3_bucket=s3_bucket,
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        aws_profile=s3_profile,
    )

    try:
        # Scan filesystem using the manager
        console.print("[dim]Scanning filesystem...[/dim]")
        all_assets = manager.scan_filesystem()

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
        )

        try:
            processed = 0
            failed = 0
            skipped = 0

            for i, asset in enumerate(filtered_assets, 1):
                try:
                    console.print(
                        f"\n[blue]Processing {i}/{len(filtered_assets)}: {asset.path.name}[/blue]"
                    )

                    summary = converter.process_gdb(
                        gdb_path=asset.path, simplify_tolerance=simplify_tolerance
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
            console.print(f"\n[bold]Batch processing complete![/bold]")
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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def diagnose_gdb(gdb_path: Path, layer: Optional[str], verbose: bool):
    """
    Diagnose FileGDB structure and identify potential issues.

    Use this command to investigate geometry problems before processing.

    Examples:
        gcover qa diagnose /path/to/issue.gdb
        gcover qa diagnose /path/to/issue.gdb --layer IssuePolygons
    """
    import fiona

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

        console.print(f"\n[bold green]✅ Diagnosis complete[/bold green]")
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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def show_stats(
    ctx,
    qa_type: Optional[str],
    rc_version: Optional[str],
    days_back: int,
    export_csv: Optional[Path],
    verbose: bool,
):
    """
    Display QA statistics from the database.

    Examples:
        gcover qa stats --qa-type Topology --days-back 7
        gcover qa stats --rc-version 2030-12-31 --export-csv results.csv
    """
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

    # Get S3 settings
    s3_bucket = qa_config.get_s3_bucket(global_config)
    s3_profile = qa_config.get_s3_profile(global_config)

    converter = FileGDBConverter(
        db_path=qa_config.db_path,
        temp_dir=qa_config.temp_dir,
        s3_bucket=s3_bucket,
        s3_profile=s3_profile,
        max_workers=global_config.max_workers,
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
        console.print(f"\n[bold]Summary[/bold]")
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
        max_workers=qa_config.max_workers,
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
    import pandas as pd
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
