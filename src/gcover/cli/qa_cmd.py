#!/usr/bin/env python3
"""
CLI commands for processing verification FileGDB results.

Extends the existing lg-gcover CLI with verification processing capabilities.
Renamed file handling for qa_converter.py
"""

import sys
import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
import pandas as pd

from gcover.config import load_config, AppConfig  # TODO
from gcover.config.models import GDBConfig, GlobalConfig, QAConfig

from loguru import logger

console = Console()


# Create a temporary config object that includes S3 settings for the converter
# (This assumes your FileGDBConverter expects s3_bucket and s3_profile attributes)
class ConfigWrapper:
    def __init__(self, qa_config, global_config):
        # Copy QA config attributes
        for attr in ["db_path", "temp_dir", "max_workers"]:
            if hasattr(qa_config, attr):
                setattr(self, attr, getattr(qa_config, attr))

        # Add S3 settings from global config
        self.s3_bucket = global_config.s3.bucket
        self.s3_profile = global_config.s3.profile


def get_qa_config(ctx):
    """Helper to get QA configuration with global S3 settings"""
    config_manager = ctx.obj["config_manager"]

    # Try to get QA-specific config, fall back to GDB config
    try:
        qa_config = config_manager.get_config("qa")
    except ValueError:
        # QA config not available, use GDB config as fallback
        qa_config = config_manager.get_config("gdb")

    global_config = config_manager.get_global_config()

    return qa_config, global_config


def get_configs(ctx) -> tuple[QAConfig, GlobalConfig, str, bool]:
    app_config: AppConfig = load_config(
        environment=ctx.obj["environment"]
    )  # ctx.obj["app_config"]
    logger.info(f"env: {ctx.obj['environment']}")
    return (
        app_config.qa,
        app_config.global_,
        ctx.obj["environment"],
        ctx.obj.get("verbose", False),
    )


@click.group()
def qa():
    """Process verification FileGDB results."""
    pass


@qa.command()
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path))
@click.option("--no-upload", is_flag=True, help="Skip S3 upload")
@click.option(
    "--format",
    type=click.Choice(["geoparquet", "geojson", "both"]),
    default="geoparquet",
)
@click.option("--simplify-tolerance", type=float)
@click.pass_context
def process(
    ctx,
    gdb_path: Path,
    output_dir: Optional[Path],
    no_upload: bool,
    format: str,
    simplify_tolerance: Optional[float],
):
    """
    Process a single verification FileGDB.

    Converts spatial layers to web formats and generates statistics.
    Handles complex geometries that may cause issues with standard readers.

    Examples:
        # Basic processing
        gcover verification process /path/to/issue.gdb

        # With geometry simplification for complex polygons
        gcover verification process /path/to/issue.gdb --simplify-tolerance 1.0

        # Verbose output for debugging
        gcover verification process /path/to/issue.gdb --verbose
    """
    # from ..gdb.config import load_config TODO
    try:
        # TODO qa_config, global_config = get_qa_config(ctx)
        qa_config, global_config, environment, verbose = get_configs(ctx)

        # Get S3 settings from global config
        s3_bucket = qa_config.get_s3_bucket(global_config)
        s3_profile = qa_config.get_s3_profile(global_config)

    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Make sure your configuration includes global S3 settings:")
        console.print("  global.s3.bucket, global.s3.profile")
        raise click.Abort()

    if verbose:
        console.print(f"[dim]S3 Bucket: {s3_bucket}[/dim]")
        console.print(f"[dim]S3 Profile: {s3_profile or 'default'}[/dim]")

    if simplify_tolerance:
        console.print(
            f"[yellow]⚠️  Geometry simplification enabled (tolerance: {simplify_tolerance})[/yellow]"
        )
        console.print("This will reduce geometry complexity but may affect precision.")

        # Import and use the QA converter
    from ..gdb.qa_converter import FileGDBConverter

    # TODO: fix config
    config_wrapper = ConfigWrapper(qa_config, global_config)
    console.print(config_wrapper)
    converter = FileGDBConverter(config=config_wrapper)

    try:
        summary = converter.process_gdb(
            gdb_path=gdb_path,
            output_dir=output_dir,
            upload_to_s3=not no_upload,
            output_format=format,
            simplify_tolerance=simplify_tolerance,
        )

        # Display summary
        console.print(f"\n[green]✅ Processing complete![/green]")
        console.print(f"S3 Bucket: {s3_bucket}")
        console.print(f"Total Features: {summary.total_features:,}")

        # Display summary
        console.print(f"\n[green]✅ Processing complete![/green]")
        console.print(f"Verification Type: {summary.verification_type}")
        console.print(f"RC Version: {summary.rc_version}")
        console.print(f"Timestamp: {summary.timestamp}")
        console.print(f"Total Features: {summary.total_features:,}")
        console.print(f"S3 Bucket: {s3_bucket}")

        # Show layer breakdown
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

            table.add_row(layer_name, f"{stats.feature_count:,}", top_test, top_issue)

        console.print(table)

        # Show helpful tips if processing was successful
        if summary.total_features > 0:
            console.print(f"\n[dim]💡 Tips:[/dim]")
            if not simplify_tolerance:
                console.print(
                    "[dim]  - If you encounter geometry errors, try --simplify-tolerance 1.0[/dim]"
                )
            console.print(
                "[dim]  - Use --verbose for detailed processing information[/dim]"
            )
            console.print(
                "[dim]  - Check 'gcover verification stats' for analysis[/dim]"
            )

    except Exception as e:
        console.print(f"[red]❌ Processing failed: {e}[/red]")
        if not verbose:
            console.print("[dim]Use --verbose for detailed error information[/dim]")
        raise

    finally:
        converter.close()


@qa.command()
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option("--layer", help="Specific layer to diagnose (default: all layers)")
def diagnose(gdb_path: Path, layer: Optional[str]):
    """
    Diagnose FileGDB structure and identify potential issues.

    Use this command to investigate geometry problems before processing.

    Example:
        gcover verification diagnose /path/to/issue.gdb
        gcover verification diagnose /path/to/issue.gdb --layer IssuePolygons
    """
    import fiona
    from rich.panel import Panel

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
                console.print(
                    f"[red]Layer '{layer}' not found. Available layers: {', '.join(layers)}[/red]"
                )
                return

        # Diagnose each layer
        for layer_name in layers:
            try:
                console.print(f"\n[cyan]📊 Layer: {layer_name}[/cyan]")

                with fiona.open(str(gdb_path), layer=layer_name) as src:
                    # Basic info
                    info_table = Table(show_header=False, box=None)
                    info_table.add_column("Property", style="dim")
                    info_table.add_column("Value")

                    info_table.add_row("Feature Count", f"{len(src):,}")
                    info_table.add_row(
                        "Geometry Type", str(src.schema.get("geometry", "None"))
                    )
                    info_table.add_row("CRS", str(src.crs))

                    # Field information
                    fields = src.schema.get("properties", {})
                    field_list = ", ".join(fields.keys()) if fields else "None"
                    info_table.add_row(
                        "Fields",
                        field_list[:100] + ("..." if len(field_list) > 100 else ""),
                    )

                    console.print(info_table)

                    # Sample features to check for geometry complexity
                    if len(src) > 0:
                        complex_geoms = 0
                        invalid_geoms = 0
                        total_parts = 0
                        max_parts = 0

                        sample_size = min(100, len(src))  # Sample first 100 features

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

                        # Geometry analysis results
                        if src.schema.get("geometry") != "None":
                            geom_table = Table(
                                title="Geometry Analysis", show_header=False, box=None
                            )
                            geom_table.add_column("Metric", style="dim")
                            geom_table.add_column("Value")

                            if total_parts > 0:
                                avg_parts = total_parts / sample_size
                                geom_table.add_row(
                                    "Avg Parts/Feature", f"{avg_parts:.1f}"
                                )
                                geom_table.add_row("Max Parts/Feature", str(max_parts))

                            geom_table.add_row(
                                "Complex Geometries", f"{complex_geoms}/{sample_size}"
                            )
                            geom_table.add_row(
                                "Invalid Geometries", f"{invalid_geoms}/{sample_size}"
                            )

                            console.print(geom_table)

                            # Recommendations
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
                                        "\n".join(
                                            f"• {rec}" for rec in recommendations
                                        ),
                                        title="[yellow]⚠️  Recommendations[/yellow]",
                                        border_style="yellow",
                                    )
                                )

            except Exception as e:
                console.print(f"[red]Error diagnosing layer {layer_name}: {e}[/red]")

        # Overall recommendations
        console.print(f"\n[bold green]✅ Diagnosis complete[/bold green]")
        console.print(
            "[dim]Use 'gcover verification process --help' for processing options[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Failed to diagnose FileGDB: {e}[/red]")


@qa.command("process-all")
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--layer", default="IssuePolygons", help="Layer to test (default: IssuePolygons)"
)
@click.option(
    "--max-features",
    type=int,
    default=10,
    help="Maximum features to test read (default: 10)",
)
def test_read(gdb_path: Path, layer: str, max_features: int):
    """
    Test reading a FileGDB layer with different strategies.

    Use this to determine the best approach for problematic FileGDBs.

    Example:
        gcover verification test-read /path/to/issue.gdb --layer IssuePolygons
    """
    from ..gdb.qa_converter import FileGDBConverter

    # from ..gdb.config import load_config TODO
    from ...config import load_config, GDBConfig, SDEConfig, SchemaConfig

    console.print(f"[bold blue]🧪 Testing read strategies for:[/bold blue] {gdb_path}")
    console.print(f"[dim]Layer: {layer}, Max features: {max_features}[/dim]")

    try:
        config = load_config()
        converter = FileGDBConverter(config=config)

        # Test each reading strategy individually
        strategies = [
            ("PyOGRIO", "pyogrio"),
            ("GeoPandas", "geopandas"),
            ("Fiona Fallback", "fiona"),
        ]

        for strategy_name, strategy_method in strategies:
            console.print(f"\n[cyan]📖 Testing {strategy_name}...[/cyan]")

            try:
                if strategy_method == "pyogrio":
                    import pyogrio

                    result = pyogrio.read_dataframe(
                        str(gdb_path),
                        layer=layer,
                        max_features=max_features,
                        use_arrow=False,
                    )

                elif strategy_method == "geopandas":
                    import geopandas as gpd
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        result = gpd.read_file(str(gdb_path), layer=layer)
                        if len(result) > max_features:
                            result = result.head(max_features)

                elif strategy_method == "fiona":
                    result = converter._read_with_fiona_fallback(gdb_path, layer)
                    if result is not None and len(result) > max_features:
                        result = result.head(max_features)

                if result is not None and not result.empty:
                    console.print(
                        f"[green]✅ {strategy_name}: Read {len(result)} features successfully[/green]"
                    )

                    # Show some stats
                    if hasattr(result, "geometry") and "geometry" in result.columns:
                        valid_geoms = result.geometry.is_valid.sum()
                        console.print(
                            f"[dim]   Valid geometries: {valid_geoms}/{len(result)}[/dim]"
                        )
                else:
                    console.print(
                        f"[yellow]⚠️  {strategy_name}: No data returned[/yellow]"
                    )

            except Exception as e:
                console.print(f"[red]❌ {strategy_name}: {str(e)[:100]}...[/red]")

        converter.close()

        console.print(f"\n[bold green]✅ Testing complete[/bold green]")
        console.print(
            "[dim]The working strategy will be used automatically during processing[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Failed to diagnose FileGDB: {e}[/red]")


@qa.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--pattern", default="**/issue.gdb", help="Glob pattern to find GDB files"
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
@click.pass_context
def process_all(
    ctx,
    directory: Path,
    pattern: str,
    dry_run: bool,
    simplify_tolerance: Optional[float],
):
    """
    Process multiple verification FileGDBs in a directory.

    Example:
        gcover verification batch /media/marco/SANDISK/Verifications
        gcover verification batch /path/to/verifications --simplify-tolerance 1.0
    """
    # from ..gdb.config import load_config TODO
    from gcover.config import load_config, GDBConfig, SDEConfig, SchemaConfig, QAConfig
    from gcover.gdb.qa_converter import FileGDBConverter

    try:
        # TODO qa_config, global_config = get_qa_config(ctx)
        qa_config, global_config, environment, verbose = get_configs(ctx)

        # Get S3 settings from global config
        s3_bucket = qa_config.get_s3_bucket(global_config)
        s3_profile = qa_config.get_s3_profile(global_config)

    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Make sure your configuration includes global S3 settings:")
        console.print("  global.s3.bucket, global.s3.profile")
        raise click.Abort()

    """try:
        config = load_config(config_file, environment)
    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()"""

    # Find all matching GDB files
    gdb_files = list(directory.glob(pattern))

    if not gdb_files:
        console.print(f"[red]No GDB files found matching pattern: {pattern}[/red]")
        return

    console.print(f"Found {len(gdb_files)} GDB files to process")
    console.print(f"S3 Bucket: {s3_bucket}")

    if simplify_tolerance:
        console.print(f"[yellow]Geometry simplification: {simplify_tolerance}[/yellow]")

    if dry_run:
        table = Table(title="Files to Process (Dry Run)")
        table.add_column("Path")
        table.add_column("Size")
        table.add_column("Modified")

        for gdb_path in sorted(gdb_files):
            try:
                size = sum(f.stat().st_size for f in gdb_path.rglob("*") if f.is_file())
                modified = gdb_path.stat().st_mtime
                import datetime

                mod_time = datetime.datetime.fromtimestamp(modified).strftime(
                    "%Y-%m-%d %H:%M"
                )
                table.add_row(str(gdb_path), f"{size / 1024 / 1024:.1f} MB", mod_time)
            except Exception:
                table.add_row(str(gdb_path), "N/A", "N/A")

        console.print(table)
        return

    # Process each file
    # TODO: fix config
    config_wrapper = ConfigWrapper(qa_config, global_config)
    console.print(config_wrapper)
    converter = FileGDBConverter(config=config_wrapper)

    try:
        processed = 0
        failed = 0
        skipped = 0

        for gdb_path in gdb_files:
            try:
                console.print(
                    f"\n[blue]Processing {processed + failed + skipped + 1}/{len(gdb_files)}: {gdb_path.name}[/blue]"
                )

                summary = converter.process_gdb(
                    gdb_path, simplify_tolerance=simplify_tolerance
                )

                if summary.total_features > 0:
                    processed += 1
                    console.print(
                        f"[green]✅ Processed {summary.total_features:,} features[/green]"
                    )
                else:
                    skipped += 1
                    console.print(
                        f"[yellow]⚠️  Skipped - no processable features[/yellow]"
                    )

            except Exception as e:
                failed += 1
                console.print(f"[red]❌ Failed: {e}[/red]")
                continue

        console.print(f"\n[bold]Batch processing complete![/bold]")
        console.print(f"✅ Processed: {processed}")
        console.print(f"⚠️  Skipped: {skipped}")
        console.print(f"❌ Failed: {failed}")

        if failed > 0:
            console.print(
                "[dim]💡 For failed files, try using --simplify-tolerance or diagnose them individually[/dim]"
            )

    finally:
        converter.close()


'''
@click.option(
    "--pattern", default="**/issue.gdb", help="Glob pattern to find GDB files"
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
@click.option(
    "--environment",
    "-e",
    default="development",
    help="Environment (development/production)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without actually processing",
)
def batch(
    directory: Path,
    pattern: str,
    config_file: Optional[Path],
    environment: str,
    dry_run: bool,
):
    """
    Process multiple verification FileGDBs in a directory.

    Example:
        gcover verification batch /media/marco/SANDISK/Verifications
    """
    # from ..gdb.config import load_config TODO
    from ...config import load_config, GDBConfig, SDEConfig, SchemaConfig
    from ..gdb.qa_converter import FileGDBConverter

    try:
        config = load_config(config_file, environment)
    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()

    # Find all matching GDB files
    gdb_files = list(directory.glob(pattern))

    if not gdb_files:
        console.print(f"[red]No GDB files found matching pattern: {pattern}[/red]")
        return

    console.print(f"Found {len(gdb_files)} GDB files to process")
    console.print(f"S3 Bucket: {config.s3_bucket}")

    if dry_run:
        table = Table(title="Files to Process (Dry Run)")
        table.add_column("Path")
        table.add_column("Size")

        for gdb_path in sorted(gdb_files):
            try:
                size = sum(f.stat().st_size for f in gdb_path.rglob("*") if f.is_file())
                table.add_row(str(gdb_path), f"{size / 1024 / 1024:.1f} MB")
            except Exception:
                table.add_row(str(gdb_path), "N/A")

        console.print(table)
        return

    # Process each file
    converter = FileGDBConverter(config=config)

    try:
        processed = 0
        failed = 0

        for gdb_path in gdb_files:
            try:
                console.print(
                    f"\n[blue]Processing {processed + 1}/{len(gdb_files)}: {gdb_path.name}[/blue]"
                )
                summary = converter.process_gdb(gdb_path)
                processed += 1
                console.print(
                    f"[green]✅ Processed {summary.total_features:,} features[/green]"
                )

            except Exception as e:
                failed += 1
                console.print(f"[red]❌ Failed: {e}[/red]")
                continue

        console.print(f"\n[bold]Batch processing complete![/bold]")
        console.print(f"Processed: {processed}")
        console.print(f"Failed: {failed}")

    finally:
        converter.close()
'''


@qa.command()
@click.option(
    "--qa-type",
    type=click.Choice(["Topology", "TechnicalQualityAssurance"]),
    help="Filter by QA type",
)
@click.option("--rc-version", help="Filter by RC version (e.g., 2030-12-31)")
@click.option("--days-back", type=int, default=30, help="Number of days to look back")
@click.option(
    "--export-csv", type=click.Path(path_type=Path), help="Export results to CSV file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def stats(
    ctx,
    qa_type: Optional[str],
    rc_version: Optional[str],
    days_back: int,
    export_csv: Optional[Path],
    verbose: bool,
):
    """
    Display verification statistics from the database.

    Example:
        gcover verification stats --verification-type Topology --days-back 7
    """
    # from ..gdb.config import load_config TODO
    from gcover.config import load_config, GDBConfig, SDEConfig, SchemaConfig
    from gcover.gdb.qa_converter import FileGDBConverter
    from gcover.core.config import convert_rc, get_all_rcs

    from loguru import logger

    log_level = "INFO"

    if verbose:
        log_level = "DEBUG"
        console.print("[dim]Verbose logging enabled[/dim]")
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    try:
        # config = load_config(config_file, environment)
        qa_config, global_config, environment, verbose = get_configs(ctx)
        console.log(qa_config)
    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()

    # Check if database exists
    # verification_db = config.db_path.parent / "verification_stats.duckdb"
    verification_db = qa_config.db_path
    console.print(f"[dim]Using DB {verification_db}[/dim]")
    if not verification_db.exists():
        console.print(f"[red]Statistics database not found: {verification_db}[/red]")
        console.print("Run 'gcover verification process' first to generate statistics.")
        return

    config_wrapper = ConfigWrapper(qa_config, global_config)
    console.print(config_wrapper)
    converter = FileGDBConverter(config=config_wrapper)

    try:
        if rc_version:
            rc_version = convert_rc(rc_version.upper(), force="long")
            logger.debug(rc_version)
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
        table.add_column("Verification Type")
        table.add_column("RC Version")
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

    finally:
        converter.close()


@qa.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
@click.option(
    "--environment",
    "-e",
    default="development",
    help="Environment (development/production)",
)
def dashboard(config_file: Optional[Path], environment: str):
    """
    Generate a simple HTML dashboard for verification statistics.

    Creates a local HTML file with charts and statistics.
    """
    # from ..gdb.config import load_config TODO
    from ...config import load_config, GDBConfig, SDEConfig, SchemaConfig
    from ..gdb.qa_converter import FileGDBConverter

    try:
        config = load_config(config_file, environment)
    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()

    # Check if database exists
    verification_db = config.db_path.parent / "verification_stats.duckdb"
    if not verification_db.exists():
        console.print(f"[red]Statistics database not found: {verification_db}[/red]")
        return

    converter = FileGDBConverter(config=config)

    try:
        # Get recent data
        df = converter.get_statistics_summary(days_back=90)

        if df.empty:
            console.print("[yellow]No data available for dashboard[/yellow]")
            return

        # Generate HTML dashboard
        html_content = _generate_dashboard_html(df)

        dashboard_path = Path("verification_dashboard.html")
        dashboard_path.write_text(html_content)

        console.print(
            f"[green]Dashboard generated: {dashboard_path.absolute()}[/green]"
        )
        console.print(f"Open in browser: file://{dashboard_path.absolute()}")

    finally:
        converter.close()


def _generate_dashboard_html(df: pd.DataFrame) -> str:
    """Generate HTML dashboard content."""
    # Simple HTML template with basic charts
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>GeoCover Verification Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; 
                  background: #f5f5f5; border-radius: 5px; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>GeoCover Verification Dashboard</h1>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Issues</h3>
            <p style="font-size: 2em; color: #e74c3c;">{df["total_count"].sum():,}</p>
        </div>
        <div class="metric">
            <h3>Unique Tests</h3>
            <p style="font-size: 2em; color: #3498db;">{df["test_name"].nunique()}</p>
        </div>
        <div class="metric">
            <h3>Error Issues</h3>
            <p style="font-size: 2em; color: #e74c3c;">
                {df[df["issue_type"].str.lower() == "error"]["total_count"].sum():,}
            </p>
        </div>
        <div class="metric">
            <h3>Warning Issues</h3>
            <p style="font-size: 2em; color: #f39c12;">
                {df[df["issue_type"].str.lower() == "warning"]["total_count"].sum():,}
            </p>
        </div>
    </div>
    
    <div id="top-tests" class="chart"></div>
    <div id="issue-types" class="chart"></div>
    
    <script>
        // Top tests chart
        var topTests = {
        df.groupby("test_name")["total_count"].sum().head(10).to_dict()
    };
        var topTestsData = [{{
            x: Object.keys(topTests),
            y: Object.values(topTests),
            type: 'bar'
        }}];
        Plotly.newPlot('top-tests', topTestsData, {{
            title: 'Top 10 Tests by Issue Count',
            xaxis: {{ tickangle: -45 }}
        }});
        
        // Issue types pie chart
        var issueTypes = {df.groupby("issue_type")["total_count"].sum().to_dict()};
        var issueTypesData = [{{
            values: Object.values(issueTypes),
            labels: Object.keys(issueTypes),
            type: 'pie'
        }}];
        Plotly.newPlot('issue-types', issueTypesData, {{
            title: 'Issues by Type'
        }});
    </script>
    
    <h2>Recent Issues (Top 20)</h2>
    <table border="1" style="width: 100%; border-collapse: collapse;">
        <tr>
            <th>Test Name</th>
            <th>Issue Type</th>
            <th>Count</th>
            <th>Latest Run</th>
        </tr>
        {
        "".join(
            [
                f"<tr><td>{row['test_name']}</td><td>{row['issue_type']}</td>"
                f"<td>{row['total_count']:,}</td><td>{row['latest_run']}</td></tr>"
                for _, row in df.head(20).iterrows()
            ]
        )
    }
    </table>
    
</body>
</html>
"""


# Register the command group with the main CLI
# This would be added to your main CLI module
def register_verification_commands(main_cli):
    """Register verification commands with the main CLI."""
    main_cli.add_command(qa)
