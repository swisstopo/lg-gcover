#!/usr/bin/env python3
"""
CLI for GDB Asset Management System
"""

import click
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta

import duckdb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

# Import our GDB management classes (assuming they're in a module)

from gcover.gdb.manager import GDBAssetManager
from gcover.gdb.storage import S3Uploader, MetadataDB

# from gcover.gdb.config import  load_config TODO
from gcover.config import load_config, AppConfig
from gcover.config.models import GDBConfig, GlobalConfig
from gcover.gdb.assets import (
    GDBAsset,
    BackupGDBAsset,
    VerificationGDBAsset,
    IncrementGDBAsset,
    AssetType,
    ReleaseCandidate,
)


from loguru import logger


console = Console()


def get_configs(ctx) -> tuple[GDBConfig, GlobalConfig, str, bool]:
    app_config: AppConfig = load_config(
        environment=ctx.obj["environment"]
    )  # ctx.obj["app_config"]
    logger.info(f"env: {ctx.obj['environment']}")
    return (
        app_config.gdb,
        app_config.global_,
        ctx.obj["environment"],
        ctx.obj.get("verbose", False),
    )


@click.group()
@click.pass_context
def gdb(ctx):
    """GDB Asset Management commands"""
    # Ensure context object exists and has required keys
    if ctx.obj is None:
        ctx.ensure_object(dict)

    # Set defaults if not provided by parent gcover command
    ctx.obj.setdefault("environment", "development")
    ctx.obj.setdefault("verbose", False)
    ctx.obj.setdefault("config_path", None)


@gdb.command()
@click.pass_context
def init(ctx):
    """Initialize GDB management system"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    rprint(f"[cyan]Initializing GDB system ({environment})...[/cyan]")

    try:
        # Test database
        db = MetadataDB(gdb_config.db_path)
        rprint(f"[green]✅ Database initialized: {gdb_config.db_path}[/green]")

        # Test S3 using global config
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        s3 = S3Uploader(s3_bucket, s3_profile)
        rprint("[green]✅ S3 connection ready[/green]")

        if verbose:
            rprint(f"[dim]S3 Bucket: {s3_bucket}[/dim]")
            rprint(f"[dim]S3 Profile: {s3_profile or 'default'}[/dim]")
            rprint(f"[dim]S3 Region: {global_config.s3.region or 'default'}[/dim]")
            rprint(f"[dim]Temp Dir: {gdb_config.temp_dir}[/dim]")

        rprint("[green]Initialization complete![/green]")

    except Exception as e:
        rprint(f"[red]Initialization failed: {e}[/red]")
        sys.exit(1)


@gdb.command()
@click.pass_context
def scan(ctx):
    """Scan filesystem for GDB assets"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Get S3 settings from global config
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,  # From global config
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,  # From global config
        )

        rprint("[cyan]Scanning filesystem...[/cyan]")
        if verbose:
            for name, path in gdb_config.base_paths.items():
                rprint(f"[dim]  {name}: {path}[/dim]")

        assets = manager.scan_filesystem()

        if assets:
            # Display results (same as before)
            by_type = {}
            for asset in assets:
                asset_type = asset.info.asset_type.value
                if asset_type not in by_type:
                    by_type[asset_type] = []
                by_type[asset_type].append(asset)

            table = Table(title="GDB Assets Found")
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right", style="magenta")
            table.add_column("Latest", style="yellow")

            for asset_type, asset_list in by_type.items():
                latest = max(asset_list, key=lambda a: a.info.timestamp)
                table.add_row(
                    asset_type,
                    str(len(asset_list)),
                    latest.info.timestamp.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(table)
            rprint(f"\n[green]Total: {len(assets)} GDB assets found[/green]")
        else:
            rprint("[yellow]No GDB assets found[/yellow]")

    except Exception as e:
        rprint(f"[red]Scan failed: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@gdb.command()
@click.option("--dry-run", is_flag=True, help="Show what would be done")
@click.pass_context
def sync(ctx, dry_run):
    """Sync GDB assets to S3 and database"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Get S3 settings from global config
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,
        )

        if dry_run:
            rprint("[yellow]DRY RUN MODE - No changes will be made[/yellow]")

            assets = manager.scan_filesystem()
            new_assets = [
                a for a in assets if not manager.metadata_db.asset_exists(a.path)
            ]

            if new_assets:
                rprint(f"[cyan]Would sync {len(new_assets)} new assets to:[/cyan]")
                rprint(f"[cyan]  S3 Bucket: {s3_bucket}[/cyan]")
                for asset in new_assets[:10]:
                    rprint(f"  - {asset.path.name}")
                if len(new_assets) > 10:
                    rprint(f"  ... and {len(new_assets) - 10} more")
            else:
                rprint("[green]No new assets to sync[/green]")
        else:
            rprint("[cyan]Starting sync...[/cyan]")
            if verbose:
                rprint(f"[dim]Environment: {environment}[/dim]")
                rprint(f"[dim]S3 Bucket: {s3_bucket}[/dim]")
                rprint(f"[dim]Workers: {gdb_config.processing.max_workers}[/dim]")

            stats = manager.sync_all()

            table = Table(title="Sync Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="magenta")

            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))

            console.print(table)

            if stats.get("failed", 0) > 0:
                rprint(
                    "[yellow]Some assets failed to sync. Check logs for details.[/yellow]"
                )
            else:
                rprint("[green]All assets synced successfully![/green]")

    except Exception as e:
        rprint(f"[red]Sync failed: {e}[/red]")
        sys.exit(1)


# TODO: use central config
@gdb.command("list-assets")
@click.option(
    "--type",
    "asset_type",
    type=click.Choice([t.value for t in AssetType]),
    help="Filter by asset type",
)
@click.option(
    "--rc", type=click.Choice(["RC1", "RC2"]), help="Filter by release candidate"
)
@click.option("--since", type=str, help="Show assets since date (YYYY-MM-DD)")
@click.option("--limit", type=int, default=20, help="Limit number of results")
@click.pass_context
def list_assets(ctx, asset_type, rc, since, limit):
    """List GDB assets from database"""

    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        db = MetadataDB(gdb_config.db_path)

        query = "SELECT * FROM gdb_assets WHERE 1=1"
        params = []

        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)

        if rc:
            rc_value = (
                ReleaseCandidate.RC1.value
                if rc == "RC1"
                else ReleaseCandidate.RC2.value
            )
            query += " AND release_candidate = ?"
            params.append(rc_value)

        if since:
            query += " AND timestamp >= ?"
            params.append(datetime.strptime(since, "%Y-%m-%d"))

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with duckdb.connect(str(db.db_path)) as conn:
            results = conn.execute(query, params).fetchall()
            columns = [desc[0] for desc in conn.description]

        if not results:
            rprint("[yellow]No assets found matching criteria[/yellow]")
            return

        table = Table(title="GDB Assets")
        table.add_column("Path", style="cyan", max_width=40)
        table.add_column("Type", style="green")
        table.add_column("RC", style="yellow")
        table.add_column("Date", style="magenta")
        table.add_column("Size", justify="right", style="blue")
        table.add_column("Uploaded", style="red")

        for row in results:
            data = dict(zip(columns, row))
            table.add_row(
                Path(data["path"]).name,
                data["asset_type"],
                "RC1"
                if data["release_candidate"] == ReleaseCandidate.RC1.value
                else "RC2",
                data["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                f"{data['file_size'] / 1024 / 1024:.1f} MB"
                if data["file_size"]
                else "N/A",
                "✅" if data["uploaded"] else "❌",
            )

        console.print(table)
    except Exception as e:
        rprint(f"[red]Sync failed: {e}[/red]")
        if ctx.obj["verbose"]:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


# TODO: use central config
@gdb.command()
@click.argument("search_term")
@click.option("--download", is_flag=True, help="Download the asset")
@click.option(
    "--output-dir", type=click.Path(), default="./downloads", help="Download directory"
)
@click.pass_context
def search(ctx, search_term, download, output_dir):
    """Search for GDB assets in the database"""

    gdb_config, global_config, environment, verbose = get_config(ctx)

    try:
        db = MetadataDB(gdb_config.db_path)

        query = """
        SELECT * FROM gdb_assets 
        WHERE path ILIKE ? OR asset_type ILIKE ?
        ORDER BY timestamp DESC
      """

        search_pattern = f"%{search_term}%"

        with duckdb.connect(str(db.db_path)) as conn:
            results = conn.execute(query, [search_pattern, search_pattern]).fetchall()
            columns = [desc[0] for desc in conn.description]

        if not results:
            rprint(f"[yellow]No assets found matching '{search_term}'[/yellow]")
            return

        rprint(f"[green]Found {len(results)} matching assets[/green]")

        for i, row in enumerate(results, 1):
            data = dict(zip(columns, row))
            rprint(f"\n[cyan]{i}.[/cyan] {Path(data['path']).name}")
            rprint(f"   Type: {data['asset_type']}")
            rprint(
                f"   RC: {'RC1' if data['release_candidate'] == ReleaseCandidate.RC1.value else 'RC2'}"
            )
            rprint(f"   Date: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            rprint(
                f"   Size: {data['file_size'] / 1024 / 1024:.1f} MB"
                if data["file_size"]
                else "N/A"
            )
            rprint(f"   S3 Key: {data['s3_key']}")

            if download and data["s3_key"] and data["uploaded"]:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                s3_uploader = S3Uploader(config.s3_bucket)
                filename = Path(data["s3_key"]).name
                local_path = output_path / filename

                try:
                    s3_uploader.s3_client.download_file(
                        config.s3_bucket, data["s3_key"], str(local_path)
                    )
                    rprint(f"   [green]Downloaded to: {local_path}[/green]")
                except Exception as e:
                    rprint(f"   [red]Download failed: {e}[/red]")
    except Exception as e:
        rprint(f"[red]Sync failed: {e}[/red]")
        if ctx.obj["verbose"]:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@gdb.command()
@click.pass_context
def status(ctx):
    """Show system status and statistics"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        db = MetadataDB(gdb_config.db_path)
        s3_bucket = gdb_config.get_s3_bucket(global_config)

        with duckdb.connect(str(db.db_path)) as conn:
            # Basic stats (same as before)
            total = conn.execute("SELECT COUNT(*) FROM gdb_assets").fetchone()[0]
            uploaded = conn.execute(
                "SELECT COUNT(*) FROM gdb_assets WHERE uploaded = true"
            ).fetchone()[0]

            size_result = conn.execute(
                "SELECT SUM(file_size) FROM gdb_assets WHERE file_size IS NOT NULL"
            ).fetchone()[0]
            total_size_gb = round(size_result / (1024**3), 2) if size_result else 0

            by_type = conn.execute("""
                SELECT asset_type, COUNT(*) 
                FROM gdb_assets 
                GROUP BY asset_type
            """).fetchall()

        # Display results
        table = Table(title="System Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")

        table.add_row("Total Assets", str(total))
        table.add_row(
            "Uploaded to S3",
            f"{uploaded} ({uploaded / total * 100:.1f}%)" if total > 0 else "0",
        )
        table.add_row("Total Size", f"{total_size_gb} GB")
        table.add_row("Environment", environment)
        table.add_row("S3 Bucket", s3_bucket)  # From global config
        table.add_row("Database", str(gdb_config.db_path))

        console.print(table)

        if by_type:
            rprint("\n[cyan]Assets by Type:[/cyan]")
            for asset_type, count in by_type:
                rprint(f"  {asset_type}: {count}")

        if verbose:
            rprint(f"\n[dim]Configuration Details:[/dim]")
            rprint(f"[dim]  Log Level: {global_config.log_level}[/dim]")
            rprint(f"[dim]  Max Workers: {gdb_config.processing.max_workers}[/dim]")
            rprint(
                f"[dim]  Compression: {gdb_config.processing.compression_level}[/dim]"
            )
            rprint(f"[dim]  Temp Dir: {gdb_config.temp_dir}[/dim]")
            rprint(f"[dim]  S3 Profile: {global_config.s3.profile or 'default'}[/dim]")

    except Exception as e:
        rprint(f"[red]Status check failed: {e}[/red]")
        sys.exit(1)


@gdb.command()
@click.argument("gdb_path", type=click.Path(exists=True))
@click.pass_context
def process(ctx, gdb_path):
    """Process a single GDB asset"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Get S3 settings from global config
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,
        )

        asset = manager.create_asset(Path(gdb_path))
        rprint(f"[cyan]Processing: {gdb_path}[/cyan]")
        rprint(f"Type: {asset.info.asset_type.value}")
        rprint(f"RC: {asset.info.release_candidate.short_name}")
        rprint(f"Date: {asset.info.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        success = manager.process_asset(asset)

        if success:
            rprint("[green]✅ Successfully processed[/green]")
        else:
            rprint("[red]❌ Processing failed[/red]")
            sys.exit(1)

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        sys.exit(1)


@gdb.command("process-all")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be processed without doing it"
)
@click.option(
    "--force", is_flag=True, help="Reprocess assets even if already in database"
)
@click.option(
    "--filter-type",
    type=click.Choice([t.value for t in AssetType]),
    help="Only process specific asset type",
)
@click.option(
    "--filter-rc",
    type=click.Choice(["RC1", "RC2"]),
    help="Only process specific release candidate",
)
@click.option(
    "--since", type=str, help="Only process assets modified since date (YYYY-MM-DD)"
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help="Number of parallel workers (be careful with disk I/O)",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue processing other assets if one fails",
)
@click.pass_context
def process_all(
    ctx, dry_run, force, filter_type, filter_rc, since, max_workers, continue_on_error
):
    """Process all GDB assets found by filesystem scan"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Get S3 settings from global config
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,
        )

        rprint("[cyan]Scanning filesystem for GDB assets...[/cyan]")
        assets = manager.scan_filesystem()

        if not assets:
            rprint("[yellow]No GDB assets found[/yellow]")
            return

        # Apply filters
        filtered_assets = []
        since_date = None
        if since:
            try:
                since_date = datetime.strptime(since, "%Y-%m-%d")
            except ValueError:
                rprint(f"[red]Invalid date format: {since}. Use YYYY-MM-DD[/red]")
                sys.exit(1)

        for asset in assets:
            # Filter by type
            if filter_type and asset.info.asset_type.value != filter_type:
                continue

            # Filter by RC
            if filter_rc:
                rc_value = (
                    ReleaseCandidate.RC1 if filter_rc == "RC1" else ReleaseCandidate.RC2
                )
                if asset.info.release_candidate != rc_value:
                    continue

            # Filter by date
            if since_date and asset.info.timestamp < since_date:
                continue

            # Check if already processed (unless force is specified)
            if not force and manager.metadata_db.asset_exists(asset.path):
                if verbose:
                    rprint(f"[dim]Skipping already processed: {asset.path.name}[/dim]")
                continue

            filtered_assets.append(asset)

        if not filtered_assets:
            rprint("[yellow]No assets match the specified criteria[/yellow]")
            return

        # Show summary
        rprint(f"\n[green]Found {len(filtered_assets)} assets to process[/green]")

        # Group by type for summary
        by_type = {}
        for asset in filtered_assets:
            asset_type = asset.info.asset_type.value
            by_type.setdefault(asset_type, []).append(asset)

        summary_table = Table(title="Assets to Process")
        summary_table.add_column("Type", style="cyan")
        summary_table.add_column("Count", justify="right", style="magenta")
        summary_table.add_column("Size Range", style="yellow")

        for asset_type, asset_list in by_type.items():
            sizes = []
            for asset in asset_list:
                try:
                    size = sum(
                        f.stat().st_size for f in asset.path.rglob("*") if f.is_file()
                    )
                    sizes.append(size)
                except:
                    pass

            if sizes:
                min_size = min(sizes) / (1024**2)  # MB
                max_size = max(sizes) / (1024**2)  # MB
                size_range = f"{min_size:.1f}-{max_size:.1f} MB"
            else:
                size_range = "Unknown"

            summary_table.add_row(asset_type, str(len(asset_list)), size_range)

        console.print(summary_table)

        if dry_run:
            rprint("\n[yellow]DRY RUN - Would process these assets:[/yellow]")
            for asset in filtered_assets[:10]:  # Show first 10
                rprint(f"  - {asset.path.name} ({asset.info.asset_type.value})")
            if len(filtered_assets) > 10:
                rprint(f"  ... and {len(filtered_assets) - 10} more")
            return

        # Confirm before processing
        if not click.confirm(f"\nProcess {len(filtered_assets)} assets?"):
            rprint("[yellow]Cancelled[/yellow]")
            return

        # Process assets
        rprint(f"\n[cyan]Processing {len(filtered_assets)} assets...[/cyan]")

        stats = {"processed": 0, "failed": 0, "skipped": 0, "total_size": 0}

        failed_assets = []

        if max_workers > 1:
            rprint(f"[cyan]Using {max_workers} parallel workers[/cyan]")
            # For parallel processing, you'd need to implement this
            # For now, fall back to serial processing
            rprint(
                "[yellow]Parallel processing not implemented yet, using serial processing[/yellow]"
            )
            max_workers = 1

        with Progress(console=console) as progress:
            task = progress.add_task(
                "Processing GDB assets...", total=len(filtered_assets)
            )

            for i, asset in enumerate(filtered_assets, 1):
                progress.update(
                    task, advance=1, description=f"Processing {asset.path.name}..."
                )

                if verbose:
                    rprint(
                        f"\n[cyan]{i}/{len(filtered_assets)}: {asset.path.name}[/cyan]"
                    )
                    rprint(f"  Type: {asset.info.asset_type.value}")
                    rprint(f"  RC: {asset.info.release_candidate.short_name}")
                    rprint(f"  Date: {asset.info.timestamp.strftime('%Y-%m-%d %H:%M')}")

                try:
                    # Calculate size before processing
                    try:
                        size = sum(
                            f.stat().st_size
                            for f in asset.path.rglob("*")
                            if f.is_file()
                        )
                        stats["total_size"] += size
                    except:
                        pass

                    success = manager.process_asset(asset)

                    if success:
                        stats["processed"] += 1
                        if verbose:
                            rprint(f"  [green]✅ Success[/green]")
                    else:
                        stats["failed"] += 1
                        failed_assets.append(asset.path.name)
                        if verbose:
                            rprint(f"  [red]❌ Failed[/red]")

                        if not continue_on_error:
                            rprint(
                                f"\n[red]Processing failed for {asset.path.name}. Use --continue-on-error to skip failures.[/red]"
                            )
                            break

                except Exception as e:
                    stats["failed"] += 1
                    failed_assets.append(asset.path.name)

                    if verbose:
                        rprint(f"  [red]❌ Error: {e}[/red]")

                    if not continue_on_error:
                        rprint(
                            f"\n[red]Processing failed for {asset.path.name}: {e}[/red]"
                        )
                        rprint("[red]Use --continue-on-error to skip failures.[/red]")
                        break

        # Final results
        rprint(f"\n[cyan]Processing Complete![/cyan]")

        results_table = Table(title="Processing Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Count", justify="right", style="magenta")

        results_table.add_row(
            "Processed Successfully", f"[green]{stats['processed']}[/green]"
        )
        results_table.add_row("Failed", f"[red]{stats['failed']}[/red]")
        results_table.add_row(
            "Total Size Processed", f"{stats['total_size'] / (1024**3):.2f} GB"
        )

        success_rate = (
            stats["processed"] / len(filtered_assets) * 100 if filtered_assets else 0
        )
        results_table.add_row("Success Rate", f"{success_rate:.1f}%")

        console.print(results_table)

        if failed_assets:
            rprint(f"\n[red]Failed assets:[/red]")
            for failed in failed_assets[:10]:  # Show first 10 failures
                rprint(f"  - {failed}")
            if len(failed_assets) > 10:
                rprint(f"  ... and {len(failed_assets) - 10} more")

        if stats["failed"] > 0 and not continue_on_error:
            sys.exit(1)

    except Exception as e:
        rprint(f"[red]Process-all failed: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@gdb.command("clean-temp")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.pass_context
def clean_temp(ctx, dry_run):
    """Clean temporary zip files from processing"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)
    temp_dir = Path(gdb_config.temp_dir)

    if not temp_dir.exists():
        rprint("[yellow]Temp directory doesn't exist[/yellow]")
        return

    zip_files = [f for f in temp_dir.glob("*.zip")]
    if not zip_files:
        rprint("[green]No temporary files to clean[/green]")
        return

    total_size = sum(f.stat().st_size for f in zip_files)

    if dry_run:
        rprint(
            f"[yellow]Would delete {len(zip_files)} files ({total_size / (1024**2):.1f} MB)[/yellow]"
        )
        for f in zip_files[:5]:  # Show first 5
            rprint(f"  - {f.name}")
        if len(zip_files) > 5:
            rprint(f"  ... and {len(zip_files) - 5} more")
    else:
        if click.confirm(
            f"Delete {len(zip_files)} temporary files ({total_size / (1024**2):.1f} MB)?"
        ):
            for f in zip_files:
                f.unlink()
            rprint(f"[green]Deleted {len(zip_files)} temporary files[/green]")


@gdb.command("validate")
@click.option("--check-s3", is_flag=True, help="Validate S3 uploads exist")
@click.option("--check-integrity", is_flag=True, help="Validate file integrity")
@click.pass_context
def validate(ctx, check_s3, check_integrity):
    """Validate processed assets"""

    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        db = MetadataDB(gdb_config.db_path)

        with duckdb.connect(str(db.db_path)) as conn:
            results = conn.execute("""
                SELECT path, s3_key, uploaded, file_hash 
                FROM gdb_assets 
                WHERE uploaded = true
            """).fetchall()

        if not results:
            rprint("[yellow]No uploaded assets to validate[/yellow]")
            return

        rprint(f"[cyan]Validating {len(results)} uploaded assets...[/cyan]")

        issues = []

        with Progress(console=console) as progress:
            task = progress.add_task("Validating...", total=len(results))

            for path_str, s3_key, uploaded, stored_hash in results:
                progress.advance(task)
                path = Path(path_str)

                # Check if local file still exists
                if not path.exists():
                    issues.append(f"Local file missing: {path.name}")
                    continue

                # Check S3 if requested
                if check_s3:
                    try:
                        s3_uploader = S3Uploader(config.s3_bucket, config.s3_profile)
                        if not s3_uploader.file_exists(s3_key):
                            issues.append(f"S3 file missing: {s3_key}")
                    except Exception as e:
                        issues.append(f"S3 check failed for {s3_key}: {e}")

                # Check integrity if requested
                if check_integrity and stored_hash:
                    try:
                        current_hash = GDBAsset._compute_directory_hash(path)
                        if current_hash != stored_hash:
                            issues.append(f"Hash mismatch: {path.name}")
                    except Exception as e:
                        issues.append(f"Hash check failed for {path.name}: {e}")

        if issues:
            rprint(f"\n[red]Found {len(issues)} validation issues:[/red]")
            for issue in issues[:10]:
                rprint(f"  - {issue}")
            if len(issues) > 10:
                rprint(f"  ... and {len(issues) - 10} more")
        else:
            rprint("[green]✅ All validations passed[/green]")

    except Exception as e:
        rprint(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)


@gdb.command("stats")
@click.option("--by-date", is_flag=True, help="Show statistics by date")
@click.option("--by-type", is_flag=True, help="Show statistics by type")
@click.option("--storage", is_flag=True, help="Show storage statistics")
@click.pass_context
def stats(ctx, by_date, by_type, storage):
    """Show detailed statistics"""

    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        db = MetadataDB(gdb_config.db_path)

        with duckdb.connect(str(db.db_path)) as conn:
            if by_date:
                results = conn.execute("""
                    SELECT DATE_TRUNC('month', timestamp) as month,
                           asset_type,
                           COUNT(*) as count,
                           SUM(file_size) as total_size
                    FROM gdb_assets 
                    GROUP BY month, asset_type
                    ORDER BY month DESC, asset_type
                """).fetchall()

                if results:
                    table = Table(title="Assets by Month")
                    table.add_column("Month", style="cyan")
                    table.add_column("Type", style="green")
                    table.add_column("Count", justify="right", style="magenta")
                    table.add_column("Size", justify="right", style="blue")

                    for month, asset_type, count, size in results:
                        size_str = f"{size / (1024**3):.2f} GB" if size else "N/A"
                        table.add_row(
                            month.strftime("%Y-%m"), asset_type, str(count), size_str
                        )

                    console.print(table)

            if by_type:
                results = conn.execute("""
                    SELECT asset_type,
                           release_candidate,
                           COUNT(*) as count,
                           AVG(file_size) as avg_size,
                           SUM(file_size) as total_size
                    FROM gdb_assets 
                    GROUP BY asset_type, release_candidate
                    ORDER BY asset_type, release_candidate
                """).fetchall()

                if results:
                    table = Table(title="Assets by Type and RC")
                    table.add_column("Type", style="cyan")
                    table.add_column("RC", style="yellow")
                    table.add_column("Count", justify="right", style="magenta")
                    table.add_column("Avg Size", justify="right", style="blue")
                    table.add_column("Total Size", justify="right", style="green")

                    for asset_type, rc, count, avg_size, total_size in results:
                        rc_name = "RC1" if rc == "2016-12-31" else "RC2"
                        avg_str = (
                            f"{avg_size / (1024**2):.1f} MB" if avg_size else "N/A"
                        )
                        total_str = (
                            f"{total_size / (1024**3):.2f} GB" if total_size else "N/A"
                        )

                        table.add_row(
                            asset_type, rc_name, str(count), avg_str, total_str
                        )

                    console.print(table)

            if storage:
                # Storage efficiency stats
                local_results = conn.execute("""
                    SELECT SUM(file_size) as total_local_size,
                           COUNT(*) as total_count
                    FROM gdb_assets
                """).fetchone()

                uploaded_results = conn.execute("""
                    SELECT COUNT(*) as uploaded_count
                    FROM gdb_assets 
                    WHERE uploaded = true
                """).fetchone()

                if local_results and uploaded_results:
                    total_size_gb = (
                        local_results[0] / (1024**3) if local_results[0] else 0
                    )
                    total_count = local_results[1]
                    uploaded_count = uploaded_results[0]

                    table = Table(title="Storage Statistics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", justify="right", style="magenta")

                    table.add_row("Total Local Size", f"{total_size_gb:.2f} GB")
                    table.add_row("Total Assets", str(total_count))
                    table.add_row(
                        "Uploaded to S3",
                        f"{uploaded_count} ({uploaded_count / total_count * 100:.1f}%)",
                    )
                    table.add_row(
                        "Avg Asset Size",
                        f"{total_size_gb * 1024 / total_count:.1f} MB"
                        if total_count > 0
                        else "N/A",
                    )

                    console.print(table)

            # If no specific stats requested, show general overview
            if not any([by_date, by_type, storage]):
                rprint(
                    "[yellow]Use --by-date, --by-type, or --storage to see detailed statistics[/yellow]"
                )
                # Show basic overview
                basic_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN uploaded THEN 1 END) as uploaded,
                        SUM(file_size) / (1024*1024*1024.0) as total_gb,
                        COUNT(DISTINCT asset_type) as types
                    FROM gdb_assets
                """).fetchone()

                if basic_stats:
                    total, uploaded, total_gb, types = basic_stats
                    rprint(f"[green]Total Assets: {total}[/green]")
                    rprint(
                        f"[green]Uploaded: {uploaded} ({uploaded / total * 100:.1f}%)[/green]"
                    )
                    rprint(f"[green]Total Size: {total_gb:.2f} GB[/green]")
                    rprint(f"[green]Asset Types: {types}[/green]")

    except Exception as e:
        rprint(f"[red]Stats failed: {e}[/red]")
        sys.exit(1)


@gdb.command("latest-by-rc")
@click.option(
    "--type",
    "asset_type",
    type=click.Choice([t.value for t in AssetType]),
    help="Filter by asset type (e.g., verification_topology)",
)
@click.option(
    "--days-back",
    type=int,
    default=30,
    help="Only consider assets from the last N days",
)
@click.option(
    "--show-couple",
    is_flag=True,
    help="Also show if they form a release couple (created close together)",
)
@click.pass_context
def latest_by_rc(ctx, asset_type, days_back, show_couple):
    """Show the latest asset for each RC (RC1/RC2)"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Create manager instance (reusing existing logic)
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,
        )

        # Get latest assets
        latest_assets = manager.get_latest_assets_by_rc(
            asset_type=asset_type, days_back=days_back
        )

        if not latest_assets:
            asset_filter = f" for {asset_type}" if asset_type else ""
            rprint(
                f"[yellow]No assets found{asset_filter} in the last {days_back} days[/yellow]"
            )
            return

        # Display results
        table = Table(
            title=f"Latest Assets by RC{' - ' + asset_type if asset_type else ''}"
        )
        table.add_column("RC", style="cyan", width=8)
        table.add_column("Date", style="magenta", width=20)
        table.add_column("Type", style="green", width=25)
        table.add_column("File", style="yellow", max_width=40)
        table.add_column("Size", justify="right", style="blue", width=12)
        table.add_column("Status", style="red", width=8)

        for rc_name in ["RC1", "RC2"]:
            if rc_name in latest_assets:
                data = latest_assets[rc_name]
                table.add_row(
                    rc_name,
                    data["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    data["asset_type"],
                    Path(data["path"]).name,
                    f"{data['file_size'] / 1024 / 1024:.1f} MB"
                    if data["file_size"]
                    else "N/A",
                    "✅" if data["uploaded"] else "❌",
                )
            else:
                table.add_row(rc_name, "Not found", "-", "-", "-", "-")

        console.print(table)

        # Show release couple information
        if show_couple and len(latest_assets) == 2:
            couple = manager.get_latest_release_couple(asset_type=asset_type)
            if couple:
                rc1_date, rc2_date = couple
                days_apart = abs((rc1_date - rc2_date).days)
                hours_apart = abs((rc1_date - rc2_date).total_seconds() / 3600)

                if days_apart == 0:
                    time_diff = f"{hours_apart:.1f} hours apart"
                else:
                    time_diff = f"{days_apart} days apart"

                rprint(f"\n[green]✅ Release Couple Found:[/green]")
                rprint(f"  RC1: {rc1_date.strftime('%Y-%m-%d %H:%M')}")
                rprint(f"  RC2: {rc2_date.strftime('%Y-%m-%d %H:%M')}")
                rprint(f"  Time difference: {time_diff}")
            else:
                rprint(
                    f"\n[yellow]⚠️  Latest RC1 and RC2 are not close enough to form a release couple[/yellow]"
                )

        # Summary for script usage
        if latest_assets:
            rprint(f"\n[dim]Latest dates:[/dim]")
            for rc_name, data in latest_assets.items():
                rprint(
                    f"[dim]  {rc_name}: {data['timestamp'].strftime('%Y-%m-%d')}[/dim]"
                )

    except Exception as e:
        rprint(f"[red]Command failed: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@gdb.command("latest-topology")
@click.option(
    "--max-days-apart",
    type=int,
    default=7,
    help="Maximum days between RC1 and RC2 to consider them a couple",
)
@click.pass_context
def latest_topology(ctx, max_days_apart):
    """Show the latest topology verification tests for each RC"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Create manager instance
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,
        )

        # Get latest topology verification for each RC
        latest_assets = manager.get_latest_assets_by_rc(
            asset_type="verification_topology"
        )

        if not latest_assets:
            rprint("[yellow]No topology verification tests found[/yellow]")
            return

        # Display results
        table = Table(title="Latest Topology Verification Tests")
        table.add_column("RC", style="cyan", width=8)
        table.add_column("Test Date", style="magenta", width=20)
        table.add_column("File", style="yellow", max_width=40)
        table.add_column("Size", justify="right", style="blue", width=12)
        table.add_column("Status", style="red", width=8)

        dates_found = []
        for rc_name in ["RC1", "RC2"]:
            if rc_name in latest_assets:
                data = latest_assets[rc_name]
                dates_found.append(data["timestamp"])
                table.add_row(
                    rc_name,
                    data["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    Path(data["path"]).name,
                    f"{data['file_size'] / 1024 / 1024:.1f} MB"
                    if data["file_size"]
                    else "N/A",
                    "✅" if data["uploaded"] else "❌",
                )
            else:
                table.add_row(rc_name, "Not found", "-", "-", "-")

        console.print(table)

        # Check if they form a release couple
        if len(dates_found) == 2:
            days_diff = abs((dates_found[0] - dates_found[1]).days)
            hours_diff = abs((dates_found[0] - dates_found[1]).total_seconds() / 3600)

            if days_diff <= max_days_apart:
                if days_diff == 0:
                    time_desc = f"{hours_diff:.1f} hours apart"
                else:
                    time_desc = f"{days_diff} days apart"

                rprint(f"\n[green]✅ Latest Release Couple:[/green] {time_desc}")

                # Show the dates in the format the user requested
                latest_dates = sorted(
                    [d.strftime("%Y-%m-%d") for d in dates_found], reverse=True
                )
                rprint(
                    f"[green]Latest tests: {latest_dates[0]} and {latest_dates[1]}[/green]"
                )
            else:
                rprint(
                    f"\n[yellow]⚠️  RC1 and RC2 are {days_diff} days apart (max allowed: {max_days_apart})[/yellow]"
                )
                rprint("[yellow]They don't form a release couple[/yellow]")

        # Answer the user's specific question
        if len(latest_assets) >= 1:
            rprint(
                f"\n[cyan]Answer: The latest topology verification tests are:[/cyan]"
            )
            for rc_name in ["RC1", "RC2"]:
                if rc_name in latest_assets:
                    date_str = latest_assets[rc_name]["timestamp"].strftime("%Y-%m-%d")
                    rprint(f"  {rc_name}: [bold]{date_str}[/bold]")

    except Exception as e:
        rprint(f"[red]Command failed: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@gdb.command("latest-verifications")
@click.pass_context
def latest_verifications(ctx):
    """Show latest verification runs for all verification types"""
    gdb_config, global_config, environment, verbose = get_configs(ctx)

    try:
        # Create manager instance
        s3_bucket = gdb_config.get_s3_bucket(global_config)
        s3_profile = gdb_config.get_s3_profile(global_config)

        manager = GDBAssetManager(
            base_paths=gdb_config.base_paths,
            s3_bucket=s3_bucket,
            db_path=gdb_config.db_path,
            temp_dir=gdb_config.temp_dir,
            aws_profile=s3_profile,
        )

        # Get all latest verification runs
        verification_runs = manager.get_latest_verification_runs()

        if not verification_runs:
            rprint("[yellow]No verification runs found[/yellow]")
            return

        # Display results grouped by verification type
        for verification_type, runs in verification_runs.items():
            # Clean up the verification type name for display
            display_name = (
                verification_type.replace("verification_", "").replace("_", " ").title()
            )

            table = Table(title=f"Latest {display_name} Verification")
            table.add_column("RC", style="cyan", width=8)
            table.add_column("Date", style="magenta", width=20)
            table.add_column("File", style="yellow", max_width=40)
            table.add_column("Size", justify="right", style="blue", width=12)
            table.add_column("Status", style="red", width=8)

            # Group by RC for display
            runs_by_rc = {run["rc_name"]: run for run in runs}

            dates_for_couple = []
            for rc_name in ["RC1", "RC2"]:
                if rc_name in runs_by_rc:
                    run = runs_by_rc[rc_name]
                    dates_for_couple.append(run["timestamp"])
                    table.add_row(
                        rc_name,
                        run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        Path(run["path"]).name,
                        f"{run['file_size'] / 1024 / 1024:.1f} MB"
                        if run["file_size"]
                        else "N/A",
                        "✅" if run["uploaded"] else "❌",
                    )
                else:
                    table.add_row(rc_name, "Not found", "-", "-", "-")

            console.print(table)

            # Show couple info for this verification type
            if len(dates_for_couple) == 2:
                days_diff = abs((dates_for_couple[0] - dates_for_couple[1]).days)
                if days_diff <= 7:  # Same max_days_apart logic
                    latest_dates = sorted(
                        [d.strftime("%Y-%m-%d") for d in dates_for_couple], reverse=True
                    )
                    rprint(
                        f"[dim]  → Release couple: {latest_dates[0]} and {latest_dates[1]}[/dim]"
                    )

            rprint()  # Empty line between verification types

    except Exception as e:
        rprint(f"[red]Command failed: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


# Also add this utility function that can be used in scripts
def get_latest_topology_dates(db_path: str) -> Optional[Tuple[str, str]]:
    """
    Utility function to get latest topology verification dates.

    Args:
        db_path: Path to the DuckDB database

    Returns:
        Tuple of (RC1_date, RC2_date) as strings in YYYY-MM-DD format, or None

    Example:
        >>> dates = get_latest_topology_dates("gdb_metadata.duckdb")
        >>> if dates:
        >>>     print(f"Latest RC1: {dates[0]}, Latest RC2: {dates[1]}")
    """
    try:
        with duckdb.connect(db_path) as conn:
            query = """
            WITH ranked_assets AS (
                SELECT *,
                       CASE 
                           WHEN release_candidate = '2016-12-31' THEN 'RC1'
                           WHEN release_candidate = '2030-12-31' THEN 'RC2'
                           ELSE 'Unknown'
                       END as rc_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY release_candidate 
                           ORDER BY timestamp DESC
                       ) as rn
                FROM gdb_assets 
                WHERE asset_type = 'verification_topology'
            )
            SELECT rc_name, timestamp::DATE as date_only
            FROM ranked_assets 
            WHERE rn = 1 AND rc_name IN ('RC1', 'RC2')
            ORDER BY rc_name
            """

            results = conn.execute(query).fetchall()

            if len(results) == 2:
                return (str(results[0][1]), str(results[1][1]))  # RC1, RC2 dates
            else:
                return None

    except Exception as e:
        logger.error(f"Error getting latest topology dates: {e}")
        return None


if __name__ == "__main__":
    gdb()
