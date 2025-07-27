#!/usr/bin/env python3
"""
CLI for GDB Asset Management System
"""

import click
import sys
from pathlib import Path
from typing import Optional, List
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


console = Console()



def get_configs(ctx) -> tuple[GDBConfig, GlobalConfig, str, bool]:
    app_config: AppConfig =  load_config()   #ctx.obj["app_config"]
    return (
        app_config.gdb,
        app_config.global_,
        ctx.obj["environment"],
        ctx.obj.get("verbose", False)
    )



@click.group()
@click.pass_context
def gdb(ctx):
    """GDB Asset Management commands"""
    pass


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
                rprint(f"[dim]Workers: {gdb_config.max_workers}[/dim]")

            stats = manager.sync_all()

            table = Table(title="Sync Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="magenta")

            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))

            console.print(table)

            if stats.get("failed", 0) > 0:
                rprint("[yellow]Some assets failed to sync. Check logs for details.[/yellow]")
            else:
                rprint("[green]All assets synced successfully![/green]")

    except Exception as e:
        rprint(f"[red]Sync failed: {e}[/red]")
        sys.exit(1)

# TODO: use central config
@gdb.command()
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
def list(ctx, asset_type, rc, since, limit):
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
    """Search for GDB assets"""


    gdb_config, global_config, environment, verbose = get_gdb_config(ctx)

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
    gdb_config, global_config, environment, verbose = get_gdb_config(ctx)

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
            rprint(f"[dim]  Max Workers: {gdb_config.max_workers}[/dim]")
            rprint(f"[dim]  Compression: {gdb_config.compression_level}[/dim]")
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
    gdb_config, global_config, environment, verbose = get_gdb_config(ctx)

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


if __name__ == "__main__":
    gdb()
