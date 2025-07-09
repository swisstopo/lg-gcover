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

from gcover.gdb.manager import (GDBAssetManager,  S3Uploader, MetadataDB)
from gcover.gdb.assets import (GDBAsset, BackupGDBAsset, VerificationGDBAsset, IncrementGDBAsset, AssetType, ReleaseCandidate)


console = Console()




@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def gdb(ctx, config, verbose):
    """GDB Asset Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    # Default configuration - should be loaded from config file in real implementation
    ctx.obj['base_paths'] = {
        'backup': Path("/media/marco/SANDISK/GCOVER"),
        'verification': Path("/media/marco/SANDISK/Verifications"),
        'increment': Path("/media/marco/SANDISK/Increment")
    }
    ctx.obj['s3_bucket'] = "your-gdb-bucket"
    ctx.obj['db_path'] = "gdb_metadata.duckdb"
    ctx.obj['temp_dir'] = "/tmp/gdb_zips"


@gdb.command()
@click.pass_context
def scan(ctx):
    """Scan filesystem for GDB assets"""
    manager = GDBAssetManager(
        base_paths=ctx.obj['base_paths'],
        s3_bucket=ctx.obj['s3_bucket'],
        db_path=ctx.obj['db_path'],
        temp_dir=ctx.obj['temp_dir']
    )
    
    with Progress() as progress:
        task = progress.add_task("Scanning filesystem...", total=None)
        assets = manager.scan_filesystem()
        progress.update(task, completed=100)
    
    rprint(f"[green]Found {len(assets)} GDB assets[/green]")
    
    # Group by type
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
            latest.info.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    console.print(table)


@gdb.command()
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
@click.pass_context
def sync(ctx, dry_run):
    """Sync all GDB assets to S3 and database"""
    manager = GDBAssetManager(
        base_paths=ctx.obj['base_paths'],
        s3_bucket=ctx.obj['s3_bucket'],
        db_path=ctx.obj['db_path'],
        temp_dir=ctx.obj['temp_dir']
    )
    
    if dry_run:
        rprint("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
        assets = manager.scan_filesystem()
        
        new_assets = []
        for asset in assets:
            if not manager.metadata_db.asset_exists(asset.path):
                new_assets.append(asset)
        
        if new_assets:
            table = Table(title="Assets to be synced")
            table.add_column("Path", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("RC", style="yellow")
            table.add_column("Date", style="magenta")
            
            for asset in new_assets:
                table.add_row(
                    str(asset.path.relative_to(asset.path.parents[3])),
                    asset.info.asset_type.value,
                    asset.info.release_candidate.short_name,
                    asset.info.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            console.print(table)
            rprint(f"[green]Would sync {len(new_assets)} new assets[/green]")
        else:
            rprint("[green]No new assets to sync[/green]")
    else:
        stats = manager.sync_all()
        
        table = Table(title="Sync Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        
        for key, value in stats.items():
            table.add_row(key.title(), str(value))
        
        console.print(table)


@gdb.command()
@click.option('--type', 'asset_type', type=click.Choice([t.value for t in AssetType]), help='Filter by asset type')
@click.option('--rc', type=click.Choice(['RC1', 'RC2']), help='Filter by release candidate')
@click.option('--since', type=str, help='Show assets since date (YYYY-MM-DD)')
@click.option('--limit', type=int, default=20, help='Limit number of results')
@click.pass_context
def list(ctx, asset_type, rc, since, limit):
    """List GDB assets from database"""
    db = MetadataDB(ctx.obj['db_path'])
    
    query = "SELECT * FROM gdb_assets WHERE 1=1"
    params = []
    
    if asset_type:
        query += " AND asset_type = ?"
        params.append(asset_type)
    
    if rc:
        rc_value = ReleaseCandidate.RC1.value if rc == 'RC1' else ReleaseCandidate.RC2.value
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
            Path(data['path']).name,
            data['asset_type'],
            'RC1' if data['release_candidate'] == ReleaseCandidate.RC1.value else 'RC2',
            data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            f"{data['file_size'] / 1024 / 1024:.1f} MB" if data['file_size'] else "N/A",
            "✅" if data['uploaded'] else "❌"
        )
    
    console.print(table)


@gdb.command()
@click.argument('search_term')
@click.option('--download', is_flag=True, help='Download the asset')
@click.option('--output-dir', type=click.Path(), default="./downloads", help='Download directory')
@click.pass_context
def search(ctx, search_term, download, output_dir):
    """Search for GDB assets"""
    db = MetadataDB(ctx.obj['db_path'])
    
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
        rprint(f"   RC: {'RC1' if data['release_candidate'] == ReleaseCandidate.RC1.value else 'RC2'}")
        rprint(f"   Date: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        rprint(f"   Size: {data['file_size'] / 1024 / 1024:.1f} MB" if data['file_size'] else "N/A")
        rprint(f"   S3 Key: {data['s3_key']}")
        
        if download and data['s3_key'] and data['uploaded']:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            s3_uploader = S3Uploader(ctx.obj['s3_bucket'])
            filename = Path(data['s3_key']).name
            local_path = output_path / filename
            
            try:
                s3_uploader.s3_client.download_file(
                    ctx.obj['s3_bucket'], 
                    data['s3_key'], 
                    str(local_path)
                )
                rprint(f"   [green]Downloaded to: {local_path}[/green]")
            except Exception as e:
                rprint(f"   [red]Download failed: {e}[/red]")


@gdb.command()
@click.pass_context
def status(ctx):
    """Show system status and statistics"""
    db = MetadataDB(ctx.obj['db_path'])
    
    with duckdb.connect(str(db.db_path)) as conn:
        # Overall stats
        total_assets = conn.execute("SELECT COUNT(*) FROM gdb_assets").fetchone()[0]
        uploaded_assets = conn.execute("SELECT COUNT(*) FROM gdb_assets WHERE uploaded = true").fetchone()[0]
        total_size = conn.execute("SELECT SUM(file_size) FROM gdb_assets WHERE file_size IS NOT NULL").fetchone()[0]
        
        # By type
        by_type = conn.execute("""
            SELECT asset_type, COUNT(*), SUM(file_size) 
            FROM gdb_assets 
            GROUP BY asset_type
        """).fetchall()
        
        # By RC
        by_rc = conn.execute("""
            SELECT release_candidate, COUNT(*), SUM(file_size) 
            FROM gdb_assets 
            GROUP BY release_candidate
        """).fetchall()
        
        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent = conn.execute("""
            SELECT COUNT(*) FROM gdb_assets 
            WHERE created_at >= ?
        """, [week_ago]).fetchone()[0]
    
    # Overall status
    table = Table(title="System Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    
    table.add_row("Total Assets", str(total_assets))
    table.add_row("Uploaded to S3", f"{uploaded_assets} ({uploaded_assets/total_assets*100:.1f}%)" if total_assets > 0 else "0")
    table.add_row("Total Size", f"{total_size/1024/1024/1024:.2f} GB" if total_size else "N/A")
    table.add_row("New (Last 7 days)", str(recent))
    
    console.print(table)
    
    # By type
    if by_type:
        table = Table(title="Assets by Type")
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Size", justify="right", style="blue")
        
        for asset_type, count, size in by_type:
            table.add_row(
                asset_type,
                str(count),
                f"{size/1024/1024/1024:.2f} GB" if size else "N/A"
            )
        
        console.print(table)
    
    # By RC
    if by_rc:
        table = Table(title="Assets by Release Candidate")
        table.add_column("RC", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Size", justify="right", style="blue")
        
        for rc, count, size in by_rc:
            rc_name = 'RC1' if rc == ReleaseCandidate.RC1.value else 'RC2'
            table.add_row(
                rc_name,
                str(count),
                f"{size/1024/1024/1024:.2f} GB" if size else "N/A"
            )
        
        console.print(table)


@gdb.command()
@click.argument('gdb_path', type=click.Path(exists=True))
@click.pass_context
def process(ctx, gdb_path):
    """Process a single GDB asset"""
    manager = GDBAssetManager(
        base_paths=ctx.obj['base_paths'],
        s3_bucket=ctx.obj['s3_bucket'],
        db_path=ctx.obj['db_path'],
        temp_dir=ctx.obj['temp_dir']
    )
    
    try:
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


@gdb.command()
@click.pass_context
def init(ctx):
    """Initialize the system (create database, check S3 connection)"""
    rprint("[cyan]Initializing GDB Asset Management System...[/cyan]")
    
    # Initialize database
    try:
        db = MetadataDB(ctx.obj['db_path'])
        rprint("[green]✅ Database initialized[/green]")
    except Exception as e:
        rprint(f"[red]❌ Database initialization failed: {e}[/red]")
        sys.exit(1)
    
    # Test S3 connection
    try:
        s3_uploader = S3Uploader(ctx.obj['s3_bucket'])
        # Try to list bucket to test connection
        s3_uploader.s3_client.head_bucket(Bucket=ctx.obj['s3_bucket'])
        rprint("[green]✅ S3 connection successful[/green]")
    except Exception as e:
        rprint(f"[red]❌ S3 connection failed: {e}[/red]")
        rprint("[yellow]Make sure your AWS credentials are configured[/yellow]")
    
    # Check base paths
    for name, path in ctx.obj['base_paths'].items():
        if path.exists():
            rprint(f"[green]✅ {name.title()} path exists: {path}[/green]")
        else:
            rprint(f"[yellow]⚠️  {name.title()} path not found: {path}[/yellow]")
    
    rprint("[green]Initialization complete![/green]")


if __name__ == '__main__':
    cli()
