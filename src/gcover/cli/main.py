#!/usr/bin/env python
"""
Main CLI entry point for gcover.
"""

import sys
from pathlib import Path
from rich import print as rprint

import click
from loguru import logger

# Ajouter le dossier parent au path si nécessaire (pour le développement)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from gcover import __version__
    from gcover.utils.imports import HAS_ARCPY
except ImportError:
    __version__ = "unknown"
    HAS_ARCPY = False

# from ..config import load_config

from gcover.config import load_config, AppConfig
from gcover.utils.logging import setup_logging, gcover_logger

env_map = {
    "prod": "production",
    "production": "production",
    "dev": "development",
    "development": "development",
    "sandisk": "sandisk",
    "integration": "integration",
    "int": "integration",
}


def confirm_extended(prompt: str, default=True):
    response = click.prompt(prompt + " [y/n]", default="y" if default else "n")
    response = response.strip().lower()
    return response in {"y", "yes", "o", "oui"}


@click.group(context_settings={"show_default": True})
@click.version_option(version=__version__, prog_name="gcover")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option(
    "--env",
    "-e",
    type=click.Choice(env_map.keys()),
    default="development",
    help="Environment (dev/prod)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output and debug logging"
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Custom log file path (default: auto-generated)",
)
@click.option("--log-info", is_flag=True, help="Show logging configuration and exit")
@click.pass_context
def cli(ctx, config, log_file, log_info, env, verbose):
    """gcover - Swiss GeoCover data processing toolkit"""
    ctx.ensure_object(dict)
    ctx.obj["has_arcpy"] = HAS_ARCPY

    # Normalize environment name
    try:
        environment = env_map[env.lower()]
    except KeyError:
        raise click.BadParameter(f"Unsupported environment: {env}")

    try:
        # Load centralized configuration
        app_config: AppConfig = load_config(environment=environment)

        # ctx.obj["config_manager"] = config_manager
        ctx.obj["config_path"] = config
        ctx.obj["environment"] = environment
        ctx.obj["verbose"] = verbose

        global_config = app_config.global_

        # print(global_config.logging)

        if verbose:
            rprint(f"[cyan]Environment: {environment}[/cyan]")
            rprint(f"[cyan]Log Level: {global_config.log_level}[/cyan]")
            rprint(f"[cyan]Bucket name: {global_config.s3.bucket}[/cyan]")
            rprint(f"[cyan]Proxy: {global_config.proxy}[/cyan]")
            rprint(f"[cyan]Temp Dir: {global_config.temp_dir}[/cyan]")
            rprint(f"[cyan]Has arcpy: {HAS_ARCPY}[/cyan]")

        # Setup centralized logging FIRST (before any other operations)
        setup_logging(verbose=verbose, log_file=log_file, environment=env)

        # Show logging info and exit if requested
        if log_info:
            gcover_logger.show_log_info()
            ctx.exit()

        # Log the startup
        logger.info(f"GCover CLI started (environment: {env})")
        logger.debug(f"Configuration: config={config}, verbose={verbose}")

    except Exception as e:
        rprint(f"[red]Configuration error: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Display information about the gcover installation."""
    click.echo(f"gcover version: {__version__}")
    click.echo(f"ArcPy available: {'Yes' if HAS_ARCPY else 'No'}")
    click.echo(f"Python version: {sys.version.split()[0]}")

    # Lister les modules disponibles
    click.echo("\nAvailable modules:")
    modules = []
    try:
        from gcover import bridge

        modules.append("✓ bridge (GeoPandas <-> ESRI)")
    except ImportError:
        modules.append("✗ bridge (not available)")

    try:
        from gcover import schema

        modules.append("✓ schema (Schema management)")
    except ImportError:
        modules.append("✗ schema (not available)")

    try:
        from gcover import qa

        modules.append("✓ qa (Quality assurance)")
    except ImportError:
        modules.append("✗ qa (not available)")

    try:
        from gcover import manage

        modules.append("✓ manage (GDB management)")
    except ImportError:
        modules.append("✗ manage (not available)")

    try:
        from gcover import gdb

        modules.append("✓ gdb (GDB management)")
    except ImportError:
        modules.append("✗ gdb (not available)")

    try:
        from gcover import sde

        modules.append("✓ sde (SDE management)")
    except ImportError:
        modules.append("✗ sde (not available)")

    for module in modules:
        click.echo(f"  {module}")


# Additional logging commands
@cli.group()
def logs():
    """Logging and diagnostics commands."""
    pass


@logs.command("show")
def show_logs():
    """Show current logging configuration."""
    gcover_logger.show_log_info()


@logs.command("debug")
@click.pass_context
def enable_debug(ctx):
    """Enable debug logging dynamically."""
    gcover_logger.enable_debug_mode()
    logger.debug("Debug logging enabled dynamically")
    click.echo("✅ Debug logging enabled")


@logs.command("tail")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
def tail_logs(lines):
    """Show recent log entries."""
    log_file = gcover_logger.get_log_file_path()

    if not log_file or not log_file.exists():
        click.echo("❌ No log file found")
        return

    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:]

        click.echo(f"📄 Last {len(recent_lines)} lines from {log_file}:")
        click.echo("─" * 60)
        for line in recent_lines:
            click.echo(line.rstrip())

    except Exception as e:
        click.echo(f"❌ Error reading log file: {e}")


# Import des sous-commandes si disponibles
try:
    from .bridge_cmd import bridge_commands

    cli.add_command(bridge_commands)
except ImportError:
    pass

try:
    from .schema_cmd import schema

    cli.add_command(schema)
except ImportError:
    pass

try:
    from .gdb_cmd import gdb

    cli.add_command(gdb)
except ImportError:
    pass

try:
    from .qa_cmd import qa_commands

    cli.add_command(qa_commands)
except ImportError:
    pass

try:
    from .manage_cmd import manage

    cli.add_command(manage)
except ImportError:
    pass

try:
    from .sde_cmd import sde_commands

    cli.add_command(sde_commands)
except ImportError:
    pass


def main() -> None:
    """Point d'entrée principal."""
    try:
        cli()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
