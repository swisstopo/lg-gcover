#!/usr/bin/env python
"""
Main CLI entry point for gcover.
"""

import sys
from pathlib import Path
from rich import print as rprint

import click

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

env_map = {
    "prod": "production",
    "production": "production",
    "dev": "development",
    "development": "development",
    "sandisk": "sandisk",
    "integration": "integration",
    "int": "integration",
}


@click.group()
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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, config, env, verbose):
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
        ctx.obj["environment"] = environment
        ctx.obj["verbose"] = verbose

        if verbose:
            global_config = app_config.global_
            rprint(f"[cyan]Environment: {environment}[/cyan]")
            rprint(f"[cyan]Log Level: {global_config.log_level}[/cyan]")
            rprint(f"[cyan]Bucket name: {global_config.s3.bucket}[/cyan]")
            rprint(f"[cyan]Temp Dir: {global_config.temp_dir}[/cyan]")
            rprint(f"[cyan]Has arcpy: {HAS_ARCPY}[/cyan]")

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


# Import des sous-commandes si disponibles
try:
    from .bridge_cmd import bridge

    cli.add_command(bridge)
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
