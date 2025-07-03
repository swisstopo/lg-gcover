#!/usr/bin/env python
"""
Main CLI entry point for gcover.
"""

import sys
from pathlib import Path

import click

# Ajouter le dossier parent au path si nécessaire (pour le développement)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from gcover import __version__
    from gcover.utils.imports import HAS_ARCPY
except ImportError:
    __version__ = "unknown"
    HAS_ARCPY = False


@click.group()
@click.version_option(version=__version__, prog_name="gcover")
@click.pass_context
def cli(ctx):
    """
    gcover - Geological vector data management tool.

    A comprehensive toolkit for working with geological vector data,
    providing bridge functionality between GeoPandas and ESRI formats,
    schema management, quality assurance, and geodatabase management.
    """
    # Contexte global si nécessaire
    ctx.ensure_object(dict)
    ctx.obj["has_arcpy"] = HAS_ARCPY


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
        from gcover import geometry

        modules.append("✓ geometry (GDB cleanup)")
    except ImportError:
        modules.append("✗ geometry (not available)")

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
    from .qa_cmd import qa

    cli.add_command(qa)
except ImportError:
    pass

try:
    from .manage_cmd import manage

    cli.add_command(manage)
except ImportError:
    pass

try:
    from .geometry_cmd import geometry

    cli.add_command(geometry)
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
