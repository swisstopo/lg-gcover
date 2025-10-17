# gcover/cli/sde_cmd.py
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from tabulate import tabulate
import geopandas as gpd
import pandas as pd


from gcover.config import SDE_INSTANCES, AppConfig, load_config  # TODO
from gcover.sde import SDEConnectionManager, create_bridge

from gcover.arcpy_compat import HAS_ARCPY, arcpy

from gcover.sde.feature_class_explorer import (
    list_all_feature_classes,
    display_feature_class_tree,
    display_feature_class_table,
    get_feature_class_info,
    search_feature_classes,
    display_search_results,
)


console = Console()


@click.group(name="sde")
def sde_commands():
    """Commandes de gestion des connexions SDE"""
    pass


@sde_commands.command("versions")
@click.option(
    "--instance",
    "-i",
    type=click.Choice(list(SDE_INSTANCES.values())),
    multiple=True,
    help="Instances à vérifier",
)
@click.option(
    "--user-only",
    "-u",
    is_flag=True,
    help="Afficher seulement les versions utilisateur",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Format de sortie",
)
def list_versions(instance, user_only, format):
    """Liste les versions disponibles sur les instances SDE"""

    instances = instance or list(SDE_INSTANCES.values())

    with SDEConnectionManager() as conn_mgr:
        current_user = os.getlogin().upper()
        all_versions = []

        for inst in instances:
            click.echo(f"🔍 Vérification instance: {inst}")

            try:
                versions = conn_mgr.get_versions(inst)

                if user_only:
                    versions = [
                        v for v in versions if current_user in v["name"].upper()
                    ]

                # Ajouter l'instance à chaque version pour l'export
                for v in versions:
                    v["instance"] = inst
                    all_versions.append(v)

                if format == "table":
                    _display_versions_table(inst, versions, current_user)

            except Exception as e:
                click.echo(f"❌ Erreur pour {inst}: {e}", err=True)

        # Export autres formats
        if format == "json":
            import json

            click.echo(json.dumps(all_versions, indent=2))
        elif format == "csv":
            _export_versions_csv(all_versions)


def _display_versions_table(instance: str, versions: List[dict], current_user: str):
    """Affiche les versions sous forme de tableau"""
    if not versions:
        click.echo(f"  ℹ️  Aucune version trouvée pour {instance}")
        return

    click.echo(f"\n📊 {instance}")
    click.echo("=" * 60)

    table_data = []
    for v in versions:
        status = []
        if v["isOwner"]:
            status.append("👤 Owner")
        if v["writable"]:
            status.append("✏️ Writable")
        if current_user in v["name"].upper():
            status.append("⭐ User")

        table_data.append([v["name"], v["parent"] or "-", " ".join(status) or "-"])

    headers = ["Version", "Parent", "Status"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


def _export_versions_csv(versions: List[dict]):
    """Export CSV des versions"""
    import csv
    import io

    output = io.StringIO()
    if versions:
        fieldnames = ["instance", "name", "parent", "isOwner", "writable"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for v in versions:
            writer.writerow(
                {
                    "instance": v["instance"],
                    "name": v["name"],
                    "parent": v["parent"] or "",
                    "isOwner": v["isOwner"],
                    "writable": v["writable"],
                }
            )

    click.echo(output.getvalue())


@sde_commands.command("connections")
@click.option(
    "--cleanup", is_flag=True, help="Nettoyer les connexions actives après affichage"
)
def list_connections(cleanup):
    """Liste et optionnellement nettoie les connexions SDE actives"""

    with SDEConnectionManager() as conn_mgr:
        connections = conn_mgr.list_active_connections()

        if not connections:
            click.echo("ℹ️  Aucune connexion SDE active")
            return

        click.echo(f"🔗 {len(connections)} connexion(s) active(s):")

        table_data = [
            [conn["instance"], conn["version"], conn["path"]] for conn in connections
        ]

        headers = ["Instance", "Version", "Chemin SDE"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

        if cleanup:
            if click.confirm("🗑️  Nettoyer toutes les connexions ?"):
                conn_mgr.cleanup_all()
                click.echo("✅ Connexions nettoyées")


@sde_commands.command("connect")
@click.option(
    "--instance",
    "-i",
    type=click.Choice(list(SDE_INSTANCES.values())),
    prompt=True,
    help="Instance SDE",
)
@click.option(
    "--interactive", is_flag=True, help="Mode interactif pour sélection de version"
)
def quick_connect(instance, interactive):
    """Test rapide de connexion à une instance SDE"""

    with SDEConnectionManager() as conn_mgr:
        try:
            if interactive:
                versions = conn_mgr.get_versions(instance)
                if not versions:
                    click.echo(f"❌ Aucune version disponible pour {instance}")
                    return

                click.echo(f"\n📋 Versions disponibles pour {instance}:")
                for i, v in enumerate(versions, 1):
                    status = " (Writable)" if v["writable"] else " (Read-only)"
                    click.echo(f"  {i}. {v['name']}{status}")

                choice = click.prompt("Choisir une version", type=int)
                if 1 <= choice <= len(versions):
                    selected = versions[choice - 1]
                    version = selected["name"]
                else:
                    click.echo("❌ Choix invalide")
                    return
            else:
                version = "SDE.DEFAULT"

            # Test de connexion
            sde_path = conn_mgr.create_connection(instance, version)
            click.echo("✅ Connexion réussie:")
            click.echo(f"   Instance: {instance}")
            click.echo(f"   Version: {version}")
            click.echo(f"   Fichier SDE: {sde_path}")

            # Test d'accès aux feature classes
            with click.progressbar(label="Test d'accès aux données") as bar:
                try:
                    import arcpy

                    arcpy.env.workspace = str(sde_path)
                    datasets = arcpy.ListDatasets()
                    bar.update(1)
                    click.echo(f"   📁 {len(datasets)} dataset(s) trouvé(s)")
                except Exception as e:
                    click.echo(f"   ⚠️  Erreur accès données: {e}")

        except Exception as e:
            click.echo(f"❌ Erreur de connexion: {e}", err=True)


@sde_commands.command("user-versions")
@click.option(
    "--instance",
    "-i",
    type=click.Choice(list(SDE_INSTANCES.values())),
    multiple=True,
    help="Instances à vérifier (toutes par défaut)",
)
def find_user_versions(instance):
    """Trouve automatiquement les versions utilisateur"""

    instances = instance or list(SDE_INSTANCES.values())
    current_user = os.getlogin().upper()

    click.echo(f"👤 Recherche versions pour utilisateur: {current_user}")

    with SDEConnectionManager() as conn_mgr:
        user_versions = {}
        for inst in instances:
            user_versions[inst] = []
            try:
                versions = conn_mgr.get_versions(inst)
                for v in versions:
                    if current_user in v["name"].upper() or v["isOwner"]:
                        user_versions[inst].append(v)
            except Exception as e:
                click.echo(f"❌ Erreur pour {inst}: {e}")

        # Affichage résultats
        found_any = False
        for inst, versions in user_versions.items():
            if versions:
                found_any = True
                click.echo(f"\n📁 {inst}:")
                for v in versions:
                    status = "✏️ Writable" if v["writable"] else "👁️ Read-only"
                    click.echo(f"  • {v['name']} ({status})")

        if not found_any:
            click.echo("ℹ️  Aucune version utilisateur trouvée")


# =============================================================================
# CONNECTION MANAGEMENT COMMANDS (existing, enhanced)
# =============================================================================


@sde_commands.command("versions")
@click.option(
    "--instance",
    "-i",
    type=click.Choice(
        list(SDE_INSTANCES.values())
        if "SDE_INSTANCES" in globals()
        else ["GCOVERP", "GCOVERI"]
    ),
    multiple=True,
    help="Instances to check",
)
@click.option("--user-only", "-u", is_flag=True, help="Show only user versions")
@click.option("--writable-only", "-w", is_flag=True, help="Show only writable versions")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def list_versions(instance, user_only, writable_only, format):
    """List available versions on SDE instances"""
    instances = instance or ["GCOVERP", "GCOVERI"]

    with SDEConnectionManager() as conn_mgr:
        current_user = os.getlogin().upper()
        all_versions = []

        for inst in instances:
            click.echo(f"🔍 Checking instance: {inst}")

            try:
                versions = conn_mgr.get_versions(inst)

                # Apply filters
                if user_only:
                    versions = [
                        v for v in versions if current_user in v["name"].upper()
                    ]
                if writable_only:
                    versions = [v for v in versions if v.get("writable", False)]

                # Add instance info
                for v in versions:
                    v["instance"] = inst
                    all_versions.append(v)

                if format == "table":
                    _display_versions_table(inst, versions, current_user)

            except Exception as e:
                click.echo(f"❌ Error for {inst}: {e}", err=True)

        # Export other formats
        if format == "json":
            import json

            click.echo(json.dumps(all_versions, indent=2))
        elif format == "csv":
            _export_versions_csv(all_versions)


@sde_commands.command("connect-test")
@click.option("--instance", "-i", default="GCOVERP", help="SDE instance name")
@click.option(
    "--version", "-v", help="Specific version (auto-detected if not provided)"
)
@click.option(
    "--version-type",
    type=click.Choice(["user_writable", "user_any", "default"]),
    default="user_writable",
    help="Type of version to find automatically",
)
def test_connection(instance, version, version_type):
    """Test SDE connection and show bridge info"""
    try:
        with create_bridge(
            instance=instance, version=version, version_type=version_type
        ) as bridge:
            click.echo("✅ Connection successful!")
            click.echo(f"   Instance: {bridge.instance}")
            click.echo(f"   Version: {bridge.version_name}")
            click.echo(f"   RC: {bridge.rc_full} ({bridge.rc_short})")
            click.echo(
                f"   Writable: {'✏️ Yes' if bridge.is_writable else '👁️ Read-only'}"
            )
            click.echo(f"   Workspace: {bridge.workspace}")

            # Test dataset access
            try:
                import arcpy

                arcpy.env.workspace = bridge.workspace
                datasets = arcpy.ListDatasets()
                click.echo(f"   📁 {len(datasets)} dataset(s) accessible")
                if datasets:
                    click.echo(f"      Examples: {', '.join(datasets[:3])}")
            except Exception as e:
                click.echo(f"   ⚠️ Dataset access error: {e}")

    except Exception as e:
        click.echo(f"❌ Connection failed: {e}", err=True)
        sys.exit(1)


# =============================================================================
# DATA EXPORT COMMANDS
# =============================================================================


@sde_commands.command("export")
@click.argument("feature_class")
@click.argument("output_path", type=click.Path())
@click.option("--instance", "-i", default="GCOVERP", help="SDE instance name")
@click.option(
    "--version", "-v", help="Specific version (auto-detected if not provided)"
)
@click.option(
    "--layer-name", "-l", help="Output layer name (defaults to feature class name)"
)
@click.option("--where", "-w", help="SQL WHERE clause for filtering")
@click.option("--bbox", type=str, help='Bounding box as "minx,miny,maxx,maxy"')
@click.option("--fields", help="Comma-separated list of fields to export")
@click.option("--max-features", type=int, help="Maximum number of features to export")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["GPKG", "GeoJSON", "ESRI Shapefile"]),
    default="GPKG",
    help="Output format",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output file")
def export_data(
    feature_class,
    output_path,
    instance,
    version,
    layer_name,
    where,
    bbox,
    fields,
    max_features,
    format,
    overwrite,
):
    """Export feature class data to file"""

    output_path = Path(output_path)

    # Check if output exists
    if output_path.exists() and not overwrite:
        click.echo(f"❌ Output file exists: {output_path}")
        click.echo("   Use --overwrite to replace it")
        sys.exit(1)

    # Parse options
    export_kwargs = {}
    if where:
        export_kwargs["where_clause"] = where
    if bbox:
        try:
            bbox_coords = [float(x.strip()) for x in bbox.split(",")]
            if len(bbox_coords) != 4:
                raise ValueError("bbox must have 4 coordinates")
            export_kwargs["bbox"] = tuple(bbox_coords)
        except Exception:
            click.echo("❌ Invalid bbox format. Use: minx,miny,maxx,maxy")
            sys.exit(1)
    if fields:
        export_kwargs["fields"] = [f.strip() for f in fields.split(",")]
    if max_features:
        export_kwargs["max_features"] = max_features

    try:
        with create_bridge(instance=instance, version=version) as bridge:
            click.echo(f"🔄 Exporting {feature_class}")
            click.echo(f"   From: {bridge.instance}::{bridge.version_name}")
            click.echo(f"   To: {output_path}")

            if export_kwargs:
                click.echo(f"   Options: {export_kwargs}")

            result_path = bridge.export_to_file(
                feature_class=feature_class,
                output_path=output_path,
                layer_name=layer_name,
                driver=format,
                **export_kwargs,
            )

            click.echo(f"✅ Export completed: {result_path}")

            # Show file info
            try:
                gdf = gpd.read_file(result_path)
                click.echo(f"   📊 {len(gdf)} features exported")
                click.echo(
                    f"   📄 {len(gdf.columns)} fields: {', '.join(gdf.columns[:5])}{'...' if len(gdf.columns) > 5 else ''}"
                )
                if hasattr(gdf, "crs") and gdf.crs:
                    click.echo(f"   🗺️  CRS: {gdf.crs}")

                # Show file size
                size_mb = result_path.stat().st_size / 1024 / 1024
                click.echo(f"   💾 File size: {size_mb:.1f} MB")

            except Exception:
                pass  # Don't fail on info display

    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)
        sys.exit(1)


@sde_commands.command("export-bulk")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--instance", "-i", default="GCOVERP", help="SDE instance name")
@click.option(
    "--version", "-v", help="Specific version (auto-detected if not provided)"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory (defaults to current directory)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["GPKG", "GeoJSON"]),
    default="GPKG",
    help="Output format",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def export_bulk(config_file, instance, version, output_dir, format, overwrite):
    """Bulk export multiple feature classes using YAML/JSON config"""

    config_file = Path(config_file)
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(exist_ok=True)

    # Load configuration
    try:
        import yaml

        with open(config_file) as f:
            if config_file.suffix.lower() == ".json":
                import json

                config = json.load(f)
            else:
                config = yaml.safe_load(f)
    except Exception as e:
        click.echo(f"❌ Error reading config file: {e}")
        sys.exit(1)

    # Validate config structure
    if "exports" not in config:
        click.echo("❌ Config file must contain 'exports' section")
        sys.exit(1)

    exports = config["exports"]
    if not isinstance(exports, list):
        click.echo("❌ 'exports' must be a list of export configurations")
        sys.exit(1)

    # Process exports
    try:
        with create_bridge(instance=instance, version=version) as bridge:
            click.echo(
                f"🔄 Starting bulk export from {bridge.instance}::{bridge.version_name}"
            )
            click.echo(f"   Output directory: {output_dir}")

            success_count = 0
            error_count = 0

            for i, export_config in enumerate(exports, 1):
                feature_class = export_config.get("feature_class")
                output_file = export_config.get("output_file")

                if not feature_class or not output_file:
                    click.echo(f"❌ Export {i}: Missing feature_class or output_file")
                    error_count += 1
                    continue

                output_path = output_dir / output_file

                # Check overwrite
                if output_path.exists() and not overwrite:
                    click.echo(f"⏭️  Export {i}: Skipping {output_path} (exists)")
                    continue

                try:
                    click.echo(
                        f"🔄 Export {i}/{len(exports)}: {feature_class} -> {output_path.name}"
                    )

                    # Extract export options
                    export_kwargs = {
                        k: v
                        for k, v in export_config.items()
                        if k not in ("feature_class", "output_file")
                    }

                    bridge.export_to_file(
                        feature_class=feature_class,
                        output_path=output_path,
                        driver=format,
                        **export_kwargs,
                    )

                    success_count += 1
                    click.echo("   ✅ Completed")

                except Exception as e:
                    click.echo(f"   ❌ Failed: {e}")
                    error_count += 1

            click.echo("\n📊 Bulk export completed:")
            click.echo(f"   ✅ Success: {success_count}")
            click.echo(f"   ❌ Errors: {error_count}")

    except Exception as e:
        click.echo(f"❌ Bulk export failed: {e}", err=True)
        sys.exit(1)


# =============================================================================
# DATA IMPORT COMMANDS
# =============================================================================


@sde_commands.command("import")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("feature_class")
@click.option("--instance", "-i", default="GCOVERP", help="SDE instance name")
@click.option(
    "--version", "-v", help="Specific version (auto-detected if not provided)"
)
@click.option("--layer", "-l", help="Layer name for multi-layer files (e.g., GPKG)")
@click.option(
    "--operation",
    type=click.Choice(["insert", "update", "delete", "upsert"]),
    default="update",
    help="Operation type",
)
@click.option("--update-fields", help="Comma-separated list of fields to update")
@click.option(
    "--no-attributes", is_flag=True, help="Skip attribute updates (geometry only)"
)
@click.option(
    "--no-geometry", is_flag=True, help="Skip geometry updates (attributes only)"
)
@click.option("--operator", help="Operator name (defaults to CLI user)")
@click.option(
    "--chunk-size", type=int, default=1000, help="Features per transaction chunk"
)
@click.option("--dryrun", is_flag=True, help="Test run without making changes")
@click.option("--confirm", is_flag=True, help="Require confirmation before import")
def import_data(
    input_path,
    feature_class,
    instance,
    version,
    layer,
    operation,
    update_fields,
    no_attributes,
    no_geometry,
    operator,
    chunk_size,
    dryrun,
    confirm,
):
    """Import data from file to feature class"""

    input_path = Path(input_path)

    # Parse update fields
    update_fields_list = None
    if update_fields:
        update_fields_list = [f.strip() for f in update_fields.split(",")]

    # Set update flags
    update_attributes = not no_attributes
    update_geometry = not no_geometry

    try:
        # Preview data first
        try:
            if layer:
                preview_gdf = gpd.read_file(input_path, layer=layer, rows=5)
                total_features = len(gpd.read_file(input_path, layer=layer))
            else:
                preview_gdf = gpd.read_file(input_path, rows=5)
                total_features = len(gpd.read_file(input_path))

            click.echo(f"📄 Input file: {input_path}")
            if layer:
                click.echo(f"   Layer: {layer}")
            click.echo(f"   Features: {total_features}")
            click.echo(f"   Fields: {list(preview_gdf.columns)}")
            click.echo(
                f"   CRS: {preview_gdf.crs if hasattr(preview_gdf, 'crs') else 'Unknown'}"
            )

        except Exception as e:
            click.echo(f"⚠️  Could not preview file: {e}")
            if not click.confirm("Continue anyway?"):
                sys.exit(1)

        with create_bridge(instance=instance, version=version) as bridge:
            click.echo(f"\n🎯 Target: {bridge.instance}::{bridge.version_name}")
            click.echo(f"   Feature class: {feature_class}")
            click.echo(f"   Operation: {operation}")
            click.echo(
                f"   Writable: {'✏️ Yes' if bridge.is_writable else '❌ Read-only'}"
            )

            if not bridge.is_writable and operation in ("insert", "update", "upsert"):
                click.echo("❌ Cannot perform write operations on read-only version")
                sys.exit(1)

            # Import options summary
            options = []
            if update_fields_list:
                options.append(f"Fields: {', '.join(update_fields_list)}")
            if not update_attributes:
                options.append("Geometry only")
            if not update_geometry:
                options.append("Attributes only")
            if operator:
                options.append(f"Operator: {operator}")
            if dryrun:
                options.append("DRY RUN")

            if options:
                click.echo(f"   Options: {', '.join(options)}")

            # Confirmation
            if confirm and not dryrun:
                if not click.confirm(f"\n⚠️  Proceed with {operation} operation?"):
                    click.echo("Operation cancelled")
                    sys.exit(0)

            click.echo(f"\n🔄 Starting {operation} operation...")

            result = bridge.import_from_file(
                file_path=input_path,
                feature_class=feature_class,
                layer=layer,
                operation=operation,
                update_fields=update_fields_list,
                update_attributes=update_attributes,
                update_geometry=update_geometry,
                operator=operator,
                chunk_size=chunk_size,
                dryrun=dryrun,
            )

            # Display results
            click.echo(f"\n📊 {operation.title()} Results:")
            click.echo(f"   ✅ Success: {result.get('success_count', 0)}")

            errors = result.get("errors", [])
            if errors:
                click.echo(f"   ❌ Errors: {len(errors)}")
                if len(errors) <= 5:
                    for error in errors:
                        click.echo(f"      • {error}")
                else:
                    for error in errors[:5]:
                        click.echo(f"      • {error}")
                    click.echo(f"      ... and {len(errors) - 5} more errors")

            # Operation-specific results
            if "details" in result:
                details = result["details"]
                for op, res in details.items():
                    click.echo(
                        f"   {op.title()}: {res.get('success_count', 0)} features"
                    )

            if dryrun:
                click.echo("\n💡 This was a dry run. Use --no-dryrun to apply changes.")
            else:
                click.echo("\n✅ Operation completed successfully!")

    except Exception as e:
        click.echo(f"❌ Import failed: {e}", err=True)
        sys.exit(1)


@sde_commands.command("list-feature-classes")
@click.option(
    "--instance",
    "-i",
    default="GCOVERP",
    help="Instance SDE à explorer"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["tree", "table"], case_sensitive=False),
    default="tree",
    help="Format d'affichage (tree ou table)"
)
@click.option(
    "--search",
    "-s",
    help="Rechercher des feature classes par nom"
)
@click.option(
    "--info",
    help="Afficher les détails d'une feature class (ex: TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK)"
)
@click.option(
    "--no-paths",
    is_flag=True,
    help="Ne pas afficher les chemins complets (format tree uniquement)"
)
def list_feature_classes(instance, format, search, info, no_paths):
    """
    Liste les feature classes disponibles dans une instance SDE.

    Exemples:

        # Afficher l'arborescence complète
        gcover sde list-feature-classes

        # Afficher sous forme de tableau
        gcover sde list-feature-classes --format table

        # Rechercher des feature classes
        gcover sde list-feature-classes --search bedrock

        # Afficher les détails d'une feature class
        gcover sde list-feature-classes --info TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK
    """
    if not HAS_ARCPY:
        console.print("[bold red]❌ Requires ArcGIS Pro[/bold red]")
        raise click.Abort()

    try:
        with SDEConnectionManager() as conn_mgr:
            # Connect to instance
            with console.status(f"[bold green]Connexion à {instance}..."):
                conn = conn_mgr.create_connection(instance)
                workspace = str(conn)

            console.print(f"[green]✓[/green] Connecté à: [cyan]{workspace}[/cyan]\n")

            # If --info specified, show details and exit
            if info:
                console.rule(f"[bold blue]Détails: {info}[/bold blue]")
                console.print()
                success = get_feature_class_info(workspace, info)
                if not success:
                    raise click.ClickException(f"Feature class non trouvée: {info}")
                return

            # Scan database
            with console.status("[bold green]Analyse de la structure..."):
                results = list_all_feature_classes(workspace)

            # If --search specified, show search results
            if search:
                console.print()
                console.rule(f"[bold blue]Recherche: '{search}'[/bold blue]")
                console.print()

                matches = search_feature_classes(workspace, search)
                display_search_results(matches, search)

                # Offer to show details
                if matches and Confirm.ask("\nAfficher les détails du premier résultat?", default=False):
                    console.print()
                    console.rule("[bold blue]Détails[/bold blue]")
                    console.print()
                    get_feature_class_info(workspace, matches[0]["path"])

                return

            # Display all feature classes
            console.print()
            if format == "tree":
                display_feature_class_tree(results, workspace, show_paths=not no_paths)
            else:
                display_feature_class_table(results, workspace)

            console.print()

    except Exception as e:
        console.print(f"[bold red]Erreur:[/bold red] {e}")
        raise click.ClickException(str(e))


@sde_commands.command("inspect")
@click.argument("feature_class_path")
@click.option(
    "--instance",
    "-i",
    default="GCOVERP",
    help="Instance SDE"
)
@click.option(
    "--sample",
    "-n",
    type=int,
    default=5,
    help="Nombre d'enregistrements à afficher en exemple"
)
def inspect_feature_class(feature_class_path, instance, sample):
    """
    Inspecte une feature class en détail.

    Affiche:
    - Métadonnées de la feature class
    - Liste des champs
    - Exemples d'enregistrements

    Exemples:

        gcover sde inspect TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK

        gcover sde inspect "TOPGIS_GC.GC_BEDROCK" --sample 10
    """
    import arcpy


    try:
        with SDEConnectionManager() as conn_mgr:
            # Connect
            with console.status(f"[bold green]Connexion à {instance}..."):
                conn = conn_mgr.create_connection(instance)
                workspace = str(conn)

            console.print(f"[green]✓[/green] Connecté à: [cyan]{workspace}[/cyan]\n")

            # Show metadata
            console.rule(f"[bold blue]{feature_class_path}[/bold blue]")
            console.print()
            get_feature_class_info(workspace, feature_class_path)

            # Show sample data
            console.print()
            console.rule("[bold blue]Exemples d'enregistrements[/bold blue]")
            console.print()

            arcpy.env.workspace = workspace
            # full_path = f"{workspace}\\{feature_class_path.replace('/', '\\')}"
            full_path = str(Path(workspace) / Path(feature_class_path))

            # Get all fields except geometry
            fields = [f.name for f in arcpy.ListFields(full_path)
                      if f.type not in ('Geometry', 'OID')][:10]  # Limit to 10 fields for display

            # Create sample data table
            sample_table = Table(
                title=f"Premiers {sample} enregistrements",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold yellow"
            )

            # Add columns
            for field in fields:
                sample_table.add_column(field, style="cyan")

            # Read sample data
            with arcpy.da.SearchCursor(full_path, fields) as cursor:
                for i, row in enumerate(cursor):
                    if i >= sample:
                        break
                    sample_table.add_row(*[str(val) if val is not None else "[dim]NULL[/dim]" for val in row])

            console.print(sample_table)

            if len(fields) > 10:
                console.print(f"\n[dim]Note: Seuls les 10 premiers champs sont affichés[/dim]")

    except Exception as e:
        console.print(f"[bold red]Erreur:[/bold red] {e}")
        raise click.ClickException(str(e))


@sde_commands.command("export-structure")
@click.option(
    "--instance",
    "-i",
    default="GCOVERP",
    help="Instance SDE"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Fichier de sortie (JSON ou YAML)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    default="json",
    help="Format de sortie"
)
def export_structure(instance, output, format):
    """
    Exporte la structure de la base de données (feature classes, champs, etc.)

    Utile pour:
    - Documentation
    - Comparaison entre versions
    - Génération de code

    Exemples:

        gcover sde export-structure -o structure.json

        gcover sde export-structure -o structure.yaml --format yaml
    """
    import arcpy
    import json
    from pathlib import Path

    try:
        with SDEConnectionManager() as conn_mgr:
            # Connect
            with console.status(f"[bold green]Connexion à {instance}..."):
                conn = conn_mgr.create_connection(instance)
                workspace = str(conn)

            console.print(f"[green]✓[/green] Connecté à: [cyan]{workspace}[/cyan]\n")

            # Scan structure
            with console.status("[bold green]Extraction de la structure..."):
                arcpy.env.workspace = workspace
                results = list_all_feature_classes(workspace)

                # Build detailed structure
                structure = {
                    "workspace": workspace,
                    "instance": instance,
                    "standalone_feature_classes": [],
                    "feature_datasets": {}
                }

                # Process standalone feature classes
                for fc in results["standalone"]:
                    full_path = f"{workspace}\\{fc}"
                    desc = arcpy.Describe(full_path)
                    fields = [{"name": f.name, "type": f.type, "length": getattr(f, 'length', None)}
                              for f in arcpy.ListFields(full_path)]

                    structure["standalone_feature_classes"].append({
                        "name": fc,
                        "type": desc.shapeType,
                        "spatial_reference": desc.spatialReference.name,
                        "fields": fields
                    })

                # Process feature datasets
                for dataset, fcs in results["datasets"].items():
                    structure["feature_datasets"][dataset] = []

                    for fc in fcs:
                        full_path = f"{workspace}\\{dataset}\\{fc}"
                        desc = arcpy.Describe(full_path)
                        fields = [{"name": f.name, "type": f.type, "length": getattr(f, 'length', None)}
                                  for f in arcpy.ListFields(full_path)]

                        structure["feature_datasets"][dataset].append({
                            "name": fc,
                            "type": desc.shapeType,
                            "spatial_reference": desc.spatialReference.name,
                            "fields": fields
                        })

            # Save to file
            if output:
                output_path = Path(output)

                if format == "yaml":
                    import yaml
                    with open(output_path, 'w', encoding='utf-8') as f:
                        yaml.dump(structure, f, default_flow_style=False, allow_unicode=True)
                else:  # json
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(structure, f, indent=2, ensure_ascii=False)

                console.print(f"[green]✓[/green] Structure exportée vers: [cyan]{output_path}[/cyan]")
            else:
                # Print to console
                import json
                console.print_json(json.dumps(structure, indent=2))

    except Exception as e:
        console.print(f"[bold red]Erreur:[/bold red] {e}")
        raise click.ClickException(str(e))

@sde_commands.command("sync")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("feature_class")
@click.option("--instance", "-i", default="GCOVERP", help="SDE instance name")
@click.option(
    "--version", "-v", help="Specific version (auto-detected if not provided)"
)
@click.option(
    "--operation-field",
    default="_operation",
    help="Field containing operation type (insert/update/delete)",
)
@click.option("--layer", "-l", help="Layer name for multi-layer files")
@click.option("--operator", help="Operator name for changes")
@click.option("--dryrun", is_flag=True, help="Test run without making changes")
@click.option(
    "--confirm-deletes",
    is_flag=True,
    default=True,
    help="Require confirmation for delete operations",
)
@click.option("--no-progress", is_flag=True, help="Disable progress bars")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
def sync_data(
    input_path,
    feature_class,
    instance,
    version,
    operation_field,
    layer,
    operator,
    dryrun,
    confirm_deletes,
    no_progress,
    quiet,
):
    """Synchronize data using operation field (insert/update/delete) with progress tracking"""

    input_path = Path(input_path)

    # Configure progress display
    show_progress = not (no_progress or quiet)

    try:
        # Load and validate data
        if layer:
            gdf = gpd.read_file(input_path, layer=layer)
        else:
            gdf = gpd.read_file(input_path)

        if operation_field not in gdf.columns:
            click.echo(f"❌ Operation field '{operation_field}' not found in data")
            click.echo(f"   Available fields: {list(gdf.columns)}")
            sys.exit(1)

        # Analyze operations
        ops_summary = gdf[operation_field].value_counts().to_dict()

        if not quiet:
            click.echo(f"📄 Input file: {input_path}")
            click.echo(f"   Total features: {len(gdf):,}")
            click.echo(f"   Operations: {ops_summary}")

        # Check for delete operations
        deletes = ops_summary.get("delete", 0)
        if deletes > 0 and confirm_deletes and not dryrun and not quiet:
            click.echo(f"\n⚠️  {deletes} features will be DELETED")
            if not click.confirm("Continue with delete operations?"):
                click.echo("Operation cancelled")
                sys.exit(0)

        with create_bridge(
            instance=instance, version=version, show_progress=show_progress
        ) as bridge:
            if not quiet:
                click.echo(f"\n🎯 Target: {bridge.instance}::{bridge.version_name}")
                click.echo(f"   Feature class: {feature_class}")

            if not bridge.is_writable:
                click.echo("❌ Cannot perform sync on read-only version")
                sys.exit(1)

            if not quiet:
                click.echo("🔄 Starting synchronization...")

            # Custom progress callback for detailed sync progress
            progress_info = {"last_update": 0, "start_time": dt.now()}

            def sync_progress_callback(current: int, total: int):
                if not quiet and not show_progress:
                    if current - progress_info["last_update"] > max(total * 0.05, 50):
                        elapsed = (
                            dt.now() - progress_info["start_time"]
                        ).total_seconds()
                        rate = current / elapsed if elapsed > 0 else 0
                        click.echo(
                            f"   🔄 Synchronizing: {current:,}/{total:,} features "
                            f"({current / total * 100:.1f}%) - {rate:.0f} features/sec"
                        )
                        progress_info["last_update"] = current

            # Use the execute_operations method from your original bridge
            # This would need to be implemented in the new bridge class
            result = bridge.import_from_geodataframe(
                gdf=gdf,
                feature_class=feature_class,
                operation="sync",  # Special sync operation
                operation_field=operation_field,
                operator=operator,
                dryrun=dryrun,
                progress_callback=sync_progress_callback if not show_progress else None,
            )

            # Display results
            if not quiet:
                click.echo("\n📊 Synchronization Results:")
                for operation, count in ops_summary.items():
                    success = (
                        result.get("details", {})
                        .get(operation, {})
                        .get("success_count", 0)
                    )
                    click.echo(
                        f"   {operation.title()}: {success:,}/{count:,} successful"
                    )

                if dryrun:
                    click.echo(
                        "\n💡 This was a dry run. Remove --dryrun to apply changes."
                    )
                else:
                    click.echo("\n✅ Synchronization completed!")

    except Exception as e:
        click.echo(f"❌ Sync failed: {e}", err=True)
        sys.exit(1)


# =============================================================================
# UTILITY FUNCTIONS (from original code, enhanced)
# =============================================================================


def _display_versions_table(instance: str, versions: List[dict], current_user: str):
    """Display versions as a formatted table"""
    if not versions:
        click.echo(f"  ℹ️  No versions found for {instance}")
        return

    click.echo(f"\n📊 {instance}")
    click.echo("=" * 80)

    table_data = []
    for v in versions:
        status = []
        if v["isOwner"]:
            status.append("👤 Owner")
        if v.get("writable", False):
            status.append("✏️ Writable")
        if current_user in v["name"].upper():
            status.append("⭐ User")

        table_data.append([v["name"], v["parent"] or "-", " ".join(status) or "-"])

    headers = ["Version", "Parent", "Status"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


def _export_versions_csv(versions: List[dict]):
    """Export versions to CSV format"""
    import csv
    import io

    output = io.StringIO()
    if versions:
        fieldnames = ["instance", "name", "parent", "isOwner", "writable"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for v in versions:
            writer.writerow(
                {
                    "instance": v["instance"],
                    "name": v["name"],
                    "parent": v["parent"] or "",
                    "isOwner": v["isOwner"],
                    "writable": v.get("writable", False),
                }
            )

    click.echo(output.getvalue())
