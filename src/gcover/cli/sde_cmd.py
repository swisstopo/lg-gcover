# gcover/cli/sde_cmd.py
import os
import click
from tabulate import tabulate
from typing import List

from ..sde import SDEConnectionManager
from ..config import SDE_INSTANCES  # Si vous avez un fichier config


@click.group(name="sde")
def sde_commands():
    """Commandes de gestion des connexions SDE"""
    pass


@sde_commands.command("versions")
@click.option('--instance', '-i',
              type=click.Choice(list(SDE_INSTANCES.values())),
              multiple=True,
              help='Instances à vérifier')
@click.option('--user-only', '-u', is_flag=True,
              help='Afficher seulement les versions utilisateur')
@click.option('--format', '-f',
              type=click.Choice(['table', 'json', 'csv']),
              default='table',
              help='Format de sortie')
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
                    versions = [v for v in versions if current_user in v["name"].upper()]

                # Ajouter l'instance à chaque version pour l'export
                for v in versions:
                    v['instance'] = inst
                    all_versions.append(v)

                if format == 'table':
                    _display_versions_table(inst, versions, current_user)

            except Exception as e:
                click.echo(f"❌ Erreur pour {inst}: {e}", err=True)

        # Export autres formats
        if format == 'json':
            import json
            click.echo(json.dumps(all_versions, indent=2))
        elif format == 'csv':
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

        table_data.append([
            v["name"],
            v["parent"] or "-",
            " ".join(status) or "-"
        ])

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
            writer.writerow({
                "instance": v["instance"],
                "name": v["name"],
                "parent": v["parent"] or "",
                "isOwner": v["isOwner"],
                "writable": v["writable"]
            })

    click.echo(output.getvalue())


@sde_commands.command("connections")
@click.option('--cleanup', is_flag=True,
              help='Nettoyer les connexions actives après affichage')
def list_connections(cleanup):
    """Liste et optionnellement nettoie les connexions SDE actives"""

    with SDEConnectionManager() as conn_mgr:
        connections = conn_mgr.list_active_connections()

        if not connections:
            click.echo("ℹ️  Aucune connexion SDE active")
            return

        click.echo(f"🔗 {len(connections)} connexion(s) active(s):")

        table_data = [
            [conn["instance"], conn["version"], conn["path"]]
            for conn in connections
        ]

        headers = ["Instance", "Version", "Chemin SDE"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

        if cleanup:
            if click.confirm("🗑️  Nettoyer toutes les connexions ?"):
                conn_mgr.cleanup_all()
                click.echo("✅ Connexions nettoyées")


@sde_commands.command("connect")
@click.option('--instance', '-i',
              type=click.Choice(list(SDE_INSTANCES.values())),
              prompt=True,
              help='Instance SDE')
@click.option('--interactive', is_flag=True,
              help='Mode interactif pour sélection de version')
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
            click.echo(f"✅ Connexion réussie:")
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
@click.option('--instance', '-i',
              type=click.Choice(list(SDE_INSTANCES.values())),
              multiple=True,
              help='Instances à vérifier (toutes par défaut)')
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