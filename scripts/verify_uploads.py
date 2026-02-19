#!/usr/bin/env python3
"""
Script de vérification des uploads S3 vs métadonnées DuckDB
Vérifie que les assets marqués comme 'uploaded' dans la base de données
existent réellement dans S3.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import click
import duckdb
from loguru import logger
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

# Ajuster selon votre structure de projet
from gcover.config import load_config
from gcover.gdb.storage import MetadataDB, S3Uploader

console = Console()


def verify_asset_uploads(
    db_path: str,
    s3_config: str,
    fix_discrepancies: bool = False,
    asset_type: str = None,
    rc_filter: str = None,
) -> Dict[str, List]:
    """
    Vérifie la cohérence entre les assets marqués 'uploaded' dans DuckDB
    et leur présence effective dans S3.

    Args:
        db_path: Chemin vers la base DuckDB
        s3_bucket: Nom du bucket S3
        s3_profile: Profil AWS (optionnel)
        fix_discrepancies: Si True, met à jour le flag 'uploaded' en cas de discordance
        asset_type: Filtrer par type d'asset (optionnel)
        rc_filter: Filtrer par RC (RC1 ou RC2, optionnel)

    Returns:
        Dict contenant les résultats de la vérification
    """

    results = {
        "verified": [],  # Assets correctement uploadés
        "missing_in_s3": [],  # Marqués uploaded mais absents de S3
        "not_uploaded": [],  # Marqués non-uploaded (info)
        "errors": [],  # Erreurs lors de la vérification
    }

    # Connexion S3
    s3_bucket = s3_config.bucket
    try:
        s3_uploader = S3Uploader(
            bucket_name=s3_bucket,
            lambda_endpoint=s3_config.lambda_endpoint,
            totp_secret=s3_config.totp_secret,
            proxy_config=s3_config.proxy,
        )
        console.print(f"[green]✓ Connexion S3 établie (bucket: {s3_bucket})[/green]")
    except Exception as e:
        console.print(f"[red]✗ Erreur connexion S3: {e}[/red]")
        return results

    # Requête DuckDB
    query = """
        SELECT 
            id, 
            path, 
            s3_key, 
            uploaded, 
            asset_type, 
            release_candidate,
            timestamp
        FROM gdb_assets
        WHERE 1=1
    """
    params = []

    if asset_type:
        query += " AND asset_type = ?"
        params.append(asset_type)

    if rc_filter:
        rc_value = "2016-12-31" if rc_filter == "RC1" else "2030-12-31"
        query += " AND release_candidate = ?"
        params.append(rc_value)

    query += " ORDER BY timestamp DESC"

    # Lecture des assets
    with duckdb.connect(db_path) as conn:
        assets = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]

    if not assets:
        console.print("[yellow]Aucun asset trouvé dans la base[/yellow]")
        return results

    console.print(f"[cyan]Vérification de {len(assets)} assets...[/cyan]\n")

    # Vérification avec progress bar
    with Progress() as progress:
        task = progress.add_task("Vérification des uploads...", total=len(assets))

        for asset_data in assets:
            asset = dict(zip(columns, asset_data))
            asset_id = asset["id"]
            path = asset["path"]
            s3_key = asset["s3_key"]
            uploaded = asset["uploaded"]

            progress.update(task, advance=1)

            # Cas 1: Marqué comme non-uploadé
            if not uploaded:
                results["not_uploaded"].append(asset)
                continue

            # Cas 2: Marqué comme uploadé mais pas de s3_key
            if not s3_key:
                results["errors"].append(
                    {"asset": asset, "error": "Marqué 'uploaded' mais s3_key est NULL"}
                )
                continue

            # Cas 3: Vérification S3
            try:
                # exists_in_s3 = s3_uploader.file_exists(s3_key)

                presigned_data = s3_uploader._get_presigned_url(s3_key, 999)

                if not presigned_data:
                    raise Exception("Exception geting presinged")

                # Check if file already exists (status 409)
                status_code = presigned_data.get("status_code")
                if status_code == 409:
                    results["verified"].append(asset)
                else:
                    results["missing_in_s3"].append(asset)

                    # Correction optionnelle
                    if fix_discrepancies:
                        with duckdb.connect(db_path) as conn:
                            conn.execute(
                                "UPDATE gdb_assets SET uploaded = FALSE WHERE id = ?",
                                [asset_id],
                            )
                        logger.warning(
                            f"Corrigé: asset {asset_id} ({Path(path).name}) "
                            f"marqué comme non-uploadé"
                        )

            except Exception as e:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                results["errors"].append({"asset": asset, "error": str(e)})

    return results


@click.command()
@click.option(
    "--db-path",
    type=click.Path(exists=True),
    help="Chemin vers la base DuckDB (défaut: depuis config)",
)
@click.option("--bucket", type=str, help="Nom du bucket S3 (défaut: depuis config)")
@click.option("--profile", type=str, help="Profil AWS (défaut: depuis config)")
@click.option(
    "--environment",
    type=click.Choice(["development", "integration", "production"]),
    default="development",
    help="Environnement de configuration",
)
@click.option(
    "--asset-type",
    type=str,
    help="Filtrer par type d'asset (ex: verification_topology)",
)
@click.option(
    "--rc", type=click.Choice(["RC1", "RC2"]), help="Filtrer par Release Candidate"
)
@click.option(
    "--fix",
    is_flag=True,
    help="Corriger automatiquement les discordances (marque comme non-uploadé)",
)
@click.option(
    "--show-all",
    is_flag=True,
    help="Afficher tous les assets, y compris ceux non-uploadés",
)
@click.option("--verbose", is_flag=True, help="Mode verbeux")
def main(db_path, bucket, profile, environment, asset_type, rc, fix, show_all, verbose):
    """
    Vérifie la cohérence entre les assets marqués 'uploaded' dans DuckDB
    et leur présence effective dans S3.

    Exemples:
        # Vérification basique
        python verify_uploads.py

        # Avec correction automatique
        python verify_uploads.py --fix

        # Filtrer par type
        python verify_uploads.py --asset-type verification_topology

        # Filtrer par RC
        python verify_uploads.py --rc RC1

        # Environnement production
        python verify_uploads.py --environment production
    """

    if verbose:
        logger.enable("gcover")

    console.print("[bold cyan]═══ Vérification des uploads S3 ═══[/bold cyan]\n")

    # Chargement de la configuration
    try:
        config = load_config(environment=environment)

        # Utilisation des valeurs de config par défaut si non spécifiées
        db_path = db_path or str(config.gdb.db_path)
        bucket = bucket or config.gdb.get_s3_bucket(config.global_)
        profile = profile or config.gdb.get_s3_profile(config.global_)

        s3_config = config.global_.s3
        print(s3_config)

        console.print(f"[dim]Environnement: {environment}[/dim]")
        console.print(f"[dim]Base de données: {db_path}[/dim]")
        console.print(f"[dim]Bucket S3: {bucket}[/dim]")
        if profile:
            console.print(f"[dim]Profil AWS: {profile}[/dim]")
        console.print()

    except Exception as e:
        console.print(f"[red]Erreur chargement config: {e}[/red]")
        if not db_path or not bucket:
            console.print("[red]Spécifiez --db-path et --bucket manuellement[/red]")
            sys.exit(1)

    # Avertissement en cas de correction
    if fix:
        console.print("[bold yellow]⚠ MODE CORRECTION ACTIVÉ[/bold yellow]")
        console.print(
            "[yellow]Les assets manquants seront marqués comme non-uploadés[/yellow]\n"
        )
        if not click.confirm("Continuer ?"):
            console.print("[red]Annulé[/red]")
            sys.exit(0)

    # Vérification
    results = verify_asset_uploads(
        db_path=db_path,
        s3_config=s3_config,
        fix_discrepancies=fix,
        asset_type=asset_type,
        rc_filter=rc,
    )

    # Affichage des résultats
    console.print("\n[bold cyan]═══ Résultats ═══[/bold cyan]\n")

    # Statistiques globales
    total_checked = len(results["verified"]) + len(results["missing_in_s3"])
    stats_table = Table(title="Statistiques")
    stats_table.add_column("Catégorie", style="cyan")
    stats_table.add_column("Nombre", justify="right", style="magenta")

    stats_table.add_row("✓ Vérifiés (OK)", f"[green]{len(results['verified'])}[/green]")
    stats_table.add_row(
        "✗ Manquants dans S3", f"[red]{len(results['missing_in_s3'])}[/red]"
    )
    stats_table.add_row(
        "⊘ Non uploadés (DB)", f"[dim]{len(results['not_uploaded'])}[/dim]"
    )
    stats_table.add_row("⚠ Erreurs", f"[yellow]{len(results['errors'])}[/yellow]")

    if total_checked > 0:
        success_rate = (len(results["verified"]) / total_checked) * 100
        stats_table.add_row(
            "Taux de réussite",
            f"[{'green' if success_rate > 95 else 'yellow'}]{success_rate:.1f}%[/]",
        )

    console.print(stats_table)

    # Détails des problèmes
    if results["missing_in_s3"]:
        console.print("\n[bold red]Assets manquants dans S3:[/bold red]")
        missing_table = Table()
        missing_table.add_column("Fichier", style="yellow", max_width=40)
        missing_table.add_column("Type", style="cyan")
        missing_table.add_column("RC", style="magenta")
        missing_table.add_column("Date", style="dim")
        missing_table.add_column("S3 Key", style="red", max_width=50)

        for asset in results["missing_in_s3"][:20]:  # Limiter à 20
            rc_name = "RC1" if asset["release_candidate"] == "2016-12-31" else "RC2"
            missing_table.add_row(
                Path(asset["path"]).name,
                asset["asset_type"],
                rc_name,
                asset["timestamp"].strftime("%Y-%m-%d"),
                asset["s3_key"],
            )

        console.print(missing_table)

        if len(results["missing_in_s3"]) > 20:
            console.print(
                f"[dim]... et {len(results['missing_in_s3']) - 20} autres[/dim]"
            )

        if fix:
            console.print("\n[green]✓ Base de données corrigée[/green]")

    # Erreurs
    if results["errors"]:
        console.print("\n[bold yellow]Erreurs rencontrées:[/bold yellow]")
        for i, error_info in enumerate(results["errors"][:10], 1):
            asset = error_info["asset"]
            error = error_info["error"]
            console.print(f"  {i}. {Path(asset['path']).name}: {error}")

        if len(results["errors"]) > 10:
            console.print(
                f"[dim]... et {len(results['errors']) - 10} autres erreurs[/dim]"
            )

    # Assets non-uploadés (optionnel)
    if show_all and results["not_uploaded"]:
        console.print(
            f"\n[dim]Assets non-uploadés: {len(results['not_uploaded'])}[/dim]"
        )

    # Code de sortie
    if results["missing_in_s3"] or results["errors"]:
        console.print(
            f"\n[bold yellow]⚠ {len(results['missing_in_s3']) + len(results['errors'])} "
            f"problème(s) détecté(s)[/bold yellow]"
        )
        sys.exit(1)
    else:
        console.print(
            "\n[bold green]✓ Tous les assets uploadés sont cohérents ![/bold green]"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
