# connection_manager.py
import os
import tempfile
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class SDEConnectionManager:
    """Gestionnaire centralisé des connexions SDE pour Enterprise Geodatabase"""

    def __init__(self, version: str = "SDE.DEFAULT", instance: str = "GCOVERP"):
        self._connections: Dict[str, Path] = {}  # Cache des connexions actives
        self._temp_dirs: List[Path] = []  # Suivi des répertoires temporaires
        self._version = version
        self._instance = instance

    def __enter__(self):
        users_version = self.find_user_versions(instances=["GCOVERP"])

        if users_version and "GCOVERP" in users_version:
            version_list = users_version["GCOVERP"]

            if version_list:
                user_version = version_list[0]
                self.create_connection(instance="GCOVERP", version=user_version)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()

    def create_connection(
        self, instance: str, version: str = "SDE.DEFAULT", reuse_existing: bool = True
    ) -> Path:
        """
        Crée ou réutilise une connexion SDE

        Args:
            instance: Instance de base (ex: "GCOVERP")
            version: Version à utiliser
            reuse_existing: Réutiliser une connexion existante si disponible

        Returns:
            Path vers le fichier .sde
        """
        connection_key = f"{instance}_{version}"

        # Vérifier si une connexion existe déjà
        if reuse_existing and connection_key in self._connections:
            sde_path = self._connections[connection_key]
            if sde_path.exists():
                logger.debug(f"Réutilisation connexion existante: {sde_path}")
                return sde_path
            else:
                # Nettoyer la référence obsolète
                del self._connections[connection_key]

        # Créer une nouvelle connexion
        return self._create_new_connection(instance, version, connection_key)

    def _create_new_connection(self, instance: str, version: str, key: str) -> Path:
        """Crée une nouvelle connexion SDE"""
        from gcover.arcpy_compat import HAS_ARCPY, arcpy

        # Créer répertoire temporaire
        temp_dir = Path(tempfile.mkdtemp(prefix="sde_conn_"))
        self._temp_dirs.append(temp_dir)

        # Générer nom unique
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        sde_name = f"conn_{instance}_{timestamp}.sde"
        sde_path = temp_dir / sde_name

        try:
            # Créer la connexion
            arcpy.management.CreateDatabaseConnection(
                out_folder_path=str(temp_dir),
                out_name=sde_name,
                database_platform="ORACLE",
                instance=instance,
                account_authentication="OPERATING_SYSTEM_AUTH",
                username="",
                password="",
                version=version,
                save_user_pass="DO_NOT_SAVE_USERNAME",
            )

            # Stocker dans le cache
            self._connections[key] = sde_path
            logger.info(f"Connexion créée: {sde_path}")
            return sde_path

        except Exception as e:
            logger.error(f"Erreur création connexion: {e}")
            # Nettoyer en cas d'erreur
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def get_versions(self, instance: str) -> List[Dict]:
        """Récupère la liste des versions disponibles"""
        from gcover.arcpy_compat import HAS_ARCPY, arcpy

        temp_connection = self.create_connection(instance, "SDE.DEFAULT")

        try:
            versions = []
            ver_list = arcpy.da.ListVersions(str(temp_connection))

            for version in ver_list:
                versions.append(
                    {
                        "name": version.name,
                        "parent": version.parentVersionName,
                        "isOwner": version.isOwner,
                        "writable": version.isOwner,
                        "instance": instance,
                    }
                )
            return versions

        except Exception as e:
            logger.error(f"Erreur récupération versions: {e}")
            return []

    def find_user_versions(self, instances: List[str] = None) -> Dict[str, List[Dict]]:
        """Trouve les versions utilisateur sur plusieurs instances"""
        if instances is None:
            instances = ["GCOVERP", "GCOVERI"]

        current_user = os.getlogin().upper()
        user_versions = {}

        for instance in instances:
            try:
                versions = self.get_versions(instance)
                logger.debug(versions)
                user_versions[instance] = [
                    v
                    for v in versions
                    if current_user in v["name"].upper() or v["isOwner"]
                ]
            except Exception as e:
                logger.error(f"Erreur pour instance {instance}: {e}")
                user_versions[instance] = []

        return user_versions

    def cleanup_connection(self, instance: str, version: str = None):
        """Nettoie une connexion spécifique"""
        if version:
            key = f"{instance}_{version}"
            if key in self._connections:
                sde_path = self._connections[key]
                if sde_path.exists():
                    sde_path.unlink()
                del self._connections[key]
        else:
            # Nettoyer toutes les connexions de cette instance
            keys_to_remove = [
                k for k in self._connections.keys() if k.startswith(f"{instance}_")
            ]
            for key in keys_to_remove:
                self.cleanup_connection(*key.split("_", 1))

    def cleanup_all(self):
        """Nettoie toutes les connexions et répertoires temporaires"""
        # Supprimer les fichiers SDE
        for sde_path in self._connections.values():
            if sde_path.exists():
                try:
                    sde_path.unlink()
                except OSError:
                    pass

        # Supprimer les répertoires temporaires
        import shutil

        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass

        self._connections.clear()
        self._temp_dirs.clear()

    def list_active_connections(self) -> List[Dict]:
        """Liste les connexions actives"""
        active = []
        for key, path in self._connections.items():
            if path.exists():
                instance, version = key.split("_", 1)
                active.append(
                    {
                        "instance": instance,
                        "version": version,
                        "path": str(path),
                        "key": key,
                    }
                )
        return active
