
from pathlib import Path
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple

from loguru import logger
from .connection_manager import SDEConnectionManager

class LayerBridge:
    def __init__(self, uuid_field=None, instance="GCOVERP", version="SDE.DEFAULT",
                 connection_manager: SDEConnectionManager = None):
        """
        Bridge vers les couches géologiques avec gestionnaire de connexions

        Args:
            connection_manager: Gestionnaire de connexions partagé (optionnel)
        """
        self.instance = instance
        self.version_name = version
        self.uuid_field = uuid_field
        self.date_of_change = "DATEOFCHANGE"
        self.operator = "OPERATOR"
        self.mandatory_fields = ["OPERATOR", "DATEOFCHANGE"]

        # Utiliser le gestionnaire fourni ou en créer un
        self.conn_manager = connection_manager or SDEConnectionManager()
        self._owns_connection_manager = connection_manager is None

        self.workspace = None
        self._read_only = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_connection_manager:
            self.conn_manager.cleanup_all()

    def connect(self, version=None):
        """Connexion via le gestionnaire"""
        version_to_use = version or self.version_name

        sde_path = self.conn_manager.create_connection(
            self.instance,
            version_to_use
        )

        # Configurer arcpy
        import arcpy
        arcpy.env.workspace = str(sde_path)
        self.workspace = str(sde_path)

        return self.workspace

    def get_versions(self):
        """Déléguer au gestionnaire de connexions"""
        return self.conn_manager.get_versions(self.instance)