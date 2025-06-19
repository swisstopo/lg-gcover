import geopandas as gpd

from ..utils.imports import require_arcpy


@require_arcpy
class SDEConnection:
    """Gestion des connexions SDE."""

    def __init__(self, connection_string: str):
        self.connection = connection_string

    def to_geodataframe(self, query: str) -> gpd.GeoDataFrame:
        """Convertit une requête SDE en GeoDataFrame."""
        # Implementation avec arcpy

    def from_geodataframe(
        self, gdf: gpd.GeoDataFrame, target_table: str, mode: str = "append"
    ):
        """Exporte un GeoDataFrame vers SDE."""
        # Implementation avec arcpy
