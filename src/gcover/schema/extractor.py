import json
from pathlib import Path
from typing import Optional, Union

from ..utils.imports import require_arcpy
from gcover.models import ESRISchema
from gcover.transformer import transform_esri_json


@require_arcpy
def extract_schema(
    source: Union[str, Path],
    output_dir: Optional[Path] = None,
    name: Optional[str] = None,
    formats: list[str] = ["json"],
) -> ESRISchema:
    """
    Extrait le schéma d'une geodatabase en utilisant arcpy.

    Args:
        source: Chemin vers SDE ou GDB
        output_dir: Répertoire de sortie pour le rapport
        name: Nom du rapport
        formats: Formats de sortie (json, html, etc.)

    Returns:
        ESRISchema: Schéma extrait
    """
    import tempfile

    from gcover.arcpy_compat import HAS_ARCPY, arcpy

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())

    if name is None:
        name = f"schema_report_{Path(source).stem}"

    # Générer le rapport ESRI
    arcpy.management.GenerateSchemaReport(
        in_dataset=str(source), out_location=str(output_dir), name=name, formats=formats
    )

    # Lire le JSON généré
    json_file = output_dir / f"{name}.json"
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    # Transformer en ESRISchema
    return transform_esri_json(data)


def can_extract_schema() -> bool:
    """Vérifie si l'extraction de schéma est disponible."""
    from ..utils.imports import HAS_ARCPY

    return HAS_ARCPY
