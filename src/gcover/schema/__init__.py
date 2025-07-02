"""
Schema management module for gcover.

This module provides tools for:
- Extracting schemas from ESRI geodatabases (requires arcpy)
- Comparing schemas and tracking changes
- Generating documentation and diagrams
- Transforming between different schema formats
"""

# Import des classes principales
from .differ import (
    ChangeType,
    DomainChange,
    FieldChange,
    RelationshipChange,
    SchemaDiff,
    SubtypeChange,
    TableChange,
)
from .exporters.json import export_esri_schema_to_json
from .exporters.plantuml import generate_plantuml_from_schema
from .models import (
    CodedDomain,
    CodedValue,
    ESRISchema,
    FeatureClass,
    Field,
    Index,
    RangeDomain,
    RelationshipClass,
    Subtype,
    SubtypeValue,
    Table,
)
from .reporter import generate_report
from .transformer import transform_esri_json

# Import conditionnel de l'extracteur (nécessite arcpy)
try:
    from .extractor import can_extract_schema, extract_schema
except ImportError:
    # arcpy n'est pas disponible
    def extract_schema(*args, **kwargs):
        raise ImportError("extract_schema requires arcpy")

    def can_extract_schema():
        return False


# Import des exporteurs
# TODO
"""from .exporters import (
    export_esri_schema_to_json,
    export_schema_diff_to_json,
    generate_plantuml_from_schema
)"""

__all__ = [
    # Classes principales
    "ESRISchema",
    "Field",
    "FeatureClass",
    "Table",
    "Index",
    "RelationshipClass",
    "Subtype",
    "SubtypeValue",
    "RangeDomain",
    "CodedDomain",
    "CodedValue",firef
    "SchemaDiff",
    # Fonctions
    "extract_schema",
    "can_extract_schema",
    "transform_esri_json",
    "export_esri_schema_to_json",
    # "export_schema_diff_to_json",
    "generate_plantuml_from_schema",
    "generate_report",
]
