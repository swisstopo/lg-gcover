"""
Schema management module for gcover.

This module provides tools for:
- Extracting schemas from ESRI geodatabases (requires arcpy)
- Comparing schemas and tracking changes
- Generating documentation and diagrams
- Transforming between different schema formats
"""

from gcover.schema.serializer import (save_esri_schema_to_file,
                                      serialize_domains_only,
                                      serialize_esri_schema_to_dict,
                                      serialize_esri_schema_to_json,
                                      serialize_feature_classes_only)

# Import des classes principales
from gcover.schema.differ import (ChangeType, DomainChange, FieldChange, RelationshipChange,
                     SchemaDiff, SubtypeChange, TableChange)
from gcover.schema.exporters.json import export_esri_schema_to_json
from gcover.schema.exporters.plantuml import generate_plantuml_from_schema
from gcover.schema.models import (CodedDomain, CodedValue, ESRISchema, FeatureClass, Field,
                     Index, RangeDomain, RelationshipClass, Subtype,
                     SubtypeValue, Table)
from gcover.schema.reporter import generate_report, schema_diff_to_dict
from gcover.schema.simple_transformer import transform_esri_flat_json
from gcover.schema.transformer import transform_esri_json

# Import conditionnel de l'extracteur (nécessite arcpy)
try:
    from gcover.schema.extractor import can_extract_schema, extract_schema
except ImportError:
    # arcpy n'est pas disponible
    def extract_schema(*args, **kwargs):
        raise ImportError("extract_schema requires arcpy")

    def can_extract_schema():
        return False


# Import des exporteurs
from gcover.schema.exporters.json import (export_esri_schema_to_json,  # TODO PlantUML
                             export_schema_diff_to_json)
from gcover.schema.filegdb_parser import parse_filegdb_to_esri_schema

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
    "CodedValue",
    "SchemaDiff",
    # Fonctions
    "extract_schema",
    "can_extract_schema",
    "transform_esri_json",
    "transform_esri_flat_json",
    "export_esri_schema_to_json",
    "export_schema_diff_to_json",
    "generate_plantuml_from_schema",
    "generate_report",
    "serialize_esri_schema_to_dict",
    "serialize_esri_schema_to_json",
    "save_esri_schema_to_file",
    "serialize_domains_only",
    "serialize_feature_classes_only",
    "parse_filegdb_to_esri_schema",
    "generate_report",
    "schema_diff_to_dict",
]
