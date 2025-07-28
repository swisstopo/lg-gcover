"""
JSON serializer for ESRISchema dataclasses.

This module provides functions to serialize ESRISchema dataclass instances
to JSON-compatible dictionaries, which is the inverse operation of the
transform_esri_json deserializer.
"""

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Union
from datetime import datetime

# Import your dataclasses
from .models import (
    CodedValue,
    CodedDomain,
    RangeDomain,
    Field,
    SubtypeValue,
    Subtype,
    RelationshipClass,
    Index,
    FeatureClass,
    Table,
    ESRISchema,
)


def serialize_esri_schema_to_dict(schema: ESRISchema) -> Dict[str, Any]:
    """
    Serialize an ESRISchema instance to a JSON-compatible dictionary.

    Args:
        schema: ESRISchema instance to serialize

    Returns:
        Dictionary that can be serialized to JSON
    """
    result = {}

    # Serialize coded domains
    if schema.coded_domains:
        result["coded_domains"] = {
            name: serialize_coded_domain(domain)
            for name, domain in schema.coded_domains.items()
        }

    # Serialize range domains
    if schema.range_domains:
        result["range_domains"] = {
            name: serialize_range_domain(domain)
            for name, domain in schema.range_domains.items()
        }

    # Serialize subtypes
    if schema.subtypes:
        result["subtypes"] = {
            name: serialize_subtype(subtype)
            for name, subtype in schema.subtypes.items()
        }

    # Serialize relationships
    if schema.relationships:
        result["relationships"] = {
            name: serialize_relationship_class(rel)
            for name, rel in schema.relationships.items()
        }

    # Serialize feature classes
    if schema.feature_classes:
        result["feature_classes"] = {
            name: serialize_feature_class(fc)
            for name, fc in schema.feature_classes.items()
        }

    # Serialize tables
    if schema.tables:
        result["tables"] = {
            name: serialize_table(table) for name, table in schema.tables.items()
        }

    # Serialize metadata
    if schema.metadata:
        result["metadata"] = dict(schema.metadata)

    return result


def serialize_coded_domain(domain: CodedDomain) -> Dict[str, Any]:
    """Serialize a CodedDomain to dictionary."""
    result = {
        "name": domain.name,
        "type": "codedValue",  # Add explicit type for ESRI compatibility
    }

    if domain.description is not None:
        result["description"] = domain.description
    if domain.field_type is not None:
        result["field_type"] = domain.field_type

    if domain.coded_values:
        result["coded_values"] = [
            serialize_coded_value(cv) for cv in domain.coded_values
        ]

    return result


def serialize_coded_value(coded_value: CodedValue) -> Dict[str, Any]:
    """Serialize a CodedValue to dictionary."""
    return {"code": coded_value.code, "name": coded_value.name}


def serialize_range_domain(domain: RangeDomain) -> Dict[str, Any]:
    """Serialize a RangeDomain to dictionary."""
    result = {
        "name": domain.name,
        "type": "range",  # Add explicit type for ESRI compatibility
    }

    if domain.description is not None:
        result["description"] = domain.description
    if domain.field_type is not None:
        result["field_type"] = domain.field_type
    if domain.min_value is not None:
        result["min_value"] = domain.min_value
    if domain.max_value is not None:
        result["max_value"] = domain.max_value

    return result


def serialize_field(field: Field) -> Dict[str, Any]:
    """Serialize a Field to dictionary."""
    result = {"name": field.name}

    if field.alias is not None:
        result["alias"] = field.alias
    if field.type is not None:
        result["type"] = field.type
    if field.length is not None:
        result["length"] = field.length
    if field.precision is not None:
        result["precision"] = field.precision
    if field.scale is not None:
        result["scale"] = field.scale

    result["nullable"] = field.nullable

    if field.default_value is not None:
        result["default_value"] = field.default_value
    if field.domain is not None:
        result["domain"] = field.domain

    return result


def serialize_subtype_value(subtype_value: SubtypeValue) -> Dict[str, Any]:
    """Serialize a SubtypeValue to dictionary."""
    return {"code": subtype_value.code, "name": subtype_value.name}


def serialize_subtype(subtype: Subtype) -> Dict[str, Any]:
    """Serialize a Subtype to dictionary."""
    result = {"name": subtype.name, "subtype_field": subtype.subtype_field}

    if subtype.default_subtype is not None:
        result["default_subtype"] = subtype.default_subtype

    if subtype.subtypes:
        result["subtypes"] = [serialize_subtype_value(sv) for sv in subtype.subtypes]

    if subtype.field_domains:
        result["field_domains"] = dict(subtype.field_domains)

    return result


def serialize_relationship_class(rel: RelationshipClass) -> Dict[str, Any]:
    """Serialize a RelationshipClass to dictionary."""
    result = {
        "name": rel.name,
        "origin_table": rel.origin_table,
        "destination_table": rel.destination_table,
        "relationship_type": rel.relationship_type,
    }

    if rel.forward_label is not None:
        result["forward_label"] = rel.forward_label
    if rel.backward_label is not None:
        result["backward_label"] = rel.backward_label
    if rel.cardinality is not None:
        result["cardinality"] = rel.cardinality

    result["is_composite"] = rel.is_composite
    result["is_attributed"] = rel.is_attributed

    if rel.origin_primary_key is not None:
        result["origin_primary_key"] = rel.origin_primary_key
    if rel.origin_foreign_key is not None:
        result["origin_foreign_key"] = rel.origin_foreign_key
    if rel.destination_primary_key is not None:
        result["destination_primary_key"] = rel.destination_primary_key
    if rel.destination_foreign_key is not None:
        result["destination_foreign_key"] = rel.destination_foreign_key

    return result


def serialize_index(index: Index) -> Dict[str, Any]:
    """Serialize an Index to dictionary."""
    return {
        "name": index.name,
        "fields": list(index.fields),
        "is_unique": index.is_unique,
        "is_ascending": index.is_ascending,
    }


def serialize_feature_class(fc: FeatureClass) -> Dict[str, Any]:
    """Serialize a FeatureClass to dictionary."""
    result = {"name": fc.name}

    if fc.alias is not None:
        result["alias"] = fc.alias
    if fc.feature_type is not None:
        result["feature_type"] = fc.feature_type
    if fc.geometry_type is not None:
        result["geometry_type"] = fc.geometry_type
    if fc.spatial_reference is not None:
        result["spatial_reference"] = fc.spatial_reference

    result["has_z"] = fc.has_z
    result["has_m"] = fc.has_m

    if fc.fields:
        result["fields"] = [serialize_field(field) for field in fc.fields]

    if fc.indexes:
        result["indexes"] = [serialize_index(index) for index in fc.indexes]

    if fc.subtypes is not None:
        result["subtypes"] = serialize_subtype(fc.subtypes)

    if fc.default_subtype is not None:
        result["default_subtype"] = fc.default_subtype
    if fc.subtype_field is not None:
        result["subtype_field"] = fc.subtype_field
    if fc.object_id_field is not None:
        result["object_id_field"] = fc.object_id_field
    if fc.global_id_field is not None:
        result["global_id_field"] = fc.global_id_field
    if fc.shape_field is not None:
        result["shape_field"] = fc.shape_field
    if fc.area_field is not None:
        result["area_field"] = fc.area_field
    if fc.length_field is not None:
        result["length_field"] = fc.length_field
    if fc.primary_key is not None:
        result["primary_key"] = fc.primary_key

    if fc.foreign_keys:
        result["foreign_keys"] = dict(fc.foreign_keys)

    return result


def serialize_table(table: Table) -> Dict[str, Any]:
    """Serialize a Table to dictionary."""
    result = {"name": table.name}

    if table.alias is not None:
        result["alias"] = table.alias

    if table.fields:
        result["fields"] = [serialize_field(field) for field in table.fields]

    if table.indexes:
        result["indexes"] = [serialize_index(index) for index in table.indexes]

    if table.subtypes is not None:
        result["subtypes"] = serialize_subtype(table.subtypes)

    if table.default_subtype is not None:
        result["default_subtype"] = table.default_subtype
    if table.subtype_field is not None:
        result["subtype_field"] = table.subtype_field
    if table.object_id_field is not None:
        result["object_id_field"] = table.object_id_field
    if table.global_id_field is not None:
        result["global_id_field"] = table.global_id_field
    if table.primary_key is not None:
        result["primary_key"] = table.primary_key

    if table.foreign_keys:
        result["foreign_keys"] = dict(table.foreign_keys)

    return result


def serialize_esri_schema_to_json(
    schema: ESRISchema,
    indent: int = 2,
    ensure_ascii: bool = False,
    add_timestamp: bool = True,
) -> str:
    """
    Serialize an ESRISchema instance to JSON string.

    Args:
        schema: ESRISchema instance to serialize
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters
        add_timestamp: Whether to add export timestamp to metadata

    Returns:
        JSON string representation of the schema
    """
    # Get dictionary representation
    schema_dict = serialize_esri_schema_to_dict(schema)

    # Add export timestamp if requested
    if add_timestamp:
        if "metadata" not in schema_dict:
            schema_dict["metadata"] = {}
        schema_dict["metadata"]["exported_at"] = datetime.now().isoformat()

    return json.dumps(schema_dict, indent=indent, ensure_ascii=ensure_ascii)


def save_esri_schema_to_file(
    schema: ESRISchema,
    filepath: str,
    indent: int = 2,
    ensure_ascii: bool = False,
    add_timestamp: bool = True,
) -> None:
    """
    Save an ESRISchema instance to a JSON file.

    Args:
        schema: ESRISchema instance to save
        filepath: Path to save the JSON file
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters
        add_timestamp: Whether to add export timestamp to metadata
    """
    json_string = serialize_esri_schema_to_json(
        schema, indent=indent, ensure_ascii=ensure_ascii, add_timestamp=add_timestamp
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json_string)


# Convenience functions for partial serialization
def serialize_domains_only(schema: ESRISchema) -> Dict[str, Any]:
    """Serialize only the domains from an ESRISchema."""
    result = {}

    if schema.coded_domains:
        result["coded_domains"] = {
            name: serialize_coded_domain(domain)
            for name, domain in schema.coded_domains.items()
        }

    if schema.range_domains:
        result["range_domains"] = {
            name: serialize_range_domain(domain)
            for name, domain in schema.range_domains.items()
        }

    return result


def serialize_feature_classes_only(schema: ESRISchema) -> Dict[str, Any]:
    """Serialize only the feature classes from an ESRISchema."""
    if schema.feature_classes:
        return {
            "feature_classes": {
                name: serialize_feature_class(fc)
                for name, fc in schema.feature_classes.items()
            }
        }
    return {}


def serialize_tables_only(schema: ESRISchema) -> Dict[str, Any]:
    """Serialize only the tables from an ESRISchema."""
    if schema.tables:
        return {
            "tables": {
                name: serialize_table(table) for name, table in schema.tables.items()
            }
        }
    return {}


# Example usage function
def example_usage():
    """Example of how to use the serializer functions."""
    # Assuming you have a schema object loaded
    # schema = transform_esri_json(your_esri_json_data)

    # Serialize to dictionary
    # schema_dict = serialize_esri_schema_to_dict(schema)

    # Serialize to JSON string
    # json_string = serialize_esri_schema_to_json(schema)

    # Save to file
    # save_esri_schema_to_file(schema, "exported_schema.json")

    # Serialize only specific parts
    # domains_only = serialize_domains_only(schema)
    # feature_classes_only = serialize_feature_classes_only(schema)

    pass


if __name__ == "__main__":
    # If running as script, show example usage
    print("ESRISchema JSON Serializer")
    print("Import this module and use the serialization functions:")
    print("  - serialize_esri_schema_to_dict(schema)")
    print("  - serialize_esri_schema_to_json(schema)")
    print("  - save_esri_schema_to_file(schema, 'file.json')")
