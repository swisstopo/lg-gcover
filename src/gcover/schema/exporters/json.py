from typing import Any

from ..models import ESRISchema


def convert_field_type(esri_type):
    """Convert ESRI field types to simplified types"""
    type_mapping = {
        "esriFieldTypeOID": "OID",
        "esriFieldTypeInteger": "Integer",
        "esriFieldTypeSmallInteger": "SmallInteger",
        "esriFieldTypeDouble": "Double",
        "esriFieldTypeSingle": "Single",
        "esriFieldTypeString": "String",
        "esriFieldTypeDate": "Date",
        "esriFieldTypeBlob": "Blob",
        "esriFieldTypeRaster": "Raster",
        "esriFieldTypeGUID": "GUID",
        "esriFieldTypeGlobalID": "GlobalID",
        "esriFieldTypeXML": "XML",
        "esriFieldTypeGeometry": "Geometry",
    }

    return type_mapping.get(esri_type, esri_type)


def export_esri_schema_to_json(schema: ESRISchema, version: int = 2) -> dict[str, Any]:
    """
    Export ESRISchema instance to JSON format matching the original structure.

    Args:
        schema: ESRISchema instance to export
        version: Schema version number (default: 2)

    Returns:
        Dictionary ready to be serialized to JSON
    """
    result = {"version": version}

    # Add metadata if available
    if schema.metadata:
        result["metadata"] = schema.metadata

    # Export coded domains
    if schema.coded_domains:
        coded_domains_dict = {}
        for domain_name, domain in schema.coded_domains.items():
            domain_dict = {"type": "CodedValue", "codedValues": {}}

            # Convert coded values to dict format
            for coded_value in domain.coded_values:
                # Convert code to string for consistency
                code_str = str(coded_value.code)
                domain_dict["codedValues"][code_str] = coded_value.name

            # Add description if available
            if domain.description:
                domain_dict["description"] = domain.description

            coded_domains_dict[domain_name] = domain_dict

        result["coded_domain"] = coded_domains_dict

    # Export tables
    if schema.tables:
        tables_dict = {}
        for table_name, table in schema.tables.items():
            table_dict = {"fields": []}

            # Export fields
            for field in table.fields:
                field_dict = {
                    "name": field.name,
                    "type": convert_field_type(field.type),
                    "length": field.length,
                    "domain": field.domain,
                }
                table_dict["fields"].append(field_dict)

            tables_dict[table_name] = table_dict

        result["tables"] = tables_dict

    # Export feature classes
    if schema.feature_classes:
        featclasses_dict = {}
        for fc_name, fc in schema.feature_classes.items():
            fc_dict = {"fields": []}

            # Export fields
            for field in fc.fields:
                field_dict = {
                    "name": field.name,
                    "type": convert_field_type(field.type),
                    "length": field.length,
                    "domain": field.domain,
                }
                fc_dict["fields"].append(field_dict)

            featclasses_dict[fc_name] = fc_dict

        result["featclasses"] = featclasses_dict

    # Export relationships
    if schema.relationships:
        relationships_dict = {}
        for rel_name, rel in schema.relationships.items():
            rel_dict = {
                "origin": [rel.origin_table] if rel.origin_table else [],
                "destination": [rel.destination_table] if rel.destination_table else [],
                "forwardPathLabel": rel.forward_label,
                "backwardPathLabel": rel.backward_label,
                "isAttachmentRelationship": False,  # Default value
                "cardinality": rel.cardinality or rel.relationship_type,
                "originClassKeys": [],
                "destinationClassKeys": [],
                "is_attributed": rel.is_attributed,
            }

            # Build origin class keys
            if rel.origin_primary_key:
                rel_dict["originClassKeys"].append(
                    {
                        "objectKeyName": rel.origin_primary_key,
                        "keyRole": "OriginPrimary",
                        "classKeyName": "",
                        "datasetType": "RelationshipClassKey",
                    }
                )
            if rel.origin_foreign_key:
                rel_dict["originClassKeys"].append(
                    {
                        "objectKeyName": rel.origin_foreign_key,
                        "keyRole": "OriginForeign",
                        "classKeyName": "",
                        "datasetType": "RelationshipClassKey",
                    }
                )

            # Build destination class keys
            if rel.destination_primary_key:
                rel_dict["destinationClassKeys"].append(
                    {
                        "objectKeyName": rel.destination_primary_key,
                        "keyRole": "DestinationPrimary",
                        "classKeyName": "",
                        "datasetType": "RelationshipClassKey",
                    }
                )
            if rel.destination_foreign_key:
                rel_dict["destinationClassKeys"].append(
                    {
                        "objectKeyName": rel.destination_foreign_key,
                        "keyRole": "DestinationForeign",
                        "classKeyName": "",
                        "datasetType": "RelationshipClassKey",
                    }
                )

            relationships_dict[rel_name] = rel_dict

        result["relationships"] = relationships_dict

    # Export subtypes - simple format as in your example
    if schema.subtypes:
        subtypes_dict = {}

        # Check if we have the simple "ExtractedSubtypes" from your extract_subtypes
        if "ExtractedSubtypes" in schema.subtypes:
            extracted = schema.subtypes["ExtractedSubtypes"]
            for subtype_value in extracted.subtypes:
                code_str = str(subtype_value.code)
                subtypes_dict[code_str] = subtype_value.name
        else:
            # Handle other subtypes - flatten them into simple code:name format
            for subtype_name, subtype in schema.subtypes.items():
                for subtype_value in subtype.subtypes:
                    code_str = str(subtype_value.code)
                    subtypes_dict[code_str] = subtype_value.name

        if subtypes_dict:
            result["subtypes"] = subtypes_dict

    return result


from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from ..differ import SchemaDiff


def export_schema_diff_to_json(diff: SchemaDiff) -> Dict[str, Any]:
    """
    Export SchemaDiff instance to a JSON-serializable dictionary.

    Args:
        diff: SchemaDiff instance to export

    Returns:
        Dictionary ready to be serialized to JSON
    """

    def serialize_change(change):
        """Helper to serialize a change object"""
        result = {
            "change_type": change.change_type.value,
            "name": getattr(change, "domain_name", None)
            or getattr(change, "table_name", None)
            or getattr(change, "relationship_name", None)
            or getattr(change, "subtype_name", None)
            or getattr(change, "field_name", None),
        }

        # Add property changes if any
        if hasattr(change, "property_changes") and change.property_changes:
            result["property_changes"] = {
                prop: {"old": old, "new": new}
                for prop, (old, new) in change.property_changes.items()
            }

        # Add specific change type data
        if hasattr(change, "coded_value_changes") and change.coded_value_changes:
            result["coded_value_changes"] = {
                code: change_type.value
                for code, change_type in change.coded_value_changes.items()
            }

        if hasattr(change, "value_changes") and change.value_changes:
            result["value_changes"] = {
                code: change_type.value
                for code, change_type in change.value_changes.items()
            }

        if hasattr(change, "field_changes") and change.field_changes:
            result["field_changes"] = [
                serialize_change(fc) for fc in change.field_changes
            ]

        return result

    # Build the export structure
    export_data = {
        "summary": diff.get_summary(),
        "changes": {
            "domains": [serialize_change(c) for c in diff.domain_changes],
            "tables": [serialize_change(c) for c in diff.table_changes],
            "feature_classes": [
                serialize_change(c) for c in diff.feature_class_changes
            ],
            "relationships": [serialize_change(c) for c in diff.relationship_changes],
            "subtypes": [serialize_change(c) for c in diff.subtype_changes],
        },
        "has_changes": diff.has_changes(),
        "metadata": {
            "old_schema_name": diff.old_schema.metadata.get("name")
            if diff.old_schema.metadata
            else None,
            "new_schema_name": diff.new_schema.metadata.get("name")
            if diff.new_schema.metadata
            else None,
            "comparison_date": datetime.now().isoformat(),
        },
    }

    # Remove empty change lists
    export_data["changes"] = {k: v for k, v in export_data["changes"].items() if v}

    return export_data
