"""
Transformer for converting ESRI JSON schema reports to ESRISchema objects.
"""

import sys
import traceback
from typing import Any, Union

from loguru import logger

# Import the dataclasses from the previous artifact
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

logger.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
logger.add(sys.stderr, level="INFO")


def remove_prefix_from_name(
    text: Union[str, list[str]], prefix: str = "TOPGIS_GC."
) -> Union[str, list[str]]:
    """
    Remove a prefix from a string or list of strings while preserving the original type.

    Args:
        text: String or list of strings to process
        prefix: Prefix to remove (default: "TOPGIS_GC.")

    Returns:
        Cleaned string or list of strings with prefix removed
    """

    def _remove_single_prefix(s: str) -> str:
        return s[len(prefix) :] if s.startswith(prefix) else s

    if isinstance(text, str):
        return _remove_single_prefix(text)
    elif isinstance(text, list):
        cleaned = [_remove_single_prefix(s) for s in text]
        return cleaned[0] if len(cleaned) == 1 else cleaned
    else:
        return text


def transform_esri_json(
    input_data: dict[str, Any],
    filter_prefix: str = "TOPGIS_GC.GC_",
    exclude_metadata_fields: bool = True,
    remove_prefix: bool = False,
    add_prefix_to_gdb: bool = False,
) -> ESRISchema:
    """
    Transform ESRI JSON format into ESRISchema dataclass.
    Handles multiple section types: coded_domains, subtypes, relationships, featureclasses, tables

    Args:
        input_data: Dictionary containing ESRI JSON data
        filter_prefix: Only import items with this prefix (set to None to import all)
        exclude_metadata_fields: If True, exclude metadata fields like REVISION_MONTH, etc.
        remove_prefix: Remove prefix from table, relationship and feature class name

    Returns:
        ESRISchema instance with all parsed data
    """
    # Define metadata fields to exclude
    EXCLUDED_FIELDS = {
        "REVISION_MONTH",
        "CREATION_DAY",
        "REVISION_DAY",
        "CREATION_MONTH",
        "REVISION_YEAR",
        "CREATION_YEAR",
        "REVISION_DATE",
        "CREATION_DATE",
        "LAST_UPDATE",
        "CREATED_USER",
        "LAST_USER",
    }

    EXCLUDED_TABLES = {
        "TOPGIS_GC.GC_CONFLICT_POLYGON",
        "TOPGIS_GC.GC_ERRORS_LINE",
        "TOPGIS_GC.GC_ERRORS_ROW",
        "TOPGIS_GC.GC_CONFLICT_ROW",
        "TOPGIS_GC.GC_VERSION",
        "TOPGIS_GC.GC_ERRORS_MULTIPOINT",
        "TOPGIS_GC.GC_ERRORS_POLYGON",
        "TOPGIS_GC.GC_REVISIONSEBENE",
    }

    def should_import_item(name: str) -> bool:
        """Check if an item should be imported based on filters"""
        if not name:
            return False
        # Exclude items ending with _I
        if name.endswith("_I"):
            return False
        if name in EXCLUDED_TABLES:
            return False
        # If filter_prefix is set, only include items with that prefix
        if filter_prefix:
            return name.startswith(filter_prefix)
        return True

    def should_import_cd_item(name: str) -> bool:
        if not name:
            return False
        # Exclude items ending with _I
        if not name.startswith("GC_"):
            return False
        return True

    def filter_field_data(field_data: dict[str, Any]) -> bool:
        """Check if a field should be imported"""
        if exclude_metadata_fields and field_data.get("name") in EXCLUDED_FIELDS:
            return False
        return True

    def add_missing_prefix(name, prefix="TOPGIS_GC."):
        if not name.startswith(prefix):
            return f"{prefix}{name}"
        return name

    if remove_prefix and not (filter_prefix is None or filter_prefix == ""):
        raise Exception(
            f"With remove_prefix, you must set a filter_prefix: {filter_prefix}"
        )

    schema = ESRISchema()

    try:
        # Add metadata (unchanged)
        if any(
            key in input_data
            for key in ["dateExported", "datasetType", "catalogPath", "name"]
        ):
            schema.metadata = {
                "dateExported": input_data.get("dateExported"),
                "datasetType": input_data.get("datasetType"),
                "catalogPath": input_data.get("catalogPath"),
                "name": input_data.get("name"),
                "version": getattr(
                    globals().get("SCHEMA_VERSION"), "SCHEMA_VERSION", "1.0"
                ),
            }

        # Local GDB misses 'TOPGIS_GC.' prefix
        is_local_database = (
            input_data.get("workspaceType") == "esriLocalDatabaseWorkspace"
        )

        add_prefix_to_gdb = add_prefix_to_gdb and is_local_database
        logger.info(f"Is local database: {is_local_database}")
        logger.info(f"Add prefix to names: {add_prefix_to_gdb}")

        # Process domains - WITH FILTERING
        logger.info("Processing domains...")
        if "domains" in input_data:
            for domain in input_data["domains"]:
                domain_name = domain.get("name")
                if not domain_name or not should_import_cd_item(domain_name):
                    continue

                domain_type = domain.get("type")

                if domain_type == "codedValue":
                    # Create CodedDomain
                    coded_domain = CodedDomain(
                        name=domain_name,
                        description=domain.get("description"),
                        field_type=domain.get("fieldType"),
                    )

                    # Process coded values
                    if "codedValues" in domain:
                        for value_pair in domain["codedValues"]:
                            coded_value = CodedValue(
                                code=value_pair["code"], name=value_pair["name"]
                            )
                            coded_domain.coded_values.append(coded_value)

                    schema.coded_domains[domain_name] = coded_domain

                elif domain_type == "range":
                    # Create RangeDomain
                    range_domain = RangeDomain(
                        name=domain_name,
                        description=domain.get("description"),
                        field_type=domain.get("fieldType"),
                        min_value=domain.get("range", [None, None])[0],
                        max_value=domain.get("range", [None, None])[1],
                    )
                    schema.range_domains[domain_name] = range_domain

        # Extract and process all components
        tables_list = []
        featclasses_list = []
        relationships_list = []
        subtypes_list = []

        logger.info("Extracting datasets...")
        # Pass filter function to extract_datasets
        extract_datasets_filtered(
            input_data,
            tables_list,
            featclasses_list,
            relationships_list,
            subtypes_list,
            should_import_item,
        )

        # Process tables - WITH FIELD FILTERING
        logger.info("Processing tables...")
        for table_item in tables_list:
            if isinstance(table_item, dict):
                for table_name, table_data in table_item.items():
                    if add_prefix_to_gdb:  # TODO
                        table_name = add_missing_prefix(table_name)
                        logger.info(f"New table name: {table_name}")
                    if not should_import_item(table_name):
                        continue

                    # Filter fields before creating table
                    if isinstance(table_data, list):
                        filtered_fields = [
                            field for field in table_data if filter_field_data(field)
                        ]
                        table = create_table_from_data(table_name, filtered_fields)
                    else:
                        # Handle dict format
                        if "fields" in table_data:
                            table_data["fields"] = [
                                field
                                for field in table_data["fields"]
                                if filter_field_data(field)
                            ]
                        table = create_table_from_data(table_name, table_data)

                    schema.tables[table_name] = table

        # Process feature classes - WITH FIELD FILTERING
        logger.info("Processing feature classes...")
        for featclass_item in featclasses_list:
            if isinstance(featclass_item, dict):
                for fc_name, fc_data in featclass_item.items():
                    if add_prefix_to_gdb:  # TODO
                        fc_name = add_missing_prefix(fc_name)
                        logger.info(f"New Feature class name: {fc_name}")
                    if not should_import_item(fc_name):
                        continue

                    # Filter fields before creating feature class
                    if "fields" in fc_data and isinstance(fc_data["fields"], list):
                        fc_data["fields"] = [
                            field
                            for field in fc_data["fields"]
                            if filter_field_data(field)
                        ]

                    feature_class = create_feature_class_from_data(fc_name, fc_data)
                    schema.feature_classes[fc_name] = feature_class

        # Process relationships - WITH FILTERING
        logger.info("Processing relationships...")
        for rel_data in relationships_list:
            if isinstance(rel_data, dict) and "name" in rel_data:
                rel_name = rel_data.pop("name")
                if add_prefix_to_gdb:  # TODO  shold also modificy the classes
                    fc_name = add_missing_prefix(fc_name)
                    logger.info(f"New relationship name: {fc_name}")
                if not should_import_item(rel_name):
                    continue

                # Also check if origin and destination should be imported
                origin_tables = rel_data.get("origin", [])
                destination_tables = rel_data.get("destination", [])

                logger.info(f"Origin tables: {origin_tables}")

                # Skip if origin or destination tables are filtered out
                if origin_tables and not should_import_item(origin_tables[0]):
                    continue
                if destination_tables and not should_import_item(destination_tables[0]):
                    continue

                relationship = create_relationship_from_data(rel_name, rel_data)
                schema.relationships[rel_name] = relationship

        # Process subtypes
        logger.info("Processing subtypes...")
        subtypes_dict = {}
        extract_subtypes(input_data, subtypes_dict)

        # Create a simple Subtype object from the flat dictionary
        # Since your extract_subtypes only gets code->name mappings,
        # we'll create a generic subtype container
        if subtypes_dict:
            # Create a generic subtype container
            generic_subtype = Subtype(
                name="ExtractedSubtypes",
                subtype_field="SUBTYPE",  # Default field name, update if known
                subtypes=[
                    SubtypeValue(code=code, name=name)
                    for code, name in subtypes_dict.items()
                ],
            )
            schema.subtypes["ExtractedSubtypes"] = generic_subtype

        # Also look for more detailed subtype information in feature classes/tables
        process_detailed_subtypes(input_data, schema)

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error transforming ESRI JSON: {e}")
        logger.error(error_details)
        raise

    return schema


def create_field_from_data(field_data: dict[str, Any]) -> Field:
    """Create a Field instance from field data dictionary"""
    return Field(
        name=field_data.get("name", ""),
        alias=field_data.get("alias"),
        type=field_data.get("type"),
        length=field_data.get("length"),
        precision=field_data.get("precision"),
        scale=field_data.get("scale"),
        nullable=field_data.get("nullable", True),
        default_value=field_data.get("defaultValue"),
        domain=field_data.get("domain"),
    )


def create_table_from_data(
    table_name: str, table_data: Union[dict[str, Any], list[dict[str, Any]]]
) -> Table:
    """Create a Table instance from table data dictionary"""
    table = Table(
        name=table_name,
        alias=table_data.get("alias") if isinstance(table_data, dict) else None,
        object_id_field=table_data.get("objectIdField")
        if isinstance(table_data, dict)
        else None,
        global_id_field=table_data.get("globalIdField")
        if isinstance(table_data, dict)
        else None,
        subtype_field=table_data.get("subtypeField")
        if isinstance(table_data, dict)
        else None,
        default_subtype=table_data.get("defaultSubtype")
        if isinstance(table_data, dict)
        else None,
    )

    # Use object_id_field if set
    if table.object_id_field and not table.primary_key:
        table.primary_key = table.object_id_field

    # Process fields - handle both your format and standard format
    fields_data = None
    if isinstance(table_data, list):
        # Your format: table_data is the field list directly
        fields_data = table_data
    elif isinstance(table_data, dict) and "fields" in table_data:
        fields_data = table_data["fields"]

    if fields_data:
        for field_data in fields_data:
            if isinstance(field_data, dict):
                field = create_field_from_data(field_data)
                table.fields.append(field)

    # Auto-detect primary key
    # ESRI doesn't set explicitely PK
    if not table.primary_key:
        for field in table.fields:
            # Check for OBJECTID field
            if field.type in ["OID", "esriFieldTypeOID"]:
                table.primary_key = field.name
                break
            # Check for GlobalID/UUID as fallback
            elif field.name.upper() in ["GLOBALID", "UUID"] and field.type in [
                "GUID",
                "esriFieldTypeGUID",
            ]:
                table.primary_key = field.name
                break
            # Check if field name is OBJECTID (sometimes type might be different)
            elif field.name.upper() == "OBJECTID":
                table.primary_key = field.name
                break

    # Process indexes if available
    if isinstance(table_data, dict) and "indexes" in table_data:
        for index_data in table_data["indexes"]:
            if isinstance(index_data, dict):
                index = Index(
                    name=index_data.get("name", ""),
                    fields=index_data.get("fields", []),
                    is_unique=index_data.get("isUnique", False),
                    is_ascending=index_data.get("isAscending", True),
                )
                table.indexes.append(index)

    return table


def create_feature_class_from_data(
    fc_name: str, fc_data: dict[str, Any]
) -> FeatureClass:
    """Create a FeatureClass instance from feature class data dictionary"""
    fc = FeatureClass(
        name=fc_name,
        alias=fc_data.get("alias"),
        feature_type=fc_data.get("featureType"),
        geometry_type=fc_data.get("geometryType"),
        spatial_reference=fc_data.get("spatialReference"),
        has_z=fc_data.get("hasZ", False),
        has_m=fc_data.get("hasM", False),
        object_id_field=fc_data.get("objectIdField"),
        global_id_field=fc_data.get("globalIdField"),
        shape_field=fc_data.get("shapeField"),
        area_field=fc_data.get("areaField"),
        length_field=fc_data.get("lengthField"),
        subtype_field=fc_data.get("subtypeField"),
        default_subtype=fc_data.get("defaultSubtype"),
    )

    # Process fields - handle your format
    if "fields" in fc_data:
        fields_data = fc_data["fields"]
        if isinstance(fields_data, list):
            for field_data in fields_data:
                if isinstance(field_data, dict):
                    field = create_field_from_data(field_data)
                    fc.fields.append(field)

    # Process indexes if available
    if "indexes" in fc_data:
        for index_data in fc_data["indexes"]:
            if isinstance(index_data, dict):
                index = Index(
                    name=index_data.get("name", ""),
                    fields=index_data.get("fields", []),
                    is_unique=index_data.get("isUnique", False),
                    is_ascending=index_data.get("isAscending", True),
                )
                fc.indexes.append(index)

    return fc


def create_relationship_from_data(
    rel_name: str, rel_data: dict[str, Any]
) -> RelationshipClass:
    """Create a RelationshipClass instance from relationship data dictionary"""
    # Handle origin and destination - they come as lists in your format
    origin_tables = rel_data.get("origin", [])
    destination_tables = rel_data.get("destination", [])

    origin_table = origin_tables[0] if origin_tables else ""
    destination_table = destination_tables[0] if destination_tables else ""

    # Map cardinality values
    cardinality_map = {
        "esriRelCardinalityOneToMany": "OneToMany",
        "esriRelCardinalityManyToMany": "ManyToMany",
        "esriRelCardinalityOneToOne": "OneToOne",
    }
    cardinality = rel_data.get("cardinality", "")
    if cardinality in cardinality_map:
        cardinality = cardinality_map[cardinality]

    # Extract key information
    origin_keys = rel_data.get("originClassKeys", [])
    dest_keys = rel_data.get("destinationClassKeys", [])

    origin_primary_key = None
    origin_foreign_key = None
    dest_primary_key = None
    dest_foreign_key = None

    # Parse keys based on keyRole
    for key in origin_keys:
        if key.get("keyRole") == "Primary":
            origin_primary_key = key.get("objectKeyName")
        elif key.get("keyRole") == "Foreign":
            origin_foreign_key = key.get("objectKeyName")

    for key in dest_keys:
        if key.get("keyRole") == "Primary":
            dest_primary_key = key.get("objectKeyName")
        elif key.get("keyRole") == "Foreign":
            dest_foreign_key = key.get("objectKeyName")

    return RelationshipClass(
        name=rel_name,
        origin_table=origin_table,
        destination_table=destination_table,
        relationship_type=cardinality,
        forward_label=rel_data.get("forwardPathLabel"),
        backward_label=rel_data.get("backwardPathLabel"),
        cardinality=cardinality,
        is_composite=rel_data.get("isComposite", False),
        is_attributed=rel_data.get("is_attributed", False),
        origin_primary_key=origin_primary_key,
        origin_foreign_key=origin_foreign_key,
        destination_primary_key=dest_primary_key,
        destination_foreign_key=dest_foreign_key,
    )


def create_subtype_from_data(
    subtype_name: str, subtype_data: dict[str, Any]
) -> Subtype:
    """Create a Subtype instance from subtype data dictionary"""
    subtype = Subtype(
        name=subtype_name,
        subtype_field=subtype_data.get("subtypeField", ""),
        default_subtype=subtype_data.get("defaultSubtype"),
    )

    # Process subtype values
    if "subtypes" in subtype_data:
        for code, name in subtype_data["subtypes"].items():
            # Handle both dictionary and list formats
            if isinstance(name, str):
                subtype_value = SubtypeValue(code=code, name=name)
                subtype.subtypes.append(subtype_value)
            elif isinstance(name, dict) and "name" in name:
                subtype_value = SubtypeValue(code=code, name=name["name"])
                subtype.subtypes.append(subtype_value)

    # Process field-level domain assignments
    if "fieldDomains" in subtype_data:
        for subtype_code, field_domains in subtype_data["fieldDomains"].items():
            if isinstance(field_domains, dict):
                subtype.field_domains[subtype_code] = field_domains

    return subtype


def convert_field_type(esri_type):
    """
    Convert ESRI field type to a standardized type.
    Implement this based on your needs.
    """
    # Add your field type conversion logic here
    return esri_type


def get_field_list(data):
    field_list = []

    for field in data["fields"]["fieldArray"]:
        field_info = {
            "name": field.get("name"),
            "type": convert_field_type(field.get("type")),
            "length": field.get("length"),
            "domain": field.get("domain").get("domainName")
            if "domain" in field
            else None,
        }
        field_list.append(field_info)
    return field_list


def extract_datasets_filtered(
    data,
    tables_list,
    featclasses_list,
    relationships_list,
    subtypes_list,
    should_import_func,
    parent_path="",
):
    """
    Modified version of extract_datasets that applies filtering during extraction
    """
    if isinstance(data, dict):
        # Check if this is a table with fields information
        if (
            "name" in data
            and "datasetType" in data
            and data.get("datasetType") == "esriDTTable"
        ):
            table_name = data.get("name")

            if not table_name.endswith("_I") and should_import_func(table_name):
                # Create basic table info
                table_info = {
                    "name": table_name,
                }

                # Add parent path information if available
                if parent_path:
                    table_info["path"] = f"{parent_path}/{table_name}"

                # Extract field information if available
                if "fields" in data and "fieldArray" in data["fields"]:
                    field_list = get_field_list(data)

                    # Store the table with its fields
                    tables_list.append({table_name: field_list})
                else:
                    # If no fields available, just add the basic info
                    tables_list.append(table_info)

        # Handle feature datasets
        elif (
            "name" in data
            and "datasetType" in data
            and data.get("datasetType") == "esriDTFeatureClass"
        ):
            feat_name = data.get("name")

            if not feat_name.endswith("_I"):
                dataset_info = {
                    "name": feat_name,
                }

                if parent_path:
                    dataset_info["path"] = f"{parent_path}/{data.get('name')}"

                # Extract field information if available
                if "fields" in data and "fieldArray" in data["fields"]:
                    field_list = get_field_list(data)
                else:
                    # If no fields available, just add the basic info
                    field_list = []
                featclasses_list.append({feat_name: {"fields": field_list}})

        # Handle relationship classes
        elif (
            "name" in data
            and "datasetType" in data
            and data.get("datasetType") == "esriDTRelationshipClass"
        ):
            rel_name = data.get("name")

            if not rel_name.endswith("_I"):
                # TODO: simplify or change datamodel code
                transform_keys = (
                    lambda item: {
                        "objectKeyName": item["objectKeyName"],
                        "keyRole": item["keyRole"].replace("esriRelKeyRole", ""),
                        "classKeyName": item["classKeyName"],
                        "datasetType": item["datasetType"],
                    }
                    if item
                    else None
                )
                try:
                    originKeys = (
                        list(
                            filter(
                                None, map(transform_keys, data.get("originClassKeys"))
                            )
                        )
                        if data.get("originClassKeys")
                        else []
                    )
                    destinationKeys = (
                        list(
                            filter(
                                None,
                                map(transform_keys, data.get("destinationClassKeys")),
                            )
                        )
                        if data.get("destinationClassKeys")
                        else []
                    )

                    logger.debug(
                        f"originKeys: {originKeys}, destinationKeys: {destinationKeys}"
                    )

                except Exception as e:
                    logger.error(f"Error processing relationship keys: {e}")
                    originKeys = []
                    destinationKeys = []

                rel_info = {
                    "name": data.get("name"),
                    "origin": [item["name"] for item in data.get("originClassNames")]
                    if data.get("originClassNames")
                    else [],
                    "destination": [
                        item["name"] for item in data.get("destinationClassNames")
                    ]
                    if data.get("destinationClassNames")
                    else [],
                    "forwardPathLabel": data.get("forwardPathLabel"),
                    "backwardPathLabel": data.get("backwardPathLabel"),
                    "isAttachmentRelationship": data.get("isAttachmentRelationship"),
                    "cardinality": data.get("cardinality"),
                    "originClassKeys": originKeys,
                    "destinationClassKeys": destinationKeys,
                    "is_attributed": data.get("is_attributed"),
                }

                if parent_path:
                    rel_info["path"] = f"{parent_path}/{data.get('name')}"

                relationships_list.append(rel_info)

        # Check for nested datasets
        if "datasets" in data and isinstance(data.get("datasets"), list):
            current_path = parent_path
            if "name" in data:
                current_path = (
                    f"{parent_path}/{data.get('name')}"
                    if parent_path
                    else data.get("name")
                )

            # Process each nested dataset
            for dataset in data.get("datasets"):
                """extract_datasets(
                    dataset,
                    tables_list,
                    featclasses_list,
                    relationships_list,
                    subtypes_list,
                    current_path,
                )"""

                extract_datasets_filtered(
                    dataset,
                    tables_list,
                    featclasses_list,
                    relationships_list,
                    subtypes_list,
                    should_import_func,
                    parent_path=current_path,
                )

        # Continue searching in other keys
        for key, value in data.items():
            if key != "datasets":  # Already processed above
                """extract_datasets(
                    value,
                    tables_list,
                    featclasses_list,
                    relationships_list,
                    subtypes_list,
                    parent_path,
                )"""
                extract_datasets_filtered(
                    value,
                    tables_list,
                    featclasses_list,
                    relationships_list,
                    subtypes_list,
                    should_import_func,
                    parent_path=parent_path,
                )

    elif isinstance(data, list):
        for item in data:
            """extract_datasets(
                item,
                tables_list,
                featclasses_list,
                relationships_list,
                subtypes_list,
                parent_path,
            )"""
            extract_datasets_filtered(
                item,
                tables_list,
                featclasses_list,
                relationships_list,
                subtypes_list,
                should_import_func,
                parent_path=parent_path,
            )


def extract_subtypes(data, subtypes_dict):
    """
    Recursively search for 'subtypes' key in the JSON structure and extract the data.
    Converts it to a simple dictionary with subtypeCode as key and subtypeName as value.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "subtypes" and isinstance(value, list):
                # Process the subtypes list
                for subtype in value:
                    if "subtypeCode" in subtype and "subtypeName" in subtype:
                        # Convert code to string if it's numeric
                        code = (
                            str(subtype["subtypeCode"])
                            if isinstance(subtype["subtypeCode"], (int, float))
                            else subtype["subtypeCode"]
                        )
                        subtypes_dict[code] = subtype["subtypeName"]
            else:
                # Continue searching deeper in the structure
                extract_subtypes(value, subtypes_dict)
    elif isinstance(data, list):
        for item in data:
            extract_subtypes(item, subtypes_dict)


def process_detailed_subtypes(data, schema: ESRISchema, parent_path=""):
    """
    Recursively search for detailed subtype information in feature classes and tables.
    This looks for subtype field definitions and field-level domain assignments.
    """
    if isinstance(data, dict):
        # Check if this is a feature class or table with subtype info
        if "subtypeField" in data or "subtypes" in data:
            parent_name = data.get("name") or parent_path.split(".")[-1]

            if "subtypeField" in data and "subtypes" in data:
                subtype = Subtype(
                    name=f"{parent_name}_Subtypes",
                    subtype_field=data["subtypeField"],
                    default_subtype=data.get("defaultSubtype"),
                )

                # Process subtypes
                if isinstance(data["subtypes"], list):
                    for st in data["subtypes"]:
                        if (
                            isinstance(st, dict)
                            and "subtypeCode" in st
                            and "subtypeName" in st
                        ):
                            subtype.subtypes.append(
                                SubtypeValue(
                                    code=st["subtypeCode"], name=st["subtypeName"]
                                )
                            )

                            # Check for field-level domains
                            if "fieldDomains" in st:
                                subtype.field_domains[st["subtypeCode"]] = st[
                                    "fieldDomains"
                                ]

                # Store the detailed subtype
                schema.subtypes[subtype.name] = subtype

                # Link to parent if we can find it
                parent = schema.get_feature_class_or_table(parent_name)
                if parent:
                    parent.subtypes = subtype
                    parent.subtype_field = subtype.subtype_field
                    parent.default_subtype = subtype.default_subtype

        # Continue searching recursively
        for key, value in data.items():
            new_path = f"{parent_path}.{key}" if parent_path else key
            process_detailed_subtypes(value, schema, new_path)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{parent_path}[{i}]"
            process_detailed_subtypes(item, schema, new_path)


def main():
    # Example usage:
    import json
    import os

    if os.name == "nt":
        BASE_DIR = r"H:\code\lg-geology-data-model\exports"
    else:
        BASE_DIR = "/home/marco/code/github.com/lg-geology-data-model/exports/"
    old_json_path = os.path.join(BASE_DIR, "2025_05_26/GCOVERP.json")

    with open(old_json_path) as f:
        esri_json_data = json.loads(f.read())
    schema = transform_esri_json(esri_json_data, exclude_metadata_fields=False)
    logger.info(f"Loaded {len(schema.feature_classes)} feature classes")
    logger.info(f"Loaded {len(schema.coded_domains)} coded domains")
    logger.info(f"Total subtypes: {schema.get_subtype_count()}")
    logger.info(f"Summary: {schema.get_subtype_summary()}")
    logger.info(f"Loaded {len(schema.tables)} tables")

    logger.info(schema.get_schema_summary())

    for feat_class in schema.feature_classes:
        logger.info(feat_class)
    # logger.info(schema.get_relationships_for_class("TOPGIS_GC.GC_BEDROCK"))

    validation_errors = schema.validate_domain_references()
    if validation_errors:
        logger.error(f"Validation errors: {validation_errors}")

    from export_esri_schema import export_esri_schema_to_json

    json_data = export_esri_schema_to_json(schema)

    # Write UTF-8 file (works on both Linux and Windows)
    json_output_path = "output.json"
    logger.info(f"File written to: {json_output_path}")
    with open(json_output_path, "w", encoding="utf-8", newline="") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
