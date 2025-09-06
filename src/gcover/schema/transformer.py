"""
Robust transformer for converting ESRI JSON schema reports to ESRISchema objects.
This is a rewritten version that systematically finds all datasets and applies
consistent filtering logic.
"""

import sys
import traceback
from typing import Any, Union, List, Dict, Set

from loguru import logger

from gcover.config import EXCLUDED_TABLES, DEFAULT_EXCLUDED_FIELDS

# Import the dataclasses
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

logger.remove()
logger.add(sys.stderr, level="INFO")


def normalize_table_name(name: str, default_prefix: str = "TOPGIS_GC.") -> str:
    """
    Normalize a table name by ensuring consistent handling of prefixes.

    Args:
        name: Original table name
        default_prefix: Default prefix to add if missing

    Returns:
        Normalized table name
    """
    if not name:
        return name

    # If name already has a prefix (contains a dot), return as-is
    if "." in name:
        return name

    # If name starts with GC_, add the default prefix
    if name.startswith("GC_"):
        return f"{default_prefix}{name}"

    return name


def get_base_table_name(name: str) -> str:
    """
    Get the base table name without any prefix.

    Args:
        name: Full table name (may include prefix)

    Returns:
        Base table name without prefix
    """
    if "." in name:
        return name.split(".", 1)[1]
    return name


def should_import_table(
    name: str, target_prefix: str = "GC_", excluded_tables: Set[str] = None
) -> bool:
    """
    Determine if a table/feature class should be imported based on robust filtering rules.

    Args:
        name: Table name (may be prefixed or not)
        target_prefix: Prefix that we want to include (e.g., "GC_")
        excluded_tables: Set of table names to exclude (can be prefixed or not)

    Returns:
        True if table should be imported
    """
    if not name:
        return False

    # Get the base name (without prefix)
    base_name = get_base_table_name(name)

    # Check exclusions against both full name and base name
    if excluded_tables:
        if name in excluded_tables or base_name in excluded_tables:
            return False

    # Exclude items ending with _I
    if base_name.endswith("_I"):
        return False

    # Only include items that start with target prefix (after removing schema prefix)
    if target_prefix:
        return base_name.startswith(target_prefix)

    return True


def extract_all_datasets(data: Any, path: str = "") -> Dict[str, List[Dict[str, Any]]]:
    """
    Systematically extract all datasets from ESRI JSON structure.
    Returns organized dictionary with separate lists for each dataset type.

    Args:
        data: JSON data structure
        path: Current path in JSON (for debugging)

    Returns:
        Dictionary with keys: 'tables', 'feature_classes', 'relationships'
    """
    results = {"tables": [], "feature_classes": [], "relationships": []}

    def process_item(item: Dict[str, Any], current_path: str):
        """Process a single item that might be a dataset."""
        if not isinstance(item, dict):
            return

        dataset_type = item.get("datasetType")
        name = item.get("name")

        if not name or not dataset_type:
            return

        # Add path information
        item_with_path = dict(item)
        item_with_path["_extraction_path"] = current_path

        if dataset_type == "esriDTTable":
            logger.debug(f"Found table: {name} at {current_path}")
            results["tables"].append(item_with_path)

        elif dataset_type == "esriDTFeatureClass":
            logger.debug(f"Found feature class: {name} at {current_path}")
            results["feature_classes"].append(item_with_path)

        elif dataset_type == "esriDTRelationshipClass":
            logger.debug(f"Found relationship: {name} at {current_path}")
            results["relationships"].append(item_with_path)

    def traverse(obj: Any, current_path: str):
        """Recursively traverse the JSON structure."""
        if isinstance(obj, dict):
            # Check if this dict represents a dataset
            process_item(obj, current_path)

            # Continue traversing all dictionary values
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                traverse(value, new_path)

        elif isinstance(obj, list):
            # Traverse all list items
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]"
                traverse(item, new_path)

    # Start traversal
    traverse(data, path)

    logger.info(
        f"Extraction summary: {len(results['tables'])} tables, "
        f"{len(results['feature_classes'])} feature classes, "
        f"{len(results['relationships'])} relationships"
    )

    return results


def create_field_from_esri_data(field_data: Dict[str, Any]) -> Field:
    """Create a Field instance from ESRI field data."""
    return Field(
        name=field_data.get("name", ""),
        alias=field_data.get("alias"),
        type=field_data.get("type"),
        length=field_data.get("length"),
        precision=field_data.get("precision"),
        scale=field_data.get("scale"),
        nullable=field_data.get("nullable", True),
        default_value=field_data.get("defaultValue"),
        domain=field_data.get("domain", {}).get("domainName")
        if "domain" in field_data
        else None,
    )


def extract_fields_from_esri_data(dataset_data: Dict[str, Any]) -> List[Field]:
    """Extract fields from ESRI dataset data structure."""
    fields = []

    # Handle different possible field structures
    if "fields" in dataset_data:
        fields_data = dataset_data["fields"]

        # Handle fieldArray structure
        if isinstance(fields_data, dict) and "fieldArray" in fields_data:
            field_array = fields_data["fieldArray"]
            if isinstance(field_array, list):
                for field_data in field_array:
                    if isinstance(field_data, dict):
                        field = create_field_from_esri_data(field_data)
                        fields.append(field)

        # Handle direct list of fields
        elif isinstance(fields_data, list):
            for field_data in fields_data:
                if isinstance(field_data, dict):
                    field = create_field_from_esri_data(field_data)
                    fields.append(field)

    return fields


def create_table_from_esri_data(dataset_data: Dict[str, Any]) -> Table:
    """Create a Table instance from ESRI dataset data."""
    name = dataset_data.get("name", "")

    table = Table(
        name=name,
        alias=dataset_data.get("alias"),
        object_id_field=dataset_data.get("oidFieldName"),
        global_id_field=dataset_data.get("globalIdFieldName"),
    )

    # Extract fields
    table.fields = extract_fields_from_esri_data(dataset_data)

    # Auto-detect primary key
    if not table.primary_key:
        for field in table.fields:
            if field.type in ["OID", "esriFieldTypeOID"]:
                table.primary_key = field.name
                break
            elif field.name.upper() == "OBJECTID":
                table.primary_key = field.name
                break

    return table


def create_feature_class_from_esri_data(dataset_data: Dict[str, Any]) -> FeatureClass:
    """Create a FeatureClass instance from ESRI dataset data."""
    name = dataset_data.get("name", "")

    fc = FeatureClass(
        name=name,
        alias=dataset_data.get("alias"),
        geometry_type=dataset_data.get("geometryType"),
        has_z=dataset_data.get("hasZ", False),
        has_m=dataset_data.get("hasM", False),
        object_id_field=dataset_data.get("oidFieldName"),
        global_id_field=dataset_data.get("globalIdFieldName"),
        shape_field=dataset_data.get("shapeFieldName"),
        spatial_reference=dataset_data.get("spatialReference"),
    )

    # Extract fields
    fc.fields = extract_fields_from_esri_data(dataset_data)

    # Auto-detect primary key
    if not fc.primary_key:
        for field in fc.fields:
            if field.type in ["OID", "esriFieldTypeOID"]:
                fc.primary_key = field.name
                break
            elif field.name.upper() == "OBJECTID":
                fc.primary_key = field.name
                break

    return fc


def create_relationship_from_esri_data(rel_data: Dict[str, Any]) -> RelationshipClass:
    """Create a RelationshipClass instance from ESRI relationship data."""
    name = rel_data.get("name", "")

    # Extract origin and destination table names
    origin_tables = []
    dest_tables = []

    if "originClassNames" in rel_data:
        origin_tables = [
            item.get("name", "")
            for item in rel_data["originClassNames"]
            if isinstance(item, dict)
        ]

    if "destinationClassNames" in rel_data:
        dest_tables = [
            item.get("name", "")
            for item in rel_data["destinationClassNames"]
            if isinstance(item, dict)
        ]

    origin_table = origin_tables[0] if origin_tables else ""
    destination_table = dest_tables[0] if dest_tables else ""

    # Map cardinality
    cardinality_map = {
        "esriRelCardinalityOneToMany": "OneToMany",
        "esriRelCardinalityManyToMany": "ManyToMany",
        "esriRelCardinalityOneToOne": "OneToOne",
    }
    cardinality = rel_data.get("cardinality", "")
    if cardinality in cardinality_map:
        cardinality = cardinality_map[cardinality]

    # Extract key information
    origin_primary_key = None
    origin_foreign_key = None
    dest_primary_key = None
    dest_foreign_key = None

    # Process origin keys
    if "originClassKeys" in rel_data:
        for key in rel_data["originClassKeys"]:
            if isinstance(key, dict):
                key_role = key.get("keyRole", "").replace("esriRelKeyRole", "")
                if key_role == "Primary":
                    origin_primary_key = key.get("objectKeyName")
                elif key_role == "Foreign":
                    origin_foreign_key = key.get("objectKeyName")

    # Process destination keys
    if "destinationClassKeys" in rel_data:
        for key in rel_data["destinationClassKeys"]:
            if isinstance(key, dict):
                key_role = key.get("keyRole", "").replace("esriRelKeyRole", "")
                if key_role == "Primary":
                    dest_primary_key = key.get("objectKeyName")
                elif key_role == "Foreign":
                    dest_foreign_key = key.get("objectKeyName")

    return RelationshipClass(
        name=name,
        origin_table=origin_table,
        destination_table=destination_table,
        relationship_type=cardinality,
        forward_label=rel_data.get("forwardPathLabel"),
        backward_label=rel_data.get("backwardPathLabel"),
        cardinality=cardinality,
        is_composite=rel_data.get("isComposite", False),
        is_attributed=rel_data.get("isAttachmentRelationship", False),
        origin_primary_key=origin_primary_key,
        origin_foreign_key=origin_foreign_key,
        destination_primary_key=dest_primary_key,
        destination_foreign_key=dest_foreign_key,
    )


def extract_domains_from_esri_data(
    input_data: Dict[str, Any], target_prefix: str = "GC_"
) -> tuple[Dict[str, CodedDomain], Dict[str, RangeDomain]]:
    """Extract domains from ESRI JSON data."""
    coded_domains = {}
    range_domains = {}

    if "domains" not in input_data:
        return coded_domains, range_domains

    for domain_data in input_data["domains"]:
        domain_name = domain_data.get("name", "")

        # Apply filtering to domains too
        if target_prefix and not get_base_table_name(domain_name).startswith(
            target_prefix
        ):
            continue

        domain_type = domain_data.get("type")

        if domain_type == "codedValue":
            coded_domain = CodedDomain(
                name=domain_name,
                description=domain_data.get("description"),
                field_type=domain_data.get("fieldType"),
            )

            # Process coded values
            if "codedValues" in domain_data:
                for value_pair in domain_data["codedValues"]:
                    coded_value = CodedValue(
                        code=value_pair.get("code"), name=value_pair.get("name", "")
                    )
                    coded_domain.coded_values.append(coded_value)

            coded_domains[domain_name] = coded_domain

        elif domain_type == "range":
            range_domain = RangeDomain(
                name=domain_name,
                description=domain_data.get("description"),
                field_type=domain_data.get("fieldType"),
                min_value=domain_data.get("range", [None, None])[0],
                max_value=domain_data.get("range", [None, None])[1],
            )
            range_domains[domain_name] = range_domain

    return coded_domains, range_domains


def extract_subtypes_from_esri_data(input_data: Dict[str, Any]) -> Dict[str, Subtype]:
    """Extract subtypes from ESRI JSON data."""
    subtypes = {}

    def find_subtypes(obj: Any, path: str = ""):
        """Recursively find subtype definitions."""
        if isinstance(obj, dict):
            # Look for subtype arrays
            if "subtypes" in obj and isinstance(obj["subtypes"], list):
                parent_name = obj.get(
                    "name", path.split(".")[-1] if path else "Unknown"
                )

                subtype = Subtype(
                    name=f"{parent_name}_Subtypes",
                    subtype_field=obj.get("subtypeField", "SUBTYPE"),
                    default_subtype=obj.get("defaultSubtype"),
                )

                for st_data in obj["subtypes"]:
                    if isinstance(st_data, dict):
                        subtype_value = SubtypeValue(
                            code=st_data.get("subtypeCode"),
                            name=st_data.get("subtypeName", ""),
                        )
                        subtype.subtypes.append(subtype_value)

                if subtype.subtypes:
                    subtypes[subtype.name] = subtype

            # Continue searching
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                find_subtypes(value, new_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                find_subtypes(item, new_path)

    find_subtypes(input_data)
    return subtypes


def transform_esri_json(
    input_data: Dict[str, Any],
    target_prefix: str = "GC_",
    excluded_tables: Set[str] = None,
    exclude_metadata_fields: bool = True,
    default_schema_prefix: str = "TOPGIS_GC.",
) -> ESRISchema:
    """
    Robust transformer for ESRI JSON format into ESRISchema dataclass.

    Args:
        input_data: Dictionary containing ESRI JSON data
        target_prefix: Only import items with this base prefix (e.g., "GC_")
        excluded_tables: Set of table names to exclude
        exclude_metadata_fields: If True, exclude metadata fields
        default_schema_prefix: Default schema prefix to add to non-prefixed names

    Returns:
        ESRISchema instance with all parsed data
    """
    if excluded_tables is None:
        excluded_tables = EXCLUDED_TABLES

    # Define metadata fields to exclude
    EXCLUDED_FIELDS = DEFAULT_EXCLUDED_FIELDS if exclude_metadata_fields else set()

    schema = ESRISchema()

    try:
        # Add metadata
        schema.metadata = {
            "dateExported": input_data.get("dateExported"),
            "datasetType": input_data.get("datasetType"),
            "catalogPath": input_data.get("catalogPath"),
            "name": input_data.get("name"),
            "workspaceType": input_data.get("workspaceType"),
        }

        # Extract domains
        logger.info("Extracting domains...")
        coded_domains, range_domains = extract_domains_from_esri_data(
            input_data, target_prefix
        )
        schema.coded_domains = coded_domains
        schema.range_domains = range_domains
        logger.info(
            f"Found {len(coded_domains)} coded domains, {len(range_domains)} range domains"
        )

        # Extract all datasets systematically
        logger.info("Extracting datasets...")
        all_datasets = extract_all_datasets(input_data)

        # Process tables
        logger.info("Processing tables...")
        tables_processed = 0
        for table_data in all_datasets["tables"]:
            table_name = table_data.get("name", "")

            if should_import_table(table_name, target_prefix, excluded_tables):
                try:
                    table = create_table_from_esri_data(table_data)

                    # Filter fields if needed
                    if EXCLUDED_FIELDS:
                        table.fields = [
                            f for f in table.fields if f.name not in EXCLUDED_FIELDS
                        ]

                    schema.tables[table_name] = table
                    tables_processed += 1
                    logger.debug(f"Processed table: {table_name}")
                except Exception as e:
                    logger.error(f"Error processing table {table_name}: {e}")

        logger.info(f"Processed {tables_processed} tables")

        # Process feature classes
        logger.info("Processing feature classes...")
        fc_processed = 0
        for fc_data in all_datasets["feature_classes"]:
            fc_name = fc_data.get("name", "")

            if should_import_table(fc_name, target_prefix, excluded_tables):
                try:
                    feature_class = create_feature_class_from_esri_data(fc_data)

                    # Filter fields if needed
                    if EXCLUDED_FIELDS:
                        feature_class.fields = [
                            f
                            for f in feature_class.fields
                            if f.name not in EXCLUDED_FIELDS
                        ]

                    schema.feature_classes[fc_name] = feature_class
                    fc_processed += 1
                    logger.debug(f"Processed feature class: {fc_name}")
                except Exception as e:
                    logger.error(f"Error processing feature class {fc_name}: {e}")

        logger.info(f"Processed {fc_processed} feature classes")

        # Process relationships
        logger.info("Processing relationships...")
        rel_processed = 0
        for rel_data in all_datasets["relationships"]:
            rel_name = rel_data.get("name", "")

            if should_import_table(rel_name, target_prefix, excluded_tables):
                try:
                    relationship = create_relationship_from_esri_data(rel_data)
                    schema.relationships[rel_name] = relationship
                    rel_processed += 1
                    logger.debug(f"Processed relationship: {rel_name}")
                except Exception as e:
                    logger.error(f"Error processing relationship {rel_name}: {e}")

        logger.info(f"Processed {rel_processed} relationships")

        # Extract subtypes
        logger.info("Extracting subtypes...")
        subtypes = extract_subtypes_from_esri_data(input_data)
        schema.subtypes = subtypes
        logger.info(f"Found {len(subtypes)} subtype collections")

        # Final processing
        schema.infer_keys_from_relationships()
        schema.detect_primary_keys()

        # Set metadata
        schema.set_metadata_from_esri_json(input_data)

        # Validation
        validation_errors = schema.validate_domain_references()
        if validation_errors:
            logger.warning(f"Domain validation errors: {validation_errors}")

        logger.info("Transform completed successfully!")
        logger.info(f"Final summary: {schema.get_schema_summary()}")

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error transforming ESRI JSON: {e}")
        logger.error(error_details)
        raise

    return schema


def main():
    """Example usage of the robust transformer."""
    import json
    import os

    # Example paths
    if os.name == "nt":
        base_dir = r"H:\code\lg-geology-data-model\exports"
    else:
        base_dir = "/home/marco/code/github.com/lg-geology-data-model/exports/"

    json_path = os.path.join(base_dir, "2025-05-26/GCOVERP_export.json")

    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return

    # Load and transform
    with open(json_path, "r", encoding="utf-8") as f:
        esri_json_data = json.load(f)

    schema = transform_esri_json_robust(
        esri_json_data, target_prefix="GC_", exclude_metadata_fields=True
    )

    print(f"Transformation completed!")
    print(f"Summary: {schema.get_schema_summary()}")

    # List some results
    print(f"\nTables found:")
    for name in list(schema.tables.keys())[:5]:
        print(f"  - {name}")

    print(f"\nFeature classes found:")
    for name in list(schema.feature_classes.keys())[:5]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
