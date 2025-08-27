# src/gcover/schema/simple_transformer.py
"""
Simple ESRI JSON transformer - creates flat JSON structure.
This is a lightweight alternative to the full dataclass-based transformer.
"""

import json
import traceback
from typing import Any, Dict
from loguru import logger

SCHEMA_VERSION = 2

# Copy all the functions from document 43:
# - convert_field_type()
# - extract_subtypes()
# - get_field_list()
# - extract_datasets_simple()
# - extract_datasets()
# - transform_esri_json()
# - transform_esri_json_file()


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


def extract_datasets_simple(data, datasets_list, parent_path=""):
    """
    Recursively search for 'datasets' objects in the JSON structure and extract data
    for specified dataset types.
    """
    target_types = ["esriDTTable", "esriDTFeatureClass", "esriDTRelationshipClass"]

    if isinstance(data, dict):
        # Check if this is a dataset object
        if (
            "name" in data
            and "datasetType" in data
            and data.get("datasetType") in target_types
        ):
            # Extract the required attributes
            dataset_info = {
                "name": data.get("name"),
                "datasetType": data.get("datasetType"),
            }

            # Add parent path information if available
            if parent_path:
                dataset_info["path"] = f"{parent_path}/{data.get('name')}"

            datasets_list.append(dataset_info)

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
                extract_datasets_simple(dataset, datasets_list, current_path)

        # Continue searching in other keys
        for key, value in data.items():
            if key != "datasets":  # Already processed above
                extract_datasets_simple(value, datasets_list, parent_path)

    elif isinstance(data, list):
        for item in data:
            extract_datasets_simple(item, datasets_list, parent_path)


def extract_datasets(
    data,
    tables_list,
    featclasses_list,
    relationships_list,
    subtypes_list,
    parent_path="",
):
    """
    Recursively search for datasets in the JSON structure and split them by type.
    For tables, extract field information.
    """
    # TODO:
    if isinstance(data, dict):
        # Check if this is a table with fields information
        if (
            "name" in data
            and "datasetType" in data
            and data.get("datasetType") == "esriDTTable"
        ):
            table_name = data.get("name")

            if not table_name.endswith("_I"):
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

                    # Store the table with its fields

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
                    click.secho(e, fg="red")

                rel_info = {
                    "name": data.get("name"),
                    "origin": [item["name"] for item in data.get("originClassNames")],
                    "destination": [
                        item["name"] for item in data.get("destinationClassNames")
                    ],
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
                extract_datasets(
                    dataset,
                    tables_list,
                    featclasses_list,
                    relationships_list,
                    subtypes_list,
                    current_path,
                )

        # Continue searching in other keys
        for key, value in data.items():
            if key != "datasets":  # Already processed above
                extract_datasets(
                    value,
                    tables_list,
                    featclasses_list,
                    relationships_list,
                    subtypes_list,
                    parent_path,
                )

    elif isinstance(data, list):
        for item in data:
            extract_datasets(
                item,
                tables_list,
                featclasses_list,
                relationships_list,
                subtypes_list,
                parent_path,
            )


def transform_esri_flat_json(input_data):
    """
    Transform ESRI JSON format into custom structured format.
    Handles multiple section types: coded_domains, subtypes, relationships, featureclasses, tables
    """
    result = {"version": SCHEMA_VERSION}
    try:
        # Add metadata if you want to keep it
        if "dateExported" in input_data:
            result["metadata"] = {
                "dateExported": input_data.get("dateExported"),
                "datasetType": input_data.get("datasetType"),
                "catalogPath": input_data.get("catalogPath"),
                "name": input_data.get("name"),
            }

        # Process domains
        logger.info(f"Processing domains...")
        if "domains" in input_data:
            domain_result = {}
            for domain in input_data["domains"]:
                if domain.get("type") == "codedValue":
                    domain_name = domain.get("name")
                    if domain_name:
                        # Create our custom structure for this domain
                        domain_result[domain_name] = {
                            "type": "CodedValue",
                            "codedValues": {},
                        }

                        # Add description if available
                        if "description" in domain:
                            domain_result[domain_name]["description"] = domain[
                                "description"
                            ]

                        # Process coded values and swap the structure
                        if "codedValues" in domain:
                            for value_pair in domain["codedValues"]:
                                # Convert code to string if it's numeric
                                code = (
                                    str(value_pair["code"])
                                    if isinstance(value_pair["code"], (int, float))
                                    else value_pair["code"]
                                )
                                domain_result[domain_name]["codedValues"][code] = (
                                    value_pair["name"]
                                )
            result["coded_domain"] = domain_result

        # Extract and process datasets from any level of the JSON
        """datasets_list = []
    extract_datasets_simple(input_data, datasets_list)

    if datasets_list:
        result["datasets"] = datasets_list"""

        # Extract and process datasets from any level of the JSON
        tables_list = []
        featclasses_list = []
        relationships_list = []
        subtypes_list = []

        logger.info(f"Extracting datasets...")

        extract_datasets(
            input_data, tables_list, featclasses_list, relationships_list, subtypes_list
        )

        logger.info(f"processing tables...")
        processed_tables = {}
        for table_item in tables_list:
            if isinstance(table_item, dict):
                # If it's a table with fields
                for table_name, fields in table_item.items():
                    processed_tables[table_name] = {"fields": fields}

        if processed_tables:
            result["tables"] = processed_tables

        if featclasses_list:
            featclasses_dict = {k: v for d in featclasses_list for k, v in d.items()}

            result["featclasses"] = featclasses_dict

        if relationships_list:
            result["relationships"] = {
                rel.pop("name"): rel for rel in relationships_list if "name" in rel
            }

        # Process subtypes
        # Extract and process subtypes from any level of the JSON
        logger.info(f"Extracting subtypes...")
        subtypes_dict = {}
        extract_subtypes(input_data, subtypes_dict)

        # logger.debug(subtypes_dict)

        if subtypes_dict:
            result["subtypes"] = subtypes_dict
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(e)
        logger.error(error_details)

    return result


def transform_esri_json_file(input_file, output_file):
    """Process an ESRI JSON file and convert it to the custom format"""
    try:
        # Read input file
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        if (
            input_data.get("datasetType") != "DEWorkspace"
            or input_data.get("majorVersion", 0) < 3
        ):
            raise Exception("Error: Not an ESRI 'DEWorkspace' export JSON")

        # Transform data
        output_data = transform_esri_flat_json(input_data)

        # Write output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        click.secho(
            f"Transformation complete. Output written to {output_file}", fg="green"
        )
        return True
    except Exception as e:
        click.secho(f"Error processing file: {e}", fg="red")
        error_details = traceback.format_exc()
        click.secho(error_details, fg="red")
    return False
