from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class CodedValue:
    """Represents a coded value in a domain"""

    code: Union[str, int, float]
    name: str


@dataclass
class CodedDomain:
    """Represents a coded value domain"""

    name: str
    description: Optional[str] = None
    field_type: Optional[str] = None
    coded_values: list[CodedValue] = field(default_factory=list)


@dataclass
class RangeDomain:
    """Represents a range domain"""

    name: str
    description: Optional[str] = None
    field_type: Optional[str] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None


@dataclass
class Field:
    """Represents a field in a feature class or table"""

    name: str
    alias: Optional[str] = None
    type: Optional[str] = None
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    default_value: Any = None
    domain: Optional[str] = None  # Reference to domain name


@dataclass
class SubtypeValue:
    """Represents a subtype value"""

    code: Union[int, str]
    name: str


@dataclass
class Subtype:
    """Represents a subtype definition"""

    name: str
    subtype_field: str
    default_subtype: Optional[Union[int, str]] = None
    subtypes: list[SubtypeValue] = field(default_factory=list)
    # Field-level domain assignments per subtype
    field_domains: dict[Union[int, str], dict[str, str]] = field(default_factory=dict)


@dataclass
class RelationshipClass:
    """Represents a relationship between feature classes/tables"""

    name: str
    origin_table: str
    destination_table: str
    relationship_type: str  # e.g., "OneToMany", "ManyToMany"
    forward_label: Optional[str] = None
    backward_label: Optional[str] = None
    cardinality: Optional[str] = None
    is_composite: bool = False
    is_attributed: bool = False
    origin_primary_key: Optional[str] = None
    origin_foreign_key: Optional[str] = None
    destination_primary_key: Optional[str] = None
    destination_foreign_key: Optional[str] = None


@dataclass
class Index:
    """Represents an index on a feature class or table"""

    name: str
    fields: list[str]
    is_unique: bool = False
    is_ascending: bool = True


@dataclass
class FeatureClass:
    """Represents a feature class"""

    name: str
    alias: Optional[str] = None
    feature_type: Optional[str] = None  # e.g., "Point", "Polyline", "Polygon"
    geometry_type: Optional[str] = None
    spatial_reference: Optional[dict[str, Any]] = None
    has_z: bool = False
    has_m: bool = False
    fields: list[Field] = field(default_factory=list)
    indexes: list[Index] = field(default_factory=list)
    subtypes: Optional[Subtype] = None
    default_subtype: Optional[Union[int, str]] = None
    subtype_field: Optional[str] = None
    object_id_field: Optional[str] = None
    global_id_field: Optional[str] = None
    shape_field: Optional[str] = None
    area_field: Optional[str] = None
    length_field: Optional[str] = None
    # Key fields
    primary_key: Optional[str] = None  # Primary key field name
    foreign_keys: dict[str, str] = field(
        default_factory=dict
    )  # field_name -> referenced_table


@dataclass
class Table:
    """Represents a table (non-spatial)"""

    name: str
    alias: Optional[str] = None
    fields: list[Field] = field(default_factory=list)
    indexes: list[Index] = field(default_factory=list)
    subtypes: Optional[Subtype] = None
    default_subtype: Optional[Union[int, str]] = None
    subtype_field: Optional[str] = None
    object_id_field: Optional[str] = None
    global_id_field: Optional[str] = None
    # Key fields
    primary_key: Optional[str] = None  # Primary key field name
    foreign_keys: dict[str, str] = field(
        default_factory=dict
    )  # field_name -> referenced_table


@dataclass
class ESRISchema:
    """
    Main container for ESRI geodatabase schema information.
    Holds all the different components that make up a geodatabase schema.
    """

    # Domains (both coded and range)
    coded_domains: dict[str, CodedDomain] = field(default_factory=dict)
    range_domains: dict[str, RangeDomain] = field(default_factory=dict)

    # Subtypes
    subtypes: dict[str, Subtype] = field(default_factory=dict)

    # Relationships
    relationships: dict[str, RelationshipClass] = field(default_factory=dict)

    # Feature classes and tables
    feature_classes: dict[str, FeatureClass] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def geological_datamodel_version(self):
        return self.metadata.get("geological_datamodel_version")

    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @geological_datamodel_version.setter
    def geological_datamodel_version(self, value):
        self.metadata["geological_datamodel_version"] = value

    def get_domain(self, domain_name: str) -> Optional[Union[CodedDomain, RangeDomain]]:
        """Get a domain by name, checking both coded and range domains"""
        if domain_name in self.coded_domains:
            return self.coded_domains[domain_name]
        return self.range_domains.get(domain_name)

    def get_all_domains(self) -> dict[str, Union[CodedDomain, RangeDomain]]:
        """Get all domains (both coded and range)"""
        all_domains = {}
        all_domains.update(self.coded_domains)
        all_domains.update(self.range_domains)
        return all_domains

    def get_feature_class_or_table(
        self, name: str
    ) -> Optional[Union[FeatureClass, Table]]:
        """Get a feature class or table by name"""
        if name in self.feature_classes:
            return self.feature_classes[name]
        return self.tables.get(name)

    def get_relationships_for_class(self, class_name: str) -> list[RelationshipClass]:
        """Get all relationships involving a specific feature class or table"""
        relationships = []
        for rel in self.relationships.values():
            if rel.origin_table == class_name or rel.destination_table == class_name:
                relationships.append(rel)
        return relationships

    def validate_domain_references(self) -> list[str]:
        """Validate that all domain references in fields exist"""
        errors = []
        all_domains = self.get_all_domains()

        # Check feature classes
        for fc_name, fc in self.feature_classes.items():
            for field in fc.fields:
                if field.domain and field.domain not in all_domains:
                    errors.append(
                        f"Feature class '{fc_name}' field '{field.name}' references non-existent domain '{field.domain}'"
                    )

        # Check tables
        for table_name, table in self.tables.items():
            for field in table.fields:
                if field.domain and field.domain not in all_domains:
                    errors.append(
                        f"Table '{table_name}' field '{field.name}' references non-existent domain '{field.domain}'"
                    )

        return errors

    def get_schema_summary(self) -> dict[str, int]:
        """Get a summary of the schema contents"""
        return {
            "coded_domains": len(self.coded_domains),
            "range_domains": len(self.range_domains),
            "subtypes": sum(
                len(st.subtypes) for st in self.subtypes.values()
            ),  # Count individual values
            "subtype_collections": len(self.subtypes),  # Count collections
            "relationships": len(self.relationships),
            "feature_classes": len(self.feature_classes),
            "tables": len(self.tables),
        }

    def get_subtype_count(self) -> int:
        """Get the total number of individual subtype values"""
        count = 0
        for subtype in self.subtypes.values():
            count += len(subtype.subtypes)
        return count

    def get_subtype_summary(self) -> dict[str, int]:
        """Get a summary of subtypes by collection"""
        return {name: len(subtype.subtypes) for name, subtype in self.subtypes.items()}

    def infer_keys_from_relationships(self):
        """
        Infer primary and foreign keys from relationship definitions.
        This updates the primary_key and foreign_keys attributes of tables and feature classes.
        """
        for rel in self.relationships.values():
            # Get origin and destination objects
            origin = self.get_feature_class_or_table(rel.origin_table)
            destination = self.get_feature_class_or_table(rel.destination_table)

            if origin and destination:
                # Set primary keys if not already set
                if rel.origin_primary_key and not origin.primary_key:
                    origin.primary_key = rel.origin_primary_key

                if rel.destination_primary_key and not destination.primary_key:
                    destination.primary_key = rel.destination_primary_key

                # Set foreign keys
                if rel.origin_foreign_key:
                    origin.foreign_keys[rel.origin_foreign_key] = rel.destination_table

                if rel.destination_foreign_key:
                    destination.foreign_keys[rel.destination_foreign_key] = (
                        rel.origin_table
                    )

    def detect_primary_keys(self):
        """
        Auto-detect primary keys for tables and feature classes that don't have them set.
        Called after infer_keys_from_relationships().
        """

        def find_pk(obj):
            if obj.primary_key:
                return  # Already set

            # First priority: object_id_field
            if obj.object_id_field:
                obj.primary_key = obj.object_id_field
                return

            # Second priority: OID type field
            for field in obj.fields:
                if field.type in ["OID", "esriFieldTypeOID"]:
                    obj.primary_key = field.name
                    return

            # Third priority: GlobalID/UUID
            for field in obj.fields:
                if field.name.upper() in [
                    "GLOBALID",
                    "UUID",
                    "GUID",
                ] and field.type in ["GUID", "esriFieldTypeGUID", "GlobalID"]:
                    obj.primary_key = field.name
                    return

            # Last resort: field named OBJECTID
            for field in obj.fields:
                if field.name.upper() == "OBJECTID":
                    obj.primary_key = field.name
                    return

        # Apply to all tables and feature classes
        for table in self.tables.values():
            find_pk(table)

        for fc in self.feature_classes.values():
            find_pk(fc)

    def get_all_coded_values(self) -> dict[str, list[dict[str, Any]]]:
        """
        Extract all numerical values from subtypes and coded domains.

        Returns:
            Dictionary with two keys:
            - 'coded_domains': List of all coded values from domains
            - 'subtypes': List of all subtype values

            Each item in the lists contains:
            - source: Name of the domain or subtype collection
            - code: The numerical/string code
            - name: The textual description
            - type: 'domain' or 'subtype'
        """
        result = {"coded_domains": [], "subtypes": []}

        # Extract from coded domains
        for domain_name, domain in self.coded_domains.items():
            for coded_value in domain.coded_values:
                result["coded_domains"].append(
                    {
                        "source": domain_name,
                        "code": coded_value.code,
                        "name": coded_value.name,
                        "type": "domain",
                    }
                )

        # Extract from subtypes
        for subtype_name, subtype in self.subtypes.items():
            for subtype_value in subtype.subtypes:
                result["subtypes"].append(
                    {
                        "source": subtype_name,
                        "code": subtype_value.code,
                        "name": subtype_value.name,
                        "type": "subtype",
                    }
                )

        return result

    def get_coded_values_flat(self) -> list[dict[str, Any]]:
        """
        Get a flat list of all coded values from both domains and subtypes.

        Returns:
            List of dictionaries, each containing:
            - source: Name of the domain or subtype collection
            - code: The numerical/string code
            - name: The textual description
            - type: 'domain' or 'subtype'
        """
        values = self.get_all_coded_values()
        return values["coded_domains"] + values["subtypes"]

    def get_coded_values_by_source(self) -> dict[str, dict[Union[str, int], str]]:
        """
        Get all coded values organized by their source.

        Returns:
            Dictionary where:
            - Keys are source names (domain or subtype names)
            - Values are dictionaries mapping codes to names
        """
        result = {}

        # Add coded domains
        for domain_name, domain in self.coded_domains.items():
            result[domain_name] = {cv.code: cv.name for cv in domain.coded_values}

        # Add subtypes
        for subtype_name, subtype in self.subtypes.items():
            result[subtype_name] = {sv.code: sv.name for sv in subtype.subtypes}

        return result

    def find_coded_value(
        self, code: Union[str, int], source: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Find all occurrences of a specific code across domains and subtypes.

        Args:
            code: The code to search for
            source: Optional source name to limit search

        Returns:
            List of matches, each containing source, code, name, and type
        """
        matches = []
        code_str = str(code)  # Convert to string for comparison

        if source:
            # Search specific source
            if source in self.coded_domains:
                domain = self.coded_domains[source]
                for cv in domain.coded_values:
                    if str(cv.code) == code_str:
                        matches.append(
                            {
                                "source": source,
                                "code": cv.code,
                                "name": cv.name,
                                "type": "domain",
                            }
                        )

            if source in self.subtypes:
                subtype = self.subtypes[source]
                for sv in subtype.subtypes:
                    if str(sv.code) == code_str:
                        matches.append(
                            {
                                "source": source,
                                "code": sv.code,
                                "name": sv.name,
                                "type": "subtype",
                            }
                        )
        else:
            # Search all sources
            all_values = self.get_coded_values_flat()
            matches = [v for v in all_values if str(v["code"]) == code_str]

        return matches

    def export_coded_values_to_csv(self, filepath: str, delimiter: str = ","):
        """
        Export all coded values to a CSV file.

        Args:
            filepath: Path to save the CSV file
            delimiter: CSV delimiter (default: comma)
        """
        import csv

        all_values = self.get_coded_values_flat()

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["source", "type", "code", "name"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)

            writer.writeheader()
            for value in all_values:
                writer.writerow(value)

    def get_coded_values_statistics(self) -> dict[str, Any]:
        """
        Get statistics about coded values in the schema.

        Returns:
            Dictionary with statistics including counts and data types
        """
        all_values = self.get_coded_values_flat()

        stats = {
            "total_values": len(all_values),
            "domain_values": len([v for v in all_values if v["type"] == "domain"]),
            "subtype_values": len([v for v in all_values if v["type"] == "subtype"]),
            "unique_sources": len(set(v["source"] for v in all_values)),
            "numeric_codes": 0,
            "string_codes": 0,
            "sources": {},
        }

        # Count numeric vs string codes
        for value in all_values:
            if isinstance(value["code"], (int, float)):
                stats["numeric_codes"] += 1
            else:
                stats["string_codes"] += 1

        # Count by source
        for value in all_values:
            source = value["source"]
            if source not in stats["sources"]:
                stats["sources"][source] = 0
            stats["sources"][source] += 1

        return stats


"""
def main():

        import os
        import json
        from transform_esri_json import transform_esri_json

        # Example usage:
        if os.name == "nt":
            BASE_DIR = r"H:\\code\\lg-geology-data-model\\exports"
        else:
            BASE_DIR = "/home/marco/code/github.com/lg-geology-data-model/exports/"

        schema_path = os.path.join(BASE_DIR, "2025-05-26/GCOVERP_export.json")

        with open(schema_path, "r") as f:
            esri_json_data = json.loads(f.read())

        schema = transform_esri_json(esri_json_data)


        print(schema.get_schema_summary())
        # Get all coded values
        all_values = schema.get_all_coded_values()
        print(f"Found {len(all_values['coded_domains'])} domain values")
        print(f"Found {len(all_values['subtypes'])} subtype values")

        # Get flat list
        flat_list = schema.get_coded_values_flat()
        for value in flat_list[:5]:  # Show first 5
            print(f"{value['source']}: {value['code']} = {value['name']}")

        # Get values organized by source
        by_source = schema.get_coded_values_by_source()
        for source, values in by_source.items():
            print(f"\n{source}:")
            for code, name in list(values.items())[:3]:  # Show first 3
                print(f"  {code}: {name}")

        # Find specific code
        matches = schema.find_coded_value(15602006)
        for match in matches:
            print(f"Found code {match['code']} in {match['source']}: {match['name']}")

        # Export to CSV
        schema.export_coded_values_to_csv('coded_values.csv')

        # Get statistics
        stats = schema.get_coded_values_statistics()
        print(f"Total coded values: {stats['total_values']}")
        print(f"Numeric codes: {stats['numeric_codes']}")
        print(f"String codes: {stats['string_codes']}")

if __name__ == '__main__':
    main()"""
