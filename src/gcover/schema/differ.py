from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from loguru import logger


class ChangeType(Enum):
    """Types of changes that can occur between schemas"""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class FieldChange:
    """Represents a change in a field"""

    field_name: str
    change_type: ChangeType
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    property_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class DomainChange:
    """Represents a change in a domain"""

    domain_name: str
    change_type: ChangeType
    old_domain: Optional[Union["CodedDomain", "RangeDomain"]] = None
    new_domain: Optional[Union["CodedDomain", "RangeDomain"]] = None
    coded_value_changes: dict[str, ChangeType] = field(default_factory=dict)
    property_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class TableChange:
    """Represents a change in a table or feature class"""

    table_name: str
    change_type: ChangeType
    old_table: Optional[Union["Table", "FeatureClass"]] = None
    new_table: Optional[Union["Table", "FeatureClass"]] = None
    field_changes: list[FieldChange] = field(default_factory=list)
    property_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class RelationshipChange:
    """Represents a change in a relationship"""

    relationship_name: str
    change_type: ChangeType
    old_relationship: Optional["RelationshipClass"] = None
    new_relationship: Optional["RelationshipClass"] = None
    property_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class SubtypeChange:
    """Represents a change in subtypes"""

    subtype_name: str
    change_type: ChangeType
    old_subtype: Optional["Subtype"] = None
    new_subtype: Optional["Subtype"] = None
    value_changes: dict[str, ChangeType] = field(default_factory=dict)
    property_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


class SchemaDiff:
    """
    Class for comparing two ESRISchema instances and tracking differences
    """

    def __init__(self, old_schema: "ESRISchema", new_schema: "ESRISchema"):
        """
        Initialize with two schemas to compare

        Args:
            old_schema: The original/baseline schema
            new_schema: The new/modified schema
        """
        self.old_schema = old_schema
        self.new_schema = new_schema

        # Storage for all changes
        self.domain_changes: list[DomainChange] = []
        self.table_changes: list[TableChange] = []
        self.feature_class_changes: list[TableChange] = []
        self.relationship_changes: list[RelationshipChange] = []
        self.subtype_changes: list[SubtypeChange] = []

        self.metadata = {}  # TODO

        self._set_metadata()

        # Perform the diff
        self._compute_diff()

    def _set_metadata(self):
        metadata = {}
        metadata['old_schema'] = {'name': self.old_schema.get('name', 'unknown')}
        metadata['new_schema'] = {'name': self.new_schema.get('name', 'unknown')}

        self.metadata = metadata


    def _compute_diff(self):
        """Compute all differences between the schemas"""
        self._diff_domains()
        self._diff_tables()
        self._diff_feature_classes()
        self._diff_relationships()
        self._diff_subtypes()

    def _diff_domains(self):
        """Compare coded and range domains"""
        # Get all domain names from both schemas
        old_domains = self.old_schema.get_all_domains()
        new_domains = self.new_schema.get_all_domains()

        old_names = set(old_domains.keys())
        new_names = set(new_domains.keys())

        # Find added, removed, and common domains
        added = new_names - old_names
        removed = old_names - new_names
        common = old_names & new_names

        # Process removed domains
        for name in removed:
            self.domain_changes.append(
                DomainChange(
                    domain_name=name,
                    change_type=ChangeType.REMOVED,
                    old_domain=old_domains[name],
                )
            )

        # Process added domains
        for name in added:
            self.domain_changes.append(
                DomainChange(
                    domain_name=name,
                    change_type=ChangeType.ADDED,
                    new_domain=new_domains[name],
                )
            )

        # Process potentially modified domains
        for name in common:
            old_domain = old_domains[name]
            new_domain = new_domains[name]

            # Check if domain type changed
            if type(old_domain) != type(new_domain):
                self.domain_changes.append(
                    DomainChange(
                        domain_name=name,
                        change_type=ChangeType.MODIFIED,
                        old_domain=old_domain,
                        new_domain=new_domain,
                        property_changes={
                            "type": (
                                type(old_domain).__name__,
                                type(new_domain).__name__,
                            )
                        },
                    )
                )
                continue

            # Compare domain properties
            changes = self._compare_domain_properties(old_domain, new_domain)
            if changes:
                self.domain_changes.append(changes)

    def _compare_domain_properties(
        self, old_domain, new_domain
    ) -> Optional[DomainChange]:
        """Compare properties of two domains of the same type"""
        property_changes = {}
        coded_value_changes = {}

        # Compare basic properties
        if old_domain.description != new_domain.description:
            property_changes["description"] = (
                old_domain.description,
                new_domain.description,
            )

        if old_domain.field_type != new_domain.field_type:
            property_changes["field_type"] = (
                old_domain.field_type,
                new_domain.field_type,
            )

        # For coded domains, compare coded values
        if hasattr(old_domain, "coded_values"):
            old_values = {str(cv.code): cv.name for cv in old_domain.coded_values}
            new_values = {str(cv.code): cv.name for cv in new_domain.coded_values}

            old_codes = set(old_values.keys())
            new_codes = set(new_values.keys())

            # Find changes in coded values
            for code in old_codes - new_codes:
                coded_value_changes[code] = ChangeType.REMOVED

            for code in new_codes - old_codes:
                coded_value_changes[code] = ChangeType.ADDED

            for code in old_codes & new_codes:
                if old_values[code] != new_values[code]:
                    coded_value_changes[code] = ChangeType.MODIFIED

        # For range domains, compare min/max
        elif hasattr(old_domain, "min_value"):
            if old_domain.min_value != new_domain.min_value:
                property_changes["min_value"] = (
                    old_domain.min_value,
                    new_domain.min_value,
                )
            if old_domain.max_value != new_domain.max_value:
                property_changes["max_value"] = (
                    old_domain.max_value,
                    new_domain.max_value,
                )

        # Return change object if any changes found
        if property_changes or coded_value_changes:
            return DomainChange(
                domain_name=old_domain.name,
                change_type=ChangeType.MODIFIED,
                old_domain=old_domain,
                new_domain=new_domain,
                property_changes=property_changes,
                coded_value_changes=coded_value_changes,
            )

        return None

    def _diff_tables(self):
        """Compare tables between schemas"""
        old_names = set(self.old_schema.tables.keys())
        new_names = set(self.new_schema.tables.keys())

        # Process removed tables
        for name in old_names - new_names:
            self.table_changes.append(
                TableChange(
                    table_name=name,
                    change_type=ChangeType.REMOVED,
                    old_table=self.old_schema.tables[name],
                )
            )

        # Process added tables
        for name in new_names - old_names:
            self.table_changes.append(
                TableChange(
                    table_name=name,
                    change_type=ChangeType.ADDED,
                    new_table=self.new_schema.tables[name],
                )
            )

        # Process potentially modified tables
        for name in old_names & new_names:
            changes = self._compare_table_properties(
                self.old_schema.tables[name], self.new_schema.tables[name], name
            )
            if changes:
                self.table_changes.append(changes)

    def _diff_feature_classes(self):
        """Compare feature classes between schemas"""
        old_names = set(self.old_schema.feature_classes.keys())
        new_names = set(self.new_schema.feature_classes.keys())

        # Process removed feature classes
        for name in old_names - new_names:
            self.feature_class_changes.append(
                TableChange(
                    table_name=name,
                    change_type=ChangeType.REMOVED,
                    old_table=self.old_schema.feature_classes[name],
                )
            )

        # Process added feature classes
        for name in new_names - old_names:
            self.feature_class_changes.append(
                TableChange(
                    table_name=name,
                    change_type=ChangeType.ADDED,
                    new_table=self.new_schema.feature_classes[name],
                )
            )

        # Process potentially modified feature classes
        for name in old_names & new_names:
            changes = self._compare_feature_class_properties(
                self.old_schema.feature_classes[name],
                self.new_schema.feature_classes[name],
                name,
            )
            if changes:
                self.feature_class_changes.append(changes)

    def _compare_table_properties(
        self, old_table, new_table, table_name
    ) -> Optional[TableChange]:
        """Compare properties of two tables"""
        property_changes = {}
        field_changes = []

        # Compare basic properties
        props_to_compare = [
            "alias",
            "object_id_field",
            "global_id_field",
            "subtype_field",
            "default_subtype",
        ]

        for prop in props_to_compare:
            old_val = getattr(old_table, prop, None)
            new_val = getattr(new_table, prop, None)
            if old_val != new_val:
                property_changes[prop] = (old_val, new_val)

        # Compare fields
        field_changes = self._compare_fields(old_table.fields, new_table.fields)

        # Return change object if any changes found
        if property_changes or field_changes:
            return TableChange(
                table_name=table_name,
                change_type=ChangeType.MODIFIED,
                old_table=old_table,
                new_table=new_table,
                property_changes=property_changes,
                field_changes=field_changes,
            )

        return None

    def _compare_feature_class_properties(
        self, old_fc, new_fc, fc_name
    ) -> Optional[TableChange]:
        """Compare properties of two feature classes"""
        # Start with table comparison
        changes = self._compare_table_properties(old_fc, new_fc, fc_name)

        # Add feature class specific properties
        fc_props = [
            "feature_type",
            "geometry_type",
            "has_z",
            "has_m",
            "shape_field",
            "area_field",
            "length_field",
        ]

        property_changes = changes.property_changes if changes else {}

        for prop in fc_props:
            old_val = getattr(old_fc, prop, None)
            new_val = getattr(new_fc, prop, None)
            if old_val != new_val:
                property_changes[prop] = (old_val, new_val)

        # Check spatial reference changes
        if old_fc.spatial_reference != new_fc.spatial_reference:
            property_changes["spatial_reference"] = (
                old_fc.spatial_reference,
                new_fc.spatial_reference,
            )

        # Return change object if any changes found
        if property_changes or (changes and changes.field_changes):
            return TableChange(
                table_name=fc_name,
                change_type=ChangeType.MODIFIED,
                old_table=old_fc,
                new_table=new_fc,
                property_changes=property_changes,
                field_changes=changes.field_changes if changes else [],
            )

        return None

    def _compare_fields(
        self, old_fields: list["Field"], new_fields: list["Field"]
    ) -> list[FieldChange]:
        """Compare two lists of fields"""
        field_changes = []

        old_field_dict = {f.name: f for f in old_fields}
        new_field_dict = {f.name: f for f in new_fields}

        old_names = set(old_field_dict.keys())
        new_names = set(new_field_dict.keys())

        # Removed fields
        for name in old_names - new_names:
            field_changes.append(
                FieldChange(
                    field_name=name,
                    change_type=ChangeType.REMOVED,
                    old_value=old_field_dict[name],
                )
            )

        # Added fields
        for name in new_names - old_names:
            field_changes.append(
                FieldChange(
                    field_name=name,
                    change_type=ChangeType.ADDED,
                    new_value=new_field_dict[name],
                )
            )

        # Modified fields
        for name in old_names & new_names:
            old_field = old_field_dict[name]
            new_field = new_field_dict[name]

            property_changes = {}

            # Compare field properties
            for prop in [
                "alias",
                "type",
                "length",
                "precision",
                "scale",
                "nullable",
                "default_value",
                "domain",
            ]:
                old_val = getattr(old_field, prop, None)
                new_val = getattr(new_field, prop, None)
                if old_val != new_val:
                    property_changes[prop] = (old_val, new_val)

            if property_changes:
                field_changes.append(
                    FieldChange(
                        field_name=name,
                        change_type=ChangeType.MODIFIED,
                        old_value=old_field,
                        new_value=new_field,
                        property_changes=property_changes,
                    )
                )

        return field_changes

    def _diff_relationships(self):
        """Compare relationships between schemas"""
        old_names = set(self.old_schema.relationships.keys())
        new_names = set(self.new_schema.relationships.keys())

        # Process removed relationships
        for name in old_names - new_names:
            self.relationship_changes.append(
                RelationshipChange(
                    relationship_name=name,
                    change_type=ChangeType.REMOVED,
                    old_relationship=self.old_schema.relationships[name],
                )
            )

        # Process added relationships
        for name in new_names - old_names:
            self.relationship_changes.append(
                RelationshipChange(
                    relationship_name=name,
                    change_type=ChangeType.ADDED,
                    new_relationship=self.new_schema.relationships[name],
                )
            )

        # Process potentially modified relationships
        for name in old_names & new_names:
            old_rel = self.old_schema.relationships[name]
            new_rel = self.new_schema.relationships[name]

            property_changes = {}

            # Compare relationship properties
            props_to_compare = [
                "origin_table",
                "destination_table",
                "relationship_type",
                "forward_label",
                "backward_label",
                "cardinality",
                "is_composite",
                "is_attributed",
                "origin_primary_key",
                "origin_foreign_key",
                "destination_primary_key",
                "destination_foreign_key",
            ]

            for prop in props_to_compare:
                old_val = getattr(old_rel, prop, None)
                new_val = getattr(new_rel, prop, None)
                if old_val != new_val:
                    property_changes[prop] = (old_val, new_val)

            if property_changes:
                self.relationship_changes.append(
                    RelationshipChange(
                        relationship_name=name,
                        change_type=ChangeType.MODIFIED,
                        old_relationship=old_rel,
                        new_relationship=new_rel,
                        property_changes=property_changes,
                    )
                )

    def _diff_subtypes(self):
        """Compare subtypes between schemas"""
        old_names = set(self.old_schema.subtypes.keys())
        new_names = set(self.new_schema.subtypes.keys())

        # Process removed subtypes
        for name in old_names - new_names:
            self.subtype_changes.append(
                SubtypeChange(
                    subtype_name=name,
                    change_type=ChangeType.REMOVED,
                    old_subtype=self.old_schema.subtypes[name],
                )
            )

        # Process added subtypes
        for name in new_names - old_names:
            self.subtype_changes.append(
                SubtypeChange(
                    subtype_name=name,
                    change_type=ChangeType.ADDED,
                    new_subtype=self.new_schema.subtypes[name],
                )
            )

        # Process potentially modified subtypes
        for name in old_names & new_names:
            old_subtype = self.old_schema.subtypes[name]
            new_subtype = self.new_schema.subtypes[name]

            property_changes = {}
            value_changes = {}

            # Compare basic properties
            if old_subtype.subtype_field != new_subtype.subtype_field:
                property_changes["subtype_field"] = (
                    old_subtype.subtype_field,
                    new_subtype.subtype_field,
                )

            if old_subtype.default_subtype != new_subtype.default_subtype:
                property_changes["default_subtype"] = (
                    old_subtype.default_subtype,
                    new_subtype.default_subtype,
                )

            # Compare subtype values
            old_values = {str(sv.code): sv.name for sv in old_subtype.subtypes}
            new_values = {str(sv.code): sv.name for sv in new_subtype.subtypes}

            old_codes = set(old_values.keys())
            new_codes = set(new_values.keys())

            for code in old_codes - new_codes:
                value_changes[code] = ChangeType.REMOVED

            for code in new_codes - old_codes:
                value_changes[code] = ChangeType.ADDED

            for code in old_codes & new_codes:
                if old_values[code] != new_values[code]:
                    value_changes[code] = ChangeType.MODIFIED

            if property_changes or value_changes:
                self.subtype_changes.append(
                    SubtypeChange(
                        subtype_name=name,
                        change_type=ChangeType.MODIFIED,
                        old_subtype=old_subtype,
                        new_subtype=new_subtype,
                        property_changes=property_changes,
                        value_changes=value_changes,
                    )
                )

    def has_changes(self) -> bool:
        """Check if any changes were detected"""
        return bool(
            self.domain_changes
            or self.table_changes
            or self.feature_class_changes
            or self.relationship_changes
            or self.subtype_changes
        )

    def get_summary(self) -> dict[str, dict[str, int]]:
        """Get a summary of all changes by type"""
        summary = {
            "domains": self._summarize_changes(self.domain_changes),
            "tables": self._summarize_changes(self.table_changes),
            "feature_classes": self._summarize_changes(self.feature_class_changes),
            "relationships": self._summarize_changes(self.relationship_changes),
            "subtypes": self._summarize_changes(self.subtype_changes),
        }
        return summary

    def _summarize_changes(self, changes: list) -> dict[str, int]:
        """Summarize a list of changes by type"""
        summary = {"added": 0, "removed": 0, "modified": 0}

        for change in changes:
            summary[change.change_type.value] += 1

        return summary

    def generate_report(self, detailed: bool = True) -> str:
        """
        Generate a human-readable report of all changes

        Args:
            detailed: If True, include detailed property changes

        Returns:
            Formatted string report
        """
        report = ["Schema Comparison Report", "=" * 50, ""]

        # Summary
        summary = self.get_summary()
        report.append("Summary:")
        for category, counts in summary.items():
            total = sum(counts.values())
            if total > 0:
                report.append(
                    f"  {category.replace('_', ' ').title()}: "
                    f"{counts['added']} added, "
                    f"{counts['removed']} removed, "
                    f"{counts['modified']} modified"
                )

        report.append("")

        # Detailed changes
        if detailed:
            # Domains
            if self.domain_changes:
                report.append("Domain Changes:")
                for change in self.domain_changes:
                    report.append(
                        f"  - {change.domain_name}: {change.change_type.value}"
                    )
                    if change.property_changes:
                        for prop, (old, new) in change.property_changes.items():
                            report.append(f"      {prop}: {old} -> {new}")
                    if change.coded_value_changes:
                        report.append(
                            f"      Coded values: {len(change.coded_value_changes)} changes"
                        )
                report.append("")

            # Tables
            if self.table_changes:
                report.append("Table Changes:")
                for change in self.table_changes:
                    report.append(
                        f"  - {change.table_name}: {change.change_type.value}"
                    )
                    if change.field_changes:
                        report.append(
                            f"      Fields: {len(change.field_changes)} changes"
                        )
                report.append("")

            # Feature Classes
            if self.feature_class_changes:
                report.append("Feature Class Changes:")
                for change in self.feature_class_changes:
                    report.append(
                        f"  - {change.table_name}: {change.change_type.value}"
                    )
                    if change.field_changes:
                        report.append(
                            f"      Fields: {len(change.field_changes)} changes"
                        )
                report.append("")

            # Relationships
            if self.relationship_changes:
                report.append("Relationship Changes:")
                for change in self.relationship_changes:
                    report.append(
                        f"  - {change.relationship_name}: {change.change_type.value}"
                    )
                    if change.property_changes:
                        for prop, (old, new) in change.property_changes.items():
                            report.append(f"      {prop}: {old} -> {new}")
                report.append("")

            # Subtypes
            if self.subtype_changes:
                report.append("Subtype Changes:")
                for change in self.subtype_changes:
                    report.append(
                        f"  - {change.subtype_name}: {change.change_type.value}"
                    )
                    if change.value_changes:
                        report.append(
                            f"      Values: {len(change.value_changes)} changes"
                        )
                report.append("")

        return "\n".join(report)


def main():
    import json
    import os

    from export_schema_diff import export_schema_diff_to_json
    from transform_esri_json import transform_esri_json

    diff_pairs = [
        {
            "from_version": "pre-4.0",
            "to_version": "4.0",
            "from_path": "2022-11-30/esri_export.json",
            "to_path": "2025-05-19/esri_export_gdb.json",
        },
        {
            "from_version": "4.0",
            "to_version": "4.1",
            "from_path": "2025-05-19/esri_export_gdb.json",
            "to_path": "2025-06-04/esri_report.json",
        },
    ]

    EXPORT_TYPE = "GDB"

    if EXPORT_TYPE == "GDB":
        FILTER_PREFIX = "GC_"
    else:
        FILTER_PREFIX = "TOPGIS_GC.GC_"

    # Example usage:
    if os.name == "nt":
        BASE_DIR = r"H:\code\lg-geology-data-model\exports"
    else:
        BASE_DIR = "/home/marco/code/github.com/lg-geology-data-model/exports/"

    for pair in diff_pairs:
        from_version, to_version, from_path, to_path = pair.values()

        old_schema_path = os.path.join(BASE_DIR, from_path)
        new_schema_path = os.path.join(BASE_DIR, to_path)

        output_dir = os.path.join("output", f"{from_version}-{to_version}")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        with open(old_schema_path) as f:
            old_esri_json_data = json.loads(f.read())
        with open(new_schema_path) as f:
            new_esri_json_data = json.loads(f.read())
        old_schema = transform_esri_json(
            old_esri_json_data, filter_prefix=FILTER_PREFIX
        )
        new_schema = transform_esri_json(
            new_esri_json_data, filter_prefix=FILTER_PREFIX
        )

        print(
            f"==================================================\n\n  Comparing {from_version} with {to_version}\n"
        )
        # print(old_schema.get_schema_summary())
        # print(new_schema.get_schema_summary())

        diff = SchemaDiff(old_schema, new_schema)
        print(diff.generate_report())

        # Check specific changes
        if diff.has_changes():
            for change in diff.domain_changes:
                if change.change_type == ChangeType.MODIFIED:
                    print(f"Domain {change.domain_name} was modified")

            for change in diff.table_changes:
                print("  ")
                print(f"Table {change.table_name}: {change.change_type}")
                for field in change.field_changes:
                    print(f"  {field.field_name}: {field.change_type}")

        # Get summary
        summary = diff.get_summary()
        print(f"Total domain changes: {sum(summary['domains'].values())}")

        diff_data = export_schema_diff_to_json(diff)
        schema_diff_json_path = os.path.join(output_dir, "schema-diff.json")
        logger.info(f"Saving the diff (from deepdiff) to {schema_diff_json_path}")
        with open(schema_diff_json_path, "w", encoding="utf-8") as f:
            json.dump(diff_data, f, ensure_ascii=False, indent=4)

        # Generate report
        import json

        from jinja2 import Template

        # Generate diff
        # diff = SchemaDiff(old_schema, new_schema)
        # diff_data = export_schema_diff_to_json(diff)

        # Load template
        with open("templates/schema_diff_template.md.j2") as f:
            template = Template(f.read())

        # Render markdown
        markdown_output = template.render(**diff_data)

        # Save result
        with open(os.path.join(output_dir, "schema_diff_report.md"), "w") as f:
            f.write(markdown_output)

        # Or convert to HTML using a markdown processor
        import markdown

        html = markdown.markdown(markdown_output, extensions=["tables", "extra"])
        with open(os.path.join(output_dir, "schema_diff_report.html"), "w") as f:
            f.write(html)

        # Load and render the template
        with open("templates/schema_diff_template.html.j2") as f:
            template = Template(f.read())

        html_output = template.render(**diff_data)

        # Save the result
        with open(os.path.join(output_dir, "schema_diff_full_report.html"), "w") as f:
            f.write(html_output)

        # Save the result
        with open(os.path.join(output_dir, "README.txt"), "w") as f:
            f.write(f"Old schema path: {old_schema_path}\n")
            f.write(f"New schema path: {new_schema_path}\n\n")

            f.write(str(old_schema.get_schema_summary()))
            f.write("\n")
            f.write(str(new_schema.get_schema_summary()))


if __name__ == "__main__":
    main()
