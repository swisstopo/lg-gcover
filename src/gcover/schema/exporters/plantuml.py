import re
from typing import Union

MAX_FIELDS_PER_TABLE = 80

IGNORE_FIELDS = [
    "OPERATOR",
    "DATEOFCREATION",
    "DATEOFCHANGE",
    "CREATION_YEAR",
    "CREATION_MONTH",
    "REVISION_YEAR",
    "REVISION_MONTH",
    "REASONFORCHANGE",
    "OBJECTORIGIN",
    "OBJECTORIGIN_YEAR",
    "OBJECTORIGIN_MONTH",
    "KIND",
    "RC_ID",
    "WU_ID",
    "RC_ID_CREATION",
    "WU_ID_CREATION",
    "REVISION_QUALITY",
    "ORIGINAL_ORIGIN",
    "INTEGRATION_OBJECT_UUID",
    "SHAPE.AREA",
    "SHAPE.LEN",
    "MORE_INFO",
]


SKINPARAMS = [
    "@startuml",
    "!theme plain",
    "skinparam pageWidth 420mm",
    "skinparam pageHeight 297mm",
    "skinparam dpi 150",
    "left to right direction",
    # Optimized for A3 landscape with centered core entities
    "skinparam linetype ortho",
    "skinparam nodesep 40",
    "skinparam ranksep 60",
    "skinparam minClassWidth 100",
    "skinparam linetype ortho",
    "skinparam backgroundcolor white",
    "skinparam defaultFontSize 12",
    'skinparam defaultFontName "Sans-Serif"',
]

"""
This is OK

ta

!define Table(name,desc) class name as desc << (T,#FFAAAA) >>

Table(TOPGIS_GC_GC_CONFLICT_ROW, "TOPGIS_GC.GC_CONFLICT_ROW") {
  + OBJECTID : esriFieldTypeOID

"""

import re
from typing import Union

from loguru import logger


def _ignore_field(fieldname):
    if fieldname in IGNORE_FIELDS:
        return True
    return False


def extract_prefix(name: str) -> str:
    """Extract prefix from a table/feature class name (e.g., TOPGIS_GC from TOPGIS_GC.GC_TABLE)"""
    if "." in name:
        return name.split(".")[0]
    return "default"


def sanitize_name(name: str) -> str:
    """Sanitize name for PlantUML (remove special characters)"""
    return re.sub(r"[^\w]", "_", name)


def remove_prefixes(
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


def _get_config(**kwargs):
    defaults = {
        "font_size": 10,
        "line_type": "ortho",
        "line_color": "#E1F5FE",
        "font_color": "#E1F5FE",
        "theme": "plain",
        "regular_table_color": "#E1F5FE",
        "spatial_table_color": "#E8F5E8",
        "junction_table_color": "#FFF8E1",
        "relation_table_color": "#F3E5F5",
        "bg_color": "#F3E5F5",
        "page_width": "420mm",  # A3 landscape width
        "page_height": "297mm",  # A3 landscape height
        "layout_direction": "left to right direction",
        "core_entities": ["BEDROCK", "UNCO_DEPOSIT"],  # Your core entities
    }

    config = {**defaults, **kwargs}

    return config


def _get_legend(config):
    lines = [
        "skinparam legend {",
        "backgroundColor  # GhostWhite",
        " entrySeparator  # GhostWhite",
        "}",
        "legend top left",
        " **Legend**",
        "  <:key:> = Primary Key",  # 🔑
        "  <:link:> = Foreign Key",  #  🔗
        "  <:globe_with_meridians:> = Geometry Field",  # 🌐
        "|  Entitiy  |= Color   |",
        f"|= Tables |<{config['regular_table_color']}>  |",
        f"|= Feature Classes |<{config['spatial_table_color']}>  |",
        f"|= Junction Tables |<{config['junction_table_color']}>  |",
        f"|= Domains |<{config['relation_table_color']}>  |",
        "endlegend",
        "",
    ]

    return lines


def generate_plantuml_from_schema(
    schema: "ESRISchema",
    title: str = "Database Schema",
    show_fields: bool = True,
    show_domains: bool = True,
    show_relationships: bool = True,
    show_junction_tables: bool = True,
    max_fields_per_table: int = 20,
    add_legend: bool = True,
    group_by_prefix: bool = True,
    prefix: str = None,
) -> str:
    """
    Generate a PlantUML diagram from an ESRISchema instance.

    Args:
        schema: ESRISchema instance to visualize
        title: Title for the diagram
        show_fields: Whether to show fields in tables
        show_domains: Whether to show domain relationships (set to False to hide domains completely)
        show_relationships: Whether to show relationships
        show_junction_tables: Whether to show junction tables
        max_fields_per_table: Maximum number of fields to show per table (0 for all)
        add_legend: Add legend to the diagram
        group_by_prefix: Whether to group tables by prefix (e.g., TOPGIS_GC)
        prefix: Prefix to remove from all names (e.g., "TOPGIS_GC.")

    Returns:
        PlantUML diagram as a string
    """
    # First, infer keys from relationships
    schema.infer_keys_from_relationships()

    # Then auto-detect remaining primary keys
    schema.detect_primary_keys()

    lines = SKINPARAMS + [
        "",
        f"title {title}",
        "",
        "skinparam class {",
        "    BackgroundColor #FFFFCC",
        "    BorderColor Black",
        "    ArrowColor Black",
        "}",
        "",
        # "hide empty members",
        "",
    ]

    # Organize tables/feature classes by prefix if requested
    groups = {}

    config = _get_config()

    # Add legend
    if add_legend:
        lines += _get_legend(config)

    # Process tables
    for table_name, table in schema.tables.items():
        prefix = extract_prefix(table_name) if group_by_prefix else "default"
        if prefix not in groups:
            groups[prefix] = {"tables": [], "feature_classes": []}
        groups[prefix]["tables"].append((table_name, table))

    # Process feature classes
    for fc_name, fc in schema.feature_classes.items():
        prefix = extract_prefix(fc_name) if group_by_prefix else "default"
        if prefix not in groups:
            groups[prefix] = {"tables": [], "feature_classes": []}
        groups[prefix]["feature_classes"].append((fc_name, fc))

    # Generate diagram content
    for prefix, items in sorted(groups.items()):
        if group_by_prefix and prefix != "default":
            lines.append(f'package "{prefix}" {{')

        # Add tables
        for table_name, table in items["tables"]:
            lines.extend(
                generate_table_uml(
                    table_name, table, show_fields, max_fields_per_table, config
                )
            )
            lines.append("")

        # Add feature classes
        for fc_name, fc in items["feature_classes"]:
            lines.extend(
                generate_feature_class_uml(
                    fc_name, fc, show_fields, max_fields_per_table, config
                )
            )
            lines.append("")

        # TODO
        # Add junction tables if requested
        if show_junction_tables and show_relationships:
            junction_lines = generate_junction_tables(schema, prefix, config)
            if junction_lines:
                lines.append("' Junction Tables for Many-to-Many Relationships")
                lines.extend(junction_lines)

        if group_by_prefix and prefix != "default":
            lines.append("}")
            lines.append("")

    # Add domains if requested
    if show_domains and (schema.coded_domains or schema.range_domains):
        lines.append("' Domains section (optional - set show_domains=False to hide)")
        lines.append('package "Domains" <<Frame>> {')

        # Coded domains
        for domain_name, domain in schema.coded_domains.items():
            lines.extend(
                generate_domain_uml(domain_name, domain, "coded", prefix, config)
            )
            lines.append("")

        # Range domains
        for domain_name, domain in schema.range_domains.items():
            lines.extend(
                generate_domain_uml(domain_name, domain, "range", prefix, config)
            )
            lines.append("")

        lines.append("}")
        lines.append("")

    # Add relationships
    if show_relationships:
        lines.append("' Relationships")

        if show_junction_tables:
            # Add junction table relationships for M:N
            junction_rels = generate_junction_relationships(schema, prefix, config)
            if junction_rels:
                lines.extend(junction_rels)
                lines.append("")

            # Add regular relationships (skip M:N as they're handled by junction tables)
            for rel_name, rel in schema.relationships.items():
                if rel.cardinality not in [
                    "ManyToMany",
                    "esriRelCardinalityManyToMany",
                ]:
                    lines.append(generate_relationship_uml(rel, prefix, config))
        else:
            # Add all relationships as-is
            for rel_name, rel in schema.relationships.items():
                lines.append(generate_relationship_uml(rel, prefix, config))

        # Add domain relationships only if domains are shown
        if show_domains:
            lines.append("")
            lines.append("' Domain usage relationships")
            lines.extend(generate_domain_relationships(schema, prefix, config))

    lines.append("")
    lines.append("@enduml")

    return "\n".join(lines)


def generate_field_line(field, pk_name=None, fk_dict=None, shape_field=None):
    """Generate a single field line for PlantUML with proper escaping"""
    if fk_dict is None:
        fk_dict = {}

    field_type = field.type or "Unknown"

    # Escape special characters in field names and types
    field_name = field.name.replace('"', '\\"')
    field_type = field_type.replace('"', '\\"')

    '''# Use Unicode symbols instead of emoji
    if field.name == pk_name:
        # Use ▶ or ● or ★ instead of 🔑
        field_line = f"  + <b>▶ {field_name}</b> : {field_type}"
    elif field.name in fk_dict:
        # Use → or ▷ instead of 🔗
        field_line = f"  + ▷ {field_name} : {field_type} → {ref_table_short}"
    elif field.name == shape_field:
        # Use ◉ or ⬡ instead of 🌐
        field_line = f"  + ◉ {field_name} : Geometry"'''

    if field.name == pk_name:
        field_line = f"  + {{field}} <:key:><b>{field_name}</b> : {field_type} <<PK>>"
        logger.debug(pk_name)
    elif field.name == shape_field:
        field_line = (
            f"  + {{field}}  <:globe_with_meridians:> {field_name} : Geometry <<SHAPE>>"
        )
        logger.debug("field.name: GEOMETRY")
    else:
        field_line = f"  + {field_name} : {field_type}"
        logger.debug(f"{field.name}, {pk_name}")
        # Add annotations
        annotations = []
        if field.name in fk_dict:
            annotations.append("FK")
        if field.domain:
            # Escape domain name
            domain_name = field.domain.replace('"', '\\"')
            annotations.append(f"domain:{domain_name}")
        if not field.nullable:
            annotations.append("NOT NULL")

        if annotations:
            field_line += f" <<{', '.join(annotations)}>>"

    return field_line
    """Generate PlantUML for a table"""
    safe_name = sanitize_name(table_name)
    display_name = table.alias or table_name

    lines = [f'Table({safe_name}, "{display_name}") {{']

    if show_fields:
        # Add primary key first
        if table.primary_key:
            pk_field = next(
                (f for f in table.fields if f.name == table.primary_key), None
            )
            if pk_field:
                lines.append(
                    f"  + {{field}} <b>{pk_field.name}</b> : {pk_field.type or 'Unknown'} <<PK>>"
                )

        # Add other fields
        field_count = 0
        for field in table.fields:
            if max_fields > 0 and field_count >= max_fields:
                lines.append("  .. truncated ..")
                break

            if field.name != table.primary_key:  # Skip if already shown as PK
                field_type = field.type or "Unknown"
                field_line = f"  + {field.name} : {field_type}"

                # Add annotations
                annotations = []
                if field.name in table.foreign_keys:
                    annotations.append("FK")
                if field.domain:
                    annotations.append(f"domain:{field.domain}")
                if not field.nullable:
                    annotations.append("NOT NULL")

                if annotations:
                    field_line += f" <<{', '.join(annotations)}>>"

                lines.append(field_line)
                field_count += 1

    lines.append("}")
    return lines


def generate_feature_class_uml(
    fc_name: str, fc: "FeatureClass", show_fields: bool, max_fields: int, config: dict
) -> list[str]:
    """Generate PlantUML for a feature class"""
    safe_name = sanitize_name(fc_name)
    display_name = (fc.alias or fc_name).replace('"', '\\"')
    # TODO
    safe_name = remove_prefixes(fc_name)
    display_name = safe_name

    # Add geometry type to display name
    # TODO
    geom_type = fc.geometry_type or fc.feature_type or "Geometry"
    # display_name = f"{display_name} [{geom_type}]"
    display_name = f"{display_name}"

    # lines = [f'FeatureClass({safe_name}, "{display_name}") {{']
    lines = [
        f'entity "{safe_name} " as {display_name} <<Featureclass>> {config["spatial_table_color"]}   {{'
    ]

    if show_fields:
        logger.debug(f"PK: , {fc.primary_key}")
        # Add primary key first
        if fc.primary_key:
            pk_field = next((f for f in fc.fields if f.name == fc.primary_key), None)
            if pk_field:
                lines.append(
                    generate_field_line(pk_field, fc.primary_key, fc.foreign_keys)
                )

        # Add geometry field
        if fc.shape_field:
            shape_field_obj = next(
                (f for f in fc.fields if f.name == fc.shape_field), None
            )
            if shape_field_obj:
                lines.append(
                    generate_field_line(
                        shape_field_obj, fc.primary_key, fc.foreign_keys, fc.shape_field
                    )
                )
            else:
                # If shape field not in fields list, add it anyway
                lines.append(f"  + {{field}} {fc.shape_field} : Geometry <<SHAPE>>")

        # Add other fields
        field_count = 0
        for field in fc.fields:
            if max_fields > 0 and field_count >= max_fields:
                lines.append("  .. truncated ..")
                break

            # Skip if already shown
            if field.name not in [fc.primary_key, fc.shape_field] and not _ignore_field(
                field.name
            ):
                lines.append(
                    generate_field_line(field, fc.primary_key, fc.foreign_keys)
                )
                field_count += 1

    lines.append("}")
    return lines


def generate_domain_uml(
    domain_name: str,
    domain: Union["CodedDomain", "RangeDomain"],
    domain_type: str,
    prefix: str,
    config: dict,
) -> list[str]:
    """Generate PlantUML for a domain"""
    # TODO
    safe_name = sanitize_name(domain_name)
    safe_name = remove_prefixes(fc_name)
    display_name = safe_name

    # Handle case where domain might be None or invalid
    if not domain or not hasattr(domain, "name"):
        display_name = domain_name
    else:
        display_name = domain.description if domain.description else domain_name

    lines = [f'Domain({safe_name}, "{display_name}") {{']

    if domain_type == "coded" and hasattr(domain, "coded_values"):
        lines.append("  --")
        # Show first few coded values
        for i, cv in enumerate(domain.coded_values[:5]):
            lines.append(f"  {cv.code} : {cv.name}")
        if len(domain.coded_values) > 5:
            lines.append(f"  ... ({len(domain.coded_values) - 5} more values)")
    elif domain_type == "range" and hasattr(domain, "min_value"):
        lines.append(f"  Range: {domain.min_value} - {domain.max_value}")

    lines.append("}")
    return lines


def generate_relationship_uml(
    rel: "RelationshipClass", prefix: str, config: dict
) -> str:
    """Generate PlantUML relationship notation"""
    origin_safe = sanitize_name(rel.origin_table)
    dest_safe = sanitize_name(rel.destination_table)

    # TODO
    origin_safe = remove_prefixes(rel.origin_table)
    dest_safe = remove_prefixes(rel.destination_table)

    # Determine cardinality notation
    cardinality_map = {
        "OneToOne": "||--||",
        "OneToMany": "||--o{",
        "ManyToMany": "}o--o{",
        "esriRelCardinalityOneToOne": "||--||",
        "esriRelCardinalityOneToMany": "||--o{",
        "esriRelCardinalityManyToMany": "}o--o{",
    }

    notation = cardinality_map.get(rel.cardinality or rel.relationship_type, "--")

    # Build relationship label
    label_parts = []
    if rel.forward_label:
        label_parts.append(rel.forward_label)
    if rel.is_composite:
        label_parts.append("composite")

    label = f' : "{" / ".join(label_parts)}"' if label_parts else ""

    logger.debug(f"forward {rel.forward_label}, ")

    return f"{origin_safe} {notation} {dest_safe}{label}"


def generate_domain_relationships(schema: "ESRISchema") -> list[str]:
    """Generate relationships between tables/feature classes and their domains"""
    lines = []
    domain_usage = {}  # domain_name -> list of (table_name, field_name)

    # Collect domain usage from all tables and feature classes
    all_objects = list(schema.tables.items()) + list(schema.feature_classes.items())

    for obj_name, obj in all_objects:
        for field in obj.fields:
            if field.domain:
                if field.domain not in domain_usage:
                    domain_usage[field.domain] = []
                domain_usage[field.domain].append((obj_name, field.name))

    # Generate domain relationship lines
    for domain_name, usages in domain_usage.items():
        safe_domain = sanitize_name(domain_name)
        # TODO
        safe_domain = remove_prefixes(domain_name)

        # Check if domain exists
        if domain_name in schema.coded_domains or domain_name in schema.range_domains:
            for table_name, field_name in usages:
                safe_table = sanitize_name(table_name)
                safe_table = remove_prefixes(table_name)  # TODO

                lines.append(f"{safe_table} ..> {safe_domain} : uses on {field_name}")

    return lines


def generate_simplified_plantuml(
    schema: "ESRISchema",
    title: str = "Database Schema Overview",
    prefix: str = "TOPGIS_GC.",
) -> str:
    """
    Generate a simplified PlantUML diagram showing only tables and relationships.

    Args:
        schema: ESRISchema instance to visualize
        title: Title for the diagram

    Returns:
        Simplified PlantUML diagram as a string
    """
    schema.infer_keys_from_relationships()

    config = _get_config()

    lines = [
        "@startuml",
        "!theme plain",
        "",
        f"title {title}",
        "",
        "skinparam class {",
        "    BackgroundColor LightYellow",
        "    BorderColor Black",
        "}",
        "skinparam linetype ortho",
        "",
    ]

    # Add all tables and feature classes as simple boxes
    all_objects = list(schema.tables.items()) + list(schema.feature_classes.items())

    for obj_name, obj in all_objects:
        safe_name = sanitize_name(obj_name)
        display_name = getattr(obj, "alias", None) or obj_name
        # TODO
        safe_name = remove_prefixes(obj_name)
        display_name = safe_name
        lines.append(f'class {safe_name} as "{display_name}"')

    lines.append("")

    # Add relationships
    for rel in schema.relationships.values():
        lines.append(generate_relationship_uml(rel, prefix, config))

    lines.append("")
    lines.append("@enduml")

    return "\n".join(lines)


# Generated by Claude, but he's lazy and never print it :-(
def generate_junction_tables(
    schema: "ESRISchema", remove_prefix: str, config: dict
) -> list[str]:
    """
    Generate junction/association tables for many-to-many relationships.
    Returns PlantUML class definitions for junction tables.
    """
    lines = []

    for rel_name, rel in schema.relationships.items():
        # Check if it's a many-to-many relationship
        if rel.cardinality in ["ManyToMany", "esriRelCardinalityManyToMany"]:
            # Generate junction table name
            # clean_rel_name = clean_name(rel_name, remove_prefix)
            clean_rel_name = remove_prefixes(rel_name)
            safe_name = sanitize_name(
                rel_name + "_JT"
            )  # Add _JT suffix for junction table

            # Extract table names for display
            # origin_clean = clean_name(rel.origin_table, remove_prefix)
            # dest_clean = clean_name(rel.destination_table, remove_prefix)
            origin_clean = remove_prefixes(rel.origin_table)
            dest_clean = remove_prefixes(rel.destination_table)

            display_name = f"{clean_rel_name} (Junction)"

            # Create junction table with composite color (light purple)
            lines.append(
                f'class {safe_name} as "{display_name}"  {config["junction_table_color"]} {{'
            )  #  #E6E6FA
            lines.append("  .. Junction Table ..")

            # Add foreign keys as composite primary key
            if rel.origin_foreign_key:
                lines.append(
                    f"  + <b>🔑🔗 {rel.origin_foreign_key}</b> : UUID → {origin_clean}"
                )
            else:
                lines.append(
                    f"  + <b>🔑🔗 {origin_clean}_ID</b> : UUID → {origin_clean}"
                )

            if rel.destination_foreign_key:
                lines.append(
                    f"  + <b>🔑🔗 {rel.destination_foreign_key}</b> : UUID → {dest_clean}"
                )
            else:
                lines.append(f"  + <b>🔑🔗 {dest_clean}_ID</b> : UUID → {dest_clean}")

            # Add any additional fields that might be on the relationship
            if rel.is_attributed:
                lines.append("  --")
                lines.append("  + RELATIONSHIP_DATE : Date")
                lines.append("  + RELATIONSHIP_TYPE : String")

            lines.append("}")
            lines.append("")

    return lines


def generate_junction_relationships(
    schema: "ESRISchema", remove_prefix: str, config: dict
) -> list[str]:
    """
    Generate relationships between junction tables and their connected tables.
    Replaces the many-to-many relationship with two one-to-many relationships.
    """
    lines = []

    for rel_name, rel in schema.relationships.items():
        if rel.cardinality in ["ManyToMany", "esriRelCardinalityManyToMany"]:
            origin_safe = sanitize_name(rel.origin_table)
            dest_safe = sanitize_name(rel.destination_table)
            junction_safe = sanitize_name(rel_name + "_JT")
            origin_safe = remove_prefixes(rel.origin_table)
            dest_safe = remove_prefixes(rel.destination_table)

            # Origin to Junction (one-to-many)
            lines.append(f'{origin_safe} ||--o{{ {junction_safe} : "has many"')

            # Junction to Destination (many-to-one)
            lines.append(f'{junction_safe} }}o--|| {dest_safe} : "references"')

    return lines


def generate_table_uml(
    fc_name: str, fc: "Table", show_fields: bool, max_fields: int, config: dict
) -> list[str]:
    """Generate PlantUML for a feature class"""
    safe_name = sanitize_name(fc_name)
    display_name = (fc.alias or fc_name).replace('"', '\\"')
    # TODO
    safe_name = remove_prefixes(fc_name)
    display_name = safe_name

    """entity  TOPGIS_GC_GC_CONFLICT_ROW <<T>> #FFE4B5 { """

    lines = [f'Table({safe_name}, "{display_name}") {{']

    # lines =  [f'entity ({safe_name} <<Table>> #FFE4B5 {{']
    lines = [
        f'entity "{safe_name} " as {display_name} <<Table>> {config["regular_table_color"]} {{'
    ]

    if show_fields:
        # Add primary key first
        if fc.primary_key:
            pk_field = next((f for f in fc.fields if f.name == fc.primary_key), None)
            if pk_field:
                lines.append(
                    generate_field_line(pk_field, fc.primary_key, fc.foreign_keys)
                )

        # Add geometry field
        """ if fc.shape_field:
            shape_field_obj = next((f for f in fc.fields if f.name == fc.shape_field), None)
            if shape_field_obj:
                lines.append(generate_field_line(shape_field_obj, fc.primary_key, fc.foreign_keys, fc.shape_field))
            else:
                # If shape field not in fields list, add it anyway
                lines.append(f"  + {{field}} {fc.shape_field} : Geometry <<SHAPE>>")"""

        # Add other fields
        field_count = 0
        for field in fc.fields:
            if max_fields > 0 and field_count >= max_fields:
                lines.append("  .. truncated ..")
                break

            # Skip if already shown
            if field.name not in [fc.primary_key] and not _ignore_field(field.name):
                lines.append(
                    generate_field_line(field, fc.primary_key, fc.foreign_keys)
                )
                field_count += 1

    lines.append("}")
    return lines


def main():
    import json
    import os
    from datetime import datetime

    from transform_esri_json import transform_esri_json

    now = datetime.today().strftime("%B %Y")

    DIAGRAM_TITLE = f"Gecover 4.1 —  {now}"

    if os.name == "nt":
        BASE_DIR = r"H:\code\lg-geology-data-model\exports"
    else:
        BASE_DIR = "/home/marco/code/github.com/lg-geology-data-model/exports/"
    old_json_path = os.path.join(BASE_DIR, "2025_05_26/GCOVERP_export.json")

    logger.info(f"Loading ESRI report from: {old_json_path}")
    with open(old_json_path) as f:
        esri_json_data = json.loads(f.read())
    logger.info("Generating ESRI Schema...")
    schema = transform_esri_json(esri_json_data, exclude_metadata_fields=False)
    logger.info(f"Loaded {len(schema.feature_classes)} feature classes")
    logger.info(f"Loaded {len(schema.coded_domains)} coded domains")
    logger.info(f"Total subtypes: {schema.get_subtype_count()}")
    logger.info(f"Subtypes summary: {schema.get_subtype_summary()}")
    logger.info(f"Loaded {len(schema.tables)} tables")

    logger.info(schema.get_schema_summary())
    # Generate full diagram
    puml_full = generate_plantuml_from_schema(
        schema,
        show_domains=False,
        max_fields_per_table=MAX_FIELDS_PER_TABLE,
        group_by_prefix=False,
        title=DIAGRAM_TITLE,
    )

    # Generate simplified diagram
    puml_simple = generate_simplified_plantuml(schema, title=DIAGRAM_TITLE)

    # Save to file
    puml_file_path = "schema_diagram.puml"
    logger.info(f"PUML saved to {puml_file_path}")
    with open(puml_file_path, "w") as f:
        f.write(puml_full)

    # Save to file
    puml_file_path = "simple_schema_diagram.puml"
    logger.info(f"PUML saved to {puml_file_path}")
    with open(puml_file_path, "w") as f:
        f.write(puml_simple)


if __name__ == "__main__":
    main()
