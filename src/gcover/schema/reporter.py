"""
Schema report generation module using Jinja2 templates.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger

from .differ import SchemaDiff, ChangeType


def _convert_enum_to_string(obj):
    """Convert ChangeType enums to strings for JSON serialization."""
    if isinstance(obj, ChangeType):
        return obj.value
    if isinstance(obj, dict):
        return {key: _convert_enum_to_string(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_enum_to_string(item) for item in obj]
    return obj


def schema_diff_to_dict(diff: SchemaDiff) -> Dict[str, Any]:
    """
    Convert SchemaDiff to dictionary format for templates.

    Args:
        diff: SchemaDiff instance

    Returns:
        Dictionary representation suitable for JSON export and templates
    """

    # Helper function to extract coded value changes
    def extract_coded_value_changes(change):
        if hasattr(change, "coded_value_changes") and change.coded_value_changes:
            return {
                code: change_type.value
                for code, change_type in change.coded_value_changes.items()
            }
        return {}

    # Helper function to extract property changes
    def extract_property_changes(change):
        if hasattr(change, "property_changes") and change.property_changes:
            return {
                prop: {"old": old, "new": new}
                for prop, (old, new) in change.property_changes.items()
            }
        return {}

    # Helper function to extract field changes
    def extract_field_changes(table_change):
        field_changes = []
        for field_change in table_change.field_changes:
            field_data = {
                "change_type": field_change.change_type.value,
                "name": field_change.field_name,
            }

            if field_change.property_changes:
                field_data["property_changes"] = {
                    prop: {"old": old, "new": new}
                    for prop, (old, new) in field_change.property_changes.items()
                }

            field_changes.append(field_data)
        return field_changes

    # Process domain changes
    domain_changes = []
    for change in diff.domain_changes:
        domain_data = {
            "change_type": change.change_type.value,
            "name": change.domain_name,
        }

        coded_value_changes = extract_coded_value_changes(change)
        if coded_value_changes:
            domain_data["coded_value_changes"] = coded_value_changes

        property_changes = extract_property_changes(change)
        if property_changes:
            domain_data["property_changes"] = property_changes

        domain_changes.append(domain_data)

    # Process table changes
    table_changes = []
    for change in diff.table_changes:
        table_data = {
            "change_type": change.change_type.value,
            "name": change.table_name,
        }

        field_changes = extract_field_changes(change)
        if field_changes:
            table_data["field_changes"] = field_changes

        property_changes = extract_property_changes(change)
        if property_changes:
            table_data["property_changes"] = property_changes

        table_changes.append(table_data)

    # Process feature class changes
    feature_class_changes = []
    for change in diff.feature_class_changes:
        fc_data = {"change_type": change.change_type.value, "name": change.table_name}

        field_changes = extract_field_changes(change)
        if field_changes:
            fc_data["field_changes"] = field_changes

        property_changes = extract_property_changes(change)
        if property_changes:
            fc_data["property_changes"] = property_changes

        feature_class_changes.append(fc_data)

    # Process relationship changes
    relationship_changes = []
    for change in diff.relationship_changes:
        rel_data = {
            "change_type": change.change_type.value,
            "name": change.relationship_name,
        }

        property_changes = extract_property_changes(change)
        if property_changes:
            rel_data["property_changes"] = property_changes

        relationship_changes.append(rel_data)

    # Process subtype changes
    subtype_changes = []
    for change in diff.subtype_changes:
        subtype_data = {
            "change_type": change.change_type.value,
            "name": change.subtype_name,
        }

        if hasattr(change, "value_changes") and change.value_changes:
            subtype_data["value_changes"] = {
                code: change_type.value
                for code, change_type in change.value_changes.items()
            }

        property_changes = extract_property_changes(change)
        if property_changes:
            subtype_data["property_changes"] = property_changes

        subtype_changes.append(subtype_data)

    # Build final structure
    result = {
        "summary": diff.get_summary(),
        "changes": {
            "domains": domain_changes,
            "tables": table_changes,
            "feature_classes": feature_class_changes,
            "relationships": relationship_changes,
            "subtypes": subtype_changes,
        },
        "has_changes": diff.has_changes(),
        "metadata": {
            "old_schema_name": getattr(diff.old_schema, "name", "Unknown"),
            "new_schema_name": getattr(diff.new_schema, "name", "Unknown"),
            "comparison_date": datetime.now().isoformat(),
        },
    }

    return _convert_enum_to_string(result)


def get_template_environment(template_dir: Optional[Path] = None) -> Environment:
    """
    Create Jinja2 environment with custom filters.

    Args:
        template_dir: Path to template directory. If None, uses default.

    Returns:
        Configured Jinja2 Environment
    """
    if template_dir is None:
        # Default to templates directory relative to this file
        template_dir = Path(__file__).parent / "templates"

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Custom filters
    def format_change_type(change_type: str) -> str:
        """Format change type for display."""
        icons = {"added": "➕", "removed": "➖", "modified": "🔄"}
        return f"{icons.get(change_type, '❓')} {change_type.title()}"

    def change_count(changes: List[Dict], change_type: str) -> int:
        """Count changes of a specific type."""
        return sum(1 for change in changes if change.get("change_type") == change_type)

    def total_changes(summary: Dict) -> int:
        """Calculate total number of changes from summary."""
        total = 0
        for category in summary.values():
            if isinstance(category, dict):
                total += sum(category.values())
        return total

    def format_datetime(iso_string: str) -> str:
        """Format ISO datetime string."""
        try:
            dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return iso_string

    # Register filters
    env.filters["format_change_type"] = format_change_type
    env.filters["change_count"] = change_count
    env.filters["total_changes"] = total_changes
    env.filters["format_datetime"] = format_datetime

    return env


def generate_report(
    diff: SchemaDiff,
    template: str = "full",
    format: str = "html",
    template_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate a report from SchemaDiff using Jinja2 templates.

    Args:
        diff: SchemaDiff instance
        template: Template type ("summary", "full", "minimal")
        format: Output format ("html", "markdown", "json")
        template_dir: Path to custom template directory
        output_file: Optional output file path

    Returns:
        Generated report as string
    """
    logger.info(f"Generating {format} report using {template} template")

    # Convert diff to dictionary
    diff_data = schema_diff_to_dict(diff)

    if format == "json":
        # Return JSON directly
        report_content = json.dumps(diff_data, indent=2, ensure_ascii=False)
    else:
        # Use Jinja2 template
        env = get_template_environment(template_dir)

        # Determine template file name
        template_name = f"schema_diff_{template}.{format}.j2"

        try:
            template_obj = env.get_template(template_name)
        except Exception as e:
            logger.error(f"Template {template_name} not found: {e}")
            # Fallback to inline template
            if format == "markdown":
                template_obj = Template(_get_default_markdown_template())
            else:
                template_obj = Template(_get_default_html_template())

        # Render template
        report_content = template_obj.render(**diff_data)

    # Save to file if specified
    if output_file:
        Path(output_file).write_text(report_content, encoding="utf-8")
        logger.info(f"Report saved to {output_file}")

    return report_content


def _get_default_markdown_template() -> str:
    """Default Markdown template as fallback."""
    return """# Schema Comparison Report

**Comparison Date:** {{ metadata.comparison_date | format_datetime }}  
**Old Schema:** `{{ metadata.old_schema_name }}`  
**New Schema:** `{{ metadata.new_schema_name }}`

## Summary

{% set total = summary | total_changes -%}
{% if total == 0 -%}
✅ **No changes detected**
{% else -%}
📊 **Total Changes:** {{ total }}

| Category | Added | Removed | Modified |
|----------|-------|---------|----------|
{% for category, counts in summary.items() -%}
| {{ category.replace('_', ' ').title() }} | {{ counts.added }} | {{ counts.removed }} | {{ counts.modified }} |
{% endfor -%}
{% endif %}

{% if changes.domains -%}
## Domain Changes

{% for change in changes.domains -%}
### {{ change.change_type | format_change_type }} {{ change.name }}

{% if change.coded_value_changes -%}
**Coded Value Changes:**
{% for code, change_type in change.coded_value_changes.items() -%}
- `{{ code }}`: {{ change_type | format_change_type }}
{% endfor %}
{% endif -%}

{% if change.property_changes -%}
**Property Changes:**
{% for prop, values in change.property_changes.items() -%}
- **{{ prop }}**: `{{ values.old }}` → `{{ values.new }}`
{% endfor %}
{% endif -%}

{% endfor -%}
{% endif -%}

{# Similar blocks for tables, feature_classes, relationships, subtypes #}
"""


def _get_default_html_template() -> str:
    """Default HTML template as fallback."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schema Comparison Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 0; padding: 2rem; background: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #2c3e50; }
        .summary { background: #e8f5e8; padding: 1rem; border-radius: 6px; margin: 1rem 0; }
        .change-item { margin: 1rem 0; padding: 1rem; border-left: 4px solid #3498db; background: #f7f9fc; }
        .added { border-color: #27ae60; background-color: #e8f5e8; }
        .removed { border-color: #e74c3c; background-color: #fdeaea; }
        .modified { border-color: #f39c12; background-color: #fef9e7; }
        .table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        .table th, .table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }
        .table th { background-color: #f8f9fa; font-weight: 600; }
        code { background: #f1f3f4; padding: 0.2rem 0.4rem; border-radius: 3px; font-size: 0.9em; }
        .badge { padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8em; font-weight: 500; }
        .badge.added { background: #d4edda; color: #155724; }
        .badge.removed { background: #f8d7da; color: #721c24; }
        .badge.modified { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Schema Comparison Report</h1>
        
        <div class="metadata">
            <p><strong>Comparison Date:</strong> {{ metadata.comparison_date | format_datetime }}</p>
            <p><strong>Old Schema:</strong> <code>{{ metadata.old_schema_name }}</code></p>
            <p><strong>New Schema:</strong> <code>{{ metadata.new_schema_name }}</code></p>
        </div>

        <h2>Summary</h2>
        {% set total = summary | total_changes -%}
        <div class="summary">
            {% if total == 0 -%}
            <p>✅ <strong>No changes detected</strong></p>
            {% else -%}
            <p>📊 <strong>Total Changes:</strong> {{ total }}</p>
            
            <table class="table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Added</th>
                        <th>Removed</th>
                        <th>Modified</th>
                    </tr>
                </thead>
                <tbody>
                    {% for category, counts in summary.items() -%}
                    <tr>
                        <td>{{ category.replace('_', ' ').title() }}</td>
                        <td><span class="badge added">{{ counts.added }}</span></td>
                        <td><span class="badge removed">{{ counts.removed }}</span></td>
                        <td><span class="badge modified">{{ counts.modified }}</span></td>
                    </tr>
                    {% endfor -%}
                </tbody>
            </table>
            {% endif -%}
        </div>

        {% if changes.domains -%}
        <h2>Domain Changes</h2>
        {% for change in changes.domains -%}
        <div class="change-item {{ change.change_type }}">
            <h3>{{ change.change_type | format_change_type }} {{ change.name }}</h3>
            
            {% if change.coded_value_changes -%}
            <h4>Coded Value Changes:</h4>
            <ul>
            {% for code, change_type in change.coded_value_changes.items() -%}
                <li><code>{{ code }}</code>: <span class="badge {{ change_type }}">{{ change_type }}</span></li>
            {% endfor -%}
            </ul>
            {% endif -%}
        </div>
        {% endfor -%}
        {% endif -%}

        {# Add similar sections for tables, feature classes, etc. #}
    </div>
</body>
</html>"""
