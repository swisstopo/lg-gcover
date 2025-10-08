#!/usr/bin/env python3
"""
MapServer Mapfile and QGIS QML Generator

Generate map server configuration from ESRI classification rules.
"""

import json
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

import click
from loguru import logger
from rich.console import Console

from gcover.publish.esri_classification_extractor import (
    ClassificationClass,
    ESRIClassificationExtractor,
    LayerClassification,
)

from .esri_classification_extractor import extract_lyrx

console = Console()


class MapServerGenerator:
    """Generate MapServer mapfile CLASS sections from classifications."""

    def __init__(
        self,
        layer_type: str = "Polygon",
        use_symbol_field: bool = False,
        symbol_field: str = "SYMBOL",
    ):
        """
        Initialize generator.

        Args:
            layer_type: POLYGON, LINE, or POINT
            use_symbol_field: If True, use simple SYMBOL field expressions instead of complex field logic
            symbol_field: Name of symbol field in data (default: SYMBOL)
        """
        self.layer_type = layer_type.upper()
        self.use_symbol_field = use_symbol_field
        self.symbol_field = symbol_field

    def generate_expression_from_fields(
        self, class_obj: ClassificationClass, field_names: List[str]
    ) -> str:
        """
        Generate MapServer EXPRESSION from classification field values.

        Args:
            class_obj: Classification class with field values
            field_names: List of field names used in classification

        Returns:
            MapServer EXPRESSION string
        """
        # Build expression for each value combination (OR logic)
        expressions = []

        for field_values in class_obj.field_values:
            # Build conditions for this combination (AND logic)
            conditions = []

            for field_name, expected_value in zip(field_names, field_values):
                if expected_value == "<Null>":
                    # MapServer: empty string or NULL
                    conditions.append(
                        f'([{field_name}] eq "" OR NOT [DEFINED_{field_name}])'
                    )
                elif expected_value == "999997":
                    conditions.append(f'([{field_name}] eq "999997")')
                elif expected_value == "999999":
                    conditions.append(f'([{field_name}] eq "999999")')
                else:
                    # Regular value - MapServer uses string comparison
                    conditions.append(f'([{field_name}] eq "{expected_value}")')

            # Combine with AND
            if len(conditions) == 1:
                expressions.append(conditions[0])
            else:
                expressions.append(f"({' AND '.join(conditions)})")

        # Combine combinations with OR
        if len(expressions) == 1:
            return expressions[0]
        else:
            return f"({' OR '.join(expressions)})"

    def generate_expression_from_symbol(self, symbol_id: str) -> str:
        """
        Generate simple MapServer EXPRESSION using SYMBOL field.

        Args:
            symbol_id: Symbol identifier (e.g., "bedrock_1204", "unco_5")

        Returns:
            MapServer EXPRESSION string
        """
        return f'([{self.symbol_field}] eq "{symbol_id}")'

    def generate_class(
        self,
        class_obj: ClassificationClass,
        field_names: List[str],
        class_index: int,
        symbol_prefix: str = "class",
    ) -> str:
        """
        Generate a single MapServer CLASS block.

        Args:
            class_obj: Classification class with styling info
            field_names: Field names for expression (if not using symbol field)
            class_index: Class index for symbol ID generation
            symbol_prefix: Prefix for symbol IDs (e.g., "bedrock", "unco")

        Returns:
            MapServer CLASS block as string
        """
        if not class_obj.visible:
            return ""

        # Generate expression based on mode
        if self.use_symbol_field:
            symbol_id = f"{symbol_prefix}_{class_index}"
            expression = self.generate_expression_from_symbol(symbol_id)
        else:
            expression = self.generate_expression_from_fields(class_obj, field_names)

        # Sanitize class name
        class_name = class_obj.label.replace('"', '\\"')

        # Build CLASS block
        lines = [
            "  CLASS",
            f'    NAME "{class_name}"',
            f"    EXPRESSION {expression}",
        ]

        # Add STYLE block based on layer type and extracted symbol info
        lines.append("    STYLE")

        if self.layer_type == "POLYGON":
            self._add_polygon_style(lines, class_obj)
        elif self.layer_type == "LINE":
            self._add_line_style(lines, class_obj)
        elif self.layer_type == "POINT":
            self._add_point_style(lines, class_obj)

        lines.append("    END # STYLE")
        lines.append("  END # CLASS")
        lines.append("")

        return "\n".join(lines)

    def _add_polygon_style(self, lines: List[str], class_obj: ClassificationClass):
        """Add polygon styling from ESRI symbol_info."""
        # Extract color from symbol_info (ColorInfo object)
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Fill color
            if hasattr(symbol_info, "color") and symbol_info.color:
                color_info = symbol_info.color
                if hasattr(color_info, "r"):
                    lines.append(
                        f"      COLOR {color_info.r} {color_info.g} {color_info.b}"
                    )

                    # Transparency
                    if hasattr(color_info, "alpha"):
                        opacity = int(color_info.alpha)
                        lines.append(f"      OPACITY {opacity}")
                else:
                    lines.append("      COLOR 128 128 128")
            else:
                lines.append("      COLOR 128 128 128")

            # Outline - extract from raw_symbol
            if hasattr(symbol_info, "raw_symbol") and symbol_info.raw_symbol:
                raw = symbol_info.raw_symbol
                if "symbolLayers" in raw:
                    for layer in raw["symbolLayers"]:
                        if layer.get("type") == "CIMSolidStroke" and "color" in layer:
                            color_data = layer["color"]
                            values = color_data.get("values", [])
                            if len(values) >= 3:
                                if color_data["type"] == "CIMCMYKColor":
                                    c, m, y, k = values[:4]
                                    r = int(255 * (1 - c / 100) * (1 - k / 100))
                                    g = int(255 * (1 - m / 100) * (1 - k / 100))
                                    b = int(255 * (1 - y / 100) * (1 - k / 100))
                                else:
                                    r, g, b = values[:3]
                                lines.append(f"      OUTLINECOLOR {r} {g} {b}")

                                # Outline width
                                if "width" in layer:
                                    width = layer["width"] * 1.33  # Points to pixels
                                    lines.append(f"      WIDTH {width:.2f}")
                                break
                    else:
                        lines.append("      OUTLINECOLOR 64 64 64")
                else:
                    lines.append("      OUTLINECOLOR 64 64 64")
            else:
                lines.append("      OUTLINECOLOR 64 64 64")
        else:
            # Fallback defaults
            lines.append("      COLOR 128 128 128")
            lines.append("      OUTLINECOLOR 64 64 64")

    def _add_line_style(self, lines: List[str], class_obj: ClassificationClass):
        """Add line styling from ESRI symbol_info."""
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Line color
            if hasattr(symbol_info, "color") and symbol_info.color:
                color_info = symbol_info.color
                if hasattr(color_info, "r"):
                    lines.append(
                        f"      COLOR {color_info.r} {color_info.g} {color_info.b}"
                    )
                else:
                    lines.append("      COLOR 128 128 128")
            else:
                lines.append("      COLOR 128 128 128")

            # Line width
            if hasattr(symbol_info, "width") and symbol_info.width:
                width = symbol_info.width * 1.33  # Points to pixels
                lines.append(f"      WIDTH {width:.2f}")
            else:
                lines.append("      WIDTH 1.0")
        else:
            lines.append("      COLOR 128 128 128")
            lines.append("      WIDTH 1.0")

    def _add_point_style(self, lines: List[str], class_obj: ClassificationClass):
        """Add point styling from ESRI symbol_info."""
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Marker type
            marker_type = "circle"
            if hasattr(symbol_info, "symbol_type"):
                type_str = str(symbol_info.symbol_type)
                if "Square" in type_str:
                    marker_type = "square"
                elif "Triangle" in type_str:
                    marker_type = "triangle"
                elif "Star" in type_str:
                    marker_type = "star"

            lines.append(f'      SYMBOL "{marker_type}"')

            # Size
            if hasattr(symbol_info, "size") and symbol_info.size:
                size = symbol_info.size * 1.33  # Points to pixels
                lines.append(f"      SIZE {size:.1f}")
            else:
                lines.append("      SIZE 8")

            # Color
            if hasattr(symbol_info, "color") and symbol_info.color:
                color_info = symbol_info.color
                if hasattr(color_info, "r"):
                    lines.append(
                        f"      COLOR {color_info.r} {color_info.g} {color_info.b}"
                    )
                else:
                    lines.append("      COLOR 128 128 128")
            else:
                lines.append("      COLOR 128 128 128")
        else:
            lines.append('      SYMBOL "circle"')
            lines.append("      SIZE 8")
            lines.append("      COLOR 128 128 128")

    def generate_layer(
        self,
        classification: LayerClassification,
        layer_name: str,
        data_path: str,
        symbol_prefix: str = "class",
    ) -> str:
        """
        Generate complete MapServer LAYER block.

        Args:
            classification: Layer classification with styling
            layer_name: Name for the layer
            data_path: Path to data source (e.g., OGR connection string)
            symbol_prefix: Prefix for symbol IDs (e.g., "bedrock", "unco")

        Returns:
            Complete LAYER block
        """
        lines = [
            "LAYER",
            f'  NAME "{layer_name}"',
            f"  TYPE {self.layer_type}",
            f'  DATA "{data_path}"',
            "  STATUS ON",
            "",
        ]

        if self.use_symbol_field:
            lines.append(f"  # Styled using {self.symbol_field} field")
        else:
            lines.append("  # Styled using classification field values")

        lines.append("")

        # Generate CLASS blocks
        field_names = [f.name for f in classification.fields]

        for i, class_obj in enumerate(classification.classes):
            if not class_obj.visible:
                continue

            class_block = self.generate_class(class_obj, field_names, i, symbol_prefix)
            lines.append(class_block)

        lines.append("END # LAYER")

        return "\n".join(lines)

    def generate_symbol_file(self) -> str:
        """Generate MapServer symbol file with basic symbols."""
        lines = [
            "SYMBOLSET",
            "",
            "  # Basic symbols for GeoCover",
            "",
            "  SYMBOL",
            '    NAME "circle"',
            "    TYPE ELLIPSE",
            "    POINTS 1 1 END",
            "    FILLED TRUE",
            "  END",
            "",
            "  SYMBOL",
            '    NAME "square"',
            "    TYPE VECTOR",
            "    POINTS",
            "      0 0",
            "      0 1",
            "      1 1",
            "      1 0",
            "      0 0",
            "    END",
            "    FILLED TRUE",
            "  END",
            "",
            "  SYMBOL",
            '    NAME "triangle"',
            "    TYPE VECTOR",
            "    POINTS",
            "      0 1",
            "      0.5 0",
            "      1 1",
            "      0 1",
            "    END",
            "    FILLED TRUE",
            "  END",
            "",
            "  SYMBOL",
            '    NAME "star"',
            "    TYPE VECTOR",
            "    POINTS",
            "      0 0.375",
            "      0.35 0.375",
            "      0.5 0",
            "      0.65 0.375",
            "      1 0.375",
            "      0.75 0.625",
            "      0.875 1",
            "      0.5 0.75",
            "      0.125 1",
            "      0.25 0.625",
            "      0 0.375",
            "    END",
            "    FILLED TRUE",
            "  END",
            "",
            "  SYMBOL",
            '    NAME "dashed"',
            "    TYPE SIMPLE",
            "    PATTERN 10 5 END",
            "  END",
            "",
            "  SYMBOL",
            '    NAME "dotted"',
            "    TYPE SIMPLE",
            "    PATTERN 2 4 END",
            "  END",
            "",
            "END # SYMBOLSET",
        ]

        return "\n".join(lines)


class QGISGenerator:
    """Generate QGIS QML style files from classifications."""

    def __init__(
        self,
        geometry_type: str = "Polygon",
        use_symbol_field: bool = False,
        symbol_field: str = "SYMBOL",
    ):
        """
        Initialize generator.

        Args:
            geometry_type: Polygon, Line, or Point
            use_symbol_field: If True, use SYMBOL field for rules
            symbol_field: Name of symbol field
        """
        self.geometry_type = geometry_type
        self.use_symbol_field = use_symbol_field
        self.symbol_field = symbol_field
        self.symbol_counter = 0

    def _build_expression_from_fields(
        self, class_obj: ClassificationClass, field_names: List[str]
    ) -> str:
        """Build QGIS expression from classification field values."""
        expressions = []

        for field_values in class_obj.field_values:
            conditions = []

            for field_name, expected_value in zip(field_names, field_values):
                if expected_value == "<Null>":
                    conditions.append(f'"{field_name}" IS NULL')
                elif expected_value == "999997":
                    conditions.append(f'"{field_name}" = 999997')
                elif expected_value == "999999":
                    conditions.append(f'"{field_name}" = 999999')
                else:
                    # Try to determine if numeric or string
                    try:
                        int(expected_value)
                        conditions.append(f'"{field_name}" = {expected_value}')
                    except ValueError:
                        conditions.append(f"\"{field_name}\" = '{expected_value}'")

            # Combine with AND
            if len(conditions) == 1:
                expressions.append(conditions[0])
            else:
                expressions.append(f"({' AND '.join(conditions)})")

        # Combine with OR
        if len(expressions) == 1:
            return expressions[0]
        else:
            return f"({' OR '.join(expressions)})"

    def _build_expression_from_symbol(self, symbol_id: str) -> str:
        """Build simple QGIS expression using SYMBOL field."""
        return f"\"{self.symbol_field}\" = '{symbol_id}'"

    def generate_qml(
        self, classification: LayerClassification, symbol_prefix: str = "class"
    ) -> str:
        """
        Generate QGIS QML style file.

        Args:
            classification: Layer classification with styling
            symbol_prefix: Prefix for symbol IDs

        Returns:
            QML XML as string
        """
        import uuid

        # Start with DOCTYPE and root
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>",
            '<qgis version="3.40.0" styleCategories="Symbology">',
        ]

        # Renderer
        geometry_map = {"Polygon": "fill", "Line": "line", "Point": "marker"}
        symbol_type = geometry_map.get(self.geometry_type, "fill")

        lines.append(
            '  <renderer-v2 type="RuleRenderer" forceraster="0" enableorderby="0" symbollevels="0" referencescale="-1">'
        )

        # Rules section
        rules_key = str(uuid.uuid4())
        lines.append(f'    <rules key="{{{rules_key}}}">')

        # Collect visible classes and their symbols
        visible_classes = [
            (i, c) for i, c in enumerate(classification.classes) if c.visible
        ]
        field_names = [f.name for f in classification.fields]

        # Generate rules
        for symbol_idx, (class_idx, class_obj) in enumerate(visible_classes):
            rule_key = str(uuid.uuid4())

            # Build filter
            if self.use_symbol_field:
                symbol_id = f"{symbol_prefix}_{class_idx}"
                filter_expr = self._build_expression_from_symbol(symbol_id)
            else:
                filter_expr = self._build_expression_from_fields(class_obj, field_names)

            # Escape XML special characters in label and filter
            label_escaped = (
                class_obj.label.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            # For filter, escape quotes as &quot; for QGIS
            filter_escaped = (
                filter_expr.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

            lines.append(
                f'      <rule symbol="{symbol_idx}" label="{label_escaped}" filter="{filter_escaped}" key="{{{rule_key}}}"/>'
            )

        lines.append("    </rules>")

        # Symbols section
        lines.append("    <symbols>")

        for symbol_idx, (class_idx, class_obj) in enumerate(visible_classes):
            if self.geometry_type == "Polygon":
                symbol_xml = self._generate_polygon_symbol(symbol_idx, class_obj)
            elif self.geometry_type == "Line":
                symbol_xml = self._generate_line_symbol(symbol_idx, class_obj)
            elif self.geometry_type == "Point":
                symbol_xml = self._generate_point_symbol(symbol_idx, class_obj)

            lines.append(symbol_xml)

        lines.append("    </symbols>")
        lines.append("  </renderer-v2>")
        lines.append("</qgis>")

        return "\n".join(lines)

    def _generate_polygon_symbol(
        self, symbol_idx: int, class_obj: ClassificationClass
    ) -> str:
        """Generate QGIS 3.x polygon symbol XML."""
        import uuid

        # Extract colors from ESRI symbol_info (ColorInfo object)
        fill_color = "128,128,128,255"
        outline_color = "35,35,35,255"

        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Extract fill color from ColorInfo object
            if hasattr(symbol_info, "color") and symbol_info.color:
                color_info = symbol_info.color
                if (
                    hasattr(color_info, "r")
                    and hasattr(color_info, "g")
                    and hasattr(color_info, "b")
                ):
                    r = color_info.r
                    g = color_info.g
                    b = color_info.b
                    # ColorInfo.alpha is already 0-255
                    alpha = color_info.alpha if hasattr(color_info, "alpha") else 255
                    # QGIS format: r,g,b,alpha,rgb:r_norm,g_norm,b_norm,alpha_norm
                    r_norm = r / 255
                    g_norm = g / 255
                    b_norm = b / 255
                    alpha_norm = alpha / 255
                    fill_color = f"{r},{g},{b},{alpha},rgb:{r_norm},{g_norm},{b_norm},{alpha_norm}"

            # For outline, look in raw_symbol for stroke layers
            if hasattr(symbol_info, "raw_symbol") and symbol_info.raw_symbol:
                raw = symbol_info.raw_symbol
                if "symbolLayers" in raw:
                    for layer in raw["symbolLayers"]:
                        # Find stroke layer for outline
                        if layer.get("type") == "CIMSolidStroke" and "color" in layer:
                            color_data = layer["color"]
                            if color_data.get("type") in [
                                "CIMRGBColor",
                                "CIMCMYKColor",
                            ]:
                                values = color_data.get("values", [])
                                if len(values) >= 3:
                                    # CMYK: convert to RGB (approximate)
                                    if color_data["type"] == "CIMCMYKColor":
                                        c, m, y, k = values[:4]
                                        r = int(255 * (1 - c / 100) * (1 - k / 100))
                                        g = int(255 * (1 - m / 100) * (1 - k / 100))
                                        b = int(255 * (1 - y / 100) * (1 - k / 100))
                                    else:  # RGB
                                        r, g, b = values[:3]
                                    r_norm = r / 255
                                    g_norm = g / 255
                                    b_norm = b / 255
                                    outline_color = f"{r},{g},{b},255,rgb:{r_norm},{g_norm},{b_norm},1"
                                    break

        layer_id = str(uuid.uuid4())

        return f'''      <symbol force_rhr="0" is_animated="0" frame_rate="10" type="fill" name="{symbol_idx}" alpha="1" clip_to_extent="1">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" name="name" value=""/>
            <Option name="properties"/>
            <Option type="QString" name="type" value="collection"/>
          </Option>
        </data_defined_properties>
        <layer enabled="1" id="{{{layer_id}}}" class="SimpleFill" locked="0" pass="0">
          <Option type="Map">
            <Option type="QString" name="border_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="color" value="{fill_color}"/>
            <Option type="QString" name="joinstyle" value="bevel"/>
            <Option type="QString" name="offset" value="0,0"/>
            <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_unit" value="MM"/>
            <Option type="QString" name="outline_color" value="{outline_color}"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="0.26"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="style" value="solid"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option name="properties"/>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>'''

    def _generate_line_symbol(self, symbol_idx: int, class_obj) -> str:
        """
        Generate QGIS 3.x line symbol XML.

        Supports:
        - SimpleLine: Regular lines with solid, dash, or dot patterns
        - MarkerLine: Lines decorated with font characters (when font_family is present)

        Args:
            symbol_idx: Symbol index for naming
            class_obj: ClassificationClass object with symbol_info

        Returns:
            QGIS XML symbol definition string
        """
        # Check if we have a character marker (font-based symbol on line)
        if (
            hasattr(class_obj, "symbol_info")
            and class_obj.symbol_info
            and class_obj.symbol_info.font_family
            and class_obj.symbol_info.character_index is not None
        ):
            return self._generate_marker_line(symbol_idx, class_obj)
        else:
            return self._generate_simple_line(symbol_idx, class_obj)

    def _generate_simple_line(self, symbol_idx: int, class_obj) -> str:
        """
        Generate QGIS SimpleLine with support for solid, dash, and dot patterns.
        """
        line_color = "128,128,128,255"
        line_width = "0.26"
        use_custom_dash = "0"
        custom_dash = "5;2"
        line_style = "solid"
        cap_style = "square"
        join_style = "bevel"

        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Extract line color
            if hasattr(symbol_info, "color") and symbol_info.color:
                color_info = symbol_info.color
                if (
                    hasattr(color_info, "r")
                    and hasattr(color_info, "g")
                    and hasattr(color_info, "b")
                ):
                    r = color_info.r
                    g = color_info.g
                    b = color_info.b
                    a = getattr(color_info, "alpha", 255)
                    r_norm = r / 255
                    g_norm = g / 255
                    b_norm = b / 255
                    a_norm = a / 255
                    line_color = f"{r},{g},{b},{a},rgb:{r_norm:.3f},{g_norm:.3f},{b_norm:.3f},{a_norm:.3f}"

            # Extract line width (convert points to mm)
            if hasattr(symbol_info, "width") and symbol_info.width:
                line_width = f"{symbol_info.width * 0.352778:.2f}"

            # Extract line style
            if hasattr(symbol_info, "line_style") and symbol_info.line_style:
                line_style = symbol_info.line_style

            # Extract dash pattern
            dash_pattern = getattr(symbol_info, "dash_pattern", None)

            if line_style in ("dash", "dot") and dash_pattern:
                use_custom_dash = "1"
                # Convert ESRI pattern to QGIS format
                custom_dash = ";".join(str(float(v)) for v in dash_pattern)

            # Extract cap and join styles
            if hasattr(symbol_info, "cap_style") and symbol_info.cap_style:
                esri_to_qgis_cap = {
                    "Butt": "flat",
                    "Round": "round",
                    "Square": "square",
                }
                cap_style = esri_to_qgis_cap.get(symbol_info.cap_style, "square")

            if hasattr(symbol_info, "join_style") and symbol_info.join_style:
                esri_to_qgis_join = {
                    "Miter": "miter",
                    "Round": "round",
                    "Bevel": "bevel",
                }
                join_style = esri_to_qgis_join.get(symbol_info.join_style, "bevel")

        # Map style to QGIS line_style
        qgis_style_map = {
            "solid": "solid",
            "dash": "solid" if use_custom_dash == "1" else "dash",
            "dot": "solid" if use_custom_dash == "1" else "dot",
        }
        qgis_line_style = qgis_style_map.get(line_style, "solid")

        layer_id = str(uuid.uuid4())

        return f'''      <symbol force_rhr="0" is_animated="0" frame_rate="10" type="line" name="{symbol_idx}" alpha="1" clip_to_extent="1">
            <data_defined_properties>
              <Option type="Map">
                <Option type="QString" name="name" value=""/>
                <Option name="properties"/>
                <Option type="QString" name="type" value="collection"/>
              </Option>
            </data_defined_properties>
            <layer enabled="1" id="{{{layer_id}}}" class="SimpleLine" locked="0" pass="0">
              <Option type="Map">
                <Option type="QString" name="align_dash_pattern" value="0"/>
                <Option type="QString" name="capstyle" value="{cap_style}"/>
                <Option type="QString" name="customdash" value="{custom_dash}"/>
                <Option type="QString" name="customdash_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="customdash_unit" value="MM"/>
                <Option type="QString" name="dash_pattern_offset" value="0"/>
                <Option type="QString" name="dash_pattern_offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="dash_pattern_offset_unit" value="MM"/>
                <Option type="QString" name="draw_inside_polygon" value="0"/>
                <Option type="QString" name="joinstyle" value="{join_style}"/>
                <Option type="QString" name="line_color" value="{line_color}"/>
                <Option type="QString" name="line_style" value="{qgis_line_style}"/>
                <Option type="QString" name="line_width" value="{line_width}"/>
                <Option type="QString" name="line_width_unit" value="MM"/>
                <Option type="QString" name="offset" value="0"/>
                <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="offset_unit" value="MM"/>
                <Option type="QString" name="ring_filter" value="0"/>
                <Option type="QString" name="trim_distance_end" value="0"/>
                <Option type="QString" name="trim_distance_end_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="trim_distance_end_unit" value="MM"/>
                <Option type="QString" name="trim_distance_start" value="0"/>
                <Option type="QString" name="trim_distance_start_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="trim_distance_start_unit" value="MM"/>
                <Option type="QString" name="use_custom_dash" value="{use_custom_dash}"/>
                <Option type="QString" name="width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
              </Option>
              <data_defined_properties>
                <Option type="Map">
                  <Option type="QString" name="name" value=""/>
                  <Option name="properties"/>
                  <Option type="QString" name="type" value="collection"/>
                </Option>
              </data_defined_properties>
            </layer>
          </symbol>'''

    def _generate_marker_line(self, symbol_idx: int, class_obj) -> str:
        """
        Generate QGIS MarkerLine with FontMarker decoration.

        Used for ESRI CIMCharacterMarker on lines - places font characters
        repeatedly along the line at specified intervals.
        """
        symbol_info = class_obj.symbol_info

        # Extract font information
        font_family = symbol_info.font_family
        character_index = symbol_info.character_index
        character = chr(character_index)

        # Extract color
        marker_color = "128,128,128,255"
        if symbol_info.color:
            r = symbol_info.color.r
            g = symbol_info.color.g
            b = symbol_info.color.b
            a = getattr(symbol_info.color, "alpha", 255)
            marker_color = f"{r},{g},{b},{a}"

        # Extract size (convert points to mm)
        marker_size = "3.5"
        if symbol_info.size:
            marker_size = f"{symbol_info.size * 0.352778:.2f}"

        # Line width for the underlying line (if needed)
        line_width = "0.26"
        if hasattr(symbol_info, "width") and symbol_info.width:
            line_width = f"{symbol_info.width * 0.352778:.2f}"

        # Marker placement interval (default: 3mm spacing)
        marker_interval = "3"

        layer_id = str(uuid.uuid4())
        marker_id = str(uuid.uuid4())

        return f'''      <symbol force_rhr="0" is_animated="0" frame_rate="10" type="line" name="{symbol_idx}" alpha="1" clip_to_extent="1">
            <data_defined_properties>
              <Option type="Map">
                <Option type="QString" name="name" value=""/>
                <Option name="properties"/>
                <Option type="QString" name="type" value="collection"/>
              </Option>
            </data_defined_properties>
            <layer enabled="1" id="{{{layer_id}}}" class="MarkerLine" locked="0" pass="0">
              <Option type="Map">
                <Option type="QString" name="average_angle_length" value="4"/>
                <Option type="QString" name="average_angle_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="average_angle_unit" value="MM"/>
                <Option type="QString" name="interval" value="{marker_interval}"/>
                <Option type="QString" name="interval_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="interval_unit" value="MM"/>
                <Option type="QString" name="offset" value="0"/>
                <Option type="QString" name="offset_along_line" value="0"/>
                <Option type="QString" name="offset_along_line_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="offset_along_line_unit" value="MM"/>
                <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="offset_unit" value="MM"/>
                <Option type="bool" name="place_on_every_part" value="true"/>
                <Option type="QString" name="placements" value="Interval"/>
                <Option type="QString" name="ring_filter" value="0"/>
                <Option type="QString" name="rotate" value="1"/>
              </Option>
              <data_defined_properties>
                <Option type="Map">
                  <Option type="QString" name="name" value=""/>
                  <Option name="properties"/>
                  <Option type="QString" name="type" value="collection"/>
                </Option>
              </data_defined_properties>
              <symbol force_rhr="0" is_animated="0" frame_rate="10" type="marker" name="@{symbol_idx}@0" alpha="1" clip_to_extent="1">
                <data_defined_properties>
                  <Option type="Map">
                    <Option type="QString" name="name" value=""/>
                    <Option name="properties"/>
                    <Option type="QString" name="type" value="collection"/>
                  </Option>
                </data_defined_properties>
                <layer enabled="1" id="{{{marker_id}}}" class="FontMarker" locked="0" pass="0">
                  <Option type="Map">
                    <Option type="QString" name="angle" value="0"/>
                    <Option type="QString" name="chr" value="{character}"/>
                    <Option type="QString" name="color" value="{marker_color}"/>
                    <Option type="QString" name="font" value="{font_family}"/>
                    <Option type="QString" name="font_style" value=""/>
                    <Option type="QString" name="horizontal_anchor_point" value="1"/>
                    <Option type="QString" name="joinstyle" value="bevel"/>
                    <Option type="QString" name="offset" value="0,0"/>
                    <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                    <Option type="QString" name="offset_unit" value="MM"/>
                    <Option type="QString" name="outline_color" value="0,0,0,255"/>
                    <Option type="QString" name="outline_width" value="0"/>
                    <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                    <Option type="QString" name="outline_width_unit" value="MM"/>
                    <Option type="QString" name="size" value="{marker_size}"/>
                    <Option type="QString" name="size_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                    <Option type="QString" name="size_unit" value="MM"/>
                    <Option type="QString" name="vertical_anchor_point" value="1"/>
                  </Option>
                  <data_defined_properties>
                    <Option type="Map">
                      <Option type="QString" name="name" value=""/>
                      <Option name="properties"/>
                      <Option type="QString" name="type" value="collection"/>
                    </Option>
                  </data_defined_properties>
                </layer>
              </symbol>
            </layer>
          </symbol>'''

    def _generate_point_symbol(self, symbol_idx: int, class_obj) -> str:
        """
        Generate QGIS 3.x point symbol XML.

        Supports:
        - SimpleMarker: Basic geometric shapes (circle, square, triangle)
        - FontMarker: Character-based symbols from fonts (when font_family is present)

        Args:
            symbol_idx: Symbol index for naming
            class_obj: ClassificationClass object with symbol_info

        Returns:
            QGIS XML symbol definition string
        """
        # Check if we have a character marker (font-based symbol)
        if (
            hasattr(class_obj, "symbol_info")
            and class_obj.symbol_info
            and class_obj.symbol_info.font_family
            and class_obj.symbol_info.character_index is not None
        ):
            return self._generate_font_marker(symbol_idx, class_obj)
        else:
            return self._generate_simple_marker(symbol_idx, class_obj)

    def _generate_simple_marker(self, symbol_idx: int, class_obj) -> str:
        """Generate QGIS SimpleMarker (geometric shape) symbol."""
        marker_color = "128,128,128,255"
        marker_size = "2"
        marker_name = "circle"

        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Extract color
            if symbol_info.color:
                r = symbol_info.color.r
                g = symbol_info.color.g
                b = symbol_info.color.b
                a = symbol_info.color.alpha
                marker_color = f"{r},{g},{b},{a}"

            # Extract size (convert points to mm)
            if symbol_info.size:
                marker_size = f"{symbol_info.size * 0.352778:.2f}"

            # Map ESRI marker types to QGIS names
            # Note: This is a simplified mapping - you may need to adjust
            if hasattr(symbol_info, "raw_symbol"):
                marker_type = symbol_info.raw_symbol.get("type", "")
                marker_map = {
                    "esriSMSCircle": "circle",
                    "esriSMSSquare": "square",
                    "esriSMSTriangle": "triangle",
                    "esriSMSDiamond": "diamond",
                    "esriSMSCross": "cross",
                    "esriSMSX": "cross2",
                }
                marker_name = marker_map.get(marker_type, "circle")

        layer_id = str(uuid.uuid4())

        return f'''      <symbol force_rhr="0" is_animated="0" frame_rate="10" type="marker" name="{symbol_idx}" alpha="1" clip_to_extent="1">
            <data_defined_properties>
              <Option type="Map">
                <Option type="QString" name="name" value=""/>
                <Option name="properties"/>
                <Option type="QString" name="type" value="collection"/>
              </Option>
            </data_defined_properties>
            <layer enabled="1" id="{{{layer_id}}}" class="SimpleMarker" locked="0" pass="0">
              <Option type="Map">
                <Option type="QString" name="angle" value="0"/>
                <Option type="QString" name="cap_style" value="square"/>
                <Option type="QString" name="color" value="{marker_color}"/>
                <Option type="QString" name="horizontal_anchor_point" value="1"/>
                <Option type="QString" name="joinstyle" value="bevel"/>
                <Option type="QString" name="name" value="{marker_name}"/>
                <Option type="QString" name="offset" value="0,0"/>
                <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="offset_unit" value="MM"/>
                <Option type="QString" name="outline_color" value="35,35,35,255"/>
                <Option type="QString" name="outline_style" value="solid"/>
                <Option type="QString" name="outline_width" value="0"/>
                <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="outline_width_unit" value="MM"/>
                <Option type="QString" name="scale_method" value="diameter"/>
                <Option type="QString" name="size" value="{marker_size}"/>
                <Option type="QString" name="size_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="size_unit" value="MM"/>
                <Option type="QString" name="vertical_anchor_point" value="1"/>
              </Option>
              <data_defined_properties>
                <Option type="Map">
                  <Option type="QString" name="name" value=""/>
                  <Option name="properties"/>
                  <Option type="QString" name="type" value="collection"/>
                </Option>
              </data_defined_properties>
            </layer>
          </symbol>'''

    def _generate_font_marker(self, symbol_idx: int, class_obj) -> str:
        """
        Generate QGIS FontMarker (character-based) symbol.

        Used for ESRI CIMCharacterMarker symbols that use font characters.
        """
        symbol_info = class_obj.symbol_info

        # Extract font information
        font_family = symbol_info.font_family
        character_index = symbol_info.character_index

        # Convert character index to Unicode character
        # ESRI uses decimal character codes
        character = chr(character_index)

        # Extract color
        marker_color = "128,128,128,255"
        if symbol_info.color:
            r = symbol_info.color.r
            g = symbol_info.color.g
            b = symbol_info.color.b
            a = symbol_info.color.alpha
            marker_color = f"{r},{g},{b},{a}"

        # Extract size (convert points to mm)
        marker_size = "3.5"
        if symbol_info.size:
            marker_size = f"{symbol_info.size * 0.352778:.2f}"

        layer_id = str(uuid.uuid4())

        # Note: QGIS FontMarker uses different parameters than SimpleMarker
        return f'''      <symbol force_rhr="0" is_animated="0" frame_rate="10" type="marker" name="{symbol_idx}" alpha="1" clip_to_extent="1">
            <data_defined_properties>
              <Option type="Map">
                <Option type="QString" name="name" value=""/>
                <Option name="properties"/>
                <Option type="QString" name="type" value="collection"/>
              </Option>
            </data_defined_properties>
            <layer enabled="1" id="{{{layer_id}}}" class="FontMarker" locked="0" pass="0">
              <Option type="Map">
                <Option type="QString" name="angle" value="0"/>
                <Option type="QString" name="chr" value="{character}"/>
                <Option type="QString" name="color" value="{marker_color}"/>
                <Option type="QString" name="font" value="{font_family}"/>
                <Option type="QString" name="font_style" value=""/>
                <Option type="QString" name="horizontal_anchor_point" value="1"/>
                <Option type="QString" name="joinstyle" value="bevel"/>
                <Option type="QString" name="offset" value="0,0"/>
                <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="offset_unit" value="MM"/>
                <Option type="QString" name="outline_color" value="0,0,0,255"/>
                <Option type="QString" name="outline_width" value="0"/>
                <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="outline_width_unit" value="MM"/>
                <Option type="QString" name="size" value="{marker_size}"/>
                <Option type="QString" name="size_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="size_unit" value="MM"/>
                <Option type="QString" name="vertical_anchor_point" value="1"/>
              </Option>
              <data_defined_properties>
                <Option type="Map">
                  <Option type="QString" name="name" value=""/>
                  <Option name="properties"/>
                  <Option type="QString" name="type" value="collection"/>
                </Option>
              </data_defined_properties>
            </layer>
          </symbol>'''
