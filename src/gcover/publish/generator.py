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
from typing import Dict, List, Optional, Tuple, Set
from xml.dom import minidom

import click
from loguru import logger
from rich.console import Console

from gcover.publish.esri_classification_extractor import (
    ClassificationClass,
    ESRIClassificationExtractor,
    LayerClassification,
    SymbolType,
)

from gcover.publish.complex_polygon_generators import (
    generate_complex_polygon_qml,
    generate_complex_polygon_mapserver,
)

from .esri_classification_extractor import extract_lyrx

from gcover.publish.tooltips_enricher import LayerType

console = Console()


class MapServerGenerator:
    """Generate MapServer mapfile CLASS sections from classifications.

    1. CLASSITEM support for simplified expressions
    2. Complete pattern symbol definitions in symbols.txt
    """

    def __init__(
        self,
        layer_type: str = "Polygon",
        use_symbol_field: bool = False,
        symbol_field: str = "SYMBOL",
        font_name_prefix: str = "esri",
    ):
        """
        Initialize generator.

        Args:
            layer_type: POLYGON, LINE, or POINT
            use_symbol_field: If True, use CLASSITEM with simple expressions
            symbol_field: Name of symbol field in data (default: SYMBOL)
            font_name_prefix: Prefix for font names in FONTSET (default: "esri")
        """
        self.layer_type = layer_type.upper()
        self.use_symbol_field = use_symbol_field
        self.symbol_field = symbol_field
        self.font_name_prefix = font_name_prefix

        # Track fonts and symbols used
        self.fonts_used: Set[str] = set()
        self.pattern_symbols: List[Dict] = []  # Store pattern symbol definitions

    def generate_expression_from_fields(self, class_obj, field_names: List[str]) -> str:
        """Generate MapServer EXPRESSION from classification field values."""
        expressions = []

        for field_values in class_obj.field_values:
            conditions = []

            for field_name, expected_value in zip(field_names, field_values):
                if expected_value == "<Null>":
                    conditions.append(
                        f'([{field_name}] eq "" OR NOT [DEFINED_{field_name}])'
                    )
                elif expected_value == "999997":
                    conditions.append(f'([{field_name}] eq "999997")')
                elif expected_value == "999999":
                    conditions.append(f'([{field_name}] eq "999999")')
                else:
                    conditions.append(f'([{field_name}] eq "{expected_value}")')

            if len(conditions) == 1:
                expressions.append(conditions[0])
            else:
                expressions.append(f"({' AND '.join(conditions)})")

        if len(expressions) == 1:
            return expressions[0]
        else:
            return f"({' OR '.join(expressions)})"

    def generate_expression_from_symbol(self, symbol_id: str) -> str:
        """
        Generate simple MapServer EXPRESSION using SYMBOL field.

        When CLASSITEM is used, the expression is just the value string.
        """
        if self.use_symbol_field:
            # Simple expression when CLASSITEM is set
            return f'"{symbol_id}"'
        else:
            # Full expression syntax
            return f'([{self.symbol_field}] eq "{symbol_id}")'

    def generate_layer(
        self,
        classification,
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
            # Add CLASSITEM for simplified expressions
            lines.append(f'  CLASSITEM "{self.symbol_field}"')
            lines.append(f"  # Using CLASSITEM for simplified expressions")
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

    def generate_class(
        self,
        class_obj,
        field_names: List[str],
        class_index: int,
        symbol_prefix: str = "class",
    ) -> str:
        """Generate a single MapServer CLASS block."""
        if not class_obj.visible:
            return ""

        # Generate expression
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

        # Add STYLE blocks based on layer type
        if self.layer_type == "POLYGON":
            # Import the complex polygon generator
            from .complex_polygon_generators import generate_complex_polygon_mapserver

            generate_complex_polygon_mapserver(
                lines,
                class_obj,
                class_index,
                symbol_prefix,
                self.fonts_used,
                self.pattern_symbols,
            )
        elif self.layer_type == "LINE":
            lines.append("    STYLE")
            self._add_line_style(lines, class_obj, class_index, symbol_prefix)
            lines.append("    END # STYLE")
        elif self.layer_type == "POINT":
            lines.append("    STYLE")
            self._add_point_style(lines, class_obj, class_index, symbol_prefix)
            lines.append("    END # STYLE")

        lines.append("  END # CLASS")
        lines.append("")

        return "\n".join(lines)

    def generate_symbol_file(self, classification_list: List = None) -> str:
        """
        Generate MapServer symbol file with all symbol types.

        Args:
            classification_list: List of LayerClassification objects

        Returns:
            Complete SYMBOLSET file content
        """
        lines = [
            "SYMBOLSET",
            "",
            "  # Basic geometric symbols for GeoCover",
            "",
        ]

        # Add basic geometric symbols
        lines.extend(self._generate_basic_symbols())

        # Add line pattern symbols (dashed, dotted)
        lines.append("")
        lines.append("  # Line pattern symbols")
        lines.append("")
        lines.extend(self._generate_line_pattern_symbols())

        # Scan classifications for font symbols
        if classification_list:  # CIM objects
            # Point and line font markers
            font_symbols = self._generate_font_symbols_from_classifications(
                classification_list
            )
            if font_symbols:
                lines.append("")
                lines.append("  # TrueType font marker symbols (points and lines)")
                lines.append("")
                lines.extend(font_symbols)

            # Polygon pattern fills
            from .complex_polygon_generators import scan_and_generate_pattern_symbols

            pattern_symbols = scan_and_generate_pattern_symbols(
                classification_list, self.pattern_symbols, self.fonts_used
            )
            if pattern_symbols:
                lines.append("")
                lines.append("  # TrueType pattern fill symbols (polygons)")
                lines.append("")
                lines.extend(pattern_symbols)

        lines.append("")
        lines.append("END # SYMBOLSET")

        return "\n".join(lines)

    def _generate_basic_symbols(self) -> List[str]:
        """Generate basic geometric symbols."""
        return [
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
        ]

    def _generate_line_pattern_symbols(self) -> List[str]:
        """Generate line pattern symbols (dashed, dotted)."""
        return [
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
        ]

    def generate_fontset(self, font_paths: Dict[str, str] = None) -> str:
        """
        Generate MapServer FONTSET file.

        Uses hardcoded GeoCover fonts by default.
        """
        lines = [
            "# MapServer FONTSET file",
            "# Font definitions for TrueType fonts used in GeoCover",
            "#",
            "# Format: <alias> <font_file_path>",
            "",
        ]

        # In WMS
        # geofont                         geofontsregular.ttf
        # geofont1                        GeoFonts1.ttf

        # Hardcoded GeoCover fonts
        GEOCOVER_FONTS = {
            "geofonts": "fonts/geofontsregular.ttf",
            # "GeoFonts 1": "/home/marco/.fonts/g/GeoFonts1.ttf",
            "geofonts1": "fonts/GeoFonts1.ttf",  # Alias without space
            # "GeoFonts 2": "/home/marco/.fonts/g/GeoFonts2.ttf",
            "geofonts2": "fonts/GeoFonts2.ttf",  # Alias without space
        }

        # Override with provided paths if given
        if font_paths:
            GEOCOVER_FONTS.update(font_paths)

        # Add fonts that were actually used
        fonts_added = set()
        for font_family in sorted(self.fonts_used):
            sanitized_name = self._sanitize_font_name(font_family)

            # Avoid duplicates
            if sanitized_name in fonts_added:
                continue

            fonts_added.add(sanitized_name)

            # Get font path
            font_path = GEOCOVER_FONTS.get(font_family, f"fonts/{sanitized_name}.ttf")

            lines.append(f'{sanitized_name} "{font_path}"')
            if font_family != sanitized_name:
                lines.append(f"# Original name: {font_family}")
            lines.append("")

        # Add common system fonts
        lines.append("# Common system fonts")
        lines.append(
            'arial "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"'
        )
        lines.append(
            'arial_bold "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"'
        )
        lines.append("")

        return "\n".join(lines)

    def _sanitize_font_name(self, font_family: str) -> str:
        """Sanitize font family name for use in FONTSET."""
        return font_family.lower().replace(" ", "").replace("-", "_")

    def _add_line_style(
        self, lines: List[str], class_obj, class_index: int, symbol_prefix: str
    ):
        """Add line styling from ESRI symbol_info."""
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Check if font marker on line
            if (
                hasattr(symbol_info, "font_family")
                and symbol_info.font_family
                and hasattr(symbol_info, "character_index")
                and symbol_info.character_index is not None
            ):
                font_symbol_name = f"{symbol_prefix}_line_font_{class_index}"
                self._add_truetype_line_marker(lines, symbol_info, font_symbol_name)
                self.fonts_used.add(symbol_info.font_family)
            else:
                # Regular line
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

                if hasattr(symbol_info, "width") and symbol_info.width:
                    width = symbol_info.width * 1.33
                    lines.append(f"      WIDTH {width:.2f}")
                else:
                    lines.append("      WIDTH 1.0")

                # Dash pattern
                if (
                    hasattr(symbol_info, "line_style")
                    and symbol_info.line_style
                    and hasattr(symbol_info, "dash_pattern")
                    and symbol_info.dash_pattern
                ):
                    line_style = symbol_info.line_style
                    if line_style == "dash":
                        lines.append('      SYMBOL "dashed"')
                    elif line_style == "dot":
                        lines.append('      SYMBOL "dotted"')
        else:
            lines.append("      COLOR 128 128 128")
            lines.append("      WIDTH 1.0")

    def _add_truetype_line_marker(
        self, lines: List[str], symbol_info, font_symbol_name: str
    ):
        """Add TrueType font marker on line."""
        lines.append(f'      SYMBOL "{font_symbol_name}"')

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

        if hasattr(symbol_info, "size") and symbol_info.size:
            size = symbol_info.size * 1.33
            lines.append(f"      SIZE {size:.1f}")
        else:
            lines.append("      SIZE 10")

        lines.append("      GAP -30")

    def _add_point_style(
        self, lines: List[str], class_obj, class_index: int, symbol_prefix: str
    ):
        """Add point styling from ESRI symbol_info."""
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Check if font marker
            if (
                hasattr(symbol_info, "font_family")
                and symbol_info.font_family
                and hasattr(symbol_info, "character_index")
                and symbol_info.character_index is not None
            ):
                font_symbol_name = f"{symbol_prefix}_point_font_{class_index}"
                lines.append(f'      SYMBOL "{font_symbol_name}"')
                self.fonts_used.add(symbol_info.font_family)
            else:
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

            if hasattr(symbol_info, "size") and symbol_info.size:
                size = symbol_info.size * 1.33
                lines.append(f"      SIZE {size:.1f}")
            else:
                lines.append("      SIZE 8")

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

    def _get_prefix(self, layer_name, default="class"):
        PREFIXES = {"Fossils": "foss", "Surfaces": "surf"}

        return PREFIXES.get(layer_name, default)

    def _generate_font_symbols_from_classifications(
        self, classification_list: List
    ) -> List[str]:
        """Generate TrueType symbols for point and line markers."""
        symbols = []
        symbols_generated = set()

        for classification in classification_list:
            symbol_prefix = self._get_prefix(classification.layer_name)

            for class_index, class_obj in enumerate(classification.classes):
                if not class_obj.visible:
                    continue

                if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
                    continue

                symbol_info = class_obj.symbol_info

                if (
                    hasattr(symbol_info, "symbol_type")
                    and symbol_info.symbol_type == SymbolType.POINT
                ):
                    console.print(symbol_info)

                if (
                    hasattr(symbol_info, "font_family")
                    and symbol_info.font_family
                    and hasattr(symbol_info, "character_index")
                    and symbol_info.character_index is not None
                ):
                    font_family = symbol_info.font_family
                    character = chr(symbol_info.character_index)

                    # Generate symbol names for both point and line
                    symbol_names = [
                        f"{symbol_prefix}_point_font_{class_index}",
                        f"{symbol_prefix}_line_font_{class_index}",
                    ]

                    for symbol_name in symbol_names:
                        symbol_key = (
                            f"{font_family}_{symbol_info.character_index}_{symbol_name}"
                        )
                        if symbol_key in symbols_generated:
                            continue

                        symbols_generated.add(symbol_key)
                        self.fonts_used.add(font_family)

                        font_name = self._sanitize_font_name(font_family)

                        symbols.extend(
                            [
                                "  SYMBOL",
                                f'    NAME "{symbol_name}"',
                                "    TYPE TRUETYPE",
                                f'    FONT "{font_name}"',
                                f'    CHARACTER "{character}"',
                                "    FILLED TRUE",
                                "    ANTIALIAS TRUE",
                                "  END",
                                "",
                            ]
                        )

        return symbols


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
        """Generate QGIS 3.x polygon symbol XML with complex layer support."""
        return generate_complex_polygon_qml(symbol_idx, class_obj)

    def _generate_polygon_symbol_ori(
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
