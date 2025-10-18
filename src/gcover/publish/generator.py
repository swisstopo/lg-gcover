#!/usr/bin/env python3
"""
MapServer Mapfile and QGIS QML Generator

Generate map server configuration from ESRI classification rules.
Supports complex multi-layer symbols including pattern fills, font markers,
and sophisticated polygon styling.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
from loguru import logger
from rich.console import Console

from .esri_classification_extractor import (ClassificationClass,
                                            ESRIClassificationExtractor,
                                            LayerClassification, SymbolType)
from .symbol_models import CharacterMarkerInfo, FontSymbol, SymbolLayersInfo
from .symbol_utils import (extract_color_from_cim,
                           extract_line_style_from_effects,
                           extract_polygon_symbol_layers, sanitize_font_name)
from .tooltips_enricher import LayerType
from .utils import generate_font_image

console = Console()


class MapServerGenerator:
    """
    Generate MapServer mapfile CLASS sections from classifications.

    Features:
    - Complete pattern symbol definitions in symbols.txt
    - Font symbol tracking and deduplication
    - Complex polygon symbols with multiple layers
    - Point and line font markers
    - PDF generation of all used symbols
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

        # Track fonts and symbols used (for symbol file generation)
        self.fonts_used: Set[str] = set()
        self.pattern_symbols: List[Dict] = []
        self.symbol_registry: Dict[FontSymbol, str] = {}

    # ========================================================================
    # LAYER AND CLASS GENERATION
    # ========================================================================

    def generate_layer(
        self,
        classification,
        layer_name: str,
        data_path: str = None,
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
            Complete LAYER block as string
        """
        lines = [
            "LAYER",
            f'  NAME "{layer_name}"',
            f"  TYPE {self.layer_type}",
            "  STATUS ON",
            "",
        ]

        # Metadata
        lines.extend(
            [
                "",
                "  METADATA",
                f'    "wms_title"    "{layer_name.capitalize()}"',
                f'    "wms_abstract" "{layer_name.capitalize()}"',
                '    "wms_srs"      "EPSG:2056 EPSG:4326 EPSG:3857"',
                '    "wms_extent" "2350000 1050000 2850000 1250000"',
                "  END",
            ]
        )

        # Data source
        if data_path:
            lines.extend(
                [
                    "",
                    "  CONNECTIONTYPE POSTGIS",
                    '  CONNECTION "host=postgis port=5432 dbname=geocover user=gcover password=gcover123"',
                    f'  DATA "geom from (SELECT * from {data_path} where {self.symbol_field.lower()} IS NOT NULL) as blabla using unique gid using srid=2056"',
                ]
            )

        # Projection
        lines.extend(
            [
                "",
                "  PROJECTION",
                '    "init=epsg:2056"',
                "  END",
            ]
        )

        # Template
        lines.extend(["", '  TEMPLATE "empty"', ""])

        # CLASSITEM if using symbol field
        if self.use_symbol_field:
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
            expression = self._generate_expression_from_symbol(symbol_id)
        else:
            expression = self._generate_expression_from_fields(class_obj, field_names)

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
            self._add_polygon_styles(lines, class_obj, class_index, symbol_prefix)
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

    # ========================================================================
    # EXPRESSION GENERATION
    # ========================================================================

    def _generate_expression_from_fields(
        self, class_obj, field_names: List[str]
    ) -> str:
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

    def _generate_expression_from_symbol(self, symbol_id: str) -> str:
        """
        Generate simple MapServer EXPRESSION using SYMBOL field.

        When CLASSITEM is used, the expression is just the value string.
        """
        if self.use_symbol_field:
            return f'"{symbol_id}"'
        else:
            return f'([{self.symbol_field}] eq "{symbol_id}")'

    # ========================================================================
    # POLYGON STYLING (COMPLEX MULTI-LAYER SUPPORT)
    # ========================================================================

    def _add_polygon_styles(
        self,
        lines: List[str],
        class_obj,
        class_index: int,
        symbol_prefix: str,
    ) -> None:
        """
        Add MapServer STYLE blocks for polygon symbols.

        ENHANCED with hatch fill support using full_symbol_layers.
        """
        # NEW: Try to use full_symbol_layers first (complete extraction)
        if hasattr(class_obj, "full_symbol_layers") and class_obj.full_symbol_layers:
            self._add_polygon_styles_from_full_layers(
                lines, class_obj.full_symbol_layers, class_index, symbol_prefix
            )
            return

        # FALLBACK: Use legacy symbol_info extraction
        if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
            self._add_simple_polygon_style(lines)
            return

        symbol_info = class_obj.symbol_info

        if not hasattr(symbol_info, "raw_symbol") or not symbol_info.raw_symbol:
            self._add_simple_polygon_style(lines)
            return

        # Extract all symbol layers (legacy method)
        layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

        has_fill = False

        # Add character marker pattern fills first
        for i, marker_info in enumerate(layers_info.character_markers):
            self._add_pattern_fill_style(
                lines, marker_info, class_index, i, symbol_prefix
            )
            has_fill = True

        # Add solid fills
        for fill_info in layers_info.fills:
            if fill_info["type"] == "solid":
                self._add_solid_fill_style(lines, fill_info["color"])
                has_fill = True

        # Add outline
        if layers_info.outline:
            self._add_outline_style(lines, layers_info.outline)

        if not has_fill:
            if layers_info.outline:
                self._add_default_fill_style(lines)
            else:
                self._add_simple_polygon_style(lines)

    def _add_polygon_styles_from_full_layers(
        self,
        lines: List[str],
        full_layers,  # SymbolLayersInfo object
        class_index: int,
        symbol_prefix: str,
    ) -> None:
        """
        NEW: Add polygon styles using complete SymbolLayersInfo.

        Renders all fill types in proper order.
        """
        has_fill = False

        # Process fills in order
        for i, fill_info in enumerate(full_layers.fills):
            fill_type = fill_info.get("type", "solid")

            if fill_type == "solid":
                self._add_solid_fill_style(lines, fill_info["color"])
                has_fill = True

            elif fill_type == "hatch":
                self._add_hatch_fill_style(
                    lines, fill_info, class_index, i, symbol_prefix
                )
                has_fill = True

            elif fill_type == "character":
                marker = fill_info.get("marker_info")
                if marker:
                    self._add_pattern_fill_style(
                        lines, marker, class_index, i, symbol_prefix
                    )
                    has_fill = True

            elif fill_type == "picture":
                logger.warning(f"Picture fill not yet supported in MapServer")

            elif fill_type == "gradient":
                logger.warning(f"Gradient fill not yet supported in MapServer")

        # Add outline (top layer)
        if full_layers.outline:
            self._add_outline_style(lines, full_layers.outline)

        # Fallback if no fills
        if not has_fill:
            if full_layers.outline:
                self._add_default_fill_style(lines)
            else:
                self._add_simple_polygon_style(lines)

    def _add_hatch_fill_style(
        self,
        lines: List[str],
        hatch_info: Dict,
        class_index: int,
        fill_index: int,
        symbol_prefix: str,
    ) -> None:
        """
        NEW: Add MapServer STYLE for hatch fill pattern.
        """
        rotation = hatch_info["rotation"]
        separation = hatch_info["separation"]
        line_sym = hatch_info["line_symbol"]

        # Extract line properties
        line_color = line_sym["color"]
        line_width = line_sym["width"]
        r, g, b, a = line_color

        # Generate unique symbol name
        symbol_name = f"hatch_{int(rotation)}deg_{int(separation * 10)}"

        # Register this hatch symbol
        hatch_symbol_def = {
            "name": symbol_name,
            "type": "hatch",
            "rotation": rotation,
            "separation": separation,
            "line_width": line_width,
            "line_color": line_color,
        }

        # Add to registry (avoid duplicates)
        if not any(s.get("name") == symbol_name for s in self.pattern_symbols):
            self.pattern_symbols.append(hatch_symbol_def)

        # Convert to pixels
        separation_px = separation * 1.33
        width_px = line_width * 1.33

        # Add STYLE block
        lines.append("    STYLE")
        lines.append(f'      SYMBOL "{symbol_name}"')
        lines.append(f"      SIZE {separation_px:.2f}")
        lines.append(f"      WIDTH {width_px:.2f}")
        lines.append(f"      COLOR {r} {g} {b}")
        if a < 255:
            opacity = int((a / 255) * 100)
            lines.append(f"      OPACITY {opacity}")
        lines.append("    END # STYLE")

    def _generate_hatch_pattern_symbols(self) -> List[str]:
        """
        NEW: Generate MapServer HATCH symbols.
        """
        symbols = []

        for hatch_def in self.pattern_symbols:
            if hatch_def.get("type") != "hatch":
                continue

            name = hatch_def["name"]
            rotation = hatch_def["rotation"]

            symbols.extend(
                [
                    "  SYMBOL",
                    f'    NAME "{name}"',
                    "    TYPE HATCH",
                    f"    # Diagonal lines at {rotation}° angle",
                    "  END",
                    "",
                ]
            )

        return symbols

    def _add_pattern_fill_style(
        self,
        lines: List[str],
        marker_info: CharacterMarkerInfo,
        class_index: int,
        marker_index: int,
        symbol_prefix: str,
    ) -> None:
        """Add MapServer STYLE for character marker pattern fill."""
        r, g, b, a = marker_info.color

        # Create symbol name using font and character
        font_name = sanitize_font_name(marker_info.font_family)
        char_index = marker_info.character_index
        symbol_name = f"{font_name}_{char_index}"

        # Track font usage
        self.fonts_used.add(marker_info.font_family)

        # Convert points to pixels
        size_px = marker_info.size * 1.33

        lines.append("    STYLE")
        lines.append(f'      SYMBOL "{symbol_name}"')
        lines.append(f"      COLOR {r} {g} {b}")
        if a < 255:
            lines.append(f"      OPACITY {a}")
        lines.append(f"      SIZE {size_px:.1f}")
        lines.append("    END # STYLE")

    def _add_solid_fill_style(
        self, lines: List[str], color: Tuple[int, int, int, int]
    ) -> None:
        """Add MapServer STYLE for solid fill."""
        r, g, b, a = color

        lines.append("    STYLE")
        lines.append(f"      COLOR {r} {g} {b}")
        if a < 255:
            lines.append(f"      OPACITY {a}")
        lines.append("    END # STYLE")

    def _add_outline_style(self, lines: List[str], outline_info: Dict) -> None:
        """Add MapServer STYLE for polygon outline."""
        r, g, b, a = outline_info["color"]
        width_px = outline_info["width"] * 1.33

        lines.append("    STYLE")
        lines.append(f"      OUTLINECOLOR {r} {g} {b}")
        lines.append(f"      WIDTH {width_px:.2f}")

        # Handle line style
        line_style_info = outline_info["line_style"]
        if line_style_info["type"] == "dash":
            lines.append('      SYMBOL "dashed"')
        elif line_style_info["type"] == "dot":
            lines.append('      SYMBOL "dotted"')

        lines.append("    END # STYLE")

    def _add_simple_polygon_style(self, lines: List[str]) -> None:
        """Add simple fallback polygon style with fill and outline."""
        lines.append("    STYLE")
        lines.append("      COLOR 128 128 128")
        lines.append("      OUTLINECOLOR 64 64 64")
        lines.append("    END # STYLE")

    def _add_default_fill_style(self, lines: List[str]) -> None:
        """Add a default fill (used when only outline is present)."""
        lines.append("    STYLE")
        lines.append("      COLOR 255 255 255")  # White fill
        lines.append("    END # STYLE")

    # ========================================================================
    # LINE STYLING
    # ========================================================================

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
                # Get or create font symbol
                font_family = sanitize_font_name(symbol_info.font_family)
                char_index = symbol_info.character_index
                spec = FontSymbol(font_family, char_index)

                if spec not in self.symbol_registry:
                    font_symbol_name = f"{spec.font_family}_{spec.char_index}"
                    self.symbol_registry[spec] = font_symbol_name
                else:
                    font_symbol_name = self.symbol_registry[spec]

                self._add_truetype_line_marker(lines, symbol_info, font_symbol_name)
                self.fonts_used.add(symbol_info.font_family)
            else:
                # Regular line
                self._add_regular_line_style(lines, symbol_info)
        else:
            lines.append("      COLOR 128 128 128")
            lines.append("      WIDTH 1.0")

    def _add_regular_line_style(self, lines: List[str], symbol_info) -> None:
        """Add regular line style (no font markers)."""
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

    # ========================================================================
    # POINT STYLING
    # ========================================================================

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
                # Get or create font symbol
                font_family = sanitize_font_name(symbol_info.font_family)
                char_index = symbol_info.character_index
                spec = FontSymbol(font_family, char_index)

                if spec not in self.symbol_registry:
                    font_symbol_name = f"{spec.font_family}_{spec.char_index}"
                    self.symbol_registry[spec] = font_symbol_name
                else:
                    font_symbol_name = self.symbol_registry[spec]

                lines.append(f'      SYMBOL "{font_symbol_name}"')
                self.fonts_used.add(symbol_info.font_family)
            else:
                # Geometric marker
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

    # ========================================================================
    # SYMBOL FILE GENERATION
    # ========================================================================

    def generate_symbol_file(
        self, classification_list: List = None, prefixes: Dict = {}
    ) -> str:
        """
        Generate MapServer symbol file with all symbol types.

        Args:
            classification_list: List of LayerClassification objects
            prefixes: Layer name to prefix mapping

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

        # NEW: Add hatch pattern symbols ⭐
        if self.pattern_symbols:
            hatch_symbols = self._generate_hatch_pattern_symbols()
            if hatch_symbols:
                lines.append("")
                lines.append("  # Hatch pattern symbols (diagonal lines)")
                lines.append("")
                lines.extend(hatch_symbols)

        # Generate font symbols if classifications provided
        if classification_list:
            # Point and line font markers
            font_symbols = self._generate_font_symbols_from_classifications(
                classification_list, prefixes
            )
            if font_symbols:
                lines.append("")
                lines.append("  # TrueType font marker symbols (points and lines)")
                lines.append("")
                lines.extend(font_symbols)

            # Polygon pattern fills
            pattern_symbols = self._generate_polygon_pattern_symbols(
                classification_list
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

    def _generate_font_symbols_from_classifications(
        self, classification_list: List, prefixes: Dict
    ) -> List[str]:
        """Generate TrueType symbols for point and line markers."""
        symbols = []

        def escape_mapserver_string(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

        # Generate symbols from registry
        images = []
        for spec in sorted(
            self.symbol_registry.keys(), key=lambda s: (s.font_family, s.char_index)
        ):
            font_symbol_name = self.symbol_registry[spec]
            font_name = spec.font_family
            character = escape_mapserver_string(chr(spec.char_index))

            # Generate font image for PDF
            image = generate_font_image(
                font_symbol_name, spec.font_family, spec.char_index
            )
            if image:
                images.append(image)

            symbols.extend(
                [
                    "  SYMBOL",
                    f'    NAME "{font_symbol_name}"',
                    "    TYPE TRUETYPE",
                    f'    FONT "{font_name}"',
                    f'    CHARACTER "{character}"',
                    "    FILLED TRUE",
                    "    ANTIALIAS TRUE",
                    "  END",
                    "",
                ]
            )

        # Save PDF of all used font symbols
        if images:
            images[0].save(
                "mapserver/font_characters.pdf",
                save_all=True,
                append_images=images[1:],
            )

        return symbols

    def _generate_polygon_pattern_symbols(self, classification_list: List) -> List[str]:
        """Generate TrueType symbols for polygon pattern fills."""
        symbols = []
        symbols_generated = set()

        for classification in classification_list:
            for class_index, class_obj in enumerate(classification.classes):
                if not class_obj.visible:
                    continue

                if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
                    continue

                symbol_info = class_obj.symbol_info

                if not hasattr(symbol_info, "raw_symbol") or not symbol_info.raw_symbol:
                    continue

                # Extract symbol layers
                layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

                # Generate symbol for each character marker
                for marker_info in layers_info.character_markers:
                    font_name = sanitize_font_name(marker_info.font_family)
                    char_index = marker_info.character_index
                    symbol_name = f"{font_name}_{char_index}"

                    # Avoid duplicates
                    if symbol_name in symbols_generated:
                        continue

                    symbols_generated.add(symbol_name)
                    self.fonts_used.add(marker_info.font_family)

                    # Convert spacing from points to pixels
                    step_x_px = marker_info.step_x * 1.33
                    step_y_px = marker_info.step_y * 1.33
                    r, g, b, a = marker_info.color

                    html_code = f"&#{char_index};"
                    character = chr(char_index)

                    symbols.extend(
                        [
                            "  SYMBOL",
                            f'    NAME "{symbol_name}"',
                            "    TYPE TRUETYPE",
                            f'    FONT "{font_name}"',
                            f"    CHARACTER \"{html_code}\"   # → '{character}'",
                            "    FILLED TRUE",
                            "    ANTIALIAS TRUE",
                            f"    # Color: RGB({r}, {g}, {b})",
                            f"    # Grid spacing: {step_x_px:.1f}x{step_y_px:.1f} px",
                            "  END",
                            "",
                        ]
                    )

        return symbols

    # ========================================================================
    # FONTSET GENERATION
    # ========================================================================

    def generate_fontset(self, font_paths: Dict[str, str] = None) -> str:
        """Generate MapServer FONTSET file."""
        lines = [
            "# MapServer FONTSET file",
            "# Font definitions for TrueType fonts used in GeoCover",
            "#",
            "# Format: <alias> <font_file_path>",
            "",
        ]

        # Hardcoded GeoCover fonts
        GEOCOVER_FONTS = {
            "geofonts": "fonts/geofontsregular.ttf",
            "geofonts1": "fonts/GeoFonts1.ttf",
            "geofonts2": "fonts/GeoFonts2.ttf",
        }

        # Override with provided paths if given
        if font_paths:
            GEOCOVER_FONTS.update(font_paths)

        # Add fonts that were actually used
        fonts_added = set()
        for font_family in sorted(self.fonts_used):
            sanitized_name = sanitize_font_name(font_family)

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
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# QGIS GENERATOR (continued in next section due to length)
# ============================================================================


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

    # Expression building methods...
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
                elif expected_value in ("999997", "999999"):
                    conditions.append(f'"{field_name}" = {expected_value}')
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
        # Implementation continues...
        # (Due to length, implement similar to existing code but with integrated
        # complex polygon support from symbol_utils)
        pass
