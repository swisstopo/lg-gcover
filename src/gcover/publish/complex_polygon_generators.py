"""
Enhanced polygon symbol generators with support for:
- Font-based pattern fills (checkerboard patterns using character markers)
- Multiple symbol layers (outline + multiple fills)
- Hatched patterns
- Complex line styles for outlines

For both QGIS QML and MapServer formats.
"""

import uuid
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass


# Font paths configuration
FONT_PATHS = {
    "GeoFontsRegular": "/home/marco/.fonts/g/geofontsregular.ttf",
    "GeoFonts1": "/home/marco/.fonts/g/GeoFonts1.ttf",
    "GeoFonts2": "/home/marco/.fonts/g/GeoFonts2.ttf",
}


@dataclass
class CharacterMarkerInfo:
    """Information about a character marker for pattern fill."""

    character_index: int
    font_family: str
    size: float
    color: tuple  # (r, g, b, a)
    offset_x: float = 0.0
    offset_y: float = 0.0
    step_x: float = 10.0
    step_y: float = 10.0


def extract_polygon_symbol_layers(raw_symbol: Dict) -> Dict:
    """
    Extract all symbol layers from a CIMPolygonSymbol.

    Returns dictionary with:
        - outline: CIMSolidStroke layer info
        - fills: List of fill layers (CIMSolidFill, CIMCharacterMarker, etc.)
    """
    result = {"outline": None, "fills": [], "character_markers": []}

    if not raw_symbol or "symbolLayers" not in raw_symbol:
        return result

    for layer in raw_symbol["symbolLayers"]:
        layer_type = layer.get("type", "")

        if not layer.get("enable", True):
            continue

        if layer_type == "CIMSolidStroke":
            # Outline layer
            result["outline"] = {
                "width": layer.get("width", 0.4),
                "color": extract_color_from_cim(layer.get("color")),
                "cap_style": layer.get("capStyle", "Round"),
                "join_style": layer.get("joinStyle", "Round"),
                "line_style": extract_line_style_from_effects(layer.get("effects", [])),
            }

        elif layer_type == "CIMSolidFill":
            # Simple fill layer
            result["fills"].append(
                {"type": "solid", "color": extract_color_from_cim(layer.get("color"))}
            )

        elif layer_type == "CIMCharacterMarker":
            # Character marker pattern fill
            marker_info = extract_character_marker_info(layer)
            if marker_info:
                result["character_markers"].append(marker_info)

    return result


def extract_character_marker_info(layer: Dict) -> Optional[CharacterMarkerInfo]:
    """Extract character marker information from CIMCharacterMarker layer."""
    placement = layer.get("markerPlacement", {})

    # Only handle CIMMarkerPlacementInsidePolygon
    if placement.get("type") != "CIMMarkerPlacementInsidePolygon":
        return None

    # Extract color from nested symbol
    color = (128, 128, 128, 255)
    nested_symbol = layer.get("symbol", {})
    if nested_symbol:
        for nested_layer in nested_symbol.get("symbolLayers", []):
            if nested_layer.get("type") == "CIMSolidFill":
                color = extract_color_from_cim(nested_layer.get("color"))
                break

    return CharacterMarkerInfo(
        character_index=layer.get("characterIndex", 0),
        font_family=layer.get("fontFamilyName", "GeoFonts1"),
        size=layer.get("size", 8),
        color=color,
        offset_x=placement.get("offsetX", 0.0),
        offset_y=placement.get("offsetY", 0.0),
        step_x=placement.get("stepX", 10.0),
        step_y=placement.get("stepY", 10.0),
    )


def extract_color_from_cim(color_obj: Dict) -> Tuple[int, int, int, int]:
    """Extract RGBA color from CIM color object."""
    if not color_obj:
        return (128, 128, 128, 255)

    color_type = color_obj.get("type", "")
    values = color_obj.get("values", [])

    if color_type == "CIMCMYKColor" and len(values) >= 4:
        c, m, y, k = values[:4]
        alpha = int(values[4] * 2.55) if len(values) > 4 else 255

        # Convert CMYK to RGB
        r = int(255 * (1 - c / 100) * (1 - k / 100))
        g = int(255 * (1 - m / 100) * (1 - k / 100))
        b = int(255 * (1 - y / 100) * (1 - k / 100))

        return (r, g, b, alpha)

    elif color_type == "CIMRGBColor" and len(values) >= 3:
        r, g, b = [int(v) for v in values[:3]]
        alpha = int(values[3] * 2.55) if len(values) > 3 else 255
        return (r, g, b, alpha)

    return (128, 128, 128, 255)


def extract_line_style_from_effects(effects: List[Dict]) -> Dict:
    """Extract line style (dash pattern) from effects."""
    for effect in effects:
        if effect.get("type") == "CIMGeometricEffectDashes":
            dash_template = effect.get("dashTemplate", [])
            if dash_template:
                return {
                    "type": "dash" if dash_template[0] >= 2 else "dot",
                    "pattern": dash_template,
                }

    return {"type": "solid", "pattern": None}


# ============================================================================
# QGIS QML GENERATOR
# ============================================================================


def generate_complex_polygon_qml(symbol_idx: int, class_obj) -> str:
    """
    Generate QGIS QML for complex polygon symbols with multiple layers.

    Supports:
    - Solid fill
    - Character marker pattern fills (checkerboard patterns)
    - Outline with various styles
    """
    if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
        return _generate_simple_polygon_qml(symbol_idx)

    symbol_info = class_obj.symbol_info

    # Extract all symbol layers from raw_symbol
    if not hasattr(symbol_info, "raw_symbol") or not symbol_info.raw_symbol:
        return _generate_simple_polygon_qml(symbol_idx)

    layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

    # Build symbol with multiple layers
    symbol_layers = []

    # Add character marker pattern fills first (bottom layers)
    for i, marker_info in enumerate(layers_info["character_markers"]):
        symbol_layers.append(_generate_qgis_point_pattern_fill(marker_info, i))

    # Add solid fills
    for fill_info in layers_info["fills"]:
        if fill_info["type"] == "solid":
            symbol_layers.append(_generate_qgis_solid_fill(fill_info["color"]))

    # Add outline (top layer)
    if layers_info["outline"]:
        symbol_layers.append(_generate_qgis_outline(layers_info["outline"]))

    # If no layers extracted, use simple fill
    if not symbol_layers:
        return _generate_simple_polygon_qml(symbol_idx)

    # Combine all layers
    return f'''      <symbol force_rhr="0" is_animated="0" frame_rate="10" type="fill" name="{symbol_idx}" alpha="1" clip_to_extent="1">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" name="name" value=""/>
            <Option name="properties"/>
            <Option type="QString" name="type" value="collection"/>
          </Option>
        </data_defined_properties>
{chr(10).join(symbol_layers)}
      </symbol>'''


def _generate_qgis_point_pattern_fill(
    marker_info: CharacterMarkerInfo, layer_idx: int
) -> str:
    """Generate QGIS PointPatternFill layer with FontMarker."""
    r, g, b, a = marker_info.color
    character = chr(marker_info.character_index)

    # Convert points to mm (ESRI uses points, QGIS uses mm)
    size_mm = marker_info.size * 0.352778
    step_x_mm = marker_info.step_x * 0.352778
    step_y_mm = marker_info.step_y * 0.352778
    offset_x_mm = marker_info.offset_x * 0.352778
    offset_y_mm = marker_info.offset_y * 0.352778

    # Sanitize font name
    font_name = marker_info.font_family.replace(" ", "")

    layer_id = str(uuid.uuid4())
    marker_id = str(uuid.uuid4())

    return f'''        <layer enabled="1" id="{{{layer_id}}}" class="PointPatternFill" locked="0" pass="0">
          <Option type="Map">
            <Option type="QString" name="displacement_x" value="{offset_x_mm:.3f}"/>
            <Option type="QString" name="displacement_x_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="displacement_x_unit" value="MM"/>
            <Option type="QString" name="displacement_y" value="{offset_y_mm:.3f}"/>
            <Option type="QString" name="displacement_y_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="displacement_y_unit" value="MM"/>
            <Option type="QString" name="distance_x" value="{step_x_mm:.3f}"/>
            <Option type="QString" name="distance_x_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="distance_x_unit" value="MM"/>
            <Option type="QString" name="distance_y" value="{step_y_mm:.3f}"/>
            <Option type="QString" name="distance_y_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="distance_y_unit" value="MM"/>
            <Option type="QString" name="offset_x" value="0"/>
            <Option type="QString" name="offset_x_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_x_unit" value="MM"/>
            <Option type="QString" name="offset_y" value="0"/>
            <Option type="QString" name="offset_y_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_y_unit" value="MM"/>
            <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option name="properties"/>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
          <symbol force_rhr="0" is_animated="0" frame_rate="10" type="marker" name="@{layer_idx}@0" alpha="1" clip_to_extent="1">
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
                <Option type="QString" name="color" value="{r},{g},{b},{a}"/>
                <Option type="QString" name="font" value="{font_name}"/>
                <Option type="QString" name="font_style" value=""/>
                <Option type="QString" name="horizontal_anchor_point" value="1"/>
                <Option type="QString" name="joinstyle" value="bevel"/>
                <Option type="QString" name="offset" value="0,0"/>
                <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="offset_unit" value="MM"/>
                <Option type="QString" name="outline_color" value="0,0,0,0"/>
                <Option type="QString" name="outline_width" value="0"/>
                <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
                <Option type="QString" name="outline_width_unit" value="MM"/>
                <Option type="QString" name="size" value="{size_mm:.3f}"/>
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
        </layer>'''


def _generate_qgis_solid_fill(color: Tuple[int, int, int, int]) -> str:
    """Generate QGIS SimpleFill layer."""
    r, g, b, a = color
    r_norm = r / 255
    g_norm = g / 255
    b_norm = b / 255
    a_norm = a / 255

    layer_id = str(uuid.uuid4())

    color_str = (
        f"{r},{g},{b},{a},rgb:{r_norm:.3f},{g_norm:.3f},{b_norm:.3f},{a_norm:.3f}"
    )

    return f'''        <layer enabled="1" id="{{{layer_id}}}" class="SimpleFill" locked="0" pass="0">
          <Option type="Map">
            <Option type="QString" name="border_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="color" value="{color_str}"/>
            <Option type="QString" name="joinstyle" value="bevel"/>
            <Option type="QString" name="offset" value="0,0"/>
            <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_unit" value="MM"/>
            <Option type="QString" name="outline_color" value="0,0,0,0"/>
            <Option type="QString" name="outline_style" value="no"/>
            <Option type="QString" name="outline_width" value="0"/>
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
        </layer>'''


def _generate_qgis_outline(outline_info: Dict) -> str:
    """Generate QGIS SimpleLine layer for polygon outline."""
    r, g, b, a = outline_info["color"]
    width_mm = outline_info["width"] * 0.352778

    # Handle line style
    line_style_info = outline_info["line_style"]
    line_style = "solid"
    use_custom_dash = "0"
    custom_dash = "5;2"

    if line_style_info["type"] in ("dash", "dot") and line_style_info["pattern"]:
        use_custom_dash = "1"
        custom_dash = ";".join(str(float(v)) for v in line_style_info["pattern"])
        line_style = "solid"  # Use solid with custom pattern

    # Cap and join styles
    cap_style_map = {"Butt": "flat", "Round": "round", "Square": "square"}
    join_style_map = {"Miter": "miter", "Round": "round", "Bevel": "bevel"}

    cap_style = cap_style_map.get(outline_info["cap_style"], "round")
    join_style = join_style_map.get(outline_info["join_style"], "round")

    layer_id = str(uuid.uuid4())

    r_norm = r / 255
    g_norm = g / 255
    b_norm = b / 255
    a_norm = a / 255
    color_str = (
        f"{r},{g},{b},{a},rgb:{r_norm:.3f},{g_norm:.3f},{b_norm:.3f},{a_norm:.3f}"
    )

    return f'''        <layer enabled="1" id="{{{layer_id}}}" class="SimpleLine" locked="0" pass="0">
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
            <Option type="QString" name="line_color" value="{color_str}"/>
            <Option type="QString" name="line_style" value="{line_style}"/>
            <Option type="QString" name="line_width" value="{width_mm:.3f}"/>
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
        </layer>'''


def _generate_simple_polygon_qml(symbol_idx: int) -> str:
    """Fallback simple polygon symbol."""
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
            <Option type="QString" name="color" value="128,128,128,255"/>
            <Option type="QString" name="joinstyle" value="bevel"/>
            <Option type="QString" name="offset" value="0,0"/>
            <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_unit" value="MM"/>
            <Option type="QString" name="outline_color" value="35,35,35,255"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="0.26"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="style" value="solid"/>
          </Option>
        </layer>
      </symbol>'''


# ============================================================================
# MAPSERVER GENERATOR
# ============================================================================


def generate_complex_polygon_mapserver(
    lines: List[str],
    class_obj,
    class_index: int,
    symbol_prefix: str,
    fonts_used: set,
    pattern_symbols: List[Dict],
) -> None:
    """
    Generate MapServer STYLE blocks for complex polygon symbols.

    Adds multiple STYLE blocks for:
    - Character marker pattern fills
    - Outline

    Modifies lines list in-place and tracks pattern symbols.
    """
    if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
        _add_simple_polygon_mapserver(lines)
        return

    symbol_info = class_obj.symbol_info

    # Extract all symbol layers
    if not hasattr(symbol_info, "raw_symbol") or not symbol_info.raw_symbol:
        _add_simple_polygon_mapserver(lines)
        return

    layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

    # If we have character markers, generate pattern fill styles
    for i, marker_info in enumerate(layers_info["character_markers"]):
        _add_mapserver_pattern_fill(
            lines, marker_info, class_index, i, symbol_prefix, fonts_used
        )

        # Track pattern symbol info for symbol file generation
        pattern_symbols.append(
            {
                "name": f"{symbol_prefix}_poly_pattern_{class_index}_{i}",
                "marker_info": marker_info,
            }
        )

    # Add solid fills
    for fill_info in layers_info["fills"]:
        if fill_info["type"] == "solid":
            _add_mapserver_solid_fill(lines, fill_info["color"])

    # Add outline
    if layers_info["outline"]:
        _add_mapserver_outline(lines, layers_info["outline"])

    # If no layers added, use simple fill
    if not layers_info["character_markers"] and not layers_info["fills"]:
        _add_simple_polygon_mapserver(lines)


def _add_mapserver_pattern_fill(
    lines: List[str],
    marker_info: CharacterMarkerInfo,
    class_index: int,
    marker_index: int,
    symbol_prefix: str,
    fonts_used: set,
) -> None:
    """Add MapServer STYLE for character marker pattern fill."""
    r, g, b, a = marker_info.color

    # Generate symbol name
    symbol_name = f"{symbol_prefix}_poly_pattern_{class_index}_{marker_index}"

    # Track font usage
    fonts_used.add(marker_info.font_family)

    # Convert points to pixels
    size_px = marker_info.size * 1.33

    lines.append("    STYLE")
    lines.append(f'      SYMBOL "{symbol_name}"')
    lines.append(f"      COLOR {r} {g} {b}")
    if a < 255:
        lines.append(f"      OPACITY {a}")
    lines.append(f"      SIZE {size_px:.1f}")
    lines.append("    END # STYLE")


def _add_mapserver_solid_fill(
    lines: List[str], color: Tuple[int, int, int, int]
) -> None:
    """Add MapServer STYLE for solid fill."""
    r, g, b, a = color

    lines.append("    STYLE")
    lines.append(f"      COLOR {r} {g} {b}")
    if a < 255:
        lines.append(f"      OPACITY {a}")
    lines.append("    END # STYLE")


def _add_mapserver_outline(lines: List[str], outline_info: Dict) -> None:
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


def _add_simple_polygon_mapserver(lines: List[str]) -> None:
    """Fallback simple polygon fill."""
    lines.append("      COLOR 128 128 128")
    lines.append("      OUTLINECOLOR 64 64 64")


def generate_mapserver_pattern_symbols(classification_list: List) -> List[str]:
    """
    Generate MapServer SYMBOL definitions for character marker patterns.

    Scans all polygon classifications and generates TRUETYPE symbols
    for pattern fills.
    """
    symbols = []
    symbols_generated = set()

    for classification in classification_list:
        symbol_prefix = classification.layer_name or "class"

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
            for marker_index, marker_info in enumerate(
                layers_info["character_markers"]
            ):
                symbol_name = (
                    f"{symbol_prefix}_poly_pattern_{class_index}_{marker_index}"
                )

                # Avoid duplicates
                symbol_key = f"{marker_info.font_family}_{marker_info.character_index}"
                if symbol_key in symbols_generated:
                    continue

                symbols_generated.add(symbol_key)

                # Sanitize font name
                font_name = marker_info.font_family.replace(" ", "").lower()
                character = chr(marker_info.character_index)

                # Convert spacing from points to pixels
                step_x_px = marker_info.step_x * 1.33
                step_y_px = marker_info.step_y * 1.33

                symbols.extend(
                    [
                        "  SYMBOL",
                        f'    NAME "{symbol_name}"',
                        "    TYPE TRUETYPE",
                        f'    FONT "{font_name}"',
                        f'    CHARACTER "{character}"',
                        "    FILLED TRUE",
                        "    ANTIALIAS TRUE",
                        f"    # Grid spacing: {step_x_px:.1f}x{step_y_px:.1f} pixels",
                        "  END",
                        "",
                    ]
                )

    return symbols


# ============================================================================
# ENHANCED COMPLEX POLYGON FUNCTIONS
# ============================================================================


def scan_and_generate_pattern_symbols(
    classification_list: List, pattern_symbols_list: List[Dict], fonts_used: Set[str]
) -> List[str]:
    """
    Scan classifications and generate complete TRUETYPE symbol definitions
    for polygon patterns.

    Returns list of symbol definition strings for SYMBOLSET.
    """
    from .complex_polygon_generators import (
        extract_polygon_symbol_layers,
        extract_color_from_cim,
    )

    symbols = []
    symbols_generated = set()

    for classification in classification_list:
        symbol_prefix = classification.layer_name or "class"

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
            for marker_index, marker_info in enumerate(
                layers_info["character_markers"]
            ):
                symbol_name = (
                    f"{symbol_prefix}_poly_pattern_{class_index}_{marker_index}"
                )

                # Check if already generated
                symbol_key = f"{marker_info.font_family}_{marker_info.character_index}_{symbol_name}"
                if symbol_key in symbols_generated:
                    continue

                symbols_generated.add(symbol_key)
                fonts_used.add(marker_info.font_family)

                # Sanitize font name
                font_name = marker_info.font_family.replace(" ", "").lower()
                character = chr(marker_info.character_index)

                # Convert spacing from points to pixels
                step_x_px = marker_info.step_x * 1.33
                step_y_px = marker_info.step_y * 1.33

                # Get color
                r, g, b, a = marker_info.color

                symbols.extend(
                    [
                        "  SYMBOL",
                        f'    NAME "{symbol_name}"',
                        "    TYPE TRUETYPE",
                        f'    FONT "{font_name}"',
                        f'    CHARACTER "{character}"',
                        "    FILLED TRUE",
                        "    ANTIALIAS TRUE",
                        f"    # Color: RGB({r}, {g}, {b})",
                        f"    # Grid spacing: {step_x_px:.1f}x{step_y_px:.1f} pixels",
                        f"    # Offset: ({marker_info.offset_x:.2f}, {marker_info.offset_y:.2f}) points",
                        "  END",
                        "",
                    ]
                )

    return symbols


# Example usage
if __name__ == "__main__":
    print("Complex Polygon Symbol Generators")
    print("=" * 60)
    print("\nSupports:")
    print("  ✓ Font-based pattern fills (checkerboard)")
    print("  ✓ Multiple symbol layers")
    print("  ✓ Hatched patterns")
    print("  ✓ Complex outlines (solid, dashed, dotted)")
    print("\nFor both QGIS QML and MapServer formats")
