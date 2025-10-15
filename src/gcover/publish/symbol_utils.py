"""
Utility functions for extracting symbol information from ESRI CIM format.

Provides functions to parse CIM JSON structures and extract:
- Color information (RGB/CMYK conversion)
- Line styles and dash patterns
- Character marker information for pattern fills
- Multi-layer polygon symbols
"""

from typing import Dict, List, Optional, Tuple

from .symbol_models import CharacterMarkerInfo, SymbolLayersInfo


def extract_color_from_cim(color_obj: Dict) -> Tuple[int, int, int, int]:
    """
    Extract RGBA color tuple from CIM color object.

    Handles both CIMRGBColor and CIMCMYKColor formats.

    Args:
        color_obj: CIM color dictionary with type and values

    Returns:
        Tuple of (r, g, b, a) where all values are 0-255
    """
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
    """
    Extract line style (dash pattern) from CIM effects.

    Args:
        effects: List of CIM effect dictionaries

    Returns:
        Dictionary with 'type' (solid/dash/dot) and 'pattern' (list of values)
    """
    for effect in effects:
        if effect.get("type") == "CIMGeometricEffectDashes":
            dash_template = effect.get("dashTemplate", [])
            if dash_template:
                # Determine if dash or dot based on first value
                style_type = "dash" if dash_template[0] >= 2 else "dot"
                return {"type": style_type, "pattern": dash_template}

    return {"type": "solid", "pattern": None}


def extract_character_marker_info(layer: Dict) -> Optional[CharacterMarkerInfo]:
    """
    Extract character marker information from CIMCharacterMarker layer.

    Args:
        layer: CIM character marker layer dictionary

    Returns:
        CharacterMarkerInfo object or None if not a valid pattern fill
    """
    placement = layer.get("markerPlacement", {})

    # Only handle CIMMarkerPlacementInsidePolygon (pattern fills)
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


def extract_polygon_symbol_layers(raw_symbol: Dict) -> SymbolLayersInfo:
    """
    Extract all symbol layers from a CIMPolygonSymbol.

    Parses a complex CIM polygon symbol and extracts:
    - Outline stroke
    - Solid fill layers
    - Character marker pattern fills

    Args:
        raw_symbol: CIM polygon symbol dictionary

    Returns:
        SymbolLayersInfo with all extracted components
    """
    result = SymbolLayersInfo()

    if not raw_symbol or "symbolLayers" not in raw_symbol:
        return result

    for layer in raw_symbol["symbolLayers"]:
        layer_type = layer.get("type", "")

        # Skip disabled layers
        if not layer.get("enable", True):
            continue

        if layer_type == "CIMSolidStroke":
            # Outline layer
            result.outline = {
                "width": layer.get("width", 0.4),
                "color": extract_color_from_cim(layer.get("color")),
                "cap_style": layer.get("capStyle", "Round"),
                "join_style": layer.get("joinStyle", "Round"),
                "line_style": extract_line_style_from_effects(layer.get("effects", [])),
            }

        elif layer_type == "CIMSolidFill":
            # Simple fill layer
            result.fills.append(
                {"type": "solid", "color": extract_color_from_cim(layer.get("color"))}
            )

        elif layer_type == "CIMCharacterMarker":
            # Character marker pattern fill
            marker_info = extract_character_marker_info(layer)
            if marker_info:
                result.character_markers.append(marker_info)

    return result


def sanitize_font_name(font_family: str) -> str:
    """
    Sanitize font family name for use in MapServer/QGIS.

    Removes spaces and special characters to create a valid identifier.

    Args:
        font_family: Original font family name

    Returns:
        Sanitized font name
    """
    return font_family.lower().replace(" ", "").replace("-", "_")
