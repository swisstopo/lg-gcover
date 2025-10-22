"""
ENHANCED utility functions for extracting symbol information from ESRI CIM format.

BACKWARDS COMPATIBLE - all existing functions maintained.
NEW: Complete extraction of CIMHatchFill, CIMPictureFill, and other complex patterns.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gcover.publish.symbol_models import (CharacterMarkerInfo, GradientFillInfo,
                            HatchFillInfo, PictureFillInfo, SymbolLayersInfo)

# =============================================================================
# EXISTING FUNCTIONS (maintained for compatibility)
# =============================================================================


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

    elif color_type == "CIMHSVColor" and len(values) >= 3:
        # HSV to RGB conversion
        import colorsys

        h, s, v = values[:3]
        alpha = int(values[3] * 2.55) if len(values) > 3 else 255

        r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
        return (int(r * 255), int(g * 255), int(b * 255), alpha)

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

    # Extract rotation
    rotation = layer.get("rotation", 0.0)

    return CharacterMarkerInfo(
        character_index=layer.get("characterIndex", 0),
        font_family=layer.get("fontFamilyName", "GeoFonts1"),
        size=layer.get("size", 8),
        color=color,
        offset_x=placement.get("offsetX", 0.0),
        offset_y=placement.get("offsetY", 0.0),
        step_x=placement.get("stepX", 10.0),
        step_y=placement.get("stepY", 10.0),
        rotation=rotation,
    )


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


# =============================================================================
# ENHANCED FUNCTION (replaces old extract_polygon_symbol_layers)
# =============================================================================


def extract_polygon_symbol_layers(raw_symbol: Dict) -> SymbolLayersInfo:
    """
    ENHANCED: Extract ALL symbol layers from a CIMPolygonSymbol.

    Now extracts:
    - Outline stroke (with dash patterns)
    - Solid fill layers
    - Hatch fill patterns (CIMHatchFill) ⭐ NEW
    - Character marker patterns
    - Picture fills (CIMPictureFill) ⭐ NEW
    - Gradient fills (CIMGradientFill) ⭐ NEW

    Maintains proper layer ordering for correct rendering.

    Args:
        raw_symbol: CIM polygon symbol dictionary

    Returns:
        SymbolLayersInfo with all extracted components
    """
    result = SymbolLayersInfo()

    if not raw_symbol or "symbolLayers" not in raw_symbol:
        return result

    # Process layers in order (bottom to top rendering)
    for layer in raw_symbol["symbolLayers"]:
        layer_type = layer.get("type", "")

        # Skip disabled layers
        if not layer.get("enable", True):
            logger.debug(f"Skipping disabled layer: {layer_type}")
            continue

        try:
            if layer_type == "CIMSolidStroke":
                _extract_solid_stroke(layer, result)

            elif layer_type == "CIMSolidFill":
                _extract_solid_fill(layer, result)

            elif layer_type == "CIMHatchFill":
                _extract_hatch_fill(layer, result)

            elif layer_type == "CIMCharacterMarker":
                _extract_character_marker(layer, result)

            elif layer_type == "CIMPictureFill":
                _extract_picture_fill(layer, result)

            elif layer_type == "CIMGradientFill":
                _extract_gradient_fill(layer, result)

            elif layer_type == "CIMVectorMarker":
                # Vector markers in polygons (rare but possible)
                logger.debug(f"CIMVectorMarker in polygon - partial support")

            else:
                logger.warning(f"Unsupported polygon layer type: {layer_type}")

        except Exception as e:
            logger.error(f"Error extracting {layer_type}: {e}")

    return result


# =============================================================================
# INTERNAL EXTRACTION FUNCTIONS (NEW)
# =============================================================================


def _extract_solid_stroke(layer: Dict, result: SymbolLayersInfo):
    """Extract CIMSolidStroke as outline."""
    line_style = extract_line_style_from_effects(layer.get("effects", []))

    result.set_outline(
        {
            "width": layer.get("width", 0.4),
            "color": extract_color_from_cim(layer.get("color")),
            "cap_style": layer.get("capStyle", "Round"),
            "join_style": layer.get("joinStyle", "Round"),
            "line_style": line_style,
        }
    )


def _extract_solid_fill(layer: Dict, result: SymbolLayersInfo):
    """Extract CIMSolidFill."""
    color = extract_color_from_cim(layer.get("color"))
    result.add_solid_fill(color)


def _extract_hatch_fill(layer: Dict, result: SymbolLayersInfo):
    """
    ⭐ NEW: Extract CIMHatchFill pattern.

    Hatch fills are diagonal line patterns defined by:
    - rotation: Angle of the lines
    - separation: Spacing between lines
    - lineSymbol: Optional line properties (color, width, style)
    """
    rotation = layer.get("rotation", 0.0)
    separation = layer.get("separation", 5.0)
    offset_x = layer.get("offsetX", 0.0)
    offset_y = layer.get("offsetY", 0.0)

    # Extract line symbol properties
    line_symbol = None
    line_symbol_obj = layer.get("lineSymbol", {})

    if line_symbol_obj and "symbolLayers" in line_symbol_obj:
        # Parse nested line symbol
        for line_layer in line_symbol_obj["symbolLayers"]:
            if line_layer.get("type") == "CIMSolidStroke":
                line_style = extract_line_style_from_effects(
                    line_layer.get("effects", [])
                )

                line_symbol = {
                    "width": line_layer.get("width", 0.5),
                    "color": extract_color_from_cim(line_layer.get("color")),
                    "line_style": line_style,
                }
                break

    # Create default line symbol if none specified
    if not line_symbol:
        line_symbol = {
            "width": 0.5,
            "color": (0, 0, 0, 255),
            "line_style": {"type": "solid", "pattern": None},
        }

    hatch = HatchFillInfo(
        rotation=rotation,
        separation=separation,
        line_symbol=line_symbol,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    result.add_hatch_fill(hatch)
    logger.debug(f"Extracted hatch fill: rotation={rotation}°, separation={separation}")


def _extract_character_marker(layer: Dict, result: SymbolLayersInfo):
    """Extract CIMCharacterMarker pattern fill."""
    marker_info = extract_character_marker_info(layer)
    if marker_info:
        result.add_character_marker(marker_info)


def _extract_picture_fill(layer: Dict, result: SymbolLayersInfo):
    """
    ⭐ NEW: Extract CIMPictureFill.

    Picture fills use images (typically SVG or raster) as fill patterns.
    """
    url = layer.get("url")
    width = layer.get("width", 10.0)
    height = layer.get("height", 10.0)
    rotation = layer.get("rotation", 0.0)
    scale_x = layer.get("scaleX", 1.0)
    scale_y = layer.get("scaleY", 1.0)

    # Try to extract embedded image data
    image_data = None
    if "imageData" in layer:
        image_data = layer["imageData"]  # Base64 encoded

    picture = PictureFillInfo(
        url=url,
        data=image_data,
        width=width,
        height=height,
        rotation=rotation,
        scale_x=scale_x,
        scale_y=scale_y,
    )

    result.add_picture_fill(picture)
    logger.debug(f"Extracted picture fill: {url or 'embedded data'}")


def _extract_gradient_fill(layer: Dict, result: SymbolLayersInfo):
    """
    ⭐ NEW: Extract CIMGradientFill.

    Gradient fills have multiple color stops.
    """
    gradient_method = layer.get("gradientMethod", "Linear")
    angle = layer.get("angle", 0.0)

    # Extract color ramp
    color_ramp = layer.get("colorRamp", {})
    color_stops = color_ramp.get("colorStops", [])

    colors = []
    positions = []

    for stop in color_stops:
        color = extract_color_from_cim(stop.get("color"))
        position = stop.get("position", 0.0)
        colors.append(color)
        positions.append(position)

    # Ensure at least 2 colors for gradient
    if len(colors) < 2:
        colors = [(128, 128, 128, 255), (255, 255, 255, 255)]
        positions = [0.0, 1.0]

    gradient = GradientFillInfo(
        gradient_type=gradient_method.lower(),
        colors=colors,
        positions=positions,
        angle=angle,
    )

    result.add_gradient_fill(gradient)
    logger.debug(f"Extracted {gradient_method} gradient with {len(colors)} stops")


# =============================================================================
# NEW: Advanced Extraction Utilities
# =============================================================================


def extract_full_line_symbol(symbol_obj: Dict) -> Dict[str, Any]:
    """
    Extract complete line symbol information including character markers.

    Used for lines that use font characters as decorations.

    Returns:
        Dictionary with stroke and optional marker information
    """
    result = {"stroke": None, "markers": []}

    if not symbol_obj or "symbolLayers" not in symbol_obj:
        return result

    for layer in symbol_obj["symbolLayers"]:
        if not layer.get("enable", True):
            continue

        layer_type = layer.get("type", "")

        if layer_type == "CIMSolidStroke":
            line_style = extract_line_style_from_effects(layer.get("effects", []))
            result["stroke"] = {
                "width": layer.get("width", 0.4),
                "color": extract_color_from_cim(layer.get("color")),
                "cap_style": layer.get("capStyle", "Round"),
                "join_style": layer.get("joinStyle", "Round"),
                "line_style": line_style,
            }

        elif layer_type == "CIMCharacterMarker":
            # Character markers along lines
            placement = layer.get("markerPlacement", {})
            placement_type = placement.get("type", "")

            if placement_type == "CIMMarkerPlacementAlongLineSameSize":
                # Extract color from nested symbol
                color = (0, 0, 0, 255)
                nested_symbol = layer.get("symbol", {})
                if nested_symbol:
                    for nested_layer in nested_symbol.get("symbolLayers", []):
                        if nested_layer.get("type") == "CIMSolidFill":
                            color = extract_color_from_cim(nested_layer.get("color"))
                            break

                result["markers"].append(
                    {
                        "character_index": layer.get("characterIndex", 0),
                        "font_family": layer.get("fontFamilyName", "Arial"),
                        "size": layer.get("size", 8),
                        "color": color,
                        "rotation": layer.get("rotation", 0.0),
                        "placement": "along_line",
                    }
                )

    return result


def extract_full_point_symbol(symbol_obj: Dict) -> Dict[str, Any]:
    """
    Extract complete point symbol information.

    Handles:
    - Simple markers (geometric shapes)
    - Character markers (font-based)
    - Vector markers (custom graphics)

    Returns:
        Dictionary with marker type and properties
    """
    result = {
        "marker_type": "simple",
        "size": 8.0,
        "color": (0, 0, 0, 255),
        "rotation": 0.0,
        "font_family": None,
        "character_index": None,
    }

    if not symbol_obj or "symbolLayers" not in symbol_obj:
        return result

    for layer in symbol_obj["symbolLayers"]:
        if not layer.get("enable", True):
            continue

        layer_type = layer.get("type", "")

        if layer_type == "CIMSimpleMarker":
            result["marker_type"] = "simple"
            result["size"] = layer.get("size", 8.0)
            result["color"] = extract_color_from_cim(layer.get("color"))
            result["rotation"] = layer.get("rotation", 0.0)
            break

        elif layer_type == "CIMCharacterMarker":
            result["marker_type"] = "character"
            result["size"] = layer.get("size", 8.0)
            result["rotation"] = layer.get("rotation", 0.0)
            result["font_family"] = layer.get("fontFamilyName", "Arial")
            result["character_index"] = layer.get("characterIndex", 0)

            # Extract color from nested symbol
            nested_symbol = layer.get("symbol", {})
            if nested_symbol:
                for nested_layer in nested_symbol.get("symbolLayers", []):
                    if nested_layer.get("type") == "CIMSolidFill":
                        result["color"] = extract_color_from_cim(
                            nested_layer.get("color")
                        )
                        break
            break

        elif layer_type == "CIMVectorMarker":
            result["marker_type"] = "vector"
            result["size"] = layer.get("size", 8.0)
            result["rotation"] = layer.get("rotation", 0.0)

            # Extract color from marker graphics
            marker_graphics = layer.get("markerGraphics", [])
            for graphic in marker_graphics:
                graphic_symbol = graphic.get("symbol", {})
                if graphic_symbol:
                    for nested_layer in graphic_symbol.get("symbolLayers", []):
                        if nested_layer.get("type") == "CIMSolidFill":
                            result["color"] = extract_color_from_cim(
                                nested_layer.get("color")
                            )
                            break
                if result["color"] != (0, 0, 0, 255):
                    break
            break

    return result


def analyze_symbol_complexity(symbol_layers: SymbolLayersInfo) -> Dict[str, Any]:
    """
    Analyze the complexity of a symbol for generation decisions.

    Returns metrics to help decide if custom override is needed:
    - layer_count: Total number of layers
    - has_hatch: Contains hatch patterns
    - has_multiple_fills: Multiple fill layers
    - has_character_markers: Uses font patterns
    - complexity_score: 0-100 (higher = more complex)

    Returns:
        Dictionary with complexity metrics
    """
    metrics = {
        "layer_count": len(symbol_layers.fills) + (1 if symbol_layers.outline else 0),
        "has_hatch": False,
        "has_multiple_fills": len(symbol_layers.fills) > 1,
        "has_character_markers": len(symbol_layers.character_markers) > 0,
        "has_picture_fill": False,
        "has_gradient": False,
        "complexity_score": 0,
    }

    # Check fill types
    for fill in symbol_layers.fills:
        fill_type = fill.get("type", "solid")
        if fill_type == "hatch":
            metrics["has_hatch"] = True
        elif fill_type == "picture":
            metrics["has_picture_fill"] = True
        elif fill_type == "gradient":
            metrics["has_gradient"] = True

    # Calculate complexity score
    score = 0

    # Base layer count contribution
    score += min(metrics["layer_count"] * 10, 30)

    # Complex pattern types
    if metrics["has_hatch"]:
        score += 20
    if metrics["has_character_markers"]:
        score += 15
    if metrics["has_picture_fill"]:
        score += 25
    if metrics["has_gradient"]:
        score += 20

    # Multiple fills increase complexity
    if metrics["has_multiple_fills"]:
        score += 10

    metrics["complexity_score"] = min(score, 100)

    return metrics


def should_use_override(symbol_layers: SymbolLayersInfo, threshold: int = 70) -> bool:
    """
    Determine if a symbol should use a custom override.

    Args:
        symbol_layers: Extracted symbol information
        threshold: Complexity score threshold (default: 70)

    Returns:
        True if symbol is too complex for automatic generation
    """
    metrics = analyze_symbol_complexity(symbol_layers)
    return metrics["complexity_score"] >= threshold


# =============================================================================
# NEW: Symbol Comparison and Diffing
# =============================================================================


def compare_symbols(
    old_symbol: SymbolLayersInfo, new_symbol: SymbolLayersInfo
) -> Dict[str, Any]:
    """
    Compare two symbol definitions to detect changes.

    Useful for detecting when .lyrx styles have been updated.

    Returns:
        Dictionary with:
        - changed: Boolean
        - changes: List of change descriptions
    """
    changes = []

    # Compare outline
    if (old_symbol.outline is None) != (new_symbol.outline is None):
        changes.append("Outline added/removed")
    elif old_symbol.outline and new_symbol.outline:
        if old_symbol.outline != new_symbol.outline:
            changes.append("Outline properties changed")

    # Compare fill count
    if len(old_symbol.fills) != len(new_symbol.fills):
        changes.append(
            f"Fill count changed: {len(old_symbol.fills)} → {len(new_symbol.fills)}"
        )

    # Compare fill types
    old_types = [f.get("type") for f in old_symbol.fills]
    new_types = [f.get("type") for f in new_symbol.fills]
    if old_types != new_types:
        changes.append(f"Fill types changed: {old_types} → {new_types}")

    # Compare character markers
    if len(old_symbol.character_markers) != len(new_symbol.character_markers):
        changes.append(f"Character marker count changed")

    return {"changed": len(changes) > 0, "changes": changes}


# =============================================================================
# NEW: Export Utilities for Debugging
# =============================================================================


def export_symbol_to_dict(symbol_layers: SymbolLayersInfo) -> Dict[str, Any]:
    """
    Export complete symbol information to a serializable dictionary.

    Useful for debugging, JSON export, or custom symbol definitions.
    """
    result = {
        "outline": symbol_layers.outline,
        "fills": [],
        "layer_order": symbol_layers.layer_order,
        "complexity": analyze_symbol_complexity(symbol_layers),
    }

    for fill in symbol_layers.fills:
        fill_copy = fill.copy()

        # Convert CharacterMarkerInfo to dict if present
        if "marker_info" in fill_copy:
            marker = fill_copy["marker_info"]
            if hasattr(marker, "__dict__"):
                fill_copy["marker_info"] = {
                    "character_index": marker.character_index,
                    "font_family": marker.font_family,
                    "size": marker.size,
                    "color": marker.color,
                    "offset_x": marker.offset_x,
                    "offset_y": marker.offset_y,
                    "step_x": marker.step_x,
                    "step_y": marker.step_y,
                    "rotation": marker.rotation,
                }

        result["fills"].append(fill_copy)

    return result


def print_symbol_summary(symbol_layers: SymbolLayersInfo, name: str = "Symbol"):
    """
    Print a human-readable summary of a symbol.

    Useful for debugging during development.
    """
    print(f"\n=== {name} ===")

    # Outline
    if symbol_layers.outline:
        outline = symbol_layers.outline
        color = outline["color"]
        width = outline["width"]
        style = outline["line_style"]["type"]
        print(f"Outline: {width}pt, RGB{color[:3]}, {style}")
    else:
        print("Outline: None")

    # Fills
    print(f"\nFills ({len(symbol_layers.fills)}):")
    for i, fill in enumerate(symbol_layers.fills):
        fill_type = fill.get("type", "unknown")

        if fill_type == "solid":
            color = fill["color"]
            print(f"  {i + 1}. Solid: RGB{color[:3]}")

        elif fill_type == "hatch":
            rotation = fill["rotation"]
            separation = fill["separation"]
            print(f"  {i + 1}. Hatch: {rotation}° angle, {separation}pt spacing")

        elif fill_type == "character":
            marker = fill.get("marker_info")
            if marker:
                print(
                    f"  {i + 1}. Character: {marker.font_family} #{marker.character_index}"
                )

        elif fill_type == "picture":
            url = fill.get("url", "embedded")
            print(f"  {i + 1}. Picture: {url}")

        elif fill_type == "gradient":
            grad_type = fill.get("gradient_type")
            num_stops = len(fill.get("colors", []))
            print(f"  {i + 1}. Gradient: {grad_type}, {num_stops} stops")

    # Complexity
    metrics = analyze_symbol_complexity(symbol_layers)
    print(f"\nComplexity Score: {metrics['complexity_score']}/100")
    if should_use_override(symbol_layers):
        print("⚠️  Recommend custom override for this symbol")

    print("=" * (len(name) + 8) + "\n")
