#!/usr/bin/env python3
"""
ESRI Layer Classification Extractor

Extract classification information from ESRI ArcGIS layers (.lyrx files)
using direct JSON CIM manipulation (no arcpy dependency).

=============================================================================
EXTERNAL API (imported by other modules)
=============================================================================
- extract_lyrx_complete          → publish_cmd, console_generator, esri_classification_applicator
- explore_layer_structure        → publish_cmd
- export_classifications_to_csv  → publish_cmd
- ClassificationJSONEncoder      → publish_cmd
- ClassificationClass            → esri_classification_applicator, generator
- ESRIClassificationExtractor    → esri_classification_applicator, generator
- LayerClassification            → esri_classification_applicator, generator
- SymbolType                     → generator
- CIMColorParser                 → label_extractor_extension

=============================================================================
INTERNAL SUPPORT (used by extract_lyrx_complete)
=============================================================================
- ESRIClassificationExtractorEnhanced
- CIMSymbolParserEnhanced
- ClassificationDisplayer
- export_complete_classification_to_json()
- generate_override_template()
- apply_overrides_to_classification()
- to_serializable_dict()

=============================================================================
CLI-ONLY
=============================================================================
- export_classifications_to_json()
- classify (CLI entry point)
"""

import json
import sys
import zipfile
import difflib
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from gcover.publish.symbol_models import (
    ClassIdentifier,
    SymbolLayersInfo,
    SymbolOverrideRegistry,
)
from gcover.publish.symbol_utils import (
    analyze_symbol_complexity,
    extract_full_line_symbol,
    extract_full_point_symbol,
    extract_polygon_symbol_layers,
)
from gcover.publish.label_extractor_extension import (
    LabelInfo,
    CIMLabelParser,
    LabelInfoDisplayer,
)
from gcover.publish.rotation_extractor_extension import (
    RotationInfo,
    CIMRotationParser,
    RotationInfoDisplayer,
    format_rotation_for_mapserver,
)

console = Console()


# =============================================================================
# ENUMS & BASIC DATA MODELS
# =============================================================================


class SymbolType(Enum):
    """Supported symbol types."""

    POINT = "CIMPointSymbol"
    LINE = "CIMLineSymbol"
    POLYGON = "CIMPolygonSymbol"
    UNKNOWN = "Unknown"


@dataclass
class ColorInfo:
    """Color information extracted from various CIM color formats."""

    r: int = 0
    g: int = 0
    b: int = 0
    alpha: int = 255
    color_type: str = "RGB"
    raw_values: List[float] = field(default_factory=list)

    def to_hex(self) -> str:
        """Convert to hex color string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)


@dataclass
class SymbolInfo:
    """Symbol information extracted from CIM."""

    symbol_type: SymbolType
    size: Optional[float] = None
    width: Optional[float] = None
    color: Optional[ColorInfo] = None

    # Point symbol specific
    font_family: Optional[str] = None
    character_index: Optional[int] = None

    # Line symbol specific
    line_type: Optional[str] = None
    line_style: Optional[str] = None  # 'solid', 'dash', 'dot'
    dash_pattern: Optional[List[float]] = None  # [dash_length, space_length, ...]
    cap_style: Optional[str] = None  # 'Butt', 'Round', 'Square'
    join_style: Optional[str] = None  # 'Miter', 'Round', 'Bevel'

    # Polygon symbol specific
    fill_type: Optional[str] = None

    # Raw CIM data for advanced usage
    raw_symbol: Dict[str, Any] = field(default_factory=dict)


def truncate_label(label: str, max_length: int = 140) -> str:
    """
    Truncate label at first comma or max_length, whichever comes first.

    Args:
        label: Original label
        max_length: Maximum length (default: 140)

    Returns:
        Truncated label
    """
    if not label:
        return ""

    if len(label) <= max_length:
        return label

    comma_pos = label.find(",")
    if comma_pos > 0:
        truncated = label[:comma_pos].strip()
        if len(truncated) <= max_length:
            return truncated

    return label[: max_length - 3].strip() + "..."


@dataclass
class ClassificationClass:
    """A single classification class from CIMUniqueValueRenderer."""

    label: str
    field_values: List[List[str]]
    symbol_info: Optional[SymbolInfo] = None
    visible: bool = True
    raw_class: Dict[str, Any] = field(default_factory=dict)

    # Enhanced extraction fields
    identifier: Optional[ClassIdentifier] = None
    full_symbol_layers: Optional[SymbolLayersInfo] = None
    complexity_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-process the label to truncate if too long."""
        self.label = truncate_label(self.label)


@dataclass
class FieldInfo:
    """Field information used in classification."""

    name: str
    alias: Optional[str] = None
    type: Optional[str] = None


@dataclass
class LayerClassification:
    """Complete classification information for a layer."""

    renderer_type: str
    fields: List[FieldInfo]
    classes: List[ClassificationClass]
    default_label: Optional[str] = None
    default_symbol: Optional[SymbolInfo] = None
    layer_name: Optional[str] = None
    layer_path: Optional[str] = None
    layer_type: Optional[str] = None
    parent_group: Optional[str] = None

    # Data connection fields
    dataset: Optional[str] = None
    feature_dataset: Optional[str] = None
    definition_expression: Optional[str] = None

    # Additional layer properties
    min_scale: Optional[float] = None
    max_scale: Optional[float] = None
    visibility: bool = True
    map_label: Union[None, bool, str] = None

    label_classes: Optional[List[LabelInfo]] = None
    rotation_info: Optional[RotationInfo] = None
    raw_renderer: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# JSON ENCODER
# =============================================================================


class ClassificationJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for classification objects."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dataclass_fields__"):
            return self._dataclass_to_dict(obj)
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        else:
            return super().default(obj)

    def _dataclass_to_dict(self, obj) -> dict:
        """Convert dataclass to dictionary with proper handling of complex objects."""
        result = {}

        for field_name, field_value in asdict(obj).items():
            if field_name.startswith("raw_"):
                continue
            elif isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif isinstance(field_value, (set, tuple)):
                result[field_name] = list(field_value)
            else:
                result[field_name] = field_value

        return result


def to_serializable_dict(obj: Any) -> Dict[str, Any]:
    """Convert any object to a JSON-serializable dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)

            if field_name.startswith("raw_"):
                continue

            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif isinstance(field_value, (list, tuple)):
                result[field_name] = [
                    to_serializable_dict(item)
                    if hasattr(item, "__dataclass_fields__")
                    else item
                    for item in field_value
                ]
            elif isinstance(field_value, set):
                result[field_name] = list(field_value)
            elif hasattr(field_value, "__dataclass_fields__"):
                result[field_name] = to_serializable_dict(field_value)
            else:
                result[field_name] = field_value

        return result
    else:
        return obj


# =============================================================================
# CIM PARSERS
# =============================================================================


class CIMColorParser:
    """Parse various CIM color formats to RGB."""

    @staticmethod
    def parse_color(color_obj: Dict[str, Any]) -> Optional[ColorInfo]:
        """Parse any CIM color object to ColorInfo."""
        if not isinstance(color_obj, dict) or "type" not in color_obj:
            return None

        color_type = color_obj["type"]

        if color_type == "CIMRGBColor":
            return CIMColorParser._parse_rgb_color(color_obj)
        elif color_type == "CIMCMYKColor":
            return CIMColorParser._parse_cmyk_color(color_obj)
        elif color_type == "CIMHSVColor":
            return CIMColorParser._parse_hsv_color(color_obj)
        elif color_type == "CIMLabColor":
            return CIMColorParser._parse_lab_color(color_obj)
        else:
            logger.warning(f"Unknown color type: {color_type}")
            return ColorInfo(
                color_type=color_type, raw_values=color_obj.get("values", [])
            )

    @staticmethod
    def _parse_rgb_color(color_obj: Dict[str, Any]) -> ColorInfo:
        """Parse CIMRGBColor."""
        values = color_obj.get("values", [0, 0, 0, 100])
        r, g, b = values[:3]
        alpha = values[3] if len(values) > 3 else 100

        return ColorInfo(
            r=int(r),
            g=int(g),
            b=int(b),
            alpha=int(alpha * 2.55),
            color_type="RGB",
            raw_values=values,
        )

    @staticmethod
    def _parse_cmyk_color(color_obj: Dict[str, Any]) -> ColorInfo:
        """Parse CIMCMYKColor to RGB."""
        values = color_obj.get("values", [0, 0, 0, 0, 100])
        c, m, y, k = values[:4]
        alpha = values[4] if len(values) > 4 else 100

        r = 255 * (1 - c / 100) * (1 - k / 100)
        g = 255 * (1 - m / 100) * (1 - k / 100)
        b = 255 * (1 - y / 100) * (1 - k / 100)

        return ColorInfo(
            r=int(r),
            g=int(g),
            b=int(b),
            alpha=int(alpha * 2.55),
            color_type="CMYK",
            raw_values=values,
        )

    @staticmethod
    def _parse_hsv_color(color_obj: Dict[str, Any]) -> ColorInfo:
        """Parse CIMHSVColor to RGB."""
        import colorsys

        values = color_obj.get("values", [0, 0, 0, 100])
        h, s, v = values[:3]
        alpha = values[3] if len(values) > 3 else 100

        r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)

        return ColorInfo(
            r=int(r * 255),
            g=int(g * 255),
            b=int(b * 255),
            alpha=int(alpha * 2.55),
            color_type="HSV",
            raw_values=values,
        )

    @staticmethod
    def _parse_lab_color(color_obj: Dict[str, Any]) -> ColorInfo:
        """Parse CIMLabColor to RGB (approximate conversion)."""
        values = color_obj.get("values", [0, 0, 0, 100])
        return ColorInfo(
            r=128,
            g=128,
            b=128,
            alpha=int(values[3] * 2.55) if len(values) > 3 else 255,
            color_type="LAB",
            raw_values=values,
        )


class CIMSymbolParser:
    """Parse CIM symbol definitions to extract styling information."""

    @staticmethod
    def parse_symbol(symbol_obj: Dict[str, Any]) -> Optional[SymbolInfo]:
        """Parse any CIM symbol object."""
        if not isinstance(symbol_obj, dict) or "type" not in symbol_obj:
            return None

        symbol_type_str = symbol_obj["type"]

        if symbol_type_str == "CIMPointSymbol":
            return CIMSymbolParser._parse_point_symbol(symbol_obj)
        elif symbol_type_str == "CIMLineSymbol":
            return CIMSymbolParser._parse_line_symbol(symbol_obj)
        elif symbol_type_str == "CIMPolygonSymbol":
            return CIMSymbolParser._parse_polygon_symbol(symbol_obj)
        else:
            logger.warning(f"Unknown symbol type: {symbol_type_str}")
            return SymbolInfo(symbol_type=SymbolType.UNKNOWN, raw_symbol=symbol_obj)

    @staticmethod
    def _parse_point_symbol(symbol_obj: Dict[str, Any]) -> SymbolInfo:
        """Parse CIMPointSymbol."""
        info = SymbolInfo(symbol_type=SymbolType.POINT, raw_symbol=symbol_obj)

        symbol_layers = symbol_obj.get("symbolLayers", [])

        for layer in symbol_layers:
            if not layer.get("enable", True):
                continue

            layer_type = layer.get("type", "")

            if layer_type == "CIMCharacterMarker":
                info.font_family = layer.get("fontFamilyName")
                info.character_index = layer.get("characterIndex")
                info.size = layer.get("size")

                nested_symbol = layer.get("symbol", {})
                if nested_symbol:
                    color_info = CIMSymbolParser._extract_color_from_symbol(nested_symbol)
                    if color_info:
                        info.color = color_info

            elif layer_type == "CIMVectorMarker":
                info.size = layer.get("size")

                marker_graphics = layer.get("markerGraphics", [])
                for graphic in marker_graphics:
                    graphic_symbol = graphic.get("symbol", {})
                    if graphic_symbol:
                        color_info = CIMSymbolParser._extract_color_from_symbol(graphic_symbol)
                        if color_info:
                            info.color = color_info
                            break

                        symbol_layers_nested = graphic_symbol.get("symbolLayers", [])
                        for nested_layer in symbol_layers_nested:
                            if nested_layer.get("type") == "CIMSolidStroke":
                                stroke_color = CIMColorParser.parse_color(nested_layer.get("color"))
                                if stroke_color:
                                    info.color = stroke_color
                            elif nested_layer.get("type") == "CIMSolidFill":
                                fill_color = CIMColorParser.parse_color(nested_layer.get("color"))
                                if fill_color:
                                    info.color = fill_color

            elif layer_type == "CIMSimpleMarker":
                info.size = layer.get("size")
                color_info = CIMColorParser.parse_color(layer.get("color"))
                if color_info:
                    info.color = color_info

        return info

    @staticmethod
    def _parse_line_symbol(symbol_obj: Dict[str, Any]) -> SymbolInfo:
        """Parse CIMLineSymbol with enhanced dash pattern extraction."""
        info = SymbolInfo(
            symbol_type=SymbolType.LINE,
            raw_symbol=symbol_obj,
            line_style="solid",
        )

        symbol_layers = symbol_obj.get("symbolLayers", [])

        for layer in symbol_layers:
            if not layer.get("enable", True):
                continue

            layer_type = layer.get("type", "")

            if layer_type == "CIMSolidStroke":
                info.line_type = "CIMSolidStroke"
                info.width = layer.get("width")
                info.cap_style = layer.get("capStyle")
                info.join_style = layer.get("joinStyle")

                color_info = CIMColorParser.parse_color(layer.get("color"))
                if color_info:
                    info.color = color_info

                effects = layer.get("effects", [])
                if effects:
                    dash_info = CIMSymbolParser._extract_dash_pattern(effects)
                    if dash_info:
                        info.line_style = dash_info["style"]
                        info.dash_pattern = dash_info["pattern"]

            elif layer_type == "CIMCharacterMarker":
                info.font_family = layer.get("fontFamilyName")
                info.character_index = layer.get("characterIndex")
                info.size = layer.get("size")

        return info

    @staticmethod
    def _extract_dash_pattern(effects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract dash pattern from CIM geometric effects."""
        for effect in effects:
            effect_type = effect.get("type", "")

            if effect_type == "CIMGeometricEffectDashes":
                dash_template = effect.get("dashTemplate", [])

                if not dash_template:
                    continue

                style = CIMSymbolParser._classify_dash_pattern(dash_template)

                return {"style": style, "pattern": dash_template}

        return None

    @staticmethod
    def _classify_dash_pattern(dash_template: List[float]) -> str:
        """Classify a dash template as 'dash' or 'dot'."""
        if not dash_template:
            return "solid"

        first_dash = dash_template[0]
        DOT_THRESHOLD = 2.0

        if first_dash < DOT_THRESHOLD:
            return "dot"
        else:
            return "dash"

    @staticmethod
    def _parse_polygon_symbol(symbol_obj: Dict[str, Any]) -> SymbolInfo:
        """Parse CIMPolygonSymbol."""
        info = SymbolInfo(symbol_type=SymbolType.POLYGON, raw_symbol=symbol_obj)

        symbol_layers = symbol_obj.get("symbolLayers", [])

        for layer in symbol_layers:
            if not layer.get("enable", True):
                continue

            layer_type = layer.get("type", "")

            if layer_type == "CIMSolidFill":
                info.fill_type = "CIMSolidFill"
                color_info = CIMColorParser.parse_color(layer.get("color"))
                if color_info:
                    info.color = color_info

        return info

    @staticmethod
    def _extract_color_from_symbol(symbol_obj: Dict[str, Any]) -> Optional[ColorInfo]:
        """Extract color from nested symbol structures."""
        symbol_layers = symbol_obj.get("symbolLayers", [])

        for layer in symbol_layers:
            layer_type = layer.get("type", "")

            if layer_type in ("CIMSolidFill", "CIMSolidStroke", "CIMSimpleFill"):
                color = CIMColorParser.parse_color(layer.get("color"))
                if color:
                    return color

        marker_graphics = symbol_obj.get("markerGraphics", [])
        for graphic in marker_graphics:
            graphic_symbol = graphic.get("symbol", {})
            if graphic_symbol:
                color = CIMSymbolParser._extract_color_from_symbol(graphic_symbol)
                if color:
                    return color

        return None


class CIMSymbolParserEnhanced(CIMSymbolParser):
    """Enhanced symbol parser with complete extraction support."""

    @staticmethod
    def parse_symbol_complete(symbol_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse symbol with COMPLETE layer extraction.

        Returns both:
        - legacy SymbolInfo for backwards compatibility
        - full SymbolLayersInfo for complete symbol data
        """
        if not isinstance(symbol_obj, dict) or "type" not in symbol_obj:
            return {"legacy": None, "complete": None, "complexity": None}

        symbol_type_str = symbol_obj["type"]

        if symbol_type_str == "CIMPointSymbol":
            legacy_info = CIMSymbolParser._parse_point_symbol(symbol_obj)
            complete_info = extract_full_point_symbol(symbol_obj)
            complexity = None

        elif symbol_type_str == "CIMLineSymbol":
            legacy_info = CIMSymbolParser._parse_line_symbol(symbol_obj)
            complete_info = extract_full_line_symbol(symbol_obj)
            complexity = None

        elif symbol_type_str == "CIMPolygonSymbol":
            legacy_info = CIMSymbolParser._parse_polygon_symbol(symbol_obj)
            complete_info = extract_polygon_symbol_layers(symbol_obj)
            complexity = analyze_symbol_complexity(complete_info)

        else:
            legacy_info = SymbolInfo(symbol_type=SymbolType.UNKNOWN, raw_symbol=symbol_obj)
            complete_info = None
            complexity = None

        return {
            "legacy": legacy_info,
            "complete": complete_info,
            "complexity": complexity,
        }


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================


class ESRIClassificationExtractor:
    """Main class for extracting classification information from ESRI layers."""

    def __init__(self):
        """Initialize extractor (JSON-only mode)."""
        pass

    def extract_from_lyrx(
        self,
        lyrx_path: Union[str, Path],
        identifier_fields: Optional[Dict[str, str]] = None,
    ) -> List[LayerClassification]:
        """
        Extract classification from .lyrx layer file.

        Args:
            lyrx_path: Path to .lyrx file
            identifier_fields: Optional dictionary mapping layer names to field names
                             to use as identifiers.

        Returns:
            List of LayerClassification objects
        """
        lyrx_path = Path(lyrx_path)
        alternative_lyrx = False

        if not lyrx_path.exists():
            parent_dir = lyrx_path.parent
            candidates = [str(p.name) for p in parent_dir.glob("*.lyrx") if p != lyrx_path]
            matches = difflib.get_close_matches(str(lyrx_path.name), candidates, n=1, cutoff=0.8)
            if matches:
                lyrx_path = lyrx_path.parent / matches[0]
                logger.debug(f"Alternative .lyrx file: {lyrx_path}")
                alternative_lyrx = True
            else:
                logger.error(f"Original file not found: {lyrx_path.name}")
                logger.debug(f"Candidates: {candidates}")
                raise FileNotFoundError(f"Layer file not found: {lyrx_path}")
        else:
            logger.debug(f"Extracting classification from {lyrx_path}")

        console.print(f"Extracting from: {lyrx_path} ({'alternative' if alternative_lyrx else 'original'})")

        return self._extract_from_json(lyrx_path, identifier_fields=identifier_fields)

    def _extract_from_json(
        self,
        lyrx_path: Path,
        identifier_fields: Optional[Dict[str, str]] = None,
    ) -> List[LayerClassification]:
        """Extract classification using direct JSON parsing."""
        try:
            lyrx_data = self._load_lyrx_json(lyrx_path)
            all_layers = self._find_all_layers_in_json(lyrx_data)

            classifications = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing layers...", total=len(all_layers))

                for layer_info in all_layers:
                    layer_data = layer_info["layer"]
                    layer_path = layer_info["path"]

                    progress.update(task, description=f"Processing: {layer_path}")

                    renderer = self._find_renderer_in_layer(layer_data)
                    if renderer and renderer.get("type") == "CIMUniqueValueRenderer":
                        layer_name = layer_data.get("name", "Unknown")
                        identifier_field = None
                        if identifier_fields and layer_name in identifier_fields:
                            identifier_field = identifier_fields[layer_name]
                            logger.info(
                                f"Layer '{layer_name}' will use "
                                f"identifier_field: {identifier_field}"
                            )

                        classification = self._parse_unique_value_renderer(
                            renderer,
                            layer_data.get("name", "Unknown"),
                            identifier_field=identifier_field,
                        )

                        if classification:
                            classification.layer_path = layer_path
                            classification.layer_type = layer_data.get("type", "Unknown")

                            self._extract_layer_properties(classification, layer_data)

                            label_classes = layer_data.get("labelClasses", [])
                            if label_classes:
                                try:
                                    label_infos = CIMLabelParser.parse_label_classes(label_classes)
                                    classification.label_classes = label_infos
                                    logger.debug(
                                        f"Extracted {len(label_infos)} label class(es) "
                                        f"for layer {classification.layer_name}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not parse label classes for "
                                        f"{classification.layer_name}: {e}"
                                    )

                            if "/" in layer_path:
                                classification.parent_group = "/".join(layer_path.split("/")[:-1])

                            classifications.append(classification)

                    progress.advance(task)

            return classifications

        except Exception as e:
            logger.error(f"Error parsing {lyrx_path}: {e}")
            return []

    def _load_lyrx_json(self, lyrx_path: Path) -> Dict[str, Any]:
        """Load JSON data from .lyrx file (handles both direct JSON and ZIP container)."""
        if lyrx_path.suffix.lower() == ".lyrx":
            try:
                with open(lyrx_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                with zipfile.ZipFile(lyrx_path, "r") as zf:
                    json_files = [f for f in zf.namelist() if f.endswith(".json")]
                    if not json_files:
                        raise ValueError("No JSON file found in .lyrx container")

                    with zf.open(json_files[0]) as f:
                        return json.load(f)
        else:
            with open(lyrx_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _find_all_layers_in_json(
        self, data: Dict[str, Any], current_path: str = ""
    ) -> List[Dict[str, Any]]:
        """Recursively find all layers in JSON structure, preserving hierarchy."""
        layers = []

        def traverse(obj, path=""):
            if isinstance(obj, dict):
                obj_type = obj.get("type", "")
                obj_name = obj.get("name", "Unknown")

                current_full_path = f"{path}/{obj_name}" if path else obj_name

                if obj_type == "CIMGroupLayer":
                    logger.debug(f"Found group layer: {current_full_path}")
                    group_layers = obj.get("layers", [])

                    for sublayer in group_layers:
                        traverse(sublayer, current_full_path)

                elif obj_type in [
                    "CIMFeatureLayer",
                    "CIMRasterLayer",
                    "CIMAnnotationLayer",
                    "CIMDimensionLayer",
                ]:
                    logger.debug(f"Found feature layer: {current_full_path}")
                    layers.append(
                        {"layer": obj, "path": current_full_path, "type": obj_type}
                    )

                for key, value in obj.items():
                    if key not in ["layers"]:
                        if isinstance(value, (dict, list)):
                            traverse(value, path)

            elif isinstance(obj, list):
                for item in obj:
                    traverse(item, path)

        if "layerDefinitions" in data:
            for layer in data["layerDefinitions"]:
                traverse(layer)
        elif "layers" in data:
            for layer in data["layers"]:
                traverse(layer)
        else:
            traverse(data)

        return layers

    def _find_renderer_in_layer(self, layer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find renderer within a layer object."""
        if "renderer" in layer_data:
            return layer_data["renderer"]

        if "subLayers" in layer_data:
            for sublayer in layer_data["subLayers"]:
                renderer = self._find_renderer_in_layer(sublayer)
                if renderer:
                    return renderer

        return self._find_unique_value_renderer(layer_data)

    def _extract_layer_properties(
        self, classification: LayerClassification, layer_data: Dict[str, Any]
    ):
        """Extract additional layer properties from CIMFeatureLayer data."""
        classification.min_scale = layer_data.get("minScale")
        classification.max_scale = layer_data.get("maxScale")
        classification.visibility = layer_data.get("visibility", True)
        classification.map_label = layer_data.get("map_label", None)

        feature_table = layer_data.get("featureTable", {})
        if feature_table:
            classification.definition_expression = feature_table.get("definitionExpression")

            data_connection = feature_table.get("dataConnection", {})
            if data_connection:
                classification.dataset = data_connection.get("dataset")
                classification.feature_dataset = data_connection.get("featureDataset")

                logger.debug(
                    f"Extracted data connection for {classification.layer_name}: "
                    f"dataset={classification.dataset}, "
                    f"featureDataset={classification.feature_dataset}"
                )

        if classification.definition_expression:
            logger.debug(
                f"Definition expression for {classification.layer_name}: "
                f"{classification.definition_expression}"
            )

    def _find_unique_value_renderer(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recursively find CIMUniqueValueRenderer in JSON structure."""
        if isinstance(data, dict):
            if data.get("type") == "CIMUniqueValueRenderer":
                return data

            for value in data.values():
                result = self._find_unique_value_renderer(value)
                if result:
                    return result

        elif isinstance(data, list):
            for item in data:
                result = self._find_unique_value_renderer(item)
                if result:
                    return result

        return None

    def _parse_unique_value_renderer(
        self,
        renderer: Dict[str, Any],
        layer_name: str = None,
        layer_path: str = None,
        identifier_field: Optional[str] = None,
    ) -> Optional[LayerClassification]:
        """Parse CIMUniqueValueRenderer structure."""
        try:
            field_names = renderer.get("fields", [])
            fields = [FieldInfo(name=name) for name in field_names]

            classes = []
            groups = renderer.get("groups", [])

            for group in groups:
                group_classes = group.get("classes", [])

                for class_obj in group_classes:
                    classification_class = self._parse_classification_class(
                        class_obj,
                        layer_path=layer_path or layer_name or "Unknown",
                        class_index=len(classes),
                        identifier_field=identifier_field,
                        field_names=field_names,
                    )
                    if classification_class:
                        classes.append(classification_class)

            default_symbol = None
            if "defaultSymbol" in renderer:
                default_symbol = CIMSymbolParser.parse_symbol(
                    renderer["defaultSymbol"].get("symbol", {})
                )

            classification = LayerClassification(
                renderer_type="CIMUniqueValueRenderer",
                fields=fields,
                classes=classes,
                default_label=renderer.get("defaultLabel"),
                default_symbol=default_symbol,
                layer_name=layer_name,
                raw_renderer=renderer,
            )

            visual_variables = renderer.get("visualVariables", [])
            if visual_variables:
                try:
                    rotation_info = CIMRotationParser.parse_rotation_variables(visual_variables)
                    if rotation_info:
                        classification.rotation_info = rotation_info
                        logger.info(
                            f"Extracted rotation on field "
                            f"{rotation_info.get_simple_field_name()} "
                            f"({rotation_info.rotation_type}) "
                            f"for layer {layer_name}"
                        )
                except Exception as e:
                    logger.warning(f"Could not parse rotation for {layer_name}: {e}")

            return classification

        except Exception as e:
            logger.error(f"Error parsing unique value renderer: {e}")
            return None

    def _parse_classification_class(
        self,
        class_obj: Dict[str, Any],
        layer_path: str = None,
        class_index: int = 0,
        identifier_field: Optional[str] = None,
        field_names: Optional[List[str]] = None,
    ) -> Optional[ClassificationClass]:
        """Parse a single CIMUniqueValueClass with enhanced extraction."""
        try:
            label = class_obj.get("label", "")
            visible = class_obj.get("visible", True)

            field_values = []
            values = class_obj.get("values", [])

            for value_obj in values:
                if isinstance(value_obj, dict) and "fieldValues" in value_obj:
                    field_values.append(value_obj["fieldValues"])

            symbol_ref = class_obj.get("symbol", {})
            raw_symbol = symbol_ref.get("symbol", {})

            symbol_info = CIMSymbolParser.parse_symbol(raw_symbol)

            full_symbol_layers = None
            complexity_metrics = None

            if raw_symbol and raw_symbol.get("type") == "CIMPolygonSymbol":
                try:
                    full_symbol_layers = extract_polygon_symbol_layers(raw_symbol)
                    complexity_metrics = analyze_symbol_complexity(full_symbol_layers)
                except Exception as e:
                    logger.debug(f"Could not extract complete symbol layers: {e}")

            identifier = None
            if field_values and layer_path:
                try:
                    identifier = ClassIdentifier.create(
                        layer_path=layer_path,
                        field_values=field_values[0] if field_values else [],
                        class_index=class_index,
                        symbol_dict=raw_symbol,
                        label=label,
                    )
                except Exception as e:
                    logger.debug(f"Could not create identifier: {e}")

            return ClassificationClass(
                label=label,
                field_values=field_values,
                symbol_info=symbol_info,
                visible=visible,
                raw_class=class_obj,
                identifier=identifier,
                full_symbol_layers=full_symbol_layers,
                complexity_metrics=complexity_metrics,
            )

        except Exception as e:
            logger.error(f"Error parsing classification class: {e}")
            return None


# =============================================================================
# ENHANCED EXTRACTOR (used by extract_lyrx_complete)
# =============================================================================


class ESRIClassificationExtractorEnhanced(ESRIClassificationExtractor):
    """Enhanced extractor with override registry support."""

    def __init__(self, override_registry: Optional[SymbolOverrideRegistry] = None):
        """
        Initialize extractor with optional override registry.

        Args:
            override_registry: Optional custom symbol override registry
        """
        super().__init__()
        self.class_count = 0
        self.override_registry = override_registry or SymbolOverrideRegistry()

    def load_overrides(self, yaml_path: str):
        """Load custom symbol overrides from YAML file."""
        self.override_registry = SymbolOverrideRegistry.from_yaml(yaml_path)

    def _parse_classification_class(
        self,
        class_obj: Dict[str, Any],
        layer_path: str,
        class_index: int,
        identifier_field: Optional[str] = None,
        field_names: Optional[List[str]] = None,
    ) -> Optional[ClassificationClass]:
        """ENHANCED version of _parse_classification_class."""
        self.class_count += 1

        try:
            label = class_obj.get("label", "")
            visible = class_obj.get("visible", True)

            field_values = []
            values = class_obj.get("values", [])

            for value_obj in values:
                if isinstance(value_obj, dict) and "fieldValues" in value_obj:
                    field_values.append(value_obj["fieldValues"])

            symbol_ref = class_obj.get("symbol", {})
            raw_symbol = symbol_ref.get("symbol", {})

            # Parse with both methods
            symbol_data = CIMSymbolParserEnhanced.parse_symbol_complete(raw_symbol)

            symbol_info = symbol_data["legacy"]
            full_symbol_layers = symbol_data["complete"]
            complexity_metrics = symbol_data["complexity"]

            identifier = None

            if field_values:
                identifier_value = None

                if identifier_field and field_names:
                    try:
                        field_index = field_names.index(identifier_field)
                        if field_index < len(field_values[0]):
                            identifier_value = field_values[0][field_index]
                            if self.class_count < 10:
                                logger.info(
                                    f"Using field '{identifier_field}' value "
                                    f"'{identifier_value}' as identifier"
                                )
                            elif self.class_count == 10:
                                logger.info("more classes...")

                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Could not extract identifier from field '{identifier_field}': {e}. "
                            f"Falling back to class_index."
                        )

                identifier = ClassIdentifier.create(
                    layer_path=layer_path,
                    field_values=field_values[0] if field_values else [],
                    class_index=class_index if identifier_value is None else identifier_value,
                    symbol_dict=raw_symbol,
                    label=label,
                )

            return ClassificationClass(
                label=label,
                field_values=field_values,
                symbol_info=symbol_info,
                visible=visible,
                raw_class=class_obj,
                identifier=identifier,
                full_symbol_layers=full_symbol_layers,
                complexity_metrics=complexity_metrics,
            )

        except Exception as e:
            logger.error(f"Error parsing classification class: {e}")
            return None

    def _parse_unique_value_renderer(
        self,
        renderer: Dict[str, Any],
        layer_name: str = None,
        layer_path: str = None,
        identifier_field: Optional[str] = None,
    ) -> Optional[LayerClassification]:
        """Parse CIMUniqueValueRenderer structure with enhanced extraction."""
        try:
            field_names = renderer.get("fields", [])
            fields = [FieldInfo(name=name) for name in field_names]

            classes = []
            groups = renderer.get("groups", [])

            class_index = 0
            for group in groups:
                group_classes = group.get("classes", [])

                for class_obj in group_classes:
                    classification_class = self._parse_classification_class(
                        class_obj,
                        layer_path=layer_path or layer_name or "Unknown",
                        class_index=class_index,
                        identifier_field=identifier_field,
                        field_names=field_names,
                    )
                    if classification_class:
                        classes.append(classification_class)
                        class_index += 1

            default_symbol = None
            if "defaultSymbol" in renderer:
                default_symbol = CIMSymbolParser.parse_symbol(
                    renderer["defaultSymbol"].get("symbol", {})
                )

            classification = LayerClassification(
                renderer_type="CIMUniqueValueRenderer",
                fields=fields,
                classes=classes,
                default_label=renderer.get("defaultLabel"),
                default_symbol=default_symbol,
                layer_name=layer_name,
                raw_renderer=renderer,
            )

            visual_variables = renderer.get("visualVariables", [])
            if visual_variables:
                try:
                    rotation_info = CIMRotationParser.parse_rotation_variables(visual_variables)
                    if rotation_info:
                        classification.rotation_info = rotation_info
                        logger.info(
                            f"Extracted rotation on field "
                            f"{rotation_info.get_simple_field_name()} "
                            f"({rotation_info.rotation_type}) "
                            f"for layer {layer_name}"
                        )
                except Exception as e:
                    logger.warning(f"Could not parse rotation for {layer_name}: {e}")

            return classification

        except Exception as e:
            logger.error(f"Error parsing unique value renderer: {e}")
            return None


# =============================================================================
# CLASSIFICATION DISPLAYER
# =============================================================================


class ClassificationDisplayer:
    """Display extracted classification information using rich."""

    @staticmethod
    def display_classification(classification: LayerClassification, head=None):
        """Display a single layer classification."""
        console.print()
        insert_ellipsis = False

        title_parts = []
        if classification.parent_group:
            title_parts.append(f"[dim]{classification.parent_group}/[/dim]")
        title_parts.append(f"[bold blue]{classification.layer_name or 'Layer'}[/bold blue]")

        title_text = "".join(title_parts)
        if classification.layer_type:
            title_text += f"\n[dim]{classification.layer_type}[/dim]"
        title_text += f"\nRenderer: {classification.renderer_type}"

        console.print(Panel.fit(title_text, title="Layer Classification"))

        if classification.fields:
            console.print("\n[bold yellow]Classification Fields:[/bold yellow]")
            for field in classification.fields:
                console.print(f"  • {field.name}")

        total_rows = len(classification.classes)
        rows_to_display = classification.classes

        if head and isinstance(head, int) and head > 0:
            rows_to_display = classification.classes[:head]
            insert_ellipsis = total_rows > head

        if classification.classes:
            table = Table(title="Classification Classes", show_header=True)
            table.add_column("Label", style="cyan", width=50, no_wrap=False)
            table.add_column("Values", style="white", width=35)
            table.add_column("Symbol Type", style="magenta", width=20)
            table.add_column("Symbol Info", style="green", width=70)
            table.add_column("Visible", style="yellow", width=8)

            for class_obj in rows_to_display:
                values_str = ""
                for fv in class_obj.field_values:
                    values_str += " | ".join(str(v) for v in fv) + "\n"
                values_str = values_str.strip()

                if len(values_str) > 50:
                    values_str = values_str[:47] + "..."

                # ENHANCED: Use full_symbol_layers if available
                symbol_str = ClassificationDisplayer._format_symbol_info(
                    class_obj.symbol_info,
                    class_obj.full_symbol_layers,
                )

                display_label = class_obj.label

                table.add_row(
                    display_label,
                    values_str,
                    class_obj.symbol_info.symbol_type.value if class_obj.symbol_info else "None",
                    symbol_str,
                    "✓" if class_obj.visible else "✗",
                )

            if insert_ellipsis:
                table.add_row(
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    "[dim]...[/dim]",
                    style="dim",
                )

            console.print(table)

        if classification.label_classes:
            console.print("\n[bold yellow]Label Classes:[/bold yellow]")
            LabelInfoDisplayer.display_label_infos(classification.label_classes, console)

        if classification.rotation_info:
            console.print("\n[bold yellow]Symbol Rotation:[/bold yellow]")
            RotationInfoDisplayer.display_rotation_info(classification.rotation_info, console)

    @staticmethod
    def display_grouped_classifications(classifications: List[LayerClassification]):
        """Display multiple classifications grouped by parent."""
        groups = {}
        standalone_layers = []

        for classification in classifications:
            if classification.parent_group:
                if classification.parent_group not in groups:
                    groups[classification.parent_group] = []
                groups[classification.parent_group].append(classification)
            else:
                standalone_layers.append(classification)

        tree = Tree("🗺️  [bold blue]Layer Classifications[/bold blue]")

        for group_path, group_classifications in groups.items():
            group_parts = group_path.split("/")
            group_node = tree

            for i, part in enumerate(group_parts):
                current_path = "/".join(group_parts[: i + 1])

                existing_node = None
                for child in getattr(group_node, "_children", []):
                    if hasattr(child, "_label") and current_path in str(child._label):
                        existing_node = child
                        break

                if existing_node:
                    group_node = existing_node
                else:
                    group_node = group_node.add(f"📁 [yellow]{part}[/yellow]")

            for classification in group_classifications:
                layer_node = group_node.add(
                    f"🎨 [cyan]{classification.layer_name}[/cyan] ({classification.renderer_type})"
                )

                if classification.classes:
                    classes_info = f"{len(classification.classes)} classes"
                    if classification.fields:
                        fields_info = ", ".join([f.name for f in classification.fields])
                        classes_info += f" on [{fields_info}]"
                    layer_node.add(f"🏷️  {classes_info}")

        for classification in standalone_layers:
            layer_node = tree.add(
                f"🎨 [cyan]{classification.layer_name}[/cyan] ({classification.renderer_type})"
            )

            if classification.classes:
                classes_info = f"{len(classification.classes)} classes"
                if classification.fields:
                    fields_info = ", ".join([f.name for f in classification.fields])
                    classes_info += f" on [{fields_info}]"
                layer_node.add(f"🏷️  {classes_info}")

        console.print(tree)

    @staticmethod
    def _format_symbol_info(
        symbol_info: Optional[SymbolInfo],
        full_symbol_layers: Optional[SymbolLayersInfo] = None,
    ) -> str:
        """
        Format symbol information for display.

        ENHANCED: Now uses SymbolLayersInfo when available for complete info.
        """
        if not symbol_info and not full_symbol_layers:
            return "No symbol"

        parts = []

        # Use full_symbol_layers if available for richer info
        if full_symbol_layers:
          # Count fill layers by type
          fill_types = {}
          if hasattr(full_symbol_layers, 'fills'):
            for fill in full_symbol_layers.fills:
                fill_type = fill.get("type", "unknown")
                fill_types[fill_type] = fill_types.get(fill_type, 0) + 1

            if fill_types:
                fill_summary = ", ".join(f"{count}×{ftype}" for ftype, count in fill_types.items())
                parts.append(f"Fills: [{fill_summary}]")

            # Outline info
            if full_symbol_layers.outline:
                outline = full_symbol_layers.outline
                outline_parts = []
                if "width" in outline:
                    outline_parts.append(f"w={outline['width']:.2f}")
                if "color" in outline:
                    color = outline["color"]
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        outline_parts.append(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
                if outline_parts:
                    parts.append(f"Outline: {', '.join(outline_parts)}")

            # Character markers (hatch patterns with fonts)
            if full_symbol_layers.character_markers:
                marker_count = len(full_symbol_layers.character_markers)
                parts.append(f"CharMarkers: {marker_count}")
                # Show first marker details
                first_marker = full_symbol_layers.character_markers[0]
                parts.append(f"Font: {first_marker.font_family}")
                parts.append(f"Char: {first_marker.character_index}")

            # Show hatch details if present
            for fill in full_symbol_layers.fills:
                if fill.get("type") == "hatch":
                    parts.append(f"Hatch: rot={fill.get('rotation', 0)}° sep={fill.get('separation', 0)}")
                    break  # Just show first hatch

        # Fall back to legacy symbol_info for basic data
        if symbol_info:
            if symbol_info.color and "Color" not in " ".join(parts):
                color_hex = symbol_info.color.to_hex()
                parts.append(f"Color: {color_hex}")

            if symbol_info.size and "Size" not in " ".join(parts):
                parts.append(f"Size: {symbol_info.size:.2f}")

            if symbol_info.width and "Width" not in " ".join(parts):
                parts.append(f"Width: {symbol_info.width:.3f}")

            if symbol_info.font_family and "Font" not in " ".join(parts):
                parts.append(f"Font: {symbol_info.font_family}")

            if symbol_info.character_index is not None and "Char" not in " ".join(parts):
                parts.append(f"Char: {symbol_info.character_index}")

            # Enhanced line style information
            if symbol_info.symbol_type == SymbolType.LINE:
                if symbol_info.line_style:
                    parts.append(f"Style: {symbol_info.line_style}")

                if symbol_info.dash_pattern:
                    pattern_str = "-".join(str(v) for v in symbol_info.dash_pattern)
                    parts.append(f"Pattern: [{pattern_str}]")

                if symbol_info.cap_style:
                    parts.append(f"Cap: {symbol_info.cap_style}")

            # Identify marker type for points
            if symbol_info.symbol_type == SymbolType.POINT:
                if symbol_info.font_family:
                    if "(CharMarker)" not in " ".join(parts):
                        parts.append("(CharMarker)")
                elif symbol_info.size and not symbol_info.font_family:
                    parts.append("(VectorMarker)")

            if symbol_info.line_type and "Type" not in " ".join(parts):
                parts.append(f"Type: {symbol_info.line_type}")

            if symbol_info.fill_type and "Fill" not in " ".join(parts):
                parts.append(f"Fill: {symbol_info.fill_type}")

        return " | ".join(parts) if parts else "Unparsed symbol"


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_classifications_to_json(
    classifications: List[LayerClassification], output_path: Union[str, Path]
) -> Path:
    """Export classification results to JSON file (CLI-only)."""
    output_path = Path(output_path)

    export_data = [to_serializable_dict(classification) for classification in classifications]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, cls=ClassificationJSONEncoder)

    logger.info(f"Exported {len(classifications)} classifications to {output_path}")
    return output_path


def export_classifications_to_csv(
    classifications: List[LayerClassification], output_path: Union[str, Path]
) -> Path:
    """Export classification results to CSV file."""
    import pandas as pd

    output_path = Path(output_path)

    rows = []
    for classification in classifications:
        base_data = {
            "layer_name": classification.layer_name,
            "layer_path": classification.layer_path,
            "layer_type": classification.layer_type,
            "parent_group": classification.parent_group,
            "renderer_type": classification.renderer_type,
            "fields": ", ".join([f.name for f in classification.fields]),
            "num_classes": len(classification.classes),
        }

        for i, class_obj in enumerate(classification.classes):
            row = base_data.copy()
            row.update(
                {
                    "class_index": i,
                    "class_label": class_obj.label,
                    "class_visible": class_obj.visible,
                    "field_values": "; ".join(
                        [" | ".join(str(v) for v in fv) for fv in class_obj.field_values]
                    ),
                }
            )

            if class_obj.symbol_info:
                symbol = class_obj.symbol_info
                row.update(
                    {
                        "symbol_type": symbol.symbol_type.value,
                        "symbol_size": symbol.size,
                        "symbol_width": symbol.width,
                        "symbol_color_hex": symbol.color.to_hex() if symbol.color else None,
                        "symbol_color_rgb": str(symbol.color.to_rgb_tuple()) if symbol.color else None,
                        "font_family": symbol.font_family,
                        "character_index": symbol.character_index,
                        "line_type": symbol.line_type,
                        "fill_type": symbol.fill_type,
                    }
                )

            # ENHANCED: Add full_symbol_layers info
            if class_obj.full_symbol_layers:
                layers = class_obj.full_symbol_layers
                fill_layer_types = None
                fill_types = []
                if hasattr(layers, 'fills'):
                    fill_types = [f.get("type", "unknown") for f in layers.fills]
                    fill_layer_types = ", ".join(fill_types)
                row["fill_layer_types"] = fill_layer_types
                row["fill_layer_count"] = len(fill_types)
                row["has_outline"] = layers.get('outline')
                row["char_marker_count"] = len(layers.get('character_markers', []))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"Exported {len(rows)} classification rows to {output_path}")
    return output_path


def export_complete_classification_to_json(
    classifications: List[LayerClassification],
    output_path: Union[str, Path],
    include_complexity: bool = True,
) -> Path:
    """Export classifications with COMPLETE symbol data to JSON."""
    from gcover.publish.symbol_utils import export_symbol_to_dict

    output_path = Path(output_path)

    export_data = []

    for classification in classifications:
        layer_data = to_serializable_dict(classification)

        for i, cls in enumerate(classification.classes):
            class_data = layer_data["classes"][i]

            if cls.full_symbol_layers:
                class_data["full_symbol_layers"] = export_symbol_to_dict(cls.full_symbol_layers)

            if include_complexity and cls.complexity_metrics:
                class_data["complexity"] = cls.complexity_metrics

            if cls.identifier:
                class_data["identifier"] = cls.identifier.to_dict()

        export_data.append(layer_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported complete classifications to {output_path}")
    return output_path


def generate_override_template(
    classifications: List[LayerClassification],
    output_path: Union[str, Path],
    complexity_threshold: int = 70,
) -> Path:
    """Generate a YAML template for custom symbol overrides."""
    import yaml

    output_path = Path(output_path)

    overrides = []

    for classification in classifications:
        for cls in classification.classes:
            if cls.complexity_metrics:
                score = cls.complexity_metrics.get("complexity_score", 0)

                if score >= complexity_threshold and cls.identifier:
                    override = {
                        "identifier": cls.identifier.to_dict(),
                        "mapserver_symbol": f"TODO_{cls.identifier.to_key().replace('::', '_')}",
                        "qgis_symbol_path": f"TODO_{cls.label.replace(' ', '_')}.qml",
                        "reason": f"Complex symbol (score: {score}/100)",
                        "notes": cls.complexity_metrics,
                    }
                    overrides.append(override)

    template_data = {
        "description": "Custom symbol overrides for complex patterns",
        "generated_threshold": complexity_threshold,
        "overrides": overrides,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated override template with {len(overrides)} candidates: {output_path}")
    return output_path


# =============================================================================
# OVERRIDE APPLICATION
# =============================================================================


def apply_overrides_to_classification(
    classification: LayerClassification, override_registry: SymbolOverrideRegistry
) -> LayerClassification:
    """Apply custom symbol overrides to a classification."""
    for cls in classification.classes:
        if cls.identifier and override_registry.has_override(cls.identifier):
            override = override_registry.get_override(cls.identifier)
            cls.custom_override = override
            logger.info(f"Custom override found for {cls.label}: {override.reason}")

    return classification


# =============================================================================
# EXPLORE LAYER STRUCTURE
# =============================================================================


def explore_layer_structure(
    lyrx_path: Union[str, Path], show_all_layers: bool = False
) -> Dict[str, Any]:
    """
    Explore the structure of a .lyrx file, showing all layers and groups.

    Args:
        lyrx_path: Path to .lyrx layer file
        show_all_layers: Include layers without unique value renderers

    Returns:
        Dictionary with layer structure information
    """
    extractor = ESRIClassificationExtractorEnhanced()
    lyrx_path = Path(lyrx_path)

    console.print(f"\n[bold blue]🔍 Exploring layer structure: {lyrx_path.name}[/bold blue]\n")

    lyrx_data = extractor._load_lyrx_json(lyrx_path)
    all_layers = extractor._find_all_layers_in_json(lyrx_data)

    structure = {
        "total_layers": len(all_layers),
        "layers": [],
        "groups": set(),
        "layer_types": {},
    }

    tree = Tree("🗺️  [bold]Layer File Structure[/bold] [dim](JSON parsing)[/dim]")

    for layer_info in all_layers:
        layer_data = layer_info["layer"]
        layer_path = layer_info["path"]
        layer_type = layer_info["type"]

        has_renderer = extractor._find_renderer_in_layer(layer_data) is not None

        structure["layers"].append(
            {
                "name": layer_data.get("name", "Unknown"),
                "path": layer_path,
                "type": layer_type,
                "has_renderer": has_renderer,
            }
        )

        if "/" in layer_path:
            group_path = "/".join(layer_path.split("/")[:-1])
            structure["groups"].add(group_path)

        if layer_type not in structure["layer_types"]:
            structure["layer_types"][layer_type] = 0
        structure["layer_types"][layer_type] += 1

        parts = layer_path.split("/")
        current_node = tree

        for i, part in enumerate(parts[:-1]):
            group_path = "/".join(parts[: i + 1])
            existing_node = None

            for child in getattr(current_node, "_children", []):
                if hasattr(child, "_label") and part in str(child._label):
                    existing_node = child
                    break

            if existing_node:
                current_node = existing_node
            else:
                current_node = current_node.add(f"📁 [yellow]{part}[/yellow]")

        layer_icon = "🎨" if has_renderer else "📄"
        renderer_info = ""

        renderer = extractor._find_renderer_in_layer(layer_data)
        if renderer:
            renderer_type = renderer.get("type", "Unknown")
            if renderer_type == "CIMUniqueValueRenderer":
                renderer_info = " [green](UniqueValue)[/green]"
            else:
                renderer_info = f" [dim]({renderer_type})[/dim]"

        layer_name = layer_data.get("name", "Unknown")
        current_node.add(
            f"{layer_icon} [cyan]{layer_name}[/cyan] [dim]({layer_type})[/dim]{renderer_info}"
        )

    console.print(tree)

    console.print(f"\n[bold yellow]📊 Statistics:[/bold yellow]")
    console.print(f"  • Total layers: {structure['total_layers']}")
    console.print(f"  • Groups: {len(structure['groups'])}")
    console.print(
        f"  • Layer types: {', '.join(f'{k}({v})' for k, v in structure['layer_types'].items())}"
    )

    unique_value_layers = [layer for layer in structure["layers"] if layer.get("has_renderer")]
    console.print(f"  • Layers with renderers: {len(unique_value_layers)}")

    return structure


# =============================================================================
# MAIN CONVENIENCE FUNCTION (PUBLIC API)
# =============================================================================


def extract_lyrx_complete(
    lyrx_path: Union[str, Path],
    override_yaml: Optional[str] = None,
    display: bool = True,
    export_json: Optional[str] = None,
    generate_override_template_path: Optional[str] = None,
    identifier_fields: Optional[Dict[str, str]] = None,
    head: Optional[int] = None,
    use_arcpy: Optional[bool] = None,   # TODO: deprecated, is ignored
) -> List[LayerClassification]:
    """
    Convenience function for COMPLETE extraction with all features.

    Args:
        lyrx_path: Path to .lyrx layer file
        override_yaml: Path to custom symbol override YAML
        display: Display results with rich
        export_json: Optional path to export complete JSON
        generate_override_template_path: Generate override template for complex symbols
        identifier_fields: Optional dictionary mapping layer names to field names
        head: Optional limit on number of rows to display

    Returns:
        List of LayerClassification objects with complete symbol data
    """
    extractor = ESRIClassificationExtractorEnhanced()

    if override_yaml:
        extractor.load_overrides(override_yaml)
        logger.info(f"Loaded {len(extractor.override_registry.overrides)} overrides")

    classifications = extractor.extract_from_lyrx(lyrx_path, identifier_fields=identifier_fields)

    for classification in classifications:
        apply_overrides_to_classification(classification, extractor.override_registry)

    if display:
        if len(classifications) > 1:
            ClassificationDisplayer.display_grouped_classifications(classifications)
        else:
            for classification in classifications:
                ClassificationDisplayer.display_classification(classification, head)

    if export_json:
        export_complete_classification_to_json(classifications, export_json)

    if generate_override_template_path:
        generate_override_template(classifications, generate_override_template_path)

    return classifications


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.option("--layers", multiple=True, help="Specific layer names (for filtering)")
@click.option("--quiet", is_flag=True, help="Suppress rich display")
@click.option("--verbose", is_flag=True, help="Be verbose")
@click.option(
    "--explore",
    is_flag=True,
    help="Explore layer structure (show all layers and groups)",
)
@click.option(
    "--export", type=click.Choice(["json", "csv"]), help="Export results to file"
)
@click.option(
    "--head",
    type=int,
    default=None,
    help="Display only the first n rows",
)
@click.option(
    "--identifiers",
    type=(str, str),
    multiple=True,
    help="Map layer names to identifier fields. Example: --identifiers Bedrock GEOL_MAPPING_UNIT",
)
def classify(input, layers, quiet, explore, export, verbose, identifiers, head):
    """Extract ESRI layer classification information from .lyrx files."""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.add("classify.log", level="DEBUG")
        logger.debug("Verbose logging enabled to 'classify.log'")

    input_path = Path(input)
    identifier_fields = dict(identifiers)

    if identifier_fields:
        click.echo(f"Identifier fields: {identifier_fields}")

    if input_path.suffix.lower() != ".lyrx":
        console.print("[red]Error: Input must be .lyrx file[/red]")
        raise click.Abort()

    if explore:
        structure = explore_layer_structure(input_path)
    else:
        results = extract_lyrx_complete(
            input_path,
            display=not quiet,
            identifier_fields=identifier_fields,
            head=head,
        )

        if quiet:
            console.print(f"Extracted {len(results)} layer classifications")

        if export:
            export_path = input_path.with_suffix(f".classifications.{export}")

            if export == "json":
                export_data = [to_serializable_dict(c) for c in results]
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(
                        export_data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        cls=ClassificationJSONEncoder,
                    )
                console.print(f"[green]Exported to {export_path}[/green]")

            elif export == "csv":
                export_classifications_to_csv(results, export_path)
                console.print(f"[green]Exported to {export_path}[/green]")


if __name__ == "__main__":
    classify()
