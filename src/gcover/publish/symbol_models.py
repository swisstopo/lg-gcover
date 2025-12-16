"""
Enhanced symbol data models for comprehensive ESRI CIM extraction.

BACKWARDS COMPATIBLE with existing imports:
- FontSymbol
- CharacterMarkerInfo
- SymbolLayersInfo

NEW: Complete symbol layer types including CIMHatchFill
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# EXISTING MODELS (maintained for compatibility)
# =============================================================================


@dataclass(frozen=True)
class FontSymbol:
    """
    Immutable font symbol specification for registry tracking.
    Used to deduplicate font symbols across different layers and classes.
    """

    font_family: str
    char_index: int

    def __hash__(self):
        return hash((self.font_family, self.char_index))


@dataclass
class CharacterMarkerInfo:
    """
    Information about a character marker for pattern fill.
    Used in polygon symbols with repeating font character patterns.
    """

    character_index: int
    font_family: str
    size: float
    color: Tuple[int, int, int, int]  # (r, g, b, a)
    offset_x: float = 0.0
    offset_y: float = 0.0
    step_x: float = 10.0
    step_y: float = 10.0
    rotation: float = 0.0  # NEW: character rotation


# =============================================================================
# NEW: Extended Symbol Layer Types
# =============================================================================


class FillType(Enum):
    """Types of fill patterns."""

    SOLID = "solid"
    HATCH = "hatch"
    CHARACTER = "character"
    PICTURE = "picture"
    GRADIENT = "gradient"


@dataclass
class HatchFillInfo:
    """
    Information about a hatch fill pattern (CIMHatchFill).

    Represents diagonal lines at specified rotation/separation.
    """

    rotation: float  # Degrees
    separation: float  # Spacing between lines
    line_symbol: Optional[Dict[str, Any]] = None  # Line properties
    offset_x: float = 0.0
    offset_y: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "hatch",
            "rotation": self.rotation,
            "separation": self.separation,
            "line_symbol": self.line_symbol,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
        }


@dataclass
class PictureFillInfo:
    """Information about a picture/image fill pattern."""

    url: Optional[str] = None
    data: Optional[str] = None  # Base64 encoded image
    width: float = 10.0
    height: float = 10.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "picture",
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
        }


@dataclass
class GradientFillInfo:
    """Information about gradient fill."""

    gradient_type: str  # 'linear', 'radial', 'rectangular'
    colors: List[Tuple[int, int, int, int]]  # List of RGBA colors
    positions: List[float]  # Position of each color stop (0.0-1.0)
    angle: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gradient",
            "gradient_type": self.gradient_type,
            "colors": self.colors,
            "positions": self.positions,
            "angle": self.angle,
        }


@dataclass
class SymbolLayersInfo:
    """
    ENHANCED: Extracted symbol layer information from ESRI CIMPolygonSymbol.

    Now captures ALL layer types in proper order:
    - outline: Stroke information
    - fills: ALL fill layers (solid, hatch, character, picture, gradient)
    - character_markers: Pattern fills using font characters (DEPRECATED - use fills)
    - layer_order: Rendering order (bottom to top)
    """

    outline: Optional[Dict[str, Any]] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)
    character_markers: List[CharacterMarkerInfo] = field(default_factory=list)  # Legacy
    layer_order: List[str] = field(default_factory=list)  # NEW: render order

    def add_solid_fill(self, color: Tuple[int, int, int, int]):
        """Add a solid fill layer."""
        self.fills.append({"type": "solid", "color": color})
        self.layer_order.append(f"fill_{len(self.fills) - 1}")

    def add_hatch_fill(self, hatch: HatchFillInfo):
        """Add a hatch fill layer."""
        self.fills.append(hatch.to_dict())
        self.layer_order.append(f"fill_{len(self.fills) - 1}")

    def add_character_marker(self, marker: CharacterMarkerInfo):
        """Add a character marker fill layer."""
        # Add to both new and legacy structures
        self.character_markers.append(marker)
        self.fills.append({"type": "character", "marker_info": marker})
        self.layer_order.append(f"fill_{len(self.fills) - 1}")

    def add_picture_fill(self, picture: PictureFillInfo):
        """Add a picture fill layer."""
        self.fills.append(picture.to_dict())
        self.layer_order.append(f"fill_{len(self.fills) - 1}")

    def add_gradient_fill(self, gradient: GradientFillInfo):
        """Add a gradient fill layer."""
        self.fills.append(gradient.to_dict())
        self.layer_order.append(f"fill_{len(self.fills) - 1}")

    def set_outline(self, outline_dict: Dict[str, Any]):
        """Set the outline stroke."""
        self.outline = outline_dict
        if "outline" not in self.layer_order:
            self.layer_order.append("outline")


# =============================================================================
# NEW: Stable Class Identification System
# =============================================================================


@dataclass
class ClassIdentifier:
    """
    Stable identifier for a classification class.

    Uses field values as primary key (stable across label changes).
    Includes symbol hash to detect style changes.
    """

    layer_path: str  # e.g., "Surfaces/GC_SURFACES"
    field_values: Tuple[str, ...]  # e.g., ("15801003",) or ("15101040", "15001014")
    class_index: int  # Position in renderer (fallback for ordering)
    symbol_hash: str  # Hash of symbol structure (detects style changes)
    label: str  # Human-readable label (can change)

    @staticmethod
    def create(
        layer_path: str,
        field_values: List[str],
        class_index: int,
        symbol_dict: Dict[str, Any],
        label: str,
    ) -> "ClassIdentifier":
        """Create identifier from classification class data."""
        # Create stable hash of symbol structure
        symbol_hash = ClassIdentifier._hash_symbol(symbol_dict)

        return ClassIdentifier(
            layer_path=layer_path,
            field_values=tuple(str(v) for v in field_values),
            class_index=class_index,
            symbol_hash=symbol_hash,
            label=label,
        )

    @staticmethod
    def _hash_symbol(symbol_dict: Dict[str, Any]) -> str:
        """Create deterministic hash of symbol structure."""
        # Use sorted JSON to ensure consistency
        json_str = json.dumps(symbol_dict, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()[:12]

    def to_key(self) -> str:
        """
        Generate unique key for this class.
        Format: layer_path::field1_field2::index
        """
        field_str = "_".join(self.field_values)
        return f"{self.layer_path}::{field_str}::{self.class_index}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_path": self.layer_path,
            "field_values": list(self.field_values),
            "class_index": self.class_index,
            "symbol_hash": self.symbol_hash,
            "label": self.label,
            "key": self.to_key(),
        }


# =============================================================================
# NEW: Custom Symbol Override System
# =============================================================================


@dataclass
class SymbolOverride:
    """
    Custom symbol override for edge cases.

    Allows manual specification of MapServer/QGIS symbols
    for complex cases that can't be auto-generated.
    """

    identifier: ClassIdentifier
    mapserver_symbol: Optional[str] = None  # Symbol name in MapServer
    qgis_symbol_path: Optional[str] = None  # Path to .qml file
    reason: str = ""  # Why override is needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier.to_dict(),
            "mapserver_symbol": self.mapserver_symbol,
            "qgis_symbol_path": self.qgis_symbol_path,
            "reason": self.reason,
        }


class SymbolOverrideRegistry:
    """
    Registry for managing custom symbol overrides.

    Loads from YAML/JSON and provides lookup by class identifier.
    """

    def __init__(self):
        self.overrides: Dict[str, SymbolOverride] = {}

    def add_override(self, override: SymbolOverride):
        """Add an override to the registry."""
        key = override.identifier.to_key()
        self.overrides[key] = override

    def get_override(self, identifier: ClassIdentifier) -> Optional[SymbolOverride]:
        """Get override for a class identifier."""
        return self.overrides.get(identifier.to_key())

    def has_override(self, identifier: ClassIdentifier) -> bool:
        """Check if an override exists."""
        return identifier.to_key() in self.overrides

    @staticmethod
    def from_yaml(yaml_path: str) -> "SymbolOverrideRegistry":
        """Load overrides from YAML file."""
        from pathlib import Path

        import yaml

        registry = SymbolOverrideRegistry()
        path = Path(yaml_path)

        if not path.exists():
            return registry

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        for override_data in data.get("overrides", []):
            # Parse identifier
            id_data = override_data.get("identifier", {})
            identifier = ClassIdentifier(
                layer_path=id_data.get("layer_path", ""),
                field_values=tuple(id_data.get("field_values", [])),
                class_index=id_data.get("class_index", 0),
                symbol_hash=id_data.get("symbol_hash", ""),
                label=id_data.get("label", ""),
            )

            override = SymbolOverride(
                identifier=identifier,
                mapserver_symbol=override_data.get("mapserver_symbol"),
                qgis_symbol_path=override_data.get("qgis_symbol_path"),
                reason=override_data.get("reason", ""),
            )

            registry.add_override(override)

        return registry

    def to_yaml(self, yaml_path: str):
        """Save overrides to YAML file."""
        from pathlib import Path

        import yaml

        data = {"overrides": [o.to_dict() for o in self.overrides.values()]}

        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
