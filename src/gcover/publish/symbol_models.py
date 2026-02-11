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
import re
from dataclasses import dataclass, field
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

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
    
    def to_dict(self):
        return asdict(self)


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

class IdentifierStrategy(Enum):
    """Strategy for generating class identifiers"""

    VALUE_BASED = "value_based"  # Single field value (e.g., GMU_CODE=15801003)
    MULTI_VALUE = "multi_value"  # Multiple field values
    EXPRESSION = "expression"  # Complex expression
    INDEX = "index"  # Fallback: position in renderer
    LABEL = "label"  # Based on label (unstable but readable)


@dataclass
class ClassIdentifier:
    """
    Stable identifier for a classification class.

    The canonical_id is human-readable and based on field values when possible.
    The hash_id provides a unique fallback identifier.

    Examples:
        VALUE_BASED:  canonical_id = "gmu_15801003"
        MULTI_VALUE:  canonical_id = "kind_12501001_status_1"
        EXPRESSION:   canonical_id = "expr_a3b4c5d6"
        INDEX:        canonical_id = "idx_42"
    """

    layer_path: str  # e.g., "Surfaces/GC_SURFACES"
    # Core identification
    canonical_id: Optional[str] = None  # Human-readable stable ID
    hash_id: Optional[str] = None  # Fallback unique hash

    # Context

    strategy: Optional[IdentifierStrategy] = None

    # Source information
    field_values: Tuple[str, ...] = ()  # Actual values used
    field_names: Tuple[str, ...] = ()  # Field names
    class_index: Optional[int] = 0  # Position in renderer
    label: str = ""  # Human label (can change)

    # Symbol tracking
    symbol_hash: Optional[str] = None  # Detect symbol changes



    @classmethod
    def from_single_field(
            cls,
            layer_path: str,
            field_name: str,
            field_value: str,
            class_index: int = 0,
            label: str = "",
            symbol_dict: Optional[Dict[str, Any]] = None,
    ) -> "ClassIdentifier":
        """
        Create identifier from single field value (most common).

        Example: Bedrock classification by GMU_CODE

        Args:
            layer_path: Layer path (e.g., "Surfaces/GC_SURFACES")
            field_name: Field name (e.g., "GMU_CODE")
            field_value: Field value (e.g., "15801003")
            class_index: Position in renderer (fallback)
            label: Human-readable label
            symbol_dict: Optional symbol dictionary for hash

        Returns:
            ClassIdentifier with VALUE_BASED strategy
        """
        # Create canonical ID: fieldname_value
        canonical = f"{field_name.lower()}_{_sanitize_value(field_value)}"

        # Create hash from full context
        hash_str = f"{layer_path}::{field_name}={field_value}"
        hash_id = hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        # Symbol hash
        symbol_hash = None
        if symbol_dict:
            symbol_hash = _hash_symbol(symbol_dict)

        return cls(
            canonical_id=canonical,
            hash_id=hash_id,
            layer_path=layer_path,
            strategy=IdentifierStrategy.VALUE_BASED,
            field_values=(field_value,),
            field_names=(field_name,),
            class_index=class_index,
            label=label,
            symbol_hash=symbol_hash,
        )

    @classmethod
    def from_multiple_fields(
            cls,
            layer_path: str,
            field_names: List[str],
            field_values: List[str],
            class_index: int = 0,
            label: str = "",
            symbol_dict: Optional[Dict[str, Any]] = None,
    ) -> "ClassIdentifier":
        """
        Create identifier from multiple field values.

        Example: Point objects by KIND + STATUS

        Args:
            layer_path: Layer path
            field_names: List of field names
            field_values: List of corresponding values
            class_index: Position in renderer
            label: Human-readable label
            symbol_dict: Optional symbol dictionary

        Returns:
            ClassIdentifier with MULTI_VALUE strategy
        """
        # Create canonical ID: field1_val1_field2_val2
        parts = []
        for name, value in zip(field_names, field_values):
            parts.append(f"{name.lower()}_{_sanitize_value(value)}")
        canonical = "_".join(parts)

        # Truncate if too long
        if len(canonical) > 60:
            canonical = canonical[:60] + "_" + hashlib.md5(canonical.encode()).hexdigest()[:6]

        # Create hash
        hash_str = f"{layer_path}::{','.join(f'{n}={v}' for n, v in zip(field_names, field_values))}"
        hash_id = hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        symbol_hash = None
        if symbol_dict:
            symbol_hash = _hash_symbol(symbol_dict)

        return cls(
            canonical_id=canonical,
            hash_id=hash_id,
            layer_path=layer_path,
            strategy=IdentifierStrategy.MULTI_VALUE,
            field_values=tuple(field_values),
            field_names=tuple(field_names),
            class_index=class_index,
            label=label,
            symbol_hash=symbol_hash,
        )

    @classmethod
    def from_expression(
            cls,
            layer_path: str,
            expression: str,
            class_index: int = 0,
            label: str = "",
            symbol_dict: Optional[Dict[str, Any]] = None,
    ) -> "ClassIdentifier":
        """
        Create identifier from complex expression.

        Used for SQL WHERE clauses or complex logical expressions.

        Args:
            layer_path: Layer path
            expression: SQL or logical expression
            class_index: Position in renderer
            label: Human-readable label
            symbol_dict: Optional symbol dictionary

        Returns:
            ClassIdentifier with EXPRESSION strategy
        """
        # Try to extract a readable canonical from expression
        canonical = _extract_canonical_from_expression(expression, label)

        # Hash the expression
        hash_str = f"{layer_path}::{expression}"
        hash_id = hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        symbol_hash = None
        if symbol_dict:
            symbol_hash = _hash_symbol(symbol_dict)

        return cls(
            canonical_id=canonical,
            hash_id=hash_id,
            layer_path=layer_path,
            strategy=IdentifierStrategy.EXPRESSION,
            field_values=(expression,),
            field_names=(),
            class_index=class_index,
            label=label,
            symbol_hash=symbol_hash,
        )

    @classmethod
    def from_label(
            cls,
            layer_path: str,
            label: str,
            class_index: int = 0,
            symbol_dict: Optional[Dict[str, Any]] = None,
    ) -> "ClassIdentifier":
        """
        Create identifier from label (least stable).

        Only use when no field values or expression available.

        Args:
            layer_path: Layer path
            label: Class label
            class_index: Position in renderer
            symbol_dict: Optional symbol dictionary

        Returns:
            ClassIdentifier with LABEL strategy
        """
        canonical = _sanitize_label(label)

        hash_str = f"{layer_path}::{label}::{class_index}"
        hash_id = hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        symbol_hash = None
        if symbol_dict:
            symbol_hash = _hash_symbol(symbol_dict)

        return cls(
            canonical_id=canonical,
            hash_id=hash_id,
            layer_path=layer_path,
            strategy=IdentifierStrategy.LABEL,
            field_values=(label,),
            field_names=(),
            class_index=class_index,
            label=label,
            symbol_hash=symbol_hash,
        )

    @classmethod
    def from_index(
            cls,
            layer_path: str,
            class_index: int,
            label: str = "",
            symbol_dict: Optional[Dict[str, Any]] = None,
    ) -> "ClassIdentifier":
        """
        Create identifier from index (fallback only).

        Least stable - only use when nothing else available.

        Args:
            layer_path: Layer path
            class_index: Position in renderer
            label: Optional human-readable label
            symbol_dict: Optional symbol dictionary

        Returns:
            ClassIdentifier with INDEX strategy
        """
        canonical = f"idx_{class_index}"

        hash_str = f"{layer_path}::{class_index}::{label}"
        hash_id = hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        symbol_hash = None
        if symbol_dict:
            symbol_hash = _hash_symbol(symbol_dict)

        return cls(
            canonical_id=canonical,
            hash_id=hash_id,
            layer_path=layer_path,
            strategy=IdentifierStrategy.INDEX,
            field_values=(),
            field_names=(),
            class_index=class_index,
            label=label,
            symbol_hash=symbol_hash,
        )

    # TODO: legacy
    def to_key(self) -> str:
        """
        Generate unique key for this class.
        Format: layer_path::field1_field2::index
        """
        field_str = "_".join(self.field_values)
        return f"{self.layer_path}::{field_str}::{self.class_index}"

    def to_canonical_key(self) -> str:
        """
        Generate unique key for this class.

        Format: layer_path::canonical_id

        Returns:
            Unique key string
        """
        return f"{self.layer_path}::{self.canonical_id}"

    def to_simple_key(self) -> str:
        """
        Generate simple key (just canonical_id).

        Use when layer context is already known.

        Returns:
            Canonical ID
        """
        return self.canonical_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "canonical_id": self.canonical_id,
            "hash_id": self.hash_id,
            "layer_path": self.layer_path,
            "strategy": self.strategy.value,
            "field_values": list(self.field_values),
            "field_names": list(self.field_names),
            "class_index": self.class_index,
            "label": self.label,
            "symbol_hash": self.symbol_hash,
            "key": self.to_key(),
        }

    # TODO: legacy method
    @staticmethod
    def create(
            layer_path: str,
            field_values: List[str],
            class_index: int,
            symbol_dict: Dict[str, Any],
            label: str,
            strategy: Optional[IdentifierStrategy]=IdentifierStrategy.LABEL
    ) -> "ClassIdentifier":
        """Create identifier from classification class data."""
        # Create stable hash of symbol structure
        def _hash_symbol(symbol_dict: Dict[str, Any]) -> str:
            """Create deterministic hash of symbol structure."""
            # Use sorted JSON to ensure consistency
            json_str = json.dumps(symbol_dict, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()[:12]
        symbol_hash = _hash_symbol(symbol_dict)

        return ClassIdentifier(
            layer_path=layer_path,
            field_values=tuple(str(v) for v in field_values),
            class_index=class_index,
            symbol_hash=symbol_hash,
            label=label,
            strategy=strategy,
        )

    @classmethod
    def create_v2(
            cls,
            layer_path: str,
            field_values: List[str],
            class_index: int,
            symbol_dict: Optional[Dict[str, Any]] = None,
            label: str = "",
            identifier_field: Optional[str] = None,
            field_names: Optional[List[str]] = None,
    ) -> "ClassIdentifier":
        """
        BACKWARD-COMPATIBLE: Smart constructor that auto-detects best strategy.

        This method provides compatibility with older code that uses:
            ClassIdentifier.create(layer_path, field_values, class_index, ...)

        Auto-detects strategy based on inputs:
        - If identifier_field specified → use that single field
        - If field_values has 1 value → single field strategy
        - If field_values has multiple values → multi field strategy
        - Otherwise → fallback to index or label

        Args:
            layer_path: Layer path (e.g., "Surfaces/GC_SURFACES")
            field_values: List of field values for this class
            class_index: Position in renderer
            symbol_dict: Optional symbol dictionary for hash
            label: Human-readable label
            identifier_field: Optional specific field to use as identifier
            field_names: Optional list of field names corresponding to field_values

        Returns:
            ClassIdentifier with auto-detected strategy
        """

        # Strategy 5: Label-based (if we have a label)
        if label:
            return cls.from_label(
                layer_path=layer_path,
                label=label,
                class_index=class_index,
                symbol_dict=symbol_dict,
            )


        # Strategy 1: Specific identifier_field requested
        if identifier_field and field_names and identifier_field in field_names:
            try:
                field_idx = field_names.index(identifier_field)
                if field_idx < len(field_values):
                    field_value = field_values[field_idx]
                    return cls.from_single_field(
                        layer_path=layer_path,
                        field_name=identifier_field,
                        field_value=field_value,
                        class_index=class_index,
                        label=label,
                        symbol_dict=symbol_dict,
                    )
            except (ValueError, IndexError):
                pass  # Fall through to other strategies

        # Strategy 2: Single field value (most common)
        if field_values and len(field_values) == 1:
            if field_names and len(field_names) == 1:
                return cls.from_single_field(
                    layer_path=layer_path,
                    field_name=field_names[0],
                    field_value=field_values[0],
                    class_index=class_index,
                    label=label,
                    symbol_dict=symbol_dict,
                )
            elif field_names and len(field_names) > 0:
                # Use first field name
                return cls.from_single_field(
                    layer_path=layer_path,
                    field_name=field_names[0],
                    field_value=field_values[0],
                    class_index=class_index,
                    label=label,
                    symbol_dict=symbol_dict,
                )

        # Strategy 3: Multiple field values
        if field_values and field_names and len(field_values) == len(field_names):
            return cls.from_multiple_fields(
                layer_path=layer_path,
                field_names=field_names,
                field_values=field_values,
                class_index=class_index,
                label=label,
                symbol_dict=symbol_dict,
            )

        # Strategy 4: Multiple values but no field names
        if field_values and len(field_values) > 1:
            # Generate generic field names
            generic_field_names = [f"field_{i}" for i in range(len(field_values))]
            return cls.from_multiple_fields(
                layer_path=layer_path,
                field_names=generic_field_names,
                field_values=field_values,
                class_index=class_index,
                label=label,
                symbol_dict=symbol_dict,
            )

        # Strategy 5: Label-based (if we have a label)
        if label:
            return cls.from_label(
                layer_path=layer_path,
                label=label,
                class_index=class_index,
                symbol_dict=symbol_dict,
            )

        # Strategy 6: Index fallback (least stable)
        return cls.from_index(
            layer_path=layer_path,
            class_index=class_index,
            label=label,
            symbol_dict=symbol_dict,
        )

    def __str__(self) -> str:
        return self.to_key()

    def __repr__(self) -> str:
        return f"ClassIdentifier(canonical_id='{self.canonical_id}', strategy={self.strategy.value})"


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


# =============================================================================
# Helper Functions
# =============================================================================


def _sanitize_value(value: str) -> str:
    """Sanitize field value for use in canonical ID."""
    # Handle special values
    if value in ["<Null>", "NULL", None, ""]:
        return "null"

    # Remove non-alphanumeric characters
    sanitized = re.sub(r'[^a-z0-9_]', '_', str(value).lower())

    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


def _sanitize_label(label: str) -> str:
    """Sanitize label for use in canonical ID."""
    # Convert to lowercase
    sanitized = label.lower()

    # Replace spaces and special chars with underscore
    sanitized = re.sub(r'[^a-z0-9]+', '_', sanitized)

    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Truncate if too long
    if len(sanitized) > 40:
        sanitized = sanitized[:40]

    return sanitized


def _extract_canonical_from_expression(expression: str, label: str = "") -> str:
    """
    Try to extract a readable canonical ID from an expression.

    Falls back to label-based or hash-based ID if expression is too complex.
    """
    # Try to extract simple field=value patterns
    simple_pattern = r'([A-Z_]+)\s*=\s*["\']?([^"\'\s]+)["\']?'
    matches = re.findall(simple_pattern, expression)

    if matches and len(matches) <= 3:
        # Use field values if simple enough
        parts = [f"{field.lower()}_{_sanitize_value(value)}" for field, value in matches]
        canonical = "_".join(parts)

        if len(canonical) <= 60:
            return canonical

    # Fall back to sanitized label if available
    if label:
        return _sanitize_label(label)

    # Last resort: hash the expression
    expr_hash = hashlib.md5(expression.encode()).hexdigest()[:8]
    return f"expr_{expr_hash}"


def _hash_symbol(symbol_dict: Dict[str, Any]) -> str:
    """Create deterministic hash of symbol structure."""
    # Use sorted JSON to ensure consistency
    json_str = json.dumps(symbol_dict, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()[:12]


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    # Example 1: Single field (Bedrock)
    bedrock_id = ClassIdentifier.from_single_field(
        layer_path="Surfaces/GC_SURFACES",
        field_name="GMU_CODE",
        field_value="15801003",
        class_index=42,
        label="Granite du Mont-Blanc",

    )
    print(f"Bedrock: {bedrock_id}")
    print(f"  Key: {bedrock_id.to_key()}")
    print(f"  Strategy: {bedrock_id.strategy.value}")
    print()

    # Example 2: Multiple fields (Point objects)
    point_id = ClassIdentifier.from_multiple_fields(
        layer_path="Points/GC_POINT_OBJECTS",
        field_names=["KIND", "HSUR_STATUS"],
        field_values=["12501001", "1"],
        class_index=5,
        label="Source active",
    )
    print(f"Point: {point_id}")
    print(f"  Key: {point_id.to_key()}")
    print(f"  Strategy: {point_id.strategy.value}")
    print()

    # Example 3: Expression
    expr_id = ClassIdentifier.from_expression(
        layer_path="Fossils/GC_FOSSILS",
        expression="KIND = 14601006 AND LFOS_DIVISION = 'Triassic'",
        class_index=3,
        label="Fossiles triassiques",
    )
    print(f"Expression: {expr_id}")
    print(f"  Key: {expr_id.to_key()}")
    print(f"  Strategy: {expr_id.strategy.value}")
    print()

    # Example 4: Index fallback
    index_id = ClassIdentifier.from_index(
        layer_path="Complex/LAYER",
        class_index=99,
        label="Unknown class",
    )
    print(f"Index: {index_id}")
    print(f"  Key: {index_id.to_key()}")
    print(f"  Strategy: {index_id.strategy.value}")
