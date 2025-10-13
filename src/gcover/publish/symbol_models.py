"""
Symbol data models for MapServer and QGIS style generation.

Contains dataclasses and enums for representing symbol information
extracted from ESRI classification files.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


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

    Used in polygon symbols with repeating font character patterns
    (e.g., checkerboard fills, hatching).
    """

    character_index: int
    font_family: str
    size: float
    color: Tuple[int, int, int, int]  # (r, g, b, a)
    offset_x: float = 0.0
    offset_y: float = 0.0
    step_x: float = 10.0  # Spacing in X direction
    step_y: float = 10.0  # Spacing in Y direction


@dataclass
class SymbolLayersInfo:
    """
    Extracted symbol layer information from ESRI CIMPolygonSymbol.

    Contains all the components needed to reconstruct a complex polygon symbol:
    - outline: Stroke information
    - fills: Solid fill layers
    - character_markers: Pattern fill layers using font characters
    """

    outline: Optional[dict] = None
    fills: list = None
    character_markers: list = None

    def __post_init__(self):
        if self.fills is None:
            self.fills = []
        if self.character_markers is None:
            self.character_markers = []
