"""Layer geometry types shared by the style and generator modules."""

from enum import Enum


class LayerType(Enum):
    """Layer geometry types with different processing requirements."""

    POLYGON = "polygon"
    LINE = "line"
    POINT = "point"
