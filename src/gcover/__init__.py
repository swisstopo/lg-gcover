"""
lg-gcover - A library and CLI tool to work with geological vector data.

This package provides tools for working with geological vector data, including:
- Bridge functionality between GeoPandas and ESRI formats
- Schema management and comparison
- Quality assurance tools
- Geodatabase management utilities
"""

__version__ = "0.1.0"

# Import des modules principaux pour un accès facile
from .core.config import Config
from .utils.imports import HAS_ARCPY

# Définir ce qui est exporté avec "from gcover import *"
__all__ = [
    "__version__",
    "Config",
    "HAS_ARCPY",
]

# Metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
