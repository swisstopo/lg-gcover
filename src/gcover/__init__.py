"""
lg-gcover - A library and CLI tool to work with geological vector data.

This package provides tools for working with geological vector data, including:
- Bridge functionality between GeoPandas and ESRI formats
- Schema management and comparison
- Quality assurance tools
- Geodatabase management utilities
"""

__version__ = "0.1.0"

from gcover._version import __version__

# Import des modules principaux pour un accès facile
# from .core.config import Config
# TODO


# Définir ce qui est exporté avec "from gcover import *"
__all__ = [
    "__version__",

]

# Metadata
__author__ = "Your Name"
__email__ = "geocover@swisstopo.ch"
__license__ = "BSD-3"
