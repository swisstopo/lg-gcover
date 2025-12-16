"""
lg-gcover - A library and CLI tool to work with Geocover 2D vector data and model.

This package provides tools for working with geological vector data, including:
- ESRI File Geodatabase management utilities
- ESRI Schema Export management and comparison
- File GDB QA tests results management and tools
- Bridge functionality between GeoPandas and ESRI formats
- Publication tools (Mapserver, QGis, ArcGis)
"""

__docformat__ = 'numpy'

from gcover._version import __version__

from gcover import cli, config, core, gdb, publish, qa, schema, utils


# Définir ce qui est exporté avec "from gcover import *"
__all__ = [
    "__version__",
    "core",
    "config",
    "cli",
    "gdb",
    "publish",
    "qa",
    "schema",
    "utils",
]

# Metadata
__author__ = "Your Name"
__email__ = "geocover@swisstopo.ch"
__license__ = "BSD-3"
