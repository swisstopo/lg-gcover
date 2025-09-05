# src/gcover/config/__init__.py
"""
Unified configuration system using Pydantic
Consolidates all previous config approaches into one clean system
"""

import os
from pathlib import Path

from .models import (
    AppConfig,
    GlobalConfig,
    GDBConfig,
    SDEConfig,
    QAConfig,
    S3Config,
    SchemaConfig,
)
from .loader import load_config, get_config

# Paths
CONFIG_DIR = Path.home() / '.gcover'
CONFIG_DIR.mkdir(exist_ok=True)

# SDE Instance mapping
SDE_INSTANCES = {
    'prod': 'GCOVERP',
    'integration': 'GCOVERI',
    'GCOVERP': 'GCOVERP',
    'GCOVERI': 'GCOVERI'
}

# Versions par défaut
DEFAULT_VERSIONS = {"GCOVERP": "SDE.DEFAULT", "GCOVERI": "SDE.DEFAULT"}

DEFAULT_CHUNK_SIZE = 1024

DEFAULT_NUM_WORKERS = 4

DEFAULT_INSTANCE = 'GCOVERP'

DEFAULT_CRS = 'EPSG:2056'

EXCLUDED_TABLES = {
    "GC_CONFLICT_POLYGON",
    "GC_ERRORS_LINE",
    "GC_ERRORS_ROW",
    "GC_CONFLICT_ROW",
    "GC_VERSION",
    "GC_ERRORS_MULTIPOINT",
    "GC_ERRORS_POLYGON",
    "GC_REVISIONSEBENE",
    # Also include prefixed versions
    "TOPGIS_GC.GC_CONFLICT_POLYGON",
    "TOPGIS_GC.GC_ERRORS_LINE",
    "TOPGIS_GC.GC_ERRORS_ROW",
    "TOPGIS_GC.GC_CONFLICT_ROW",
    "TOPGIS_GC.GC_VERSION",
    "TOPGIS_GC.GC_ERRORS_MULTIPOINT",
    "TOPGIS_GC.GC_ERRORS_POLYGON",
    "TOPGIS_GC.GC_REVISIONSEBENE",
}

DEFAULT_EXCLUDED_FIELDS = {
    "REVISION_MONTH",
    "CREATION_DAY",
    "REVISION_DAY",
    "CREATION_MONTH",
    "REVISION_YEAR",
    "CREATION_YEAR",
    "REVISION_DATE",
    "CREATION_DATE",
    "LAST_UPDATE",
    "CREATED_USER",
    "LAST_USER",
}

__all__ = [
    "AppConfig",
    "GlobalConfig",
    "GDBConfig",
    "SDEConfig",
    "QAConfig",
    "S3Config",
    "SchemaConfig",
    "load_config",
    "get_config",
    "EXCLUDED_TABLES",
    "DEFAULT_EXCLUDED_FIELDS",
    "SDE_INSTANCES",
    "DEFAULT_VERSIONS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_NUM_WORKERS",
]
