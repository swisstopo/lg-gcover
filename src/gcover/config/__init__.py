# src/gcover/config/__init__.py
"""
Unified configuration system using Pydantic
Consolidates all previous config approaches into one clean system
"""

import os
from pathlib import Path
from functools import lru_cache

from gcover.config.loader import debug_config_loading, load_config
from gcover.config.models import (
    AppConfig,
    GDBConfig,
    GlobalConfig,
    QAConfig,
    S3Config,
    SchemaConfig,
    SDEConfig,
)


# Paths
@lru_cache(maxsize=1)
def get_config_dir() -> Path:
    """
    Get gcover config directory.

    Resolution order:
    1. GCOVER_CONFIG_DIR environment variable
    2. XDG_CONFIG_HOME/gcover (Linux standard)
    3. ~/.gcover (if home exists)
    4. /tmp/gcover (container fallback)
    """
    if env_dir := os.environ.get("GCOVER_CONFIG_DIR"):
        config_dir = Path(env_dir)
    elif xdg_config := os.environ.get("XDG_CONFIG_HOME"):
        config_dir = Path(xdg_config) / "gcover"
    else:
        try:
            home = Path.home()
            if home.exists() and os.access(home, os.W_OK):
                config_dir = home / ".gcover"
            else:
                raise OSError("Home not writable")
        except (OSError, RuntimeError):
            # RuntimeError: Could not determine home directory
            config_dir = Path("/tmp/gcover")

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


CONFIG_DIR = get_config_dir()

# SDE Instance mapping
SDE_INSTANCES = {
    "prod": "GCOVERP",
    "integration": "GCOVERI",
    "GCOVERP": "GCOVERP",
    "GCOVERI": "GCOVERI",
}

# Versions par défaut
DEFAULT_VERSIONS = {"GCOVERP": "SDE.DEFAULT", "GCOVERI": "SDE.DEFAULT"}

DEFAULT_CHUNK_SIZE = 1024

DEFAULT_NUM_WORKERS = 4

DEFAULT_INSTANCE = "GCOVERP"

DEFAULT_CRS = "EPSG:2056"

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
    "GlobalID",
    "OBJECTID",
    "Shape_Length",
    "Shape_Area",
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
    "debug_config_loading",
    "EXCLUDED_TABLES",
    "DEFAULT_EXCLUDED_FIELDS",
    "SDE_INSTANCES",
    "DEFAULT_VERSIONS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_NUM_WORKERS",
]
