# src/gcover/config/__init__.py
"""
Unified configuration system using Pydantic
Consolidates all previous config approaches into one clean system
"""
from .models import AppConfig, GlobalConfig, GDBConfig, SDEConfig, QAConfig, S3Config, SchemaConfig
from .loader import load_config, get_config

EXCLUDED_TABLES =  {
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
        "REVISION_MONTH", "CREATION_DAY", "REVISION_DAY", "CREATION_MONTH",
        "REVISION_YEAR", "CREATION_YEAR", "REVISION_DATE", "CREATION_DATE",
        "LAST_UPDATE", "CREATED_USER", "LAST_USER"
    }

__all__ = [
    'AppConfig', 'GlobalConfig', 'GDBConfig', 'SDEConfig', 'QAConfig', 'S3Config', 'SchemaConfig',
    'load_config', 'get_config', 'EXCLUDED_TABLES', 'DEFAULT_EXCLUDED_FIELDS'
]



