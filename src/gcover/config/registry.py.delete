# src/gcover/config/registry.py
"""
Configuration registry - auto-register all config classes
"""
from .core import get_config_manager
from .gdb import GDBConfig
from .sde import SDEConfig
from .schema import SchemaConfig


def register_all_configs():
    """Register all configuration classes"""
    manager = get_config_manager()
    manager.register_config(GDBConfig)
    manager.register_config(SDEConfig)
    manager.register_config(SchemaConfig)


# Auto-register when module is imported
register_all_configs()