# src/gcover/gdb/config.py
"""
Configuration management for GDB Asset Management
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GDBConfig:
    """Configuration for GDB Asset Management"""
    base_paths: Dict[str, Path]
    s3_bucket: str
    s3_profile: Optional[str]
    db_path: Path
    temp_dir: Path
    compression_level: int = 6
    max_workers: int = 4
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'GDBConfig':
        """Create config from dictionary"""
        return cls(
            base_paths={k: Path(v) for k, v in config_data['base_paths'].items()},
            s3_bucket=config_data['s3']['bucket'],
            s3_profile=config_data['s3'].get('profile'),
            db_path=Path(config_data['database']['path']),
            temp_dir=Path(config_data['temp_dir']),
            compression_level=config_data.get('processing', {}).get('compression_level', 6),
            max_workers=config_data.get('processing', {}).get('max_workers', 4),
            log_level=config_data.get('logging', {}).get('level', 'INFO')
        )


def load_config(config_path: Optional[Path] = None, environment: str = "development") -> GDBConfig:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file (default: search for config in standard locations)
        environment: Environment name (development, production)
    """
    if config_path is None:
        # Search for config in standard locations
        search_paths = [
            Path("config/gdb_config.yaml"),
            Path("~/.config/gcover/gdb_config.yaml").expanduser(),
            Path("/etc/gcover/gdb_config.yaml")
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError("No configuration file found in standard locations")

    with open(config_path, 'r') as f:
        # Load all YAML documents (base config + environment overrides)
        configs = list(yaml.safe_load_all(f))

    # Base configuration
    config_data = configs[0]

    # Apply environment-specific overrides
    for config in configs[1:]:
        if config and environment in str(config).lower():
            merge_configs(config_data, config)

    # Override with environment variables
    apply_env_overrides(config_data)

    return GDBConfig.from_dict(config_data)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge configuration dictionaries"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_configs(base[key], value)
        else:
            base[key] = value


def apply_env_overrides(config: Dict[str, Any]) -> None:
    """Apply environment variable overrides"""
    env_mappings = {
        'GDB_S3_BUCKET': ['s3', 'bucket'],
        'GDB_S3_PROFILE': ['s3', 'profile'],
        'GDB_DB_PATH': ['database', 'path'],
        'GDB_TEMP_DIR': ['temp_dir'],
        'GDB_LOG_LEVEL': ['logging', 'level']
    }

    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            # Navigate to the nested config location
            current = config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[config_path[-1]] = value
