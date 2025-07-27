# src/gcover/config/loader.py
"""
Configuration loader using the unified Pydantic models
"""
import os
import yaml
from pathlib import Path
from typing import Optional
from .models import AppConfig


class ConfigManager:
    """Simplified config manager using Pydantic models"""

    def __init__(self):
        self._config: Optional[AppConfig] = None

    def load_config(
            self,
            config_path: Optional[Path] = None,
            environment: str = "development"
    ) -> AppConfig:
        """Load configuration from YAML file"""
        config_data = self._load_yaml_config(config_path, environment)

        # Apply environment variable overrides
        self._apply_env_overrides(config_data)

        # Create and validate config using Pydantic
        self._config = AppConfig(**config_data)
        return self._config

    def get_config(self) -> AppConfig:
        """Get loaded configuration"""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self._config

    def _load_yaml_config(
            self,
            config_path: Optional[Path],
            environment: str
    ) -> dict:
        """Load and merge YAML configuration"""
        if config_path is None:
            config_path = self._find_config_file()

        with open(config_path, 'r') as f:
            configs = list(yaml.safe_load_all(f))

        # Base configuration
        config_data = configs[0] if configs else {}

        # Apply environment-specific overrides
        for config in configs[1:]:
            if config and self._is_environment_config(config, environment):
                self._merge_configs(config_data, config)

        return config_data

    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations"""
        search_paths = [
            Path("config/gcover_config.yaml"),
            Path("config/config.yaml"),
            Path("~/.config/gcover/config.yaml").expanduser(),
            Path("/etc/gcover/config.yaml")
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No configuration file found. Searched: {[str(p) for p in search_paths]}"
        )

    def _is_environment_config(self, config: dict, environment: str) -> bool:
        """Check if config section is for the target environment"""
        env_keys = ['environment', 'env', '_environment']
        for key in env_keys:
            if key in config and config[key] == environment:
                return True
        return False

    def _merge_configs(self, base: dict, override: dict) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key.startswith('_'):  # Skip meta keys
                continue
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self, config: dict) -> None:
        """Apply environment variable overrides"""
        # Global overrides
        self._apply_section_env_overrides(config, 'global', 'GCOVER_GLOBAL_')

        # Module-specific overrides
        for module in ['gdb', 'sde', 'schema', 'qa']:
            if module in config:
                prefix = f'GCOVER_{module.upper()}_'
                self._apply_section_env_overrides(config[module], module, prefix)

    def _apply_section_env_overrides(self, config_section: dict, section_name: str, prefix: str):
        """Apply environment overrides to a config section"""
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                config_path = env_var[len(prefix):].lower().split('_')
                self._set_nested_value(config_section, config_path, value)

    def _set_nested_value(self, config: dict, path: list, value: str) -> None:
        """Set a nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value


# Global instance
_config_manager = ConfigManager()


def load_config(
        config_path: Optional[Path] = None,
        environment: str = "development"
) -> AppConfig:
    """Load configuration and return config object"""
    return _config_manager.load_config(config_path, environment)


def get_config() -> AppConfig:
    """Get the loaded configuration"""
    return _config_manager.get_config()