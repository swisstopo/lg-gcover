# src/gcover/config/core.py
"""
Core configuration management system with global S3 settings
"""
import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, Union
from dataclasses import dataclass

T = TypeVar('T', bound='BaseConfig')

class BaseConfig(ABC):
    """Base class for all module configurations"""

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any], global_config: 'GlobalConfig' = None) -> T:
        """Create config instance from dictionary with optional global config"""
        pass

    @classmethod
    @abstractmethod
    def get_section_name(cls) -> str:
        """Return the configuration section name"""
        pass

    @classmethod
    def get_env_prefix(cls) -> str:
        """Return environment variable prefix for this config"""
        return f"GCOVER_{cls.get_section_name().upper()}_"


@dataclass
class S3Config:
    """S3 configuration settings"""
    bucket: str
    profile: Optional[str] = None
    region: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'S3Config':
        return cls(
            bucket=data['bucket'],
            profile=data.get('profile'),
            region=data.get('region')
        )


@dataclass
class GlobalConfig:
    """Global configuration settings with S3"""
    log_level: str = "INFO"
    temp_dir: Path = Path("/tmp/gcover")
    max_workers: int = 4
    s3: S3Config = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlobalConfig':
        s3_config = None
        if 's3' in data:
            s3_config = S3Config.from_dict(data['s3'])

        return cls(
            log_level=data.get('log_level', 'INFO'),
            temp_dir=Path(data.get('temp_dir', '/tmp/gcover')),
            max_workers=data.get('max_workers', 4),
            s3=s3_config
        )


class ConfigManager:
    """Central configuration manager with global S3 support"""

    def __init__(self):
        self._config_classes: Dict[str, Type[BaseConfig]] = {}
        self._loaded_configs: Dict[str, BaseConfig] = {}
        self._global_config: Optional[GlobalConfig] = None

    def register_config(self, config_class: Type[BaseConfig]) -> None:
        """Register a configuration class"""
        section_name = config_class.get_section_name()
        self._config_classes[section_name] = config_class

    def load_config(
        self,
        config_path: Optional[Path] = None,
        environment: str = "development"
    ) -> None:
        """Load configuration from YAML file"""
        config_data = self._load_yaml_config(config_path, environment)

        # Load global config first (includes S3)
        global_data = config_data.get('global', {})
        self._apply_env_overrides(global_data, 'GCOVER_GLOBAL_')
        self._global_config = GlobalConfig.from_dict(global_data)

        # Load module configs (pass global config for S3 access)
        for section_name, config_class in self._config_classes.items():
            if section_name in config_data:
                module_data = config_data[section_name].copy()
                self._apply_env_overrides(module_data, config_class.get_env_prefix())
                # Pass global config to module configs
                self._loaded_configs[section_name] = config_class.from_dict(
                    module_data, self._global_config
                )

    def get_config(self, section_name: str) -> BaseConfig:
        """Get configuration for a specific section"""
        if section_name not in self._loaded_configs:
            raise ValueError(f"Configuration for '{section_name}' not loaded")
        return self._loaded_configs[section_name]

    def get_global_config(self) -> GlobalConfig:
        """Get global configuration"""
        if self._global_config is None:
            raise ValueError("Global configuration not loaded")
        return self._global_config

    def _load_yaml_config(
        self,
        config_path: Optional[Path],
        environment: str
    ) -> Dict[str, Any]:
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

    def _is_environment_config(self, config: Dict[str, Any], environment: str) -> bool:
        """Check if config section is for the target environment"""
        env_keys = ['environment', 'env', '_environment']
        for key in env_keys:
            if key in config and config[key] == environment:
                return True
        return environment.lower() in str(config).lower()

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key.startswith('_'):  # Skip meta keys
                continue
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self, config: Dict[str, Any], prefix: str) -> None:
        """Apply environment variable overrides"""
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                config_path = env_var[len(prefix):].lower().split('_')
                self._set_nested_value(config, config_path, value)

    def _set_nested_value(
        self,
        config: Dict[str, Any],
        path: list,
        value: str
    ) -> None:
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
) -> ConfigManager:
    """Load configuration and return manager instance"""
    _config_manager.load_config(config_path, environment)
    return _config_manager


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager"""
    return _config_manager
