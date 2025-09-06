# src/gcover/config/loader.py
"""
Configuration loader supporting separate environment files
"""
import os
import yaml
from pathlib import Path
from typing import Optional
from .models import AppConfig


class ConfigManager:
    """Config manager supporting separate environment files"""

    def __init__(self):
        self._config: Optional[AppConfig] = None

    def load_config(
        self,
        config_path: Optional[Path] = None,
        environment: str = "development"
    ) -> AppConfig:
        """Load configuration with separate environment files"""
        print(f"Environment: {environment}")

        # 1. Load base configuration
        base_config_data = self._load_base_config(config_path)

        # 2. Load environment-specific overrides
        env_config_data = self._load_environment_config(config_path, environment)

        # 3. Merge environment overrides into base config
        if env_config_data:
            self._merge_configs(base_config_data, env_config_data)

        # 4. Apply environment variable overrides
        self._apply_env_overrides(base_config_data)

        # 5. Create and validate config using Pydantic
        self._config = AppConfig(**base_config_data)
        return self._config

    def get_config(self) -> AppConfig:
        """Get loaded configuration"""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self._config

    def _load_base_config(self, config_path: Optional[Path]) -> dict:
        """Load the base configuration file"""
        if config_path is None:
            config_path = self._find_base_config_file()

        print(f"🔧 Loading base config: {config_path}")

        with open(config_path, 'r') as f:
            # Handle both single YAML and multi-document YAML
            configs = list(yaml.safe_load_all(f))

        # Use first document as base, ignore environment sections
        base_config = configs[0] if configs else {}

        # If there are multiple documents in the same file, filter out environment sections
        if len(configs) > 1:
            for config in configs[1:]:
                if config and not self._is_environment_section(config):
                    self._merge_configs(base_config, config)

        return base_config

    def _load_environment_config(self, base_config_path: Optional[Path], environment: str) -> Optional[dict]:
        """Load environment-specific configuration file"""

        # Try to find environment file in multiple locations
        env_paths = self._find_environment_config_paths(base_config_path, environment)

        for env_path in env_paths:
            if env_path.exists():
                print(f"🔧 Loading environment config: {env_path}")

                with open(env_path, 'r') as f:
                    env_config = yaml.safe_load(f)

                if env_config:
                    return env_config
                else:
                    print(f"⚠️  Environment config is empty: {env_path}")

        print(f"⚠️  No environment config found for '{environment}'")
        return None

    def _find_base_config_file(self) -> Path:
        """Find the base configuration file"""
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
            f"No base configuration file found. Searched: {[str(p) for p in search_paths]}"
        )

    def _find_environment_config_paths(self, base_config_path: Optional[Path], environment: str) -> list[Path]:
        """Find possible environment configuration file paths"""

        if base_config_path:
            base_dir = base_config_path.parent
        else:
            base_dir = Path("config")

        # Multiple possible locations for environment configs
        env_paths = [
            # config/environments/development.yaml
            base_dir / "environments" / f"{environment}.yaml",
            base_dir / "environments" / f"{environment}.yml",

            # config/development.yaml
            base_dir / f"{environment}.yaml",
            base_dir / f"{environment}.yml",

            # config/env/development.yaml
            base_dir / "env" / f"{environment}.yaml",
            base_dir / "env" / f"{environment}.yml",
        ]

        return env_paths

    def _is_environment_section(self, config: dict) -> bool:
        """Check if a config section is an environment override"""
        env_keys = ['environment', 'env', '_environment']
        return any(key in config for key in env_keys)

    def _merge_configs(self, base: dict, override: dict) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key.startswith('_'):  # Skip meta keys like _environment
                continue

            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_configs(base[key], value)
            else:
                # Override or add new key
                base[key] = value
                print(f"🔧 Override: {key} = {value}")

    def _apply_env_overrides(self, config: dict) -> None:
        """Apply environment variable overrides"""
        # Global overrides
        self._apply_section_env_overrides(config.get('global', {}), 'GCOVER_GLOBAL_')

        # Module-specific overrides
        for module in ['gdb', 'sde', 'schema', 'qa']:
            if module in config:
                prefix = f'GCOVER_{module.upper()}_'
                self._apply_section_env_overrides(config[module], prefix)

    def _apply_section_env_overrides(self, config_section: dict, prefix: str):
        """Apply environment overrides to a config section"""
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                config_path = env_var[len(prefix):].lower().split('_')
                self._set_nested_value(config_section, config_path, value)
                print(f"🔧 Env override: {env_var} = {value}")

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
    """Load configuration with separate environment files"""
    return _config_manager.load_config(config_path, environment)


def get_config() -> AppConfig:
    """Get the loaded configuration"""
    return _config_manager.get_config()


# Debug helper function
def debug_config_loading(
    config_path: Optional[Path] = None,
    environment: str = "development"
) -> None:
    """Debug configuration loading process"""
    print(f"\n🔍 DEBUG: Loading config for environment '{environment}'")

    manager = ConfigManager()

    try:
        # Load base config
        base_config = manager._load_base_config(config_path)
        print(f"📄 Base config log_level: {base_config.get('global', {}).get('log_level', 'NOT_SET')}")

        # Load environment config
        env_config = manager._load_environment_config(config_path, environment)
        if env_config:
            print(f"🌍 Environment config log_level: {env_config.get('global', {}).get('log_level', 'NOT_SET')}")
        else:
            print("🌍 No environment config loaded")

        # Load full config
        app_config = manager.load_config(config_path, environment)
        print(f"✅ Final log_level: {app_config.global_.log_level}")

    except Exception as e:
        print(f"❌ Error: {e}")