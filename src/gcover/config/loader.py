# src/gcover/config/loader.py
"""
Configuration loader supporting separate environment files and secret management
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from gcover.config.models import AppConfig

console = Console()


class ConfigManager:
    """Config manager supporting separate environment files and secrets"""

    def __init__(self):
        self._config: AppConfig | None = None
        self._secrets_loaded = False

    def load_config(
        self,
        config_path: Path | None = None,
        environment: str = "development",
        load_secrets: bool = True,
    ) -> AppConfig:
        """Load configuration with separate environment files and secret management"""
        console.print(f"[blue]Environment: {environment}[/blue]")

        # 1. Load secrets first (from .env files and environment)
        if load_secrets and not self._secrets_loaded:
            self._load_secrets(environment)

        # 2. Load base configuration
        base_config_data = self._load_base_config(config_path)

        # 3. Load environment-specific overrides
        env_config_data = self._load_environment_config(config_path, environment)

        # 4. Merge environment overrides into base config
        if env_config_data:
            self._merge_configs(base_config_data, env_config_data)

        # 5. Apply environment variable overrides
        self._apply_env_overrides(base_config_data)

        # 6. Substitute secrets in configuration
        if load_secrets:
            self._substitute_secrets(base_config_data)

        # 7. Create and validate config using Pydantic
        self._config = AppConfig(**base_config_data)
        return self._config

    def get_config(self) -> AppConfig:
        """Get loaded configuration"""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self._config

    def _load_secrets(self, environment: str) -> None:
        """Load secrets from .env files and environment variables"""
        # Try to load python-dotenv if available
        try:
            from dotenv import load_dotenv

            # Load .env files in priority order
            env_files = self._find_env_files(environment)

            for env_file in env_files:
                if env_file.exists():
                    console.log(f"🔐 Loading secrets from: [blue]{env_file}[/blue]")

                    load_dotenv(
                        env_file, override=False
                    )  # Don't override existing env vars

            # Always try to load from default .env
            default_env = Path(".env")
            if default_env.exists():
                console.log(f"🔐 Loading secrets from: {default_env}")
                load_dotenv(default_env, override=False)

        except ImportError:
            console.log(
                "[yellow]⚠️  python-dotenv not available, using system environment only[/yellow]"
            )

        self._secrets_loaded = True

    def _find_env_files(self, environment: str) -> list[Path]:
        """Find .env files in priority order"""
        return [
            Path(f".env.{environment}.local"),  # Highest priority
            Path(f".env.{environment}"),
            Path(".env.local"),
            Path("config") / f".env.{environment}",
            Path("config") / ".env",
            Path.home() / ".gcover" / f".env.{environment}",
            Path.home() / ".gcover" / ".env",
        ]

    def _substitute_secrets(self, config: dict) -> None:
        """Recursively substitute secrets in configuration"""
        self._process_config_secrets(config, "GCOVER")

    def _process_config_secrets(
        self, config: dict | list | str | Any, prefix: str
    ) -> None:
        """Process configuration object to substitute secrets"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    # Recursively process nested objects
                    section_prefix = f"{prefix}_{key.upper()}"
                    self._process_config_secrets(value, section_prefix)
                elif isinstance(value, str):
                    # Process string values for secret substitution
                    config[key] = self._substitute_secret_value(value, key, prefix)

        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, (dict, list, str)):
                    self._process_config_secrets(item, prefix)

    def _substitute_secret_value(self, value: str, key: str, prefix: str) -> str | None:
        """Substitute secret values with support for partial replacement"""

        # Si la valeur ne contient pas de variables d'environnement, retourner telle quelle
        if "${" not in value:
            return value

        # Pattern pour les variables d'environnement ${ENV_VAR}
        env_var_pattern = r"\$\{([^}]+)\}"
        matches = re.findall(env_var_pattern, value)

        if not matches:
            return value

        # Garder une copie de la valeur originale pour les remplacements
        result_value = value
        substitution_made = False
        missing_vars = []

        for env_var in matches:
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Traiter la valeur (gestion des valeurs optionnelles, etc.)
                processed_value = self._process_optional_value(env_value, key)

                if processed_value is not None:
                    # Remplacer dans la chaîne résultante
                    result_value = result_value.replace(
                        f"${{{env_var}}}", str(processed_value)
                    )
                    substitution_made = True
                    console.log(
                        f"🔐 Secret substituted: {env_var} -> {self._safe_log_value(key, processed_value)}"
                    )
                else:
                    # Si processed_value est None, garder la variable non remplacée
                    missing_vars.append(env_var)
            else:
                console.log(
                    f"[yellow]⚠️  Environment variable not set: {env_var}[/yellow]"
                )
                missing_vars.append(env_var)

        # Gestion des cas d'erreur
        if missing_vars:
            # Pour les champs optionnels, retourner None si des variables sont manquantes
            if self._is_optional_field(key):
                return None

            # Pour les champs obligatoires, vous pouvez choisir de :
            # 1. Lever une exception
            # 2. Retourner la valeur partiellement substituée
            # 3. Retourner None

            # Option 1: Lever une exception (recommandé pour les champs obligatoires)
            raise ValueError(
                f"Missing environment variables for field '{key}': {', '.join(missing_vars)}"
            )

            # Option 2: Retourner la valeur partiellement substituée (décommenter si préféré)
            # console.log(f"[yellow]⚠️  Partial substitution for '{key}': missing {missing_vars}[/yellow]")
            # return result_value if substitution_made else None

        return result_value

    def _process_optional_value(self, env_value: str, key: str) -> str | None:
        """Process environment variable value, handling special cases for optional fields"""
        # Empty string means None for optional fields
        if env_value == "":
            if self._is_optional_field(key):
                return None
            return env_value

        # Handle explicit None strings
        if env_value.lower() in ("none", "null", "nil"):
            return None

        # Handle boolean-like strings for optional fields
        if env_value.lower() in ("false", "0", "no", "off", "disabled"):
            if self._is_optional_field(key):
                return None
            return env_value

        return env_value

    def _is_optional_field(self, field_name: str) -> bool:
        """Detect if a field is optional (can be None)"""
        optional_keywords = [
            "proxy",
            "http_proxy",
            "https_proxy",
            "url",
            "endpoint",
            "webhook",
            "callback",
            "notification",
            "alert",
        ]
        field_lower = field_name.lower()
        return any(keyword in field_lower for keyword in optional_keywords)

    def _is_secret_field(self, field_name: str) -> bool:
        """Detect if a field contains sensitive information"""
        secret_keywords = [
            "secret",
            "password",
            "passwd",
            "pwd",
            "token",
            "key",
            "auth",
            "credential",
            "api_key",
            "access_token",
            "private_key",
            "totp",
        ]
        field_lower = field_name.lower()
        return any(keyword in field_lower for keyword in secret_keywords)

    def _load_base_config(self, config_path: Path | None) -> dict:
        """Load the base configuration file"""
        if config_path is None:
            config_path = self._find_base_config_file()

        console.log(f"🔧 Loading base config: {config_path}")

        with open(config_path, "r") as f:
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

    def _load_environment_config(
        self, base_config_path: Path | None, environment: str
    ) -> dict | None:
        """Load environment-specific configuration file"""

        # Try to find environment file in multiple locations
        env_paths = self._find_environment_config_paths(base_config_path, environment)

        for env_path in env_paths:
            if env_path.exists():
                console.log(f"🔧 Loading environment config: {env_path}")

                with open(env_path, "r") as f:
                    env_config = yaml.safe_load(f)

                if env_config:
                    return env_config
                else:
                    console.log(
                        f"[yellow]⚠️  Environment config is empty: {env_path}[/yellow]"
                    )

        console.log(f"⚠️  No environment config found for '{environment}'")
        return None

    def _find_base_config_file(self) -> Path:
        """Find the base configuration file"""
        search_paths = [
            Path("config/gcover_config.yaml"),
            Path("config/config.yaml"),
            Path("~/.config/gcover/config.yaml").expanduser(),
            Path("/etc/gcover/config.yaml"),
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No base configuration file found. Searched: {[str(p) for p in search_paths]}"
        )

    def _find_environment_config_paths(
        self, base_config_path: Path | None, environment: str
    ) -> list[Path]:
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
        env_keys = ["environment", "env", "_environment"]
        return any(key in config for key in env_keys)

    def _merge_configs(self, base: dict, override: dict) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key.startswith("_"):  # Skip meta keys like _environment
                continue

            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_configs(base[key], value)
            else:
                # Override or add new key
                base[key] = value
                console.log(f"🔧 Override: {key} = {self._safe_log_value(key, value)}")

    def _safe_log_value(self, key: str, value: any) -> str:
        """Safely log configuration values (hide secrets)"""
        if self._is_secret_field(key) and isinstance(value, str):
            return "***" if value else "None"
        return str(value)

    def _apply_env_overrides(self, config: dict) -> None:
        """Apply environment variable overrides"""
        # Global overrides
        self._apply_section_env_overrides(config.get("global", {}), "GCOVER_GLOBAL_")

        # Module-specific overrides
        for module in ["gdb", "sde", "schema", "qa", "totp", "proxy"]:
            if module in config:
                prefix = f"GCOVER_{module.upper()}_"
                self._apply_section_env_overrides(config[module], prefix)

    def _apply_section_env_overrides(self, config_section: dict, prefix: str):
        """Apply environment overrides to a config section"""
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                config_path = env_var[len(prefix) :].lower().split("_")
                self._set_nested_value(config_section, config_path, value)
                safe_value = "***" if self._is_secret_field(config_path[-1]) else value
                console.log(f"🔧 Env override: {env_var} = {safe_value}")

    def _set_nested_value(self, config: dict, path: list[str], value: str) -> None:
        """Set a nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    # New utility methods for secret management
    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get a secret from environment variables with multiple naming conventions"""
        possible_keys = [
            key.upper(),
            f"GCOVER_{key.upper()}",
            key.lower(),
            f"gcover_{key.lower()}",
        ]

        for env_key in possible_keys:
            value = os.getenv(env_key)
            if value:
                return value

        return default

    def validate_secrets(self, required_secrets: list[str]) -> list[str]:
        """Validate that all required secrets are available"""
        missing_secrets = []

        for secret in required_secrets:
            if not self.get_secret(secret):
                missing_secrets.append(secret)

        return missing_secrets


# Global instance
_config_manager = ConfigManager()


def load_config(
    config_path: Path | None = None,
    environment: str = "development",
    load_secrets: bool = True,
) -> AppConfig:
    """Load configuration with separate environment files"""
    return _config_manager.load_config(config_path, environment, load_secrets)


def get_config() -> AppConfig:
    """Get the loaded configuration"""
    return _config_manager.get_config()


def get_secret(key: str, default: str | None = None) -> str | None:
    """Get a secret from environment variables"""
    return _config_manager.get_secret(key, default)


def validate_secrets(required_secrets: list[str]) -> list[str]:
    """Validate that all required secrets are available"""
    return _config_manager.validate_secrets(required_secrets)


# Debug helper function
def debug_config_loading(
    config_path: Path | None = None, environment: str = "development"
) -> None:
    """Debug configuration loading process"""
    console.print(f"\n🔍 DEBUG: Loading config for environment '{environment}'")

    manager = ConfigManager()

    try:
        # Load base config
        base_config = manager._load_base_config(config_path)
        console.print(
            f"📄 Base config log_level: {base_config.get('global', {}).get('log_level', 'NOT_SET')}"
        )

        # Load environment config
        env_config = manager._load_environment_config(config_path, environment)
        if env_config:
            console.print(
                f"🌍 Environment config log_level: {env_config.get('global', {}).get('log_level', 'NOT_SET')}"
            )
        else:
            console.print("🌍 No environment config loaded")

        # Load full config
        app_config = manager.load_config(config_path, environment)
        console.print(f"✅ Final log_level: {app_config.global_.log_level}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


def debug_secrets() -> None:
    """Debug secret loading and availability"""
    console.print("\n🔍 DEBUG: Secret Management")

    # Common secret keys to check
    secret_keys = [
        "sde_username",
        "sde_password",
        "totp_secret",
        "proxy_url",
        "proxy_username",
        "proxy_password",
    ]

    for key in secret_keys:
        value = get_secret(key)
        status = "✅ Found" if value else "❌ Missing"
        console.print(f"{status}: {key}")

    # Show loaded .env files
    console.print("\n📁 Environment files checked:")
    for env_file in [".env", ".env.local", "config/.env"]:
        path = Path(env_file)
        status = "✅ Exists" if path.exists() else "❌ Missing"
        console.print(f"{status}: {env_file}")
