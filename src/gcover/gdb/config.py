# src/gcover/gdb/config.py (enhanced version)
"""
Configuration management for GDB Asset Management
Enhanced with verification processing support
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class VerificationConfig:
    """Configuration for verification processing"""

    layers: List[str]
    output_formats: List[str]
    source_crs: str
    target_crs: str
    s3_prefix: str
    retention_days: int = 365

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "VerificationConfig":
        """Create verification config from dictionary"""
        return cls(
            layers=config_data.get(
                "layers", ["IssuePolygons", "IssueLines", "IssuePoints", "IssueRows"]
            ),
            output_formats=config_data.get("output_formats", ["geoparquet"]),
            source_crs=config_data.get("coordinate_systems", {}).get(
                "source_crs", "EPSG:2056"
            ),
            target_crs=config_data.get("coordinate_systems", {}).get(
                "target_crs", "EPSG:4326"
            ),
            s3_prefix=config_data.get("s3_prefix", "verifications/"),
            retention_days=config_data.get("retention_days", 365),
        )


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
    verification: Optional[VerificationConfig] = None

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "GDBConfig":
        """Create config from dictionary"""
        # Parse verification config if present
        verification_config = None
        if "verification" in config_data:
            verification_config = VerificationConfig.from_dict(
                config_data["verification"]
            )

        return cls(
            base_paths={k: Path(v) for k, v in config_data["base_paths"].items()},
            s3_bucket=config_data["s3"]["bucket"],
            s3_profile=config_data["s3"].get("profile"),
            db_path=Path(config_data["database"]["path"]),
            temp_dir=Path(config_data["temp_dir"]),
            compression_level=config_data.get("processing", {}).get(
                "compression_level", 6
            ),
            max_workers=config_data.get("processing", {}).get("max_workers", 4),
            log_level=config_data.get("logging", {}).get("level", "INFO"),
            verification=verification_config,
        )

    def get_verification_db_path(self) -> Path:
        """Get the path for verification statistics database"""
        return self.db_path.parent / "verification_stats.duckdb"


def load_config(
    config_path: Optional[Path] = None, environment: str = "development"
) -> GDBConfig:
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
            Path("/etc/gcover/gdb_config.yaml"),
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                "No configuration file found in standard locations:\n"
                + "\n".join(f"  - {p}" for p in search_paths)
                + "\n\nCreate a config file or set these environment variables:\n"
                + "  GDB_S3_BUCKET, GDB_S3_PROFILE, GDB_DB_PATH, GDB_TEMP_DIR"
            )

    with open(config_path, "r") as f:
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
        "GDB_S3_BUCKET": ["s3", "bucket"],
        "GDB_S3_PROFILE": ["s3", "profile"],
        "GDB_DB_PATH": ["database", "path"],
        "GDB_TEMP_DIR": ["temp_dir"],
        "GDB_LOG_LEVEL": ["logging", "level"],
        # Verification-specific environment variables
        "GDB_VERIFICATION_S3_PREFIX": ["verification", "s3_prefix"],
        "GDB_VERIFICATION_SOURCE_CRS": [
            "verification",
            "coordinate_systems",
            "source_crs",
        ],
        "GDB_VERIFICATION_TARGET_CRS": [
            "verification",
            "coordinate_systems",
            "target_crs",
        ],
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


def create_default_config(config_path: Path) -> None:
    """Create a default configuration file"""
    default_config = {
        "base_paths": {
            "gcover": "/media/marco/SANDISK/GCOVER",
            "verifications": "/media/marco/SANDISK/Verifications",
            "increments": "/media/marco/SANDISK/Increment/GCOVERP",
        },
        "s3": {"bucket": "your-gcover-bucket", "profile": None},
        "database": {"path": str(Path.home() / ".config/gcover/assets.duckdb")},
        "temp_dir": "/tmp/gcover",
        "processing": {"compression_level": 6, "max_workers": 4},
        "logging": {"level": "INFO"},
        "verification": {
            "layers": ["IssuePolygons", "IssueLines", "IssuePoints", "IssueRows"],
            "output_formats": ["geoparquet"],
            "coordinate_systems": {
                "source_crs": "EPSG:2056",
                "target_crs": "EPSG:4326",
            },
            "s3_prefix": "verifications/",
            "retention_days": 365,
        },
    }

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config file
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Created default configuration file: {config_path}")
    print("Please edit the configuration file to match your environment.")


if __name__ == "__main__":
    # CLI for creating default config
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create-config":
        config_path = Path("config/gdb_config.yaml")
        if len(sys.argv) > 2:
            config_path = Path(sys.argv[2])
        create_default_config(config_path)
    else:
        # Test loading configuration
        try:
            config = load_config()
            print("Configuration loaded successfully!")
            print(f"S3 Bucket: {config.s3_bucket}")
            print(f"Database: {config.db_path}")
            if config.verification:
                print(f"Verification layers: {', '.join(config.verification.layers)}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nCreate a default config with:")
            print("python -m gcover.gdb.config create-config")
