# src/gcover/config/models.py
"""
Unified Pydantic configuration models
This replaces ALL other config classes
"""

from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Dict, Optional, Any
import os
from datetime import datetime

from datetime import datetime
from typing import Dict, Optional, Any, Union
import re


class FileLoggingConfig(BaseModel):
    """File logging configuration with template support"""

    enabled: bool = True
    path: str = "logs/gcover_{environment}_{date}.log"  # Keep as string template
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "gz"

    @validator("path")
    def validate_path_template(cls, v):
        """Validate that path template has valid placeholders"""
        if not isinstance(v, str):
            raise ValueError("path must be a string template")

        # Check for valid placeholder syntax
        valid_placeholders = {"{environment}", "{date}", "{datetime}", "{timestamp}"}
        found_placeholders = set(re.findall(r"\{[^}]+\}", v))

        invalid_placeholders = found_placeholders - valid_placeholders
        if invalid_placeholders:
            raise ValueError(
                f"Invalid placeholders in path: {invalid_placeholders}. "
                f"Valid placeholders: {valid_placeholders}"
            )
        return v

    def get_resolved_path(self, environment: str) -> Path:
        """
        Resolve template placeholders in the path.

        Args:
            environment: Environment name (e.g., 'development', 'production')

        Returns:
            Path with placeholders resolved
        """
        now = datetime.now()

        resolved_path = self.path.format(
            environment=environment,
            date=now.strftime("%Y%m%d"),
            datetime=now.strftime("%Y%m%d_%H%M%S"),
            timestamp=now.strftime("%Y%m%d_%H%M%S"),
        )

        return Path(resolved_path)

    @validator("rotation")
    def validate_rotation(cls, v):
        """Validate rotation format (e.g., '10 MB', '1 GB', '1 day')"""
        if not re.match(r"^\d+\s*(MB|GB|KB|day|days|hour|hours)$", v, re.IGNORECASE):
            raise ValueError(
                "rotation must be in format like '10 MB', '1 GB', or '1 day'"
            )
        return v

    @validator("retention")
    def validate_retention(cls, v):
        """Validate retention format (e.g., '30 days', '1 week')"""
        if not re.match(
            r"^\d+\s*(day|days|week|weeks|month|months)$", v, re.IGNORECASE
        ):
            raise ValueError(
                "retention must be in format like '30 days', '1 week', '6 months'"
            )
        return v


class ConsoleLoggingConfig(BaseModel):
    """Console logging configuration"""

    format: str = "simple"  # "simple" or "detailed"
    show_time: bool = True
    show_level: bool = True
    show_path: bool = False

    @validator("format")
    def validate_format(cls, v):
        """Validate console format"""
        if v not in ["simple", "detailed"]:
            raise ValueError("format must be 'simple' or 'detailed'")
        return v


class ModuleLoggingConfig(BaseModel):
    """Module-specific logging configuration"""

    modules: Dict[str, str] = {}

    @validator("modules")
    def validate_module_levels(cls, v):
        """Validate that log levels are valid"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        for module, level in v.items():
            if level.upper() not in valid_levels:
                raise ValueError(
                    f"Invalid log level '{level}' for module '{module}'. "
                    f"Valid levels: {valid_levels}"
                )
            v[module] = level.upper()  # Normalize to uppercase
        return v


class LoggingConfig(BaseModel):
    """Complete logging configuration"""

    file: FileLoggingConfig = FileLoggingConfig()
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    modules: ModuleLoggingConfig = ModuleLoggingConfig()

    def get_file_path(self, environment: str) -> Optional[Path]:
        """
        Get resolved file path if file logging is enabled.

        Args:
            environment: Environment name

        Returns:
            Resolved Path if file logging enabled, None otherwise
        """
        if self.file.enabled:
            return self.file.get_resolved_path(environment)
        return None

    def get_log_config_for_environment(self, environment: str) -> Dict[str, Any]:
        """
        Get complete logging configuration for a specific environment.

        Args:
            environment: Environment name

        Returns:
            Dict with resolved configuration for logging setup
        """
        config = {
            "file": {
                "enabled": self.file.enabled,
                "path": self.get_file_path(environment),
                "rotation": self.file.rotation,
                "retention": self.file.retention,
                "compression": self.file.compression,
            },
            "console": {
                "format": self.console.format,
                "show_time": self.console.show_time,
                "show_level": self.console.show_level,
                "show_path": self.console.show_path,
            },
            "modules": self.modules.modules,
        }
        return config


class S3Config(BaseModel):
    """S3 configuration settings"""

    bucket: str
    profile: Optional[str] = None
    region: Optional[str] = None

    @validator("bucket")
    def bucket_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("S3 bucket cannot be empty")
        return v


class GlobalConfig(BaseModel):
    """Global configuration settings"""

    log_level: str = "INFO"
    temp_dir: Path = Path("/tmp/gcover")
    max_workers: int = 4
    s3: S3Config
    default_crs: str = "EPSG:2056"
    chunk_size: int = 1000
    logging: LoggingConfig = LoggingConfig()

    @validator("temp_dir", pre=True)
    def parse_temp_dir(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @validator("log_level")
    def log_level_must_be_valid(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    def get_logging_config(self, environment: str) -> Dict[str, Any]:
        """
        Get complete logging configuration for the current environment.

        Args:
            environment: Environment name (e.g., 'development', 'production')

        Returns:
            Dict with resolved logging configuration
        """
        return self.logging.get_log_config_for_environment(environment)

    def get_log_file_path(self, environment: str) -> Optional[Path]:
        """
        Get the resolved log file path for the current environment.

        Args:
            environment: Environment name

        Returns:
            Resolved log file path if file logging is enabled
        """
        return self.logging.get_file_path(environment)


class ProcessingConfig(BaseModel):
    """Processing configuration for modules that need it"""

    compression_level: int = 6
    max_workers: int = 4

    @validator("compression_level")
    def compression_level_must_be_valid(cls, v):
        if not 1 <= v <= 9:
            raise ValueError("compression_level must be between 1 and 9")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration"""

    path: Path

    @validator("path", pre=True)
    def parse_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v


class GDBConfig(BaseModel):
    """GDB Asset Management configuration"""

    base_paths: Dict[str, Path]
    # database: DatabaseConfig
    temp_dir: Path = Path("/tmp/gcover/gdb")
    processing: ProcessingConfig = ProcessingConfig()
    db_path: Path = Path("data/dev_gdb_metadata.duckdb")

    @validator("base_paths", pre=True)
    def parse_base_paths(cls, v):
        if isinstance(v, dict):
            return {
                k: Path(path) if not isinstance(path, Path) else path
                for k, path in v.items()
            }
        return v

    @validator("temp_dir", pre=True)
    def parse_temp_dir(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    # S3 access methods (using global config)
    def get_s3_bucket(self, global_config: GlobalConfig) -> str:
        """Get S3 bucket from global configuration"""
        return global_config.s3.bucket

    def get_s3_profile(self, global_config: GlobalConfig) -> Optional[str]:
        """Get S3 profile from global configuration"""
        return global_config.s3.profile

    def get_s3_region(self, global_config: GlobalConfig) -> Optional[str]:
        """Get S3 region from global configuration"""
        return global_config.s3.region


class SDEInstanceConfig(BaseModel):
    """Single SDE instance configuration"""

    host: str
    port: int = 5151
    database: str
    version: str = "SDE.DEFAULT"
    user: Optional[str] = None

    @validator("port")
    def port_must_be_valid(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v


class SDEConfig(BaseModel):
    """SDE connection configuration"""

    instances: Dict[str, SDEInstanceConfig]
    connection_timeout: int = 30
    temp_dir: Path = Path("/tmp/gcover/sde")
    cleanup_on_exit: bool = True

    @validator("temp_dir", pre=True)
    def parse_temp_dir(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @validator("connection_timeout")
    def timeout_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("connection_timeout must be positive")
        return v


class SchemaConfig(BaseModel):
    """Schema management configuration"""

    output_dir: Path = Path("./schemas")
    template_dir: Optional[Path] = None
    default_formats: list = ["json"]
    plantuml_path: Optional[Path] = None
    max_diagram_tables: int = 50
    include_system_tables: bool = False

    @validator("output_dir", "template_dir", "plantuml_path", pre=True)
    def parse_paths(cls, v):
        return Path(v) if v and not isinstance(v, Path) else v

    @validator("max_diagram_tables")
    def max_tables_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("max_diagram_tables must be positive")
        return v


class QAConfig(BaseModel):
    """Quality Assurance configuration"""

    output_dir: Path = Path("./qa_output")
    # database: DatabaseConfig
    db_path: Path = Path("data/prod_verification_stats.duckdb")
    temp_dir: Path = Path("/tmp/gcover/qa")
    processing: ProcessingConfig = ProcessingConfig()
    default_simplify_tolerance: Optional[float] = None

    @validator("output_dir", "temp_dir", pre=True)
    def parse_paths(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @validator("default_simplify_tolerance")
    def tolerance_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("default_simplify_tolerance must be positive")
        return v

    # S3 access methods (using global config)
    def get_s3_bucket(self, global_config: GlobalConfig) -> str:
        """Get S3 bucket from global configuration"""
        return global_config.s3.bucket

    def get_s3_profile(self, global_config: GlobalConfig) -> Optional[str]:
        """Get S3 profile from global configuration"""
        return global_config.s3.profile


class AppConfig(BaseModel):
    """Main application configuration - FIXED schema field conflict"""

    global_: GlobalConfig = Field(alias="global")
    gdb: GDBConfig
    sde: Optional[SDEConfig] = None
    schema_config: Optional[SchemaConfig] = Field(
        None, alias="schema"
    )  # 🔧 FIX: Use alias
    qa: Optional[QAConfig] = None

    class Config:
        # allow_population_by_field_name = True
        validate_by_name = True

    @validator("global_", pre=True)
    def validate_global_config(cls, v):
        if not v:
            raise ValueError("global configuration is required")
        return v
