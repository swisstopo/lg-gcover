# src/gcover/config/models.py
"""
Unified Pydantic configuration models
This replaces ALL other config classes
"""

from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Dict, Optional, Any
import os


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

    @validator("temp_dir", pre=True)
    def parse_temp_dir(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @validator("log_level")
    def log_level_must_be_valid(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


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
