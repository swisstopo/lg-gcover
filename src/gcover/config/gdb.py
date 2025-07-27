# src/gcover/config/gdb.py
"""
GDB module configuration - updated to use global S3
"""
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .core import BaseConfig, GlobalConfig


@dataclass
class GDBConfig(BaseConfig):
    """Configuration for GDB Asset Management - S3 from global config"""
    base_paths: Dict[str, Path]
    db_path: Path
    temp_dir: Path
    compression_level: int = 6
    max_workers: int = 4
    # S3 settings come from global config now

    @classmethod
    def from_dict(cls, data: Dict[str, Any], global_config: GlobalConfig = None) -> 'GDBConfig':
        """Create config from dictionary - S3 comes from global config"""
        return cls(
            base_paths={k: Path(v) for k, v in data['base_paths'].items()},
            db_path=Path(data['database']['path']),
            temp_dir=Path(data.get('temp_dir', '/tmp/gcover/gdb')),
            compression_level=data.get('processing', {}).get('compression_level', 6),
            max_workers=data.get('processing', {}).get('max_workers', 4)
        )

    @classmethod
    def get_section_name(cls) -> str:
        return "gdb"

    # Convenience properties to access global S3 settings
    def get_s3_bucket(self, global_config: GlobalConfig) -> str:
        """Get S3 bucket from global config"""
        if not global_config.s3:
            raise ValueError("S3 configuration not found in global config")
        return global_config.s3.bucket

    def get_s3_profile(self, global_config: GlobalConfig) -> Optional[str]:
        """Get S3 profile from global config"""
        if not global_config.s3:
            return None
        return global_config.s3.profile
