# src/gcover/config/gdb.py
"""
GDB module configuration
"""
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .core import BaseConfig


@dataclass
class GDBConfig(BaseConfig):
    """Configuration for GDB Asset Management"""
    base_paths: Dict[str, Path]
    s3_bucket: str
    s3_profile: Optional[str]
    db_path: Path
    temp_dir: Path
    compression_level: int = 6
    max_workers: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GDBConfig':
        """Create config from dictionary"""
        return cls(
            base_paths={k: Path(v) for k, v in data['base_paths'].items()},
            s3_bucket=data['s3']['bucket'],
            s3_profile=data['s3'].get('profile'),
            db_path=Path(data['database']['path']),
            temp_dir=Path(data.get('temp_dir', '/tmp/gcover/gdb')),
            compression_level=data.get('processing', {}).get('compression_level', 6),
            max_workers=data.get('processing', {}).get('max_workers', 4)
        )

    @classmethod
    def get_section_name(cls) -> str:
        return "gdb"

