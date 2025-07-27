# src/gcover/config/qa.py
"""
QA module configuration - using global S3
"""
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .core import BaseConfig, GlobalConfig


@dataclass
class QAConfig(BaseConfig):
    """Configuration for Quality Assurance processing - S3 from global"""
    output_dir: Path
    db_path: Path
    temp_dir: Path
    max_workers: int = 4
    default_simplify_tolerance: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], global_config: GlobalConfig = None) -> 'QAConfig':
        """Create config from dictionary"""
        return cls(
            output_dir=Path(data.get('output_dir', './qa_output')),
            db_path=Path(data['database']['path']),
            temp_dir=Path(data.get('temp_dir', '/tmp/gcover/qa')),
            max_workers=data.get('processing', {}).get('max_workers', 4),
            default_simplify_tolerance=data.get('default_simplify_tolerance')
        )

    @classmethod
    def get_section_name(cls) -> str:
        return "qa"

    # S3 access via global config
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