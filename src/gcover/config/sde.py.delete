# src/gcover/config/sde.py
"""
SDE module configuration
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .core import BaseConfig


@dataclass
class SDEInstance:
    """SDE Instance configuration"""
    name: str
    host: str
    port: int
    database: str
    version: str = "SDE.DEFAULT"
    user: Optional[str] = None


@dataclass
class SDEConfig(BaseConfig):
    """Configuration for SDE connections"""
    instances: Dict[str, SDEInstance]
    connection_timeout: int = 30
    temp_dir: Path = Path("/tmp/gcover/sde")
    cleanup_on_exit: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SDEConfig':
        """Create config from dictionary"""
        instances = {}
        for name, instance_data in data.get('instances', {}).items():
            instances[name] = SDEInstance(
                name=name,
                host=instance_data['host'],
                port=instance_data['port'],
                database=instance_data['database'],
                version=instance_data.get('version', 'SDE.DEFAULT'),
                user=instance_data.get('user')
            )

        return cls(
            instances=instances,
            connection_timeout=data.get('connection_timeout', 30),
            temp_dir=Path(data.get('temp_dir', '/tmp/gcover/sde')),
            cleanup_on_exit=data.get('cleanup_on_exit', True)
        )

    @classmethod
    def get_section_name(cls) -> str:
        return "sde"