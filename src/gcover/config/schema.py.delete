# src/gcover/config/schema.py
"""
Schema module configuration
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .core import BaseConfig


@dataclass
class SchemaConfig(BaseConfig):
    """Configuration for schema management"""
    output_dir: Path
    template_dir: Optional[Path]
    default_formats: List[str]
    plantuml_path: Optional[Path]
    max_diagram_tables: int = 50
    include_system_tables: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaConfig':
        """Create config from dictionary"""
        return cls(
            output_dir=Path(data.get('output_dir', './schemas')),
            template_dir=Path(data['template_dir']) if data.get('template_dir') else None,
            default_formats=data.get('default_formats', ['json']),
            plantuml_path=Path(data['plantuml_path']) if data.get('plantuml_path') else None,
            max_diagram_tables=data.get('max_diagram_tables', 50),
            include_system_tables=data.get('include_system_tables', False)
        )

    @classmethod
    def get_section_name(cls) -> str:
        return "schema"