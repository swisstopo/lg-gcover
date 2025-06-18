"""
Configuration management for gcover.
"""
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration principale de gcover."""

    # Chemins par défaut
    config_dir: Path = Field(default_factory=lambda: Path.home() / ".gcover")
    log_dir: Path = Field(default_factory=lambda: Path.home() / ".gcover" / "logs")

    # Configuration des connexions
    connections: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Options par défaut
    default_crs: int = 2056  # CH1903+ / LV95
    output_format: str = "geoparquet"

    # AWS
    aws_region: Optional[str] = None
    aws_bucket: Optional[str] = None

    class Config:
        """Configuration Pydantic."""

        validate_assignment = True
        extra = "allow"

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "Config":
        """
        Charge la configuration depuis un fichier.

        Args:
            config_file: Chemin vers le fichier de configuration

        Returns:
            Instance de Config
        """
        if config_file is None:
            # Chercher dans l'ordre : .gcoverrc, ~/.gcover/config.yaml
            search_paths = [
                Path.cwd() / ".gcoverrc",
                Path.home() / ".gcover" / "config.yaml",
            ]

            for path in search_paths:
                if path.exists():
                    config_file = path
                    break

        if config_file and config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)

        # Configuration par défaut
        return cls()

    def save(self, config_file: Optional[Path] = None):
        """Sauvegarde la configuration."""
        if config_file is None:
            config_file = self.config_dir / "config.yaml"

        # Créer le répertoire si nécessaire
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
