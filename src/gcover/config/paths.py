"""Default locations for gcover's local databases and cached downloads."""

import os
from pathlib import Path


def user_config_dir() -> Path:
    """``$XDG_CONFIG_HOME/gcover`` or ``~/.config/gcover`` (where the YAML config lives)."""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "gcover"


def db_dir() -> Path:
    """Directory for local DuckDB files. Override with ``GCOVER_DB_DIR``.

    Not ``GCOVER_DATA_DIR``: that one already points to the products tree.
    """
    if env := os.environ.get("GCOVER_DB_DIR"):
        return Path(env).expanduser()
    return user_config_dir() / "db"


def cache_dir() -> Path:
    """Directory for cached downloads (e.g. metadata Parquet). Override with ``GCOVER_CACHE_DIR``."""
    if env := os.environ.get("GCOVER_CACHE_DIR"):
        return Path(env).expanduser()
    return user_config_dir() / "cache"


def resolve_db_path(p: Path) -> Path:
    """Expand ``~``; resolve relative paths against :func:`db_dir`, not the CWD."""
    p = p.expanduser()
    return p if p.is_absolute() else db_dir() / p
