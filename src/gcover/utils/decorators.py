# utils/decorators.py
from functools import wraps

from .imports import HAS_ARCPY


def require_arcpy(func):
    """Décorateur pour les fonctions nécessitant arcpy."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_ARCPY:
            raise ImportError(
                f"{func.__name__} nécessite arcpy. "
                "Installez ArcGIS Pro et utilisez son environnement Python."
            )
        return func(*args, **kwargs)

    return wrapper
