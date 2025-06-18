"""
Gestion des imports optionnels, notamment arcpy.
"""
import importlib
from functools import wraps


def try_import_arcpy() -> None:
    """Tente d'importer arcpy et retourne un flag de disponibilité."""
    try:
        import arcpy

        return arcpy, True
    except ImportError:
        return None, False


# Import global
ARCPY, HAS_ARCPY = try_import_arcpy()


def require_arcpy(func):
    """
    Décorateur pour les fonctions nécessitant arcpy.

    Usage:
        @require_arcpy
        def ma_fonction():
            # Utilise arcpy
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_ARCPY:
            raise ImportError(
                f"{func.__name__} requires arcpy. "
                "Please install ArcGIS Pro and use its Python environment."
            )
        return func(*args, **kwargs)

    return wrapper


def optional_import(module_name, package=None):
    """
    Import optionnel d'un module.

    Args:
        module_name: Nom du module à importer
        package: Package parent si import relatif

    Returns:
        tuple: (module, disponible)
    """
    try:
        module = importlib.import_module(module_name, package)
        return module, True
    except ImportError:
        return None, False


# Vérifier d'autres dépendances optionnelles
BOTO3, HAS_BOTO3 = optional_import("boto3")
MATPLOTLIB, HAS_MATPLOTLIB = optional_import("matplotlib")
PLOTLY, HAS_PLOTLY = optional_import("plotly")
