"""
Gestion des imports optionnels, notamment arcpy.
"""

import importlib
from functools import wraps


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
