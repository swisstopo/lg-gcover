"""
Centralized arcpy import and compatibility layer.

This module provides a single point of import for arcpy, with graceful
degradation when arcpy is not available. All modules should import arcpy
through this module instead of directly.

Usage:
    from gcover.arcpy_compat import arcpy, HAS_ARCPY, require_arcpy

    if HAS_ARCPY:
        # Use arcpy functionality
        pass

    @require_arcpy
    def some_function():
        # This function requires arcpy
        pass
"""

import functools
import sys
from typing import Any, Callable, TypeVar, cast

from loguru import logger

# Global flag indicating if arcpy is available
HAS_ARCPY = False
arcpy: Any = None

# Try to import arcpy
try:
    import arcpy as _arcpy

    HAS_ARCPY = True
    arcpy = _arcpy
    logger.debug("arcpy successfully imported")
except ImportError as e:
    logger.debug(f"arcpy not available: {e}")

    # Create a mock arcpy module for better error messages
    class _MockArcPy:
        """Mock arcpy module that raises informative errors."""

        def __getattr__(self, name: str) -> Any:
            raise ImportError(
                f"arcpy.{name} is not available. "
                "Install ArcGIS Pro or use a compatible environment."
            )

        def __bool__(self) -> bool:
            return False

    arcpy = _MockArcPy()


F = TypeVar("F", bound=Callable[..., Any])


def require_arcpy(func: F) -> F:
    """
    Decorator to mark functions that require arcpy.

    Raises ImportError with a helpful message if arcpy is not available.

    Example:
        @require_arcpy
        def export_to_sde(data):
            arcpy.management.CopyFeatures(...)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not HAS_ARCPY:
            raise ImportError(
                f"Function '{func.__name__}' requires arcpy. "
                "Please install ArcGIS Pro or use an environment with arcpy available."
            )
        return func(*args, **kwargs)

    return cast(F, wrapper)


class ArcPyRequired:
    """
    Context manager and decorator for operations requiring arcpy.

    Can be used as a decorator for entire classes:

        @ArcPyRequired.decorate_class
        class SDEConnection:
            def connect(self):
                arcpy.CreateDatabaseConnection_management(...)
    """

    def __enter__(self):
        if not HAS_ARCPY:
            raise ImportError(
                "This operation requires arcpy. "
                "Please install ArcGIS Pro or use an environment with arcpy available."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @classmethod
    def decorate_class(cls, target_class: type) -> type:
        """
        Decorate all public methods of a class to require arcpy.

        Example:
            @ArcPyRequired.decorate_class
            class MySDEClass:
                def method1(self): ...
                def method2(self): ...
        """
        for attr_name in dir(target_class):
            if attr_name.startswith("_"):
                continue

            attr = getattr(target_class, attr_name)
            if callable(attr):
                setattr(target_class, attr_name, require_arcpy(attr))

        return target_class


def check_arcpy_availability() -> dict[str, Any]:
    """
    Check arcpy availability and return diagnostic information.

    Returns:
        Dictionary with availability status and version info
    """
    result = {
        "available": HAS_ARCPY,
        "version": None,
        "install_info": None,
    }

    if HAS_ARCPY:
        try:
            result["version"] = arcpy.GetInstallInfo()
            result["product"] = arcpy.GetInstallInfo().get("ProductName", "Unknown")
        except Exception as e:
            result["error"] = str(e)
    else:
        result["install_info"] = (
            "arcpy is not available. To use ESRI functionality:\n"
            "1. Install ArcGIS Pro\n"
            "2. Activate the Python environment from ArcGIS Pro\n"
            "3. Install this package in that environment"
        )

    return result


# Export convenience function for checking
def ensure_arcpy() -> None:
    """
    Ensure arcpy is available or raise ImportError with helpful message.

    Call this at the start of modules that absolutely require arcpy.
    """
    if not HAS_ARCPY:
        info = check_arcpy_availability()
        raise ImportError(info["install_info"])


__all__ = [
    "arcpy",
    "HAS_ARCPY",
    "require_arcpy",
    "ArcPyRequired",
    "check_arcpy_availability",
    "ensure_arcpy",
]
