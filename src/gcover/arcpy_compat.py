"""
Centralized arcpy import and compatibility layer.

This module provides a single point of import for arcpy, with graceful
degradation when arcpy is not available.

IMPORTANT: This module uses DELAYED import of arcpy to avoid DLL loading
issues in ArcGIS Pro environments. The actual import happens on first use.

Usage:
    from gcover.arcpy_compat import arcpy, HAS_ARCPY, require_arcpy

    if HAS_ARCPY:
        # Use arcpy functionality
        pass

    @require_arcpy
    def some_function():
        # This function requires arcpy
        arcpy.management.CopyFeatures(...)
"""
import sys
import functools
import threading
from typing import Any, Callable, TypeVar, cast, Optional

# Conditional import based on logging availability
try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# ============================================================================
# DELAYED ARCPY IMPORT
# ============================================================================
# We use a lazy/delayed import mechanism to avoid loading arcpy at module
# import time. This prevents DLL conflicts in ArcGIS Pro environments where
# the Python interpreter needs to be fully initialized first.

_arcpy_module: Optional[Any] = None
_arcpy_load_attempted = False
_arcpy_load_lock = threading.Lock()
_arcpy_load_error: Optional[Exception] = None


def _try_import_arcpy() -> tuple[bool, Optional[Any], Optional[Exception]]:
    """
    Attempt to import arcpy (thread-safe, called only once).

    Returns:
        (success, module, error)
    """
    global _arcpy_module, _arcpy_load_attempted, _arcpy_load_error

    with _arcpy_load_lock:
        # Already attempted?
        if _arcpy_load_attempted:
            return (_arcpy_module is not None, _arcpy_module, _arcpy_load_error)

        _arcpy_load_attempted = True

        try:
            import arcpy as _arcpy
            _arcpy_module = _arcpy
            logger.debug("arcpy successfully imported (delayed)")
            return (True, _arcpy_module, None)
        except ImportError as e:
            _arcpy_load_error = e
            logger.debug(f"arcpy not available: {e}")
            return (False, None, e)
        except Exception as e:
            # Catch DLL errors and other unexpected issues
            _arcpy_load_error = e
            logger.debug(f"arcpy import failed with unexpected error: {type(e).__name__}: {e}")
            return (False, None, e)


class _ArcPyProxy:
    """
    Proxy object that delays arcpy import until first attribute access.

    This allows the module to be imported without immediately loading arcpy,
    which can cause DLL conflicts in ArcGIS Pro environments.
    """

    def __getattr__(self, name: str) -> Any:
        """Lazy load arcpy on first attribute access."""
        success, module, error = _try_import_arcpy()

        if not success:
            if error:
                raise ImportError(
                    f"arcpy.{name} is not available.\n"
                    f"Reason: {type(error).__name__}: {error}\n"
                    f"Install ArcGIS Pro or use a compatible environment."
                )
            else:
                raise ImportError(
                    f"arcpy.{name} is not available. "
                    "Install ArcGIS Pro or use a compatible environment."
                )

        # Return the requested attribute from the real arcpy module
        return getattr(module, name)

    def __bool__(self) -> bool:
        """Check if arcpy is available."""
        success, _, _ = _try_import_arcpy()
        return success

    def __repr__(self) -> str:
        success, module, error = _try_import_arcpy()
        if success:
            return f"<arcpy proxy (loaded): {module}>"
        else:
            return f"<arcpy proxy (not available): {error}>"


# Create the proxy instance
arcpy = _ArcPyProxy()


# ============================================================================
# AVAILABILITY CHECK (fast, without importing)
# ============================================================================

def _quick_check_arcpy_available() -> bool:
    """
    Quick check if arcpy is likely available without actually importing it.

    This checks for ArcGIS Pro installation markers but doesn't guarantee
    that arcpy will actually import successfully.
    """
    # Check if we already tried to import
    if _arcpy_load_attempted:
        return _arcpy_module is not None

    # Quick heuristic checks (without importing)
    if sys.platform != "win32":
        return False  # arcpy only on Windows

    # Check for ArcGIS Pro in sys.prefix
    if "ArcGIS" in sys.prefix or "arcgispro" in sys.prefix.lower():
        return True

    # Check for arcpy in sys.modules (already loaded elsewhere)
    if "arcpy" in sys.modules:
        return True

    # Don't know for sure - will need to try importing
    return False


# Public flag - initially based on quick check
# Will be updated to actual status on first real use
HAS_ARCPY = _quick_check_arcpy_available()

# ============================================================================
# DECORATORS AND UTILITIES
# ============================================================================

F = TypeVar('F', bound=Callable[..., Any])


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
        # This will trigger the import attempt if not done yet
        success, _, error = _try_import_arcpy()

        if not success:
            raise ImportError(
                f"Function '{func.__name__}' requires arcpy.\n"
                f"Reason: {error}\n"
                "Please install ArcGIS Pro or use an environment with arcpy available."
            )
        return func(*args, **kwargs)

    return cast(F, wrapper)


class ArcPyRequired:
    """
    Context manager and decorator for operations requiring arcpy.
    """

    def __enter__(self):
        success, _, error = _try_import_arcpy()
        if not success:
            raise ImportError(
                f"This operation requires arcpy.\n"
                f"Reason: {error}\n"
                "Please install ArcGIS Pro or use an environment with arcpy available."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @classmethod
    def decorate_class(cls, target_class: type) -> type:
        """Decorate all public methods of a class to require arcpy."""
        for attr_name in dir(target_class):
            if attr_name.startswith('_'):
                continue

            attr = getattr(target_class, attr_name)
            if callable(attr):
                setattr(target_class, attr_name, require_arcpy(attr))

        return target_class


def check_arcpy_availability(force_check: bool = False) -> dict[str, Any]:
    """
    Check arcpy availability and return diagnostic information.

    Args:
        force_check: Force an actual import attempt (not just cached result)

    Returns:
        Dictionary with availability status and version info
    """
    global HAS_ARCPY

    if force_check:
        # Force a fresh import attempt
        global _arcpy_load_attempted
        _arcpy_load_attempted = False

    success, module, error = _try_import_arcpy()

    # Update the global flag
    HAS_ARCPY = success

    result = {
        "available": success,
        "version": None,
        "install_info": None,
        "error": None,
    }

    if success and module:
        try:
            info = module.GetInstallInfo()
            result["version"] = info.get("Version", "Unknown")
            result["product"] = info.get("ProductName", "Unknown")
            result["install_dir"] = info.get("InstallDir", "Unknown")
        except Exception as e:
            result["error"] = str(e)
    else:
        result["error"] = str(error) if error else "Unknown error"
        result["install_info"] = (
            "arcpy is not available. To use ESRI functionality:\n"
            "1. Install ArcGIS Pro\n"
            "2. Activate the Python environment from ArcGIS Pro\n"
            "3. Install this package in that environment\n"
            f"\nError details: {error}"
        )

    return result


def ensure_arcpy() -> None:
    """
    Ensure arcpy is available or raise ImportError with helpful message.

    Call this at the start of modules that absolutely require arcpy.
    """
    success, _, error = _try_import_arcpy()
    if not success:
        info = check_arcpy_availability()
        raise ImportError(info["install_info"])


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# On module load, do a quick check (without importing arcpy yet)
if _quick_check_arcpy_available():
    logger.debug("arcpy appears to be available (quick check)")
else:
    logger.debug("arcpy does not appear to be available (quick check)")

__all__ = [
    "arcpy",
    "HAS_ARCPY",
    "require_arcpy",
    "ArcPyRequired",
    "check_arcpy_availability",
    "ensure_arcpy",
]