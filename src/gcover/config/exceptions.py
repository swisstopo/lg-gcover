# src/gcover/config/exceptions.py
"""
Configuration-related exceptions
"""


class ConfigurationError(Exception):
    """Base exception for configuration errors"""

    pass


class ConfigurationNotFoundError(ConfigurationError):
    """Raised when configuration file is not found"""

    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails"""

    pass
