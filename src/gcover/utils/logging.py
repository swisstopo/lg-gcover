# src/gcover/utils/logging.py
"""
Centralized logging configuration for the gcover application.
"""

from loguru import logger
from rich.console import Console
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from gcover.config import load_config, AppConfig


class GCoverLogger:
    """Centralized logger for the gcover application with config integration."""

    def __init__(self):
        self.console = Console()
        self._is_configured = False
        self._current_level = "INFO"
        self._log_file = None
        self._environment = "development"

    def setup(
        self,
        verbose: bool = False,
        log_file: Optional[Path] = None,
        environment: str = "development",
        config_path: Optional[Path] = None,
    ):
        """
        Setup logging using the unified configuration system.

        Args:
            verbose: Enable debug logging (overrides config)
            log_file: Optional custom log file path (overrides config)
            environment: Environment name for config loading
            config_path: Optional path to config file
        """
        if self._is_configured:
            return  # Already configured

        self._environment = environment

        # Load application configuration
        try:
            app_config = load_config(environment=environment, config_path=config_path)
        except Exception as e:
            # Fallback to basic logging if config fails
            self._setup_fallback_logging(verbose)
            logger.warning(f"Failed to load config, using fallback logging: {e}")
            return

        # Get logging configuration from the app config
        logging_config = app_config.global_.get_logging_config(environment)

        # Determine log level
        if verbose:
            log_level = "DEBUG"
        else:
            log_level = app_config.global_.log_level

        self._current_level = log_level

        # Remove default loguru handler
        logger.remove()

        # Setup console logging
        self._setup_console_logging(log_level, verbose, logging_config["console"])

        # Setup file logging
        if log_file:
            # Use custom log file path
            self._log_file = log_file
            self._setup_file_logging(
                log_level,
                {"rotation": "10 MB", "retention": "30 days", "compression": "gz"},
            )
        elif logging_config["file"]["enabled"] and logging_config["file"]["path"]:
            # Use config-resolved log file path
            self._log_file = logging_config["file"]["path"]
            self._setup_file_logging(log_level, logging_config["file"])

        # Setup module-specific logging
        if logging_config["modules"]:
            self._setup_module_logging(logging_config["modules"])

        self._is_configured = True

        if verbose:
            self.console.print(
                f"[dim]Logging configured: level={log_level}, file={self._log_file}[/dim]"
            )

        logger.info(
            f"GCover logging initialized (level={log_level}, env={environment})"
        )

    def _setup_fallback_logging(self, verbose: bool):
        """Setup basic logging when config loading fails."""
        logger.remove()

        log_level = "DEBUG" if verbose else "INFO"
        self._current_level = log_level

        # Simple console logging
        logger.add(
            lambda msg: self.console.print(msg, end=""),
            format="<level>{level}</level>: {message}",
            level=log_level,
            colorize=True,
        )

        # Simple file logging
        fallback_log_file = Path(
            f"logs/gcover_fallback_{datetime.now().strftime('%Y%m%d')}.log"
        )
        fallback_log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(fallback_log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=log_level,
            rotation="10 MB",
        )

        self._log_file = fallback_log_file
        self._is_configured = True

    def _setup_console_logging(
        self, log_level: str, verbose: bool, console_config: Dict
    ):
        """Setup console logging based on configuration."""
        format_type = console_config.get("format", "simple")
        show_time = console_config.get("show_time", True)
        show_level = console_config.get("show_level", True)
        show_path = console_config.get("show_path", False)

        if verbose or format_type == "detailed":
            # Detailed format for debugging
            if show_path:
                format_str = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
            else:
                format_str = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}</cyan> - <level>{message}</level>"
        else:
            # Simple format
            parts = []
            if show_time:
                parts.append("<green>{time:HH:mm:ss}</green>")
            if show_level:
                parts.append("<level>{level}</level>")
            parts.append("{message}")

            format_str = " | ".join(parts)

        logger.add(
            lambda msg: self.console.print(msg, end=""),
            format=format_str,
            level=log_level,
            colorize=True,
            diagnose=verbose,
        )

    def _setup_file_logging(self, log_level: str, file_config: Dict):
        """Setup file logging based on configuration."""
        if not self._log_file:
            return

        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(self._log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=file_config.get("rotation", "10 MB"),
            retention=file_config.get("retention", "30 days"),
            compression=file_config.get("compression", "gz"),
            enqueue=True,
            diagnose=True,
        )

    def _setup_module_logging(self, modules_config: Dict[str, str]):
        """Setup module-specific log levels."""
        for module_name, level in modules_config.items():
            # Create a filter for this specific module
            def create_module_filter(module):
                return lambda record: record["name"].startswith(module)

            logger.add(
                sys.stderr,
                format="{message}",
                level=level,
                filter=create_module_filter(module_name),
            )

    def enable_debug_mode(self):
        """Enable debug mode dynamically."""
        if self._current_level != "DEBUG":
            logger.remove()  # Remove existing handlers
            self._is_configured = False  # Reset configuration flag
            self.setup(
                verbose=True, log_file=self._log_file, environment=self._environment
            )

    def get_log_file_path(self) -> Optional[Path]:
        """Get the current log file path."""
        return self._log_file

    def show_log_info(self):
        """Display logging information."""
        self.console.print(f"[bold]Logging Configuration:[/bold]")
        self.console.print(f"  Environment: {self._environment}")
        self.console.print(f"  Level: {self._current_level}")
        self.console.print(f"  Log file: {self._log_file}")
        if self._log_file and self._log_file.exists():
            size_mb = self._log_file.stat().st_size / (1024 * 1024)
            self.console.print(f"  File size: {size_mb:.2f} MB")


# Global logger instance
gcover_logger = GCoverLogger()


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    environment: str = "development",
    config_path: Optional[Path] = None,
):
    """
    Setup logging for gcover application using unified configuration.

    Args:
        verbose: Enable debug logging
        log_file: Optional custom log file path
        environment: Environment name
        config_path: Optional path to config file
    """
    gcover_logger.setup(
        verbose=verbose,
        log_file=log_file,
        environment=environment,
        config_path=config_path,
    )


# Example usage/testing
if __name__ == "__main__":
    # Test the logging setup
    print("Testing logging configuration...")

    # Test with development environment
    setup_logging(verbose=True, environment="development")

    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Show current configuration
    gcover_logger.show_log_info()
