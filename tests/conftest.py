"""Pytest configuration and fixtures."""

import pytest
import os
import sys
from loguru import logger
import io
from pathlib import Path

# Add src to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))





@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def project_root():
    """Path to project root."""
    return Path(__file__).parent.parent



def pytest_configure(config):
    """Force test environment for all pytest runs"""
    os.environ["GCOVER_ENVIRONMENT"] = "test"
    print("🧪 FORCED: GCOVER_ENVIRONMENT=test for pytest")
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(autouse=True)
def loguru_capture():
    stream = io.StringIO()
    logger.remove()
    logger.add(stream, level="INFO")
    yield stream
    logger.remove()


@pytest.fixture(scope="session")
def temp_gdb_path():
    """Provide a path for temporary test geodatabase."""
    import tempfile

    return Path(tempfile.gettempdir()) / "test_data.gdb"


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    # Prevent arcpy from trying to create licensing popups in CI
    os.environ["ESRI_CONCURRENT_LICENSE_TIMEOUT"] = "0"
