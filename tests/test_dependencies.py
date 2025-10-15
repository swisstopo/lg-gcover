"""Test that required dependencies are available."""

import pytest


def test_required_dependencies():
    """Test that all required dependencies can be imported."""
    required_deps = [
        "geopandas",
        "shapely",
        "pandas",
        "click",
        "rich",
        "pydantic",
        "yaml",  # pyyaml
        "structlog",
        "dotenv",  # python-dotenv
    ]

    missing = []
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        pytest.fail(f"Missing required dependencies: {missing}")


def test_optional_dependencies():
    """Test optional dependencies - should not fail if missing."""
    optional_deps = {
        "arcpy": "ESRI ArcGIS functionality",
        "boto3": "AWS S3 support",
        "matplotlib": "Visualization support",
    }

    available = {}
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            available[dep] = True
        except ImportError:
            available[dep] = False

    # Just log what's available (don't fail)
    print(f"\nOptional dependencies status: {available}")
