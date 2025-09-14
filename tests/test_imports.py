"""Test that all modules can be imported without errors."""

import pytest


def test_main_package_import():
    """Test that main package imports successfully."""
    import gcover

    assert gcover.__name__ == "gcover"


def test_cli_imports():
    """Test CLI module imports."""
    # These should not raise ImportError
    from gcover.cli import main
    from gcover.cli import gdb_cmd
    from gcover.cli import qa_cmd
    from gcover.cli import schema_cmd
    from gcover.cli import sde_cmd

    assert callable(main.cli)


def test_sde_imports():
    """Test SDE-related imports (may skip if arcpy unavailable)."""
    try:
        from gcover.sde import bridge
        from gcover.sde import connection_manager

        # If we get here, arcpy is available
        assert True
    except ImportError as e:
        if "arcpy" in str(e):
            pytest.skip("arcpy not available - SDE tests skipped")
        else:
            raise


def test_core_imports():
    """Test core functionality imports."""
    # Add other core modules as they exist
    # from gcover.core import something
    pass
