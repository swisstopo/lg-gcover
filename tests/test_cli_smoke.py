"""Smoke tests for CLI commands - basic functionality without real data."""

import pytest
from click.testing import CliRunner
from gcover.cli.main import cli


def test_cli_help():
    """Test that main CLI help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--env", "test", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_gdb_command_help():
    """Test that gdb subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--env", "test", "gdb", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_gdb_command_no_args():
    """Test gdb command with no arguments (should show help or fail gracefully)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--env", "test","gdb"])
    # Should either show help (exit 0) or fail gracefully (not crash)
    assert result.exit_code in [0, 1, 2]  # Common "expected" exit codes
    # Should not be a Python exception/traceback
    assert "Traceback" not in result.output


def test_gdb_command_invalid_path():
    """Test gdb command with obviously invalid path fails gracefully."""
    runner = CliRunner()
    result = runner.invoke(cli, ["gdb", "/definitely/does/not/exist.gdb"])
    # Should fail gracefully, not crash with unhandled exception
    assert result.exit_code != 0
    assert "Traceback" not in result.output or "FileNotFoundError" in result.output
