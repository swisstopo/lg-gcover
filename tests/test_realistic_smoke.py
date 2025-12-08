# tests/test_cli_realistic.py
"""Realistic CLI smoke tests using actual commands and test data."""
import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import shutil
from gcover.cli.main import cli


@pytest.fixture
def isolated_config_env(tmp_path):
    """
    Create isolated test environment with config files.

    Returns the temporary directory path for further customization.
    """
    # Read original config files
    original_config = Path("config/gcover_config.yaml").read_text(encoding="utf-8")
    original_test_config = Path("config/environments/test.yaml").read_text(encoding="utf-8")

    # Modify test config to use dummy paths (no real GDB scanning)
    modified_test_config = original_test_config.replace(
        "tests/data/examples",
        str(tmp_path / "dummy_data")
    )

    # Create directory structure in tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "environments").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "dummy_data").mkdir()
    (tmp_path / "logs").mkdir()

    # Write config files
    (config_dir / "gcover_config.yaml").write_text(original_config, encoding="utf-8")
    (config_dir / "environments" / "test.yaml").write_text(modified_test_config, encoding="utf-8")

    return tmp_path


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture  
def test_data_exists():
    """Check if test data directory exists."""
    test_data_path = Path("tests/data/examples")
    if not test_data_path.exists():
        pytest.skip("Test data directory not found - skipping realistic tests")
    return test_data_path

@pytest.fixture
def test_styles_dir_exists():
    """Check if test data directory exists."""
    test_styles_path = Path("tests/data/styles")
    if not test_styles_path.exists():
        pytest.skip("Test style directory not found - skipping realistic tests")
    return test_styles_path


# Collect all .lyrx files under tests/data/styles
styles_dir = Path(__file__).parent / "data" / "styles"
lyrx_files = list(styles_dir.glob("*.lyrx"))
output_formats= ['csv', 'json']

@pytest.mark.parametrize("lyrx_file", lyrx_files)
def test_publish_export_classification_with_test_env(runner, lyrx_file):
    """Test gdb scan command with test environment for each .lyrx file."""
    result = runner.invoke(
        cli,
        ["--env", "test", "publish", "extract-classification", str(lyrx_file)]
    )

    # Command should succeed
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Check for expected patterns in output
    output = result.output
    assert "Layer Classification" in output


@pytest.mark.parametrize("output_format", output_formats)
def test_publish_export_classification_format_with_test_env(runner, output_format):
    """Test gdb scan command with test environment for each .lyrx file."""
    lyrx_path = 'tests/data/styles/Linear_Objects.lyrx'
    result = runner.invoke(
        cli,
        ["--env", "test", "publish", "extract-classification", '--export',output_format , lyrx_path]
    )

    # Command should succeed
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Check for expected patterns in output
    output = result.output
    assert lyrx_path.replace('.lyrx', f'.classifications.{output_format}') in output

def test_gdb_scan_with_test_env(runner, test_data_exists):
    """Test gdb scan command with test environment."""
    result = runner.invoke(cli, ['--env', 'test', 'gdb', 'scan'])
    
    # Command should succeed
    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    
    # Check for expected patterns in output
    output = result.output
    
    # Configuration loading should be visible
    assert "Loading base config:" in output
    assert "Loading environment config:" in output
    assert "Environment: test" in output
    
    # Should show filesystem scanning
    assert "Scanning filesystem..." in output
    
    # Should find some assets (we know there's at least one test GDB)
    assert "GDB Assets Found" in output
    assert "verification_topology" in output
    assert "Total:" in output and "GDB assets found" in output
    
    # Should show proper formatting (rich table)
    assert "┏" in output or "│" in output  # Table formatting characters


def test_gdb_scan_count_assets(runner, test_data_exists):
    """Test that scan finds the expected number of test assets."""
    result = runner.invoke(cli, ['--env', 'test', 'gdb', 'scan'])
    
    assert result.exit_code == 0
    
    # Parse the output to check asset count
    lines = result.output.split('\n')
    total_line = [line for line in lines if "Total:" in line and "GDB assets found" in line]
    
    assert len(total_line) == 1, "Should have exactly one total line"
    
    # Should find at least 1 asset (the test GDB)
    total_line_text = total_line[0]
    assert "1 GDB assets found" in total_line_text


def test_gdb_scan_with_verbose(runner, test_data_exists):
    """Test gdb scan with verbose flag."""
    result = runner.invoke(cli, ['--env', 'test', '--verbose', 'gdb', 'scan'])
    
    assert result.exit_code == 0
    
    # Verbose should show more debug information
    output = result.output
    assert "DEBUG" in output
    assert "GDB Assets Found" in output

@pytest.mark.skip(reason="why is the metadata duckdb needed?")
def test_gdb_scan_handles_missing_data_gracefully(runner):
    """Test scan command when no GDB files are present."""
    # Use a temporary directory with no GDB files
    source = Path("config/environments/test.yaml")
    content = source.read_text(encoding="utf-8").replace("tests/data/examples", "dummy/data/examples")
    Path("dummy/data/examples").mkdir(parents=True,exist_ok = True)
    config = Path("config/gcover_config.yaml").read_text(encoding="utf-8")
    print(content)
    with runner.isolated_filesystem():
        # Create minimal config structure
        Path("config").mkdir()
        Path("config/gcover_config.yaml").write_text(config, encoding="utf-8")
        Path("config/environments").mkdir()
        Path("config/environments/test.yaml").write_text(content, encoding="utf-8")
        result = runner.invoke(cli, ['--env', 'test', 'gdb', 'init'])
        result = runner.invoke(cli, ['--env', 'test', 'gdb', 'scan'])
        print(result.output)

        # Should still succeed (no crash)
        assert result.exit_code == 0
        
        # Should handle empty results gracefully
        output = result.output
        assert "Scanning filesystem..." in output
        assert ("No GDB assets found" in output)


def test_multiple_commands_with_test_env(runner, test_data_exists):
    """Test several commands to ensure they don't crash."""
    commands_to_test = [
        ['--env', 'test', 'gdb', 'scan'],
        ['--env', 'test', 'gdb', 'list'],  # If this command exists
        ['--help'],
        ['gdb', '--help'],
    ]
    
    for cmd in commands_to_test:
        try:
            result = runner.invoke(cli, cmd)
            # Allow success (0) or expected failure (1-2) but not crashes
            assert result.exit_code in [0, 1, 2], f"Command {cmd} crashed with code {result.exit_code}: {result.output}"
            
            # Should not contain Python tracebacks (sign of unhandled exceptions)
            assert "Traceback" not in result.output, f"Command {cmd} had unhandled exception: {result.output}"
            
        except Exception as e:
            pytest.fail(f"Command {cmd} raised exception: {e}")


def test_config_loading_error_handling(runner):
    """Test that config loading errors are handled gracefully."""
    with runner.isolated_filesystem():
        # Create invalid config
        Path("config").mkdir()
        Path("config/gcover_config.yaml").write_text("invalid: yaml: content: [")
        
        result = runner.invoke(cli, ['--env', 'test', 'gdb', 'scan'])
        
        # Should fail gracefully, not crash
        assert result.exit_code != 0
        assert "Traceback" not in result.output or "YAML" in result.output


@pytest.mark.parametrize("env_name", ["test", "dev", "prod"])
def test_different_environments(runner, env_name):
    """Test that different environment names are handled properly."""
    result = runner.invoke(cli, ['--env', env_name, '--help'])
    
    # Should not crash regardless of environment
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_gdb_init_duckdb(runner, test_data_exists):
    """Test that output format is consistent and well-formed."""
    source = Path("config/environments/test.yaml")
    content = source.read_text(encoding="utf-8").replace("tests/data/examples", "dummy/data/examples")
    Path("dummy/data/examples").mkdir(parents=True, exist_ok=True)
    config = Path("config/gcover_config.yaml").read_text(encoding="utf-8")
    print(content)
    with runner.isolated_filesystem():
        # Create minimal config structure
        Path("config").mkdir()
        Path("data").mkdir()
        Path("config/gcover_config.yaml").write_text(config, encoding="utf-8")
        Path("config/environments").mkdir()
        Path("config/environments/test.yaml").write_text(content, encoding="utf-8")
        result = runner.invoke(cli, ['--env', 'test', '--verbose', 'gdb', 'init'])

        file_exists = Path("data/test_gdb_metadata.duckdb").exists()
        print(result.output)

        assert result.exit_code == 0
        assert file_exists == True

        assert 'Initialization complete!' in result.output

def test_output_format_consistency(runner, test_data_exists):
    """Test that output format is consistent and well-formed."""
    result = runner.invoke(cli, ['--env', 'test', 'gdb', 'scan'])
    
    assert result.exit_code == 0
    
    output = result.output
    
    # Check for consistent formatting
    lines = output.split('\n')
    
    # Should have structured output, not just dump
    table_lines = [line for line in lines if '┏' in line or '┡' in line or '└' in line]
    assert len(table_lines) >= 2, "Should have table formatting"
    
    # Should have summary information
    summary_lines = [line for line in lines if "Total:" in line]
    assert len(summary_lines) >= 1, "Should have summary line"


def test_performance_smoke_test(runner, test_data_exists):
    """Basic performance smoke test - should complete reasonably quickly."""
    import time
    
    start_time = time.time()
    result = runner.invoke(cli, ['--env', 'test', 'gdb', 'scan'])
    end_time = time.time()
    
    assert result.exit_code == 0
    
    # Should complete within reasonable time (10 seconds for test data)
    duration = end_time - start_time
    assert duration < 10.0, f"Command took too long: {duration:.2f} seconds"


