import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from gcover.gdb import load_config, GDBAssetManager

def test_config_loading():
    """Test configuration loading"""
    # Mock config file for testing
    with patch('gcover.gdb.config.Path.exists', return_value=True):
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = """
base_paths:
  backup: "/test/backup"
s3:
  bucket: "test-bucket"
database:
  path: "test.db"
temp_dir: "/tmp"
"""
            config = load_config()
            assert config.s3_bucket == "test-bucket"

@patch('gcover.gdb.storage.boto3')
@patch('gcover.gdb.storage.duckdb')
def test_manager_creation(mock_duckdb, mock_boto3):
    """Test manager creation with mocked dependencies"""
    config = load_config()
    manager = GDBAssetManager(
        base_paths=config.base_paths,
        s3_bucket=config.s3_bucket,
        db_path=config.db_path,
        temp_dir=config.temp_dir
    )
    assert manager is not None