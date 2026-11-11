"""Tests for db/cache path resolution and the metadata Parquet cache."""
import os
import time
from pathlib import Path

from gcover.config.models import GDBConfig, QAConfig
from gcover.config.paths import resolve_db_path
from gcover.gdb import storage


def test_relative_db_path_resolves_to_db_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GCOVER_DB_DIR", str(tmp_path))
    assert resolve_db_path(Path("x.duckdb")) == tmp_path / "x.duckdb"
    assert GDBConfig(base_paths={}).db_path == tmp_path / "dev_gdb_metadata.duckdb"
    assert QAConfig().db_path == tmp_path / "prod_verification_stats.duckdb"


def test_absolute_db_path_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv("GCOVER_DB_DIR", str(tmp_path))
    assert GDBConfig(base_paths={}, db_path="/abs/y.duckdb").db_path == Path("/abs/y.duckdb")


def test_metadata_parquet_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GCOVER_CACHE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        storage, "_fetch_metadata_to", lambda dest, *a: (calls.append(1), dest.write_bytes(b"x"))
    )
    first = storage.fetch_metadata_parquet(object())
    assert storage.fetch_metadata_parquet(object()) == first and len(calls) == 1

    old = time.time() - 2 * storage.METADATA_CACHE_MAX_AGE
    os.utime(first, (old, old))
    storage.fetch_metadata_parquet(object())
    assert len(calls) == 2


def test_failed_download_keeps_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GCOVER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "_fetch_metadata_to", lambda dest, *a: dest.write_bytes(b"good"))
    cached = storage.fetch_metadata_parquet(object())

    def boom(dest, *a):
        dest.write_bytes(b"partial")
        raise RuntimeError("net down")

    monkeypatch.setattr(storage, "_fetch_metadata_to", boom)
    try:
        storage.fetch_metadata_parquet(object(), max_age=0)
    except RuntimeError:
        pass
    assert cached.read_bytes() == b"good"
    assert [p.name for p in tmp_path.iterdir()] == [cached.name]
