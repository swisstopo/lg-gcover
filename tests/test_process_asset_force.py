"""`--force` reprocessing must not lose a previously-good DB row.

Regression coverage for a bug where GDBAssetManager.process_asset() deleted
the existing metadata row *before* attempting the (risky) zip/upload, so a
failure in between (locked FileGDB, network drop, ...) left zero rows for
that asset instead of the prior good one.
"""
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from gcover.gdb.assets import AssetType, GDBAsset, GDBAssetInfo, ReleaseCandidate
from gcover.gdb.manager import GDBAssetManager
from gcover.gdb.storage import MetadataDB, UploadResult


class StubAsset(GDBAsset):
    """Minimal GDBAsset stand-in that skips real filesystem parsing."""

    def __init__(self, path, info, should_fail=False):
        self.path = Path(path)
        self.info = info
        self._should_fail = should_fail

    def create_zip(self, output_dir):
        if self._should_fail:
            raise OSError(
                "[Errno 13] Permission denied: FileGDB is locked (QA still running)"
            )
        zip_path = Path(output_dir) / self.zip_filename
        zip_path.write_bytes(b"fake-zip-bytes")
        self.info.zip_path = zip_path
        self.info.file_size = zip_path.stat().st_size
        return zip_path

    def compute_hash(self):
        self.info.hash_md5 = "deadbeef"
        return self.info.hash_md5


def _rows(metadata_db: MetadataDB, path: Path):
    with duckdb.connect(str(metadata_db.db_path)) as conn:
        return conn.execute(
            "SELECT s3_key, uploaded FROM gdb_assets WHERE path = ?", [str(path)]
        ).fetchall()


def _make_manager(tmp_path, uploader) -> GDBAssetManager:
    mgr = GDBAssetManager.__new__(GDBAssetManager)
    mgr.bucket_name = "test-bucket"
    mgr.base_paths = {}
    mgr.temp_dir = tmp_path
    mgr.upload_to_s3 = True
    mgr.show_progress = False
    mgr.s3_uploader = uploader
    mgr.metadata_db = MetadataDB(tmp_path / "meta.duckdb")
    return mgr


def _asset_info(path: Path, uploaded: bool, s3_key=None):
    return GDBAssetInfo(
        path=path,
        asset_type=AssetType.BACKUP_WEEKLY,
        release_candidate=ReleaseCandidate.RC2,
        timestamp=datetime(2026, 8, 10, 3, 30),
        file_size=123,
        s3_key=s3_key,
        uploaded=uploaded,
        metadata={},
    )


def test_failed_force_reprocess_preserves_prior_uploaded_row(tmp_path):
    gdb_path = Path("/mock/source/20260810_0330_2030-12-31.gdb")

    uploader = MagicMock()
    manager = _make_manager(tmp_path, uploader)

    # Seed a prior, genuinely-successful upload.
    good_info = _asset_info(gdb_path, uploaded=True, s3_key="gdb-assets/RC2/backup_weekly/old.zip")
    manager.metadata_db.insert_asset(good_info)
    assert _rows(manager.metadata_db, gdb_path) == [("gdb-assets/RC2/backup_weekly/old.zip", True)]

    # Force reprocess, but the zip step fails (e.g. locked FileGDB).
    failing_info = _asset_info(gdb_path, uploaded=False)
    asset = StubAsset(gdb_path, failing_info, should_fail=True)

    success, error = manager.process_asset(asset, force=True)

    assert success is False
    assert "locked" in error

    # The previously-good row must still be there — not wiped by the failed retry.
    assert _rows(manager.metadata_db, gdb_path) == [("gdb-assets/RC2/backup_weekly/old.zip", True)]
    uploader.upload_file.assert_not_called()


def test_successful_force_reprocess_replaces_row_without_duplicates(tmp_path):
    gdb_path = Path("/mock/source/20260810_0330_2030-12-31.gdb")

    uploader = MagicMock()
    uploader.file_exists.return_value = False
    uploader.upload_file.return_value = UploadResult(
        success=True, status_code=200, s3_key="new-key", method="presigned"
    )
    manager = _make_manager(tmp_path, uploader)

    stale_info = _asset_info(gdb_path, uploaded=False, s3_key="gdb-assets/RC2/backup_weekly/stale.zip")
    manager.metadata_db.insert_asset(stale_info)

    fresh_info = _asset_info(gdb_path, uploaded=False)
    asset = StubAsset(gdb_path, fresh_info, should_fail=False)

    success, error = manager.process_asset(asset, force=True)

    assert success is True
    assert error is None

    rows = _rows(manager.metadata_db, gdb_path)
    assert len(rows) == 1  # no duplicate left behind
    assert rows[0][1] is True  # uploaded flipped to True
