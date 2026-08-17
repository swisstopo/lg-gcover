"""Retry behavior for S3Uploader's presigned-URL PUT.

Covers the failure mode seen in production: a transport-level error (read
timeout / connection drop) on the actual PUT, distinct from a non-2xx HTTP
response, which should not be retried.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from gcover.gdb.storage import S3Uploader


@pytest.fixture
def uploader(tmp_path, monkeypatch):
    monkeypatch.setattr(S3Uploader, "_init_s3_client", lambda self: None)
    u = S3Uploader(
        bucket_name="test-bucket",
        lambda_endpoint="https://example.invalid/presign",
        totp_token="123456",
        upload_method="presigned",
        show_progress=False,
        retry_backoff_seconds=0,  # keep test fast
    )
    return u


@pytest.fixture
def small_file(tmp_path):
    p = tmp_path / "asset.zip"
    p.write_bytes(b"x" * 100)
    return p


def _presigned_data():
    return {
        "presigned_url": "https://example.invalid/put",
        "headers": {},
        "status_code": 200,
    }


def test_retries_on_transport_error_then_succeeds(uploader, small_file):
    ok_response = MagicMock(status_code=200, text="")

    with patch.object(uploader, "_get_presigned_url", return_value=_presigned_data()):
        with patch(
            "gcover.gdb.storage.requests.put",
            side_effect=[requests.exceptions.ReadTimeout("stalled"), ok_response],
        ) as mock_put:
            result = uploader._upload_with_presigned_url(small_file, "gdb-assets/RC1/foo.zip")

    assert result.success is True
    assert mock_put.call_count == 2


def test_gives_up_after_max_retries(uploader, small_file):
    with patch.object(uploader, "_get_presigned_url", return_value=_presigned_data()):
        with patch(
            "gcover.gdb.storage.requests.put",
            side_effect=requests.exceptions.ReadTimeout("stalled"),
        ) as mock_put:
            result = uploader._upload_with_presigned_url(small_file, "gdb-assets/RC1/foo.zip")

    assert result.success is False
    assert mock_put.call_count == uploader.max_upload_retries


def test_non_2xx_response_is_not_retried(uploader, small_file):
    bad_response = MagicMock(status_code=500, text="server error")

    with patch.object(uploader, "_get_presigned_url", return_value=_presigned_data()):
        with patch(
            "gcover.gdb.storage.requests.put", return_value=bad_response
        ) as mock_put:
            result = uploader._upload_with_presigned_url(small_file, "gdb-assets/RC1/foo.zip")

    assert result.success is False
    assert mock_put.call_count == 1
