import pytest
from datetime import datetime, timedelta
from gcover.cli.main import parse_since
import sys

# Patch sys.exit to prevent actual exit during tests
@pytest.fixture(autouse=True)
def patch_sys_exit(monkeypatch):
    monkeypatch.setattr(sys, "exit", lambda code=1: (_ for _ in ()).throw(SystemExit(code)))

def test_absolute_date():
    result = parse_since("2025-10-01")
    assert result == datetime(2025, 10, 1)

def test_natural_language_week():
    result = parse_since("1 week ago")
    delta = datetime.today() - result
    assert 6 <= delta.days <= 8

def test_natural_language_two_week():
    result = parse_since("2 weeks ago")
    delta = datetime.today() - result
    assert 13 <= delta.days <= 15

def test_natural_language_month():
    result = parse_since("1 month ago")
    delta = datetime.today() - result
    assert 28 <= delta.days <= 31

def test_yesterday():
    result = parse_since("yesterday")
    delta = datetime.today() - result
    assert 0 <= delta.days <= 2

def test_invalid_format_raises():
    with pytest.raises(SystemExit):
        parse_since("not-a-date")

def test_empty_string_raises():
    with pytest.raises(SystemExit):
        parse_since("")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

