"""Tests for staging/diff workflow."""

import pytest
from pathlib import Path
from gcover.publish.generator import MapServerGenerator
from gcover.publish.diff_tools import detect_changes, parse_class_names


def test_should_regenerate_classes_modes():
    """Test regenerate vs frozen modes."""
    generator = MapServerGenerator(
        layer_type= 'Polygon'
    )

    # Regenerate mode - always True
    assert generator.should_regenerate_classes(
        Path("test.inc"),
        classes_mode="regenerate",
        force=False
    ) == True

    # Frozen mode - False if exists
    existing_file = Path("existing.inc")
    existing_file.touch()

    assert generator.should_regenerate_classes(
        existing_file,
        classes_mode="frozen",
        force=False
    ) == False

    # Frozen mode - True if force
    assert generator.should_regenerate_classes(
        existing_file,
        classes_mode="frozen",
        force=True
    ) == True

    existing_file.unlink()


def test_staging_generation(tmp_path):
    """Test staging file generation."""
    generator = MapServerGenerator(
        layer_type= 'Polygon'
    )

    classes_file = tmp_path / "test_classes.inc"
    new_content = "CLASS\n  NAME \"Test\"\nEND"

    staging_file = generator.generate_to_staging(classes_file, new_content)

    assert staging_file.exists()
    assert staging_file.parent.name == ".staging"
    assert staging_file.name == "test_classes.inc.new"
    assert staging_file.read_text() == new_content


def test_detect_changes():
    """Test change detection."""
    original = """
    CLASS
      NAME "Class A"
    END
    CLASS
      NAME "Class B"
    END
    """

    new = """
    CLASS
      NAME "Class A"
    END
    CLASS
      NAME "Class C"
    END
    """

    original_names = parse_class_names(original)
    new_names = parse_class_names(new)

    assert "Class A" in original_names
    assert "Class B" in original_names
    assert "Class C" in new_names

    added = new_names - original_names
    removed = original_names - new_names

    assert "Class C" in added
    assert "Class B" in removed