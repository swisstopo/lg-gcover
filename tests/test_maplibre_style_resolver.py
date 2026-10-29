"""Unit tests for gcover.publish.maplibre_style_resolver.

Pure-logic tests build synthetic mappyfile-shaped CLASS dicts directly (no
mapfile parsing involved) so they stay fast and independent of the sibling
mapserver-geocover checkout. One integration test exercises the real thing
end-to-end against that checkout, when it's available.
"""

from pathlib import Path

import pandas as pd
import pytest

from gcover.publish.maplibre_style_resolver import (
    LAYER_REGISTRY,
    SubLayer,
    _dash_bucket,
    _hatch_bucket,
    _hex,
    _resolve_area_class,
    _resolve_line_class,
    _resolve_point_class,
    _source_confidence,
    _spacing_bucket,
    resolve_all_groups,
    resolve_layer_group,
)


def test_hex():
    assert _hex([0, 89, 255]) == "#0059ff"
    assert _hex(None) is None
    assert _hex([]) is None


def test_dash_bucket_single_pair():
    assert _dash_bucket([[5, 3]]) == "dash_5_3"
    assert _dash_bucket([(5, 3)]) == "dash_5_3"


def test_dash_bucket_dash_dot():
    # e.g. "PATTERN 10 3 0.3 3" - two nested pairs must both be flattened
    assert _dash_bucket([(10, 3), (0.3, 3)]) == "dash_10_3_0.3_3"


def test_dash_bucket_empty():
    assert _dash_bucket(None) is None
    assert _dash_bucket([]) is None


def test_spacing_bucket():
    assert _spacing_bucket(-30) == "gap_-30"
    assert _spacing_bucket(None) is None


def test_hatch_bucket_rounds_and_keeps_sign():
    assert _hatch_bucket(45.0) == "hatch_45"
    assert _hatch_bucket(-45.0) == "hatch_-45"
    assert _hatch_bucket(None) == "hatch_0"


def test_resolve_line_class_plain():
    cls = {"styles": [{"color": [0, 89, 255], "width": 1.33}]}
    row = _resolve_line_class(cls)
    assert row["_line_color"] == "#0059ff"
    assert row["_line_width"] == 1.33
    assert row["_dash_class"] is None
    assert row["_icon_name"] is None
    assert row["_icon_rotation_mode"] is None


def test_resolve_line_class_dash():
    cls = {"styles": [{"color": [255, 0, 50], "width": 1.3, "pattern": [(5, 3)]}]}
    row = _resolve_line_class(cls)
    assert row["_dash_class"] == "dash_5_3"
    assert row["_line_color"] == "#ff0032"


def test_resolve_line_class_auto_rotate_decoration():
    # background line + a decoration style with ANGLE AUTO (e.g. Gero Erosionsrand)
    cls = {
        "styles": [
            {"color": [0, 89, 255], "width": 2},
            {"symbol": "inverted_triangle", "color": [0, 89, 255], "size": 5,
             "angle": "AUTO", "gap": -10},
        ]
    }
    row = _resolve_line_class(cls)
    assert row["_line_color"] == "#0059ff"
    assert row["_icon_name"] == "inverted_triangle"
    assert row["_icon_rotation_mode"] == "map"
    assert row["_spacing_class"] == "gap_-10"


def test_resolve_line_class_fixed_angle_decoration():
    cls = {"styles": [{"symbol": "geofonts1_88", "color": [130, 54, 0], "size": 8.0, "gap": -30}]}
    row = _resolve_line_class(cls)
    assert row["_icon_rotation_mode"] == "viewport"
    assert row["_line_color"] is None  # no separate background stroke style


def test_resolve_area_class_flat():
    cls = {
        "styles": [
            {"color": [204, 221, 255], "opacity": 254},
            {"outlinecolor": [0, 89, 255], "width": 1.0},
        ]
    }
    row = _resolve_area_class(cls)
    assert row["_fill_color"] == "#ccddff"
    assert row["_fill_pattern"] is None
    assert row["_outline_color"] == "#0059ff"
    assert row["_outline_width"] == 1.0


def test_resolve_area_class_custom_pattern():
    cls = {
        "styles": [
            {"symbol": "ggla_kame", "opacity": 254},
            {"outlinecolor": [115, 76, 0], "width": 1.0},
        ]
    }
    row = _resolve_area_class(cls)
    assert row["_fill_pattern"] == "ggla_kame"
    assert row["_fill_color"] is None


def test_resolve_area_class_generic_hatch():
    cls = {
        "styles": [
            {"symbol": "hatchsymbol", "color": [130, 54, 0], "angle": 0.0, "pattern": [[5, 2]]},
            {"color": [230, 205, 193], "opacity": 254},
            {"outlinecolor": [0, 89, 255], "width": 0.53},
        ]
    }
    row = _resolve_area_class(cls)
    assert row["_fill_pattern"] == "hatch_0"
    assert row["_fill_color"] == "#e6cdc1"


def test_resolve_point_class_bohrung_circle():
    cls = {"styles": [{"symbol": "circle", "size": 8.0, "color": [200, 210, 190],
                       "outlinecolor": [0, 89, 255], "width": 1.0}]}
    row = _resolve_point_class(cls, "bohrung_fels_erreicht")
    assert row["_point_group"] == "bohrung"
    assert row["_symbol_type"] == "circle"
    assert row["_circle_color"] == "#c8d2be"
    assert row["_circle_radius"] == 8.0
    assert row["_icon_name"] is None


def test_resolve_point_class_other_icon():
    cls = {"styles": [{"symbol": "geofonts1_86", "size": 35.9, "color": [38, 127, 255]}]}
    row = _resolve_point_class(cls, "pnt_objects")
    assert row["_point_group"] == "other"
    assert row["_symbol_type"] == "icon"
    assert row["_icon_name"] == "geofonts1_86"
    assert row["_icon_color"] == "#267fff"
    assert row["_icon_size"] == 35.9


def test_resolve_point_class_bound_angle():
    cls = {"styles": [{"symbol": "geofonts1_98", "angle": "[map_angle]", "size": 16.0,
                       "color": [0, 89, 255]}]}
    row = _resolve_point_class(cls, "planar_struct")
    assert row["_icon_rotate"] == "map_angle"


def test_source_confidence_no_inc_file_is_hand_merged(tmp_path: Path):
    sub = SubLayer("inline_layer", "inline_layer.map", None)
    assert _source_confidence(tmp_path, sub) == "hand_merged"


def test_source_confidence_auto_banner(tmp_path: Path):
    inc_dir = tmp_path / "classes"
    inc_dir.mkdir()
    (inc_dir / "foo_classes.inc").write_text(
        "# Auto-generated from unknown\n# Symbol prefix: foo\n# Mode: auto\n\nCLASS\nEND\n"
    )
    sub = SubLayer("foo", "foo.map", "classes/foo_classes.inc")
    assert _source_confidence(tmp_path, sub) == "auto"


def test_source_confidence_no_banner_is_hand_merged(tmp_path: Path):
    inc_dir = tmp_path / "classes"
    inc_dir.mkdir()
    (inc_dir / "bar_classes.inc").write_text("SYMBOLSCALEDENOM 12500\nCLASS\nEND\n")
    sub = SubLayer("bar", "bar.map", "classes/bar_classes.inc")
    assert _source_confidence(tmp_path, sub) == "hand_merged"


def test_resolve_layer_group_flags_duplicate_map_symbol_across_sublayers(tmp_path: Path, monkeypatch):
    layers_dir = tmp_path
    (layers_dir / "a.map").write_text(
        'LAYER\nCLASSITEM "map_symbol"\nCLASS\nEXPRESSION "shared"\nSTYLE\nCOLOR 1 2 3\nEND\nEND\nEND\n'
    )
    (layers_dir / "b.map").write_text(
        'LAYER\nCLASSITEM "map_symbol"\nCLASS\nEXPRESSION "shared"\nSTYLE\nCOLOR 4 5 6\nEND\nEND\nEND\n'
    )
    fake_registry = {
        "tecto_lines": [
            SubLayer("a", "a.map", None),
            SubLayer("b", "b.map", None),
        ]
    }
    monkeypatch.setitem(LAYER_REGISTRY, "tecto_lines", fake_registry["tecto_lines"])
    df, issues = resolve_layer_group(layers_dir, "tecto_lines")
    assert len(df) == 2
    assert any("duplicate map_symbol" in i.reason for i in issues)


def test_resolve_layer_group_unknown_group():
    with pytest.raises(ValueError):
        resolve_layer_group(Path("."), "not_a_group")


# --- Integration: exercise the real mapserver-geocover checkout, if present ---

MAPSERVER_GEOCOVER_LAYERS = Path(__file__).parent.parent.parent / "mapserver-geocover" / "mapserver" / "layers"


@pytest.mark.integration
@pytest.mark.skipif(not MAPSERVER_GEOCOVER_LAYERS.exists(),
                     reason="sibling mapserver-geocover checkout not available")
def test_resolve_all_groups_against_real_mapfiles():
    results, issues = resolve_all_groups(MAPSERVER_GEOCOVER_LAYERS)

    assert set(results) == set(LAYER_REGISTRY)
    assert not issues, f"unexpected parsing issues: {issues}"

    for group, df in results.items():
        assert len(df) > 0, f"{group} resolved to zero classes"
        assert df["map_symbol"].is_unique
        assert df["_source_confidence"].isin(["auto", "hand_merged"]).all()

    # A handful of structural invariants from docs/maplibre-style-export.md that
    # should hold regardless of exact class counts (which drift release to release).
    lines = results["lines"]
    assert (lines["_icon_rotation_mode"] == "map").sum() > 0
    assert (lines["_icon_rotation_mode"] == "viewport").sum() > 0

    points = results["points"]
    assert set(points["_point_group"].unique()) <= {"bohrung", "other"}
    assert (points["_point_group"] == "bohrung").sum() > 0
