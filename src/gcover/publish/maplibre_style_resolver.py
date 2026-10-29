# src/gcover/publish/maplibre_style_resolver.py
"""Resolve MapServer CLASS/STYLE definitions into a flat per-`map_symbol`
attribute lookup for MapLibre style baking.

See docs/maplibre-style-export.md for the full spec this implements. In short:
`geolover-app` (the Android/MapLibre consumer) bakes resolved style as extra
columns on the source data before tiling, because MapLibre has no equivalent
of a `map_symbol` -> `CLASS`/`STYLE` lookup with thousands of entries. This
module produces that lookup, one flat table per layer group, keyed on
`map_symbol`.

Source of truth is `mapserver-geocover/mapserver/layers/` (the live-serving
mapfiles) - never `.lyrx` and never `kogis_deliverables/` (a downstream,
release-cut snapshot). See "Source of truth" in the spec doc for why.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mappyfile
import pandas as pd
from loguru import logger

AUTO_BANNER_RE = re.compile(r"^\s*#\s*Mode:\s*auto\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class SubLayer:
    """One `.map` file contributing classes to a layer group."""

    name: str
    map_file: str
    inc_file: Optional[str] = None  # relative to layers_dir; None = classes inline in .map


BOHRUNG_SUBLAYERS = {"bohrung_fels_erreicht", "bohrung_fels_nicht_erreicht"}

LAYER_REGISTRY: dict[str, list[SubLayer]] = {
    "tecto_lines": [
        SubLayer("linear_bruch", "linear_bruch.map", "classes/linear_bruch_classes.inc"),
        SubLayer("linear_ueberschiebung", "linear_ueberschiebung.map",
                 "classes/linear_ueberschiebung_classes.inc"),
    ],
    "lines": [
        SubLayer("geolkontur", "geolkontur.map", None),
        SubLayer("gesteinhorizont", "gesteinhorizont.map", "classes/gesteinhorizont_classes.inc"),
        SubLayer("linear_moraenenwall", "linear_moraenenwall.map", "classes/moraenenwall_classes.inc"),
        SubLayer("linear_objects", "linear_objects.map", "classes/linear_objects_classes.inc"),
    ],
    "points": [
        SubLayer("achsenflaeche", "achsenflaeche.map", None),
        SubLayer("bohrung_fels_erreicht", "bohrung_fels_erreicht.map", None),
        SubLayer("bohrung_fels_nicht_erreicht", "bohrung_fels_nicht_erreicht.map", None),
        SubLayer("erratiker", "erratiker.map", None),
        SubLayer("planar_struct", "planar_struct.map", None),
        SubLayer("pnt_objects", "pnt_objects.map", None),
        SubLayer("pnt_orient", "pnt_orient.map", None),
        SubLayer("quelle", "quelle.map", None),
        SubLayer("sondierung", "sondierung.map", None),
        SubLayer("exploit_geomat_pt", "exploit_geomat_pt.map", None),
        SubLayer("fossils", "fossils.map", None),
    ],
    "surfaces": [
        SubLayer("surfaces", "surfaces.map", "classes/surfaces_classes.inc"),
    ],
    "unconsolidated": [
        SubLayer("unco_bachschutt", "bachschutt.map", "classes/unco_bachschutt_classes.inc"),
        SubLayer("unco_chrono_b", "unco_chrono_b.map", "classes/unco_chrono_b_classes.inc"),
        SubLayer("unco_chrono_moraene", "unco_chrono_moraene.map", None),
        SubLayer("unco_chrono_t", "unco_chrono_t.map", "classes/unco_chrono_t_classes.inc"),
        SubLayer("unco_litho", "unco_litho.map", "classes/unco_litho_classes.inc"),
    ],
}

COLUMNS_BY_GROUP: dict[str, list[str]] = {
    "tecto_lines": ["map_symbol", "_line_color", "_line_width"],
    "lines": ["map_symbol", "_line_color", "_line_width", "_dash_class",
              "_icon_name", "_icon_color", "_spacing_class", "_icon_rotation_mode"],
    "points": ["map_symbol", "_point_group", "_symbol_type", "_circle_color", "_circle_radius",
               "_icon_name", "_icon_color", "_icon_size", "_icon_rotate"],
    "surfaces": ["map_symbol", "_fill_color", "_fill_pattern", "_outline_color", "_outline_width"],
    "unconsolidated": ["map_symbol", "_fill_color", "_fill_pattern", "_outline_color", "_outline_width"],
}

EXTRA_COLUMNS = ["_source_confidence", "_sublayer"]


@dataclass
class ClassIssue:
    """A class/file this tool could not confidently resolve - never silently dropped."""

    group: str
    sublayer: str
    map_symbol: str
    reason: str


class _cwd:
    """Temporarily change the process working directory.

    `mappyfile` resolves relative `INCLUDE` paths against the *process cwd*
    when no `fn` is supplied to `loads()`, which matches how these mapfiles
    are actually deployed: their `INCLUDE "layers/classes/*.inc"` paths are
    written relative to the mapserver root, not to the including file's own
    directory (that's also why `mappyfile.open()` can't be pointed at a layer
    file directly - see docs/maplibre-style-export.md's pipeline section).
    """

    def __init__(self, path: Path):
        self._target = path
        self._previous: Optional[str] = None

    def __enter__(self):
        self._previous = os.getcwd()
        os.chdir(self._target)

    def __exit__(self, *exc):
        os.chdir(self._previous)


def load_layer_classes(layers_dir: Path, map_file: str) -> list[dict]:
    """Parse one layer `.map` file's `CLASS` blocks, resolving `INCLUDE`s."""
    mapserver_root = layers_dir.parent
    text = (layers_dir / map_file).read_text(encoding="utf-8")
    with _cwd(mapserver_root):
        parsed = mappyfile.loads(text, expand_includes=True)
    return parsed.get("classes", [])


def _source_confidence(layers_dir: Path, sublayer: SubLayer) -> str:
    """'auto' if the sublayer's `.inc` still carries the generator's auto banner,
    'hand_merged' otherwise (including sublayers with classes inline in the `.map`,
    which by convention are never auto-generated)."""
    if sublayer.inc_file is None:
        return "hand_merged"
    text = (layers_dir / sublayer.inc_file).read_text(encoding="utf-8")
    header = "\n".join(text.splitlines()[:10])
    return "auto" if AUTO_BANNER_RE.search(header) else "hand_merged"


def _hex(rgb: Optional[list]) -> Optional[str]:
    if not rgb:
        return None
    r, g, b = rgb[0], rgb[1], rgb[2]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _fmt_num(v: Any) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def _dash_bucket(pattern: Any) -> Optional[str]:
    if not pattern:
        return None
    values: list = []
    for item in pattern:
        if isinstance(item, (list, tuple)):
            values.extend(item)
        else:
            values.append(item)
    return "dash_" + "_".join(_fmt_num(v) for v in values)


def _spacing_bucket(gap: Any) -> Optional[str]:
    if gap is None:
        return None
    return f"gap_{_fmt_num(gap)}"


def _hatch_bucket(angle: Any) -> str:
    return f"hatch_{int(round(float(angle if angle is not None else 0)))}"


def _resolve_line_class(cls: dict) -> dict:
    styles = cls.get("styles", [])
    plain = next((s for s in styles if "symbol" not in s), None)
    dash_style = next((s for s in styles if s.get("pattern") and "symbol" not in s), None)
    deco_style = next((s for s in styles if "symbol" in s), None)

    return {
        "_line_color": _hex(plain.get("color")) if plain else None,
        "_line_width": plain.get("width") if plain else None,
        "_dash_class": _dash_bucket(dash_style.get("pattern")) if dash_style else None,
        "_icon_name": deco_style.get("symbol") if deco_style else None,
        "_icon_color": _hex(deco_style.get("color")) if deco_style else None,
        "_spacing_class": _spacing_bucket(deco_style.get("gap")) if deco_style else None,
        "_icon_rotation_mode": (
            ("map" if deco_style.get("angle") == "AUTO" else "viewport") if deco_style else None
        ),
    }


def _resolve_area_class(cls: dict) -> dict:
    styles = cls.get("styles", [])
    outline = next((s for s in styles if "outlinecolor" in s), None)
    hatch = next((s for s in styles if s.get("symbol") == "hatchsymbol"), None)
    fill_plain = next((s for s in styles if "color" in s and "symbol" not in s), None)
    fill_symbol = next((s for s in styles if s.get("symbol") not in (None, "hatchsymbol")), None)

    if hatch is not None:
        fill_pattern = _hatch_bucket(hatch.get("angle"))
    elif fill_symbol is not None:
        fill_pattern = fill_symbol.get("symbol")
    else:
        fill_pattern = None

    return {
        "_fill_color": _hex(fill_plain.get("color")) if fill_plain else None,
        "_fill_pattern": fill_pattern,
        "_outline_color": _hex(outline.get("outlinecolor")) if outline else None,
        "_outline_width": outline.get("width") if outline else None,
    }


def _resolve_point_class(cls: dict, sublayer_name: str) -> dict:
    styles = cls.get("styles", [])
    style = styles[0] if styles else {}
    is_circle = style.get("symbol") == "circle"

    angle = style.get("angle")
    icon_rotate = None
    if isinstance(angle, str) and angle.startswith("[") and angle.endswith("]"):
        icon_rotate = angle[1:-1]

    return {
        "_point_group": "bohrung" if sublayer_name in BOHRUNG_SUBLAYERS else "other",
        "_symbol_type": "circle" if is_circle else "icon",
        "_circle_color": _hex(style.get("color")) if is_circle else None,
        "_circle_radius": style.get("size") if is_circle else None,
        "_icon_name": None if is_circle else style.get("symbol"),
        "_icon_color": None if is_circle else _hex(style.get("color")),
        "_icon_size": None if is_circle else style.get("size"),
        "_icon_rotate": icon_rotate,
    }


def _resolve_class(group: str, cls: dict, sublayer_name: str) -> dict:
    if group == "points":
        return _resolve_point_class(cls, sublayer_name)
    if group in ("surfaces", "unconsolidated"):
        return _resolve_area_class(cls)
    return _resolve_line_class(cls)


def resolve_layer_group(layers_dir: Path, group: str) -> tuple[pd.DataFrame, list[ClassIssue]]:
    """Resolve one layer group (e.g. "lines") into a flat, `map_symbol`-keyed table.

    Returns the table plus any issues encountered (unparseable files, classes with
    no EXPRESSION, or a `map_symbol` defined in more than one sublayer) - these are
    always reported, never silently dropped or papered over with a default.
    """
    if group not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer group: {group!r}. Known: {sorted(LAYER_REGISTRY)}")

    rows: list[dict] = []
    issues: list[ClassIssue] = []
    first_seen_in: dict[str, str] = {}

    for sub in LAYER_REGISTRY[group]:
        try:
            classes = load_layer_classes(layers_dir, sub.map_file)
        except Exception as exc:
            logger.error(f"[{group}/{sub.name}] failed to parse {sub.map_file}: {exc}")
            issues.append(ClassIssue(group, sub.name, "*", f"parse error: {exc}"))
            continue

        confidence = _source_confidence(layers_dir, sub)

        for cls in classes:
            map_symbol = cls.get("expression")
            if not map_symbol or not isinstance(map_symbol, str):
                issues.append(ClassIssue(
                    group, sub.name, "?",
                    f"class {cls.get('name')!r} has no usable EXPRESSION (got {map_symbol!r})",
                ))
                continue

            if map_symbol in first_seen_in:
                issues.append(ClassIssue(
                    group, sub.name, map_symbol,
                    f"duplicate map_symbol, also defined in sublayer {first_seen_in[map_symbol]!r}",
                ))
            else:
                first_seen_in[map_symbol] = sub.name

            try:
                attrs = _resolve_class(group, cls, sub.name)
            except Exception as exc:
                issues.append(ClassIssue(group, sub.name, map_symbol, f"resolve error: {exc}"))
                continue

            row = {"map_symbol": map_symbol, **attrs,
                   "_source_confidence": confidence, "_sublayer": sub.name}
            rows.append(row)

    columns = COLUMNS_BY_GROUP[group] + EXTRA_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    return df, issues


def resolve_all_groups(
    layers_dir: Path, groups: Optional[list[str]] = None
) -> tuple[dict[str, pd.DataFrame], list[ClassIssue]]:
    """Resolve every requested layer group. Defaults to all of them."""
    groups = list(groups) if groups else list(LAYER_REGISTRY)
    results: dict[str, pd.DataFrame] = {}
    all_issues: list[ClassIssue] = []
    for group in groups:
        df, issues = resolve_layer_group(layers_dir, group)
        results[group] = df
        all_issues.extend(issues)
    return results, all_issues
