"""Static QGIS layer styles for QA output GPKG files.

Writes a ``layer_styles`` table into the GPKG so QGIS automatically applies
the correct symbology when the file is opened.  ArcGIS ignores this table.

Supported layers
----------------
IssuePolygons / IssueLines / IssuePoints  — categorised by ``IssueType``
mapsheets_sources_only                    — categorised by ``SOURCE_RC``
qa_rand_gc_buffer_50m                     — single semi-transparent fill
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Colour palette  (R, G, B, A)
# ---------------------------------------------------------------------------
_RED          = (213,   0,   0, 180)
_RED_OUT      = (150,   0,   0, 255)
_ORANGE       = (255, 140,   0, 180)
_ORANGE_OUT   = (180, 100,   0, 255)
_GREY         = (160, 160, 160, 120)
_GREY_OUT     = (100, 100, 100, 255)

_BLUE         = ( 31, 120, 180, 160)
_BLUE_OUT     = ( 10,  80, 130, 255)
_GREEN        = ( 51, 160,  44, 160)
_GREEN_OUT    = ( 20, 110,  20, 255)

_YELLOW_GHOST = (255, 200,   0,  35)
_YELLOW_OUT   = (200, 140,   0, 200)


# ---------------------------------------------------------------------------
# Low-level symbol builders
# ---------------------------------------------------------------------------

def _c(rgba: Tuple[int, int, int, int]) -> str:
    return "{},{},{},{}".format(*rgba)


def _fill_sym(idx: int, fill: Tuple, outline: Tuple, style: str = "solid") -> str:
    return (
        f'      <symbol name="{idx}" type="fill" alpha="1" clip_to_extent="1" force_rhr="0">\n'
        f'        <layer class="SimpleFill" pass="0" locked="0" enabled="1">\n'
        f'          <Option type="Map">\n'
        f'            <Option name="border_width_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'            <Option name="color" value="{_c(fill)}" type="QString"/>\n'
        f'            <Option name="joinstyle" value="miter" type="QString"/>\n'
        f'            <Option name="offset" value="0,0" type="QString"/>\n'
        f'            <Option name="offset_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'            <Option name="offset_unit" value="MM" type="QString"/>\n'
        f'            <Option name="outline_color" value="{_c(outline)}" type="QString"/>\n'
        f'            <Option name="outline_style" value="solid" type="QString"/>\n'
        f'            <Option name="outline_width" value="0.26" type="QString"/>\n'
        f'            <Option name="outline_width_unit" value="MM" type="QString"/>\n'
        f'            <Option name="style" value="{style}" type="QString"/>\n'
        f'          </Option>\n'
        f'        </layer>\n'
        f'      </symbol>'
    )


def _line_sym(idx: int, color: Tuple, width: float = 0.5) -> str:
    return (
        f'      <symbol name="{idx}" type="line" alpha="1" clip_to_extent="1" force_rhr="0">\n'
        f'        <layer class="SimpleLine" pass="0" locked="0" enabled="1">\n'
        f'          <Option type="Map">\n'
        f'            <Option name="capstyle" value="square" type="QString"/>\n'
        f'            <Option name="joinstyle" value="bevel" type="QString"/>\n'
        f'            <Option name="line_color" value="{_c(color)}" type="QString"/>\n'
        f'            <Option name="line_style" value="solid" type="QString"/>\n'
        f'            <Option name="line_width" value="{width}" type="QString"/>\n'
        f'            <Option name="line_width_unit" value="MM" type="QString"/>\n'
        f'            <Option name="offset" value="0" type="QString"/>\n'
        f'            <Option name="offset_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'            <Option name="offset_unit" value="MM" type="QString"/>\n'
        f'            <Option name="ring_filter" value="0" type="QString"/>\n'
        f'            <Option name="use_custom_dash" value="0" type="QString"/>\n'
        f'            <Option name="width_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'          </Option>\n'
        f'        </layer>\n'
        f'      </symbol>'
    )


def _marker_sym(idx: int, color: Tuple, size: float = 3.0) -> str:
    return (
        f'      <symbol name="{idx}" type="marker" alpha="1" clip_to_extent="1" force_rhr="0">\n'
        f'        <layer class="SimpleMarker" pass="0" locked="0" enabled="1">\n'
        f'          <Option type="Map">\n'
        f'            <Option name="angle" value="0" type="QString"/>\n'
        f'            <Option name="color" value="{_c(color)}" type="QString"/>\n'
        f'            <Option name="horizontal_anchor_point" value="1" type="QString"/>\n'
        f'            <Option name="joinstyle" value="bevel" type="QString"/>\n'
        f'            <Option name="name" value="circle" type="QString"/>\n'
        f'            <Option name="offset" value="0,0" type="QString"/>\n'
        f'            <Option name="offset_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'            <Option name="offset_unit" value="MM" type="QString"/>\n'
        f'            <Option name="outline_color" value="50,50,50,255" type="QString"/>\n'
        f'            <Option name="outline_style" value="solid" type="QString"/>\n'
        f'            <Option name="outline_width" value="0" type="QString"/>\n'
        f'            <Option name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'            <Option name="outline_width_unit" value="MM" type="QString"/>\n'
        f'            <Option name="scale_method" value="diameter" type="QString"/>\n'
        f'            <Option name="size" value="{size}" type="QString"/>\n'
        f'            <Option name="size_map_unit_scale" value="3x:0,0,0,0,0,0" type="QString"/>\n'
        f'            <Option name="size_unit" value="MM" type="QString"/>\n'
        f'            <Option name="vertical_anchor_point" value="1" type="QString"/>\n'
        f'          </Option>\n'
        f'        </layer>\n'
        f'      </symbol>'
    )


# ---------------------------------------------------------------------------
# QML renderer builders
# ---------------------------------------------------------------------------

# category spec: (value, label, symbol_builder_kwargs...)
# For the three layer geometry types we build all variants upfront.

_ISSUE_CATS = [
    ("Error",   "Error",   _RED,    _RED_OUT),
    ("Warning", "Warning", _ORANGE, _ORANGE_OUT),
    ("",        "(other)", _GREY,   _GREY_OUT),
]

_MAPSHEET_CATS = [
    ("RC1", "RC1", _BLUE,  _BLUE_OUT),
    ("RC2", "RC2", _GREEN, _GREEN_OUT),
    ("",    "(other)", _GREY, _GREY_OUT),
]


def _categorized_qml(attr: str, sym_type: str, cats: list) -> str:
    """Build a complete categorized-renderer QML string.

    sym_type: 'fill' | 'line' | 'marker'
    cats: list of (value, label, primary_color, secondary_color)
    """
    cat_lines = "\n".join(
        f'      <category symbol="{i}" value="{v}" label="{lbl}" render="true" type="string"/>'
        for i, (v, lbl, *_) in enumerate(cats)
    )
    if sym_type == "fill":
        sym_lines = "\n".join(
            _fill_sym(i, fill, outline)
            for i, (_, __, fill, outline) in enumerate(cats)
        )
    elif sym_type == "line":
        sym_lines = "\n".join(
            _line_sym(i, color)
            for i, (_, __, color, *___) in enumerate(cats)
        )
    else:  # marker
        sym_lines = "\n".join(
            _marker_sym(i, color)
            for i, (_, __, color, *___) in enumerate(cats)
        )

    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.0.0" styleCategories="Symbology">\n'
        f'  <renderer-v2 type="categorizedSymbol" attr="{attr}" forceraster="0"'
        ' enableorderby="0" symbollevels="0" referencescale="-1">\n'
        "    <categories>\n"
        f"{cat_lines}\n"
        "    </categories>\n"
        "    <symbols>\n"
        f"{sym_lines}\n"
        "    </symbols>\n"
        "    <rotation/>\n"
        "    <sizescale/>\n"
        "  </renderer-v2>\n"
        "  <blendMode>0</blendMode>\n"
        "  <featureBlendMode>0</featureBlendMode>\n"
        "  <layerOpacity>1</layerOpacity>\n"
        "</qgis>"
    )


def _single_symbol_qml(sym_type: str, fill: Tuple, outline: Tuple) -> str:
    """Build a single-symbol QML string (used for the rand-buffer layer)."""
    if sym_type == "fill":
        sym = _fill_sym(0, fill, outline)
    elif sym_type == "line":
        sym = _line_sym(0, fill)
    else:
        sym = _marker_sym(0, fill)

    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.0.0" styleCategories="Symbology">\n'
        '  <renderer-v2 type="singleSymbol" forceraster="0" referencescale="-1">\n'
        "    <symbols>\n"
        f"{sym}\n"
        "    </symbols>\n"
        "    <rotation/>\n"
        "    <sizescale/>\n"
        "  </renderer-v2>\n"
        "  <blendMode>0</blendMode>\n"
        "  <featureBlendMode>0</featureBlendMode>\n"
        "  <layerOpacity>1</layerOpacity>\n"
        "</qgis>"
    )


# ---------------------------------------------------------------------------
# Pre-built styles  { layer_name → QML string }
# ---------------------------------------------------------------------------

LAYER_STYLES: Dict[str, str] = {
    "IssuePolygons": _categorized_qml("IssueType", "fill",   _ISSUE_CATS),
    "IssueLines":    _categorized_qml("IssueType", "line",   _ISSUE_CATS),
    "IssuePoints":   _categorized_qml("IssueType", "marker", _ISSUE_CATS),
    "mapsheets_sources_only": _categorized_qml("SOURCE_RC", "fill", _MAPSHEET_CATS),
    "qa_rand_gc_buffer_50m":  _single_symbol_qml("fill", _YELLOW_GHOST, _YELLOW_OUT),
}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS layer_styles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    f_table_catalog   TEXT(256),
    f_table_schema    TEXT(256),
    f_table_name      TEXT(256),
    f_geometry_column TEXT(256),
    styleName         TEXT(30),
    styleQML          TEXT,
    styleSLD          TEXT,
    useAsDefault      BOOLEAN,
    description       TEXT(200),
    owner             TEXT(30),
    ui                TEXT(30),
    update_datetime   DATETIME
)
"""

_UPSERT = """
INSERT INTO layer_styles
    (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,
     styleName, styleQML, styleSLD, useAsDefault, description, owner, ui, update_datetime)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(f_table_name, f_geometry_column, styleName)
DO UPDATE SET styleQML=excluded.styleQML, update_datetime=excluded.update_datetime
"""

# Fallback insert when the table has no UNIQUE constraint (first-time population)
_INSERT = """
INSERT OR REPLACE INTO layer_styles
    (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,
     styleName, styleQML, styleSLD, useAsDefault, description, owner, ui, update_datetime)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def write_layer_styles(gpkg_path: Path) -> None:
    """Embed QGIS styles into *gpkg_path* for all known QA layers.

    Creates the ``layer_styles`` table if absent, then inserts/replaces a row
    for every layer in :data:`LAYER_STYLES` that is present in the file.
    Layers with no matching style are silently skipped.
    """
    if not gpkg_path.exists():
        logger.warning(f"GPKG not found, skipping style injection: {gpkg_path}")
        return

    import fiona

    try:
        available = set(fiona.listlayers(str(gpkg_path)))
    except Exception as e:
        logger.warning(f"Could not list GPKG layers for styling: {e}")
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    with sqlite3.connect(gpkg_path) as con:
        con.execute(_CREATE_TABLE)

        # Add a unique index if not present (makes upsert reliable)
        try:
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS layer_styles_unique "
                "ON layer_styles (f_table_name, f_geometry_column, styleName)"
            )
        except sqlite3.OperationalError:
            pass  # index already exists or table structure differs — harmless

        written = 0
        for layer_name, qml in LAYER_STYLES.items():
            if layer_name not in available:
                continue
            # Use the geometry column name actually registered in the GPKG metadata
            # (geopandas/GDAL writes 'geom', not 'geometry').
            try:
                r = con.execute(
                    "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
                    (layer_name,),
                ).fetchone()
                geom_col = r[0] if r else "geom"
            except sqlite3.OperationalError:
                geom_col = "geom"
            row = ("", "", layer_name, geom_col, "default", qml, None, 1, "", "", None, now)
            try:
                con.execute(_UPSERT, row)
            except sqlite3.OperationalError:
                con.execute(_INSERT, row)
            written += 1

        con.commit()

    if written:
        logger.info(f"Embedded QGIS styles for {written} layer(s) in {gpkg_path.name}")
    else:
        logger.debug(f"No matching layers for style injection in {gpkg_path.name}")
