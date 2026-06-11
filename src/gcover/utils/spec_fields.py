# src/gcover/utils/spec_fields.py
"""
Compute spec_de / spec_fr composite fields for GeoCover tooltip layers.

Pre-computes the "Spezifizierung" display string consumed by mf-chsdi3
htmlPopup templates, mimicking old SPEC_DESCR_D/F from geocover_tooltips.gdb.

Applies to: linear_objects, point_objects, surfaces, fossils, exploit_points.
Does NOT apply to: bedrock, unco_deposits (structured multi-column tooltips).
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

_NA: frozenset = frozenset({"not applicable", "unbekannt", "N/A", ""})

# Static label prefix strings keyed by language suffix
_DEPTH_BEDR_LABEL = {"_de": "Tiefe Fels", "_fr": "Profondeur du roc"}


def _val(v) -> str | None:
    """Return v as a displayable string, or None for no-data sentinels."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return None if s in _NA else s or None


def _join(*parts) -> str | None:
    """Comma-join non-empty display values; return None if nothing survives."""
    items = [p for p in (_val(p) for p in parts) if p]
    return ", ".join(items) if items else None


# ── Per-table compute functions ───────────────────────────────────────────────

def _spec_linear(row: pd.Series, suf: str) -> str | None:
    kind = row.get("kind")

    if kind in {14901004, 14901001, 14901002, 14901005, 14901007, 14901008}:
        return _val(row.get(f"ttec_status{suf}"))

    if kind == 13101001:
        return _join(row.get(f"lpro_litho{suf}"), row.get(f"lpro_cong_spe{suf}"))

    if kind in {11301001, 11301002}:
        return _join(row.get(f"ggla_morai_mo{suf}"), row.get(f"ggla_glac_typ{suf}"))

    if kind in {10701001, 10701002}:
        v = _val(row.get(f"aexp_status{suf}"))
        return f"Status: {v}" if v else None

    if kind == 13001001:
        v = _val(row.get(f"lgeo_status{suf}"))
        return f"Status: {v}" if v else None

    if kind == 13901001:
        try:
            a = float(row.get("pcob_altitude"))
            return f"Altitude (m): {int(a)}" if a < 999990 else None
        except (TypeError, ValueError):
            return None

    if kind == 14001001:
        try:
            a = float(row.get("pcoh_altitude"))
            return f"Altitude (m): {int(a)}" if a < 999990 else None
        except (TypeError, ValueError):
            return None

    if kind in {10201001, 10201002, 10201003, 10201004, 10201005, 10201006, 10201007}:
        return _join(row.get(f"aarc_epoch{suf}"), row.get(f"aarc_period{suf}"))

    return None


def _spec_point(row: pd.Series, suf: str) -> str | None:
    kind = row.get("kind")

    if kind in {12501001, 12501002}:
        return _join(
            row.get(f"hsur_type{suf}"),
            row.get(f"hsur_status{suf}"),
            row.get(f"hsur_flow_con{suf}"),
        )
    if kind == 12501003:
        return _join(row.get(f"hsur_status{suf}"), row.get(f"hsur_flow_con{suf}"))
    if kind == 12501006:
        return _val(row.get(f"hsur_status{suf}"))

    if kind in {12101001, 12101002}:
        v = _val(row.get(f"hcon_status{suf}"))
        return f"Status: {v}" if v else None

    if kind == 14101002:
        h = row.get("pmod_height")
        if h is not None and not (isinstance(h, float) and pd.isna(h)):
            return f"Altitude (m): {h}"
        return None

    if kind == 13801001:
        return _val(row.get(f"mpla_polarity{suf}"))

    if kind in {14401001, 14401002, 14401003, 14401004, 14401005, 14401006, 14401007, 14401008}:
        return _join(
            row.get(f"runc_rock_typ{suf}"),
            row.get(f"runc_rock_spe{suf}"),
            row.get(f"runc_status{suf}"),
        )

    if kind in {10501001, 10501002, 10501003, 10501004}:
        fm_a = _val(row.get(f"abor_fm_a{suf}"))
        if fm_a:
            return fm_a
        if kind == 10501001:
            try:
                d = float(row.get("abor_depth_bedr"))
                label = _DEPTH_BEDR_LABEL.get(suf, "Tiefe Fels")
                return f"{label}: {int(d)} m" if d < 888 else None
            except (TypeError, ValueError):
                return None
        return None

    if kind in {13501003, 13501001}:
        return _join(row.get(f"ltyp_strati{suf}"), row.get("ltyp_name"))

    if kind in {
        10101001, 10101002, 10101003, 10101004, 10101005,
        10101006, 10101007, 10101008, 10101009, 10101010,
        10101011, 10101012, 10101013,
    }:
        return _join(row.get(f"aarc_epoch{suf}"), row.get(f"aarc_period{suf}"))

    if kind in {13201002, 13201003}:
        v = _val(row.get(f"lres_status{suf}"))
        return f"Status: {v}" if v else None
    if kind in {13201001, 13201004}:
        v = _val(row.get(f"lres_material{suf}"))
        return f"Material: {v}" if v else None

    return None


def _spec_surface(row: pd.Series, suf: str) -> str | None:
    kind = row.get("kind")

    if kind == 14801002:
        v = _val(row.get(f"tdef_type{suf}"))
        return f"Type: {v}" if v else None
    if kind == 15801001:
        return _val(row.get(f"gins_main_mov{suf}"))
    if kind in {10301001, 10301002}:
        return _join(row.get(f"aarc_epoch{suf}"), row.get(f"aarc_period{suf}"))

    return None


def _spec_fossil(row: pd.Series, suf: str) -> str | None:
    return _join(row.get("system_desc"), row.get(f"lfos_division{suf}"))


def _spec_exploit(row: pd.Series, suf: str) -> str | None:
    kind = row.get("kind")

    if kind in {10601001, 10601002, 10601003}:
        v = _val(row.get(f"aexp_status{suf}"))
        return f"Status: {v}" if v else None

    return None


# ── Layer dispatch ────────────────────────────────────────────────────────────

_COMPUTE_FN = {
    "linear_objects":  _spec_linear,
    "point_objects":   _spec_point,
    "surfaces":        _spec_surface,
    "fossils":         _spec_fossil,
    "exploit_points":  _spec_exploit,
}


def add_spec_fields(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """Add spec_de and spec_fr columns to gdf for the five supported layers.

    Returns gdf unchanged for layers not in the supported set (bedrock,
    unco_deposits, exploit_polygons, …).  Columns are computed row-by-row
    from the translated _de/_fr columns already present in gdf.
    """
    fn = _COMPUTE_FN.get(layer_name)
    if fn is None:
        return gdf

    gdf = gdf.copy()
    gdf["spec_de"] = gdf.apply(lambda row: fn(row, "_de"), axis=1)
    gdf["spec_fr"] = gdf.apply(lambda row: fn(row, "_fr"), axis=1)
    return gdf
