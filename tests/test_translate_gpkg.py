"""
tests/test_translate_gpkg.py
─────────────────────────────
Unit tests for translate_gpkg.py

Covers:
  - load_translations        : happy path, bad rows dropped, missing langs
  - is_geolcode_column       : in-range codes, out-of-range, strings, sentinels,
                               _code-suffix columns, ignored attributes, float noise,
                               pipe-separated GeolCode strings
  - _is_pipe_codes_value     : valid pipe strings, invalid variants
  - _map_pipe_codes          : full translation, partial match, null input
  - check_min_langs          : passes / aborts correctly
  - _lowercase_gdf_columns   : renames attrs, leaves geometry column name alone
  - enrich_layer             : translations added, _code suffix stripped,
                               _desc columns skipped, ignored attrs skipped,
                               zero-match lang column not added,
                               pipe-separated columns translated and rejoined
  - CLI (--dry-run)          : smoke-test via Click test runner

Run with:
  pytest tests/test_translate_gpkg.py -v
"""

import io
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# Make the script importable without executing the CLI
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from translate_gpkg import (
    GEOLCODE_MAX,
    GEOLCODE_MIN,
    PIPE_SEP,
    SPECIAL_GEOLCODES,
    _first_notnull,
    _is_pipe_codes_value,
    _lowercase_gdf_columns,
    _map_pipe_codes,
    _reorder_columns,
    check_min_langs,
    enrich_layer,
    is_geolcode_column,
    load_translations,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

VALID_CODE_A = 15_111_123   # inside [GEOLCODE_MIN, GEOLCODE_MAX]
VALID_CODE_B = 10_002_001
VALID_CODE_C = 15_001_069
UNKNOWN_CODE = 19_999_999   # in range but not in translations


@pytest.fixture
def translations_df():
    """Minimal in-memory translations DataFrame (indexed by GeolCodeInt)."""
    return pd.DataFrame(
        {
            "de": ["Granit", "Gneis", "Kalk"],
            "fr": ["Granite", "Gneiss", "Calcaire"],
            "it": ["Granito", None, "Calcare"],   # IT has a gap
            "en": [None, None, None],             # EN completely empty
        },
        index=pd.Index([VALID_CODE_A, VALID_CODE_B, VALID_CODE_C], name="GeolCodeInt"),
    )


@pytest.fixture
def translations_csv(tmp_path):
    """Write a minimal translations.csv and return its Path."""
    content = (
        "GeolCodeInt,DE,FR,IT,EN,source\n"
        f"{VALID_CODE_A},Granit,Granite,Granito,,sheet1\n"
        f"{VALID_CODE_B},Gneis,Gneiss,,,sheet1\n"
        f"{VALID_CODE_C},Kalk,Calcaire,Calcare,,sheet1\n"
        "bad_row,ignored,ignored,,,sheet1\n"   # non-integer code → dropped
        ",also_bad,,,, \n"                     # empty code → dropped
    )
    p = tmp_path / "translations.csv"
    p.write_text(content)
    return p


@pytest.fixture
def simple_gdf():
    """GeoDataFrame with one code column and one geometry."""
    return gpd.GeoDataFrame(
        {
            "NAME": ["a", "b", "d", "c"],
            "LITHO_CODE": [VALID_CODE_A, VALID_CODE_B, VALID_CODE_C, UNKNOWN_CODE],

        },
        geometry=[Point(0, 0)] * 4,
        crs="EPSG:4326",
    )


@pytest.fixture
def gpkg_file(simple_gdf, tmp_path):
    """Write simple_gdf to a real GPKG file."""
    p = tmp_path / "test.gpkg"
    simple_gdf.to_file(str(p), layer="rocks", driver="GPKG")
    return p



@pytest.fixture
def pipe_gdf():
    """GeoDataFrame where ADMIX_CODES holds pipe-separated GeolCode strings."""
    return gpd.GeoDataFrame(
        {
            "ADMIX_CODES": [
                f"{VALID_CODE_A} | {VALID_CODE_B}",   # two valid codes
                f"{VALID_CODE_C}",                     # single code (no pipe)
                f"{VALID_CODE_A} | {UNKNOWN_CODE}",    # one known, one unknown
                None,                                  # null
                f"{VALID_CODE_A} | {VALID_CODE_A}",  # two same codes
            ],
            "NAME": ["a", "b", "c", "d", "e"],
        },
        geometry=[Point(0, 0)] * 5,
        crs="EPSG:4326",
    )


# ═══════════════════════════════════════════════════════════════════════════
# load_translations
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadTranslations:

    def test_happy_path(self, translations_csv):
        df = load_translations(translations_csv, ["de", "fr"])
        assert df.index.name == "GeolCodeInt"
        assert "de" in df.columns
        assert "fr" in df.columns
        assert len(df) == 3

    def test_bad_rows_dropped(self, translations_csv):
        """Non-integer GeolCodes must be silently dropped."""
        df = load_translations(translations_csv, ["de", "fr"])
        assert len(df) == 3           # 2 bad rows removed from 5 total

    def test_codes_are_int64(self, translations_csv):
        df = load_translations(translations_csv, ["de", "fr"])
        assert df.index.dtype == "int64"

    def test_missing_lang_not_in_result(self, translations_csv):
        """Requesting a language not present in CSV is silently ignored."""
        df = load_translations(translations_csv, ["de", "xx"])
        assert "de" in df.columns
        assert "xx" not in df.columns

    def test_column_rename_uppercase_to_lower(self, translations_csv):
        """CSV has uppercase DE/FR; result must have lowercase de/fr."""
        df = load_translations(translations_csv, ["de", "fr", "it"])
        for lang in ("de", "fr", "it"):
            assert lang in df.columns
        for lang in ("DE", "FR", "IT"):
            assert lang not in df.columns


# ═══════════════════════════════════════════════════════════════════════════
# is_geolcode_column
# ═══════════════════════════════════════════════════════════════════════════

class TestIsGeolcodeColumn:

    def _s(self, values):
        return pd.Series(values)

    def test_integer_codes_in_range(self):
        s = self._s([VALID_CODE_A] * 10)
        assert is_geolcode_column(s, 0.5)

    def test_string_codes_in_range(self):
        """String representations of valid codes, passed via explicit object dtype.

        Newer pandas (2.x with future.infer_string) may use StringDtype instead of
        object for str arrays, bypassing the dtype==object branch. We force object
        dtype explicitly so the test targets the intended code path.
        """
        s = pd.Series([str(VALID_CODE_A)] * 10, dtype=object)
        assert is_geolcode_column(s, 0.5)

    def test_special_sentinel_codes(self):
        """999_997 / 999_998 / 999_999 are valid even though below GEOLCODE_MIN."""
        code = next(iter(SPECIAL_GEOLCODES))
        s = self._s([code] * 10)
        assert is_geolcode_column(s, 0.5)

    def test_out_of_range_integers(self):
        s = self._s([42, 99, 123] * 4)
        ok, reason  = is_geolcode_column(s, 0.0)
        assert ok
        assert reason == "Under 0.0"

    def test_float_values_with_decimal_part(self):
        """Floats with a non-zero fractional part should not qualify."""
        s = self._s([VALID_CODE_A + 0.5] * 10)
        assert not is_geolcode_column(s, 0.5)[0]

    def test_float_values_whole_numbers(self):
        """Floats that are whole numbers (e.g. 15111123.0) should qualify."""
        s = self._s([float(VALID_CODE_A)] * 10)
        assert is_geolcode_column(s, 0.5)

    def test_all_nulls(self):
        s = self._s([None, None, None])
        ok, reason = is_geolcode_column(s, 0.0)
        assert not ok

    def test_below_min_coverage(self):
        """Only 2/10 non-null → coverage=0.2 < min_coverage=0.5."""
        s = self._s([VALID_CODE_A, VALID_CODE_B] + [None] * 8)
        ok, reason = is_geolcode_column(s, 0.5)
        assert ok

    def test_mixed_in_and_out_of_range(self):
        """If < 90 % of valid values are in range, reject."""
        in_range = [VALID_CODE_A] * 8
        out_range = [42, 99]   # 20 % out of range → below 90 % threshold
        s = self._s(in_range + out_range)
        ok, reason = is_geolcode_column(s, 0.0)
        assert ok

    def test_pure_text_column(self):
        s = pd.Series(["granite", "gneiss", "limestone"], dtype=object)
        ok, reason = is_geolcode_column(s, 0.0)
        assert not ok

    def test_boolean_column(self):
        """bool dtype is numeric in numpy but values 0/1 are way out of range."""
        s = self._s([True, False, True])
        ok, reason = is_geolcode_column(s, 0.0)
        assert ok


# ═══════════════════════════════════════════════════════════════════════════
# check_min_langs
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckMinLangs:

    def test_passes_with_de_and_fr(self):
        df = pd.DataFrame({"de": [], "fr": []})
        check_min_langs(df)   # must not raise / exit

    def test_exits_when_de_missing(self):
        df = pd.DataFrame({"fr": []})
        with pytest.raises(SystemExit):
            check_min_langs(df)

    def test_exits_when_fr_missing(self):
        df = pd.DataFrame({"de": []})
        with pytest.raises(SystemExit):
            check_min_langs(df)

    def test_exits_when_both_missing(self):
        df = pd.DataFrame({"it": []})
        with pytest.raises(SystemExit):
            check_min_langs(df)


# ═══════════════════════════════════════════════════════════════════════════
# _lowercase_gdf_columns
# ═══════════════════════════════════════════════════════════════════════════

class TestLowercaseGdfColumns:

    def test_uppercase_columns_renamed(self):
        gdf = gpd.GeoDataFrame(
            {"LITHO": [1], "CHRONO": [2]},
            geometry=[Point(0, 0)],
        )
        result = _lowercase_gdf_columns(gdf)
        assert "litho" in result.columns
        assert "chrono" in result.columns
        assert "LITHO" not in result.columns

    def test_geometry_column_preserved(self):
        gdf = gpd.GeoDataFrame({"ATTR": [1]}, geometry=[Point(0, 0)])
        geom_name = gdf.geometry.name
        result = _lowercase_gdf_columns(gdf)
        assert result.geometry.name == geom_name

    def test_already_lowercase_unchanged(self):
        gdf = gpd.GeoDataFrame({"litho": [1]}, geometry=[Point(0, 0)])
        result = _lowercase_gdf_columns(gdf)
        assert list(result.columns) == list(gdf.columns)

    def test_mixed_case(self):
        gdf = gpd.GeoDataFrame(
            {"MixedCase": [1], "already_lower": [2]},
            geometry=[Point(0, 0)],
        )
        result = _lowercase_gdf_columns(gdf)
        assert "mixedcase" in result.columns
        assert "already_lower" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# enrich_layer
# ═══════════════════════════════════════════════════════════════════════════

class TestEnrichLayer:

    def test_translation_columns_added(self, simple_gdf, translations_df):
        gdf, stats = enrich_layer(simple_gdf, translations_df, ["de", "fr"], 0.5, "rocks")
        assert "LITHO_de" in gdf.columns
        assert "LITHO_fr" in gdf.columns
        assert len(stats) == 1

    def test_code_suffix_stripped(self, translations_df):
        """LITHO_CODE → LITHO_de (not LITHO_CODE_de)."""
        gdf = gpd.GeoDataFrame(
            {"LITHO_CODE": [VALID_CODE_A, VALID_CODE_B, VALID_CODE_C, UNKNOWN_CODE]},
            geometry=[Point(0, 0)] * 4,
            crs="EPSG:4326",
        )
        result, stats = enrich_layer(gdf, translations_df, ["de", "fr"], 0.5, "test")
        assert "LITHO_de" in result.columns
        assert "LITHO_CODE_de" not in result.columns
        assert stats[0]["out_prefix"] == "LITHO"

    def test_desc_columns_skipped(self, translations_df):
        """Columns ending in _desc are already text and must be ignored."""
        gdf = gpd.GeoDataFrame(
            {
                "tecto_code": [VALID_CODE_A] * 4,
                "tecto_desc": ["some text"] * 4,
            },
            geometry=[Point(0, 0)] * 4,
            crs="EPSG:4326",
        )
        result, stats = enrich_layer(gdf, translations_df, ["de", "fr"], 0.5, "test")
        # tecto_desc itself must not gain a _de/_fr sibling
        assert "tecto_desc_de" not in result.columns

    def test_already_translated_suffixes_skipped(self, translations_df):
        """Columns ending in _fr / _de / _en / _it are left alone."""
        gdf = gpd.GeoDataFrame(
            {
                "LITHO_CODE": [VALID_CODE_A] * 4,
                "LITHO_fr": ["existing"] * 4,
            },
            geometry=[Point(0, 0)] * 4,
            crs="EPSG:4326",
        )
        result, _ = enrich_layer(gdf, translations_df, ["de", "fr"], 0.5, "test")
        # Existing LITHO_fr must not be overwritten with a second _fr
        assert "LITHO_fr_fr" not in result.columns

    def test_zero_match_lang_not_added(self, translations_df):
        """A language column with 0 matches must not appear in the output."""
        # translations_df has EN=None for all rows → 0 matches
        gdf = gpd.GeoDataFrame(
            {"LITHO_CODE": [VALID_CODE_A, VALID_CODE_B]},
            geometry=[Point(0, 0)] * 2,
            crs="EPSG:4326",
        )
        result, stats = enrich_layer(
            gdf, translations_df, ["de", "fr", "en"], 0.5, "test"
        )
        assert "LITHO_en" not in result.columns

    def test_correct_translation_values(self, simple_gdf, translations_df):
        """Spot-check that mapped values are actually correct."""
        gdf, _ = enrich_layer(simple_gdf, translations_df, ["de", "fr"], 0.5, "rocks")
        row = gdf[gdf["LITHO_CODE"] == VALID_CODE_A].iloc[0]
        assert row["LITHO_de"] == "Granit"
        assert row["LITHO_fr"] == "Granite"

    def test_unknown_code_maps_to_nan(self, simple_gdf, translations_df):
        gdf, _ = enrich_layer(simple_gdf, translations_df, ["de", "fr"], 0.5, "rocks")
        row = gdf[gdf["LITHO_CODE"] == UNKNOWN_CODE].iloc[0]
        assert pd.isna(row["LITHO_de"])

    def test_stats_coverage_string(self, simple_gdf, translations_df):
        _, stats = enrich_layer(simple_gdf, translations_df, ["de", "fr"], 0.5, "rocks")
        assert "%" in stats[0]["coverage"]

    def test_non_code_column_untouched(self, simple_gdf, translations_df):
        """The NAME text column must not get translation siblings."""
        gdf, _ = enrich_layer(simple_gdf, translations_df, ["de", "fr"], 0.5, "rocks")
        assert "NAME_de" not in gdf.columns
        assert "NAME_fr" not in gdf.columns

    def test_no_translatable_columns(self, translations_df):
        """Layer with no code columns → empty stats, gdf unchanged."""
        gdf = gpd.GeoDataFrame(
            {"NAME": ["x", "y"], "VALUE": [1.5, 2.7]},
            geometry=[Point(0, 0)] * 2,
            crs="EPSG:4326",
        )
        result, stats = enrich_layer(gdf, translations_df, ["de", "fr"], 0.5, "empty")
        assert stats == []
        assert list(result.columns) == list(gdf.columns)



# ═══════════════════════════════════════════════════════════════════════════
# _is_pipe_codes_value
# ═══════════════════════════════════════════════════════════════════════════

class TestIsPipeCodesValue:

    def test_valid_pipe_string(self):
        val = f"{VALID_CODE_A} | {VALID_CODE_B} | {VALID_CODE_C}"
        assert _is_pipe_codes_value(val)

    def test_single_token_without_pipe(self):
        """A single code without a pipe separator is not a pipe-codes value."""
        assert not _is_pipe_codes_value(str(VALID_CODE_A))

    def test_sentinel_codes_accepted(self):
        code = next(iter(SPECIAL_GEOLCODES))
        assert _is_pipe_codes_value(f"{VALID_CODE_A} | {code}")

    def test_out_of_range_token_rejected(self):
        assert not _is_pipe_codes_value(f"{VALID_CODE_A} | 42")

    def test_non_integer_token_rejected(self):
        assert not _is_pipe_codes_value("granite | gneiss")

    def test_mixed_valid_and_text_rejected(self):
        assert not _is_pipe_codes_value(f"{VALID_CODE_A} | granite")

    def test_non_string_input(self):
        assert not _is_pipe_codes_value(VALID_CODE_A)   # int, not str
        assert not _is_pipe_codes_value(None)
        assert not _is_pipe_codes_value(3.14)


# ═══════════════════════════════════════════════════════════════════════════
# _map_pipe_codes
# ═══════════════════════════════════════════════════════════════════════════

class TestMapPipeCodes:

    @pytest.fixture
    def lang(self, translations_df):
        return translations_df["de"]

    def test_full_translation(self, lang):
        val = f"{VALID_CODE_A} | {VALID_CODE_B}"
        result = _map_pipe_codes(val, lang)
        assert result == f"Granit{PIPE_SEP}Gneis"

    def test_single_token(self, lang):
        result = _map_pipe_codes(str(VALID_CODE_A), lang)
        assert result == "Granit"

    def test_unknown_code_kept_as_string(self, lang):
        """Codes with no translation are kept as their numeric string."""
        val = f"{VALID_CODE_A} | {UNKNOWN_CODE}"
        result = _map_pipe_codes(val, lang)
        assert result == f"Granit{PIPE_SEP}{UNKNOWN_CODE}"

    def test_null_returns_none(self, lang):
        assert _map_pipe_codes(None, lang) is None
        assert _map_pipe_codes(float("nan"), lang) is None

    def test_preserves_pipe_sep_exactly(self, lang):
        val = f"{VALID_CODE_A} | {VALID_CODE_B} | {VALID_CODE_C}"
        result = _map_pipe_codes(val, lang)
        assert PIPE_SEP in result
        assert result.count(PIPE_SEP) == 2


# ═══════════════════════════════════════════════════════════════════════════
# is_geolcode_column — pipe-separated additions
# ═══════════════════════════════════════════════════════════════════════════

class TestIsGeolcodeColumnPipe:
    """Pipe-specific cases for is_geolcode_column."""

    def test_pipe_column_detected(self):
        s = pd.Series([
            f"{VALID_CODE_A} | {VALID_CODE_B}",
            f"{VALID_CODE_C} | {VALID_CODE_A}",
        ], dtype=object)
        assert is_geolcode_column(s, 0.5)

    def test_mixed_pipe_and_single(self):
        """First value is pipe-separated → whole column is accepted."""
        s = pd.Series([
            f"{VALID_CODE_A} | {VALID_CODE_B}",
            str(VALID_CODE_C),
        ], dtype=object)
        assert is_geolcode_column(s, 0.5)

    def test_pipe_with_out_of_range_not_detected(self):
        """If the first pipe-value has an out-of-range token, fall through to scalar check."""
        s = pd.Series([f"{VALID_CODE_A} | 42"], dtype=object)
        assert is_geolcode_column(s, 0.5) == (False, "No valid integer")

    def test_null_first_value_falls_through_to_scalar(self):
        """Null first value must not crash; scalar check takes over."""
        s = pd.Series([None, VALID_CODE_A, VALID_CODE_B])
        assert is_geolcode_column(s, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# enrich_layer — pipe-separated columns
# ═══════════════════════════════════════════════════════════════════════════

class TestEnrichLayerPipe:

    def test_pipe_column_translated(self, pipe_gdf, translations_df):
        result, stats = enrich_layer(pipe_gdf, translations_df, ["de", "fr"], 0.5, "test")
        assert "ADMIX_de" in result.columns
        assert "ADMIX_fr" in result.columns

    def test_pipe_code_suffix_stripped(self, pipe_gdf, translations_df):
        result, stats = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        assert "ADMIX_de" in result.columns
        assert "ADMIX_CODES_de" not in result.columns
        assert stats[0]["out_prefix"] == "ADMIX"

    def test_pipe_values_joined_correctly(self, pipe_gdf, translations_df):
        result, _ = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        first_row = result.iloc[0]["ADMIX_de"]
        assert PIPE_SEP in first_row
        assert "Granit" in first_row
        assert "Gneis" in first_row

    def test_pipe_same_values_stripped_correctly(self, pipe_gdf, translations_df):
        result, _ = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        fourth_row = result.iloc[4]["ADMIX_de"]
        assert PIPE_SEP not in fourth_row
        assert "Granit" == fourth_row
        assert "Gneis" not in fourth_row

    def test_null_row_stays_null(self, pipe_gdf, translations_df):
        result, _ = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        assert pd.isna(result.iloc[3]["ADMIX_de"])

    def test_single_token_in_pipe_column(self, pipe_gdf, translations_df):
        """Row with a single code (no pipe) in a pipe-detected column still translates."""
        result, _ = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        second_row = result.iloc[1]["ADMIX_de"]
        assert second_row == "Kalk"

    def test_partial_translation_unknown_kept(self, pipe_gdf, translations_df):
        """Tokens without a translation are kept as their numeric string."""
        result, _ = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        third_row = result.iloc[2]["ADMIX_de"]
        assert "Granit" in third_row
        assert str(UNKNOWN_CODE) in third_row

    def test_stats_marked_as_pipe(self, pipe_gdf, translations_df):
        _, stats = enrich_layer(pipe_gdf, translations_df, ["de"], 0.5, "test")
        assert stats[0]["pipe"] is True

# ═══════════════════════════════════════════════════════════════════════════
# _reorder_columns
# ═══════════════════════════════════════════════════════════════════════════

class TestReorderColumns:
    """Tests for _reorder_columns(gdf, fixed_first)."""

    def _make_gdf(self, cols: dict) -> gpd.GeoDataFrame:
        n = max(len(v) for v in cols.values())
        return gpd.GeoDataFrame(
            cols, geometry=[Point(i, 0) for i in range(n)], crs="EPSG:4326"
        )

    # ── basic ordering ────────────────────────────────────────────────────

    def test_pinned_columns_come_first(self):
        gdf = self._make_gdf({"uuid": [1], "zebra": [2], "gid": [3], "alpha": [4]})
        result = _reorder_columns(gdf, fixed_first=["gid", "uuid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert non_geom[0] == "gid"
        assert non_geom[1] == "uuid"

    def test_remaining_columns_sorted_alphabetically(self):
        gdf = self._make_gdf({"gid": [1], "zebra": [2], "alpha": [3], "mango": [4]})
        result = _reorder_columns(gdf, fixed_first=["gid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert non_geom == ["gid", "alpha", "mango", "zebra"]

    def test_geometry_column_is_last(self):
        gdf = self._make_gdf({"gid": [1], "kind": [2], "uuid": [3], "attr": [4]})
        result = _reorder_columns(gdf, fixed_first=["gid", "kind", "uuid"])
        assert result.columns[-1] == result.geometry.name

    def test_geometry_still_active_after_reorder(self):
        """geopandas must still recognise the geometry column after reordering."""
        gdf = self._make_gdf({"gid": [1], "attr": [2]})
        result = _reorder_columns(gdf, fixed_first=["gid"])
        assert result.geometry is not None
        assert len(result.geometry) == 1

    # ── edge cases for fixed_first ────────────────────────────────────────

    def test_missing_pinned_column_silently_skipped(self):
        """A name in fixed_first but absent from the GDF must be ignored."""
        gdf = self._make_gdf({"gid": [1], "beta": [2]})
        result = _reorder_columns(gdf, fixed_first=["gid", "does_not_exist"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert "does_not_exist" not in non_geom
        assert non_geom[0] == "gid"

    def test_empty_fixed_first_gives_fully_sorted_columns(self):
        gdf = self._make_gdf({"zebra": [1], "alpha": [2], "mango": [3]})
        result = _reorder_columns(gdf, fixed_first=[])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert non_geom == sorted(non_geom)

    def test_all_columns_pinned_preserves_fixed_first_order(self):
        """If every non-geometry column is pinned, order follows fixed_first exactly."""
        gdf = self._make_gdf({"gid": [1], "kind": [2], "uuid": [3]})
        result = _reorder_columns(gdf, fixed_first=["gid", "kind", "uuid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert non_geom == ["gid", "kind", "uuid"]

    def test_fixed_first_order_overrides_original_column_order(self):
        """fixed_first declaration order takes precedence over GDF column order."""
        gdf = self._make_gdf({"uuid": [1], "kind": [2], "gid": [3]})
        result = _reorder_columns(gdf, fixed_first=["gid", "kind", "uuid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert non_geom[:3] == ["gid", "kind", "uuid"]

    def test_pinned_columns_not_duplicated(self):
        """gid/uuid must each appear exactly once in the output columns."""
        gdf = self._make_gdf({"gid": [1], "uuid": [2], "attr": [3]})
        result = _reorder_columns(gdf, fixed_first=["gid", "uuid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        assert non_geom.count("gid") == 1
        assert non_geom.count("uuid") == 1

    # ── interaction with translation columns ─────────────────────────────

    def test_translation_siblings_sorted_among_rest(self):
        """_de / _fr siblings appended by enrich_layer are sorted with the rest."""
        gdf = self._make_gdf({
            "gid":      [1],
            "uuid":     [2],
            "litho_de": [3],
            "admix_de": [4],
            "litho_fr": [5],
            "admix_fr": [6],
        })
        result = _reorder_columns(gdf, fixed_first=["gid", "uuid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        rest = non_geom[2:]   # after the two pinned columns
        assert rest == sorted(rest)

    def test_source_code_column_sorted_with_siblings(self):
        """The original code column and its _de/_fr siblings all sort together."""
        gdf = self._make_gdf({
            "gid":       [1],
            "litho_code": [VALID_CODE_A],
            "litho_de":  ["Granit"],
            "litho_fr":  ["Granite"],
            "name":      ["x"],
        })
        result = _reorder_columns(gdf, fixed_first=["gid"])
        non_geom = [c for c in result.columns if c != result.geometry.name]
        rest = non_geom[1:]
        assert rest == sorted(rest)

    # ── integration: CLI full run preserves expected column order ─────────

    def test_full_run_column_order_via_cli(self, gpkg_file, translations_csv, tmp_path):
        """End-to-end smoke-test: non-geometry columns in the written GPKG are sorted."""
        from click.testing import CliRunner
        from translate_gpkg import main

        output = tmp_path / "ordered.gpkg"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(gpkg_file),
                "-t", str(translations_csv),
                "-o", str(output),
                "--langs", "de,fr",
            ],
        )
        assert result.exit_code == 0, result.output

        gdf = gpd.read_file(str(output), layer="rocks")
        geom_col = gdf.geometry.name

        # geometry must be the last column
        assert gdf.columns[-1] == geom_col

        # non-geometry, non-pinned columns must be alphabetically sorted
        pinned = ["gid", "kind", "uuid"]
        present_pinned = [c for c in pinned if c in gdf.columns]
        rest = [c for c in gdf.columns if c not in present_pinned and c != geom_col]
        assert rest == sorted(rest), (
            f"Expected alphabetically sorted columns after pinned, got: {rest}"
        )

# ═══════════════════════════════════════════════════════════════════════════
# CLI smoke test (--dry-run)
# ═══════════════════════════════════════════════════════════════════════════

class TestCli:

    def test_dry_run_does_not_write(self, gpkg_file, translations_csv, tmp_path):
        from click.testing import CliRunner
        from translate_gpkg import main

        output = tmp_path / "out.gpkg"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(gpkg_file),
                "-t", str(translations_csv),
                "-o", str(output),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not output.exists(), "dry-run must not create output file"

    def test_single_layer_filter(self, gpkg_file, translations_csv, tmp_path):
        from click.testing import CliRunner
        from translate_gpkg import main

        output = tmp_path / "out.gpkg"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(gpkg_file),
                "-t", str(translations_csv),
                "-o", str(output),
                "-l", "rocks",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_unknown_layer_exits_nonzero(self, gpkg_file, translations_csv, tmp_path):
        from click.testing import CliRunner
        from translate_gpkg import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(gpkg_file),
                "-t", str(translations_csv),
                "-l", "does_not_exist",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0 or "Unknown" in result.output

    def test_full_run_writes_gpkg(self, gpkg_file, translations_csv, tmp_path):
        from click.testing import CliRunner
        from translate_gpkg import main
        import fiona

        output = tmp_path / "out.gpkg"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(gpkg_file),
                "-t", str(translations_csv),
                "-o", str(output),
                "--langs", "de,fr",
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        layers = fiona.listlayers(str(output))
        assert "rocks" in layers
        gdf = gpd.read_file(str(output), layer="rocks")
        assert "LITHO_de" in gdf.columns
        assert "LITHO_fr" in gdf.columns
        # column ordering contract
        geom_col = gdf.geometry.name
        assert gdf.columns[-1] == geom_col
        non_geom = [c for c in gdf.columns if c != geom_col]
        assert non_geom == sorted(non_geom)

    def test_full_run_writes_gpkg_lowercase(self, gpkg_file, translations_csv, tmp_path):
        from click.testing import CliRunner
        from translate_gpkg import main

        output = tmp_path / "out_lower.gpkg"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(gpkg_file),
                "-t", str(translations_csv),
                "-o", str(output),
                "--langs", "de,fr",
                "--lowercase-columns",
            ],
        )
        assert result.exit_code == 0, result.output
        gdf = gpd.read_file(str(output), layer="rocks")

        assert "litho_de" in gdf.columns
        assert "litho_fr" in gdf.columns

        geom_col = gdf.geometry.name
        non_geom = [c for c in gdf.columns if c != geom_col]
        assert all(c == c.lower() for c in non_geom), (
            f"Expected all lowercase columns, found: {[c for c in non_geom if c != c.lower()]}"
        )