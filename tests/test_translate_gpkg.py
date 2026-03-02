"""
tests/test_translate_gpkg.py
─────────────────────────────
Unit tests for translate_gpkg.py

Covers:
  - load_translations        : happy path, bad rows dropped, missing langs
  - is_geolcode_column       : in-range codes, out-of-range, strings, sentinels,
                               _code-suffix columns, ignored attributes, float noise
  - check_min_langs          : passes / aborts correctly
  - _lowercase_gdf_columns   : renames attrs, leaves geometry column name alone
  - enrich_layer             : translations added, _code suffix stripped,
                               _desc columns skipped, ignored attrs skipped,
                               zero-match lang column not added
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
    SPECIAL_GEOLCODES,
    _lowercase_gdf_columns,
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
            "LITHO_CODE": [VALID_CODE_A, VALID_CODE_B, VALID_CODE_C, UNKNOWN_CODE],
            "NAME": ["a", "b", "c", "d"],
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
        assert not is_geolcode_column(s, 0.0)

    def test_float_values_with_decimal_part(self):
        """Floats with a non-zero fractional part should not qualify."""
        s = self._s([VALID_CODE_A + 0.5] * 10)
        assert not is_geolcode_column(s, 0.5)

    def test_float_values_whole_numbers(self):
        """Floats that are whole numbers (e.g. 15111123.0) should qualify."""
        s = self._s([float(VALID_CODE_A)] * 10)
        assert is_geolcode_column(s, 0.5)

    def test_all_nulls(self):
        s = self._s([None, None, None])
        assert not is_geolcode_column(s, 0.0)

    def test_below_min_coverage(self):
        """Only 2/10 non-null → coverage=0.2 < min_coverage=0.5."""
        s = self._s([VALID_CODE_A, VALID_CODE_B] + [None] * 8)
        assert not is_geolcode_column(s, 0.5)

    def test_mixed_in_and_out_of_range(self):
        """If < 90 % of valid values are in range, reject."""
        in_range = [VALID_CODE_A] * 8
        out_range = [42, 99]   # 20 % out of range → below 90 % threshold
        s = self._s(in_range + out_range)
        assert not is_geolcode_column(s, 0.0)

    def test_pure_text_column(self):
        s = pd.Series(["granite", "gneiss", "limestone"], dtype=object)
        assert not is_geolcode_column(s, 0.0)

    def test_boolean_column(self):
        """bool dtype is numeric in numpy but values 0/1 are way out of range."""
        s = self._s([True, False, True])
        assert not is_geolcode_column(s, 0.0)


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
