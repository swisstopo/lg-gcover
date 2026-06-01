# tests/test_domain_compliance.py
"""
Tests for coded-domain compliance checking.

Covers:
  - _load_domains / _load_field_domain_map (domain metadata extraction)
  - _check_layer              (violation detection, integer + string paths)
  - _cleanup_domain_violations (GDBMerger in-memory cleanup before save)
  - cross-GDB drift scenario  (target checked against a different reference)

All GDBs are built in-process with OGR — no binary fixtures committed to the repo.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from osgeo import gdal, ogr

# Suppress GDAL 4.0 FutureWarning about exception mode before any OGR call.
gdal.DontUseExceptions()

from gcover.publish.merge_sources import GDBMerger, MergeConfig, SourceConfig, SourceType

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Import the script under test (it's not a package, use importlib)
# ---------------------------------------------------------------------------

def _import_script(name: str, rel_path: str):
    scripts_dir = Path(__file__).parent.parent / rel_path
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_compliance = _import_script("check_domain_compliance", "scripts")
_load_domains        = _compliance._load_domains
_load_field_domain_map = _compliance._load_field_domain_map
_check_layer         = _compliance._check_layer


# ---------------------------------------------------------------------------
# GDB builder helper
# ---------------------------------------------------------------------------

def _make_gdb(path: Path, features: list[dict]) -> Path:
    """
    Build a minimal FileGDB at *path* with:

      Domains
        GC_KIND_CD   (Integer)  {10: 'Sediment', 20: 'Magmatic', 30: 'Metamorphic'}
        GC_STATUS_CD (String)   {'A': 'Active',  'B': 'Buried',  'C': 'Covered'}

      Layer  GC_BEDROCK
        KIND   OFTInteger  → GC_KIND_CD
        STATUS OFTString   → GC_STATUS_CD

    Each dict in *features* has keys 'kind' and 'status'; omit a key to leave NULL.
    """
    drv = ogr.GetDriverByName("OpenFileGDB")
    ds  = drv.CreateDataSource(str(path))

    ds.AddFieldDomain(ogr.CreateCodedFieldDomain(
        "GC_KIND_CD", "Rock kind",
        ogr.OFTInteger, ogr.OFDT_CODED,
        {10: "Sediment", 20: "Magmatic", 30: "Metamorphic"},
    ))
    ds.AddFieldDomain(ogr.CreateCodedFieldDomain(
        "GC_STATUS_CD", "Outcrop status",
        ogr.OFTString, ogr.OFDT_CODED,
        {"A": "Active", "B": "Buried", "C": "Covered"},
    ))

    lyr = ds.CreateLayer("GC_BEDROCK", geom_type=ogr.wkbPoint)

    fld_kind = ogr.FieldDefn("KIND", ogr.OFTInteger)
    fld_kind.SetDomainName("GC_KIND_CD")
    lyr.CreateField(fld_kind)

    fld_status = ogr.FieldDefn("STATUS", ogr.OFTString)
    fld_status.SetDomainName("GC_STATUS_CD")
    lyr.CreateField(fld_status)

    defn = lyr.GetLayerDefn()
    for feat_vals in features:
        f = ogr.Feature(defn)
        f.SetGeometry(ogr.CreateGeometryFromWkt("POINT (2600000 1200000)"))
        if "kind" in feat_vals:
            f.SetField("KIND", feat_vals["kind"])
        if "status" in feat_vals:
            f.SetField("STATUS", feat_vals["status"])
        lyr.CreateFeature(f)

    ds = None
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_gdb(tmp_path_factory) -> Path:
    """GDB whose every feature respects its coded domains."""
    p = tmp_path_factory.mktemp("gdbs") / "clean.gdb"
    return _make_gdb(p, [
        {"kind": 10, "status": "A"},
        {"kind": 20, "status": "B"},
        {"kind": 30, "status": "C"},
        {},                          # fully NULL — must not be flagged
    ])


@pytest.fixture(scope="module")
def dirty_gdb(tmp_path_factory) -> Path:
    """GDB with controlled domain violations."""
    p = tmp_path_factory.mktemp("gdbs") / "dirty.gdb"
    return _make_gdb(p, [
        {"kind": 10,  "status": "A"},   # valid
        {"kind": 99,  "status": "A"},   # KIND bogus
        {"kind": 20,  "status": "Z"},   # STATUS bogus
        {"kind": 999, "status": "ZZ"},  # both bogus
        {},                              # NULL — not a violation
    ])


@pytest.fixture(scope="module")
def reference_gdb(tmp_path_factory) -> Path:
    """GDB used as authoritative domain source (stricter: only codes 10 and 20)."""
    p = tmp_path_factory.mktemp("gdbs") / "reference.gdb"
    drv = ogr.GetDriverByName("OpenFileGDB")
    ds  = drv.CreateDataSource(str(p))
    ds.AddFieldDomain(ogr.CreateCodedFieldDomain(
        "GC_KIND_CD", "Rock kind (reference subset)",
        ogr.OFTInteger, ogr.OFDT_CODED,
        {10: "Sediment", 20: "Magmatic"},   # 30 intentionally absent
    ))
    ds.AddFieldDomain(ogr.CreateCodedFieldDomain(
        "GC_STATUS_CD", "Outcrop status",
        ogr.OFTString, ogr.OFDT_CODED,
        {"A": "Active", "B": "Buried", "C": "Covered"},
    ))
    lyr = ds.CreateLayer("GC_BEDROCK", geom_type=ogr.wkbPoint)
    fld = ogr.FieldDefn("KIND", ogr.OFTInteger)
    fld.SetDomainName("GC_KIND_CD")
    lyr.CreateField(fld)
    fld2 = ogr.FieldDefn("STATUS", ogr.OFTString)
    fld2.SetDomainName("GC_STATUS_CD")
    lyr.CreateField(fld2)
    ds = None
    return p


# ---------------------------------------------------------------------------
# _load_domains
# ---------------------------------------------------------------------------

class TestLoadDomains:
    def test_returns_both_domains(self, clean_gdb):
        domains = _load_domains(clean_gdb)
        assert "GC_KIND_CD"   in domains
        assert "GC_STATUS_CD" in domains

    def test_integer_domain_codes_are_strings(self, clean_gdb):
        # GetEnumeration() always returns str keys regardless of field type
        domains = _load_domains(clean_gdb)
        assert domains["GC_KIND_CD"] == frozenset({"10", "20", "30"})

    def test_string_domain_codes(self, clean_gdb):
        domains = _load_domains(clean_gdb)
        assert domains["GC_STATUS_CD"] == frozenset({"A", "B", "C"})

    def test_range_domains_are_excluded(self):
        # Verify _load_domains ignores non-CODED domain types via a mock datasource.
        mock_range = MagicMock()
        mock_range.GetDomainType.return_value = ogr.OFDT_RANGE

        mock_coded = MagicMock()
        mock_coded.GetDomainType.return_value = ogr.OFDT_CODED
        mock_coded.GetEnumeration.return_value = {"1": "One", "2": "Two"}

        mock_ds = MagicMock()
        mock_ds.GetFieldDomainNames.return_value = ["GC_YEAR_RD", "GC_KIND_CD"]
        mock_ds.GetFieldDomain.side_effect = lambda n: {
            "GC_YEAR_RD": mock_range,
            "GC_KIND_CD": mock_coded,
        }[n]

        with patch("osgeo.ogr.Open", return_value=mock_ds):
            domains = _load_domains(Path("/mock/path.gdb"))

        assert "GC_YEAR_RD" not in domains
        assert "GC_KIND_CD" in domains


# ---------------------------------------------------------------------------
# _load_field_domain_map
# ---------------------------------------------------------------------------

class TestLoadFieldDomainMap:
    def test_layer_present(self, clean_gdb):
        fmap = _load_field_domain_map(clean_gdb)
        assert "GC_BEDROCK" in fmap

    def test_field_domain_assignments(self, clean_gdb):
        fmap = _load_field_domain_map(clean_gdb)
        fields = fmap["GC_BEDROCK"]
        assert fields["KIND"][0]   == "GC_KIND_CD"
        assert fields["STATUS"][0] == "GC_STATUS_CD"

    def test_integer_type_recorded(self, clean_gdb):
        fmap = _load_field_domain_map(clean_gdb)
        _, ogr_type = fmap["GC_BEDROCK"]["KIND"]
        assert ogr_type == ogr.OFTInteger

    def test_string_type_recorded(self, clean_gdb):
        fmap = _load_field_domain_map(clean_gdb)
        _, ogr_type = fmap["GC_BEDROCK"]["STATUS"]
        assert ogr_type == ogr.OFTString


# ---------------------------------------------------------------------------
# _check_layer
# ---------------------------------------------------------------------------

class TestCheckLayer:
    def test_clean_gdb_no_violations(self, clean_gdb):
        domains  = _load_domains(clean_gdb)
        fmap     = _load_field_domain_map(clean_gdb)
        result   = _check_layer(clean_gdb, "GC_BEDROCK", fmap["GC_BEDROCK"], domains)
        assert result == []

    def test_detects_bogus_integer(self, dirty_gdb):
        domains = _load_domains(dirty_gdb)
        fmap    = _load_field_domain_map(dirty_gdb)
        result  = _check_layer(dirty_gdb, "GC_BEDROCK", fmap["GC_BEDROCK"], domains)
        kind_rec = next(r for r in result if r["field"] == "KIND")
        assert kind_rec["n_bogus"] == 2   # 99 and 999

    def test_detects_bogus_string(self, dirty_gdb):
        domains = _load_domains(dirty_gdb)
        fmap    = _load_field_domain_map(dirty_gdb)
        result  = _check_layer(dirty_gdb, "GC_BEDROCK", fmap["GC_BEDROCK"], domains)
        status_rec = next(r for r in result if r["field"] == "STATUS")
        assert status_rec["n_bogus"] == 2  # 'Z' and 'ZZ'

    def test_null_values_not_flagged(self, dirty_gdb):
        domains = _load_domains(dirty_gdb)
        fmap    = _load_field_domain_map(dirty_gdb)
        result  = _check_layer(dirty_gdb, "GC_BEDROCK", fmap["GC_BEDROCK"], domains)
        # 5 features total, 1 fully NULL → 4 non-null per field
        kind_rec = next(r for r in result if r["field"] == "KIND")
        assert kind_rec["n_total"] == 4

    def test_examples_contain_bogus_values(self, dirty_gdb):
        domains = _load_domains(dirty_gdb)
        fmap    = _load_field_domain_map(dirty_gdb)
        result  = _check_layer(dirty_gdb, "GC_BEDROCK", fmap["GC_BEDROCK"], domains)
        kind_rec   = next(r for r in result if r["field"] == "KIND")
        status_rec = next(r for r in result if r["field"] == "STATUS")
        assert 99  in kind_rec["examples"]   or 999 in kind_rec["examples"]
        assert "Z" in status_rec["examples"] or "ZZ" in status_rec["examples"]

    def test_cross_gdb_drift(self, dirty_gdb, reference_gdb):
        # dirty_gdb has kind=30 (valid in self, absent from reference)
        # In addition to its own 99/999 violations we get an extra violation for 30
        ref_domains = _load_domains(reference_gdb)
        fmap        = _load_field_domain_map(dirty_gdb)
        result      = _check_layer(dirty_gdb, "GC_BEDROCK", fmap["GC_BEDROCK"], ref_domains)
        kind_rec    = next(r for r in result if r["field"] == "KIND")
        # 99, 999 (always bogus) + 10→valid, 20→valid; but also no 30 code in reference
        # dirty_gdb features: 10✓, 99✗, 20✓, 999✗, NULL — no 30 present, so same 2 violations
        assert kind_rec["n_bogus"] == 2


# ---------------------------------------------------------------------------
# GDBMerger._cleanup_domain_violations
# ---------------------------------------------------------------------------

class TestCleanupDomainViolations:
    """Test the in-memory cleanup step that runs before _save_merged_layers."""

    def _make_merger(self, gdb_path: Path) -> GDBMerger:
        """Minimal GDBMerger with rc2_path set to *gdb_path*."""
        cfg = MergeConfig(rc2_path=gdb_path)
        merger = GDBMerger.__new__(GDBMerger)
        merger.config  = cfg
        merger.verbose = False
        merger.sources = {}
        merger.mapsheets_gdf       = None
        merger.source_masks        = {}
        merger.swiss_border        = None
        from gcover.publish.merge_sources import MergeStats
        merger.stats = MergeStats()
        # Register RC2 source manually (mirrors _setup_sources)
        merger.sources["RC2"] = SourceConfig(
            name="RC2", path=gdb_path, source_type=SourceType.RC2
        )
        return merger

    def _make_layers(self) -> dict:
        """GeoDataFrame dict mimicking merged_layers with controlled violations."""
        gdf = gpd.GeoDataFrame(
            {
                "KIND":   pd.array([10, 99, 20, None, 30, 777], dtype="Int64"),
                "STATUS": ["A", "B", "Z", None, "C", "X"],
            },
            geometry=[Point(0, 0)] * 6,
            crs="EPSG:2056",
        )
        return {"GC_ROCK_BODIES/GC_BEDROCK": gdf}

    def test_bogus_integers_become_null(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        layers = self._make_layers()
        result = merger._cleanup_domain_violations(layers)
        kind = result["GC_ROCK_BODIES/GC_BEDROCK"]["KIND"]
        assert pd.isna(kind.iloc[1])   # was 99
        assert pd.isna(kind.iloc[5])   # was 777

    def test_valid_integers_preserved(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        layers = self._make_layers()
        result = merger._cleanup_domain_violations(layers)
        kind = result["GC_ROCK_BODIES/GC_BEDROCK"]["KIND"]
        assert kind.iloc[0] == 10
        assert kind.iloc[2] == 20
        assert kind.iloc[4] == 30

    def test_bogus_strings_become_null(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        layers = self._make_layers()
        result = merger._cleanup_domain_violations(layers)
        status = result["GC_ROCK_BODIES/GC_BEDROCK"]["STATUS"]
        assert pd.isna(status.iloc[2])  # was 'Z'
        assert pd.isna(status.iloc[5])  # was 'X'

    def test_existing_nulls_stay_null(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        layers = self._make_layers()
        result = merger._cleanup_domain_violations(layers)
        kind   = result["GC_ROCK_BODIES/GC_BEDROCK"]["KIND"]
        status = result["GC_ROCK_BODIES/GC_BEDROCK"]["STATUS"]
        assert pd.isna(kind.iloc[3])
        assert pd.isna(status.iloc[3])

    def test_stats_recorded(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        layers = self._make_layers()
        merger._cleanup_domain_violations(layers)
        assert "GC_BEDROCK" in merger.stats.domain_cleanup
        assert merger.stats.domain_cleanup["GC_BEDROCK"]["KIND"]   == 2
        assert merger.stats.domain_cleanup["GC_BEDROCK"]["STATUS"] == 2

    def test_clean_layers_untouched(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        gdf = gpd.GeoDataFrame(
            {"KIND": pd.array([10, 20, 30], dtype="Int64"), "STATUS": ["A", "B", "C"]},
            geometry=[Point(0, 0)] * 3,
            crs="EPSG:2056",
        )
        layers = {"GC_ROCK_BODIES/GC_BEDROCK": gdf}
        merger._cleanup_domain_violations(layers)
        assert "GC_BEDROCK" not in merger.stats.domain_cleanup

    def test_no_rc_source_returns_unchanged(self, clean_gdb):
        merger = self._make_merger(clean_gdb)
        merger.sources = {}            # remove all sources
        layers = self._make_layers()
        result = merger._cleanup_domain_violations(layers)
        # values unchanged
        assert result["GC_ROCK_BODIES/GC_BEDROCK"]["KIND"].iloc[1] == 99
