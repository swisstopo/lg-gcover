# tests/test_merge_sources.py
"""
Tests for GDBMerger class.

Uses synthetic test data (GeoPackage) to test merge logic without requiring
actual FileGDB files or arcpy.
"""

import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box, LineString
from shapely.ops import unary_union

from gcover.publish.merge_sources import (
    GDBMerger,
    MergeConfig,
    create_merge_config,
    fast_clip,

    get_expected_geometry_type,
    normalize_geodataframe_geometries,
)


# =============================================================================
# FIXTURES - Synthetic Test Data
# =============================================================================

@pytest.fixture
def crs():
    """Swiss LV95 coordinate system."""
    return "EPSG:2056"


@pytest.fixture
def test_area_bounds():
    """
    Test area bounds (simplified Swiss-like area).
    Using round numbers for easy reasoning.

    Area is 100x100 km, split into 4 quadrants for mapsheets.
    """
    return {
        "minx": 2600000,
        "miny": 1200000,
        "maxx": 2700000,
        "maxy": 1300000,
    }


@pytest.fixture
def mapsheets_gdf(test_area_bounds, crs):
    """
    Create 4 mapsheets covering the test area.

    Layout:
    +-------+-------+
    |  MS3  |  MS4  |   MS3, MS4 = RC2
    | (RC2) | (RC2) |
    +-------+-------+
    |  MS1  |  MS2  |   MS1, MS2 = RC1
    | (RC1) | (RC1) |
    +-------+-------+
    """
    b = test_area_bounds
    mid_x = (b["minx"] + b["maxx"]) / 2
    mid_y = (b["miny"] + b["maxy"]) / 2

    mapsheets = [
        # Bottom-left (RC1)
        {
            "MSH_MAP_NBR": 1,
            "MSH_MAP_TITLE": "Sheet1",
            "SOURCE_RC": "RC1",
            "geometry": box(b["minx"], b["miny"], mid_x, mid_y),
        },
        # Bottom-right (RC1)
        {
            "MSH_MAP_NBR": 2,
            "MSH_MAP_TITLE": "Sheet2",
            "SOURCE_RC": "RC1",
            "geometry": box(mid_x, b["miny"], b["maxx"], mid_y),
        },
        # Top-left (RC2)
        {
            "MSH_MAP_NBR": 3,
            "MSH_MAP_TITLE": "Sheet3",
            "SOURCE_RC": "RC2",
            "geometry": box(b["minx"], mid_y, mid_x, b["maxy"]),
        },
        # Top-right (RC2)
        {
            "MSH_MAP_NBR": 4,
            "MSH_MAP_TITLE": "Sheet4",
            "SOURCE_RC": "RC2",
            "geometry": box(mid_x, mid_y, b["maxx"], b["maxy"]),
        },
    ]

    return gpd.GeoDataFrame(mapsheets, crs=crs)


@pytest.fixture
def admin_zones_gpkg(tmp_path, mapsheets_gdf):
    """Create admin zones GeoPackage with mapsheets."""
    path = tmp_path / "admin_zones.gpkg"
    mapsheets_gdf.to_file(path, layer="mapsheets_sources_only", driver="GPKG")
    return path


@pytest.fixture
def sample_features_rc1(test_area_bounds, crs):
    """
    Create sample features for RC1 source.

    Features cover the ENTIRE test area (simulating real data where
    RC1.gdb has all of Switzerland).

    Includes:
    - Features fully in RC1 territory
    - Features fully in RC2 territory (should be excluded)
    - Features crossing the RC1/RC2 boundary (should be split)
    """
    b = test_area_bounds
    mid_x = (b["minx"] + b["maxx"]) / 2
    mid_y = (b["miny"] + b["maxy"]) / 2

    features = [
        # Feature fully in RC1 territory (bottom-left)
        {
            "UUID": "uuid-rc1-only-1",
            "GEOLCODE": 1001,
            "NAME": "RC1 Feature 1",
            "geometry": box(b["minx"] + 1000, b["miny"] + 1000,
                            b["minx"] + 10000, b["miny"] + 10000),
        },
        # Feature fully in RC1 territory (bottom-right)
        {
            "UUID": "uuid-rc1-only-2",
            "GEOLCODE": 1002,
            "NAME": "RC1 Feature 2",
            "geometry": box(mid_x + 1000, b["miny"] + 1000,
                            mid_x + 10000, b["miny"] + 10000),
        },
        # Feature crossing RC1/RC2 boundary (spans mid_y)
        {
            "UUID": "uuid-boundary-1",
            "GEOLCODE": 2001,
            "NAME": "Boundary Feature 1",
            "geometry": box(b["minx"] + 20000, mid_y - 5000,
                            b["minx"] + 30000, mid_y + 5000),
        },
        # Feature fully in RC2 territory (should be EXCLUDED from RC1 output)
        {
            "UUID": "uuid-rc2-territory-in-rc1",
            "GEOLCODE": 3001,
            "NAME": "RC2 Territory Feature (in RC1 DB)",
            "geometry": box(b["minx"] + 1000, mid_y + 10000,
                            b["minx"] + 10000, mid_y + 20000),
        },
        # Another boundary-crossing feature
        {
            "UUID": "uuid-boundary-2",
            "GEOLCODE": 2002,
            "NAME": "Boundary Feature 2",
            "geometry": box(mid_x - 5000, mid_y - 5000,
                            mid_x + 5000, mid_y + 5000),
        },
    ]

    return gpd.GeoDataFrame(features, crs=crs)


@pytest.fixture
def sample_features_rc2(test_area_bounds, crs):
    """
    Create sample features for RC2 source.

    Also covers the ENTIRE test area, with many of the same UUIDs as RC1
    (simulating that RC2 is a copy of RC1 with edits).
    """
    b = test_area_bounds
    mid_x = (b["minx"] + b["maxx"]) / 2
    mid_y = (b["miny"] + b["maxy"]) / 2

    features = [
        # Feature fully in RC2 territory (top-left)
        {
            "UUID": "uuid-rc2-only-1",
            "GEOLCODE": 4001,
            "NAME": "RC2 Feature 1",
            "geometry": box(b["minx"] + 1000, mid_y + 10000,
                            b["minx"] + 10000, mid_y + 20000),
        },
        # Feature fully in RC2 territory (top-right)
        {
            "UUID": "uuid-rc2-only-2",
            "GEOLCODE": 4002,
            "NAME": "RC2 Feature 2",
            "geometry": box(mid_x + 1000, mid_y + 10000,
                            mid_x + 10000, mid_y + 20000),
        },
        # SAME UUID as RC1 boundary feature - RC2 version (should win in RC2 territory)
        {
            "UUID": "uuid-boundary-1",
            "GEOLCODE": 2001,
            "NAME": "Boundary Feature 1 (RC2 version)",
            "geometry": box(b["minx"] + 20000, mid_y - 5000,
                            b["minx"] + 30000, mid_y + 5000),
        },
        # Feature fully in RC1 territory (should be EXCLUDED from RC2 output)
        {
            "UUID": "uuid-rc1-territory-in-rc2",
            "GEOLCODE": 5001,
            "NAME": "RC1 Territory Feature (in RC2 DB)",
            "geometry": box(b["minx"] + 40000, b["miny"] + 1000,
                            b["minx"] + 45000, b["miny"] + 10000),
        },
        # SAME UUID as RC1 boundary feature 2
        {
            "UUID": "uuid-boundary-2",
            "GEOLCODE": 2002,
            "NAME": "Boundary Feature 2 (RC2 version)",
            "geometry": box(mid_x - 5000, mid_y - 5000,
                            mid_x + 5000, mid_y + 5000),
        },
    ]

    return gpd.GeoDataFrame(features, crs=crs)


@pytest.fixture
def sample_points_rc1(test_area_bounds, crs):
    """Sample point features for RC1."""
    b = test_area_bounds
    mid_x = (b["minx"] + b["maxx"]) / 2
    mid_y = (b["miny"] + b["maxy"]) / 2

    points = [
        {"UUID": "pt-rc1-1", "KIND": 12501001, "geometry": Point(b["minx"] + 5000, b["miny"] + 5000)},
        {"UUID": "pt-rc1-2", "KIND": 12501002, "geometry": Point(mid_x + 5000, b["miny"] + 5000)},
        # Point in RC2 territory
        {"UUID": "pt-rc2-in-rc1", "KIND": 12501003, "geometry": Point(b["minx"] + 5000, mid_y + 5000)},
    ]

    return gpd.GeoDataFrame(points, crs=crs)


@pytest.fixture
def sample_points_rc2(test_area_bounds, crs):
    """Sample point features for RC2."""
    b = test_area_bounds
    mid_x = (b["minx"] + b["maxx"]) / 2
    mid_y = (b["miny"] + b["maxy"]) / 2

    points = [
        {"UUID": "pt-rc2-1", "KIND": 12501001, "geometry": Point(b["minx"] + 5000, mid_y + 5000)},
        {"UUID": "pt-rc2-2", "KIND": 12501002, "geometry": Point(mid_x + 5000, mid_y + 5000)},
        # Point in RC1 territory (should be excluded)
        {"UUID": "pt-rc1-in-rc2", "KIND": 12501003, "geometry": Point(b["minx"] + 15000, b["miny"] + 5000)},
    ]

    return gpd.GeoDataFrame(points, crs=crs)


@pytest.fixture
def rc1_gpkg(tmp_path, sample_features_rc1, sample_points_rc1):
    """Create RC1 source as GeoPackage (substitute for FileGDB in tests)."""
    path = tmp_path / "RC1.gpkg"
    sample_features_rc1.to_file(path, layer="GC_BEDROCK", driver="GPKG")
    sample_points_rc1.to_file(path, layer="GC_POINT_OBJECTS", driver="GPKG")
    return path


@pytest.fixture
def rc2_gpkg(tmp_path, sample_features_rc2, sample_points_rc2):
    """Create RC2 source as GeoPackage (substitute for FileGDB in tests)."""
    path = tmp_path / "RC2.gpkg"
    sample_features_rc2.to_file(path, layer="GC_BEDROCK", driver="GPKG")
    sample_points_rc2.to_file(path, layer="GC_POINT_OBJECTS", driver="GPKG")
    return path


@pytest.fixture
def merge_config(tmp_path, rc1_gpkg, rc2_gpkg, admin_zones_gpkg):
    """Create merge configuration for tests."""
    output_path = tmp_path / "merged.gpkg"

    return MergeConfig(
        rc1_path=rc1_gpkg,
        rc2_path=rc2_gpkg,
        admin_zones_path=admin_zones_gpkg,
        mapsheets_layer="mapsheets_sources_only",
        source_column="SOURCE_RC",
        output_path=output_path,
        spatial_layers=["GC_BEDROCK", "GC_POINT_OBJECTS"],
        non_spatial_tables=[],  # Skip for simplicity
        use_convex_hull_masks=False,  # Use exact masks
        clip_to_swiss_border=True,
        validate_geometries=True,
        preserve_z=False,
    )


# =============================================================================
# TESTS - Mask Creation
# =============================================================================

class TestMaskCreation:
    """Tests for exclusive mask creation."""

    def test_masks_are_exclusive(self, merge_config):
        """Masks should not overlap."""
        merger = GDBMerger(merge_config, verbose=False)
        merger._load_mapsheets()
        merger._create_source_masks()

        masks = merger.source_masks

        assert "RC1" in masks
        assert "RC2" in masks

        # Check no overlap
        overlap = masks["RC1"].intersection(masks["RC2"])
        assert overlap.is_empty or overlap.area < 1, "Masks should not overlap"

    def test_masks_cover_full_area(self, merge_config, mapsheets_gdf):
        """Union of all masks should cover the full mapsheet area."""
        merger = GDBMerger(merge_config, verbose=False)
        merger._load_mapsheets()
        merger._create_source_masks()

        masks = merger.source_masks
        all_masks_union = unary_union(list(masks.values()))

        full_area = unary_union(mapsheets_gdf.geometry.values)

        # Masks should cover at least 99.9% of the area
        coverage = all_masks_union.area / full_area.area
        assert coverage > 0.999, f"Masks cover only {coverage * 100:.1f}% of full area"

    def test_rc2_has_priority(self, merge_config, test_area_bounds):
        """RC2 should get its full area, RC1 gets the remainder."""
        merger = GDBMerger(merge_config, verbose=False)
        merger._load_mapsheets()
        merger._create_source_masks()

        # RC2 area should be exactly top half (2 mapsheets)
        b = test_area_bounds
        mid_y = (b["miny"] + b["maxy"]) / 2
        expected_rc2_area = (b["maxx"] - b["minx"]) * (b["maxy"] - mid_y)

        actual_rc2_area = merger.source_masks["RC2"].area

        # Allow 1% tolerance
        assert abs(actual_rc2_area - expected_rc2_area) / expected_rc2_area < 0.01


# =============================================================================
# TESTS - Feature Merging
# =============================================================================

class TestFeatureMerging:
    """Tests for merging features from multiple sources."""

    def test_no_duplicate_uuids(self, merge_config):
        """Merged output should have no duplicate UUIDs."""
        merger = GDBMerger(merge_config, verbose=False)
        stats = merger.merge()

        # Read merged output
        merged = gpd.read_file(merge_config.output_path, layer="GC_BEDROCK")

        # Check for duplicates
        uuid_counts = merged["UUID"].value_counts()
        duplicates = uuid_counts[uuid_counts > 1]

        assert len(duplicates) == 0, f"Found duplicate UUIDs: {duplicates.to_dict()}"

    def test_features_from_correct_source(self, merge_config, test_area_bounds):
        """Features should come from the correct source based on location."""
        merger = GDBMerger(merge_config, verbose=False)
        merger.merge()

        merged = gpd.read_file(merge_config.output_path, layer="GC_BEDROCK")

        b = test_area_bounds
        mid_y = (b["miny"] + b["maxy"]) / 2

        # Check that features in RC2 territory are from RC2
        rc2_territory = box(b["minx"], mid_y, b["maxx"], b["maxy"])

        for idx, row in merged.iterrows():
            centroid = row.geometry.centroid
            if rc2_territory.contains(centroid):
                # Feature centroid is in RC2 territory, should be from RC2
                assert row["_MERGE_SOURCE"] == "RC2", \
                    f"Feature {row['UUID']} in RC2 territory but from {row['_MERGE_SOURCE']}"

    def test_boundary_features_are_split(self, merge_config, test_area_bounds):
        """Features crossing source boundary should be split."""
        merger = GDBMerger(merge_config, verbose=False)
        merger.merge()

        merged = gpd.read_file(merge_config.output_path, layer="GC_BEDROCK")

        # The boundary features (uuid-boundary-1, uuid-boundary-2) should appear
        # but their geometry should be clipped to their respective source areas

        b = test_area_bounds
        mid_y = (b["miny"] + b["maxy"]) / 2

        # Check that no single feature crosses the boundary
        for idx, row in merged.iterrows():
            geom = row.geometry
            crosses_boundary = (geom.bounds[1] < mid_y) and (geom.bounds[3] > mid_y)

            # Small tolerance for boundary touching
            if crosses_boundary:
                # Check if it's just touching the boundary (acceptable)
                height_crossing = min(geom.bounds[3], mid_y + 1) - max(geom.bounds[1], mid_y - 1)
                assert height_crossing < 10, \
                    f"Feature {row['UUID']} significantly crosses boundary"

    def test_point_features_not_duplicated(self, merge_config):
        """Point features should not be duplicated."""
        merger = GDBMerger(merge_config, verbose=False)
        merger.merge()

        merged = gpd.read_file(merge_config.output_path, layer="GC_POINT_OBJECTS")

        uuid_counts = merged["UUID"].value_counts()
        duplicates = uuid_counts[uuid_counts > 1]

        assert len(duplicates) == 0, f"Duplicate point UUIDs: {duplicates.to_dict()}"

    def test_points_from_correct_source(self, merge_config, test_area_bounds):
        """Points should come from the correct source."""
        merger = GDBMerger(merge_config, verbose=False)
        merger.merge()

        merged = gpd.read_file(merge_config.output_path, layer="GC_POINT_OBJECTS")

        b = test_area_bounds
        mid_y = (b["miny"] + b["maxy"]) / 2

        for idx, row in merged.iterrows():
            geom = row.geometry
            if geom.geom_type == "MultiPoint":
                y = geom.centroid.y
            else:
                y = geom.y
            expected_source = "RC2" if y > mid_y else "RC1"
            assert row["_MERGE_SOURCE"] == expected_source, \
                f"Point {row['UUID']} at y={y} should be from {expected_source}"


# =============================================================================
# TESTS - Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_layer(self, tmp_path, admin_zones_gpkg, crs):
        """Should handle empty layers gracefully."""
        # Create sources with empty layer
        rc1_path = tmp_path / "RC1_empty.gpkg"
        rc2_path = tmp_path / "RC2_empty.gpkg"

        empty_gdf = gpd.GeoDataFrame(
            {"UUID": [], "geometry": []},
            crs=crs
        )
        empty_gdf.to_file(rc1_path, layer="GC_BEDROCK", driver="GPKG")
        empty_gdf.to_file(rc2_path, layer="GC_BEDROCK", driver="GPKG")

        config = MergeConfig(
            rc1_path=rc1_path,
            rc2_path=rc2_path,
            admin_zones_path=admin_zones_gpkg,
            output_path=tmp_path / "merged_empty.gpkg",
            spatial_layers=["GC_BEDROCK"],
            non_spatial_tables=[],
        )

        merger = GDBMerger(config, verbose=False)
        stats = merger.merge()

        assert stats.features_per_layer.get("GC_BEDROCK", 0) == 0

    def test_missing_source(self, tmp_path, rc1_gpkg, admin_zones_gpkg):
        """Should handle missing source gracefully."""
        config = MergeConfig(
            rc1_path=rc1_gpkg,
            rc2_path=tmp_path / "nonexistent.gpkg",  # Doesn't exist
            admin_zones_path=admin_zones_gpkg,
            output_path=tmp_path / "merged_missing.gpkg",
            spatial_layers=["GC_BEDROCK"],
            non_spatial_tables=[],
        )

        merger = GDBMerger(config, verbose=False)
        stats = merger.merge()

        # Should complete with warnings
        assert len(stats.warnings) > 0 or "RC2" not in merger.sources

    def test_mapsheet_filter(self, merge_config, tmp_path):
        """Should filter by mapsheet numbers when specified."""
        merge_config.mapsheet_numbers = [1, 2]  # Only RC1 mapsheets
        merge_config.output_path = tmp_path / "merged_filtered.gpkg"

        merger = GDBMerger(merge_config, verbose=False)
        merger.merge()

        merged = gpd.read_file(merge_config.output_path, layer="GC_BEDROCK")

        # All features should be from RC1
        sources = merged["_MERGE_SOURCE"].unique()
        assert list(sources) == ["RC1"], f"Expected only RC1, got {sources}"


# =============================================================================
# TESTS - Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_fast_clip_preserves_features_inside(self, crs):
        """fast_clip should preserve features fully inside the mask."""
        mask = box(0, 0, 100, 100)

        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "geometry": [
                box(10, 10, 20, 20),  # Fully inside
                box(90, 90, 110, 110),  # Partially outside
            ]
        }, crs=crs)

        result = fast_clip(gdf, mask)

        assert len(result) == 2
        # First feature should be unchanged
        assert result.iloc[0].geometry.equals(gdf.iloc[0].geometry)
        # Second feature should be clipped
        assert result.iloc[1].geometry.area < gdf.iloc[1].geometry.area

    def test_fast_clip_excludes_outside(self, crs):
        """fast_clip should exclude features fully outside the mask."""
        mask = box(0, 0, 100, 100)

        gdf = gpd.GeoDataFrame({
            "id": [1],
            "geometry": [box(200, 200, 300, 300)],  # Fully outside
        }, crs=crs)

        result = fast_clip(gdf, mask)

        assert len(result) == 0

    def test_get_expected_geometry_type(self):
        """Should return correct geometry type based on layer name."""
        assert get_expected_geometry_type("GC_BEDROCK") == "MultiPolygon"
        assert get_expected_geometry_type("GC_POINT_OBJECTS") == "MultiPoint"
        assert get_expected_geometry_type("GC_LINEAR_OBJECTS") == "MultiLineString"
        assert get_expected_geometry_type("GC_FOSSILS") == "MultiPoint"

    def test_normalize_geometry_handles_collection(self, crs):
        """Should extract correct geometry type from GeometryCollection."""
        # Simulate what happens after clipping - might get GeometryCollection
        from shapely.geometry import GeometryCollection, MultiPolygon

        mixed = GeometryCollection([
            box(0, 0, 10, 10),
            LineString([(0, 0), (10, 10)]),  # Should be filtered out
        ])

        gdf = gpd.GeoDataFrame({
            "id": [1],
            "geometry": [mixed],
        }, crs=crs)

        result = normalize_geodataframe_geometries(gdf, "MultiPolygon")

        assert len(result) == 1
        assert result.iloc[0].geometry.geom_type in ["Polygon", "MultiPolygon"]


# =============================================================================
# TESTS - Statistics
# =============================================================================

class TestMergeStatistics:
    """Tests for merge statistics."""

    def test_stats_features_per_source(self, merge_config):
        """Should track features per source."""
        merger = GDBMerger(merge_config, verbose=False)
        stats = merger.merge()

        assert "RC1" in stats.features_per_source
        assert "RC2" in stats.features_per_source
        assert stats.features_per_source["RC1"] > 0
        assert stats.features_per_source["RC2"] > 0

    def test_stats_features_per_layer(self, merge_config):
        """Should track features per layer."""
        merger = GDBMerger(merge_config, verbose=False)
        stats = merger.merge()

        assert "GC_BEDROCK" in stats.features_per_layer
        assert "GC_POINT_OBJECTS" in stats.features_per_layer

    def test_stats_layers_processed(self, merge_config):
        """Should count processed layers."""
        merger = GDBMerger(merge_config, verbose=False)
        stats = merger.merge()

        assert stats.layers_processed == 2  # GC_BEDROCK and GC_POINT_OBJECTS


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestIntegration:
    """Full integration tests."""

    def test_full_merge_workflow(self, merge_config):
        """Test complete merge workflow."""
        merger = GDBMerger(merge_config, verbose=True)
        stats = merger.merge()

        # Verify output exists
        assert merge_config.output_path.exists()

        # Verify layers were created
        import fiona
        layers = fiona.listlayers(str(merge_config.output_path))
        assert "GC_BEDROCK" in layers
        assert "GC_POINT_OBJECTS" in layers

        # Verify no errors
        assert len(stats.errors) == 0

        # Verify reasonable feature counts
        assert stats.features_per_layer["GC_BEDROCK"] > 0
        assert stats.features_per_layer["GC_POINT_OBJECTS"] > 0

        # Final verification: no duplicate UUIDs across all layers
        for layer in ["GC_BEDROCK", "GC_POINT_OBJECTS"]:
            gdf = gpd.read_file(merge_config.output_path, layer=layer)
            if "UUID" in gdf.columns:
                assert gdf["UUID"].is_unique, f"Duplicate UUIDs in {layer}"