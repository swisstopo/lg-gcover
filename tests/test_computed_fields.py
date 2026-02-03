"""
Tests for computed fields functionality.

Run with: pytest test_computed_fields.py -v
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon


from gcover.publish.esri_classification_applicator import (
    apply_computed_fields,
    validate_computed_fields,
    GEOMETRY_PROPERTIES,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def gdf_lines():
    """GeoDataFrame with LineString geometries."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "azimuth": [0, 90, 180],
            "depth": [-10, -20, -30],
            "value_a": [100, 200, 300],
            "value_b": [10, 20, 30],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(0, 0), (0, 200)]),
                LineString([(0, 0), (50, 50)]),
            ],
        },
        crs="EPSG:2056",
    )


@pytest.fixture
def gdf_polygons():
    """GeoDataFrame with Polygon geometries."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "name": ["small", "large"],
            "geometry": [
                Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),  # 10000 m²
                Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),  # 1000000 m²
            ],
        },
        crs="EPSG:2056",
    )


@pytest.fixture
def gdf_points():
    """GeoDataFrame with Point geometries."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "elevation": [100, 200, 300],
            "geometry": [
                Point(2600000, 1200000),
                Point(2600100, 1200100),
                Point(2600200, 1200200),
            ],
        },
        crs="EPSG:2056",
    )


# =============================================================================
# TESTS: BASIC ARITHMETIC
# =============================================================================


class TestArithmeticExpressions:
    """Test column arithmetic expressions."""

    def test_simple_subtraction(self, gdf_lines):
        """Test basic subtraction."""
        computed = {"result": "value_a - value_b"}
        result = apply_computed_fields(gdf_lines, computed)

        assert "result" in result.columns
        assert list(result["result"]) == [90, 180, 270]

    def test_modulo_operation(self, gdf_lines):
        """Test modulo operation."""
        computed = {"normalized": "azimuth % 360"}
        result = apply_computed_fields(gdf_lines, computed)

        assert "normalized" in result.columns
        assert list(result["normalized"]) == [0, 90, 180]

    def test_azimuth_to_map_angle(self, gdf_lines):
        """Test ESRI azimuth to MapServer angle conversion."""
        computed = {"map_angle": "(90 - azimuth) % 360"}
        result = apply_computed_fields(gdf_lines, computed)

        assert "map_angle" in result.columns
        expected = [90, 0, 270]  # (90-0)%360=90, (90-90)%360=0, (90-180)%360=270
        assert list(result["map_angle"]) == expected

    def test_negation(self, gdf_lines):
        """Test negation of values."""
        computed = {"depth_positive": "-depth"}
        result = apply_computed_fields(gdf_lines, computed)

        assert "depth_positive" in result.columns
        assert list(result["depth_positive"]) == [10, 20, 30]

    def test_multiplication(self, gdf_lines):
        """Test multiplication."""
        computed = {"doubled": "value_a * 2"}
        result = apply_computed_fields(gdf_lines, computed)

        assert list(result["doubled"]) == [200, 400, 600]

    def test_division(self, gdf_lines):
        """Test division."""
        computed = {"ratio": "value_a / value_b"}
        result = apply_computed_fields(gdf_lines, computed)

        assert list(result["ratio"]) == [10.0, 10.0, 10.0]

    def test_power(self, gdf_lines):
        """Test power operation."""
        computed = {"squared": "value_b ** 2"}
        result = apply_computed_fields(gdf_lines, computed)

        assert list(result["squared"]) == [100, 400, 900]

    def test_combined_expression(self, gdf_lines):
        """Test combined arithmetic expression."""
        computed = {"complex": "(value_a + value_b) / 2"}
        result = apply_computed_fields(gdf_lines, computed)

        assert list(result["complex"]) == [55.0, 110.0, 165.0]


# =============================================================================
# TESTS: GEOMETRY PROPERTIES
# =============================================================================


class TestGeometryProperties:
    """Test geometry-based computed fields."""

    def test_line_length(self, gdf_lines):
        """Test geometry.length for lines."""
        computed = {"line_m": "geometry.length"}
        result = apply_computed_fields(gdf_lines, computed)

        assert "line_m" in result.columns
        np.testing.assert_array_almost_equal(
            result["line_m"].values,
            [100.0, 200.0, 70.710678],  # sqrt(50²+50²) ≈ 70.71
            decimal=4,
        )

    def test_polygon_area(self, gdf_polygons):
        """Test geometry.area for polygons."""
        computed = {"area_m2": "geometry.area"}
        result = apply_computed_fields(gdf_polygons, computed)

        assert "area_m2" in result.columns
        assert list(result["area_m2"]) == [10000.0, 1000000.0]

    def test_polygon_perimeter(self, gdf_polygons):
        """Test geometry.length for polygon perimeter."""
        computed = {"perimeter_m": "geometry.length"}
        result = apply_computed_fields(gdf_polygons, computed)

        assert "perimeter_m" in result.columns
        assert list(result["perimeter_m"]) == [400.0, 4000.0]

    def test_area_conversion(self, gdf_polygons):
        """Test area unit conversion."""
        computed = {
            "area_m2": "geometry.area",
            "area_ha": "geometry.area / 10000",
            "area_km2": "geometry.area / 1000000",
        }
        result = apply_computed_fields(gdf_polygons, computed)

        assert list(result["area_ha"]) == [1.0, 100.0]
        assert list(result["area_km2"]) == [0.01, 1.0]

    def test_centroid_coordinates(self, gdf_polygons):
        """Test centroid extraction."""
        computed = {
            "center_x": "geometry.centroid.x",
            "center_y": "geometry.centroid.y",
        }
        result = apply_computed_fields(gdf_polygons, computed)

        assert "center_x" in result.columns
        assert "center_y" in result.columns
        # Small polygon centered at (50, 50)
        assert result["center_x"].iloc[0] == 50.0
        assert result["center_y"].iloc[0] == 50.0

    def test_bounds(self, gdf_polygons):
        """Test bounds extraction."""
        computed = {
            "minx": "geometry.bounds.minx",
            "maxx": "geometry.bounds.maxx",
        }
        result = apply_computed_fields(gdf_polygons, computed)

        assert result["minx"].iloc[0] == 0.0
        assert result["maxx"].iloc[0] == 100.0


# =============================================================================
# TESTS: FIELD OVERWRITING
# =============================================================================


class TestFieldOverwriting:
    """Test overwriting existing fields."""

    def test_overwrite_existing_field(self, gdf_lines):
        """Test that existing fields can be overwritten."""
        original_azimuth = list(gdf_lines["azimuth"])
        computed = {"azimuth": "azimuth + 180"}  # Overwrite azimuth
        result = apply_computed_fields(gdf_lines, computed)

        expected = [a + 180 for a in original_azimuth]
        assert list(result["azimuth"]) == expected

    def test_original_unchanged(self, gdf_lines):
        """Test that original GeoDataFrame is not modified."""
        original_azimuth = list(gdf_lines["azimuth"])
        computed = {"azimuth": "azimuth + 180"}
        _ = apply_computed_fields(gdf_lines, computed)

        # Original should be unchanged
        assert list(gdf_lines["azimuth"]) == original_azimuth


# =============================================================================
# TESTS: MULTIPLE FIELDS
# =============================================================================


class TestMultipleFields:
    """Test computing multiple fields at once."""

    def test_multiple_fields(self, gdf_lines):
        """Test computing multiple fields."""
        computed = {
            "map_angle": "(90 - azimuth) % 360",
            "depth_positive": "-depth",
            "line_m": "geometry.length",
            "sum_values": "value_a + value_b",
        }
        result = apply_computed_fields(gdf_lines, computed)

        assert all(col in result.columns for col in computed.keys())
        assert list(result["map_angle"]) == [90, 0, 270]
        assert list(result["depth_positive"]) == [10, 20, 30]
        assert list(result["sum_values"]) == [110, 220, 330]

    def test_field_order_preserved(self, gdf_lines):
        """Test that fields are added in order."""
        computed = {
            "field_a": "value_a * 2",
            "field_b": "value_b * 2",
            "field_c": "azimuth",
        }
        result = apply_computed_fields(gdf_lines, computed)

        # All fields should exist
        for field in computed.keys():
            assert field in result.columns


# =============================================================================
# TESTS: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_computed_fields(self, gdf_lines):
        """Test with empty computed_fields dict."""
        result = apply_computed_fields(gdf_lines, {})
        pd.testing.assert_frame_equal(result, gdf_lines)

    def test_none_computed_fields(self, gdf_lines):
        """Test with None computed_fields."""
        result = apply_computed_fields(gdf_lines, None)
        pd.testing.assert_frame_equal(result, gdf_lines)

    def test_missing_column_non_strict(self, gdf_lines):
        """Test that missing columns don't raise in non-strict mode."""
        computed = {"result": "nonexistent_column + 1"}
        # Should not raise, just warn
        result = apply_computed_fields(gdf_lines, computed, strict=False)
        # Field should not be created
        assert "result" not in result.columns or pd.isna(result["result"]).all()

    def test_missing_column_strict(self, gdf_lines):
        """Test that missing columns raise in strict mode."""
        computed = {"result": "nonexistent_column + 1"}
        with pytest.raises((ValueError, KeyError, pd.errors.UndefinedVariableError)):
            apply_computed_fields(gdf_lines, computed, strict=True)

    def test_division_by_zero(self, gdf_lines):
        """Test division by zero handling."""
        # Add a column with zero
        gdf_lines["zero"] = 0
        computed = {"result": "value_a / zero"}
        result = apply_computed_fields(gdf_lines, computed, strict=False)

        # Should produce inf
        assert "result" in result.columns
        assert all(np.isinf(result["result"]))

    def test_empty_geodataframe(self):
        """Test with empty GeoDataFrame."""
        gdf_empty = gpd.GeoDataFrame(
            {"azimuth": [], "geometry": []},
            crs="EPSG:2056",
        )
        computed = {"map_angle": "(90 - azimuth) % 360"}
        result = apply_computed_fields(gdf_empty, computed)

        assert "map_angle" in result.columns
        assert len(result) == 0


# =============================================================================
# TESTS: VALIDATION
# =============================================================================


class TestValidation:
    """Test computed fields validation."""

    def test_validate_valid_expressions(self, gdf_lines):
        """Test validation with valid expressions."""
        computed = {
            "map_angle": "(90 - azimuth) % 360",
            "line_m": "geometry.length",
        }
        issues = validate_computed_fields(gdf_lines, computed)
        assert len(issues) == 0

    def test_validate_missing_column(self, gdf_lines):
        """Test validation detects missing columns."""
        computed = {"result": "nonexistent + 1"}
        issues = validate_computed_fields(gdf_lines, computed)

        assert len(issues) == 1
        assert issues[0]["field"] == "result"
        assert issues[0]["issue"] == "missing_columns"
        assert "nonexistent" in issues[0]["details"]


# =============================================================================
# TESTS: GEOMETRY PROPERTIES REGISTRY
# =============================================================================


class TestGeometryPropertiesRegistry:
    """Test the GEOMETRY_PROPERTIES registry."""

    def test_all_properties_callable(self):
        """Test that all registered properties are callable."""
        for key, func in GEOMETRY_PROPERTIES.items():
            assert callable(func), f"{key} is not callable"

    def test_expected_properties_exist(self):
        """Test that expected properties are registered."""
        expected = [
            "geometry.length",
            "geometry.area",
            "geometry.centroid.x",
            "geometry.centroid.y",
        ]
        for prop in expected:
            assert prop in GEOMETRY_PROPERTIES, f"Missing property: {prop}"
