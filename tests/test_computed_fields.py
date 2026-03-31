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

# =============================================================================
# TESTS: BEARING COMPUTATION
# =============================================================================


class TestBearingComputation:
    """Test geometry.bearing and related properties."""

    @pytest.fixture
    def gdf_cardinal_lines(self):
        """Lines pointing in cardinal directions from origin."""
        return gpd.GeoDataFrame(
            {
                "direction": ["north", "east", "south", "west", "northeast"],
                "geometry": [
                    LineString([(0, 0), (0, 100)]),      # North: bearing = 0°
                    LineString([(0, 0), (100, 0)]),     # East: bearing = 90°
                    LineString([(0, 0), (0, -100)]),    # South: bearing = 180°
                    LineString([(0, 0), (-100, 0)]),    # West: bearing = 270°
                    LineString([(0, 0), (100, 100)]),   # NE: bearing = 45°
                ],
            },
            crs="EPSG:2056",
        )

    def test_bearing_cardinal_directions(self, gdf_cardinal_lines):
        """Test bearing for cardinal directions."""
        computed = {"bearing": "geometry.bearing"}
        result = apply_computed_fields(gdf_cardinal_lines, computed)

        expected = [0.0, 90.0, 180.0, 270.0, 45.0]
        np.testing.assert_array_almost_equal(
            result["bearing"].values, expected, decimal=1
        )

    def test_bearing_as_integer(self, gdf_cardinal_lines):
        """Test bearing cast to integer."""
        computed = {"bearing_int": "geometry.bearing:int"}
        result = apply_computed_fields(gdf_cardinal_lines, computed)

        assert result["bearing_int"].dtype == "Int64"
        assert list(result["bearing_int"]) == [0, 90, 180, 270, 45]

    def test_strike(self, gdf_cardinal_lines):
        """Test strike (0-180° range)."""
        computed = {"strike": "geometry.strike"}
        result = apply_computed_fields(gdf_cardinal_lines, computed)

        # Strike normalizes to 0-180: 270° → 90°, 180° → 0°
        expected = [0.0, 90.0, 0.0, 90.0, 45.0]
        np.testing.assert_array_almost_equal(
            result["strike"].values, expected, decimal=1
        )

    def test_bearing_to_map_angle(self, gdf_cardinal_lines):
        """Test conversion from bearing to MapServer angle."""
        computed = {"map_angle": "(90 - geometry.bearing) % 360:int"}
        result = apply_computed_fields(gdf_cardinal_lines, computed)

        # MapServer: 0=East, CCW positive
        # North (bearing=0) → 90, East (bearing=90) → 0, etc.
        expected = [90, 0, 270, 180, 45]
        assert list(result["map_angle"]) == expected

    def test_bearing_empty_geometry(self):
        """Test bearing with empty geometry."""
        gdf = gpd.GeoDataFrame(
            {
                "id": [1],
                "geometry": [LineString()],  # Empty
            },
            crs="EPSG:2056",
        )
        computed = {"bearing": "geometry.bearing"}
        result = apply_computed_fields(gdf, computed)

        assert pd.isna(result["bearing"].iloc[0])

    def test_bearing_point_geometry(self, gdf_points):
        """Test that bearing returns NaN for non-line geometries."""
        computed = {"bearing": "geometry.bearing"}
        result = apply_computed_fields(gdf_points, computed)

        assert all(pd.isna(result["bearing"]))


# =============================================================================
# TESTS: MULTILINESTRING BEARING
# =============================================================================


class TestMultiLineStringBearing:
    """Test bearing for MultiLineString geometries."""

    @pytest.fixture
    def gdf_multilines(self):
        """GeoDataFrame with MultiLineString geometries."""
        from shapely.geometry import MultiLineString as MLS

        return gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [
                    # Two segments: short east, long north
                    MLS([
                        LineString([(0, 0), (10, 0)]),    # 10m east
                        LineString([(0, 0), (0, 100)]),   # 100m north
                    ]),
                    # Single segment
                    MLS([
                        LineString([(0, 0), (50, 50)]),   # NE diagonal
                    ]),
                ],
            },
            crs="EPSG:2056",
        )

    def test_bearing_uses_longest(self, gdf_multilines):
        """Test that geometry.bearing uses the longest segment."""
        computed = {"bearing": "geometry.bearing"}
        result = apply_computed_fields(gdf_multilines, computed)

        # First: longest is north (100m) → bearing 0°
        # Second: single NE segment → bearing 45°
        expected = [0.0, 45.0]
        np.testing.assert_array_almost_equal(
            result["bearing"].values, expected, decimal=1
        )

    def test_bearing_weighted(self, gdf_multilines):
        """Test length-weighted bearing average."""
        computed = {"bearing_w": "geometry.bearing_weighted"}
        result = apply_computed_fields(gdf_multilines, computed)

        # Weighted average should be closer to north (longer segment)
        # but not exactly 0° due to east segment contribution
        assert 0 < result["bearing_w"].iloc[0] < 45


# =============================================================================
# TESTS: TYPE CASTING
# =============================================================================


class TestTypeCasting:
    """Test inline type casting with :type suffix."""

    def test_cast_to_int(self, gdf_lines):
        """Test :int cast."""
        computed = {"length_int": "geometry.length:int"}
        result = apply_computed_fields(gdf_lines, computed)

        assert result["length_int"].dtype == "Int64"
        assert list(result["length_int"]) == [100, 200, 71]  # Rounded

    def test_cast_to_int64(self, gdf_lines):
        """Test :Int64 cast (explicit)."""
        computed = {"length_int64": "geometry.length:Int64"}
        result = apply_computed_fields(gdf_lines, computed)

        assert result["length_int64"].dtype == "Int64"

    def test_cast_to_float(self, gdf_lines):
        """Test :float cast."""
        computed = {"value_float": "value_a:float"}
        result = apply_computed_fields(gdf_lines, computed)

        assert result["value_float"].dtype == "float64"

    def test_cast_to_string(self, gdf_lines):
        """Test :str cast."""
        computed = {"value_str": "value_a:str"}
        result = apply_computed_fields(gdf_lines, computed)

        assert result["value_str"].dtype == "string"
        assert result["value_str"].iloc[0] == "100"

    def test_cast_round(self, gdf_lines):
        """Test :round cast."""
        computed = {"length_round": "geometry.length:round"}
        result = apply_computed_fields(gdf_lines, computed)

        # Should be rounded but still float
        np.testing.assert_array_almost_equal(
            result["length_round"].values, [100.0, 200.0, 71.0], decimal=0
        )

    def test_cast_with_expression(self, gdf_polygons):
        """Test casting with arithmetic expression."""
        computed = {"area_km2": "geometry.area / 1000000:round"}
        result = apply_computed_fields(gdf_polygons, computed)

        assert list(result["area_km2"]) == [0.0, 1.0]

    def test_cast_preserves_nan(self):
        """Test that casting preserves NaN values."""
        gdf = gpd.GeoDataFrame(
            {
                "value": [1.5, np.nan, 3.7],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:2056",
        )
        computed = {"value_int": "value:int"}
        result = apply_computed_fields(gdf, computed)

        assert result["value_int"].iloc[0] == 2  # Rounded
        assert pd.isna(result["value_int"].iloc[1])  # NaN preserved
        assert result["value_int"].iloc[2] == 4


# =============================================================================
# TESTS: CONCAT FUNCTION
# =============================================================================


class TestConcatFunction:
    """Test concat() special function."""

    @pytest.fixture
    def gdf_litho(self):
        """GeoDataFrame with lithology codes."""
        return gpd.GeoDataFrame(
            {
                "litho_main": [15101001, 15101002, 15101003],
                "litho_sec": [15102001, 15102002, 0],
                "litho_ter": [15103001, 0, 0],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:2056",
        )

    def test_concat_default_separator(self, gdf_litho):
        """Test concat with default separator ' | '."""
        computed = {"combined": "concat(litho_main, litho_sec, litho_ter)"}
        result = apply_computed_fields(gdf_litho, computed)

        assert "combined" in result.columns
        assert result["combined"].iloc[0] == "15101001 | 15102001 | 15103001"

    def test_concat_custom_separator(self, gdf_litho):
        """Test concat with custom separator."""
        computed = {"combined": "concat(litho_main, litho_sec, sep='|')"}
        result = apply_computed_fields(gdf_litho, computed)

        assert result["combined"].iloc[0] == "15101001|15102001"

    def test_concat_dash_separator(self, gdf_litho):
        """Test concat with dash separator."""
        computed = {"combined": "concat(litho_main, litho_sec, sep='-')"}
        result = apply_computed_fields(gdf_litho, computed)

        assert result["combined"].iloc[0] == "15101001-15102001"

    def test_concat_single_field(self, gdf_litho):
        """Test concat with single field."""
        computed = {"single": "concat(litho_main)"}
        result = apply_computed_fields(gdf_litho, computed)

        assert result["single"].iloc[0] == "15101001"

    def test_concat_with_zeros(self, gdf_litho):
        """Test that concat includes zero values as '0'."""
        computed = {"combined": "concat(litho_main, litho_sec, litho_ter)"}
        result = apply_computed_fields(gdf_litho, computed)

        # Row with zeros should show them
        assert "0" in result["combined"].iloc[2]


# =============================================================================
# TESTS: CONCAT WITH NULLS
# =============================================================================


class TestConcatWithNulls:
    """Test concat() handling of NULL values."""

    @pytest.fixture
    def gdf_with_nulls(self):
        """GeoDataFrame with NULL values."""
        return gpd.GeoDataFrame(
            {
                "field_a": ["alpha", "beta", None],
                "field_b": ["one", None, "three"],
                "field_c": [100, 200, 300],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:2056",
        )

    def test_concat_replaces_null_with_empty(self, gdf_with_nulls):
        """Test that NULL values become empty strings in concat."""
        computed = {"combined": "concat(field_a, field_b, sep='|')"}
        result = apply_computed_fields(gdf_with_nulls, computed)

        assert result["combined"].iloc[0] == "alpha|one"
        assert result["combined"].iloc[1] == "beta|"  # NULL → empty
        assert result["combined"].iloc[2] == "|three"  # NULL → empty


# =============================================================================
# TESTS: COALESCE FUNCTION
# =============================================================================


class TestCoalesceFunction:
    """Test coalesce() special function."""

    @pytest.fixture
    def gdf_sparse(self):
        """GeoDataFrame with sparse data (many NULLs)."""
        return gpd.GeoDataFrame(
            {
                "primary": [100, None, None, 400],
                "secondary": [None, 200, None, 500],
                "tertiary": [None, None, 300, 600],
                "geometry": [Point(i, i) for i in range(4)],
            },
            crs="EPSG:2056",
        )

    def test_coalesce_basic(self, gdf_sparse):
        """Test coalesce returns first non-null value."""
        computed = {"result": "coalesce(primary, secondary, tertiary)"}
        result = apply_computed_fields(gdf_sparse, computed)

        expected = [100.0, 200.0, 300.0, 400.0]
        assert list(result["result"]) == expected

    def test_coalesce_two_fields(self, gdf_sparse):
        """Test coalesce with two fields."""
        computed = {"result": "coalesce(primary, secondary)"}
        result = apply_computed_fields(gdf_sparse, computed)

        # Third row: both primary and secondary are NULL → stays NULL
        assert result["result"].iloc[0] == 100.0
        assert result["result"].iloc[1] == 200.0
        assert pd.isna(result["result"].iloc[2])
        assert result["result"].iloc[3] == 400.0

    def test_coalesce_single_field(self, gdf_sparse):
        """Test coalesce with single field (identity)."""
        computed = {"result": "coalesce(primary)"}
        result = apply_computed_fields(gdf_sparse, computed)

        assert result["result"].iloc[0] == 100.0
        assert pd.isna(result["result"].iloc[1])

    def test_coalesce_all_null(self):
        """Test coalesce when all values are NULL."""
        gdf = gpd.GeoDataFrame(
            {
                "a": [None, None],
                "b": [None, None],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:2056",
        )
        computed = {"result": "coalesce(a, b)"}
        result = apply_computed_fields(gdf, computed)

        assert all(pd.isna(result["result"]))


# =============================================================================
# TESTS: COMBINED FEATURES
# =============================================================================


class TestCombinedFeatures:
    """Test combining multiple new features."""

    def test_bearing_cast_and_concat(self):
        """Test bearing, casting, and concat together."""
        gdf = gpd.GeoDataFrame(
            {
                "name": ["fault_1", "fault_2"],
                "type_code": [101, 102],
                "geometry": [
                    LineString([(0, 0), (100, 100)]),  # NE: 45°
                    LineString([(0, 0), (0, 100)]),   # N: 0°
                ],
            },
            crs="EPSG:2056",
        )

        computed = {
            "bearing": "geometry.bearing:int",
            "info": "concat(name, type_code, sep=' - ')",
        }
        result = apply_computed_fields(gdf, computed)

        assert result["bearing"].iloc[0] == 45
        assert result["bearing"].iloc[1] == 0
        assert result["info"].iloc[0] == "fault_1 - 101"

    def test_coalesce_then_concat(self):
        """Test coalesce followed by concat."""
        gdf = gpd.GeoDataFrame(
            {
                "code_main": [100, None],
                "code_backup": [999, 200],
                "label": ["A", "B"],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:2056",
        )

        computed = {
            "code": "coalesce(code_main, code_backup)",
            "combined": "concat(label, code_main, sep=':')",  # Uses original
        }
        result = apply_computed_fields(gdf, computed)

        assert result["code"].iloc[0] == 100.0
        assert result["code"].iloc[1] == 200.0


# =============================================================================
# TESTS: VALIDATION WITH NEW FEATURES
# =============================================================================


class TestValidationNewFeatures:
    """Test validation of new computed field features."""

    def test_validate_invalid_cast(self, gdf_lines):
        """Test validation rejects invalid cast types."""
        computed = {"result": "value_b:invalid_type"}
        issues = validate_computed_fields(gdf_lines, computed)

        assert len(issues) == 1
        assert issues[0]["issue"] == "invalid_cast"

    def test_validate_concat_missing_field(self, gdf_lines):
        """Test validation detects missing fields in concat."""
        computed = {"result": "concat(value_a, nonexistent)"}
        # This will fail at runtime, not validation (could be enhanced)
        result = apply_computed_fields(gdf_lines, computed, strict=False)
        # Should warn but not crash
        assert "result" not in result.columns or pd.isna(result["result"]).all()

    def test_validate_bearing_property_exists(self, gdf_lines):
        """Test that geometry.bearing is recognized as valid."""
        computed = {"bearing": "geometry.bearing"}
        issues = validate_computed_fields(gdf_lines, computed)

        assert len(issues) == 0
