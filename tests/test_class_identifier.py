import pytest
from gcover.publish.symbol_models import ClassIdentifier


def test_single_field():
    bedrock = ClassIdentifier.from_single_field(
        layer_path="Surfaces/GC_SURFACES",
        field_name="GMU_CODE",
        field_value="15801003",
        class_index=42,
        label="Granite du Mont-Blanc",
    )

    assert str(bedrock) == "Surfaces/GC_SURFACES::gmu_code_15801003"
    assert bedrock.to_key() == "Surfaces/GC_SURFACES::gmu_code_15801003"
    assert bedrock.strategy.value == "value_based"


def test_multiple_fields():
    point = ClassIdentifier.from_multiple_fields(
        layer_path="Points/GC_POINT_OBJECTS",
        field_names=["KIND", "HSUR_STATUS"],
        field_values=["12501001", "1"],
        class_index=5,
        label="Source active",
    )

    assert str(point) == "Points/GC_POINT_OBJECTS::kind_12501001_hsur_status_1"
    assert point.to_key() == "Points/GC_POINT_OBJECTS::kind_12501001_hsur_status_1"
    assert point.strategy.value == "multi_value"


def test_expression():
    expr = ClassIdentifier.from_expression(
        layer_path="Fossils/GC_FOSSILS",
        expression="KIND = 14601006 AND LFOS_DIVISION = 'Triassic'",
        class_index=3,
        label="Fossiles triassiques",
    )

    assert str(expr) == "Fossils/GC_FOSSILS::kind_14601006_lfos_division_triassic"
    assert expr.to_key() == "Fossils/GC_FOSSILS::kind_14601006_lfos_division_triassic"
    assert expr.strategy.value == "expression"


def test_index():
    idx = ClassIdentifier.from_index(
        layer_path="Complex/LAYER",
        class_index=99,
        label="Unknown class",
    )

    assert str(idx) == "Complex/LAYER::idx_99"
    assert idx.to_key() == "Complex/LAYER::idx_99"
    assert idx.strategy.value == "index"
