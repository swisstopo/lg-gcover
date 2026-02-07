#!/usr/bin/env python
"""
Unit tests for Phase 1: Mapfile Generation Control

Tests:
- MapfileGenerationConfig validation
- SymbolAdjustments application
- ManualMapfileHandler symbol extraction
- ClassIdentifier generation strategies
"""

import pytest
from pathlib import Path
from io import StringIO

# Assuming these are the new modules
from gcover.publish.style_config import (
    MapfileGenerationConfig,
    SymbolAdjustments,
    LayerConfig,
    ClassificationApplicationConfig,
    BatchClassificationConfig,
)
from gcover.publish.manual_mapfile_handler import (
    ManualMapfileHandler,
    MapfileSymbolUsage,
)
from gcover.publish.symbol_models import (
    ClassIdentifier,
    IdentifierStrategy,
)


# =============================================================================
# MapfileGenerationConfig Tests
# =============================================================================


class TestMapfileGenerationConfig:
    """Test MapfileGenerationConfig validation and initialization"""
    
    def test_auto_mode_default(self):
        """Auto mode should work with minimal config"""
        config = MapfileGenerationConfig(generation_mode="auto")
        assert config.generation_mode == "auto"
        assert config.manual_mapfile_path is None
        assert config.extract_symbols_for_catalog is True
    
    def test_manual_mode_requires_path(self):
        """Manual mode must have manual_mapfile_path"""
        with pytest.raises(ValueError, match="manual_mapfile_path required"):
            MapfileGenerationConfig(
                generation_mode="manual"
                # Missing manual_mapfile_path
            )
    
    def test_manual_mode_with_path(self):
        """Manual mode should work with path"""
        config = MapfileGenerationConfig(
            generation_mode="manual",
            manual_mapfile_path=Path("mapfiles/manual/test.map")
        )
        assert config.generation_mode == "manual"
        assert config.manual_mapfile_path.name == "test.map"
    
    def test_disabled_mode(self):
        """Disabled mode should work with reason"""
        config = MapfileGenerationConfig(
            generation_mode="disabled",
            reason="Not ready for production"
        )
        assert config.generation_mode == "disabled"
        assert "Not ready" in config.reason
    
    def test_symbol_adjustments_from_dict(self):
        """SymbolAdjustments should be created from dict"""
        config = MapfileGenerationConfig(
            generation_mode="auto",
            symbol_adjustments={
                "point_size_multiplier": 1.5,
                "line_width_multiplier": 0.8,
                "dash_pattern_override": [5, 10],
            }
        )
        
        assert isinstance(config.symbol_adjustments, SymbolAdjustments)
        assert config.symbol_adjustments.point_size_multiplier == 1.5
        assert config.symbol_adjustments.line_width_multiplier == 0.8
        assert config.symbol_adjustments.dash_pattern_override == [5, 10]


# =============================================================================
# SymbolAdjustments Tests
# =============================================================================


class TestSymbolAdjustments:
    """Test SymbolAdjustments behavior"""
    
    def test_defaults(self):
        """Default values should be identity (no change)"""
        adj = SymbolAdjustments()
        assert adj.point_size_multiplier == 1.0
        assert adj.line_width_multiplier == 1.0
        assert adj.dash_pattern_override is None
        assert adj.transparency_override is None
    
    def test_custom_values(self):
        """Custom values should be stored correctly"""
        adj = SymbolAdjustments(
            point_size_multiplier=1.3,
            line_width_multiplier=0.7,
            dash_pattern_override=[8, 4, 2, 4],
            transparency_override=50,
        )
        
        assert adj.point_size_multiplier == 1.3
        assert adj.line_width_multiplier == 0.7
        assert adj.dash_pattern_override == [8, 4, 2, 4]
        assert adj.transparency_override == 50
    
    def test_to_dict(self):
        """Should serialize to dict"""
        adj = SymbolAdjustments(point_size_multiplier=1.5)
        d = adj.to_dict()
        
        assert d["point_size_multiplier"] == 1.5
        assert d["line_width_multiplier"] == 1.0


# =============================================================================
# BatchClassificationConfig Tests
# =============================================================================


class TestBatchClassificationConfig:
    """Test YAML config parsing with new fields"""
    
    @pytest.fixture
    def sample_yaml(self, tmp_path):
        """Create sample YAML config"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
global:
  symbol_field: SYMBOL
  label_field: LABEL

layers:
  - gpkg_layer: test_layer
    gcover_layer: test
       
    classifications:
      - style_file: test.lyrx
        symbol_prefix: test
        mapfile_config:
            generation_mode: auto
            symbol_adjustments:
                point_size_multiplier: 1.2
            metadata:
                wms_title: "Test Layer"
""")
        return config_file
    
    def test_parse_mapfile_config(self, sample_yaml):
        """Should parse mapfile_config from YAML"""
        config = BatchClassificationConfig(sample_yaml)
        
        assert len(config.layers) == 1
        layer = config.layers[0]

        assert len(layer.classifications) == 1
        classification = layer.classifications[0]
        
        assert classification.mapfile_config is not None
        assert classification.mapfile_config.generation_mode == "auto"
        assert classification.mapfile_config.symbol_adjustments.point_size_multiplier == 1.2
        assert classification.mapfile_config.metadata["wms_title"] == "Test Layer"
    
    def test_get_manual_layers(self, tmp_path):
        """Should filter manual layers"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
global:
  symbol_field: SYMBOL

layers:
  - gpkg_layer: auto_layer
    gcover_layer: auto
 
    classifications:
      - style_file: test.lyrx
        mapfile_config:
            generation_mode: auto
  
  - gpkg_layer: manual_layer
    gcover_layer: manual
    
    classifications:
      - style_file: test.lyrx
        mapfile_config:
          generation_mode: manual
          manual_mapfile_path: manual.map
  
  - gpkg_layer: disabled_layer
    gcover_layer: disabled
    
    classifications:
      - style_file: test.lyrx
        mapfile_config:
          generation_mode: disabled
""")
        
        config = BatchClassificationConfig(config_file)
        
        manual_layers = config.get_manual_layers()
        assert len(manual_layers) == 1
        assert manual_layers[0].gpkg_layer == "manual_layer"
        
        disabled_layers = config.get_disabled_layers()
        assert len(disabled_layers) == 1
        assert disabled_layers[0].gpkg_layer == "disabled_layer"
        
        auto_layers = config.get_auto_layers()
        assert len(auto_layers) == 1
        assert auto_layers[0].gpkg_layer == "auto_layer"


# =============================================================================
# ManualMapfileHandler Tests
# =============================================================================


class TestManualMapfileHandler:
    """Test manual mapfile symbol extraction"""
    
    @pytest.fixture
    def sample_mapfile(self, tmp_path):
        """Create sample mapfile"""
        mapfile = tmp_path / "test.map"
        mapfile.write_text("""
LAYER
  NAME "test_layer"
  TYPE POLYGON
  
  CLASS
    NAME "Class 1"
    EXPRESSION ([GMU_CODE] eq "123")
    STYLE
      SYMBOL "bedrock_pattern_1"
      COLOR 200 100 50
    END
    LABEL
      FONT "geofonts"
      SIZE 10
    END
  END
  
  CLASS
    NAME "Class 2"
    EXPRESSION ([GMU_CODE] eq "456")
    STYLE
      SYMBOL "bedrock_pattern_2"
      COLOR 150 150 200
    END
    LABEL
      FONT "geofonts1"
      SIZE 12
    END
  END
END
""")
        return mapfile
    
    def test_extract_symbols(self, sample_mapfile):
        """Should extract symbol names from mapfile"""
        handler = ManualMapfileHandler()
        usage = handler.extract_symbols_from_mapfile(sample_mapfile)
        
        assert len(usage.symbols_used) == 2
        assert "bedrock_pattern_1" in usage.symbols_used
        assert "bedrock_pattern_2" in usage.symbols_used
    
    def test_extract_fonts(self, sample_mapfile):
        """Should extract font names from mapfile"""
        handler = ManualMapfileHandler()
        usage = handler.extract_symbols_from_mapfile(sample_mapfile)
        
        assert len(usage.fonts_used) == 2
        assert "geofonts" in usage.fonts_used
        assert "geofonts1" in usage.fonts_used
    
    def test_validate_expected_symbols(self, sample_mapfile):
        """Should validate expected symbols are present"""
        handler = ManualMapfileHandler()
        
        # All expected symbols present
        is_valid = handler.validate_expected_symbols(
            sample_mapfile,
            ["bedrock_pattern_1", "bedrock_pattern_2"]
        )
        assert is_valid
        
        # Missing expected symbol
        is_valid = handler.validate_expected_symbols(
            sample_mapfile,
            ["bedrock_pattern_1", "bedrock_pattern_3"]
        )
        assert not is_valid
    
    def test_merge_symbols(self, tmp_path):
        """Should merge symbols from multiple mapfiles"""
        # Create two mapfiles
        mapfile1 = tmp_path / "map1.map"
        mapfile1.write_text('LAYER\nCLASS\nSTYLE\nSYMBOL "sym1"\nEND\nEND\nEND')
        
        mapfile2 = tmp_path / "map2.map"
        mapfile2.write_text('LAYER\nCLASS\nSTYLE\nSYMBOL "sym2"\nSYMBOL "sym1"\nEND\nEND\nEND')
        
        handler = ManualMapfileHandler()
        handler.extract_symbols_from_mapfile(mapfile1)
        handler.extract_symbols_from_mapfile(mapfile2)
        
        all_symbols = handler.merge_manual_symbols()
        assert len(all_symbols) == 2
        assert "sym1" in all_symbols
        assert "sym2" in all_symbols


# =============================================================================
# ClassIdentifier Tests
# =============================================================================


class TestClassIdentifier:
    """Test stable class identifier generation"""
    
    def test_single_field_identifier(self):
        """VALUE_BASED strategy for single field"""
        identifier = ClassIdentifier.from_single_field(
            layer_path="Surfaces/GC_SURFACES",
            field_name="GMU_CODE",
            field_value="15801003",
            class_index=42,
            label="Granite du Mont-Blanc",
        )
        
        assert identifier.strategy == IdentifierStrategy.VALUE_BASED
        assert identifier.canonical_id == "gmu_code_15801003"
        assert identifier.field_values == ("15801003",)
        assert identifier.field_names == ("GMU_CODE",)
        assert identifier.class_index == 42
    
    def test_multiple_fields_identifier(self):
        """MULTI_VALUE strategy for multiple fields"""
        identifier = ClassIdentifier.from_multiple_fields(
            layer_path="Points/GC_POINT_OBJECTS",
            field_names=["KIND", "HSUR_STATUS"],
            field_values=["12501001", "1"],
            class_index=5,
            label="Source active",
        )
        
        assert identifier.strategy == IdentifierStrategy.MULTI_VALUE
        assert "kind_12501001" in identifier.canonical_id
        assert "hsur_status_1" in identifier.canonical_id
        assert identifier.field_values == ("12501001", "1")
    
    def test_expression_identifier(self):
        """EXPRESSION strategy for complex expressions"""
        identifier = ClassIdentifier.from_expression(
            layer_path="Fossils/GC_FOSSILS",
            expression="KIND = 14601006 AND LFOS_DIVISION = 'Triassic'",
            class_index=3,
            label="Fossiles triassiques",
        )
        
        assert identifier.strategy == IdentifierStrategy.EXPRESSION
        assert identifier.canonical_id.startswith("kind_14601006")
        assert identifier.field_values[0] == "KIND = 14601006 AND LFOS_DIVISION = 'Triassic'"
    
    def test_label_identifier(self):
        """LABEL strategy when only label available"""
        identifier = ClassIdentifier.from_label(
            layer_path="Complex/LAYER",
            label="Granite à biotite et muscovite",
            class_index=10,
        )
        
        assert identifier.strategy == IdentifierStrategy.LABEL
        assert identifier.canonical_id == "granite_biotite_et_muscovite"
        assert identifier.label == "Granite à biotite et muscovite"
    
    def test_index_identifier(self):
        """INDEX strategy as fallback"""
        identifier = ClassIdentifier.from_index(
            layer_path="Unknown/LAYER",
            class_index=99,
            label="Unknown class",
        )
        
        assert identifier.strategy == IdentifierStrategy.INDEX
        assert identifier.canonical_id == "idx_99"
        assert identifier.class_index == 99
    
    def test_to_key(self):
        """to_key() should generate unique key"""
        identifier = ClassIdentifier.from_single_field(
            layer_path="Surfaces/GC_SURFACES",
            field_name="GMU_CODE",
            field_value="15801003",
        )
        
        key = identifier.to_key()
        assert key == "Surfaces/GC_SURFACES::gmu_code_15801003"
    
    def test_special_characters_sanitized(self):
        """Special characters should be sanitized"""
        identifier = ClassIdentifier.from_single_field(
            layer_path="Test/LAYER",
            field_name="CODE",
            field_value="AB-123/456",
        )
        
        # Should sanitize to valid identifier
        assert identifier.canonical_id == "code_ab_123_456"
    
    def test_null_value_handling(self):
        """NULL values should be handled correctly"""
        identifier = ClassIdentifier.from_single_field(
            layer_path="Test/LAYER",
            field_name="CODE",
            field_value="<Null>",
        )
        
        assert identifier.canonical_id == "code_null"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_config_workflow(self, tmp_path):
        """Test complete workflow: YAML → Config → Filters"""
        config_file = tmp_path / "full_config.yaml"
        config_file.write_text("""
global:
  symbol_field: SYMBOL

layers:
  - gpkg_layer: auto_layer
    gcover_layer: auto
    
    classifications:
      - style_file: test.lyrx
        mapfile_config:
          generation_mode: auto
          symbol_adjustments:
            point_size_multiplier: 1.5
  
  - gpkg_layer: manual_layer
    gcover_layer: manual
    
    classifications:
      - style_file: test.lyrx
        mapfile_config:
          generation_mode: manual
          manual_mapfile_path: manual.map
          expected_symbols: [sym1, sym2]
  
  - gpkg_layer: disabled_layer
    gcover_layer: disabled
    
    classifications:
      - style_file: test.lyrx
        mapfile_config:
          generation_mode: disabled
          reason: "Not ready"
""")
        
        config = BatchClassificationConfig(config_file)
        
        # Test filtering
        assert len(config.get_auto_layers()) == 1
        assert len(config.get_manual_layers()) == 1
        assert len(config.get_disabled_layers()) == 1
        
        # Test config access
        auto_layer = config.get_auto_layers()[0]
        assert auto_layer.classifications[0].mapfile_config.symbol_adjustments.point_size_multiplier == 1.5
        
        manual_layer = config.get_manual_layers()[0]
        assert manual_layer.classifications[0].mapfile_config.expected_symbols == ["sym1", "sym2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
