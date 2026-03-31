
import pytest

from gcover.publish.generator import MapServerGenerator
import pytest


# from your_module import MapServerGenerator # Import your actual class here

@pytest.fixture
def generator():
    return MapServerGenerator()


def test_build_lang_data_ordering(generator):
    # Setup inputs based on your config
    data_input = "geom FROM (SELECT geom, gid, map_angle, strike_deg, map_symbol FROM geol.geocover_linear_objects) AS subquery USING UNIQUE gid USING SRID=2056"

    lang_fields = {
        "kind_desc": "kind_%lang%",
        "aarc_epoch_desc": "aarc_epoch_%lang%",
        "lpro_litho_desc": "lpro_litho_%lang%"
    }

    # We want a specific order: litho first, then kind
    include_items = "kind_desc,lpro_litho_desc, strike_deg"

    result = generator._build_lang_data(
        data=data_input,
        lang_fields=lang_fields,
        include_items=include_items,
        symbol_field="map_symbol",
        geom_col="geom"
    )

    # Assertions
    # geom FROM (
    #        SELECT gid,
    #               geom,
    #               kind_%lang%       AS kind_desc,
    #               lpro_litho_%lang% AS lpro_litho_desc,
    #               strike_deg,
    #               aarc_epoch_%lang% AS aarc_epoch_desc,
    #               map_angle,
    #               map_symbol,
    #               label
    #        FROM   geol.geocover_linear_objects) AS subquery using UNIQUE gid using srid=2056

    # 1. Check if geom and gid are at the very start
    assert "SELECT gid, geom," in result

    # 2. Check if include_items order is respected (litho before kind)
    # and that they are translated
    litho_idx = result.find("lpro_litho_%lang% AS lpro_litho_desc")
    kind_idx = result.find("kind_%lang% AS kind_desc")
    strike_idx = result.find("strike_deg")

    assert kind_idx  < litho_idx <  strike_idx

    # 3. Check if map_angle (from subquery but not in include_items) exists
    assert "map_angle" in result

    # 4. Check if map_symbol and label (Tail) are at the end
    # map_symbol should be after strike_deg
    symbol_idx = result.find("map_symbol")
    assert strike_idx < symbol_idx

    # 5. Check if the subquery structure is preserved
    assert result.endswith("AS subquery USING UNIQUE gid USING SRID=2056")



def test_empty_include_items(generator):
    data_input = "geom FROM (SELECT geom, gid FROM table) AS sub USING UNIQUE gid"
    lang_fields = {"name": "name_%lang%"}

    result = generator._build_lang_data(data_input, lang_fields)

    # Should still include the translation even if not in include_items
    assert "name_%lang% AS name" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])