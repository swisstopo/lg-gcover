# Mapfile generation

Generating `mapfiles` from the `.lyrx` (or any sources) is tricky. Some layers are easy to generate (e.g. `Bedrock`) while
other layers have complexes symbology.
This approach try to keep the complex, hand-edited mapfiles.


## Configuration


### ESRI .lyrx symbols ovverides

```yaml
layers:
  - gpkg_layer: GC_SURFACES
    classifications:
      - style_file: styles/Surfaces.lyrx
        mapfile_config:           # ← Au niveau classification
          generation_mode: auto   # manual | disabled
          symbol_adjustments:
            point_size_multiplier: 1.5
            line_width_multiplier: 1.3
            dash_pattern_override: [8, 4]
            transparency_override: 50
          metadata:
            gml_include_items: "all"
            wms_title: "Dépôts non-consolidés"
```

✅ point_size_multiplier → SIZE (points, labels)
✅ line_width_multiplier → WIDTH (lines, outlines)
✅ dash_pattern_override → PATTERN (lines, outlines)
✅ transparency_override → OPACITY (fills, markers)


### Manual mapfile generation

```yaml
layers:
  - gpkg_layer: GC_FOSSILS
    classifications:
      - style_file: styles/Fossils.lyrx
        mapfile_config:
          generation_mode: manual  # Don't generate, use manual file
          manual_mapfile_path: mapfiles/manual/fossils.map
          extract_symbols: true  # Still extract symbols for symbols.sym
          # Even in manual mode, we track what symbols SHOULD be there
          expected_symbols:
           - fossil_ammonite
           - fossil_plant
           - fossil_vertebrate
```

### Disabling mapfile generation

```yaml
layers:
  - gpkg_layer: GC_UNCO_DESPOSITS
    classifications:
      - style_files: styles/Unco.lyrx
        mapfile_config:
          generation_mode: disabled  # Don't generate at all, not ready
          reason: "Waiting for classification simplification"

```

### Case 1 : Inline mode (default - comportement actuel)

```yaml
# Config without  `mapfile_config` or `classes_mode` unspecified
classifications:
  - style_file: styles/Bedrock.lyrx
```

**Result** :
```mapfile
LAYER
  NAME "bedrock"
  DATA "..."
  
  CLASS
    NAME "Granite"
    ...
  END
  
  CLASS
    NAME "Gneiss"
    ...
  END
  
END
```

### Case 2 : `regenerate` mode (always regenerate)

```yaml
classifications:
  - style_file: styles/Bedrock.lyrx
    mapfile_config:
      classes_mode: regenerate
```

**Results** :
- Files : `mapfiles/classes/bedrock_classes.inc`
- Overwritten every regeneration

```mapfile
LAYER
  NAME "bedrock"
  DATA "..."
  
  INCLUDE "classes/bedrock_classes.inc"
  
END
```

### Case 3 : `frozen` mode (preserving manual edits)

```yaml
classifications:
  - style_file: styles/Lines.lyrx
    mapfile_config:
      classes_mode: frozen
```

**Comportement** :
```bash
# First time
gcover publish mapserver
# → Creates lines_classes.inc

# Second time
gcover publish mapserver
# → Keep lines_classes.inc (not overwritten)
```

### Case 4 : `staging` mode

```bash
gcover publish mapserver --staging lines
```

**Results** :
- File : `mapfiles/classes/.staging/lines_classes.inc.new`
- Keeps the original file

### Case 5 : `Force` regenerate

```bash
gcover publish mapserver --force-regenerate lines
```

**Results** :
- Overwriten even if `frozen`

## List all classifications

    gcover publish mapserver --list-classifications

## Generate one particular classification

    gcover publish mapserver --staging bedrock_rc1
    gcover publish mapserver --diff bedrock_rc2 --diff-tool meld

## Generate all `frozen` classifications

    gcover publish mapserver --staging-all