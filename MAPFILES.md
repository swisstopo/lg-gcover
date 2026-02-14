# MapfileGeneration



## Configuration



### List classifications

```bash
  gcover --env production publish mapserver --list-classifications config/esri_classifier_denormalized_geocover.yaml


  Layer: GC_ROCK_BODIES/GC_EXPLOIT_GEOMAT_PT
    - Exploit_Geomat_Pt    (auto      )
      └─ config/styles/Exploit_Geomat_Pt.lyrx

  Layer: GC_ROCK_BODIES/GC_FOSSILS
    - Fossils              (auto      )
      └─ config/styles/Fossils.lyrx

  Layer: GC_ROCK_BODIES/GC_POINT_OBJECTS
    - Point_Mpla_Planar_Structures (auto      )
      └─ config/styles/Point_Mpla_Planar_Structures.lyrx
    - Point Objects_Achsenfläche (auto      )
      └─ config/styles/Point Objects_Achsenfläche.lyrx
    - Point Objects_Orient (auto      )
      └─ config/styles/Point Objects_Orient.lyrx
    - Point_Objects_Bohrung_Fels_erreicht (auto      )
      └─ config/styles/Point_Objects_Bohrung_Fels_erreicht.lyrx
    - Point_Objects_Sondierung (auto      )
      └─ config/styles/Point_Objects_Sondierung.lyrx
    - Point_Objects_Bohrung_Fels_nicht_erreicht (auto      )
      └─ config/styles/Point_Objects_Bohrung_Fels_nicht_erreicht.lyrx
    - Point Objects_Quelle (auto      )
      └─ config/styles/Point Objects_Quelle.lyrx
    - Point Objects_Erraticker (auto      )
      └─ config/styles/Point Objects_Erraticker.lyrx
```


### ESRI .lyrx symbols overrides

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


## Mapfile generation

### Normal worklfow



