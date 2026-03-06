# Computed Fields

Add or update fields dynamically during classification using expressions.

## Configuration

Computed fields can be defined at **layer level** (applied once before all 
classifications) or at **classification level** (applied before that specific 
classification).

### Layer-level (recommended)
```yaml
layers:
  - gpkg_layer: GC_SURFACES
    computed_fields:
      area_m2: "geometry.area"
      map_angle: "(90 - azimuth) % 360"
    classifications:
      - style_file: styles/Bedrock.lyrx
        # ...
```

### Classification-level
```yaml
layers:
  - gpkg_layer: GC_LINES
    classifications:
      - style_file: styles/Faults.lyrx
        computed_fields:
          line_m: "geometry.length"
```

## Expression Syntax

Uses [pandas.DataFrame.eval()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eval.html) 
with geometry extensions.

### Column Arithmetic

| Expression | Description |
|------------|-------------|
| `column_a + column_b` | Addition |
| `column_a - column_b` | Subtraction |
| `column_a * column_b` | Multiplication |
| `column_a / column_b` | Division |
| `column_a % column_b` | Modulo |
| `column_a ** 2` | Power |
| `-column_a` | Negation |
| `(90 - azimuth) % 360` | Combined operations |

### Geometry Properties

Requires projected CRS (e.g., EPSG:2056) for metric units.

| Expression | Description | Unit |
|------------|-------------|------|
| `geometry.length` | Line length or polygon perimeter | meters |
| `geometry.area` | Polygon area | m² |
| `geometry.centroid.x` | Centroid X coordinate | CRS units |
| `geometry.centroid.y` | Centroid Y coordinate | CRS units |

### Mixed Expressions
```yaml
computed_fields:
  area_km2: "geometry.area / 1_000_000"
  compactness: "4 * 3.14159 * geometry.area / (geometry.length ** 2)"
  normalized_depth: "depth_m / max_depth"
```

## Common Use Cases

### Azimuth to MapServer Angle

ESRI azimuth (clockwise from North) → MapServer angle (counter-clockwise from East):
```yaml
computed_fields:
  map_angle: "(90 - azimuth) % 360"
```

### Bearing (lines)

```yaml
- gpkg_layer: linear_objects
  gcover_layer: GC_ROCK_BODIES/GC_LINEAR_OBJECTS
  layer_type: line
  computed_fields:
    # Basic geometry
    length_m: "geometry.length:round"
    
    # Bearing as integer (0-360)
    bearing_deg: "geometry.bearing:int"
    
    # Strike for geological features (0-180)
    strike_deg: "geometry.strike:int"
    
    # MapServer angle convention (0=East, CCW)
    map_angle: "(90 - geometry.bearing) % 360:int"
    
    # Area in km² rounded
    area_km2: "geometry.area / 1_000_000:round"
    
    # String field
    bearing_label: "geometry.bearing:str"
```

**Output example:**
```
  ✓ Created 'bearing_deg' = geometry.bearing → int
  ✓ Created 'map_angle' = (90 - geometry.bearing) % 360 → int
  ✓ Created 'length_m' = geometry.length → round
### Area/Length for Display
```yaml
computed_fields:
  area_m2: "geometry.area"
  area_ha: "geometry.area / 10_000"
  line_m: "geometry.length"
  line_km: "geometry.length / 1000"
```

### Normalize Values
```yaml
computed_fields:
  azimuth_normalized: "azimuth % 360"
  depth_positive: "-depth"  # if depth stored as negative
```

### Derived Attributes
```yaml
computed_fields:
  age_range_ma: "chrono_base - chrono_top"
  thickness_m: "top_elevation - base_elevation"
```

## Overwriting Fields

A computed field can overwrite an existing column:
```yaml
computed_fields:
  azimuth: "azimuth % 360"  # normalize in place
```

## Concatenating and coalescing

```yaml
computed_fields:
  # Default separator " | "
  litho_combined: "concat(litho_main, litho_sec, litho_ter)"
  # Result: "12000 | 300000 | 350000"
  
  # Custom separator
  litho_pipe: "concat(litho_main, litho_sec, litho_ter, sep='|')"
  # Result: "12000|300000|350000"
  
  # With spaces around pipe
  litho_spaced: "concat(litho_main, litho_sec, litho_ter, sep=' | ')"
  # Result: "12000 | 300000 | 350000"
  
  # Dash separator
  code_combined: "concat(KIND, GMU_CODE, sep='-')"
  # Result: "14334001-15203001"
  
  # Coalesce - first non-null value
  litho_primary: "coalesce(litho_main, litho_sec, litho_ter)"
```


**Output:**
```
  ✓ Created 'litho_combined' = concat(litho_main, litho_sec, litho_ter)
  ✓ Created 'litho_pipe' = concat(litho_main, litho_sec, litho_ter, sep='|')
```
The `coalesce()` function is a bonus — useful for getting the first non-null value from multiple fields.

## Execution Order

1. Load GeoDataFrame
2. Cast `field_types`
3. Apply **layer-level** `computed_fields`
4. For each classification:
   a. Apply **classification-level** `computed_fields`
   b. Apply filter
   c. Match classification rules

## Error Handling

- Invalid expressions log a warning and skip that field
- Missing source columns log a warning and skip that field
- Other fields continue processing

## Limitations

- No string operations (use `field_types` for casting)
- No conditional logic (use `filter` instead)
- No aggregations (min, max, sum across rows)
- Division by zero returns `inf` or `NaN`