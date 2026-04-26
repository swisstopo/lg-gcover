# GeoCover Data Preparation Workflow

This document describes the pipeline that transforms raw ESRI FileGDB sources into
a classified, translated GeoPackage ready for loading into the PostGIS publication database.

## Overview

```
RC1.gdb ──┐
RC2.gdb ──┼──► merge ──► denormalize ──► classify ──► translate ──► PostGIS
custom/ ──┘
```

The four stages are driven by Make targets and the `gcover publish` CLI:

```bash
make merge denormalize classify translate
```

All intermediate and final outputs land in `OUTPUT_DIR` (default: `~/DATA/Derivations/output/R16/`).

---

## Stage 1 — Merge (`make merge`)

**Command:**
```bash
gcover publish merge \
  --rc1 <DELIVERY_DIR>/RC1.gdb \
  --rc2 <DELIVERY_DIR>/RC2.gdb \
  --custom-sources-dir <DELIVERY_DIR> \
  --force-2d \
  --no-clip-to-swiss-border \
  --enrich-mapsheet-links \
  --output <OUTPUT_DIR>/master_R16.gdb
```

**What it does:**

Reads `administrative_zones.gpkg` (layer `mapsheets_sources_only`) to determine which of the ~221 mapsheets is served by RC1, RC2, or a custom GDB (column `SOURCE_RC`). For each mapsheet, it clips the matching features from the appropriate source and writes them into a single master GDB.

**Split vs. keep whole features:**

The merger intentionally does **not** split features along mapsheet boundaries (`--split-by-mapsheet` is off by default). The primary goal of GeoCover is a harmonised, seamless dataset — splitting polygon and line features at administrative boundaries fragments geometries, can introduce topology errors, and would require re-checking topology that was already validated in the RC1/RC2 FileGDBs. Mapsheet-level delivery is handled downstream (subsetting by `MSH_MAP_NBR`), not by cutting features at export time.

Key options used in production:

| Option | Effect |
|---|---|
| `--force-2d` | Strips Z coordinates (avoids FileGDB 3D compatibility issues) |
| `--no-clip-to-swiss-border` | Skips the outer Swiss border clip (mapsheet boundaries are sufficient) |
| `--enrich-mapsheet-links` | Adds `erl_link` / `ber_link` PDF notice links to every feature |
| `--custom-sources-dir` | Picks up any `*.gdb` overrides present alongside RC1/RC2 |

**Output:** `master_R16.gdb`

> Tip: `gcover publish list-sources` shows the RC1/RC2 assignment for every mapsheet.  
> `gcover publish merge --dry-run ...` previews the source assignment table without touching any data.

---

## Stage 2 — Denormalize (`make denormalize`)

**Commands (run automatically by Make):**
```bash
# 1. Import lookup / relation tables from the full RC2 delivery GDB
ogr2ogr -f "OpenFileGDB" -update -overwrite master_R16.gdb RC2.gdb <TABLE>
# repeated for: GC_GEOL_MAPPING_UNIT, GC_LITSTRAT_FORMATION_BANK, GC_CHRONO,
#               GC_EX_GEO_PLG/PNT_EXP_UNIT_GC_GMU, GC_FOSS_SYSTEM_GC_SYSTEM,
#               GC_UN_DEP_CHARACT/COMPOSIT/MAT_TYPE_GC_LITHO

# 2. Denormalize each spatial layer
scripts/denormalize_geocover.py --remove-metadata \
  -o denormalized.gpkg \
  --cd-gdb-path RC2.gdb \
  --tables <layer> \
  master_R16.gdb
# repeated for: fossils, exploit_polygons, exploit_points, linear_objects,
#               point_objects, bedrock, surfaces, unco_deposits
```

**What it does:**

The merge GDB lacks the coded-domain lookup tables (only present in the original ESRI-created GDB). `ogr2ogr` re-imports them from the full RC2 delivery. The denormalization script then joins each spatial layer to its related tables, expanding foreign-key codes into human-readable attributes and flattening the relational model into a self-contained flat layer.

**Output:** `denormalized.gpkg` — one layer per geological feature class, no external dependencies.

---

## Stage 3 — Classify (`make classify`)

**Command:**
```bash
gcover --env sandisk publish apply-config \
  --styles-dir <STYLES_DIR> \
  denormalized.gpkg \
  config/esri_classifier_denormalized_geocover.yaml
```

**What it does:**

Reads the YAML classification config, which maps each GPKG layer to one or more ESRI `.lyrx` style files. For every feature, it evaluates the classification rules extracted from the `.lyrx` (field values, filter expressions) and writes two new columns:

- `SYMBOL` — stable identifier linking the feature to a MapServer `CLASS` or QGIS rule (e.g. `bedrock_15202001`)
- `LABEL` — human-readable display label derived from the ESRI class label

The matching is fully vectorized (pandas merge), processing ~1.2 M features in ~45 s.

**Config structure (simplified):**
```yaml
global:
  symbol_field: SYMBOL
  label_field: LABEL
  treat_zero_as_null: true
layers:
  - gpkg_layer: GC_BEDROCK
    classifications:
      - style_file: styles/Bedrock.lyrx
        classification_name: Bedrock
        symbol_prefix: bedr
```

**Output:** `denormalized_classified.gpkg`

> `--dry-run` validates that all `.lyrx` files referenced in the config exist without writing any data.

---

## Stage 4 — Translate (`make translate`)

**Command:**
```bash
python scripts/translate_gpkg.py \
  -t <TRANSLATION_CSV> \
  --strati-links <STRATI_LINK_XLSX> \
  --lowercase-columns \
  --langs de,fr \
  --output denormalized_classified_translated.gpkg \
  denormalized_classified.gpkg
```

**What it does:**

Joins a geolcode translation table (exported from the geology data model repository) onto every layer to add DE/FR label columns for coded values. Also cross-references the stratigraphic link Excel to attach notice/report URLs where available.

This stage also **normalises all column names to lowercase** (`--lowercase-columns`), which is required for PostgreSQL/PostGIS compatibility. This follows a deliberate naming convention established across the pipeline:

- **UPPERCASE** fields → original attributes coming directly from the source FileGDB (e.g. `UUID`, `GEOLCODE`, `KIND`)
- **lowercase** fields → derived, added, or transformed columns introduced during denormalization, classification, or translation (e.g. `symbol`, `label`, `gmu_code`, `tecto`)

**Output:** `denormalized_classified_translated.gpkg` — the final artefact, ready for PostGIS import.

---

## File Summary

| File | Stage | Description |
|---|---|---|
| `RC1.gdb`, `RC2.gdb` | Input | ESRI FileGDB delivery sources |
| `administrative_zones.gpkg` | Input | Mapsheet boundaries + RC assignments |
| `config/esri_classifier_denormalized_geocover.yaml` | Input | Classification rules (layers → `.lyrx` mappings) |
| `styles/*/\*.lyrx` | Input | ESRI CIM symbol definitions |
| `master_R16.gdb` | Stage 1 | Spatially merged, mapsheet-clipped GDB |
| `denormalized.gpkg` | Stage 2 | Flat layers with coded domains resolved |
| `denormalized_classified.gpkg` | Stage 3 | + `SYMBOL` / `LABEL` classification columns |
| `denormalized_classified_translated.gpkg` | Stage 4 | + DE/FR labels, strati links → **PostGIS input** |

---

## Additional Make Targets

| Target | Purpose |
|---|---|
| `make surfaces-aux` | Generate auxiliary point grids for `surfaces` and `unco_deposits` (80 m spacing) |
| `make mapfiles` | Generate MapServer `.map` files and combined `symbols.sym` from the classified GPKG |
| `make clean-classify` | Remove the classified GPKG (re-triggers from denormalized) |
| `make clean-denormalize` | Remove denormalized + classified GPKGs |
| `make clean-all` | Remove all generated GDB and GPKG artefacts |
| `make merge-diagnostic` | Run the merge diagnostic script without regenerating the master GDB |
| `make geometry-check`   | Check invalid geometries and bedrock/unco coverage |


### Geometries check and coverage

#### Usage

```commandline
make geometry-check

Loading layers from GDB...
  GC_BEDROCK: 297,159 features
  GC_UNCO_DESPOSIT: 234,192 features
  Mapsheets: 220

Geometry validation...
  GC_BEDROCK: 297,146 valid  13 invalid  0 empty
  GC_UNCO_DESPOSIT: 234,186 valid  6 invalid  0 empty
  ⚠  19 issues written to 'invalid_geometries'

Building spatial indexes...

Analysing mapsheets...
  [1136] Liechtenstein ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
                       Coverage & Approach Summary                       
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                                          ┃    Count ┃    RC1 ┃    RC2 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Approach 1 (bedrock fills, unco          │       21 │      3 │     18 │
│ overlaps)                                │          │        │        │
│ Approach 2 (contiguous tiling)           │      151 │    120 │     29 │
│ Mixed / ambiguous                        │       48 │     21 │     27 │
│ Bedrock only (no unco_deposits)          │        0 │      0 │      0 │
│                                          │          │        │        │
│ Mapsheets with gaps > 100 m²             │       64 │     29 │     33 │
│ Invalid / empty geometries               │       19 │        │        │
└──────────────────────────────────────────┴──────────┴────────┴────────┘

✓ Results written to /home/marco/DATA/Derivations/output/R17/geometry_check.gpkg (220 mapsheets, 64 gaps)
✓ Report written to /home/marco/DATA/Derivations/output/R17/geometry_check.txt

```


#### Script structure

Three independent checks per mapsheet

  Load once, iterate 220 times:
  1. Read GC_BEDROCK + GC_UNCO_DESPOSIT fully into memory (~800 MB, one-time cost)
  2. Build a STRtree spatial index on each
  3. Load the 220 mapsheets from gcover.data (mapsheets_sources_only)
  4. For each mapsheet: query both indexes by bbox → clip to exact boundary → compute all metrics below

  ---
  Metrics computed per mapsheet

  ┌───────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │      Metric       │                                          How                                           │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ bedrock_coverage  │ area(unary_union(clipped_bedrock)) / mapsheet_area                                     │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ unco_coverage     │ area(unary_union(clipped_unco)) / mapsheet_area                                        │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ combined_coverage │ area(bedrock_union ∪ unco_union) / mapsheet_area                                       │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ overlap_ratio     │ area(bedrock_union ∩ unco_union) / unco_area — how much of unco sits on top of bedrock │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ gap_area          │ mapsheet_geom.difference(bedrock_union ∪ unco_union) — uncovered area                  │
  └───────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Approach detection logic

  - Approach 1 (bedrock fills, unco overlaps on top): bedrock_coverage > 0.95 AND overlap_ratio > 0.30
  - Approach 2 (contiguous tiling, bedrock + unco together cover the sheet): overlap_ratio < 0.05
  - Mixed / ambiguous: anything in between — worth flagging for manual review

  SOURCE_RC goes into every output row so gaps and ambiguous mapsheets can be directly attributed to RC1 or RC2.

  ---
  Output GPKG — three layers

  - mapsheet_metrics — one row per mapsheet: all metrics + approach classification + SOURCE_RC + MSH_MAP_NBR
  - coverage_gaps — gap polygons with mapsheet number + area (only where gap_area > threshold, default 100 m²)
  - invalid_geometries — one row per bad feature from either layer, with UUID + layer name + explain_validity() reason

  ---
  On the border question

  Since mapsheets are the definitive reference and --no-clip-to-swiss-border was used during merge, outer-border mapsheets may show systematic gaps at the national boundary. Worth adding a border_mapsheet flag (detectable if MSH_MAP_NBR is in a known list, or if the mapsheet
  geometry touches the borders_100m layer) so those false positives can be filtered in the report.


                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                    
● Clean. A few design notes worth knowing before you run it:                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                    
  - ~15 min estimate for the full 220 mapsheets with ~531k features — the progress bar shows mapsheet number + title so you can see it moving                                                                                                                                       
  - The summary table breaks down approach 1/2/mixed by RC1 vs RC2, so you can immediately see if one RC is systematically using a different approach (which could indicate a data consistency issue)                                                                               
  - bedrock only handles the edge case of mapsheets with zero unco_deposits features — classifying those as approach 2 would be misleading                                                                                                                                          
  - The mapsheet_metrics layer in the output GPKG carries all numeric columns, so you can visualise e.g. combined_cov_pct or approach directly in QGIS as a choropleth over the mapsheets 
