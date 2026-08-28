# GeoCover Data Preparation Workflow

This document describes the pipeline that transforms raw ESRI FileGDB sources into
a classified, translated GeoPackage ready for loading into the PostGIS publication database.

## Overview

```
                                   ┌─► merged_master.gdb ─► denormalize ─► classify ─► translate ─► PostGIS
RC1.gdb ──┐                       │      (flat, OGR)
RC2.gdb ──┼──► make merge ────────┤
custom/ ──┘                       └─► merged_final.gdb
                                        (ESRI-schema clone, ArcGIS Pro operators)
```

`make merge` produces **two** GDBs from the same source data in one invocation — they diverge immediately after and are consumed by two different audiences. Everything from Stage 2 onward in this document operates on `merged_master.gdb` only; `merged_final.gdb` is documented separately below.

The stages are driven by Make targets and the `gcover publish` CLI:

```bash
make merge denormalize classify translate
```

All intermediate and final outputs land in `OUTPUT_DIR` (default: `~/DATA/Derivations/output/<RELEASE>/`).

---

## Stage 1 — Merge (`make merge`)

**Command (as run by Make):**
```bash
gcover publish merge \
  --rc1 <SOURCES_DIR>/RC1.gdb \
  --rc2 <SOURCES_DIR>/RC2.gdb \
  --custom-sources-dir <SOURCES_DIR> \
  --admin-zones <GC_MAPSHEET_SOURCE> \
  --mapsheets-layer mapsheet_gc \
  --source-column BKP \
  --force-2d \
  --no-clip-to-swiss-border \
  --enrich-mapsheet-links \
  --exclude-metadata \
  --output <OUTPUT_DIR>/merged_master.gdb \
  --schema-output <OUTPUT_DIR>/merged_final.gdb \
  [--strati-links <STRATI_LINK_XLSX>]
```

**What it does:**

Reads the delivered `GC_MAPSHEET.gpkg` (layer `mapsheet_gc`) to determine which of the ~221 mapsheets is served by RC1, RC2, or a custom GDB. The source-assignment column in this raw delivery file is named `BKP` rather than `SOURCE_RC`; `--source-column BKP` tells the merge which column to read, and it's renamed to the canonical `SOURCE_RC` internally for the rest of the pipeline. (The older `--sources <xlsx>` / bundled `administrative_zones.gpkg` path still exists and already uses `SOURCE_RC` natively — `--source-column` also accepts `SOURCE_QA` for that path.) For each mapsheet, the merge clips the matching features from the appropriate source and writes them into `merged_master.gdb` — then, in the same run, clones an authoritative ESRI-schema GDB and re-injects the same merged data into it to produce `merged_final.gdb` (see below).

Make's `[2/2]` step additionally re-imports `GC_MAPSHEET` itself straight from `<GC_MAPSHEET_SOURCE>` via `ogr2ogr` (with `BER_LINK`/`ERL_LINK` computed from the `BER`/`ERL` flags), since the merge step only clips *spatial* feature classes, not this administrative layer.

**Split vs. keep whole features:**

The merger intentionally does **not** split features along mapsheet boundaries (`--split-by-mapsheet` is off by default). The primary goal of GeoCover is a harmonised, seamless dataset — splitting polygon and line features at administrative boundaries fragments geometries, can introduce topology errors, and would require re-checking topology that was already validated in the RC1/RC2 FileGDBs. Mapsheet-level delivery is handled downstream (subsetting by `MSH_MAP_NBR`), not by cutting features at export time.

Key options used in production:

| Option | Effect |
|---|---|
| `--force-2d` | Strips Z coordinates (avoids FileGDB 3D compatibility issues) |
| `--no-clip-to-swiss-border` | Skips the outer Swiss border clip (mapsheet boundaries are sufficient) |
| `--enrich-mapsheet-links` | Adds `erl_link` / `ber_link` PDF notice links to every feature |
| `--custom-sources-dir` | Picks up any `*.gdb` overrides present alongside RC1/RC2 |
| `--schema-output` | Also produce the ESRI-schema-preserving `merged_final.gdb` (see below) |
| `--strati-links` | Optional; injects `strati_link` on `GC_BEDROCK` via an Excel GMU-code lookup |

**Output:** `merged_master.gdb`, `merged_final.gdb`

> Tip: `gcover publish list-sources` shows the RC1/RC2 assignment for every mapsheet.  
> `gcover publish merge --dry-run ...` previews the source assignment table without touching any data.

### Why two GDBs — preserving the ESRI schema

`merged_master.gdb` is built with GDAL/geopandas, which is what lets `gcover publish merge` clip and recombine RC1/RC2/custom sources at all — but GDAL cannot *create* three ESRI-specific schema elements: coded value domains, relationship classes (junction tables linking spatial layers to attribute tables), and the `GC_ROCK_BODIES` feature dataset grouping. Anything downstream that only needs a flat, open-source-readable GPKG (Stages 2–4 below) is fine without them. ArcGIS Pro operators are not — they need a GDB that still looks and behaves like a genuine ESRI delivery.

`merged_final.gdb` solves this via `patch_schema_gdb()` (`src/gcover/publish/patch_schema.py`), invoked automatically when `--schema-output` is passed:

1. **Clone** — `shutil.copytree` of the authoritative schema GDB (RC2.gdb by default, `--schema-gdb` to override) — a byte-perfect copy, so domains/relationships/feature dataset all survive untouched.
2. **Truncate** — every spatial layer and reference table in the clone is emptied via OGR `DeleteFeature` (SQL `DELETE` isn't supported by the OpenFileGDB write driver).
3. **Append** — `gdal.VectorTranslate(accessMode="append", explodeCollections=True)` bulk-inserts the corresponding layer from `merged_master.gdb`. `explodeCollections` undoes geopandas' MultiPoint promotion so geometry types match the ESRI-authored schema.
4. **Recompute spatial index** — a truncate+append cycle leaves each layer's stored extent as a stale union of the deleted and newly-inserted features' bounds (OGR never recomputes it automatically), which is what makes the FileGDB's spatial index look out of sync afterward. `_refresh_index()` runs the OpenFileGDB driver's `RECOMPUTE EXTENT ON <layer>` and `REPACK <layer>` special SQL statements on every patched layer to fix this.
5. **Extra fields / strati_link / GC_MAPSHEET** — `_MERGE_SOURCE`, `ERL_LINK`, `BER_LINK` are added via `CreateField()` (absent from the ESRI schema clone); `strati_link` is optionally injected on `GC_BEDROCK`; `GC_MAPSHEET` is optionally replaced from the admin-zones source instead of kept from the RC2 clone.

**Known manual step:** at least one ArcGIS-internal topology system table (working name recalled as something like *LINES_TOP* — needs re-confirming, not yet re-identified precisely) is **not enumerable via OGR/GDAL at all**, so `patch_schema_gdb()` cannot drop it programmatically the way it drops `BEDROCK_TOPOLOGY` / `GC_ROCK_BODIES_TOPO` / `T_1_*` (which *are* OGR-visible feature classes, dropped automatically). This table must still be located and deleted manually in ArcGIS Pro / ArcCatalog after `merged_final.gdb` is produced, before handing it to ArcGIS Pro operators.

**Validation:** `make domain-check-final` checks `merged_final.gdb`'s data against RC2's coded domains; `make schema-snapshot-final` + `git diff config/merged_final_schema.json` catches unexpected schema drift (see [Data Checks](#data-checks) below).

---

## Stage 2 — Denormalize (`make denormalize`)

**Commands (run automatically by Make):**
```bash
# 1. Import lookup / relation tables from the full RC2 delivery GDB
ogr2ogr -f "OpenFileGDB" -update -overwrite merged_master.gdb RC2.gdb <TABLE>
# repeated for: GC_GEOL_MAPPING_UNIT, GC_LITSTRAT_FORMATION_BANK, GC_CHRONO,
#               GC_EX_GEO_PLG/PNT_EXP_UNIT_GC_GMU, GC_FOSS_SYSTEM_GC_SYSTEM,
#               GC_UN_DEP_CHARACT/COMPOSIT/MAT_TYPE_GC_LITHO

# 2. Denormalize each spatial layer
scripts/denormalize_geocover.py --remove-metadata \
  -o denormalized.gpkg \
  --cd-gdb-path RC2.gdb \
  --tables <layer> \
  merged_master.gdb
# repeated for: fossils, exploit_polygons, exploit_points, linear_objects,
#               point_objects, bedrock, surfaces, unco_deposits

# 3. Copy GC_MAPSHEET straight through (no denormalization)
ogr2ogr -f GPKG -update -overwrite denormalized.gpkg merged_master.gdb GC_MAPSHEET -nln mapsheet
```

**What it does:**

`merged_master.gdb` (the *flat* output — deliberately, this stage never touches `merged_final.gdb`) lacks the coded-domain lookup tables, since those only exist in the original ESRI-created GDB. `ogr2ogr` re-imports them from the full RC2 delivery. The denormalization script then joins each spatial layer to its related tables via `denormalize_simple_relationship()` (junction-table joins, e.g. `fossils` ↔ `GC_SYSTEM` via `GC_FOSS_SYSTEM_GC_SYSTEM`) or the `"special"`/`"copy"` methods for layers with more complex or no relational structure, expanding foreign-key codes into human-readable attributes and flattening the relational model into a self-contained flat layer.

A relationship-table row should resolve each source feature to **exactly one** lookup entry — a duplicated foreign key in the relationship table (bad source data) would otherwise fan a single feature out into multiple output rows. `denormalize_simple_relationship()` dedupes on the source feature's key before joining and logs a warning naming the affected relationship table when this happens, so a `pipeline-check` count mismatch here (see below) is traceable back to a specific junction table.

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
  --output swissgeocover2d.gpkg \
  denormalized_classified.gpkg
```

**What it does:**

Joins a geolcode translation table (exported from the geology data model repository) onto every layer to add DE/FR label columns for coded values. Also cross-references the stratigraphic link Excel to attach notice/report URLs where available.

This stage also **normalises all column names to lowercase** (`--lowercase-columns`), which is required for PostgreSQL/PostGIS compatibility. This follows a deliberate naming convention established across the pipeline:

- **UPPERCASE** fields → original attributes coming directly from the source FileGDB (e.g. `UUID`, `GEOLCODE`, `KIND`)
- **lowercase** fields → derived, added, or transformed columns introduced during denormalization, classification, or translation (e.g. `symbol`, `label`, `gmu_code`, `tecto`)

**Output:** `swissgeocover2d.gpkg` — the final artefact, ready for PostGIS import.

---

## File Summary

| File | Stage | Description |
|---|---|---|
| `RC1.gdb`, `RC2.gdb` | Input | ESRI FileGDB delivery sources |
| `GC_MAPSHEET.gpkg` | Input | Mapsheet boundaries + RC assignments (layer `mapsheet_gc`, column `BKP`) |
| `config/esri_classifier_denormalized_geocover.yaml` | Input | Classification rules (layers → `.lyrx` mappings) |
| `styles/*/\*.lyrx` | Input | ESRI CIM symbol definitions |
| `merged_master.gdb` | Stage 1 | Spatially merged, mapsheet-clipped GDB — flat, feeds Stages 2–4 |
| `merged_final.gdb` | Stage 1 | Same data, re-injected into an ESRI-schema clone — domains/relationships/feature dataset preserved, handed to ArcGIS Pro operators (not used downstream in this doc) |
| `denormalized.gpkg` | Stage 2 | Flat layers with coded domains resolved |
| `denormalized_classified.gpkg` | Stage 3 | + `SYMBOL` / `LABEL` classification columns |
| `swissgeocover2d.gpkg` | Stage 4 | + DE/FR labels, strati links → **PostGIS input** |

---

## Additional Make Targets

| Target | Purpose |
|---|---|
| `make surfaces-aux` | Generate auxiliary point grids for `surfaces` and `unco_deposits` (80 m spacing) |
| `make mapfiles` | Generate MapServer `.map` files and combined `symbols.sym` from the classified GPKG |
| `make clean-classify` | Remove the classified GPKG (re-triggers from denormalized) |
| `make clean-denormalize` | Remove denormalized + classified GPKGs |
| `make clean-merge` | Remove `merged_master.gdb` + `merged_final.gdb` |
| `make clean-all` | Remove all generated GDB and GPKG artefacts |
| `make merge-diagnostic` | Run the merge diagnostic script without regenerating the master GDB |
| `make pipeline-check` | Feature-count consistency across merge → denormalize → classify → translate (see [Data Checks](#data-checks)) |
| `make geometry-check` | Check invalid geometries and bedrock/unco coverage |
| `make line-topology-check` | Check tectonic line topology against `GC_BEDROCK` boundaries per mapsheet (`MAPSHEET=<nbr>` to restrict) |
| `make polygon-topology-check` | Check polygon micro-gaps and overlaps within/across layers per mapsheet (`MAPSHEET=<nbr>` to restrict) |
| `make coverage-check` | Classification coverage — extracts unclassified features |
| `make domain-check` | Coded-domain compliance: RC1/RC2 self-check, `merged_final.gdb` vs RC2, each custom source vs RC2 |
| `make schema-snapshot-translated` / `make schema-snapshot-final` | Snapshot `swissgeocover2d.gpkg` / `merged_final.gdb` schema to `config/*.json` — diff against the git-committed contract to catch schema drift |


## Data Checks

A handful of `make` targets validate the pipeline output at different points. They're independent of each other — run whichever is relevant to what you just regenerated.

### Pipeline feature-count check (`make pipeline-check`)

Runs `scripts/check_pipeline_counts.py`, which counts features per layer at each stage — `merged_master.gdb` → `denormalized.gpkg` → `denormalized_classified.gpkg` → `swissgeocover2d.gpkg` — and flags any layer whose count changes somewhere it shouldn't.

```bash
make pipeline-check
```

```
                                          Pipeline feature counts
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Layer                        ┃   merged (GDB) ┃   denormalized ┃     classified ┃     translated ┃ OK?   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ bedrock                      │        291,274 │        291,128 │        291,128 │        291,129 │ ✗     │
│ exploit_points               │          1,812 │          1,812 │          1,812 │          1,812 │ ✓     │
│ exploit_polygons             │         13,586 │         13,585 │              — │              — │ ✗     │
│ fossils                      │          2,745 │          2,759 │          2,759 │          2,759 │ ✗     │
│ ...                          │                │                │                │                │       │
└──────────────────────────────┴────────────────┴────────────────┴────────────────┴────────────────┴───────┘
✗ Feature count mismatches detected!
```

**How to read a mismatch, by which stage it appears at:**

- **merged → denormalized**, layer count *increases*: a duplicated foreign key in a relationship table (e.g. `GC_FOSS_SYSTEM_GC_SYSTEM`) is fanning one source feature out into several output rows in `denormalize_simple_relationship()`. Fixed automatically since the dedup was added (`scripts/denormalize_geocover.py`) — check the console/log for a `duplicate ... relationship(s) — keeping first` warning naming the offending junction table.
- **merged → denormalized**, layer count *decreases* (most layers, most runs): expected — denormalize drops rows that fail the relational join or geometry cleanup. This is the "is the merged DB complete" question and is tracked separately from this check; `pipeline-check` only tells you *that* it changed, not whether the drop is legitimate.
- **denormalized → classified**, layer *disappears entirely* (shown as `—`): check whether every classification for that layer is `active: False` in `config/esri_classifier_denormalized_geocover.yaml` — `apply-config` drops a layer outright when it has zero active classifications, rather than passing it through unclassified (`GC_MAPSHEET` gets an explicit passthrough copy in the Makefile instead; nothing else does). This can be an intentional, documented pause on a layer (e.g. `exploit_polygons`, not published because rock-mining-area data isn't updated fast enough to be accurate) — check the `# comment` next to `active: False` before assuming it's a bug.
- **classified → translated**, count changes at all: translate should be a pure enrichment step (adds `_de`/`_fr`/etc. columns, never touches feature count). Any change here points at a duplicate key in one of the join sources — e.g. a duplicated `GeolCode_GMU` in `_Update_stratiLINK.xlsx` fanning out `_strati_links()`'s left join on `GC_BEDROCK` (also fixed with a dedup, in `scripts/translate_gpkg.py`).

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
