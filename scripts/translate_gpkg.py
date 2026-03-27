#!/usr/bin/env python3
"""
translate_gpkg.py
─────────────────
Enrich a GeoPackage by adding translated label columns for every field
whose values look like GeolCodes (integers in [10 000 000 – 20 000 000]).

For each qualifying column `FOO`, up to four new columns are appended:
  FOO_de  FOO_fr  FOO_it  FOO_en

Only languages that have at least 80 % coverage in translations.csv are
added by default (always includes DE and FR as minimum requirement).

Usage
-----
  python translate_gpkg.py [OPTIONS] GPKG

Options
-------
  -t, --translations PATH   Path to translations.csv  [required]
  -l, --layer TEXT          Process only this layer (default: all layers)
  -o, --output PATH         Output GPKG (default: overwrite input)
  --min-coverage FLOAT      Min. fraction of non-null values in a column
                            to consider it a code column [default: 0.5]
  --langs TEXT              Comma-separated language list [default: de,fr,it,en]
  --dry-run                 Report what would be done, don't write anything
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click
import fiona
import geopandas as gpd
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from rich.table import Table

console = Console()

GEOLCODE_MIN = 999_995
GEOLCODE_MAX = 20_000_000

SPECIAL_GEOLCODES = {999_997, 999_998, 999_999}

ATTRIBUTES_TO_IGNORE = [
    "pmod_height",
    "aexp_depth_tot",
    "pcob_altitude",
    "abor_depth_tot",
    "abor_depth_fm_a",
    "abor_depth_fm_b",
    "abor_ref_number",
    "uuid",
    "objectorigin",
    "more_info",
    "printed",
    "map_symbol",
    "label",
    "_merge_source",
    "symbol",
    "rbed_litstrat_link",
    "area_m2",
    "map_angle",
    "geol_mapping_unit_att_uuid",
    "runc_litsrat_link",
    "bearing_deg",
    "strike_deg",
    "azimuth",
    "dip",
    "hcon_depth",
    "length_m",
    "ttec_name",
    "runc_orig_descr",
    "lpro_orig_descr",

]



TRANSLATED_SUFFIXES = ("_desc", "_fr", "_de", "_it", "_en")

FIXED_FIRST_COLUMNS = ["gid", "kind", "kind_de", "kind_fr", "kind_it", "kind_en", "uuid","label", "map_symbol", "label_de", "label_fr"]

PIPE_SEP = " | "

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _first_notnull(series: pd.Series):
    """Return the first non-null value of a Series, or None."""
    notnull = series.dropna()
    return notnull.iloc[0] if len(notnull) else None


def _is_pipe_codes_value(val) -> bool:
    """Return True if val looks like a pipe-separated list of GeolCodes.

    Example: "14511001 | 14511003 | 14511005"
    All tokens must be integers in the valid GeolCode range (or sentinel values).
    """
    if not isinstance(val, str) or PIPE_SEP not in val:
        return False
    try:
        codes = [int(p.strip()) for p in val.split("|") if p.strip()]
    except ValueError:
        return False
    if not codes:
        return False
    return all(
        (GEOLCODE_MIN <= c <= GEOLCODE_MAX) or c in SPECIAL_GEOLCODES
        for c in codes
    )

def _map_pipe_codes(val, lang_series: pd.Series) -> "str | None":
        """Translate a pipe-separated string of GeolCodes.

        Each integer token is looked up in lang_series; untranslated tokens are
        kept as their numeric string so no information is silently lost.

        If all translated values are identical, returns a single value instead
        of repeating it — e.g. "15001025 | 15001025" → "Burdigalien".
        """
        if pd.isna(val):
            return None
        parts = [p.strip() for p in str(val).split("|") if p.strip()]
        translated = []
        for part in parts:
            try:
                label = lang_series.get(int(part))
                translated.append(label if label is not None else part)
            except ValueError:
                translated.append(part)

        if not translated:
            return None

        # Deduplicate while preserving order —
        # "Burdigalien | Burdigalien" → "Burdigalien"
        # "Burdigalien | Aquitanien"  → "Burdigalien | Aquitanien"  (unchanged)
        seen: set = set()
        deduped: list = []
        for t in translated:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        return PIPE_SEP.join(deduped)

def load_translations(path: Path, langs: list[str]) -> pd.DataFrame:
    """Load translations CSV, keep only requested language columns.

    Rows whose first column cannot be coerced to integer are silently dropped
    (count is reported to the caller via a module-level side-channel so that
    the CLI can display it with Rich).
    """
    # Read first column as str so we can coerce cleanly
    df = pd.read_csv(path, dtype=str)
    code_col = df.columns[0]  # whatever it's named (GeolCodeInt, GeolCode, …)

    before = len(df)
    df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
    dropped = df[code_col].isna().sum()
    df = df.dropna(subset=[code_col])
    df[code_col] = df[code_col].astype("int64")
    df = df.rename(columns={code_col: "GeolCodeInt"})

    if dropped:
        console.print(
            f"  [yellow]⚠[/]  Dropped [bold]{dropped}[/] / {before} translation row(s) "
            f"— GeolCode not castable to integer"
        )

    lang_cols = {lang: lang.upper() for lang in langs}
    keep_cols = ["GeolCodeInt"] + [c for c in lang_cols.values() if c in df.columns]
    df = df[keep_cols].copy()

    # Rename to target suffix names  e.g. "DE" → "de"
    df = df.rename(columns={v: k for k, v in lang_cols.items()})
    return df.set_index("GeolCodeInt")


def is_geolcode_column(series: pd.Series, min_coverage: float) -> bool:
    """Return True if the series contains plain GeolCodes OR pipe-separated GeolCodes.

    Pipe-separated detection is based on the first non-null value only, which is
    sufficient because denormalized columns are consistently formatted throughout.
    """
    # ── Fast path: pipe-separated check on the first non-null value ──────────
    first = _first_notnull(series)
    if first is not None and _is_pipe_codes_value(str(first)):
        return (True, "Pipe values")

    # ── Normal path: scalar integer codes ────────────────────────────────────
    # Step 1: convert to numeric (float), coercing errors to NaN
    if series.dtype == object:
        converted = pd.to_numeric(series, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        converted = series.astype("float64")
    else:
        return (False, "Cannot convert to numeric")

    # Step 2: keep only whole numbers (or NaN)
    converted = converted.where(converted.isna() | (converted % 1 == 0))

    # Step 3: cast to nullable Int64
    converted = converted.astype("Int64")

    # Step 4: continue with your existing logic
    valid = converted.dropna()
    if len(valid) == 0:
        return (False, "No valid integer")

    # coverage = len(valid) / len(series)
    # console.print(f"coverage: {coverage}")
    # if coverage < min_coverage:
    #    return False

    # Values inside the normal range
    in_range = valid.between(GEOLCODE_MIN, GEOLCODE_MAX)
    # Values equal to special sentinel codes
    is_special = valid.isin(SPECIAL_GEOLCODES)
    valid_values = (in_range | is_special).sum()

    valid_values_in_range = (valid_values / len(valid))
    is_under_coverage = valid_values_in_range >= min_coverage


    return  (is_under_coverage, "OK" if not is_under_coverage else f"Under {min_coverage}")  #0.90  # 90 % of valid values in range


def _lowercase_gdf_columns(gdf):
    """Rename all non-geometry columns to lowercase.

    The geometry column is intentionally kept as-is so that geopandas
    internals (which track the active geometry by name) remain correct.
    """
    geom_col = gdf.geometry.name
    rename_map = {
        col: col.lower()
        for col in gdf.columns
        if col != geom_col and col != col.lower()
    }
    if rename_map:
        console.print(
            f"  Lowercasing {len(rename_map)} column(s): {list(rename_map.keys())}"
        )
    return gdf.rename(columns=rename_map)

def _reorder_columns(
    gdf: gpd.GeoDataFrame,
    fixed_first: list[str],
) -> gpd.GeoDataFrame:
    """Reorder columns: pinned columns first, then all others alphabetically, geometry last."""
    geom_col = gdf.geometry.name
    head = [c for c in fixed_first if c in gdf.columns]
    rest = sorted(
        c for c in gdf.columns
        if c not in head and c != geom_col
    )
    return gdf[head + rest + [geom_col]]


def enrich_layer(
    gdf: gpd.GeoDataFrame,
    translations: pd.DataFrame,
    langs: list[str],
    min_coverage: float,
    layer_name: str,
) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Add translated columns to GDF; return (enriched_gdf, stats_list)."""
    stats = []
    ignored_cols = []

    # table = Table(title=f"Translating {layer_name}", show_lines=False, header_style="bold cyan")
    # console.print(table)
    non_geo_cols = [c for c in gdf.columns if c != gdf.geometry.name]

    for col in non_geo_cols:
        # Already human-readable text – skip
        if col.lower().endswith(TRANSLATED_SUFFIXES):
            console.print(
                f"[orange]Ignoring {col} (already translated)[/orange]"
            )
            continue

        if col.lower() in ATTRIBUTES_TO_IGNORE:
            console.print(
                f"[orange]Ignoring {col} (column to ignore)[/orange]"
            )
            continue

        is_valid_geolcode, reason = is_geolcode_column(gdf[col], min_coverage)
        if not is_valid_geolcode:
            console.print(
                f"Ignoring {col} ({reason})",style="#FFA500"
            )
            continue

        col_lower = col.lower()
        if col_lower.endswith("_codes"):
            out_prefix = col[: -len("_codes")]
        elif col_lower.endswith("_code"):
            out_prefix = col[: -len("_code")]
        else:
            out_prefix = col

        # Detect whether the column holds pipe-separated values
        first = _first_notnull(gdf[col])
        is_pipe = first is not None and _is_pipe_codes_value(str(first))

        added_langs = []

        for lang in langs:
            if lang not in translations.columns:
                continue
            out_col = f"{out_prefix}_{lang}"
            lang_series = translations[lang]

            if is_pipe:
                mapped = gdf[col].apply(_map_pipe_codes, lang_series=lang_series)
            else:
                codes = (
                    pd.to_numeric(gdf[col], errors="coerce")
                    .where(lambda s: s.isna() | (s % 1 == 0))
                    .astype("Int64")
                )
                mapped = codes.map(lang_series)

            if mapped.notna().sum() == 0:
                continue
            gdf[out_col] = mapped
            added_langs.append(lang)

        if added_langs:
            if is_pipe:
                n_codes = gdf[col].notna().sum()
                ref_lang = next((l for l in langs if l in translations.columns), None)
                n_translated = (
                    gdf[col].apply(_map_pipe_codes, lang_series=translations[ref_lang]).notna().sum()
                    if ref_lang else 0
                )
            else:
                codes = (
                    pd.to_numeric(gdf[col], errors="coerce")
                    .where(lambda s: s.isna() | (s % 1 == 0))
                    .astype("Int64")
                )
                n_codes = codes.notna().sum()
                ref_lang = next((l for l in langs if l in translations.columns), None)
                n_translated = codes.map(translations[ref_lang]).notna().sum() if ref_lang else 0

            stats.append({
                "layer": layer_name,
                "column": col,
                "out_prefix": out_prefix,
                "langs": ", ".join(added_langs),
                "codes_found": int(n_codes),
                "translated": int(n_translated),
                "coverage": f"{n_translated / n_codes * 100:.1f}%" if n_codes else "–",
                "pipe": is_pipe,
            })

    return gdf, stats



def _strati_links(bedrock: gpd.GeoDataFrame,xlsx_path
    ) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Add translated columns to GDF; return (enriched_gdf, stats_list)."""
    stats = []

    strati_link_col = "stratiLINK"

    # --- Lire la table de correspondance depuis le xlsx ---
    strati_df = pd.read_excel(
        xlsx_path,
        usecols=["GeolCode_GMU", strati_link_col],
        dtype={"GeolCode_GMU": "Int64"},  # nullable int, cohérent avec le GPKG
    )
    strati_df = strati_df.rename(columns={strati_link_col: "strati_link"})
    strati_df = strati_df.dropna(subset=["GeolCode_GMU"])


    # --- Merge (left join pour garder toutes les features) ---
    bedrock = bedrock.merge(
        strati_df,
        left_on="GMU_CODE",
        right_on="GeolCode_GMU",
        how="left",
    ).drop(columns=["GeolCode_GMU"])  # supprime la colonne redondante

    return bedrock


def check_min_langs(translations: pd.DataFrame) -> None:
    """Abort if DE and FR are not both present."""
    missing = [lang for lang in ("de", "fr") if lang not in translations.columns]
    if missing:
        console.print(
            f"[bold red]✗[/] translations.csv missing required language(s): {missing}"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("gpkg", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-t",
    "--translations",
    "trans_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to translations.csv (GeolCodeInt, DE, FR, IT, EN, ...)",
)
@click.option(
    "-s",
    "--strati-links",
    "strati_links_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to strati links xlsx",
)
@click.option(
    "-l",
    "--layer",
    "layer_filter",
    default=None,
    help="Process only this layer (default: all layers)",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output GPKG path (default: overwrite input)",
)
@click.option(
    "--min-coverage",
    default=0.4,
    show_default=True,
    type=float,
    help="Min. fraction of non-null values required to treat column as code column",
)
@click.option(
    "--langs",
    default="de,fr,it,en",
    show_default=True,
    help="Comma-separated language suffixes to add",
)
@click.option(
    "--lowercase-columns",
    is_flag=True,
    default=False,
    help=(
        "Normalize all attribute column names to lowercase before writing. "
        "The geometry column is preserved as-is."
    ),
)
@click.option(
    "-c",
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to BatchClassificationConfig YAML. When provided, "
        "label_formulas defined per layer are computed after translation."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would be done without writing anything",
)
def main(
    gpkg: Path,
    trans_path: Path,
    layer_filter: Optional[str],
    output_path: Optional[Path],
    min_coverage: float,
    langs: str,
    lowercase_columns: bool,
    strati_links_path: Optional[Path],
    config_path: Optional[Path],
    dry_run: bool,
) -> None:
    """Add GeolCode translation columns (_de, _fr, …) to every layer of a GPKG."""

    lang_list = [l.strip().lower() for l in langs.split(",")]
    output_path = output_path or gpkg

    console.rule("[bold cyan]GeoCover – GPKG Translation Enrichment")

    # ── Load translations ────────────────────────────────────────────────────
    console.print(f"[dim]Loading translations from[/] [bold]{trans_path}[/]")
    try:
        translations = load_translations(trans_path, lang_list)
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to load translations:[/] {exc}")
        sys.exit(1)

    check_min_langs(translations)

    available_langs = [l for l in lang_list if l in translations.columns]
    console.print(
        f"  [green]✓[/] {len(translations):,} codes loaded  │  "
        f"Languages: {', '.join(f'[bold]{l}[/]' for l in available_langs)}"
    )

    # ── Load label formulas from BatchClassificationConfig (optional) ────────
    batch_config = None
    if config_path:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
            from gcover.publish.style_config import BatchClassificationConfig
            from gcover.publish.esri_classification_applicator import apply_computed_fields
            batch_config = BatchClassificationConfig(config_path)
            # Get the list
            names_with_labels = []
            for lc in batch_config.layers:
                for cls_cfg in lc.classifications:
                    if cls_cfg.label_formulas:
                        names_with_labels.append(cls_cfg.classification_name)
            n_with_labels = len(names_with_labels)

            # Format for display
            names_str = ", ".join(names_with_labels)

            console.print(
                f"  [green]✓[/] Config loaded — "
                f"{n_with_labels} layer(s) ({names_str}) have label_formulas"
            )
        except Exception as exc:
            console.print(f"[bold red]✗ Failed to load config:[/] {exc}")
            sys.exit(1)

    # ── Discover layers ──────────────────────────────────────────────────────
    try:
        all_layers = fiona.listlayers(str(gpkg))
    except Exception as exc:
        console.print(f"[bold red]✗ Cannot read GPKG:[/] {exc}")
        sys.exit(1)

    layers = [layer_filter] if layer_filter else all_layers
    unknown = [l for l in layers if l not in all_layers]
    if unknown:
        console.print(f"[bold red]✗ Unknown layer(s): {unknown}[/]")
        sys.exit(1)

    total_layers = len(layers)
    console.print(f"  [green]✓[/] {total_layers} layer(s) to process\n")

    # ── Process layers ───────────────────────────────────────────────────────
    all_stats: list[dict] = []
    enriched: dict[str, gpd.GeoDataFrame] = {}
    new_labels= {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing layers…", total=total_layers)

        for idx, lyr in enumerate(layers):
            progress.update(task, description=f"{idx + 1}/{total_layers}[cyan]{lyr}[/]")
            try:
                gdf = gpd.read_file(str(gpkg), layer=lyr)
            except Exception as exc:
                console.print(f"  [yellow]⚠[/]  Skipping [bold]{lyr}[/]: {exc}")
                progress.advance(task)
                continue

            if strati_links_path and 'bedrock' in lyr:
                console.print("Adding strati links to Bedrock")
                gdf = _strati_links(gdf, strati_links_path)

            if lowercase_columns:
                console.print(f" [blue] Converting to lower case[/blue]")
                gdf = _lowercase_gdf_columns(gdf)


            gdf, stats = enrich_layer(
                gdf, translations, available_langs, min_coverage, lyr
            )

            # ── Apply label formulas from config ─────────────────────────────
            if batch_config:
                layer_cfg = batch_config.get_layer_config(lyr)
                if layer_cfg:
                    processed_chunks = []

                    # Track which indices we've processed if you need to keep "unmatched" rows later
                    all_matched_indices = []

                    for cls_cfg in layer_cfg.classifications:
                        prefix = cls_cfg.symbol_prefix

                        # Efficiently filter using the string accessor
                        mask = gdf['map_symbol'].str.startswith(prefix, na=False)
                        gdf_subset = gdf[mask].copy()

                        if not gdf_subset.empty:
                            console.print(f"  Computing labels for [bold]{prefix}[/] ({len(gdf_subset)} rows)…")

                            # Apply formulas to the subset
                            gdf_subset = apply_computed_fields(gdf_subset, cls_cfg.label_formulas)
                            processed_chunks.append(gdf_subset)
                            if cls_cfg.label_formulas:
                                new_labels[lyr] = ', '.join(cls_cfg.label_formulas.keys())
                            

                    if processed_chunks:
                        # Re-combine into a GeoDataFrame
                        # Using gpd.GeoDataFrame constructor ensures spatial metadata is locked in
                        final_gdf = gpd.GeoDataFrame(pd.concat(processed_chunks, ignore_index=True))

                        # Restore the original CRS if it was lost during concat
                        final_gdf.set_crs(gdf.crs, allow_override=True, inplace=True)
                        gdf = final_gdf

            enriched[lyr] = gdf
            console.print(f" [green] ✓ Layer {lyr} translated[/green]")
            table = Table(title="Layer → Labels")

            table.add_column("Layer", style="cyan", no_wrap=True)
            table.add_column("Labels", style="green")

            for layer, labels in new_labels.items():
              table.add_row(layer, labels)
            console.print(table)
            
            all_stats.extend(stats)
            progress.advance(task)

    # ── Summary table ────────────────────────────────────────────────────────
    console.print()
    if not all_stats:
        console.print("[yellow]⚠  No translatable code columns found.[/]")
    else:
        table = Table(
            title="Translation summary", show_lines=False, header_style="bold cyan"
        )
        table.add_column("Layer", style="dim")
        table.add_column("Source column")
        table.add_column("Output prefix", style="cyan")
        table.add_column("Languages", style="green")
        table.add_column("Codes", justify="right")
        table.add_column("Translated", justify="right")
        table.add_column("Coverage", justify="right")
        
        for s in all_stats:
            src, out = s["column"], s["out_prefix"]
            out_display = out if out == src else f"[cyan]{out}[/cyan]"
            table.add_row(
                s["layer"],
                src,
                out_display,
                s["langs"],
                str(s["codes_found"]),
                str(s["translated"]),
                s["coverage"],
                
            )

        console.print(table)
        console.print(
            f"\n  [bold green]{len(all_stats)}[/] column(s) enriched across "
            f"[bold green]{len({s['layer'] for s in all_stats})}[/] layer(s)"
        )

    # ── Write output ─────────────────────────────────────────────────────────
    if dry_run:
        console.print("\n[bold yellow]⚑  Dry-run – no file written.[/]")
        return

    if not all_stats:
        console.print("\n[dim]Nothing to write.[/]")
        return

    console.print(f"\n[dim]Writing to[/] [bold]{output_path}[/]")

    # Write all layers (enriched or untouched pass-through)
    write_layers = set(enriched.keys())
    passthrough = [l for l in all_layers if l not in write_layers]

    tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    tmp_path = tmp.name
    tmp.close()  # important: GDAL needs to open it itself

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        total = len(all_layers)
        task = progress.add_task("Writing…", total=total)

        # Write enriched layers
        first_write = True

        for lyr, gdf in enriched.items():
            progress.update(task, description=f"[cyan]{lyr}[/]")
            gdf = _reorder_columns(gdf, fixed_first=FIXED_FIRST_COLUMNS)
            mode = "w" if first_write else "a"
            gdf.to_file(str(tmp_path), layer=lyr, driver="GPKG", mode=mode)
            first_write = False
            progress.advance(task)

        # Pass-through layers that were not processed
        for lyr in passthrough:
            progress.update(task, description=f"[dim]{lyr}[/]")
            try:
                gdf = gpd.read_file(str(gpkg), layer=lyr)
                gdf = _reorder_columns(gdf, fixed_first=FIXED_FIRST_COLUMNS)
                mode = "w" if first_write else "a"
                gdf.to_file(str(tmp_path), layer=lyr, driver="GPKG", mode=mode)
                first_write = False
            except Exception as exc:
                console.print(f"  [yellow]⚠[/]  Could not copy [bold]{lyr}[/]: {exc}")
            progress.advance(task)
    if os.path.exists(tmp_path):
        import shutil

        shutil.copy(tmp_path, output_path)
        os.remove(tmp_path)

    console.print(f"\n  [bold green]✓[/] Done → [bold]{output_path}[/]\n")


if __name__ == "__main__":
    main()
