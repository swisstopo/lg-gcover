#!/usr/bin/env python3
"""Report competing classifications: rules that overlap inside a .lyrx, and features
labelled by two or more classifications of the same layer.

Complement of check_classification_coverage.py: that one finds features covered by
no classification, this one finds features covered by several. When the symbol field
is written, the first rule/classification wins (drop_duplicates / preserve_existing),
so overlaps are silent in the classified GPKG.

Two levels:
  1. Within one .lyrx: the same value combination mapped to different classes. Static
     check on the rules, no data needed.
  2. Across the classifications of a spatial table: a feature that passes the filter
     AND has a matching rule in two or more active classifications.
"""

import sys
from pathlib import Path

import click
import fiona
import geopandas as gpd
import pandas as pd
from rich.console import Console
from rich.table import Table

from check_classification_coverage import SKIP_LAYERS, _apply_filter
from gcover.publish import vectorized_classification as vc
from gcover.publish.esri_classification_extractor import extract_lyrx_complete
from gcover.publish.style_config import BatchClassificationConfig

console = Console(record=True)


def _class_name(cls_cfg) -> str:
    return cls_cfg.classification_name or Path(cls_cfg.style_file).stem


def _load_classification(cls_cfg):
    """Pick the LayerClassification of the .lyrx the way the batch classifier does."""
    import difflib

    classifications = extract_lyrx_complete(cls_cfg.style_file, display=False)
    if cls_cfg.classification_name:
        for c in classifications:
            if difflib.get_close_matches(cls_cfg.classification_name, [c.layer_name], n=1, cutoff=0.8):
                return c
        return None
    return classifications[0] if len(classifications) == 1 else None


def _lyrx_conflicts(classification, cls_cfg) -> pd.DataFrame:
    """Value combinations of one .lyrx that map to more than one (symbol, label)."""
    field_names = [f.name for f in classification.fields]
    lookup = vc.build_lookup_table(
        classification, cls_cfg.symbol_prefix or "x", field_names,
        identifier_field=cls_cfg.identifier_field,
    )
    if lookup.empty:
        return lookup
    target = lookup.drop_duplicates(subset=field_names + ["_SYMBOL", "_LABEL"])
    return target[target.duplicated(subset=field_names, keep=False)]


def _rule_mask(gdf, classification, cls_cfg, treat_zero, field_types) -> pd.Series:
    """True where a feature has a matching rule (same normalisation as the classifier)."""
    quiet, vc.console.quiet = vc.console.quiet, True
    try:
        app = vc.VectorizedClassificationApplicator(
            classification=classification,
            symbol_prefix=cls_cfg.symbol_prefix,
            field_mapping=cls_cfg.fields,
            identifier_field=cls_cfg.identifier_field,
            treat_zero_as_null=treat_zero,
        )
    finally:
        vc.console.quiet = quiet
    ok, missing = app.check_required_fields(gdf)
    if not ok:
        raise KeyError(f"missing fields {missing}")
    if app.lookup_df.empty:
        return pd.Series(False, index=gdf.index)

    keys = pd.DataFrame(index=gdf.index)
    for gpkg_field, class_field in zip(app.gpkg_field_names, app.classification_field_names):
        col = gdf[gpkg_field]
        if str(field_types.get(gpkg_field, "")).lower().startswith("int"):
            col = pd.to_numeric(col, errors="coerce").round().astype("Int64")
        keys[class_field] = vc.normalize_join_column(col, treat_zero)
    names = app.classification_field_names
    known = pd.MultiIndex.from_frame(app.lookup_df[names])
    return pd.Series(pd.MultiIndex.from_frame(keys[names]).isin(known), index=gdf.index)


@click.command()
@click.argument("classified_gpkg", type=click.Path(exists=True))
@click.argument("config", type=click.Path(exists=True))
@click.option("--output-gpkg", type=click.Path(), default=None,
              help="Write overlapping features to this GPKG (column 'matched_classes')")
@click.option("--report", type=click.Path(), default=None,
              help="Write plain-text summary to this file")
@click.option("--top-n", default=15, show_default=True,
              help="Number of most frequent class combinations / conflicts to show")
@click.option("--styles-dir", type=click.Path(exists=True), default=None,
              help="Directory holding the .lyrx files (default: next to the config)")
@click.option("--layer", "only_layer", default=None, help="Only check this GPKG layer")
@click.option("--fail-on-overlap", is_flag=True,
              help="Exit with status 1 if any overlap is found")
def main(classified_gpkg, config, output_gpkg, report, top_n, styles_dir, only_layer, fail_on_overlap):
    """Report competing classifications (inside a .lyrx and across a layer)."""
    gpkg_path = Path(classified_gpkg)
    out_gpkg = Path(output_gpkg) if output_gpkg else None
    if out_gpkg and out_gpkg.exists():
        out_gpkg.unlink()

    cfg = BatchClassificationConfig(Path(config), Path(styles_dir) if styles_dir else None)
    all_layers = [l for l in fiona.listlayers(str(gpkg_path)) if l not in SKIP_LAYERS and only_layer in (None, l)]

    summaries = []
    any_overlap = False

    for layer_name in all_layers:
        layer_cfg = cfg.get_layer_config(layer_name)
        if not layer_cfg:
            continue
        active = [c for c in layer_cfg.classifications if c.active]
        if not active:
            continue
        field_types = layer_cfg.field_types or {}
        console.print(f"\nLayer: [bold cyan]{layer_name}[/]")

        # --- Level 1: rule conflicts inside each .lyrx ---------------------------
        loaded = []  # (name, cls_cfg, classification, treat_zero)
        n_conflicts = 0
        for cls_cfg in active:
            name = _class_name(cls_cfg)
            try:
                classification = _load_classification(cls_cfg)
            except Exception as exc:
                console.print(f"  [yellow]⚠[/]  {name}: cannot load {cls_cfg.style_file} ({exc})")
                continue
            if classification is None:
                console.print(f"  [yellow]⚠[/]  {name}: classification not found in {Path(cls_cfg.style_file).name}")
                continue
            treat_zero = cls_cfg.treat_zero_as_null
            loaded.append((name, cls_cfg, classification,
                           cfg.treat_zero_as_null if treat_zero is None else treat_zero))

            conflicts = _lyrx_conflicts(classification, cls_cfg)
            if conflicts.empty:
                continue
            field_names = [f.name for f in classification.fields]
            n_keys = len(conflicts.groupby(field_names, dropna=False))
            n_conflicts += n_keys
            console.print(f"  [bold yellow]⚠ {name}[/]: {n_keys} value combination(s) mapped to several classes "
                          f"([dim]{Path(cls_cfg.style_file).name}[/])")
            table = Table(header_style="bold cyan", show_lines=False)
            table.add_column("Key", max_width=60)
            table.add_column("Competing classes")
            for i, (key, grp) in enumerate(conflicts.groupby(field_names, dropna=False, sort=False)):
                if i >= top_n:
                    console.print(f"  ... and {n_keys - top_n} more")
                    break
                key = key if isinstance(key, tuple) else (key,)
                table.add_row(" | ".join(str(k) for k in key), " + ".join(grp["_LABEL"].astype(str)))
            console.print(table)

        # --- Level 2: features labelled by 2+ classifications --------------------
        n_overlap, total = 0, 0
        if len(loaded) >= 2:
            gdf = gpd.read_file(str(gpkg_path), layer=layer_name)
            total = len(gdf)
            names = [n for n, *_ in loaded]
            matches = pd.DataFrame(False, index=gdf.index, columns=names)
            for name, cls_cfg, classification, treat_zero in loaded:
                try:
                    rules = _rule_mask(gdf, classification, cls_cfg, treat_zero, field_types)
                except KeyError as exc:
                    console.print(f"  [yellow]⚠[/]  {name}: {exc}, skipped")
                    continue
                in_filter = _apply_filter(gdf, cls_cfg.filter or classification.definition_expression,
                                          field_types).index
                matches[name] = rules & matches.index.isin(in_filter)

            n_matched = matches.sum(axis=1)
            overlap = n_matched >= 2
            n_overlap = int(overlap.sum())
            status = "[green]✓ no overlap[/]" if not n_overlap else f"[bold yellow]⚠ {n_overlap:,} overlapping[/]"
            console.print(f"  Across classifications: {status} | Total: {total:,} | "
                          f"Max classes per feature: {int(n_matched.max()) if total else 0}")

            if n_overlap:
                combos = matches[overlap].apply(lambda r: " + ".join(r.index[r.values]), axis=1)
                counts = combos.value_counts()
                table = Table(title="Overlapping class combinations", header_style="bold cyan", show_lines=False)
                table.add_column("Count", justify="right", width=9)
                table.add_column("Classifications")
                for combo, n in counts.head(top_n).items():
                    table.add_row(f"{n:,}", combo)
                console.print(table)
                if len(counts) > top_n:
                    console.print(f"  ... and {len(counts) - top_n} more combinations")
                if out_gpkg:
                    out = gdf[overlap].copy()
                    out["matched_classes"] = combos
                    out.to_file(str(out_gpkg), layer=layer_name, driver="GPKG", append=out_gpkg.exists())
        elif len(loaded) == 1 and not n_conflicts:
            console.print("  [green]✓ no rule conflict[/] (single classification)")

        if n_conflicts or n_overlap:
            any_overlap = True
        summaries.append((layer_name, total, n_conflicts, n_overlap))

    summary = Table(title="Overlap Summary", header_style="bold cyan", show_lines=False)
    summary.add_column("Layer", style="dim", width=28)
    summary.add_column("Rule conflicts (.lyrx)", justify="right", width=22)
    summary.add_column("Features in 2+ classes", justify="right", width=22)
    for layer_name, total, n_conf, n_overlap in summaries:
        summary.add_row(layer_name, f"{n_conf:,}" if n_conf else "—", f"{n_overlap:,}" if n_overlap else "—",
                        style="yellow" if n_conf or n_overlap else None)
    console.print("\n")
    console.print(summary)

    if out_gpkg and out_gpkg.exists():
        console.print(f"\n[green]✓[/] Overlapping features written to [bold]{out_gpkg}[/]")
    if report:
        Path(report).write_text(console.export_text())
        console.print(f"[green]✓[/] Report written to [bold]{report}[/]")

    if any_overlap and fail_on_overlap:
        sys.exit(1)


if __name__ == "__main__":
    main()
