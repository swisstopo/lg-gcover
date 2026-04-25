#!/usr/bin/env python3
"""Report classification coverage of the classified GPKG and extract unclassified features."""

import re
import sys
from pathlib import Path

import click
import fiona
import geopandas as gpd
import pandas as pd
from rich.console import Console
from rich.table import Table

from gcover.publish.style_config import BatchClassificationConfig
from gcover.publish.utils import translate_esri_to_pandas

console = Console(record=True)

SKIP_LAYERS = {"aux_points_aspect"}
EXCLUDE_COLUMNS = {"geometry", "fid", "ogc_fid"}


def _fields_from_filters(classifications) -> list[str]:
    """Extract column names referenced in filter expressions."""
    tokens = set()
    for cls in classifications:
        if cls.filter:
            tokens |= set(re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', cls.filter))
    return sorted(tokens)


def _fallback_fields(gdf: gpd.GeoDataFrame, symbol_field: str, max_cols: int = 8) -> list[str]:
    """When no filter fields exist, pick the most discriminating non-system columns."""
    skip = EXCLUDE_COLUMNS | {symbol_field, "label", "map_symbol"}
    candidates = [c for c in gdf.columns if c not in skip and c.lower() not in skip]
    # Rank by number of unique values descending — most discriminating first
    ranked = sorted(candidates, key=lambda c: gdf[c].nunique(), reverse=True)
    return ranked[:max_cols]


def _apply_filter(gdf: gpd.GeoDataFrame, esri_filter: str | None, field_types: dict) -> gpd.GeoDataFrame:
    """Apply an ESRI filter expression quietly (no console output)."""
    if not esri_filter:
        return gdf
    try:
        work = gdf.copy()
        for col, dtype in field_types.items():
            if col in work.columns and str(dtype).lower() in ("int", "integer"):
                work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
        return work.query(translate_esri_to_pandas(esri_filter))
    except Exception as exc:
        console.print(f"  [yellow]⚠[/]  Filter skipped ({exc}): [dim]{esri_filter[:80]}[/]")
        return gdf


@click.command()
@click.argument("classified_gpkg", type=click.Path(exists=True))
@click.argument("config", type=click.Path(exists=True))
@click.option("--output-gpkg", type=click.Path(), default=None,
              help="Write unclassified features to this GPKG")
@click.option("--report", type=click.Path(), default=None,
              help="Write plain-text summary to this file")
@click.option("--top-n", default=15, show_default=True,
              help="Number of top unclassified attribute patterns to show")
def main(classified_gpkg, config, output_gpkg, report, top_n):
    """Report classification coverage and extract unclassified features."""
    gpkg_path = Path(classified_gpkg)
    out_gpkg = Path(output_gpkg) if output_gpkg else None
    if out_gpkg and out_gpkg.exists():
        out_gpkg.unlink()

    cfg = BatchClassificationConfig(Path(config))
    symbol_field = cfg.symbol_field

    all_layers = [l for l in fiona.listlayers(str(gpkg_path)) if l not in SKIP_LAYERS]

    overall_total = 0
    overall_unclassified = 0
    layer_summaries = []
    any_incomplete = False

    for layer_name in all_layers:
        gdf = gpd.read_file(str(gpkg_path), layer=layer_name)
        layer_cfg = cfg.get_layer_config(layer_name)

        total = len(gdf)
        unclassified = gdf[gdf[symbol_field].isna()]
        n_unclassified = len(unclassified)
        n_classified = total - n_unclassified
        coverage = n_classified / total * 100 if total > 0 else 100.0

        overall_total += total
        overall_unclassified += n_unclassified
        if n_unclassified:
            any_incomplete = True

        status = (
            "[green]✓ OK[/]" if n_unclassified == 0
            else f"[bold red]✗ INCOMPLETE ({n_unclassified:,} missing)[/]"
        )
        console.print(f"\nLayer: [bold cyan]{layer_name}[/] — {status}")
        console.print(f"  Total: {total:,} | Classified: {n_classified:,} | Coverage: {coverage:.1f}%")

        breakdown_rows = []
        if layer_cfg:
            field_types = layer_cfg.field_types or {}
            active = [c for c in layer_cfg.classifications if c.active]

            table = Table(title="Classification Breakdown", header_style="bold cyan", show_lines=False)
            table.add_column("Classification", max_width=42)
            table.add_column("Filter", max_width=34)
            table.add_column("Filtered", justify="right", width=10)
            table.add_column("Matched", justify="right", width=9)
            table.add_column("Coverage", justify="right", width=10)

            for cls_cfg in active:
                filtered = _apply_filter(gdf, cls_cfg.filter, field_types)
                n_filtered = len(filtered)
                n_matched = int(filtered[symbol_field].notna().sum())
                cls_cov = n_matched / n_filtered * 100 if n_filtered else 0.0
                cov_str = f"{cls_cov:.1f}%"
                name = cls_cfg.classification_name or Path(cls_cfg.style_file).stem
                flt = cls_cfg.filter or "—"
                if len(flt) > 33:
                    flt = flt[:30] + "…"
                row_style = "" if cls_cov >= 100 else "yellow" if cls_cov >= 95 else "red"
                table.add_row(name, flt, f"{n_filtered:,}", f"{n_matched:,}", cov_str,
                              style=row_style if row_style else None)
                breakdown_rows.append((name, n_filtered, n_matched, cls_cov))

            console.print(table)

        # Top unclassified patterns
        if n_unclassified > 0 and layer_cfg:
            raw_fields = _fields_from_filters(layer_cfg.classifications)
            key_fields = [f for f in raw_fields if f in unclassified.columns]

            if not key_fields:
                key_fields = _fallback_fields(unclassified, symbol_field)
                console.print(f"  [dim](no filter expressions — showing most discriminating columns)[/]")

            if key_fields:
                patterns = (
                    unclassified[key_fields]
                    .fillna("<NULL>")
                    .astype(str)
                    .groupby(key_fields)
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Count", ascending=False)
                )
                shown = patterns.head(top_n)
                rest = len(patterns) - len(shown)
                rest_count = patterns.iloc[top_n:]["Count"].sum() if rest > 0 else 0

                console.print(f"\n⚠️   Unclassified feature patterns (top {top_n}):")
                console.print(f"   Fields: {' | '.join(key_fields)}\n")

                pat_table = Table(show_header=True, header_style="dim", show_lines=False, box=None)
                pat_table.add_column("Count", justify="right", width=7)
                for f in key_fields:
                    pat_table.add_column(f, max_width=16)
                for _, row in shown.iterrows():
                    pat_table.add_row(str(int(row["Count"])), *[str(row[f]) for f in key_fields])
                console.print(pat_table)

                if rest > 0:
                    console.print(f"\n  ... and {rest} more patterns ({rest_count:,} features)")

        layer_summaries.append((layer_name, total, n_classified, n_unclassified, coverage))

        # Export unclassified features
        if n_unclassified > 0 and out_gpkg:
            unclassified.to_file(str(out_gpkg), layer=layer_name, driver="GPKG",
                                 append=out_gpkg.exists())

    # Overall summary
    overall_coverage = (overall_total - overall_unclassified) / overall_total * 100 if overall_total else 100.0
    console.print("\n")

    summary_table = Table(title="Overall Summary", header_style="bold cyan", show_lines=False)
    summary_table.add_column("Layer", style="dim", width=28)
    summary_table.add_column("Total", justify="right", width=10)
    summary_table.add_column("Classified", justify="right", width=12)
    summary_table.add_column("Missing", justify="right", width=9)
    summary_table.add_column("Coverage", justify="right", width=10)

    for layer_name, total, n_classified, n_unclassified, coverage in layer_summaries:
        cov_str = f"{coverage:.1f}%"
        style = None if n_unclassified == 0 else "yellow" if coverage >= 95 else "red"
        summary_table.add_row(layer_name, f"{total:,}", f"{n_classified:,}",
                              f"{n_unclassified:,}" if n_unclassified else "—",
                              cov_str, style=style)

    summary_table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{overall_total:,}[/]",
        f"[bold]{overall_total - overall_unclassified:,}[/]",
        f"[bold red]{overall_unclassified:,}[/]" if overall_unclassified else "[bold]—[/]",
        f"[bold]{overall_coverage:.1f}%[/]",
    )
    console.print(summary_table)

    if out_gpkg and any_incomplete:
        console.print(f"\n[green]✓[/] Unclassified features written to [bold]{out_gpkg}[/]")

    if report:
        Path(report).write_text(console.export_text())
        console.print(f"[green]✓[/] Report written to [bold]{report}[/]")

    if any_incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()
