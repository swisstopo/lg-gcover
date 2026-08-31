#!/usr/bin/env python3
"""
Check that each layer's config `filter:` — a single-field pre-filter used to
build the mapfile DATA statement's WHERE clause — covers every value used by
that same layer's own active .lyrx classification classes.

A filter missing a value silently drops matching features from the mapfile
output before they ever reach CLASS matching. This is exactly what happened
to unco_chrono_b's RUNC_LITHO filter: a new class ("Hangschutt/strukturierter
Hangschutt, Holozän") introduced litho code 15101084, but the hand-maintained
`filter: RUNC_LITHO IN (15101009, 15101015)` was never updated to match.

Only simple single-field filters — `FIELD IN (...)`, `FIELD NOT IN (...)`,
`FIELD=value` — where FIELD is also one of the layer's own .lyrx
classification fields are checked. Compound/multi-field filters (e.g.
combining KIND with a depth threshold to split one code across two layers)
encode business logic beyond pure classification coverage and are skipped
rather than guessed at.

Usage:
    python scripts/check_filter_coverage.py config/esri_classifier_denormalized_geocover.yaml \\
        --styles-dir /path/to/Styles/2026-08-25/styles
"""

import re
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcover.publish.esri_classification_extractor import extract_lyrx_complete

console = Console()

FILTER_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(NOT\s+IN|IN|==|=)\s*\(?\s*([^()]+?)\s*\)?\s*$",
    re.IGNORECASE,
)


def parse_simple_filter(expr: str):
    """Return (field, op, {values}) for a simple single-field filter, or None.

    None means "not a simple single-field filter" (e.g. it combines multiple
    fields with AND/OR) — the caller should skip it rather than guess.
    """
    m = FILTER_RE.match(expr)
    if not m:
        return None
    field, op, raw_values = m.groups()
    op = op.upper().replace(" ", " ").strip()
    op = "NOT IN" if op.startswith("NOT") else op
    try:
        values = {int(v.strip()) for v in raw_values.split(",") if v.strip()}
    except ValueError:
        return None
    if not values:
        return None
    return field, op, values


def load_lyrx_classes(lyrx_path: Path):
    """All ClassificationClass objects across every layer defined in lyrx_path."""
    layers = extract_lyrx_complete(lyrx_path, display=False)
    out = []
    for layer in layers:
        field_names = [f.name.upper() for f in layer.fields]
        out.append((field_names, layer.classes))
    return out


def check_classification(gpkg_layer: str, cls_cfg: dict, styles_dir: Path):
    """Return a list of gap dicts for one classification block, or []."""
    filter_expr = cls_cfg.get("filter")
    if not filter_expr:
        return []

    parsed = parse_simple_filter(str(filter_expr))
    if parsed is None:
        return []  # compound/unparseable filter — not checkable, skip

    field, op, filter_values = parsed
    style_file = cls_cfg.get("style_file")
    if not style_file:
        return []

    lyrx_path = styles_dir / Path(style_file).name
    if not lyrx_path.exists():
        console.print(f"  [yellow]⚠[/]  {gpkg_layer}: {style_file} not found under {styles_dir}, skipping")
        return []

    try:
        layer_defs = load_lyrx_classes(lyrx_path)
    except Exception as exc:
        console.print(f"  [yellow]⚠[/]  {gpkg_layer}: failed to parse {style_file}: {exc}")
        return []

    gaps = []
    for field_names, classes in layer_defs:
        if field.upper() not in field_names:
            continue  # filter field isn't one of this layer's classification fields
        idx = field_names.index(field.upper())

        used: dict[int, list[str]] = {}
        for c in classes:
            if not c.visible:
                continue
            for fv in c.field_values:
                if idx >= len(fv):
                    continue
                try:
                    val = int(fv[idx])
                except (ValueError, TypeError):
                    continue
                used.setdefault(val, []).append(c.label)

        if op == "IN":
            missing = sorted(set(used) - filter_values)
        elif op == "NOT IN":
            missing = sorted(set(used) & filter_values)  # classified but excluded
        else:  # "=" / "=="
            missing = sorted(set(used) - filter_values)

        for val in missing:
            gaps.append({
                "gpkg_layer": gpkg_layer,
                "style_file": style_file,
                "field": field,
                "op": op,
                "value": val,
                "classes": sorted(set(used[val])),
            })

    return gaps


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--styles-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory containing the .lyrx style files referenced by style_file:")
def main(config_path: Path, styles_dir: Path):
    """Check config filter: coverage against each layer's active .lyrx classification."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    all_gaps = []
    checked = 0
    for layer in cfg.get("layers", []):
        gpkg_layer = layer.get("gpkg_layer", "?")
        for cls_cfg in layer.get("classifications", []):
            if cls_cfg.get("filter"):
                checked += 1
            all_gaps.extend(check_classification(gpkg_layer, cls_cfg, styles_dir))

    console.print(f"\nChecked {checked} filter(s) with a simple single-field expression.\n")

    if not all_gaps:
        console.print("[green]✓ Every checkable filter covers all values used by its active .lyrx classes.[/green]")
        return

    table = Table(title="Filter coverage gaps", show_lines=False)
    table.add_column("Layer", style="cyan")
    table.add_column("Style file")
    table.add_column("Field")
    table.add_column("Op")
    table.add_column("Missing value", justify="right", style="bold red")
    table.add_column("Used by class(es)")

    for g in all_gaps:
        table.add_row(
            g["gpkg_layer"], g["style_file"], g["field"], g["op"],
            str(g["value"]), ", ".join(g["classes"]),
        )
    console.print(table)
    console.print(f"\n[bold red]✗ {len(all_gaps)} filter coverage gap(s) detected![/bold red]")
    sys.exit(1)


if __name__ == "__main__":
    main()
