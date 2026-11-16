#!/usr/bin/env python3
"""
Compare each config `filter:` with the definition query stored in the block's
own .lyrx (`featureTable.definitionExpression`).

For every classification block that has a simple single-field `filter:`
(`FIELD IN (...)`, `FIELD NOT IN (...)`, `FIELD=value`), the leading simple
clause of the .lyrx definition query is parsed the same way and the two value
sets are compared. Reported:

  * operator mismatch (IN vs NOT IN)
  * values only in the config filter
  * values only in the .lyrx definition query
  * extra clauses in the .lyrx query not representable in the config filter
    (e.g. `AND (PRINTED = 1 OR PRINTED IS NULL)`), as informational notes

Blocks without a config filter, with a compound config filter, or whose .lyrx
has no usable definition query are skipped.

Usage:
    python scripts/check_filter_vs_lyrx.py config/esri_classifier_denormalized_geocover.yaml \\
        --styles-dir /path/to/Styles/2026-11-12/styles
"""

import json
import re
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from check_filter_coverage import parse_simple_filter  # noqa: E402

console = Console()

# Leading simple clause of a definition query; anything after it is "rest".
LEAD_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(NOT\s+IN|IN|==|=)\s*"
    r"(\([^()]*\)|-?\d+)\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_definition_query(expr: str):
    """Return (field, op, {values}, rest) for the leading simple clause, or None."""
    m = LEAD_RE.match(expr)
    if not m:
        return None
    field, op, vals, rest = m.groups()
    parsed = parse_simple_filter(f"{field} {op} {vals}")
    if parsed is None:
        return None
    f, o, v = parsed
    return f, o, v, rest.strip()


def lyrx_definition_queries(lyrx_path: Path):
    """{layer name: definitionExpression} for layers that define one."""
    with lyrx_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for layer in data.get("layerDefinitions", []):
        expr = layer.get("featureTable", {}).get("definitionExpression")
        if expr:
            out[layer.get("name", "")] = expr
    return out


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pick_query(queries: dict, cls_cfg: dict):
    """Pick the definition query belonging to this block, or None if ambiguous."""
    if len(queries) == 1:
        return next(iter(queries.values()))
    key = norm(str(cls_cfg.get("classification_name", "")))
    for name, expr in queries.items():
        if key and norm(name) == key:
            return expr
    return None


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--styles-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory containing the .lyrx style files referenced by style_file:")
def main(config_path: Path, styles_dir: Path):
    """Compare config filter: with the .lyrx definition query."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    diffs, notes, skipped = [], [], []
    checked = 0
    for layer in cfg.get("layers", []):
        gpkg_layer = layer.get("gpkg_layer", "?")
        for cls_cfg in layer.get("classifications", []):
            filter_expr = cls_cfg.get("filter")
            style_file = cls_cfg.get("style_file")
            if not filter_expr or not style_file:
                continue
            if cls_cfg.get("active") is False:
                continue  # inactive block: not published, nothing to check
            where = f"{gpkg_layer} / {Path(style_file).name}"

            cfg_f = parse_simple_filter(str(filter_expr))
            if cfg_f is None:
                skipped.append((where, "compound config filter"))
                continue

            lyrx_path = styles_dir / Path(style_file).name
            if not lyrx_path.exists():
                skipped.append((where, "lyrx not found"))
                continue

            queries = lyrx_definition_queries(lyrx_path)
            expr = pick_query(queries, cls_cfg)
            if expr is None:
                skipped.append((where, "no / ambiguous .lyrx definition query"))
                continue

            lyrx_f = parse_definition_query(expr)
            if lyrx_f is None:
                skipped.append((where, f"unparseable .lyrx query: {expr[:60]}"))
                continue

            checked += 1
            c_field, c_op, c_vals = cfg_f
            l_field, l_op, l_vals, rest = lyrx_f

            if c_field.upper() != l_field.upper():
                diffs.append((where, "field", f"config {c_field} vs lyrx {l_field}"))
                continue
            if c_op != l_op and not ({c_op, l_op} <= {"=", "=="}):
                diffs.append((where, "operator", f"config {c_op} vs lyrx {l_op}"))
                continue
            only_cfg = sorted(c_vals - l_vals)
            only_lyrx = sorted(l_vals - c_vals)
            if only_cfg:
                diffs.append((where, "only in config", ", ".join(map(str, only_cfg))))
            if only_lyrx:
                diffs.append((where, "only in lyrx", ", ".join(map(str, only_lyrx))))
            if rest:
                notes.append((where, rest))

    console.print(f"\nCompared {checked} filter(s) with their .lyrx definition query.\n")

    for where, why in skipped:
        console.print(f"  [yellow]skipped[/] {where}: {why}")
    for where, rest in notes:
        console.print(f"  [dim]note[/] {where}: lyrx also has: {rest}")

    if not diffs:
        console.print("\n[green]✓ All compared filters match their .lyrx definition query.[/green]")
        return

    table = Table(title="Config filter vs .lyrx definition query")
    table.add_column("Block", style="cyan")
    table.add_column("Difference")
    table.add_column("Detail", style="bold red", overflow="fold")
    for row in diffs:
        table.add_row(*row)
    console.print(table)
    console.print(f"\n[bold red]✗ {len(diffs)} difference(s) detected![/bold red]")
    sys.exit(1)


if __name__ == "__main__":
    main()
