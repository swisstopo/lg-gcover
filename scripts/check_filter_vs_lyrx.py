#!/usr/bin/env python3
"""
Compare each config `filter:` with the definition query stored in the block's
own .lyrx (`featureTable.definitionExpression`).

For every classification block that has a `filter:` made of simple clauses
joined by AND (`FIELD IN (...)`, `FIELD NOT IN (...)`, `FIELD=value`,
`FIELD<>value`), the .lyrx definition query is split and parsed the same way and
each config clause is compared with the .lyrx clause on the same field. Reported:

  * operator mismatch (IN vs NOT IN)
  * values only in the config filter
  * values only in the .lyrx definition query
  * config clause with no counterpart in the .lyrx query
  * extra clauses in the .lyrx query not present in the config filter
    (e.g. `AND (PRINTED = 1 OR PRINTED IS NULL)`), as informational notes

Blocks without a config filter, with a config filter using OR or other
non-simple clauses, or whose .lyrx
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


console = Console()

# One simple clause: FIELD (=|==|<>|!=|IN|NOT IN) value(s)
CLAUSE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(NOT\s+IN|IN|==|=|<>|!=)\s*"
    r"\(?\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\)?\s*$",
    re.IGNORECASE,
)


def split_and(expr: str) -> list:
    """Split on top-level AND (not inside parentheses)."""
    parts, depth, start = [], 0, 0
    for m in re.finditer(r"[()]|\bAND\b", expr, re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            parts.append(expr[start:m.start()])
            start = m.end()
    parts.append(expr[start:])
    return [p.strip() for p in parts if p.strip()]


def parse_clause(clause: str):
    """(FIELD, 'IN'|'NOT IN', {ints}) for a simple clause, else None."""
    m = CLAUSE_RE.match(clause)
    if not m:
        return None
    field, op, raw = m.groups()
    op = re.sub(r"\s+", " ", op.upper())
    op = "NOT IN" if op in ("NOT IN", "<>", "!=") else "IN"
    return field.upper(), op, {int(v) for v in raw.split(",")}


def parse_and_filter(expr: str):
    """Split an AND-joined filter into (simple clauses, other raw clauses)."""
    clauses, other = [], []
    for part in split_and(expr):
        parsed = parse_clause(part)
        if parsed:
            clauses.append(parsed)
        else:
            other.append(part)
    return clauses, other


def fmt_clause(c) -> str:
    return f"{c[0]} {c[1]} {', '.join(map(str, sorted(c[2])))}"


def compare_filters(cfg_clauses, lyrx_clauses, lyrx_other):
    """Return ([(kind, detail)] differences, [str] informational notes)."""
    diffs, notes = [], []
    lyrx_by_field = {c[0]: c for c in lyrx_clauses}
    for field, op, vals in cfg_clauses:
        match = lyrx_by_field.pop(field, None)
        if match is None:
            diffs.append(("clause only in config", fmt_clause((field, op, vals))))
            continue
        _, l_op, l_vals = match
        if op != l_op:
            diffs.append(("operator", f"{field}: config {op} vs lyrx {l_op}"))
            continue
        for label, extra in (("only in config", vals - l_vals), ("only in lyrx", l_vals - vals)):
            if extra:
                diffs.append((label, f"{field}: {', '.join(map(str, sorted(extra)))}"))
    notes.extend(fmt_clause(c) for c in lyrx_by_field.values())
    notes.extend(lyrx_other)
    return diffs, notes


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

            cfg_clauses, cfg_other = parse_and_filter(str(filter_expr))
            if cfg_other or not cfg_clauses:
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

            lyrx_clauses, lyrx_other = parse_and_filter(expr)
            if not lyrx_clauses:
                skipped.append((where, f"unparseable .lyrx query: {expr[:60]}"))
                continue

            checked += 1
            block_diffs, block_notes = compare_filters(cfg_clauses, lyrx_clauses, lyrx_other)
            diffs.extend((where, kind, detail) for kind, detail in block_diffs)
            if block_notes:
                notes.append((where, " AND ".join(block_notes)))

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
