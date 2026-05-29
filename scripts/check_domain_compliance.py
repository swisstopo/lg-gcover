#!/usr/bin/env python3
"""Check a FileGDB's field values against coded value domains.

By default the GDB is checked against its own domains.
Pass --reference to use another GDB's domains instead — useful for
detecting drift in custom sources (e.g. Saas.gdb) against the
authoritative model (RC2.gdb).

Examples
--------
  # Self-check
  python scripts/check_domain_compliance.py ~/DATA/.../RC2.gdb

  # Drift check: Saas.gdb values vs. RC2.gdb domains
  python scripts/check_domain_compliance.py ~/DATA/.../Saas.gdb \\
      --reference ~/DATA/.../RC2.gdb

  # Only certain layers, write a report
  python scripts/check_domain_compliance.py ~/DATA/.../Saas.gdb \\
      --reference ~/DATA/.../RC2.gdb \\
      --layers GC_BEDROCK GC_UNCO_DESPOSIT \\
      --report saas_drift.txt
"""

import sys
from pathlib import Path

import click
import fiona
import geopandas as gpd
import pandas as pd
from osgeo import ogr
from rich.console import Console
from rich.table import Table

console = Console()
report_console = Console(record=True)

# Layers that carry no coded-domain fields — skip silently
_SKIP_LAYERS = {"aux_points_aspect", "GC_BEDROCK_D", "GC_BEDROCK_A"}

# How many bad example values to show per field
_N_EXAMPLES = 5


# ── Domain helpers ────────────────────────────────────────────────────────────

def _load_domains(gdb_path: Path) -> dict[str, frozenset]:
    """Return {domain_name: frozenset(str_codes)} for every coded-value domain."""
    ds = ogr.Open(str(gdb_path), 0)
    if ds is None:
        raise click.ClickException(f"Cannot open GDB: {gdb_path}")
    result: dict[str, frozenset] = {}
    for dname in (ds.GetFieldDomainNames() or []):
        dom = ds.GetFieldDomain(dname)
        if dom is None or dom.GetDomainType() != ogr.OFDT_CODED:
            continue
        result[dname] = frozenset(str(k) for k in dom.GetEnumeration())
    ds = None
    return result


def _load_field_domain_map(gdb_path: Path) -> dict[str, dict[str, tuple[str, int]]]:
    """Return {layer_name: {field_name: (domain_name, ogr_type)}}."""
    ds = ogr.Open(str(gdb_path), 0)
    if ds is None:
        raise click.ClickException(f"Cannot open GDB: {gdb_path}")
    result: dict[str, dict[str, tuple[str, int]]] = {}
    for i in range(ds.GetLayerCount()):
        lyr = ds.GetLayerByIndex(i)
        name = lyr.GetName()
        defn = lyr.GetLayerDefn()
        fields = {}
        for j in range(defn.GetFieldCount()):
            fld = defn.GetFieldDefn(j)
            dname = fld.GetDomainName()
            if dname:
                fields[fld.GetNameRef()] = (dname, fld.GetType())
        if fields:
            result[name] = fields
    ds = None
    return result


def _valid_set(str_codes: frozenset, ogr_type: int) -> frozenset:
    """Cast domain codes to the right Python type for comparison."""
    if ogr_type in (ogr.OFTInteger, ogr.OFTInteger64):
        try:
            return frozenset(int(c) for c in str_codes)
        except ValueError:
            pass
    elif ogr_type == ogr.OFTReal:
        try:
            return frozenset(float(c) for c in str_codes)
        except ValueError:
            pass
    return str_codes


# ── Per-layer check ───────────────────────────────────────────────────────────

def _check_layer(
    gdb_path: Path,
    layer_name: str,
    field_map: dict[str, tuple[str, int]],
    domains: dict[str, frozenset],
) -> list[dict]:
    """
    Returns a list of violation records:
      {field, domain, n_total, n_bogus, pct, examples}
    Empty list = clean.
    """
    try:
        gdf = gpd.read_file(str(gdb_path), layer=layer_name)
    except Exception as exc:
        console.print(f"  [yellow]⚠[/] Could not read {layer_name}: {exc}")
        return []

    records = []
    for field, (dname, ogr_type) in sorted(field_map.items()):
        if field not in gdf.columns:
            continue
        if dname not in domains:
            continue

        valid = _valid_set(domains[dname], ogr_type)
        col = gdf[field]
        non_null = col.notna()
        n_total = int(non_null.sum())
        if n_total == 0:
            continue

        if ogr_type in (ogr.OFTInteger, ogr.OFTInteger64, ogr.OFTReal):
            bogus_mask = non_null & ~col.isin(valid)
        else:
            bogus_mask = non_null & ~col.astype(str).isin(valid)

        n_bogus = int(bogus_mask.sum())
        if n_bogus == 0:
            continue

        examples = (
            col[bogus_mask]
            .value_counts()
            .head(_N_EXAMPLES)
            .index.tolist()
        )
        records.append({
            "field":    field,
            "domain":   dname,
            "n_total":  n_total,
            "n_bogus":  n_bogus,
            "pct":      n_bogus / n_total * 100,
            "examples": examples,
        })

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("target_gdb", type=click.Path(exists=True))
@click.option("--reference", "reference_gdb", default=None, type=click.Path(exists=True),
              help="GDB whose coded domains are used as the authority (default: target itself).")
@click.option("--layers", "-l", multiple=True, metavar="LAYER",
              help="Restrict check to these layer names (repeatable). Default: all layers.")
@click.option("--report", type=click.Path(), default=None,
              help="Write plain-text summary to this file.")
@click.option("--show-clean", is_flag=True, default=False,
              help="Also list layers/fields that passed (no violations).")
def main(target_gdb, reference_gdb, layers, report, show_clean):
    """Check a FileGDB's field values against coded value domains.

    TARGET_GDB is the GDB to validate.
    Use --reference to supply a different authoritative domain source.
    """
    target_path    = Path(target_gdb)
    reference_path = Path(reference_gdb) if reference_gdb else target_path
    same_source    = reference_path == target_path

    console.print(f"\n[bold]Target  :[/] {target_path}")
    if not same_source:
        console.print(f"[bold]Reference:[/] {reference_path}")
    else:
        console.print("[bold]Reference:[/] [dim](self — checking against own domains)[/]")

    # ── Load domain definitions from reference ────────────────────────────────
    try:
        domains = _load_domains(reference_path)
    except click.ClickException as e:
        console.print(f"[red]✗ {e.format_message()}[/]")
        sys.exit(1)

    console.print(f"  {len(domains)} coded domain(s) loaded from reference\n")

    # ── Load field→domain map from target ─────────────────────────────────────
    try:
        field_domain_map = _load_field_domain_map(target_path)
    except click.ClickException as e:
        console.print(f"[red]✗ {e.format_message()}[/]")
        sys.exit(1)

    # ── Layer list ────────────────────────────────────────────────────────────
    try:
        all_layers = fiona.listlayers(str(target_path))
    except Exception as exc:
        console.print(f"[red]✗ Cannot list layers in {target_path}: {exc}[/]")
        sys.exit(1)

    target_layers = [l for l in all_layers if l not in _SKIP_LAYERS]
    if layers:
        target_layers = [l for l in target_layers if l in layers]
        missing = set(layers) - set(target_layers)
        if missing:
            console.print(f"[yellow]⚠[/] Requested layers not found: {', '.join(sorted(missing))}")

    # ── Check each layer ──────────────────────────────────────────────────────
    summary_rows: list[dict] = []
    total_bogus  = 0

    for layer_name in target_layers:
        field_map = field_domain_map.get(layer_name, {})
        if not field_map:
            if show_clean:
                console.print(f"[dim]{layer_name}: no domain-bound fields[/]")
            continue

        violations = _check_layer(target_path, layer_name, field_map, domains)

        if not violations:
            if show_clean:
                console.print(f"[green]✓[/] {layer_name}: all domain-bound fields clean")
            continue

        # Print per-layer violation table
        tbl = Table(
            title=f"[bold red]✗[/] {layer_name}",
            show_header=True,
            header_style="bold",
        )
        tbl.add_column("Field",       style="cyan",   no_wrap=True)
        tbl.add_column("Domain",      style="dim",    no_wrap=True)
        tbl.add_column("Non-null",    justify="right")
        tbl.add_column("Bogus",       justify="right", style="red")
        tbl.add_column("%",           justify="right", style="red")
        tbl.add_column(f"Top-{_N_EXAMPLES} bogus values", style="yellow")

        for v in violations:
            ex = ", ".join(str(e) for e in v["examples"])
            tbl.add_row(
                v["field"],
                v["domain"],
                f"{v['n_total']:,}",
                f"{v['n_bogus']:,}",
                f"{v['pct']:.1f}",
                ex,
            )
            summary_rows.append({"layer": layer_name, **v})
            total_bogus += v["n_bogus"]

        console.print(tbl)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    if not summary_rows:
        console.print("[bold green]✓ All domain-bound fields are clean — no violations found.[/]")
    else:
        n_layers = len({r["layer"] for r in summary_rows})
        n_fields = len(summary_rows)
        console.print(
            f"[bold red]✗ {total_bogus:,} bogus value(s) in "
            f"{n_fields} field(s) across {n_layers} layer(s)[/]"
        )

        summary_tbl = Table(title="Summary", show_header=True, header_style="bold")
        summary_tbl.add_column("Layer",  style="cyan",   min_width=24)
        summary_tbl.add_column("Field",  style="yellow", min_width=22)
        summary_tbl.add_column("Domain", style="dim",    min_width=32)
        summary_tbl.add_column("Bogus",  justify="right", style="red")
        summary_tbl.add_column("%",      justify="right", style="red")
        for r in sorted(summary_rows, key=lambda x: -x["n_bogus"]):
            summary_tbl.add_row(
                r["layer"], r["field"], r["domain"],
                f"{r['n_bogus']:,}", f"{r['pct']:.1f}",
            )
        console.print(summary_tbl)

    # ── Optional report file ──────────────────────────────────────────────────
    if report:
        report_path = Path(report)
        lines = [
            f"Domain compliance report",
            f"  Target   : {target_path}",
            f"  Reference: {reference_path}",
            f"",
        ]
        if not summary_rows:
            lines.append("RESULT: CLEAN — no violations found.")
        else:
            lines.append(f"RESULT: {total_bogus:,} bogus value(s) found\n")
            lines.append(f"{'Layer':<30} {'Field':<25} {'Domain':<35} {'Bogus':>8} {'%':>6}  Examples")
            lines.append("-" * 120)
            for r in sorted(summary_rows, key=lambda x: -x["n_bogus"]):
                ex = ", ".join(str(e) for e in r["examples"])
                lines.append(
                    f"{r['layer']:<30} {r['field']:<25} {r['domain']:<35} "
                    f"{r['n_bogus']:>8,} {r['pct']:>5.1f}%  {ex}"
                )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"\n[dim]Report written → {report_path}[/]")

    sys.exit(1 if summary_rows else 0)


if __name__ == "__main__":
    main()
