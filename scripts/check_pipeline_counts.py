#!/usr/bin/env python3
"""Check feature counts are consistent across the merge→denormalize→classify→translate pipeline."""

import sys
from pathlib import Path

import click
import fiona
import yaml
from rich.console import Console
from rich.table import Table

SKIP_LAYERS = {"aux_points_aspect"}

console = Console()


def count_layers(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {
        layer: len(fiona.open(str(path), layer=layer))
        for layer in fiona.listlayers(str(path))
        if layer not in SKIP_LAYERS
    }


def gdb_layer_map(config_path: Path) -> dict[str, str]:
    """Return {gpkg_layer: gdb_layer_name} from the config YAML, stripping the group prefix."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    mapping = {}
    for layer in cfg.get("layers", []):
        gpkg = layer.get("gpkg_layer")
        gcover = layer.get("gcover_layer", "")
        # Strip group prefix: "GC_ROCK_BODIES/GC_BEDROCK" → "GC_BEDROCK"
        gdb_name = gcover.split("/")[-1] if "/" in gcover else gcover
        if gpkg and gdb_name:
            mapping[gpkg] = gdb_name
    return mapping


def count_gdb_layers(gdb_path: Path, layer_map: dict[str, str]) -> dict[str, int]:
    """Count features in the GDB, keyed by gpkg_layer name."""
    if not gdb_path.exists():
        return {}
    available = set(fiona.listlayers(str(gdb_path)))
    counts = {}
    for gpkg_layer, gdb_layer in layer_map.items():
        if gdb_layer in available:
            counts[gpkg_layer] = len(fiona.open(str(gdb_path), layer=gdb_layer))
        else:
            console.print(f"  [yellow]⚠[/]  GDB layer [bold]{gdb_layer}[/] not found (for {gpkg_layer})")
    return counts


@click.command()
@click.argument("denormalized", type=click.Path(exists=True))
@click.argument("classified", type=click.Path(exists=True))
@click.argument("translated", type=click.Path(exists=True))
@click.option("--gdb", type=click.Path(), default=None, help="Merged FileGDB path")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Config YAML with gpkg_layer/gcover_layer mapping (required with --gdb)")
def main(denormalized, classified, translated, gdb, config_path):
    """Check feature counts across the pipeline GPKGs for regressions."""
    stages = [
        ("denormalized", Path(denormalized)),
        ("classified", Path(classified)),
        ("translated", Path(translated)),
    ]

    counts = {label: count_layers(path) for label, path in stages}

    # Prepend merged GDB column if provided
    if gdb:
        if not config_path:
            console.print("[red]--config is required when --gdb is specified[/]")
            sys.exit(1)
        layer_map = gdb_layer_map(Path(config_path))
        gdb_counts = count_gdb_layers(Path(gdb), layer_map)
        counts = {"merged (GDB)": gdb_counts, **counts}
        stages = [("merged (GDB)", Path(gdb))] + stages

    all_layers = sorted({layer for c in counts.values() for layer in c})

    table = Table(title="Pipeline feature counts", show_lines=False, header_style="bold cyan")
    table.add_column("Layer", style="dim", width=28)
    for label, _ in stages:
        table.add_column(label, justify="right", width=14)
    table.add_column("OK?", width=5)

    ok = True
    for layer in all_layers:
        row_counts = [counts[label].get(layer, None) for label, _ in stages]
        present = [c for c in row_counts if c is not None]
        consistent = len(set(present)) == 1

        if not consistent:
            ok = False
            status = "[bold red]✗[/]"
        else:
            status = "[green]✓[/]"

        table.add_row(
            layer,
            *[
                (f"[red]{c:,}[/]" if not consistent and c != max(present) else f"{c:,}")
                if c is not None
                else "[dim]—[/]"
                for c in row_counts
            ],
            status,
        )

    totals = [sum(counts[label].get(l, 0) for l in all_layers) for label, _ in stages]
    total_consistent = len(set(totals)) == 1
    table.add_row(
        "[bold]TOTAL[/]",
        *[
            (f"[red]{t:,}[/]" if not total_consistent and t != max(totals) else f"[bold]{t:,}[/]")
            for t in totals
        ],
        "[green]✓[/]" if total_consistent else "[bold red]✗[/]",
    )

    console.print(table)

    if ok:
        console.print("[green]✓ All feature counts are consistent across the pipeline.[/green]")
    else:
        console.print("[bold red]✗ Feature count mismatches detected![/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
