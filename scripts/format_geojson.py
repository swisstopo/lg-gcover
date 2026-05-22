#!/usr/bin/env python3
"""Format a GeoJSON file for clean git diffs.

Properties are pretty-printed (one attribute per line); geometry is kept on a
single compact line so coordinate noise does not pollute diffs.
"""

import hashlib
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

console = Console()

_GEOM_COORD_PRECISION = 2  # metres → 1 cm, enough to absorb float noise


def _round_coords(obj):
    if isinstance(obj, float):
        return round(obj, _GEOM_COORD_PRECISION)
    if isinstance(obj, list):
        return [_round_coords(v) for v in obj]
    return obj


def geom_hash(geom: dict) -> str:
    normalized = json.dumps(
        {"type": geom["type"], "coordinates": _round_coords(geom["coordinates"])},
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode()).hexdigest()[:10]


def fmt_props(props: dict, geom: dict) -> str:
    enriched = {"_geom_hash": geom_hash(geom), **props}
    s = json.dumps(enriched, indent=2, ensure_ascii=False)
    lines = s.splitlines()
    return lines[0] + "\n" + "\n".join("      " + l for l in lines[1:])


def fmt_geom(geom) -> str:
    return json.dumps(geom, separators=(",", ":"), ensure_ascii=False)


def format_geojson(data: dict) -> str:
    features = []
    for feat in data["features"]:
        features.append(
            "    {\n"
            '      "type": "Feature",\n'
            f'      "properties": {fmt_props(feat["properties"], feat["geometry"])},\n'
            f'      "geometry": {fmt_geom(feat["geometry"])}\n'
            "    }"
        )

    return (
        "{\n"
        f'  "type": {json.dumps(data["type"])},\n'
        f'  "name": {json.dumps(data["name"])},\n'
        '  "features": [\n'
        + ",\n".join(features)
        + "\n  ]\n}\n"
    )


@click.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--check", is_flag=True, help="Check only — exit 1 if file would change.")
def main(file: Path, check: bool):
    """Format FILE for clean git diffs: indented properties, compact geometry."""
    with open(file) as f:
        data = json.load(f)

    n = len(data["features"])
    formatted = format_geojson(data)

    if check:
        if file.read_text() != formatted:
            console.print(Text(f"✗ {file} would be reformatted", style="bold red"))
            sys.exit(1)
        console.print(Text(f"✓ {file} is already formatted", style="green"))
        return

    file.write_text(formatted)
    console.print(f"[green]✓[/green] Formatted [bold]{file}[/bold] ([cyan]{n}[/cyan] features)")


if __name__ == "__main__":
    main()
