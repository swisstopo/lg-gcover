#!/usr/bin/env python3
"""
Compare two versions of ESRI .lyrx style files and report what changed.

Reuses ESRIClassificationExtractor (extract_lyrx_complete) to parse each
side, then diffs classes per layer using the same identifier logic as the
real pipeline (label-slug by default, matching esri_classification_applicator
and generator).

Change kinds detected, per layer:
  - new_class / deleted_class : a class identifier only exists on one side
  - renamed                   : same class conditions, different label
                                 (paired from unmatched added/removed classes)
  - condition_changed         : same identifier, different field_values
  - symbology_changed         : same identifier, different symbol appearance
  - visibility_changed        : same identifier, visible flag flipped

Usage:
    # Two single files
    python scripts/diff_lyrx.py old/Bedrock.lyrx new/Bedrock.lyrx

    # Two directories (matched by filename)
    python scripts/diff_lyrx.py old_styles/ new_styles/

    # Only show symbology and condition changes, export full diff to JSON
    python scripts/diff_lyrx.py old/ new/ --only symbology_changed --only condition_changed \\
        --export json -o lyrx_diff.json
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcover.publish.esri_classification_extractor import (
    ClassificationClass,
    extract_lyrx_complete,
)

console = Console()


# =============================================================================
# COMPARABLE SIGNATURES
# =============================================================================


def identifier_key(class_obj: ClassificationClass) -> str:
    """Same identifier value the pipeline computes (label-slug by default)."""
    if class_obj.identifier:
        try:
            return str(class_obj.identifier.to_key().split("::")[-1])
        except Exception:
            pass
    return class_obj.label


def field_values_key(class_obj: ClassificationClass) -> frozenset:
    """Order-independent snapshot of a class's matching conditions."""
    return frozenset(tuple(fv) for fv in class_obj.field_values)


def format_conditions(class_obj: ClassificationClass) -> str:
    return "; ".join(" & ".join(fv) for fv in class_obj.field_values) or "(default)"


def _symbol_field(symbol_info, name: str):
    if not symbol_info:
        return None
    if name == "color":
        return symbol_info.color.to_hex() if symbol_info.color else None
    if name == "alpha":
        return symbol_info.color.alpha if symbol_info.color else None
    if name == "symbol_type":
        return symbol_info.symbol_type.value if symbol_info.symbol_type else None
    if name == "dash_pattern":
        return tuple(symbol_info.dash_pattern) if symbol_info.dash_pattern else None
    return getattr(symbol_info, name, None)


SYMBOL_FIELDS = [
    "symbol_type", "color", "alpha", "size", "width", "line_style",
    "dash_pattern", "cap_style", "join_style", "fill_type",
    "font_family", "character_index",
]


def symbol_signature(class_obj: ClassificationClass) -> tuple:
    return tuple(_symbol_field(class_obj.symbol_info, f) for f in SYMBOL_FIELDS)


def describe_symbol_diff(old_c: ClassificationClass, new_c: ClassificationClass) -> str:
    parts = []
    for f in SYMBOL_FIELDS:
        ov = _symbol_field(old_c.symbol_info, f)
        nv = _symbol_field(new_c.symbol_info, f)
        if ov != nv:
            parts.append(f"{f}: {ov!r} -> {nv!r}")
    return ", ".join(parts) if parts else "symbol changed"


# =============================================================================
# DIFF DATA MODEL
# =============================================================================


@dataclass
class ClassDiff:
    layer: str
    kind: str
    label: str
    old_label: Optional[str] = None
    identifier: Optional[str] = None
    details: str = ""


@dataclass
class LyrxDiffResult:
    old_file: str
    new_file: str
    new_layers: List[str] = field(default_factory=list)
    removed_layers: List[str] = field(default_factory=list)
    changes: List[ClassDiff] = field(default_factory=list)


KIND_STYLES = {
    "new_class": ("+", "green"),
    "deleted_class": ("-", "red"),
    "renamed": ("~", "yellow"),
    "condition_changed": ("≠", "magenta"),
    "symbology_changed": ("\U0001F3A8", "cyan"),
    "visibility_changed": ("\U0001F441", "blue"),
    "fields_changed": ("!", "bold red"),
}


# =============================================================================
# CORE DIFF LOGIC
# =============================================================================


def load_classifications(lyrx_path: Path) -> Dict[str, "LayerClassification"]:
    """layer_path -> LayerClassification (label-slug identifiers, like the pipeline default)."""
    classifications = extract_lyrx_complete(lyrx_path, display=False)
    return {(c.layer_path or c.layer_name): c for c in classifications}


def diff_layer(layer_name: str, old_layer, new_layer) -> List[ClassDiff]:
    changes: List[ClassDiff] = []

    old_field_names = [f.name for f in old_layer.fields]
    new_field_names = [f.name for f in new_layer.fields]
    if old_field_names != new_field_names:
        changes.append(ClassDiff(
            layer=layer_name, kind="fields_changed", label="",
            details=f"renderer fields {old_field_names} -> {new_field_names} "
                    f"(classes are not comparable field-for-field below)",
        ))

    old_classes = {identifier_key(c): c for c in old_layer.classes}
    new_classes = {identifier_key(c): c for c in new_layer.classes}

    old_ids, new_ids = set(old_classes), set(new_classes)
    common = old_ids & new_ids
    added = new_ids - old_ids
    removed = old_ids - new_ids

    # Pair up added/removed classes that share identical conditions -> rename
    old_fv_index: Dict[frozenset, List[str]] = {}
    for cid in removed:
        old_fv_index.setdefault(field_values_key(old_classes[cid]), []).append(cid)

    paired_added: Set[str] = set()
    paired_removed: Set[str] = set()

    for cid in sorted(added):
        candidates = old_fv_index.get(field_values_key(new_classes[cid]))
        if candidates:
            old_cid = candidates.pop(0)
            paired_added.add(cid)
            paired_removed.add(old_cid)

            old_c, new_c = old_classes[old_cid], new_classes[cid]
            changes.append(ClassDiff(
                layer=layer_name, kind="renamed",
                label=new_c.label, old_label=old_c.label, identifier=cid,
                details=f"identifier {old_cid!r} -> {cid!r}",
            ))
            if symbol_signature(old_c) != symbol_signature(new_c):
                changes.append(ClassDiff(
                    layer=layer_name, kind="symbology_changed",
                    label=new_c.label, identifier=cid,
                    details=describe_symbol_diff(old_c, new_c),
                ))

    for cid in sorted(added - paired_added):
        c = new_classes[cid]
        changes.append(ClassDiff(
            layer=layer_name, kind="new_class", label=c.label, identifier=cid,
            details=f"conditions: {format_conditions(c)}",
        ))

    for cid in sorted(removed - paired_removed):
        c = old_classes[cid]
        changes.append(ClassDiff(
            layer=layer_name, kind="deleted_class", label=c.label, identifier=cid,
            details=f"conditions: {format_conditions(c)}",
        ))

    for cid in sorted(common):
        old_c, new_c = old_classes[cid], new_classes[cid]

        if field_values_key(old_c) != field_values_key(new_c):
            changes.append(ClassDiff(
                layer=layer_name, kind="condition_changed",
                label=new_c.label, identifier=cid,
                details=f"{format_conditions(old_c)} -> {format_conditions(new_c)}",
            ))

        if symbol_signature(old_c) != symbol_signature(new_c):
            changes.append(ClassDiff(
                layer=layer_name, kind="symbology_changed",
                label=new_c.label, identifier=cid,
                details=describe_symbol_diff(old_c, new_c),
            ))

        if old_c.visible != new_c.visible:
            changes.append(ClassDiff(
                layer=layer_name, kind="visibility_changed",
                label=new_c.label, identifier=cid,
                details=f"{old_c.visible} -> {new_c.visible}",
            ))

    return changes


def diff_lyrx_files(old_path: Path, new_path: Path) -> LyrxDiffResult:
    old_layers = load_classifications(old_path)
    new_layers = load_classifications(new_path)

    result = LyrxDiffResult(old_file=str(old_path), new_file=str(new_path))
    result.new_layers = sorted(set(new_layers) - set(old_layers))
    result.removed_layers = sorted(set(old_layers) - set(new_layers))

    for layer_name in sorted(set(old_layers) & set(new_layers)):
        result.changes.extend(
            diff_layer(layer_name, old_layers[layer_name], new_layers[layer_name])
        )

    return result


# =============================================================================
# DISPLAY
# =============================================================================


def display_result(result: LyrxDiffResult) -> None:
    console.print(Panel.fit(
        f"[bold]{Path(result.old_file).name}[/bold] -> [bold]{Path(result.new_file).name}[/bold]"
    ))

    if result.new_layers:
        console.print(f"[green]+ New layers:[/green] {', '.join(result.new_layers)}")
    if result.removed_layers:
        console.print(f"[red]- Removed layers:[/red] {', '.join(result.removed_layers)}")

    if not result.changes and not result.new_layers and not result.removed_layers:
        console.print("[dim]No differences detected[/dim]\n")
        return

    if result.changes:
        table = Table(show_header=True, header_style="bold")
        table.add_column("", width=2)
        table.add_column("Layer", style="cyan", overflow="fold")
        table.add_column("Type", style="white")
        table.add_column("Label", style="white", max_width=40, overflow="fold")
        table.add_column("Identifier", style="yellow", overflow="fold")
        table.add_column("Details", style="dim", max_width=60, overflow="fold")

        for c in result.changes:
            symbol, style = KIND_STYLES.get(c.kind, ("?", "white"))
            table.add_row(
                f"[{style}]{symbol}[/{style}]", c.layer, c.kind, c.label,
                c.identifier or "", c.details,
            )

        console.print(table)

        counts: Dict[str, int] = {}
        for c in result.changes:
            counts[c.kind] = counts.get(c.kind, 0) + 1
        summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        console.print(f"[bold]Summary:[/bold] {summary}")

    console.print()


# =============================================================================
# CLI
# =============================================================================


def find_lyrx_pairs(old_dir: Path, new_dir: Path, pattern: str):
    old_files = {p.name: p for p in old_dir.glob(pattern)}
    new_files = {p.name: p for p in new_dir.glob(pattern)}

    common = sorted(set(old_files) & set(new_files))
    only_old = sorted(set(old_files) - set(new_files))
    only_new = sorted(set(new_files) - set(old_files))

    return common, old_files, new_files, only_old, only_new


@click.command()
@click.argument("old", type=click.Path(exists=True, path_type=Path))
@click.argument("new", type=click.Path(exists=True, path_type=Path))
@click.option("--pattern", default="*.lyrx", show_default=True,
              help="Glob pattern used when comparing directories")
@click.option("--only", multiple=True, type=click.Choice(sorted(KIND_STYLES)),
              help="Restrict output to these change kinds (repeatable)")
@click.option("--export", type=click.Choice(["json"]), help="Export the diff to a file")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("lyrx_diff.json"),
              show_default=True, help="Export output path")
@click.option("--quiet", is_flag=True, help="Suppress rich tables, print summary counts only")
def main(old: Path, new: Path, pattern: str, only, export, output: Path, quiet: bool):
    """
    Compare .lyrx style files between OLD and NEW.

    OLD and NEW are either two single .lyrx files, or two directories of
    .lyrx files matched by filename (use --pattern to filter).
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    only_set = set(only) if only else None

    if old.is_dir() and new.is_dir():
        common, old_files, new_files, only_old, only_new = find_lyrx_pairs(old, new, pattern)

        if only_old:
            console.print(f"[red]Only in {old}:[/red] {', '.join(only_old)}")
        if only_new:
            console.print(f"[green]Only in {new}:[/green] {', '.join(only_new)}")
        if not common:
            console.print("[yellow]No matching .lyrx filenames found in both directories[/yellow]")

        results = []
        for name in common:
            try:
                results.append(diff_lyrx_files(old_files[name], new_files[name]))
            except Exception as exc:
                console.print(f"[red]ERROR comparing {name}: {exc}[/red]")

    elif old.is_file() and new.is_file():
        results = [diff_lyrx_files(old, new)]
    else:
        raise click.UsageError("OLD and NEW must both be files or both be directories")

    if only_set:
        for r in results:
            r.changes = [c for c in r.changes if c.kind in only_set]

    if not quiet:
        for r in results:
            display_result(r)
    else:
        for r in results:
            counts: Dict[str, int] = {}
            for c in r.changes:
                counts[c.kind] = counts.get(c.kind, 0) + 1
            if counts or r.new_layers or r.removed_layers:
                console.print(
                    f"{Path(r.new_file).name}: "
                    f"+layers={len(r.new_layers)} -layers={len(r.removed_layers)} "
                    + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                )

    if export == "json":
        output.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        console.print(f"[green]Exported diff to {output}[/green]")


if __name__ == "__main__":
    main()
