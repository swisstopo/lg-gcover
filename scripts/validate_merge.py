#!/usr/bin/env python3
"""
gcover publish validate-merge
──────────────────────────────
Post-merge validation: compares feature counts and geometry health between
source GDBs and the merged output, per layer.

Metrics reported per layer:
  • n_source   — features in each source GDB, filtered to its exclusive mask
  • n_merged   — features in the merged output GDB
  • delta      — n_merged − n_source_total  (negative = loss, positive = gain)
  • boundary_candidates — source features that cross the mask edge (clip risk)
  • boundary_splits     — degenerate geometries produced by clipping
                          (Lines / Points dropped by normalize_gdf)
  • n_invalid  — invalid geometries in merged output

Exit codes (useful for CI):
  0 — all checks passed (warnings may be present)
  1 — at least one ERROR threshold breached
  2 — configuration / input error

Usage:
    python validate_merge.py \\
        --rc1  /path/to/RC1.gdb \\
        --rc2  /path/to/RC2.gdb \\
        --merged /path/to/merged.gdb \\
        --admin data/administrative_zones.gpkg \\
        [--warn-loss 1.0] \\
        [--error-loss 5.0] \\
        [--allow-invalid] \\
        [--json report.json]

Integration with merge_sources.py:
    Add a call to MergeValidator.run() at the end of GDBMerger.merge().
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import warnings

import click
import geopandas as gpd
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box
from shapely import intersects, is_valid, is_empty, within
from shapely.ops import unary_union
from shapely import difference

from gcover.publish.merge_sources import _discover_filegdbs

# Suppress OGR geometry warnings (unclosed rings, 100-part polygons, etc.)
# Same approach as merge_sources.py
os.environ["OGR_GEOMETRY_ACCEPT_UNCLOSED_RING"] = "NO"
os.environ["METHOD"] = "ONLY_CCW"
warnings.filterwarnings("ignore", message=".*organizePolygons.*",  category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Non closed ring.*",   category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*closed ring.*",       category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPATIAL_LAYERS = [
    "GC_BEDROCK",
    "GC_UNCO_DESPOSIT",
    "GC_SURFACES",
    "GC_LINEAR_OBJECTS",
    "GC_POINT_OBJECTS",
    "GC_FOSSILS",
    "GC_EXPLOIT_GEOMAT_PLG",
    "GC_EXPLOIT_GEOMAT_PT",
    # "GC_MAPSHEET",
]

# Geometry types that normalize_gdf silently drops (boundary artefacts)
DEGENERATE_TYPES = {"Point", "LineString", "MultiLineString", "MultiPoint"}

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    OK      = "OK"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


@dataclass
class LayerReport:
    layer: str
    source_counts: Dict[str, int]       # {"RC1": 1200, "RC2": 800}
    n_source_total: int                 # sum of source counts
    n_merged: int                       # features in merged output
    delta: int                          # n_merged − n_source_total
    delta_pct: float                    # delta / n_source_total * 100
    boundary_candidates: int            # features crossing mask edge (per source, summed)
    boundary_splits: int                # degenerate geoms dropped by normalization
    n_invalid_merged: int               # invalid geoms in merged output
    geometry_types_merged: Dict[str, int]
    severity: Severity = Severity.OK
    messages: List[str] = field(default_factory=list)


@dataclass
class ValidationThresholds:
    """Configurable thresholds that control WARNING vs ERROR classification."""
    # Feature loss
    warn_loss_pct: float  = 1.0    # delta < -1%  → WARNING
    error_loss_pct: float = 5.0    # delta < -5%  → ERROR
    # Feature gain (unexpected duplicates)
    warn_gain_pct: float  = 1.0    # delta > +1%  → WARNING
    error_gain_pct: float = 5.0    # delta > +5%  → ERROR
    # Invalid geometries in merged output
    allow_invalid: bool   = False  # any invalid  → ERROR (unless True)
    # Missing layers in merged output
    allow_missing: bool   = False  # missing layer → ERROR (unless True)


@dataclass
class ValidationReport:
    layers: List[LayerReport] = field(default_factory=list)
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    elapsed_sec: float = 0.0
    has_errors: bool   = False
    has_warnings: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert Severity enums to strings for JSON serialisation
        for lr in d["layers"]:
            lr["severity"] = lr["severity"].value if hasattr(lr["severity"], "value") else lr["severity"]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers  (mirrors merge_sources._create_source_masks)
# ─────────────────────────────────────────────────────────────────────────────

def build_exclusive_masks(
    admin_path: Path,
    layer: str = "mapsheets_sources_only",
    source_col: str = "SOURCE_RC",
    mapsheet_numbers: Optional[List[int]] = None,
) -> Tuple[Dict[str, object], gpd.GeoDataFrame]:
    """
    Returns (exclusive_masks, mapsheets_gdf) mirroring GDBMerger._create_source_masks.
    RC2 > RC1 priority.
    """
    gdf = gpd.read_file(admin_path, layer=layer)
    if mapsheet_numbers:
        gdf = gdf[gdf["MSH_MAP_NBR"].isin(mapsheet_numbers)]

    all_sources = gdf[source_col].unique().tolist()

    priority = [s for s in ("RC2", "RC1") if s in all_sources]
    priority += sorted(s for s in all_sources if s not in priority)

    masks: Dict[str, object] = {}
    claimed = None

    for src in priority:
        sheets = gdf[gdf[source_col] == src]
        if sheets.empty:
            continue
        area = unary_union(sheets.geometry.values)
        if not area.is_valid:
            area = area.buffer(0)

        exclusive = difference(area, claimed) if claimed is not None else area
        if exclusive is None or exclusive.is_empty:
            continue
        if not exclusive.is_valid:
            exclusive = exclusive.buffer(0)

        masks[src] = exclusive
        claimed = unary_union([claimed, area]) if claimed is not None else area

    return masks, gdf


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_source_features(
    gdb_path: Path,
    layer_name: str,
    mask,
) -> Tuple[int, int]:
    """
    Returns (n_filtered, n_boundary_candidates) for a source GDB / layer / mask.

    n_filtered            — features that intersect the exclusive mask
    n_boundary_candidates — subset that cross the mask edge (clip risk)
    """
    try:
        gdf = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio", mask=mask)
    except Exception as e:
        logger.warning(f"  Could not read {layer_name} from {gdb_path}: {e}")
        return 0, 0

    if gdf.empty:
        return 0, 0

    geoms = gdf.geometry.values
    does_inter = intersects(geoms, mask)
    gdf = gdf[does_inter]
    if gdf.empty:
        return 0, 0

    is_win = within(gdf.geometry.values, mask)
    n_candidates = int((~is_win).sum())
    return len(gdf), n_candidates


def count_merged_features(
    merged_path: Path,
    layer_name: str,
) -> Tuple[int, int, Dict[str, int]]:
    """
    Returns (n_total, n_invalid, geometry_types_dict) for a merged layer.
    """
    try:
        gdf = gpd.read_file(merged_path, layer=layer_name, engine="pyogrio")
    except Exception as e:
        logger.debug(f"  Layer {layer_name} not found in merged output: {e}")
        return -1, 0, {}   # -1 signals layer missing

    if gdf.empty:
        return 0, 0, {}

    type_counts: Dict[str, int] = {}
    n_invalid = 0
    for geom in gdf.geometry:
        if geom is None or is_empty(geom):
            t = "(empty)"
        else:
            t = geom.geom_type
            if not is_valid(geom):
                n_invalid += 1
        type_counts[t] = type_counts.get(t, 0) + 1

    return len(gdf), n_invalid, type_counts


def estimate_boundary_splits(
    gdb_path: Path,
    layer_name: str,
    mask,
) -> int:
    """
    Simulates fast_clip on boundary candidates and counts how many produce
    degenerate (non-polygon) geometries — these are what normalize_gdf drops.
    Returns the count of such split artefacts.
    """
    from shapely import intersection as shapely_intersection

    try:
        gdf = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio", mask=mask)
    except Exception:
        return 0

    if gdf.empty:
        return 0

    geoms = gdf.geometry.values
    does_inter = intersects(geoms, mask)
    is_win     = within(geoms, mask)
    needs_clip = does_inter & ~is_win
    candidates = gdf[needs_clip]

    if candidates.empty:
        return 0

    clipped_geoms = shapely_intersection(candidates.geometry.values, mask)
    n_splits = sum(
        1 for g in clipped_geoms
        if g is not None and not is_empty(g) and g.geom_type in DEGENERATE_TYPES
    )
    return n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Core validator
# ─────────────────────────────────────────────────────────────────────────────

class MergeValidator:
    """
    Validates a merged GDB against its source GDBs.

    Can be used standalone (CLI) or embedded at the end of GDBMerger.merge().

    Example embedded use:
        validator = MergeValidator(
            sources={"RC1": config.rc1_path, "RC2": config.rc2_path},
            merged_path=config.output_path,
            admin_path=config.admin_zones_path,
            custom_sources_dir=config.custom_sources_dir,
        )
        report = validator.run()
        if report.has_errors:
            raise RuntimeError("Merge validation failed — check logs")
    """

    def __init__(
        self,
        sources: Dict[str, Path],           # {"RC1": Path(...), "RC2": Path(...)}
        merged_path: Path,
        admin_path: Path,
        custom_sources_dir: Optional[Path] = None,
        layers: Optional[List[str]] = None,
        thresholds: Optional[ValidationThresholds] = None,
        mapsheet_numbers: Optional[List[int]] = None,
        verbose: bool = False,
    ):
        self.merged_path = merged_path
        self.admin_path = admin_path
        self.layers = layers or SPATIAL_LAYERS
        self.thresholds = thresholds or ValidationThresholds()
        self.mapsheet_numbers = mapsheet_numbers
        self.verbose = verbose
        self.masks: Dict[str, object] = {}

        # Build full sources dict: standard RC1/RC2 + anything in custom_sources_dir.
        # Keys must match the SOURCE_RC values found in the mapsheets layer
        # (e.g. "20300501_Saas.gdb", "BCK_2016").
        self.sources: Dict[str, Path] = {k: v for k, v in sources.items() if v is not None}
        if custom_sources_dir and custom_sources_dir.exists():
            for gdb_path, stem, full_name in _discover_filegdbs(custom_sources_dir):
                # Register under BOTH the stem ("BCK_2016") and full name
                # ("BCK_2016.gdb") so we match whatever SOURCE_RC contains.
                self.sources.setdefault(stem,      gdb_path)
                self.sources.setdefault(full_name, gdb_path)
                console.print(f"Registered custom source: {stem!r} or {full_name!r} -> {gdb_path}")


    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> ValidationReport:
        t0 = time.time()
        report = ValidationReport(thresholds=self.thresholds)

        console.print("\n[bold blue]🔍 Post-merge validation[/bold blue]\n")

        # Build exclusive masks (same logic as the merger)
        console.print("[dim]Building exclusive source masks...[/dim]")
        self.masks, _ = build_exclusive_masks(
            self.admin_path,
            mapsheet_numbers=self.mapsheet_numbers,
        )
        console.print(f"  Sources found in mapsheets: {list(self.masks.keys())}\n")

        for layer_name in self.layers:
            lr = self._validate_layer(layer_name)
            self._classify(lr)
            report.layers.append(lr)
            if lr.severity == Severity.ERROR:
                report.has_errors = True
            elif lr.severity == Severity.WARNING:
                report.has_warnings = True

        report.elapsed_sec = time.time() - t0
        self._print_report(report)
        return report

    # ── per-layer validation ──────────────────────────────────────────────────

    def _validate_layer(self, layer_name: str) -> LayerReport:
        console.print(f"[cyan]  {layer_name}[/cyan]")

        # 1. Count features per source GDB
        source_counts: Dict[str, int] = {}
        total_candidates = 0
        total_splits = 0

        for src_name, mask in self.masks.items():
            src_path = self.sources.get(src_name)
            if src_path is None:
                console.print(f"    [yellow]⚠ No GDB configured for source {src_name}[/yellow]")
                source_counts[src_name] = 0
                continue

            n, n_cand = count_source_features(src_path, layer_name, mask)
            source_counts[src_name] = n
            total_candidates += n_cand

            if self.verbose and n_cand > 0:
                splits = estimate_boundary_splits(src_path, layer_name, mask)
                total_splits += splits

        n_source_total = sum(source_counts.values())

        # 2. Count merged output
        n_merged, n_invalid, geom_types = count_merged_features(
            self.merged_path, layer_name
        )

        # 3. Compute delta
        if n_merged == -1:
            delta = -n_source_total
            delta_pct = -100.0
        elif n_source_total == 0:
            delta = n_merged
            delta_pct = 0.0
        else:
            delta = n_merged - n_source_total
            delta_pct = delta / n_source_total * 100

        return LayerReport(
            layer=layer_name,
            source_counts=source_counts,
            n_source_total=n_source_total,
            n_merged=max(n_merged, 0),
            delta=delta,
            delta_pct=delta_pct,
            boundary_candidates=total_candidates,
            boundary_splits=total_splits,
            n_invalid_merged=n_invalid,
            geometry_types_merged=geom_types,
        )

    # ── threshold classification ──────────────────────────────────────────────

    def _classify(self, lr: LayerReport) -> None:
        t = self.thresholds

        # Missing layer
        if lr.n_merged == 0 and lr.n_source_total > 0:
            if not t.allow_missing:
                lr.severity = Severity.ERROR
                lr.messages.append(
                    f"Layer missing from merged output ({lr.n_source_total} expected)"
                )
            else:
                lr.severity = Severity.WARNING
                lr.messages.append("Layer missing from merged output (allowed)")
            return

        # Feature loss
        if lr.delta_pct < -t.error_loss_pct:
            lr.severity = Severity.ERROR
            lr.messages.append(
                f"Feature loss {lr.delta_pct:.1f}% exceeds error threshold "
                f"(-{t.error_loss_pct}%): {abs(lr.delta)} features lost"
            )
        elif lr.delta_pct < -t.warn_loss_pct:
            lr.severity = Severity.WARNING
            lr.messages.append(
                f"Feature loss {lr.delta_pct:.1f}% exceeds warning threshold "
                f"(-{t.warn_loss_pct}%): {abs(lr.delta)} features lost"
            )

        # Unexpected gain
        if lr.delta_pct > t.error_gain_pct:
            lr.severity = Severity.ERROR
            lr.messages.append(
                f"Feature gain +{lr.delta_pct:.1f}% exceeds error threshold "
                f"(+{t.error_gain_pct}%): possible duplicates ({lr.delta} extra)"
            )
        elif lr.delta_pct > t.warn_gain_pct:
            if lr.severity != Severity.ERROR:
                lr.severity = Severity.WARNING
            lr.messages.append(
                f"Feature gain +{lr.delta_pct:.1f}% exceeds warning threshold "
                f"(+{t.warn_gain_pct}%): check for duplicates"
            )

        # Invalid geometries
        if lr.n_invalid_merged > 0:
            if not t.allow_invalid:
                lr.severity = Severity.ERROR
                lr.messages.append(
                    f"{lr.n_invalid_merged} invalid geometries in merged output"
                )
            else:
                if lr.severity == Severity.OK:
                    lr.severity = Severity.WARNING
                lr.messages.append(
                    f"{lr.n_invalid_merged} invalid geometries (allowed by config)"
                )

        # Boundary splits: only INFO (expected behaviour)
        if lr.boundary_splits > 0:
            lr.messages.append(
                f"{lr.boundary_splits} boundary artefacts (lines/points) "
                f"dropped by normalization — expected"
            )

        if not lr.messages:
            lr.severity = Severity.OK

    # ── rich output ───────────────────────────────────────────────────────────

    def _print_report(self, report: ValidationReport) -> None:
        # ── per-layer detail table ────────────────────────────────────────────
        table = Table(
            title="Merge validation — spatial layers",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Layer",             style="cyan",    no_wrap=True)

        # One column per source
        for src in self.masks.keys():
            table.add_column(f"n_{src}", justify="right", style="dim")

        table.add_column("n_merged",          justify="right", style="white")
        table.add_column("delta",             justify="right")
        table.add_column("Δ %",               justify="right")
        table.add_column("boundary\ncand.",   justify="right", style="dim")
        table.add_column("boundary\ndrops",   justify="right", style="dim")
        table.add_column("invalid",           justify="right")
        table.add_column("status",            justify="center")

        SEV_STYLE = {
            Severity.OK:      ("[green]✓ OK[/green]",      ""),
            Severity.INFO:    ("[blue]ℹ INFO[/blue]",      ""),
            Severity.WARNING: ("[yellow]⚠ WARN[/yellow]",  "yellow"),
            Severity.ERROR:   ("[red]✗ ERROR[/red]",       "red"),
        }

        for lr in report.layers:
            sev_label, row_style = SEV_STYLE[lr.severity]

            # delta formatting
            if lr.delta < 0:
                delta_str = f"[red]{lr.delta}[/red]"
                dpct_str  = f"[red]{lr.delta_pct:.1f}%[/red]"
            elif lr.delta > 0:
                delta_str = f"[yellow]+{lr.delta}[/yellow]"
                dpct_str  = f"[yellow]+{lr.delta_pct:.1f}%[/yellow]"
            else:
                delta_str = "[green]0[/green]"
                dpct_str  = "[green]0.0%[/green]"

            invalid_str = (
                f"[red]{lr.n_invalid_merged}[/red]"
                if lr.n_invalid_merged > 0
                else "[green]0[/green]"
            )

            row = [lr.layer]
            for src in self.masks.keys():
                row.append(str(lr.source_counts.get(src, "—")))
            row += [
                str(lr.n_merged),
                delta_str,
                dpct_str,
                str(lr.boundary_candidates) if lr.boundary_candidates else "—",
                str(lr.boundary_splits)      if lr.boundary_splits      else "—",
                invalid_str,
                sev_label,
            ]
            table.add_row(*row, style=row_style)

        console.print(table)

        # ── messages ─────────────────────────────────────────────────────────
        for lr in report.layers:
            for msg in lr.messages:
                if lr.severity == Severity.ERROR:
                    console.print(f"  [red]✗[/red] [bold]{lr.layer}[/bold]: {msg}")
                elif lr.severity == Severity.WARNING:
                    console.print(f"  [yellow]⚠[/yellow] {lr.layer}: {msg}")
                elif lr.severity == Severity.INFO:
                    console.print(f"  [blue]ℹ[/blue] {lr.layer}: {msg}")

        # ── summary banner ────────────────────────────────────────────────────
        total_source   = sum(lr.n_source_total for lr in report.layers)
        total_merged   = sum(lr.n_merged       for lr in report.layers)
        total_invalid  = sum(lr.n_invalid_merged for lr in report.layers)
        total_splits   = sum(lr.boundary_splits  for lr in report.layers)
        n_errors   = sum(1 for lr in report.layers if lr.severity == Severity.ERROR)
        n_warnings = sum(1 for lr in report.layers if lr.severity == Severity.WARNING)

        console.print()
        summary = Table(title="Summary", box=box.SIMPLE, show_header=False)
        summary.add_column("metric", style="cyan")
        summary.add_column("value",  style="white")
        summary.add_row("Total source features",  f"{total_source:,}")
        summary.add_row("Total merged features",  f"{total_merged:,}")
        summary.add_row("Net delta",
                        f"[green]{total_merged - total_source:+,}[/green]"
                        if total_merged >= total_source
                        else f"[red]{total_merged - total_source:+,}[/red]")
        summary.add_row("Boundary artefacts dropped", str(total_splits) if total_splits else "—")
        summary.add_row("Invalid geometries in output",
                        f"[red]{total_invalid}[/red]" if total_invalid else "[green]0[/green]")
        summary.add_row("Layers with warnings", str(n_warnings))
        summary.add_row("Layers with errors",
                        f"[red]{n_errors}[/red]" if n_errors else "[green]0[/green]")
        summary.add_row("Elapsed", f"{report.elapsed_sec:.1f}s")
        console.print(summary)

        if report.has_errors:
            console.print("[bold red]❌  Validation FAILED — merge should not be published[/bold red]\n")
        elif report.has_warnings:
            console.print("[bold yellow]⚠  Validation passed with warnings — review before publishing[/bold yellow]\n")
        else:
            console.print("[bold green]✅  Validation passed[/bold green]\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--rc1",    type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--rc2",    type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--custom-sources-dir", type=click.Path(exists=True, path_type=Path), default=None,
              help="Directory containing custom *.gdb sources (e.g. Saas.gdb, BCK_2016.gdb)")
@click.option("--merged", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to the merged output GDB/GPKG")
@click.option("--admin",  type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to administrative_zones.gpkg")
@click.option("--warn-loss",  default=1.0,  show_default=True,
              help="Feature loss %% that triggers a WARNING")
@click.option("--error-loss", default=5.0,  show_default=True,
              help="Feature loss %% that triggers an ERROR (exit 1)")
@click.option("--allow-invalid", is_flag=True, default=False,
              help="Downgrade invalid geometry errors to warnings")
@click.option("--allow-missing", is_flag=True, default=False,
              help="Downgrade missing layer errors to warnings")
@click.option("--mapsheet", multiple=True, type=int, default=None,
              help="Restrict validation to specific mapsheet numbers")
@click.option("--json", "json_path", type=click.Path(path_type=Path), default=None,
              help="Write full report as JSON (useful as CI artifact)")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Also estimate boundary artefact counts (slower)")
def cli(rc1, rc2, custom_sources_dir, merged, admin, warn_loss, error_loss,
        allow_invalid, allow_missing, mapsheet, json_path, verbose):
    """Validate a merged GeoCover GDB against its RC1/RC2/custom sources."""

    sources = {}
    if rc1:
        sources["RC1"] = rc1
    if rc2:
        sources["RC2"] = rc2

    if not sources and not custom_sources_dir:
        console.print("[red]Provide at least --rc1, --rc2, or --custom-sources-dir[/red]")
        sys.exit(2)

    thresholds = ValidationThresholds(
        warn_loss_pct  = warn_loss,
        error_loss_pct = error_loss,
        allow_invalid  = allow_invalid,
        allow_missing  = allow_missing,
    )

    validator = MergeValidator(
        sources              = sources,
        merged_path          = merged,
        admin_path           = admin,
        custom_sources_dir   = custom_sources_dir,
        thresholds           = thresholds,
        mapsheet_numbers     = list(mapsheet) if mapsheet else None,
        verbose              = verbose,
    )

    report = validator.run()

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        console.print(f"[dim]Report written to {json_path}[/dim]")

    sys.exit(1 if report.has_errors else 0)


if __name__ == "__main__":
    cli()
