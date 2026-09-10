#!/usr/bin/env python3
"""Check geometry quality and UUID format in any OGR-readable file (GDB, GPKG…).

Three checks per layer:

  many-parts     organizePolygons() received a polygon with more than 100 parts.
                 Detected by counting rings in the shapely geometry (one read).

  unclosed-ring  Non closed ring detected.
                 GDAL auto-fixes these silently, so detection uses a GDAL error
                 handler installed during an OGR forward pass. Requires osgeo.

  bogus-uuid     UUID values that don't match ESRI canonical format
                 {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX} (uppercase hex, braces).
                 Detects: lowercase, missing braces, malformed, or NULL.

many-parts and unclosed-ring rows also carry an OGC IsValid() verdict (via
shapely/GEOS), so a warning raised by GDAL's read-time heuristics can be told
apart from an actually invalid geometry — many-parts is just organizePolygons()
flagging a polygon as slow to assemble, and unclosed-ring is auto-closed by
GDAL on read; neither implies the result is invalid.

Every detail row also carries the last-modification date (DATEOFCHANGE) and the
OPERATOR who last edited the feature, so a flagged geometry can be traced back to
its author.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from shapely import wkb
from shapely.validation import explain_validity

console = Console()

# ESRI / TOPGIS audit fields carried on every GeoCover feature.
DATE_FIELD = "DATEOFCHANGE"
OPERATOR_FIELD = "OPERATOR"
OBJECTID_FIELD = "OBJECTID"

POLYGON_LAYERS = [
    "GC_BEDROCK",
    "GC_SURFACES",
    "GC_UNCO_DESPOSIT",
    "GC_EXPLOIT_GEOMAT_PLG",
    "GC_MAPSHEET",
]

# Canonical ESRI UUID: {8-4-4-4-12 uppercase hex, enclosed in braces}
_VALID_UUID = re.compile(
    r"^\{[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\}$"
)


# ---------------------------------------------------------------------------
# Ring-count helper
# ---------------------------------------------------------------------------

def _audit_str(value, *, is_date: bool = False) -> str:
    """Render a DATEOFCHANGE / OPERATOR value for display ('—' when absent)."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s in ("", "None", "NaT", "nan"):
        return "—"
    if is_date and len(s) >= 10:
        s = s[:10]  # keep the date, drop any time component
    return s


def audit_fields(row) -> tuple[str, str]:
    """Return (modification-date, operator) display strings for a GDF row."""
    return (
        _audit_str(row.get(DATE_FIELD), is_date=True),
        _audit_str(row.get(OPERATOR_FIELD)),
    )


def row_objectid(row, idx):
    """Return the OBJECTID for a GDF row.

    Prefers an explicit OBJECTID attribute column when the driver exposes
    one; otherwise falls back to *idx*, which is the OGR FID (read with
    fid_as_index=True) — for ESRI File Geodatabases the FID equals OBJECTID.
    """
    val = row.get(OBJECTID_FIELD)
    return idx if val is None else val


def validity_info(geom) -> tuple[bool, str]:
    """Return (is_valid, reason) for a shapely geometry via OGC IsValid() / GEOS.

    reason is '' when valid. Distinguishes a genuinely invalid geometry from a
    read-time GDAL heuristic warning (many-parts, unclosed-ring), which does
    not by itself imply invalidity.
    """
    if geom is None:
        return False, "no geometry"
    if geom.is_valid:
        return True, ""
    return False, explain_validity(geom)


def ogr_geom_to_shapely(geom):
    """Convert an osgeo.ogr.Geometry to shapely via WKB, or None."""
    if geom is None:
        return None
    return wkb.loads(bytes(geom.ExportToWkb()))


def ring_count(geom) -> int:
    if geom is None:
        return 0
    t = geom.geom_type
    if t == "Polygon":
        return 1 + len(list(geom.interiors))
    if t == "MultiPolygon":
        return sum(1 + len(list(p.interiors)) for p in geom.geoms)
    return 0


# ---------------------------------------------------------------------------
# Many-parts check  (uses already-loaded GDF)
# ---------------------------------------------------------------------------

def check_many_parts(gdf: gpd.GeoDataFrame, threshold: int = 100) -> list[dict]:
    """Return rows whose ring count exceeds *threshold*."""
    results = []
    for idx, row in gdf.iterrows():
        n = ring_count(row.geometry)
        if n > threshold:
            date, operator = audit_fields(row)
            is_valid, reason = validity_info(row.geometry)
            results.append({
                "objectid": row_objectid(row, idx),
                "uuid": row.get("UUID", "—"),
                "geom_type": row.geometry.geom_type if row.geometry else "—",
                "rings": n,
                "date": date,
                "operator": operator,
                "valid": is_valid,
                "reason": reason,
            })
    return results


# ---------------------------------------------------------------------------
# Unclosed-ring check  (single OGR pass with GDAL error handler)
# ---------------------------------------------------------------------------

def check_unclosed_rings(path: Path, layer: str) -> list[dict]:
    """Return features triggering a 'Non closed ring' GDAL error.

    GDAL auto-closes rings before returning geometry to Python, so the only
    reliable detection is an error handler. Crucially, GDAL materializes (and
    thus validates/warns about) a feature's geometry *while fetching it* —
    inside GetNextFeature() itself, not afterwards — so the flag is reset
    immediately before each fetch and read immediately after, with the FID
    taken from the feature that fetch just returned. (An earlier version of
    this function read a stale FID left over from the *previous* iteration,
    silently misattributing every hit to the wrong feature.) One forward
    pass, O(n). Requires osgeo.
    """
    try:
        from osgeo import gdal, ogr
    except ImportError:
        console.print("[yellow]osgeo not available — unclosed-ring check skipped.[/yellow]")
        return []

    gdal.DontUseExceptions()

    flagged = [False]

    def _handler(_level: int, _num: int, msg: str) -> None:
        if "Non closed ring" in msg:
            flagged[0] = True

    results: list[dict] = []
    gdal.PushErrorHandler(_handler)
    try:
        ds = ogr.Open(str(path))
        if ds is None:
            console.print(f"[red]OGR could not open {path}[/red]")
            return []
        lyr = ds.GetLayerByName(layer)
        if lyr is None:
            return []
        defn = lyr.GetLayerDefn()
        uuid_idx = defn.GetFieldIndex("UUID")
        date_idx = defn.GetFieldIndex(DATE_FIELD)
        op_idx = defn.GetFieldIndex(OPERATOR_FIELD)
        objectid_idx = defn.GetFieldIndex(OBJECTID_FIELD)
        lyr.ResetReading()
        while True:
            flagged[0] = False
            feat = lyr.GetNextFeature()
            if feat is None:
                break
            if not flagged[0]:
                continue
            is_valid, reason = validity_info(ogr_geom_to_shapely(feat.GetGeometryRef()))
            results.append({
                "objectid": (
                    feat.GetField(objectid_idx) if objectid_idx >= 0 else feat.GetFID()
                ),
                "uuid": (feat.GetField(uuid_idx) if uuid_idx >= 0 else None) or "—",
                "date": _audit_str(
                    feat.GetFieldAsString(date_idx) if date_idx >= 0 else None,
                    is_date=True,
                ),
                "operator": _audit_str(
                    feat.GetField(op_idx) if op_idx >= 0 else None
                ),
                "valid": is_valid,
                "reason": reason,
            })
        ds = None
    finally:
        gdal.PopErrorHandler()

    return results


# ---------------------------------------------------------------------------
# Bogus-UUID check  (vectorised, uses already-loaded GDF)
# ---------------------------------------------------------------------------

def _uuid_category(u: str | None) -> str | None:
    """Return a short label for what's wrong, or None if the UUID is valid."""
    if u is None or str(u).strip() in ("", "None", "nan"):
        return "null"
    s = str(u)
    if _VALID_UUID.match(s):
        return None
    # Diagnose the most common failures in priority order
    has_braces = s.startswith("{") and s.endswith("}")
    inner = s[1:-1] if has_braces else s
    parts = inner.split("-")
    correct_groups = len(parts) == 5 and [len(p) for p in parts] == [8, 4, 4, 4, 12]
    if not has_braces and correct_groups and inner == inner.upper():
        return "missing-braces"
    if correct_groups and inner != inner.upper():
        return "lowercase"
    return "malformed"


def check_bogus_uuids(gdf: gpd.GeoDataFrame, max_rows: int = 50) -> list[dict]:
    """Return rows with UUID values outside ESRI canonical format."""
    if "UUID" not in gdf.columns:
        return []

    results = []
    for idx, row in gdf.iterrows():
        cat = _uuid_category(row.get("UUID"))
        if cat is not None:
            date, operator = audit_fields(row)
            results.append({
                "objectid": row_objectid(row, idx),
                "uuid": str(row.get("UUID", "")),
                "issue": cat,
                "date": date,
                "operator": operator,
            })
        if len(results) >= max_rows:
            break
    return results


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _uuid_cell(uuid: str) -> str:
    """Return a rich-markup string: magenta + issue label when UUID is bogus."""
    cat = _uuid_category(uuid)
    if cat is None:
        return uuid
    return f"[magenta]{uuid} [dim]({cat})[/dim][/magenta]"


def _valid_cell(r: dict) -> str:
    """Return a rich-markup string for the IsValid() verdict of a flagged row."""
    if "valid" not in r:
        return "[dim]—[/dim]"
    if r["valid"]:
        return "[green]valid[/green]"
    return f"[bold red]INVALID[/bold red] [dim]({r.get('reason', '')})[/dim]"


def _print_many_parts(layer: str, issues: list[dict]) -> None:
    t = Table(title=f"[yellow]Many-parts polygons[/yellow] — {layer}", box=box.SIMPLE_HEAD)
    t.add_column("OBJECTID", justify="right", style="dim")
    t.add_column("UUID")
    t.add_column("Geom type")
    t.add_column("Ring count", justify="right", style="bold red")
    t.add_column("IsValid", style="cyan")
    t.add_column("Changed", style="cyan")
    t.add_column("Operator", style="cyan")
    for r in sorted(issues, key=lambda x: -x["rings"]):
        t.add_row(
            str(r["objectid"]), _uuid_cell(str(r["uuid"])), r["geom_type"], str(r["rings"]),
            _valid_cell(r), r.get("date", "—"), r.get("operator", "—"),
        )
    console.print(t)


def _print_unclosed(layer: str, issues: list[dict]) -> None:
    t = Table(title=f"[red]Unclosed-ring features[/red] — {layer}", box=box.SIMPLE_HEAD)
    t.add_column("OBJECTID", justify="right", style="dim")
    t.add_column("UUID")
    t.add_column("IsValid", style="cyan")
    t.add_column("Changed", style="cyan")
    t.add_column("Operator", style="cyan")
    for r in issues:
        t.add_row(
            str(r["objectid"]), _uuid_cell(str(r["uuid"])),
            _valid_cell(r), r.get("date", "—"), r.get("operator", "—"),
        )
    console.print(t)


def _print_bogus_uuids(layer: str, issues: list[dict], total_bad: int) -> None:
    title = f"[magenta]Bogus UUIDs[/magenta] — {layer}"
    if total_bad > len(issues):
        title += f" [dim](showing {len(issues)} of {total_bad})[/dim]"
    t = Table(title=title, box=box.SIMPLE_HEAD)
    t.add_column("OBJECTID", justify="right", style="dim")
    t.add_column("UUID value")
    t.add_column("Issue", style="magenta")
    t.add_column("Changed", style="cyan")
    t.add_column("Operator", style="cyan")
    for r in issues:
        t.add_row(
            str(r["objectid"]), r["uuid"], r["issue"],
            r.get("date", "—"), r.get("operator", "—"),
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Core scan  (one read per layer, reused by all checks)
# ---------------------------------------------------------------------------

def scan_file(
    path: Path,
    layers: list[str],
    parts_threshold: int,
    skip_unclosed: bool,
    skip_uuid: bool,
) -> None:
    console.print(Panel(f"[bold]{path.name}[/bold]\n[dim]{path}[/dim]", expand=False))

    any_issue = False
    for layer in layers:
        console.print(f"  [cyan]{layer}[/cyan]…", end=" ")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gdf = gpd.read_file(path, layer=layer, fid_as_index=True)
            msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        except Exception as exc:
            console.print(f"[dim]skipped ({exc})[/dim]")
            continue

        has_parts    = any("100 parts" in m for m in msgs)
        has_unclosed = any("Non closed ring" in m for m in msgs)

        # UUID check is pure in-memory — run unconditionally (unless --skip-uuid)
        uuid_issues: list[dict] = []
        total_uuid_bad = 0
        if not skip_uuid and "UUID" in gdf.columns:
            # Count all bad rows for the summary, but cap the detail table
            all_bad_mask = (
                gdf["UUID"].isna()
                | ~gdf["UUID"].astype(str).str.fullmatch(
                    r"\{[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\}"
                )
            )
            total_uuid_bad = int(all_bad_mask.sum())
            if total_uuid_bad:
                uuid_issues = check_bogus_uuids(gdf)

        tags = []
        if has_parts:
            tags.append("[yellow]many-parts[/yellow]")
        if has_unclosed:
            tags.append("[red]unclosed-ring[/red]")
        if total_uuid_bad:
            tags.append(f"[magenta]bogus-uuid:{total_uuid_bad}[/magenta]")
        console.print(", ".join(tags) if tags else "[green]OK[/green]")

        if has_parts:
            issues = check_many_parts(gdf, parts_threshold)
            if issues:
                _print_many_parts(layer, issues)
                any_issue = True

        if has_unclosed and not skip_unclosed:
            console.print(f"   [dim]Locating unclosed-ring feature(s) in {layer}…[/dim]")
            issues = check_unclosed_rings(path, layer)
            if issues:
                _print_unclosed(layer, issues)
                any_issue = True

        if uuid_issues:
            _print_bogus_uuids(layer, uuid_issues, total_uuid_bad)
            any_issue = True

    if not any_issue:
        console.print("[green]No issues found.[/green]")
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--layers", "layer_names",
    default=",".join(POLYGON_LAYERS), show_default=True,
    help="Comma-separated list of layers to check.",
)
@click.option(
    "--parts-threshold", default=100, show_default=True,
    help="Ring-count threshold for the many-parts check.",
)
@click.option("--skip-unclosed", is_flag=True, default=False,
              help="Skip the unclosed-ring OGR pass.")
@click.option("--skip-uuid", is_flag=True, default=False,
              help="Skip the bogus-UUID check.")
def main(
    source_path: Path,
    layer_names: str,
    parts_threshold: int,
    skip_unclosed: bool,
    skip_uuid: bool,
) -> None:
    """Check geometry quality and UUID format in SOURCE_PATH (GDB, GPKG, …).

    Checks per layer: many-parts polygons, unclosed rings, bogus UUIDs.
    Each layer is read once; all three checks share that GDF.

    \b
    Examples:
      check_gdb_geometry.py RC2.gdb
      check_gdb_geometry.py denormalized.gpkg --layers bedrock,surfaces,linear_objects
      check_gdb_geometry.py RC1.gdb --layers GC_BEDROCK --skip-unclosed
    """
    layers = [l.strip() for l in layer_names.split(",") if l.strip()]
    scan_file(source_path, layers, parts_threshold, skip_unclosed, skip_uuid)


if __name__ == "__main__":
    main()
