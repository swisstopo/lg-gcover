#!/usr/bin/env python3
"""Highlight content differences between two xlsx files (first sheet only)."""

import io
import sys
from pathlib import Path
from typing import Any

import click
import openpyxl
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()


def load_sheet(path: Path, key_col: int = 0) -> tuple[list[str], dict[Any, list[Any]]]:
    """Return (headers, {key: row_values}) for the first sheet."""
    raw = path.read_bytes()
    wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return [], {}
    headers = [str(h) if h is not None else "" for h in rows[0]]
    data: dict[Any, list[Any]] = {}
    for row in rows[1:]:
        if all(v is None for v in row):
            continue
        key = row[key_col]
        data[key] = list(row)
    return headers, data


def cell_str(v: Any) -> str:
    return "" if v is None else str(v)


@click.command()
@click.argument("new_file", type=click.Path(exists=True, path_type=Path),
                default="src/gcover/data/GC_Sources_PA.xlsx")
@click.argument("old_file", type=click.Path(exists=True, path_type=Path),
                default="src/gcover/data/GC_Sources_PA.previous.xlsx")
@click.option("--key-col", default=0, show_default=True,
              help="Column index (0-based) used as row identifier.")
@click.option("--no-unchanged", is_flag=True, default=False,
              help="Hide rows that are identical in both files.")
@click.option("--cols", multiple=True, metavar="COL",
              help="Restrict diff to these column names (repeatable). "
                   "E.g. --cols BKP --cols Version")
def main(new_file: Path, old_file: Path, key_col: int, no_unchanged: bool,
         cols: tuple[str, ...]) -> None:
    """Show a rich diff table between NEW_FILE and OLD_FILE (first sheet only).

    Rows are matched by the first column value (override with --key-col).
    Legend: green = added, red = removed, yellow = changed cell.
    """
    headers_new, new_data = load_sheet(new_file, key_col)
    headers_old, old_data = load_sheet(old_file, key_col)

    headers = headers_new or headers_old

    # Resolve --cols to a set of column indices; unknown names are warned about.
    if cols:
        unknown = [c for c in cols if c not in headers]
        if unknown:
            console.print(f"[red]Unknown column(s): {', '.join(unknown)}[/red]")
            console.print(f"Available: {', '.join(h for h in headers if h)}")
            raise SystemExit(1)
        filter_indices: set[int] | None = {headers.index(c) for c in cols}
    else:
        filter_indices = None  # all columns

    def row_differs(new_row: list[Any], old_row: list[Any]) -> bool:
        if filter_indices is None:
            return new_row != old_row
        return any(
            new_row[i] != old_row[i]
            for i in filter_indices
            if i < len(new_row) and i < len(old_row)
        )

    all_keys = list(dict.fromkeys(list(new_data.keys()) + list(old_data.keys())))

    added = [k for k in all_keys if k in new_data and k not in old_data]
    removed = [k for k in all_keys if k in old_data and k not in new_data]
    changed = [
        k for k in all_keys
        if k in new_data and k in old_data and row_differs(new_data[k], old_data[k])
    ]
    unchanged = [
        k for k in all_keys
        if k in new_data and k in old_data and not row_differs(new_data[k], old_data[k])
    ]

    console.print()
    console.print(f"[bold]NEW :[/bold] {new_file.name}")
    console.print(f"[bold]OLD :[/bold] {old_file.name}")
    if filter_indices is not None:
        console.print(f"[bold]COLS:[/bold] {', '.join(cols)}")
    console.print()
    console.print(
        f"[green]{len(added)} added[/green]  "
        f"[red]{len(removed)} removed[/red]  "
        f"[yellow]{len(changed)} changed[/yellow]  "
        f"{len(unchanged)} unchanged"
    )
    console.print()

    if not (added or removed or changed) and no_unchanged:
        console.print("[dim]No differences found.[/dim]")
        return

    # Columns to display: always key col + filtered cols (or all)
    if filter_indices is not None:
        display_indices = sorted({key_col} | filter_indices)
    else:
        display_indices = list(range(len(headers)))

    display_headers = [headers[i] for i in display_indices]

    table = Table(box=box.SIMPLE_HEAD, show_lines=True, expand=False)
    table.add_column("Δ", width=3, no_wrap=True)
    for h in display_headers:
        table.add_column(h, overflow="fold")

    def fmt_row(row: list[Any], old_row: list[Any] | None, style: str) -> list[Text]:
        cells = []
        for i in display_indices:
            v = row[i] if i < len(row) else None
            txt = cell_str(v)
            if (old_row is not None and i < len(old_row) and v != old_row[i]
                    and (filter_indices is None or i in filter_indices)):
                cells.append(Text(txt, style="bold yellow"))
            else:
                cells.append(Text(txt, style=style))
        return cells

    for k in all_keys:
        if k in added:
            row_cells = fmt_row(new_data[k], None, "green")
            table.add_row(Text("+", style="bold green"), *row_cells)
        elif k in removed:
            row_cells = fmt_row(old_data[k], None, "red")
            table.add_row(Text("-", style="bold red"), *row_cells)
        elif k in changed:
            old_cells = fmt_row(old_data[k], None, "dim red")
            new_cells = fmt_row(new_data[k], old_data[k], "")
            table.add_row(Text("~", style="bold yellow"), *old_cells)
            table.add_row(Text("→", style="bold yellow"), *new_cells)
        elif not no_unchanged:
            row_cells = fmt_row(new_data[k], None, "dim")
            table.add_row(Text(" ", style=""), *row_cells)

    console.print(table)

    # Summary of which columns changed most
    if changed and filter_indices is None:
        col_counts: dict[str, int] = {}
        for k in changed:
            for i, (nv, ov) in enumerate(zip(new_data[k], old_data[k])):
                if nv != ov and i < len(headers):
                    col_counts[headers[i]] = col_counts.get(headers[i], 0) + 1
        console.print("[bold]Columns with changes:[/bold]")
        for col, cnt in sorted(col_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {col}: {cnt}")
        console.print()


if __name__ == "__main__":
    main()
