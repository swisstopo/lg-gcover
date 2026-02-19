import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint

from rich.console import Console
from shapely.geometry import Point
from rich.progress import Progress

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from rich.progress import Progress
from rich import print as rprint
from rich.panel import Panel


import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint
from rich.progress import Progress
from rich import print as rprint
from rich.table import Table

import geopandas as gpd
import numpy as np
import click
from shapely.geometry import Point, MultiPoint
from rich.progress import Progress
from rich import print as rprint
from rich.table import Table


@click.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help="Path to input GPKG.")
@click.option('--layer', '-l', required=True, help="Layer name inside the GPKG.", default="surfaces")
@click.option('--output', '-o', default="hex_multipoints.gpkg", help="Output filename.")
@click.option('--spacing', '-s', default=80.0, help="Distance between points in meters.")
@click.option('--buffer', '-b', default=40.0, help="Inset distance from polygon edge.")
def generate_grid(input, layer, output, spacing, buffer):
    """Generates a hexagonal MultiPoint grid within specified geologic features."""

    rprint(f"[bold blue]→ Reading layer:[/bold blue] {layer}")
    gdf = gpd.read_file(input, layer=layer)

    # 1. Define Target Symbols
    target_symbols = [
        "surfaces_gins_sackungsgebiet",
        "surfaces_gins_gebiet_mit_hakenwurf",
        "surfaces_gins_rutschgebiet",
        "surfaces_gins_gebiet_mit_solifluktion"
    ]

    # 2. Filter Attributes
    # We always keep map_symbol, plus your requested list if they exist
    keep_cols = ['KIND', 'UUID', 'label', 'map_symbol', 'geometry']
    existing_cols = [c for c in keep_cols if c in gdf.columns]

    filtered_gdf = gdf[gdf['map_symbol'].isin(target_symbols)][existing_cols].copy()

    if filtered_gdf.empty:
        rprint("[bold red]Error:[/bold red] No matching features found for the specified symbols.")
        return

    results = []
    dy = spacing * np.sqrt(3) / 2  # Vertical row spacing

    # 3. Process Features
    with Progress() as progress:
        task = progress.add_task("[yellow]Generating hex grid...", total=len(filtered_gdf))

        for _, row in filtered_gdf.iterrows():
            # Apply inset
            geom = row.geometry
            inset_geom = geom.buffer(-abs(buffer))

            if inset_geom.is_empty:
                progress.update(task, advance=1)
                continue

            # Bounding box for the grid
            xmin, ymin, xmax, ymax = inset_geom.bounds
            rows = np.arange(ymin, ymax + dy, dy)
            cols = np.arange(xmin, xmax + spacing, spacing)

            feature_points = []
            for i, y in enumerate(rows):
                x_offset = (spacing / 2) if i % 2 == 1 else 0
                for x in cols:
                    p = Point(x + x_offset, y)
                    if inset_geom.contains(p):
                        feature_points.append(p)

            if feature_points:
                new_row = row.copy()
                new_row.geometry = MultiPoint(feature_points)
                new_row['pt_count'] = len(feature_points)
                results.append(new_row)

            progress.update(task, advance=1)

    # 4. Save and Report
    if results:
        out_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)
        out_gdf.to_file(output, layer=f"{layer }_aux_points", driver="GPKG")

        # Summary Table
        table = Table(title="Generation Results")
        table.add_column("Symbol", style="cyan")
        table.add_column("Points", justify="right", style="magenta")

        for symbol in target_symbols:
            count = out_gdf[out_gdf['map_symbol'] == symbol]['pt_count'].sum()
            if count > 0:
                table.add_row(symbol, str(int(count)))

        rprint(table)
        rprint(f"[bold green]Success![/bold green] File saved to {output}")
    else:
        rprint("[bold red]No points generated. Check your buffer/spacing settings.[/bold red]")


if __name__ == '__main__':
    generate_grid()





