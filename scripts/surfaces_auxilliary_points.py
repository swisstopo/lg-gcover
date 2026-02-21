import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.affinity import translate

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
    # TODO: change names if using cannonical IDs
    target_symbols = [
        "surfaces_gins_sackungsgebiet",
        "surfaces_gins_gebiet_mit_hakenwurf",
        "surfaces_gins_rutschgebiet",
        "surfaces_gins_gebiet_mit_solifluktion",
        "unco_litho_rutschmasse",
        "unco_litho_zerruettete_sackungsmasse"
    ]

    # 2. Filter Attributes
    # We always keep map_symbol, plus your requested list if they exist
    keep_cols = ['KIND', 'UUID', 'label', 'map_symbol', 'geometry']
    existing_cols = [c for c in keep_cols if c in gdf.columns]

    filtered_gdf = gdf[gdf['map_symbol'].isin(target_symbols)][existing_cols].copy()

    #filtered_gdf["translate"] = filtered_gdf["map_symbol"] == "unco_litho_zerruettete_sackungsmasse"

    if filtered_gdf.empty:
        rprint("[bold red]Error:[/bold red] No matching features found for the specified symbols.")
        return

    results = []
    dx = spacing
    dy = spacing * (math.sqrt(3) / 2)

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating aligned grid...", total=len(filtered_gdf))

        for _, row in filtered_gdf.iterrows():
            geom = row.geometry
            inset_geom = geom.buffer(-abs(buffer))
            if inset_geom.is_empty:
                progress.update(task, advance=1)
                continue

            xmin, ymin, xmax, ymax = inset_geom.bounds

            # --- GLOBAL ALIGNMENT LOGIC ---
            # Snap the starting coordinates to the nearest multiple of dx/dy
            # relative to the CRS origin (0,0)
            start_y = math.floor(ymin / dy) * dy
            end_y = math.ceil(ymax / dy) * dy

            feature_points = []

            # Use a numeric index to determine row offsetting consistently
            for y_coord in np.arange(start_y, end_y + (dy / 2), dy):
                # Row index relative to global origin
                row_idx = round(y_coord / dy)

                # Offset every "odd" global row by half dx
                global_x_offset = (dx / 2) if row_idx % 2 != 0 else 0

                # Align start_x to the global grid
                start_x = math.floor((xmin - global_x_offset) / dx) * dx + global_x_offset

                for x_coord in np.arange(start_x, xmax + (dx / 2), dx):
                    p = Point(x_coord, y_coord)
                    if inset_geom.contains(p):
                        feature_points.append(p)

            if feature_points:
                new_row = row.copy()
                new_row.geometry = MultiPoint(feature_points)
                results.append(new_row)

            progress.update(task, advance=1)

    # 4. Save and Report
    if results:
        out_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)
        out_gdf.to_file(output, layer=f"{layer}_aux_points", driver="GPKG")


        # Orignal geometries:

        filtered_gdf.to_file(output, layer=f"filtered_{layer}", driver="GPKG")

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





