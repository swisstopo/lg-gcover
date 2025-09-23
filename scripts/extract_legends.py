import arcpy
import os
import arcpy
import os
import json
from pathlib import Path
import traceback
import sys

from loguru import logger

# --- Configuration ---
template_path = r"C:\Program Files\ArcGIS\Pro\Resources\ArcToolBox\Services\routingservices\data\Blank.aprx"  # Default blank template
new_aprx_path = r"Y:\ArcGis\ExtractLegends.aprx"

lyrx_files = [
    r"\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\NEPRO\GoTOP\Lyrx_P\GC_Pro_4.1@Default.lyrx"
]  # Layer files to load

output_dir = Path(r"X:mom\\mapserver_style\export_json")
output_dir.mkdir(parents=True, exist_ok=True)

# Remove default sink
logger.remove()

# Add console output
logger.add(sys.stdout, level="INFO", colorize=True)

# Add file output
logger.add(
    output_dir / "extract.log", level="DEBUG", rotation="5 MB", compression="zip"
)


import re
from pathlib import Path


def sanitize_filename(name: str, replacement_map=None) -> str:
    """
    Replace invalid filename characters with safe alternatives.
    Default replacements include: < > : " / \ | ? * → mapped to readable tokens.
    """
    if replacement_map is None:
        replacement_map = {
            "<": "lt",
            ">": "gt",
            ":": "-",  # colon often replaced with dash
            '"': "",  # quotes removed
            "/": "-",  # slash replaced with dash
            "\\": "-",  # backslash replaced with dash
            "|": "-",  # pipe replaced with dash
            "?": "",  # question mark removed
            "*": "x",  # asterisk replaced with 'x'
        }

    # Replace each invalid character
    for char, replacement in replacement_map.items():
        name = name.replace(char, replacement)

    # Optionally strip leading/trailing whitespace
    return name.strip()


def merge_headings(headings, headings_alias):
    final_headings = []
    for alias, heading in zip(headings_alias, headings):
        final_headings.append(alias.upper() if heading is None else heading)
    return final_headings


def process_layers(layer):

    if layer.isGroupLayer:
        logger.info(f"===Processing group: {layer.name}===")
        for sublayer in layer.listLayers():
            process_layers(sublayer)

    result = {"renderer": {}, "query_defn": {}}

    if layer.isFeatureLayer:
        sym = layer.symbology
        renderer = sym.renderer
        result["renderer"]["type"] = renderer.type

        logger.info(f"------ Processing layer {layer.name} - {renderer.type} ------")

        if layer.supports("dataSource"):
            result["dataSource"] = layer.dataSource

        if layer.supports("DefinitionQuery"):
            queries = layer.listDefinitionQueries()
            result["query_defn"] = [{k: v} for q in queries for k, v in q.items()]

        fields = {f.aliasName: f.name for f in arcpy.Describe(layer).fields}

        if renderer.type == "SimpleRenderer":
            # Extract relevant properties
            # Extract basic symbol info
            renderer = sym.renderer
            symbol = renderer.symbol
            symbol_info = {
                "type": type(symbol).__name__,  # e.g., SimpleFillSymbol
                "color": symbol.color,
                "size": getattr(symbol, "size", None),
                "style": getattr(symbol, "styleName", None),
                "name": getattr(symbol, "name", None),
                "angle": getattr(symbol, "angle", None),
                "width": getattr(symbol, "width", None),
                "outline": getattr(symbol, "outline", None),
            }

            renderer_dict = {
                "type": renderer.type,
                "label": renderer.label,
                "description": renderer.description,
                "symbol": symbol_info,
            }

            logger.debug(renderer_dict)

        elif hasattr(renderer, "groups"):
            renderer_dict = {"fields": renderer.fields, "groups": []}

            logger.info(f"GROUPS: {len(renderer.groups)}")

            for group in renderer.groups:
                logger.info(f"New group: {group.heading}")
                headings = [v.strip() for v in group.heading.split(",")]
                group_dict = {"headings": headings, "labels": [], "values": []}

                for item in group.items:
                    logger.debug(f"New item")
                    logger.info(f"   label={item.label}")
                    logger.debug(f"   values={item.values}")
                    group_dict["labels"].append(item.label)
                    cleaned_values = [
                        [None if val == "<Null>" else val for val in sublist]
                        for sublist in item.values
                    ]
                    # Extract symbol info
                    symbol = item.symbol
                    color = None
                    symbol_dict = {
                        "type": type(symbol).__name__,
                        "size": getattr(symbol, "size", None),
                        "style": getattr(symbol, "styleName", None),
                        "name": getattr(symbol, "name", None),
                        "angle": getattr(symbol, "angle", None),
                        "width": getattr(symbol, "width", None),
                        "outline": getattr(symbol, "outline", None)
                    }
                    try:
                        color = symbol.color
                    except RuntimeError as e:
                        tb = sys.exc_info()[2]
                        tbinfo = traceback.format_tb(tb)[0]
                        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str( sys.exc_info()[1])
                        msgs = "ArcPy ERRORS:\n" + arcpy.GetMessages(2) + "\n"
                        logger.error(f"Error with color: {msgs}")

                    symbol_dict['color'] = color


                    # Add symbol to item
                    group_dict.setdefault("symbols", []).append(symbol_dict)
                    group_dict["values"].append(cleaned_values)

                renderer_dict["groups"].append(group_dict)
        else:
            logger.error(f"Layer {layer.name}: {renderer.type}")
            return result

        try:
            result["renderer"] = renderer_dict

            output_path = os.path.join(
                output_dir, sanitize_filename(f"{layer.name}.json")
            )
            with open(output_path, "w") as f:
                json.dump(renderer_dict, f, indent=4)
                logger.info(f"Written {layer.name} to {output_path}")
        except Exception as e:
            logger.error(f"Cannot write symbology: {layer.name}: {e}")

    return result


def export_map_symbology(mp):
    for lyr in mp.listLayers():
        process_layers(lyr)


def main():
    # --- Create new project from template ---
    if not os.path.exists(new_aprx_path):
        os.remove(new_aprx_path)

    arcpy.mp.ArcGISProject(template_path).saveACopy(new_aprx_path)
    logger.info(f"Created new project: {new_aprx_path}")

    # --- Open the new project ---
    aprx = arcpy.mp.ArcGISProject(new_aprx_path)

    my_map = next((m for m in aprx.listMaps() if m.name == "MyMap"), None)
    if not my_map:
        # --- Add a new map ---
        my_map = aprx.createMap("MyMap")
        logger.info(f"Added new map: {my_map.name}")

        # --- Add layers to the map ---
        for lyrx_path in lyrx_files:
            if os.path.exists(lyrx_path):
                my_map.addLayer(arcpy.mp.LayerFile(lyrx_path))
                logger.info(f"Added layer: {lyrx_path}")
            else:
                logger.info(f"Layer file not found: {lyrx_path}")

    export_map_symbology(my_map)
    logger.info(f"All symbols exported to {output_dir}")

    # --- Save the project ---
    aprx.save()
    logger.info(f"Project saved to {new_aprx_path}")


if __name__ == "__main__":
    main()
