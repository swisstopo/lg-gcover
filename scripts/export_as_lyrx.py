import string
import arcpy
import os
from datetime import datetime
import re

"""

Import files form r'\\v0t0020a.adr.admin.ch\prod\lgX\TOPGIS\Bases\_Layerfiles\Default'

"""


def makeSafeFilename(fname: str) -> str:
    # Replace forbidden characters explicitly
    fname = fname.replace("<", "lt_").replace(">", "gt_")

    # Build valid character set: ASCII + Latin-1 supplement
    latin1_chars = "".join(chr(i) for i in range(256))  # full Latin-1 range
    valid_chars = frozenset(
        "-_.() " + string.ascii_letters + string.digits + latin1_chars
    )

    # Keep only valid characters
    safe = "".join(c for c in fname if c in valid_chars)

    # Replace spaces with underscores
    safe = safe.replace(" ", "_")

    # Collapse multiple underscores into one
    safe = re.sub(r"_+", "_", safe)

    # Strip leading/trailing underscores or dots
    safe = safe.strip("._")

    return safe


# Current project
aprx = arcpy.mp.ArcGISProject("CURRENT")

# Choose a map (first one here, but you can loop through all maps if needed)
m = aprx.listMaps()[0]

# Base output folder
base_folder = r"Y:\temp\lyrx_export"

# Add current date (YYYY-MM-DD)
date_str = datetime.now().strftime("%Y-%m-%d")
out_folder = os.path.join(base_folder, date_str)

# Make sure folder exists
os.makedirs(out_folder, exist_ok=True)

# Loop through layers and export each as lyrx
for lyr in m.listLayers():
    if lyr.isFeatureLayer or lyr.isRasterLayer:  # only valid layer types
        safe_name = makeSafeFilename(lyr.name)
        out_path = os.path.join(out_folder, f"{safe_name}.lyrx")
        try:
            lyr.saveACopy(out_path)
            print(f"Saved {lyr.name} → {out_path}")
        except OSError as e:
            print(f"Bad Error: {e}")
