from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Union
from gcover.publish.diff_tools import detect_changes, print_changes, launch_diff_tool
from gcover.publish.generator import MapServerGenerator

from loguru import logger

def find_classifications_by_name(batch_config, classification_name: str):
    """
    Find classifications by classification_name.

    Args:
        batch_config: BatchClassificationConfig
        classification_name: Name from .lyrx (e.g., 'bedrock_rc1')

    Returns:
        List of tuples: [(layer_config, classification), ...]
    """
    results = []

    for layer_config in batch_config.layers:
        for classification in layer_config.classifications:
            cls_name = getattr(classification, 'classification_name', None)

            if cls_name == classification_name:
                results.append((layer_config, classification))

    return results


def list_all_classification_names(batch_config):
    """
    List all classification names in config.

    Returns:
        List of dicts with classification info
    """
    classifications = []

    for layer_config in batch_config.layers:
        for classification in layer_config.classifications:
            cls_name = getattr(classification, 'classification_name', None)

            if cls_name:
                mapfile_config = getattr(classification, 'mapfile_config', None)
                mode = 'auto'
                if mapfile_config:
                    mode = getattr(mapfile_config, 'classes_mode', 'auto')

                classifications.append({
                    'name': cls_name,
                    'layer': layer_config.gcover_layer,
                    'style_file': getattr(classification, 'style_file', 'unknown'),
                    'mode': mode
                })

    return classifications


def get_all_frozen_classifications(batch_config):
    """
    Get all frozen classifications as tuples.

    Returns:
        List of tuples: [(layer_config, classification), ...]
    """
    results = []

    for layer_config in batch_config.layers:
        for classification in layer_config.classifications:
            mapfile_config = getattr(classification, 'mapfile_config', None)
            if mapfile_config and mapfile_config.classes_mode == 'frozen':
                results.append((layer_config, classification))

    return results


def get_all_classifications(batch_config):
    """
    Get all classifications as tuples.

    Returns:
        List of tuples: [(layer_config, classification), ...]
    """
    results = []

    for layer_config in batch_config.layers:
        for classification in layer_config.classifications:
            results.append((layer_config, classification))

    return results

def handle_staging_result(
    staging_file_path,
    symbol_prefix: str,
    mapfile_config,
    output_dir: Path,
    diff_tool: Optional[str] = None,
):
    """
    Handle result from generate_layer() in staging mode.

    This function:
    1. Determines the original file path
    2. Detects changes between original and staging
    3. Prints changes to console
    4. Optionally launches diff tool

    Args:
        staging_file_path: Path returned by generate_layer() (str or Path)
        symbol_prefix: Symbol prefix for this layer
        mapfile_config: MapfileGenerationConfig (can be None)
        output_dir: Base output directory
        diff_tool: Diff tool to launch (None = don't launch)
    """
    from pathlib import Path

    # Convert to Path if string
    staging_file = Path(staging_file_path)

    # Determine original file path
    if (
        mapfile_config
        and hasattr(mapfile_config, "classes_file")
        and mapfile_config.classes_file
    ):
        # Explicit path from config
        original_file = Path(mapfile_config.classes_file)
    else:
        # Default path: output_dir/classes/<symbol_prefix>_classes.inc
        original_file = output_dir / "classes" / f"{symbol_prefix}_classes.inc"

    logger.info("")
    logger.info(f"✓ Generated staging file: {staging_file}")
    logger.info("")

    # Detect changes if original exists
    if original_file.exists():
        changes = detect_changes(original_file, staging_file)
        print_changes(changes)

        logger.info("")
        logger.info(f"Compare with:")
        logger.info(f"  {diff_tool or 'meld'} {original_file} {staging_file}")
        logger.info("")
    else:
        logger.info("ℹ️  Original file doesn't exist yet (first generation)")
        logger.info(f"   Will be created at: {original_file}")
        logger.info("")

    # Launch diff tool if requested
    if diff_tool:
        if not original_file.exists():
            logger.warning(f"Cannot launch diff tool - original file doesn't exist")
            logger.info(f"   Create it first with: gcover publish mapserver")
        else:
            success = launch_diff_tool(original_file, staging_file, diff_tool)

            if success:
                logger.info("")
                logger.info("💡 Tip: After merging in diff tool:")
                logger.info("   1. Save changes to original file")
                logger.info("   2. Close diff tool")
                logger.info(f"   3. git add {original_file}")
                logger.info(f"   4. git commit -m 'Merge changes from .lyrx update'")
                logger.info("")


def export_unique_items_to_excel(
        data_dict: dict,
        output_path: Path,
        columns: list = None
) -> Path:
    """
    Extracts unique, stripped strings from a dict of comma-separated values
    and saves them to an Excel file.
    """
    if columns is None:
        columns = ["id", "de", "fr"]

    # 1. Extraction Logic
    # We use a set comprehension for uniqueness and built-in filtering
    unique_items = {
        item.strip()
        for value in data_dict.values()
        if value  # Handles None, empty strings, or empty lists
        for item in str(value).split(',')
        if item.strip()
    }

    # 2. DataFrame Creation
    # We sort here to ensure the Excel file is predictable/ordered
    df = pd.DataFrame(sorted(unique_items), columns=[columns[0]])

    # Add any extra empty columns requested
    for col in columns[1:]:
        df[col] = ""

    # 3. Save with error handling
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
        return output_path
    except Exception as e:
        raise IOError(f"Failed to save Excel file to {output_path}: {e}")