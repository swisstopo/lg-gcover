"""Tools for comparing and merging mapfile classes."""

import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassChange:
    """Represents a change in a CLASS definition."""
    class_name: str
    change_type: str  # 'added', 'removed', 'modified'
    details: List[str] = None


def detect_changes(
        original_file: Path,
        new_file: Path
) -> Dict[str, List[ClassChange]]:
    """
    Detect what changed between original and new classes.

    Returns:
        Dict with 'added', 'removed', 'modified' lists
    """
    if not original_file.exists():
        return {'added': [], 'removed': [], 'modified': []}

    original_classes = parse_class_names(original_file.read_text())
    new_classes = parse_class_names(new_file.read_text())

    added = [ClassChange(name, 'added') for name in new_classes - original_classes]
    removed = [ClassChange(name, 'removed') for name in original_classes - new_classes]

    # Simple detection - could be enhanced
    modified = []

    return {
        'added': added,
        'removed': removed,
        'modified': modified
    }


def parse_class_names(content: str) -> set:
    """Extract CLASS names from mapfile content."""
    import re

    names = set()
    for line in content.split('\n'):
        match = re.search(r'^\s*NAME\s+"([^"]+)"', line)
        if match:
            names.add(match.group(1))

    return names


def print_changes(changes: Dict[str, List[ClassChange]]):
    """Print detected changes to console."""

    if not any(changes.values()):
        logger.info("No changes detected")
        return

    logger.info("Changes detected:")
    logger.info("")

    if changes['added']:
        logger.info(f"  Added classes ({len(changes['added'])}):")
        for change in changes['added']:
            logger.info(f"    + {change.class_name}")
        logger.info("")

    if changes['removed']:
        logger.info(f"  Removed classes ({len(changes['removed'])}):")
        for change in changes['removed']:
            logger.info(f"    - {change.class_name}")
        logger.info("")

    if changes['modified']:
        logger.info(f"  Modified classes ({len(changes['modified'])}):")
        for change in changes['modified']:
            logger.info(f"    ~ {change.class_name}")
        logger.info("")


def launch_diff_tool(
        original_file: Path,
        new_file: Path,
        tool: str = "meld"
) -> bool:
    """
    Launch external diff tool for visual comparison.

    Args:
        original_file: Original file path
        new_file: New/staging file path
        tool: Diff tool to use (meld, vscode, kdiff3, etc.)

    Returns:
        True if tool launched successfully
    """
    tool_commands = {
        'meld': ['meld', str(original_file), str(new_file)],
        'vscode': ['code', '--diff', str(original_file), str(new_file)],
        'kdiff3': ['kdiff3', str(original_file), str(new_file)],
        'bcompare': ['bcompare', str(original_file), str(new_file)],
        'vimdiff': ['vimdiff', str(original_file), str(new_file)],
    }

    if tool not in tool_commands:
        logger.error(f"Unknown diff tool: {tool}")
        logger.info(f"Available tools: {', '.join(tool_commands.keys())}")
        return False

    cmd = tool_commands[tool]

    logger.info(f"Launching {tool} for comparison...")
    logger.info(f"  Original: {original_file}")
    logger.info(f"  New:      {new_file}")

    try:
        subprocess.run(cmd, check=False)
        return True
    except FileNotFoundError:
        logger.error(f"{tool} not found.")
        logger.info(f"Install with: sudo apt install {tool}")
        logger.info(f"Manual comparison: {tool} {original_file} {new_file}")
        return False