"""
Configuration management for gcover.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

# Dictionaries
RELEASE_CANDIDATES = {"RC1": "2026-12-31", "RC2": "2030-12-31"}

long_to_short = {v: k for k, v in RELEASE_CANDIDATES.items()}


def get_all_rcs():
    """
    Returns all RCs.

    Returns:
        list: ['RC1', '2030-12-21', ...]
    """
    flat_list = []
    for k, v in RELEASE_CANDIDATES.items():
        flat_list.extend([k, v])
    return flat_list


def convert_rc(value, force=None):
    """
    Converts between short and long RC forms.

    Parameters:
        value (str): The input value to convert.
        force (str): Optional. 'short' forces long-to-short,
                     'long' forces short-to-long.
                     If None, auto-detects direction.

    Returns:
        str or None: Converted value or None if not found.
    """
    if force == "short":
        return long_to_short.get(value)
    elif force == "long":
        return RELEASE_CANDIDATES.get(value)
    else:
        # Auto-detect mode
        if value in RELEASE_CANDIDATES:
            return RELEASE_CANDIDATES[value]
        elif value in long_to_short:
            return long_to_short[value]
        else:
            return None
