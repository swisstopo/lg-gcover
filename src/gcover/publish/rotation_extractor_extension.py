#!/usr/bin/env python3
"""
Extension for ESRI Classification Extractor - Rotation Support

Adds parsing for CIMRotationVisualVariable to extract symbol rotation
information from renderer visualVariables.

Usage:
    Add the RotationInfo dataclass and parsing methods to your existing
    esri_classification_extractor.py file.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger


# =============================================================================
# NEW DATACLASS: RotationInfo
# =============================================================================

@dataclass
class RotationInfo:
    """
    Symbol rotation information extracted from CIMRotationVisualVariable.
    
    Focuses on Z-axis rotation (2D map rotation) which is the most common
    for 2D cartography.
    """
    # Rotation field/expression (e.g., "AZIMUTH", "[AZIMUTH]")
    field_name: Optional[str] = None
    expression: Optional[str] = None  # Full expression like "[AZIMUTH]"
    
    # Rotation type
    rotation_type: Optional[str] = None  # "Geographic" or "Arithmetic"
    
    # Variable info type
    variable_type: Optional[str] = None  # "Expression", "Random", "None"
    
    # Range information (if applicable)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Raw data for reference
    raw_rotation_variable: Dict[str, Any] = field(default_factory=dict)
    
    def get_simple_field_name(self) -> Optional[str]:
        """
        Extract simple field name from expression.
        
        Examples:
            "[AZIMUTH]" -> "AZIMUTH"
            "AZIMUTH" -> "AZIMUTH"
        
        Returns:
            Simple field name without brackets, or None
        """
        if self.field_name:
            return self.field_name
            
        if not self.expression:
            return None
            
        # Remove brackets if present
        expr = self.expression.strip()
        if expr.startswith('[') and expr.endswith(']'):
            return expr[1:-1]
        
        return expr
    
    def is_geographic(self) -> bool:
        """Check if rotation type is Geographic (vs Arithmetic)."""
        return self.rotation_type == "Geographic"
    
    def is_expression_based(self) -> bool:
        """Check if rotation is based on a field expression (vs random)."""
        return self.variable_type == "Expression"


# =============================================================================
# NEW PARSER CLASS: CIMRotationParser
# =============================================================================

class CIMRotationParser:
    """
    Parser for CIMRotationVisualVariable from renderer.
    
    Extracts rotation information used to rotate symbols based on
    attribute values (e.g., azimuth, direction).
    """
    
    @staticmethod
    def parse_rotation_variables(
        visual_variables: List[Dict[str, Any]]
    ) -> Optional[RotationInfo]:
        """
        Parse rotation information from visualVariables array.
        
        Looks for CIMRotationVisualVariable and extracts Z-axis rotation
        information (2D map rotation).
        
        Args:
            visual_variables: List of visual variable dictionaries from renderer
            
        Returns:
            RotationInfo object if rotation variable found, else None
        """
        if not visual_variables:
            return None
        
        for var in visual_variables:
            if var.get('type') == 'CIMRotationVisualVariable':
                return CIMRotationParser._parse_rotation_variable(var)
        
        return None
    
    @staticmethod
    def _parse_rotation_variable(
        rotation_var: Dict[str, Any]
    ) -> Optional[RotationInfo]:
        """
        Parse a single CIMRotationVisualVariable.
        
        Focuses on Z-axis (2D map rotation).
        
        Args:
            rotation_var: CIMRotationVisualVariable dictionary
            
        Returns:
            RotationInfo object or None if parsing fails
        """
        try:
            # Extract Z-axis rotation info (2D rotation)
            z_info = rotation_var.get('visualVariableInfoZ', {})
            
            if not z_info:
                logger.debug("No visualVariableInfoZ found in rotation variable")
                return None
            
            # Extract expression
            expression = z_info.get('expression', '')
            field_name = CIMRotationParser._extract_field_name(expression)
            
            # Extract variable type
            variable_type = z_info.get('visualVariableInfoType', 'None')
            
            # Extract rotation type (Geographic vs Arithmetic)
            rotation_type = rotation_var.get('rotationTypeZ')
            
            # Extract value range if present
            min_value = z_info.get('minValue')
            max_value = z_info.get('maxValue')
            
            # If no min/max, check for randomMax (used for random rotation)
            if max_value is None:
                max_value = z_info.get('randomMax')
            
            return RotationInfo(
                field_name=field_name,
                expression=expression,
                rotation_type=rotation_type,
                variable_type=variable_type,
                min_value=min_value,
                max_value=max_value,
                raw_rotation_variable=rotation_var
            )
            
        except Exception as e:
            logger.warning(f"Error parsing rotation variable: {e}")
            return None
    
    @staticmethod
    def _extract_field_name(expression: str) -> Optional[str]:
        """
        Extract field name from rotation expression.
        
        Args:
            expression: Rotation expression (e.g., "[AZIMUTH]", "AZIMUTH")
            
        Returns:
            Field name without brackets, or None
        """
        import re
        
        if not expression:
            return None
        
        # Remove brackets if present
        expr = expression.strip()
        if expr.startswith('[') and expr.endswith(']'):
            return expr[1:-1]
        
        # Find fields in brackets (for complex expressions)
        matches = re.findall(r'\[([^\]]+)\]', expr)
        if matches:
            return matches[0]  # Return first field
        
        # If no brackets, assume it's a direct field name
        if expr and expr.replace('_', '').isalnum():
            return expr
        
        return None


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def format_rotation_for_mapserver(rotation_info: RotationInfo) -> str:
    """
    Generate MapServer rotation configuration from RotationInfo.
    
    Args:
        rotation_info: RotationInfo object
        
    Returns:
        MapServer ANGLE configuration string
    
    Example:
        >>> rotation = RotationInfo(field_name="AZIMUTH", rotation_type="Geographic")
        >>> format_rotation_for_mapserver(rotation)
        'ANGLE [AZIMUTH]  # Geographic rotation'
    """
    if not rotation_info or not rotation_info.field_name:
        return ""
    
    field = rotation_info.field_name
    
    # MapServer ANGLE syntax
    config = f"ANGLE [{field}]"
    
    # Add comment about rotation type
    if rotation_info.rotation_type:
        config += f"  # {rotation_info.rotation_type} rotation"
    
    return config


# =============================================================================
# DISPLAY HELPER
# =============================================================================

class RotationInfoDisplayer:
    """Display helper for rotation information using rich."""
    
    @staticmethod
    def display_rotation_info(rotation_info: RotationInfo, console):
        """
        Display rotation information.
        
        Args:
            rotation_info: RotationInfo object to display
            console: Rich console object
        """
        from rich.table import Table
        from rich.panel import Panel
        
        if not rotation_info:
            return
        
        table = Table(title="Symbol Rotation", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        # Field
        field_name = rotation_info.get_simple_field_name()
        if field_name:
            table.add_row("Field", field_name)
        
        # Expression (if different from field)
        if rotation_info.expression and rotation_info.expression != f"[{field_name}]":
            table.add_row("Expression", rotation_info.expression)
        
        # Rotation type
        if rotation_info.rotation_type:
            rotation_desc = rotation_info.rotation_type
            if rotation_info.rotation_type == "Geographic":
                rotation_desc += " (0° = North)"
            elif rotation_info.rotation_type == "Arithmetic":
                rotation_desc += " (0° = East)"
            table.add_row("Type", rotation_desc)
        
        # Variable type
        if rotation_info.variable_type:
            var_type_desc = rotation_info.variable_type
            if rotation_info.variable_type == "Expression":
                var_type_desc += " (field-based)"
            elif rotation_info.variable_type == "Random":
                var_type_desc += " (random rotation)"
            table.add_row("Variable Type", var_type_desc)
        
        # Value range
        if rotation_info.min_value is not None or rotation_info.max_value is not None:
            min_val = rotation_info.min_value or 0
            max_val = rotation_info.max_value or 360
            table.add_row("Range", f"{min_val}° - {max_val}°")
        
        console.print(Panel(table))


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

def integration_instructions():
    """Print integration instructions."""
    return """
# =============================================================================
# INTEGRATION STEPS
# =============================================================================

## Step 1: Add RotationInfo to LayerClassification

In esri_classification_extractor.py, modify LayerClassification dataclass:

```python
@dataclass
class LayerClassification:
    # ... existing fields ...
    
    label_classes: Optional[List[LabelInfo]] = None
    
    # NEW: Symbol rotation information
    rotation_info: Optional[RotationInfo] = None
    
    raw_renderer: Dict[str, Any] = field(default_factory=dict)
```

## Step 2: Import the parser

At the top of esri_classification_extractor.py:

```python
from gcover.publish.rotation_extractor_extension import (
    RotationInfo,
    CIMRotationParser,
    RotationInfoDisplayer,
    format_rotation_for_mapserver
)
```

## Step 3: Parse rotation in _parse_unique_value_renderer

In the _parse_unique_value_renderer method, add after creating classification:

```python
def _parse_unique_value_renderer(
    self, renderer: Dict[str, Any], layer_name: str = None, layer_path: str = None
) -> Optional[LayerClassification]:
    try:
        # ... existing code to create classification ...
        
        classification = LayerClassification(
            renderer_type="CIMUniqueValueRenderer",
            fields=fields,
            classes=classes,
            # ... other fields ...
        )
        
        # NEW: Extract rotation information
        visual_variables = renderer.get('visualVariables', [])
        if visual_variables:
            rotation_info = CIMRotationParser.parse_rotation_variables(visual_variables)
            if rotation_info:
                classification.rotation_info = rotation_info
                logger.info(
                    f"Extracted rotation on field {rotation_info.get_simple_field_name()} "
                    f"({rotation_info.rotation_type})"
                )
        
        return classification
        
    except Exception as e:
        logger.error(f"Error parsing unique value renderer: {e}")
        return None
```

## Step 4: Display rotation (optional)

In ClassificationDisplayer.display_classification(), add:

```python
@staticmethod
def display_classification(classification: LayerClassification):
    # ... existing display code ...
    
    # Display rotation info
    if classification.rotation_info:
        console.print("\\n[bold yellow]Symbol Rotation:[/bold yellow]")
        RotationInfoDisplayer.display_rotation_info(
            classification.rotation_info,
            console
        )
```

## Step 5: Export to MapServer

When generating MapServer configuration:

```python
# In your MapServer export function
if classification.rotation_info:
    rotation_config = format_rotation_for_mapserver(classification.rotation_info)
    # Add to your LAYER or CLASS block
    mapfile += f"  {rotation_config}\\n"
```

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

## Example 1: Extract and display rotation

```python
from gcover.publish.esri_classification_extractor import ESRIClassificationExtractor

extractor = ESRIClassificationExtractor(use_arcpy=False)
classifications = extractor.extract_from_lyrx('your_layer.lyrx')

for classification in classifications:
    if classification.rotation_info:
        print(f"Layer: {classification.layer_name}")
        print(f"  Rotation field: {classification.rotation_info.get_simple_field_name()}")
        print(f"  Rotation type: {classification.rotation_info.rotation_type}")
```

## Example 2: Generate MapServer configuration

```python
for classification in classifications:
    print(f"LAYER")
    print(f"  NAME \\"{classification.layer_name}\\"")
    
    # Add rotation if present
    if classification.rotation_info:
        field = classification.rotation_info.get_simple_field_name()
        print(f"  ANGLE [{field}]")
    
    print(f"END")
```

## Example 3: Export to JSON

```python
import json

export_data = {
    'layer_name': classification.layer_name,
    'rotation': {
        'field': classification.rotation_info.get_simple_field_name(),
        'type': classification.rotation_info.rotation_type,
        'expression': classification.rotation_info.expression
    } if classification.rotation_info else None
}

with open('config.json', 'w') as f:
    json.dump(export_data, f, indent=2)
```
"""


if __name__ == "__main__":
    print(integration_instructions())
