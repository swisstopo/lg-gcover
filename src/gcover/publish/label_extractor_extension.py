#!/usr/bin/env python3
"""
Extension for ESRI Classification Extractor - Label Support

Adds label parsing capabilities to extract basic labeling information
from CIMLabelClass definitions in .lyrx files.

Usage:
    Add the LabelInfo dataclass and parsing methods to your existing
    esri_classification_extractor.py file.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger


# =============================================================================
# NEW DATACLASS: LabelInfo
# =============================================================================

@dataclass
class LabelInfo:
    """
    Basic label information extracted from CIMLabelClass.
    
    Focuses on essential properties that can be easily translated
    to other mapping systems (e.g., MapServer).
    """
    # Which field to use for labeling (e.g., "DIP", "AZIMUTH")
    field_name: Optional[str] = None
    expression: Optional[str] = None  # Full expression like "[DIP]" or "[FIELD1] & '-' & [FIELD2]"
    
    # Font properties
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    font_color: Optional['ColorInfo'] = None  # Reuse existing ColorInfo class
    
    # Additional simple properties
    font_style: Optional[str] = None  # "Regular", "Bold", "Italic", etc.
    min_scale: Optional[float] = None
    max_scale: Optional[float] = None
    
    # Visibility and conditions
    visible: bool = True
    where_clause: Optional[str] = None  # SQL filter for which features get labeled
    
    # Raw data for reference
    raw_label_class: Dict[str, Any] = field(default_factory=dict)
    
    def get_simple_field_name(self) -> Optional[str]:
        """
        Extract simple field name from expression.
        
        Examples:
            "[DIP]" -> "DIP"
            "[AZIMUTH]" -> "AZIMUTH"
            "[FIELD1] & '-' & [FIELD2]" -> "FIELD1" (first field)
        
        Returns:
            Simple field name without brackets, or None if complex
        """
        if self.field_name:
            return self.field_name
            
        if not self.expression:
            return None
            
        # Simple case: just a field in brackets
        if self.expression.startswith('[') and self.expression.endswith(']'):
            return self.expression[1:-1]
        
        # Complex expression: extract first field
        import re
        matches = re.findall(r'\[([^\]]+)\]', self.expression)
        if matches:
            return matches[0]
        
        return None


# =============================================================================
# NEW PARSER CLASS: CIMLabelParser
# =============================================================================

class CIMLabelParser:
    """
    Parser for CIMLabelClass definitions.
    
    Extracts basic labeling properties that are commonly supported
    across different mapping systems.
    """
    
    @staticmethod
    def parse_label_classes(label_classes: List[Dict[str, Any]]) -> List[LabelInfo]:
        """
        Parse multiple label classes from a layer.
        
        Args:
            label_classes: List of CIMLabelClass dictionaries
            
        Returns:
            List of LabelInfo objects
        """
        results = []
        
        for label_class in label_classes:
            label_info = CIMLabelParser.parse_label_class(label_class)
            if label_info:
                results.append(label_info)
        
        return results
    
    @staticmethod
    def parse_label_class(label_class: Dict[str, Any]) -> Optional[LabelInfo]:
        """
        Parse a single CIMLabelClass.
        
        Args:
            label_class: CIMLabelClass dictionary
            
        Returns:
            LabelInfo object or None if parsing fails
        """
        try:
            # Import ColorInfo from the main module
            from gcover.publish.esri_classification_extractor import CIMColorParser
            
            # Extract basic properties
            expression = label_class.get('expression', '')
            visible = label_class.get('visibility', True)
            where_clause = label_class.get('whereClause')
            
            # Extract field name from expression
            field_name = CIMLabelParser._extract_field_name(expression)
            
            # Extract text symbol
            text_symbol_ref = label_class.get('textSymbol', {})
            text_symbol = text_symbol_ref.get('symbol', {})
            
            # Extract font properties
            font_size = text_symbol.get('height')
            font_family = text_symbol.get('fontFamilyName')
            font_style = text_symbol.get('fontStyleName', 'Regular')
            
            # Extract color from the nested symbol structure
            # Path: textSymbol -> symbol -> symbol -> symbolLayers -> color
            font_color = CIMLabelParser._extract_label_color(text_symbol)
            
            # Extract scale range (from Maplex properties if available)
            min_scale = None
            max_scale = None
            
            maplex_props = label_class.get('maplexLabelPlacementProperties', {})
            if maplex_props:
                # Maplex doesn't store scale directly in label class
                # Scale is typically at layer level
                pass
            
            return LabelInfo(
                field_name=field_name,
                expression=expression,
                font_size=font_size,
                font_family=font_family,
                font_color=font_color,
                font_style=font_style,
                visible=visible,
                where_clause=where_clause,
                min_scale=min_scale,
                max_scale=max_scale,
                raw_label_class=label_class
            )
            
        except Exception as e:
            logger.warning(f"Error parsing label class: {e}")
            return None
    
    @staticmethod
    def _extract_field_name(expression: str) -> Optional[str]:
        """
        Extract field name from label expression.
        
        Args:
            expression: Label expression (e.g., "[DIP]", "[FIELD1] & [FIELD2]")
            
        Returns:
            First field name found, or None
        """
        import re
        
        if not expression:
            return None
        
        # Find all fields in brackets
        matches = re.findall(r'\[([^\]]+)\]', expression)
        
        if matches:
            return matches[0]  # Return first field
        
        return None
    
    @staticmethod
    def _extract_label_color(text_symbol: Dict[str, Any]) -> Optional['ColorInfo']:
        """
        Extract color from text symbol structure.
        
        The color is typically at:
        textSymbol -> symbol -> symbol -> symbolLayers -> color
        
        Args:
            text_symbol: CIMTextSymbol dictionary
            
        Returns:
            ColorInfo object or None
        """
        try:
            from gcover.publish.esri_classification_extractor import CIMColorParser
            
            # Navigate to the nested symbol (polygon symbol that draws the text)
            nested_symbol = text_symbol.get('symbol', {})
            
            if not nested_symbol:
                return None
            
            # Look for symbol layers
            symbol_layers = nested_symbol.get('symbolLayers', [])
            
            for layer in symbol_layers:
                layer_type = layer.get('type', '')
                
                # Text is usually drawn with a solid fill
                if layer_type == 'CIMSolidFill':
                    color_obj = layer.get('color')
                    if color_obj:
                        return CIMColorParser.parse_color(color_obj)
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract label color: {e}")
            return None


# =============================================================================
# EXTENSION TO ClassificationClass
# =============================================================================

# Add this field to the existing ClassificationClass dataclass:
#
# @dataclass
# class ClassificationClass:
#     # ... existing fields ...
#     
#     # NEW: Label information
#     label_info: Optional[List[LabelInfo]] = None  # Multiple label classes possible


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

def integrate_labels_into_extractor():
    """
    Example showing how to integrate label parsing into the main extractor.
    
    Add this method to ESRIClassificationExtractor class:
    """
    example_code = '''
    def _extract_layer_with_labels(
        self, layer_data: Dict[str, Any]
    ) -> Optional[LayerClassification]:
        """
        Extract classification and label information from a layer.
        
        This extends the existing _extract_from_json method.
        """
        # First, extract classification as usual
        classification = self._extract_layer_properties(layer_data)
        
        if not classification:
            return None
        
        # NEW: Extract label classes
        label_classes = layer_data.get('labelClasses', [])
        
        if label_classes:
            label_infos = CIMLabelParser.parse_label_classes(label_classes)
            
            # Option 1: Add to LayerClassification
            classification.label_classes = label_infos
            
            # Option 2: Add to each ClassificationClass if needed
            # This would require matching labels to specific classes
            # based on SQL expressions or other criteria
        
        return classification
    '''
    
    print(example_code)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """
    Example of how to use the label extractor.
    """
    example = '''
    # In your main extraction code:
    
    from esri_classification_extractor import ESRIClassificationExtractor
    from label_extractor_extension import CIMLabelParser, LabelInfo
    
    # Load your .lyrx file
    extractor = ESRIClassificationExtractor(use_arcpy=False)
    lyrx_data = extractor._load_lyrx_json('your_layer.lyrx')
    
    # Find layer definitions
    layer_definitions = lyrx_data.get('layerDefinitions', [])
    
    for layer_def in layer_definitions:
        # Extract label classes
        label_classes = layer_def.get('labelClasses', [])
        
        if label_classes:
            label_infos = CIMLabelParser.parse_label_classes(label_classes)
            
            print(f"Found {len(label_infos)} label classes")
            
            for label_info in label_infos:
                print(f"  Field: {label_info.get_simple_field_name()}")
                print(f"  Font: {label_info.font_family} {label_info.font_size}pt")
                
                if label_info.font_color:
                    print(f"  Color: {label_info.font_color.to_hex()}")
                
                if label_info.where_clause:
                    print(f"  Filter: {label_info.where_clause}")
    '''
    
    print(example)


# =============================================================================
# DISPLAY HELPER
# =============================================================================

class LabelInfoDisplayer:
    """
    Display helper for label information using rich.
    """
    
    @staticmethod
    def display_label_info(label_info: LabelInfo, console):
        """
        Display a single label info object.
        
        Args:
            label_info: LabelInfo object to display
            console: Rich console object
        """
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(title="Label Configuration", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        # Field
        field_name = label_info.get_simple_field_name()
        table.add_row("Field", field_name or "N/A")
        
        # Expression (if complex)
        if label_info.expression and label_info.expression != f"[{field_name}]":
            table.add_row("Expression", label_info.expression)
        
        # Font
        if label_info.font_family:
            font_desc = f"{label_info.font_family}"
            if label_info.font_style and label_info.font_style != "Regular":
                font_desc += f" {label_info.font_style}"
            table.add_row("Font", font_desc)
        
        if label_info.font_size:
            table.add_row("Size", f"{label_info.font_size} pt")
        
        # Color
        if label_info.font_color:
            color_hex = label_info.font_color.to_hex()
            color_rgb = label_info.font_color.to_rgb_tuple()
            table.add_row(
                "Color",
                f"{color_hex} (RGB: {color_rgb})"
            )
        
        # Filter
        if label_info.where_clause:
            table.add_row("Filter", label_info.where_clause)
        
        # Visibility
        visibility = "✓ Visible" if label_info.visible else "✗ Hidden"
        table.add_row("Status", visibility)
        
        console.print(Panel(table))
    
    @staticmethod
    def display_label_infos(label_infos: List[LabelInfo], console):
        """
        Display multiple label infos.
        
        Args:
            label_infos: List of LabelInfo objects
            console: Rich console object
        """
        from rich.tree import Tree
        
        tree = Tree("🏷️  [bold]Label Classes[/bold]")
        
        for i, label_info in enumerate(label_infos):
            field_name = label_info.get_simple_field_name() or "Unknown"
            
            # Build label description
            desc_parts = [f"[cyan]{field_name}[/cyan]"]
            
            if label_info.font_size:
                desc_parts.append(f"{label_info.font_size}pt")
            
            if label_info.font_color:
                desc_parts.append(label_info.font_color.to_hex())
            
            if not label_info.visible:
                desc_parts.append("[dim](hidden)[/dim]")
            
            node = tree.add(" ".join(desc_parts))
            
            # Add filter if present
            if label_info.where_clause:
                node.add(f"[yellow]Filter:[/yellow] {label_info.where_clause}")
        
        console.print(tree)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("INTEGRATION INSTRUCTIONS")
    print("="*70)
    
    print("""
    1. Add LabelInfo dataclass to your esri_classification_extractor.py
    
    2. Add CIMLabelParser class to your esri_classification_extractor.py
    
    3. Extend ClassificationClass with label_info field:
       
       @dataclass
       class ClassificationClass:
           # ... existing fields ...
           label_info: Optional[List[LabelInfo]] = None
    
    4. OR extend LayerClassification with label_classes field:
       
       @dataclass
       class LayerClassification:
           # ... existing fields ...
           label_classes: Optional[List[LabelInfo]] = None
    
    5. In _extract_from_json, add label extraction:
       
       # After extracting renderer
       label_classes = layer_data.get('labelClasses', [])
       if label_classes:
           classification.label_classes = CIMLabelParser.parse_label_classes(label_classes)
    
    6. Use LabelInfoDisplayer to show label information
    """)
    
    example_usage()
