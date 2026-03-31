#!/usr/bin/env python
"""
Manual Mapfile Handler

Extract symbol information from manually-created mapfiles to:
1. Include symbols in the global symbols.sym catalog
2. Validate expected symbols are present
3. Track manual mapfile dependencies
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class MapfileSymbolUsage:
    """Track symbol usage in a manual mapfile"""
    
    mapfile_path: Path
    layer_name: str
    symbols_used: Set[str] = field(default_factory=set)
    fonts_used: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        return {
            "mapfile_path": str(self.mapfile_path),
            "layer_name": self.layer_name,
            "symbols_used": sorted(self.symbols_used),
            "fonts_used": sorted(self.fonts_used),
        }


class ManualMapfileHandler:
    """
    Extract and manage symbols from manually-created mapfiles.
    
    Use cases:
    - Extract SYMBOL names to include in symbols.sym catalog
    - Validate expected symbols are present
    - Track font dependencies
    - Generate reports on manual mapfile usage
    """
    
    def __init__(self):
        """Initialize handler"""
        self.manual_mapfiles: Dict[str, MapfileSymbolUsage] = {}
    
    def extract_symbols_from_mapfile(self, mapfile_path: Path, layer_name: str = None) -> MapfileSymbolUsage:
        """
        Parse manual mapfile and extract symbol and font references.
        
        Args:
            mapfile_path: Path to mapfile (.map)
            layer_name: Optional layer name for tracking
            
        Returns:
            MapfileSymbolUsage with extracted symbols and fonts
        """
        if not mapfile_path.exists():
            logger.error(f"Mapfile not found: {mapfile_path}")
            return MapfileSymbolUsage(mapfile_path, layer_name or mapfile_path.stem)
        
        logger.info(f"Extracting symbols from {mapfile_path}")
        
        with open(mapfile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract SYMBOL "name" references from STYLE blocks
        # Pattern: SYMBOL "symbol_name"
        symbol_pattern = re.compile(r'SYMBOL\s+"([^"]+)"', re.IGNORECASE)
        symbols = set(symbol_pattern.findall(content))
        
        # Extract FONT "name" references
        # Pattern: FONT "font_name"
        font_pattern = re.compile(r'FONT\s+"([^"]+)"', re.IGNORECASE)
        fonts = set(font_pattern.findall(content))
        
        # Extract layer name if not provided
        if not layer_name:
            layer_match = re.search(r'NAME\s+"([^"]+)"', content, re.IGNORECASE)
            layer_name = layer_match.group(1) if layer_match else mapfile_path.stem
        
        usage = MapfileSymbolUsage(
            mapfile_path=mapfile_path,
            layer_name=layer_name,
            symbols_used=symbols,
            fonts_used=fonts,
        )
        
        self.manual_mapfiles[str(mapfile_path)] = usage
        
        logger.debug(f"  Found {len(symbols)} symbols: {', '.join(sorted(symbols))}")
        logger.debug(f"  Found {len(fonts)} fonts: {', '.join(sorted(fonts))}")
        
        return usage
    
    def validate_expected_symbols(
        self, 
        mapfile_path: Path, 
        expected_symbols: List[str]
    ) -> bool:
        """
        Validate that manual mapfile contains expected symbols.
        
        Args:
            mapfile_path: Path to mapfile
            expected_symbols: List of expected symbol names
            
        Returns:
            True if all expected symbols are present
        """
        usage = self.extract_symbols_from_mapfile(mapfile_path)
        
        missing_symbols = set(expected_symbols) - usage.symbols_used
        unexpected_symbols = usage.symbols_used - set(expected_symbols)
        
        if missing_symbols:
            logger.warning(f"Missing expected symbols in {mapfile_path.name}:")
            for sym in sorted(missing_symbols):
                logger.warning(f"  - {sym}")
        
        if unexpected_symbols:
            logger.info(f"Unexpected symbols in {mapfile_path.name}:")
            for sym in sorted(unexpected_symbols):
                logger.info(f"  + {sym}")
        
        return len(missing_symbols) == 0
    
    def merge_manual_symbols(self) -> Set[str]:
        """
        Get all unique symbols used across all manual mapfiles.
        
        Returns:
            Set of symbol names to include in symbols.sym
        """
        all_symbols = set()
        for usage in self.manual_mapfiles.values():
            all_symbols.update(usage.symbols_used)
        return all_symbols
    
    def merge_manual_fonts(self) -> Set[str]:
        """
        Get all unique fonts used across all manual mapfiles.
        
        Returns:
            Set of font names to include in fonts.txt
        """
        all_fonts = set()
        for usage in self.manual_mapfiles.values():
            all_fonts.update(usage.fonts_used)
        return all_fonts
    
    def generate_report(self) -> str:
        """
        Generate summary report of manual mapfile usage.
        
        Returns:
            Formatted report string
        """
        if not self.manual_mapfiles:
            return "No manual mapfiles tracked"
        
        table = Table(title="Manual Mapfile Symbol Usage", show_header=True)
        table.add_column("Mapfile", style="cyan")
        table.add_column("Layer", style="yellow")
        table.add_column("Symbols", justify="right", style="green")
        table.add_column("Fonts", justify="right", style="blue")
        
        for usage in self.manual_mapfiles.values():
            table.add_row(
                usage.mapfile_path.name,
                usage.layer_name,
                str(len(usage.symbols_used)),
                str(len(usage.fonts_used)),
            )
        
        # Add totals
        total_symbols = len(self.merge_manual_symbols())
        total_fonts = len(self.merge_manual_fonts())
        table.add_row(
            "[bold]TOTAL (unique)[/bold]",
            "",
            f"[bold]{total_symbols}[/bold]",
            f"[bold]{total_fonts}[/bold]",
        )
        
        from io import StringIO
        output = StringIO()
        console.print(table, file=output)
        return output.getvalue()
    
    def extract_class_definitions(self, mapfile_path: Path) -> List[Dict]:
        """
        Extract CLASS definitions from manual mapfile.
        
        Useful for understanding what was manually defined.
        
        Args:
            mapfile_path: Path to mapfile
            
        Returns:
            List of class definitions with name and expression
        """
        if not mapfile_path.exists():
            logger.error(f"Mapfile not found: {mapfile_path}")
            return []
        
        with open(mapfile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        classes = []
        
        # Extract CLASS blocks
        # Pattern: CLASS ... NAME "..." ... EXPRESSION (...) ... END
        class_pattern = re.compile(
            r'CLASS\s+.*?NAME\s+"([^"]+)".*?EXPRESSION\s+([^\n]+?)(?=\s+STYLE|\s+LABEL|\s+END)',
            re.DOTALL | re.IGNORECASE
        )
        
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            expression = match.group(2).strip()
            
            classes.append({
                "name": class_name,
                "expression": expression,
            })
        
        logger.debug(f"Extracted {len(classes)} CLASS definitions from {mapfile_path.name}")
        
        return classes


# =============================================================================
# CLI for testing
# =============================================================================


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python manual_mapfile_handler.py <mapfile_path> [expected_symbols...]")
        sys.exit(1)
    
    mapfile_path = Path(sys.argv[1])
    expected_symbols = sys.argv[2:] if len(sys.argv) > 2 else []
    
    handler = ManualMapfileHandler()
    
    # Extract symbols
    usage = handler.extract_symbols_from_mapfile(mapfile_path)
    
    console.print("\n[bold green]Extraction Results:[/bold green]")
    console.print(f"  Mapfile: {usage.mapfile_path}")
    console.print(f"  Layer: {usage.layer_name}")
    console.print(f"  Symbols: {', '.join(sorted(usage.symbols_used))}")
    console.print(f"  Fonts: {', '.join(sorted(usage.fonts_used))}")
    
    # Validate if expected symbols provided
    if expected_symbols:
        console.print("\n[bold yellow]Validation:[/bold yellow]")
        is_valid = handler.validate_expected_symbols(mapfile_path, expected_symbols)
        
        if is_valid:
            console.print("[green]✓ All expected symbols present[/green]")
        else:
            console.print("[red]✗ Some expected symbols missing[/red]")
    
    # Extract class definitions
    classes = handler.extract_class_definitions(mapfile_path)
    if classes:
        console.print(f"\n[bold cyan]Found {len(classes)} CLASS definitions:[/bold cyan]")
        for cls in classes[:10]:  # Show first 10
            console.print(f"  • {cls['name']}: {cls['expression']}")
