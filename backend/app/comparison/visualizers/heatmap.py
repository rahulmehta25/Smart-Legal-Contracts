"""
Change heatmap visualizer for document comparison.
Creates visual heatmaps showing density and intensity of changes.
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..diff_engine import DiffResult, DiffType, ComparisonResult
from ..semantic_comparison import SemanticDiff
from ..legal_change_detector import LegalRisk, LegalRiskLevel


@dataclass
class HeatmapCell:
    """Represents a cell in the change heatmap."""
    row: int
    col: int
    intensity: float
    change_count: int
    change_types: List[str]
    content_preview: str
    tooltip: str


class ChangeHeatmapVisualizer:
    """
    Creates heatmap visualizations of document changes.
    Shows change density, intensity, and patterns across the document.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (20, 50)):
        """
        Initialize heatmap visualizer.
        
        Args:
            grid_size: Tuple of (rows, columns) for the heatmap grid
        """
        self.grid_rows, self.grid_cols = grid_size
    
    def create_change_heatmap(self, old_content: str, new_content: str,
                            comparison_result: ComparisonResult,
                            semantic_diffs: List[SemanticDiff] = None,
                            legal_risks: List[LegalRisk] = None) -> Dict[str, Any]:
        """
        Create change density heatmap.
        
        Args:
            old_content: Original document content
            new_content: Modified document content
            comparison_result: Comparison results
            semantic_diffs: Optional semantic analysis
            legal_risks: Optional legal risk analysis
            
        Returns:
            Heatmap data and visualization
        """
        # Calculate document grid mapping
        content_length = len(new_content)
        char_per_cell = max(1, content_length // (self.grid_rows * self.grid_cols))
        
        # Initialize heatmap grid
        heatmap_data = np.zeros((self.grid_rows, self.grid_cols))
        cell_details = {}
        
        # Process surface-level changes
        change_weights = {
            DiffType.INSERTION: 1.0,
            DiffType.DELETION: 1.2,
            DiffType.MODIFICATION: 1.5,
            DiffType.MOVE: 0.8
        }
        
        for diff in comparison_result.differences:
            weight = change_weights.get(diff.diff_type, 1.0)
            self._add_change_to_heatmap(
                heatmap_data, cell_details, diff.new_position, weight,
                diff.diff_type.value, diff.new_content, char_per_cell
            )
        
        # Process semantic changes with higher weights
        if semantic_diffs:
            for semantic_diff in semantic_diffs:
                weight = 2.0 - semantic_diff.semantic_similarity  # Higher weight for more semantic change
                self._add_change_to_heatmap(
                    heatmap_data, cell_details, semantic_diff.position, weight,
                    f"semantic_{semantic_diff.change_type.value}", 
                    semantic_diff.new_segment[:50], char_per_cell
                )
        
        # Process legal risks with highest weights
        if legal_risks:
            risk_weights = {
                LegalRiskLevel.CRITICAL: 5.0,
                LegalRiskLevel.HIGH: 3.0,
                LegalRiskLevel.MEDIUM: 2.0,
                LegalRiskLevel.LOW: 1.0,
                LegalRiskLevel.INFORMATIONAL: 0.5
            }
            
            for legal_risk in legal_risks:
                weight = risk_weights.get(legal_risk.risk_level, 2.0)
                self._add_change_to_heatmap(
                    heatmap_data, cell_details, legal_risk.position, weight,
                    f"legal_{legal_risk.risk_level.value}", 
                    legal_risk.new_language[:50], char_per_cell
                )
        
        # Normalize heatmap data
        max_intensity = np.max(heatmap_data)
        if max_intensity > 0:
            heatmap_data = heatmap_data / max_intensity
        
        # Generate visualizations
        html_heatmap = self._generate_html_heatmap(heatmap_data, cell_details)
        svg_heatmap = self._generate_svg_heatmap(heatmap_data, cell_details)
        
        return {
            'heatmap_data': heatmap_data.tolist(),
            'cell_details': cell_details,
            'html_visualization': html_heatmap,
            'svg_visualization': svg_heatmap,
            'statistics': {
                'max_intensity': float(max_intensity),
                'total_changes': len(comparison_result.differences),
                'high_intensity_cells': int(np.sum(heatmap_data > 0.7)),
                'medium_intensity_cells': int(np.sum((heatmap_data > 0.3) & (heatmap_data <= 0.7))),
                'low_intensity_cells': int(np.sum((heatmap_data > 0) & (heatmap_data <= 0.3)))
            }
        }
    
    def create_section_heatmap(self, content: str, comparison_result: ComparisonResult,
                             section_boundaries: List[Tuple[int, int, str]] = None) -> Dict[str, Any]:
        """
        Create section-based heatmap showing changes by document sections.
        
        Args:
            content: Document content
            comparison_result: Comparison results
            section_boundaries: Optional list of (start, end, title) tuples for sections
            
        Returns:
            Section-based heatmap data
        """
        if not section_boundaries:
            # Auto-detect sections based on paragraph breaks
            section_boundaries = self._detect_sections(content)
        
        section_heatmap = []
        
        for start, end, title in section_boundaries:
            section_changes = []
            section_intensity = 0.0
            
            # Find changes within this section
            for diff in comparison_result.differences:
                if self._position_overlaps_range(diff.new_position, (start, end)):
                    section_changes.append(diff)
                    # Weight by change type
                    if diff.diff_type == DiffType.DELETION:
                        section_intensity += 1.5
                    elif diff.diff_type == DiffType.MODIFICATION:
                        section_intensity += 1.2
                    else:
                        section_intensity += 1.0
            
            section_heatmap.append({
                'title': title,
                'start_pos': start,
                'end_pos': end,
                'change_count': len(section_changes),
                'intensity': section_intensity,
                'changes': [self._diff_to_dict(diff) for diff in section_changes]
            })
        
        # Normalize intensities
        max_intensity = max((section['intensity'] for section in section_heatmap), default=1.0)
        for section in section_heatmap:
            section['normalized_intensity'] = section['intensity'] / max_intensity
        
        # Generate visualization
        html_section_heatmap = self._generate_html_section_heatmap(section_heatmap)
        
        return {
            'section_heatmap': section_heatmap,
            'html_visualization': html_section_heatmap,
            'statistics': {
                'total_sections': len(section_heatmap),
                'sections_with_changes': len([s for s in section_heatmap if s['change_count'] > 0]),
                'max_section_intensity': max_intensity,
                'avg_changes_per_section': sum(s['change_count'] for s in section_heatmap) / len(section_heatmap) if section_heatmap else 0
            }
        }
    
    def _add_change_to_heatmap(self, heatmap_data: np.ndarray, cell_details: Dict,
                              position: Tuple[int, int], weight: float,
                              change_type: str, content_preview: str, char_per_cell: int):
        """Add a change to the heatmap grid."""
        start_pos = position[0]
        cell_index = start_pos // char_per_cell
        
        # Convert to row, col coordinates
        row = cell_index // self.grid_cols
        col = cell_index % self.grid_cols
        
        # Ensure within bounds
        row = min(row, self.grid_rows - 1)
        col = min(col, self.grid_cols - 1)
        
        # Add intensity
        heatmap_data[row, col] += weight
        
        # Track cell details
        cell_key = f"{row}_{col}"
        if cell_key not in cell_details:
            cell_details[cell_key] = {
                'row': row,
                'col': col,
                'changes': [],
                'total_intensity': 0.0
            }
        
        cell_details[cell_key]['changes'].append({
            'type': change_type,
            'weight': weight,
            'preview': content_preview
        })
        cell_details[cell_key]['total_intensity'] += weight
    
    def _detect_sections(self, content: str) -> List[Tuple[int, int, str]]:
        """Auto-detect sections in the document."""
        sections = []
        lines = content.split('\n')
        current_pos = 0
        section_start = 0
        
        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            
            # Check if line looks like a header
            if self._is_header_line(line):
                if i > 0:  # Not the first line
                    # End previous section
                    sections.append((section_start, current_pos, f"Section {len(sections) + 1}"))
                    section_start = current_pos
            
            current_pos += line_len
        
        # Add final section
        if section_start < len(content):
            sections.append((section_start, len(content), f"Section {len(sections) + 1}"))
        
        return sections
    
    def _is_header_line(self, line: str) -> bool:
        """Determine if a line is likely a header."""
        line = line.strip()
        if not line:
            return False
        
        # Check for common header patterns
        header_patterns = [
            line.startswith('#'),  # Markdown
            line.isupper() and len(line) < 100,  # All caps
            line.endswith(':') and len(line) < 100,  # Colon ending
            bool(re.match(r'^\d+\.', line)),  # Numbered
            bool(re.match(r'^[A-Z][^.]+$', line)) and len(line) < 100  # Title case
        ]
        
        return any(header_patterns)
    
    def _position_overlaps_range(self, position: Tuple[int, int], range_bounds: Tuple[int, int]) -> bool:
        """Check if a position overlaps with a range."""
        return not (position[1] <= range_bounds[0] or position[0] >= range_bounds[1])
    
    def _diff_to_dict(self, diff: DiffResult) -> Dict[str, Any]:
        """Convert diff result to dictionary."""
        return {
            'type': diff.diff_type.value,
            'old_content': diff.old_content[:100],
            'new_content': diff.new_content[:100],
            'confidence': diff.confidence,
            'position': diff.new_position
        }
    
    def _generate_html_heatmap(self, heatmap_data: np.ndarray, cell_details: Dict) -> str:
        """Generate HTML visualization of the heatmap."""
        cell_size = 20
        width = self.grid_cols * cell_size
        height = self.grid_rows * cell_size
        
        # Generate heatmap cells
        cells_html = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                intensity = heatmap_data[row, col]
                cell_key = f"{row}_{col}"
                
                # Color mapping based on intensity
                if intensity == 0:
                    color = "#f8f9fa"
                elif intensity < 0.3:
                    color = "#fff3cd"  # Light yellow
                elif intensity < 0.7:
                    color = "#ffc107"  # Yellow
                else:
                    color = "#dc3545"  # Red
                
                # Tooltip
                tooltip = f"Row {row}, Col {col}\\nIntensity: {intensity:.2f}"
                if cell_key in cell_details:
                    details = cell_details[cell_key]
                    tooltip += f"\\nChanges: {len(details['changes'])}"
                
                x = col * cell_size
                y = row * cell_size
                
                cells_html.append(f"""
                    <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" 
                          fill="{color}" stroke="#ddd" stroke-width="0.5"
                          title="{tooltip}" class="heatmap-cell" 
                          data-intensity="{intensity:.3f}" data-row="{row}" data-col="{col}">
                    </rect>
                """)
        
        # Generate legend
        legend_html = """
        <div class="heatmap-legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f8f9fa;"></div>
                <span>No changes</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #fff3cd;"></div>
                <span>Low intensity</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffc107;"></div>
                <span>Medium intensity</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #dc3545;"></div>
                <span>High intensity</span>
            </div>
        </div>
        """
        
        css_styles = """
        <style>
            .heatmap-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .heatmap-title {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
                color: #2c3e50;
            }
            .heatmap-svg {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            .heatmap-cell:hover {
                stroke: #007bff;
                stroke-width: 2;
                cursor: pointer;
            }
            .heatmap-legend {
                display: flex;
                gap: 20px;
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .legend-color {
                width: 20px;
                height: 16px;
                border: 1px solid #ddd;
                border-radius: 2px;
            }
        </style>
        """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Change Heatmap</title>
            {css_styles}
        </head>
        <body>
            <div class="heatmap-container">
                <div class="heatmap-title">Document Change Heatmap</div>
                <svg class="heatmap-svg" width="{width}" height="{height}">
                    {''.join(cells_html)}
                </svg>
                {legend_html}
            </div>
        </body>
        </html>
        """
    
    def _generate_svg_heatmap(self, heatmap_data: np.ndarray, cell_details: Dict) -> str:
        """Generate standalone SVG heatmap."""
        cell_size = 20
        width = self.grid_cols * cell_size
        height = self.grid_rows * cell_size
        
        cells_svg = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                intensity = heatmap_data[row, col]
                
                # Color mapping
                if intensity == 0:
                    color = "#f8f9fa"
                elif intensity < 0.3:
                    color = "#fff3cd"
                elif intensity < 0.7:
                    color = "#ffc107" 
                else:
                    color = "#dc3545"
                
                x = col * cell_size
                y = row * cell_size
                
                cells_svg.append(f"""
                    <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" 
                          fill="{color}" stroke="#ddd" stroke-width="0.5"/>
                """)
        
        return f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            {''.join(cells_svg)}
        </svg>
        """
    
    def _generate_html_section_heatmap(self, section_heatmap: List[Dict]) -> str:
        """Generate HTML for section-based heatmap."""
        sections_html = []
        
        for section in section_heatmap:
            intensity = section['normalized_intensity']
            
            # Color based on intensity
            if intensity == 0:
                bar_color = "#f8f9fa"
            elif intensity < 0.3:
                bar_color = "#fff3cd"
            elif intensity < 0.7:
                bar_color = "#ffc107"
            else:
                bar_color = "#dc3545"
            
            bar_width = max(10, intensity * 100)  # Percentage width
            
            sections_html.append(f"""
            <div class="section-row">
                <div class="section-title">{section['title']}</div>
                <div class="section-bar-container">
                    <div class="section-bar" style="width: {bar_width}%; background-color: {bar_color};"></div>
                </div>
                <div class="section-stats">
                    <span class="change-count">{section['change_count']} changes</span>
                    <span class="intensity-score">Intensity: {intensity:.2f}</span>
                </div>
            </div>
            """)
        
        css_styles = """
        <style>
            .section-heatmap {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 20px;
                background: white;
                border-radius: 8px;
            }
            .section-heatmap-title {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 20px;
                color: #2c3e50;
            }
            .section-row {
                display: flex;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid #ecf0f1;
            }
            .section-title {
                flex: 0 0 200px;
                font-weight: 500;
                color: #2c3e50;
            }
            .section-bar-container {
                flex: 1;
                height: 20px;
                background: #ecf0f1;
                border-radius: 10px;
                margin: 0 15px;
                position: relative;
            }
            .section-bar {
                height: 100%;
                border-radius: 10px;
                transition: width 0.3s ease;
            }
            .section-stats {
                flex: 0 0 150px;
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                font-size: 12px;
                color: #6c757d;
            }
        </style>
        """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Section Change Heatmap</title>
            {css_styles}
        </head>
        <body>
            <div class="section-heatmap">
                <div class="section-heatmap-title">Changes by Document Section</div>
                {''.join(sections_html)}
            </div>
        </body>
        </html>
        """


# Utility functions
def create_change_heatmap(old_content: str, new_content: str,
                         comparison_result: ComparisonResult,
                         grid_size: Tuple[int, int] = (20, 50)) -> Dict[str, Any]:
    """
    Quick utility to create a change heatmap.
    
    Args:
        old_content: Original document
        new_content: Modified document
        comparison_result: Comparison results
        grid_size: Heatmap grid dimensions
        
    Returns:
        Heatmap visualization data
    """
    visualizer = ChangeHeatmapVisualizer(grid_size)
    return visualizer.create_change_heatmap(old_content, new_content, comparison_result)