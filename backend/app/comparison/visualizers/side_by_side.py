"""
Side-by-side comparison visualizer.
Creates side-by-side views of document differences with highlighting.
"""

import html
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..diff_engine import DiffResult, DiffType, ComparisonResult
from ..semantic_comparison import SemanticDiff, SemanticChangeType
from ..legal_change_detector import LegalRisk, LegalRiskLevel


class HighlightStyle(Enum):
    """Different highlight styles for changes."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    MOVE = "move"
    SEMANTIC = "semantic"
    LEGAL_RISK = "legal_risk"


@dataclass
class HighlightedSegment:
    """A segment of text with highlighting information."""
    text: str
    style: HighlightStyle
    start_pos: int
    end_pos: int
    tooltip: str = ""
    metadata: Dict[str, Any] = None


class SideBySideVisualizer:
    """
    Creates side-by-side comparison views with rich highlighting.
    Supports HTML, plain text, and structured output formats.
    """
    
    def __init__(self):
        self.css_styles = self._generate_css_styles()
    
    def visualize_comparison(self, old_content: str, new_content: str,
                           comparison_result: ComparisonResult,
                           semantic_diffs: List[SemanticDiff] = None,
                           legal_risks: List[LegalRisk] = None,
                           format_type: str = "html") -> Dict[str, Any]:
        """
        Create side-by-side visualization of document comparison.
        
        Args:
            old_content: Original document content
            new_content: Modified document content
            comparison_result: Results from diff engine
            semantic_diffs: Optional semantic analysis results
            legal_risks: Optional legal risk analysis results
            format_type: Output format ("html", "text", "json")
            
        Returns:
            Visualization data in requested format
        """
        # Build highlighted segments for both sides
        old_segments = self._build_highlighted_segments(
            old_content, comparison_result.differences, "old",
            semantic_diffs, legal_risks
        )
        new_segments = self._build_highlighted_segments(
            new_content, comparison_result.differences, "new",
            semantic_diffs, legal_risks
        )
        
        if format_type == "html":
            return self._generate_html_output(
                old_segments, new_segments, comparison_result,
                semantic_diffs, legal_risks
            )
        elif format_type == "text":
            return self._generate_text_output(old_segments, new_segments)
        elif format_type == "json":
            return self._generate_json_output(
                old_segments, new_segments, comparison_result
            )
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _build_highlighted_segments(self, content: str, differences: List[DiffResult],
                                   side: str, semantic_diffs: List[SemanticDiff] = None,
                                   legal_risks: List[LegalRisk] = None) -> List[HighlightedSegment]:
        """Build highlighted segments for one side of the comparison."""
        segments = []
        last_pos = 0
        
        # Sort differences by position
        sorted_diffs = sorted(differences, key=lambda d: d.old_position[0])
        
        for diff in sorted_diffs:
            # Get position based on which side we're processing
            if side == "old":
                pos = diff.old_position
                diff_content = diff.old_content
            else:
                pos = diff.new_position
                diff_content = diff.new_content
            
            # Skip if this diff doesn't apply to current side
            if not diff_content and diff.diff_type in [DiffType.INSERTION, DiffType.DELETION]:
                if (side == "old" and diff.diff_type == DiffType.INSERTION) or \
                   (side == "new" and diff.diff_type == DiffType.DELETION):
                    continue
            
            # Add unchanged content before this difference
            if pos[0] > last_pos:
                unchanged_text = content[last_pos:pos[0]]
                if unchanged_text:
                    segments.append(HighlightedSegment(
                        text=unchanged_text,
                        style=HighlightStyle.ADDITION,  # No highlighting
                        start_pos=last_pos,
                        end_pos=pos[0]
                    ))
            
            # Add highlighted difference
            if diff_content:
                style = self._get_highlight_style(diff.diff_type)
                tooltip = self._generate_diff_tooltip(diff)
                
                segments.append(HighlightedSegment(
                    text=diff_content,
                    style=style,
                    start_pos=pos[0],
                    end_pos=pos[1],
                    tooltip=tooltip,
                    metadata={'diff': diff}
                ))
            
            last_pos = pos[1]
        
        # Add remaining unchanged content
        if last_pos < len(content):
            remaining_text = content[last_pos:]
            segments.append(HighlightedSegment(
                text=remaining_text,
                style=HighlightStyle.ADDITION,  # No highlighting
                start_pos=last_pos,
                end_pos=len(content)
            ))
        
        # Add semantic and legal highlighting
        if semantic_diffs:
            segments = self._add_semantic_highlighting(segments, semantic_diffs, side)
        
        if legal_risks:
            segments = self._add_legal_risk_highlighting(segments, legal_risks, side)
        
        return segments
    
    def _get_highlight_style(self, diff_type: DiffType) -> HighlightStyle:
        """Map diff type to highlight style."""
        mapping = {
            DiffType.INSERTION: HighlightStyle.ADDITION,
            DiffType.DELETION: HighlightStyle.DELETION,
            DiffType.MODIFICATION: HighlightStyle.MODIFICATION,
            DiffType.MOVE: HighlightStyle.MOVE
        }
        return mapping.get(diff_type, HighlightStyle.MODIFICATION)
    
    def _generate_diff_tooltip(self, diff: DiffResult) -> str:
        """Generate tooltip text for a difference."""
        tooltip_parts = [
            f"Type: {diff.diff_type.value}",
            f"Level: {diff.level.value}",
            f"Confidence: {diff.confidence:.2f}"
        ]
        
        if diff.context:
            tooltip_parts.append(f"Context: {diff.context[:100]}...")
        
        return " | ".join(tooltip_parts)
    
    def _add_semantic_highlighting(self, segments: List[HighlightedSegment],
                                 semantic_diffs: List[SemanticDiff],
                                 side: str) -> List[HighlightedSegment]:
        """Add semantic highlighting to segments."""
        # This would overlay semantic highlighting on existing segments
        # Implementation would merge overlapping highlights
        return segments
    
    def _add_legal_risk_highlighting(self, segments: List[HighlightedSegment],
                                   legal_risks: List[LegalRisk],
                                   side: str) -> List[HighlightedSegment]:
        """Add legal risk highlighting to segments."""
        # Similar to semantic highlighting
        return segments
    
    def _generate_html_output(self, old_segments: List[HighlightedSegment],
                            new_segments: List[HighlightedSegment],
                            comparison_result: ComparisonResult,
                            semantic_diffs: List[SemanticDiff] = None,
                            legal_risks: List[LegalRisk] = None) -> Dict[str, Any]:
        """Generate HTML output for side-by-side comparison."""
        
        def segments_to_html(segments: List[HighlightedSegment]) -> str:
            html_parts = []
            for segment in segments:
                escaped_text = html.escape(segment.text)
                
                if segment.style != HighlightStyle.ADDITION:  # No highlighting for unchanged
                    css_class = f"diff-{segment.style.value}"
                    title_attr = f' title="{html.escape(segment.tooltip)}"' if segment.tooltip else ""
                    html_parts.append(
                        f'<span class="{css_class}"{title_attr}>{escaped_text}</span>'
                    )
                else:
                    html_parts.append(escaped_text)
            
            return ''.join(html_parts)
        
        old_html = segments_to_html(old_segments)
        new_html = segments_to_html(new_segments)
        
        # Generate statistics
        stats_html = self._generate_statistics_html(comparison_result, semantic_diffs, legal_risks)
        
        # Create complete HTML document
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Comparison</title>
            <style>{self.css_styles}</style>
        </head>
        <body>
            <div class="comparison-header">
                <h1>Document Comparison</h1>
                {stats_html}
            </div>
            
            <div class="comparison-container">
                <div class="side-panel">
                    <div class="panel-header">Original Document</div>
                    <div class="content-panel">
                        <pre class="document-content">{old_html}</pre>
                    </div>
                </div>
                
                <div class="divider"></div>
                
                <div class="side-panel">
                    <div class="panel-header">Modified Document</div>
                    <div class="content-panel">
                        <pre class="document-content">{new_html}</pre>
                    </div>
                </div>
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color diff-addition"></span>
                    <span>Additions</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color diff-deletion"></span>
                    <span>Deletions</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color diff-modification"></span>
                    <span>Modifications</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color diff-move"></span>
                    <span>Moved Content</span>
                </div>
            </div>
        </body>
        </html>
        """
        
        return {
            'format': 'html',
            'content': html_content,
            'old_segments': len(old_segments),
            'new_segments': len(new_segments)
        }
    
    def _generate_statistics_html(self, comparison_result: ComparisonResult,
                                semantic_diffs: List[SemanticDiff] = None,
                                legal_risks: List[LegalRisk] = None) -> str:
        """Generate HTML for comparison statistics."""
        stats = comparison_result.statistics
        
        stats_html = f"""
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-label">Similarity Score:</span>
                <span class="stat-value">{comparison_result.similarity_score:.1%}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Changes:</span>
                <span class="stat-value">{stats.get('total_differences', 0)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Insertions:</span>
                <span class="stat-value">{stats.get('insertions', 0)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Deletions:</span>
                <span class="stat-value">{stats.get('deletions', 0)}</span>
            </div>
        """
        
        if semantic_diffs:
            stats_html += f"""
            <div class="stat-item">
                <span class="stat-label">Semantic Changes:</span>
                <span class="stat-value">{len(semantic_diffs)}</span>
            </div>
            """
        
        if legal_risks:
            risk_counts = {}
            for risk in legal_risks:
                risk_counts[risk.risk_level] = risk_counts.get(risk.risk_level, 0) + 1
            
            stats_html += f"""
            <div class="stat-item">
                <span class="stat-label">Legal Risks:</span>
                <span class="stat-value">{len(legal_risks)}</span>
            </div>
            """
        
        stats_html += "</div>"
        return stats_html
    
    def _generate_text_output(self, old_segments: List[HighlightedSegment],
                            new_segments: List[HighlightedSegment]) -> Dict[str, Any]:
        """Generate plain text output."""
        def segments_to_text(segments: List[HighlightedSegment]) -> str:
            text_parts = []
            for segment in segments:
                if segment.style == HighlightStyle.DELETION:
                    text_parts.append(f"[-{segment.text}-]")
                elif segment.style == HighlightStyle.ADDITION:
                    text_parts.append(f"[+{segment.text}+]")
                elif segment.style == HighlightStyle.MODIFICATION:
                    text_parts.append(f"[~{segment.text}~]")
                else:
                    text_parts.append(segment.text)
            return ''.join(text_parts)
        
        return {
            'format': 'text',
            'old_content': segments_to_text(old_segments),
            'new_content': segments_to_text(new_segments)
        }
    
    def _generate_json_output(self, old_segments: List[HighlightedSegment],
                            new_segments: List[HighlightedSegment],
                            comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Generate JSON output."""
        def segment_to_dict(segment: HighlightedSegment) -> Dict[str, Any]:
            return {
                'text': segment.text,
                'style': segment.style.value,
                'start_pos': segment.start_pos,
                'end_pos': segment.end_pos,
                'tooltip': segment.tooltip,
                'metadata': segment.metadata or {}
            }
        
        return {
            'format': 'json',
            'old_segments': [segment_to_dict(seg) for seg in old_segments],
            'new_segments': [segment_to_dict(seg) for seg in new_segments],
            'statistics': comparison_result.statistics,
            'similarity_score': comparison_result.similarity_score
        }
    
    def _generate_css_styles(self) -> str:
        """Generate CSS styles for the comparison view."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .comparison-header {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .comparison-header h1 {
            margin: 0 0 15px 0;
            color: #333;
        }
        
        .stats-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
            min-width: 120px;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }
        
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        
        .comparison-container {
            display: flex;
            gap: 20px;
            height: calc(100vh - 200px);
        }
        
        .side-panel {
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .panel-header {
            background: #4a90e2;
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 16px;
        }
        
        .content-panel {
            flex: 1;
            overflow: auto;
            padding: 0;
        }
        
        .document-content {
            margin: 0;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .divider {
            width: 4px;
            background: #ddd;
            border-radius: 2px;
        }
        
        .diff-addition {
            background-color: #d4edda;
            color: #155724;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .diff-deletion {
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 4px;
            border-radius: 3px;
            text-decoration: line-through;
        }
        
        .diff-modification {
            background-color: #fff3cd;
            color: #856404;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .diff-move {
            background-color: #cce5ff;
            color: #004085;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px dashed #004085;
        }
        
        .diff-semantic {
            background-color: #e2e3ff;
            color: #383d41;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .diff-legal_risk {
            background-color: #ffcccc;
            color: #cc0000;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .legend {
            margin-top: 20px;
            padding: 15px 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 16px;
            border-radius: 3px;
        }
        
        @media (max-width: 768px) {
            .comparison-container {
                flex-direction: column;
                height: auto;
            }
            
            .side-panel {
                height: 400px;
            }
            
            .stats-container {
                justify-content: center;
            }
            
            .legend {
                justify-content: center;
            }
        }
        """


# Utility function
def create_side_by_side_view(old_content: str, new_content: str,
                           comparison_result: ComparisonResult,
                           format_type: str = "html") -> str:
    """
    Quick utility to create a side-by-side comparison view.
    
    Args:
        old_content: Original document
        new_content: Modified document  
        comparison_result: Comparison results
        format_type: Output format
        
    Returns:
        Formatted comparison view
    """
    visualizer = SideBySideVisualizer()
    result = visualizer.visualize_comparison(
        old_content, new_content, comparison_result, format_type=format_type
    )
    
    if format_type == "html":
        return result['content']
    else:
        return str(result)