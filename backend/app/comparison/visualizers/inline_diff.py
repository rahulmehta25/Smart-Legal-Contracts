"""
Inline diff visualizer for document changes.
Shows changes within the context of the document flow.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..diff_engine import DiffResult, DiffType, ComparisonResult
from ..semantic_comparison import SemanticDiff
from ..legal_change_detector import LegalRisk, LegalRiskLevel


class AnnotationType(Enum):
    """Types of annotations for inline diff."""
    INSERTION = "insertion"
    DELETION = "deletion"
    MODIFICATION = "modification"
    SEMANTIC_CHANGE = "semantic_change"
    LEGAL_RISK = "legal_risk"
    COMMENT = "comment"


@dataclass
class InlineAnnotation:
    """An annotation for inline diff display."""
    annotation_id: str
    annotation_type: AnnotationType
    start_pos: int
    end_pos: int
    content: str
    tooltip: str
    severity: str = "medium"
    metadata: Dict[str, Any] = None


class InlineDiffVisualizer:
    """
    Creates inline diff views showing changes within document context.
    Provides annotations, comments, and contextual highlighting.
    """
    
    def __init__(self):
        self.annotation_counter = 0
    
    def visualize_inline_diff(self, old_content: str, new_content: str,
                            comparison_result: ComparisonResult,
                            semantic_diffs: List[SemanticDiff] = None,
                            legal_risks: List[LegalRisk] = None,
                            show_context: bool = True) -> Dict[str, Any]:
        """
        Create inline diff visualization.
        
        Args:
            old_content: Original document content
            new_content: Modified document content
            comparison_result: Comparison results
            semantic_diffs: Optional semantic analysis
            legal_risks: Optional legal risk analysis
            show_context: Whether to show surrounding context
            
        Returns:
            Inline diff visualization data
        """
        # Create merged content with inline annotations
        merged_content, annotations = self._create_merged_content_with_annotations(
            old_content, new_content, comparison_result.differences,
            semantic_diffs, legal_risks
        )
        
        # Generate HTML output
        html_content = self._generate_html_inline_diff(
            merged_content, annotations, comparison_result, show_context
        )
        
        # Generate text output
        text_content = self._generate_text_inline_diff(merged_content, annotations)
        
        return {
            'merged_content': merged_content,
            'annotations': [self._annotation_to_dict(ann) for ann in annotations],
            'html_output': html_content,
            'text_output': text_content,
            'statistics': {
                'total_annotations': len(annotations),
                'insertions': len([a for a in annotations if a.annotation_type == AnnotationType.INSERTION]),
                'deletions': len([a for a in annotations if a.annotation_type == AnnotationType.DELETION]),
                'modifications': len([a for a in annotations if a.annotation_type == AnnotationType.MODIFICATION]),
                'semantic_changes': len([a for a in annotations if a.annotation_type == AnnotationType.SEMANTIC_CHANGE]),
                'legal_risks': len([a for a in annotations if a.annotation_type == AnnotationType.LEGAL_RISK])
            }
        }
    
    def _create_merged_content_with_annotations(self, old_content: str, new_content: str,
                                              differences: List[DiffResult],
                                              semantic_diffs: List[SemanticDiff] = None,
                                              legal_risks: List[LegalRisk] = None) -> Tuple[str, List[InlineAnnotation]]:
        """Create merged content with inline annotations."""
        # Start with the new content as base
        merged_content = new_content
        annotations = []
        
        # Process diff results to create annotations
        for diff in differences:
            annotation = self._create_annotation_from_diff(diff, merged_content)
            if annotation:
                annotations.append(annotation)
        
        # Add semantic annotations
        if semantic_diffs:
            for semantic_diff in semantic_diffs:
                annotation = self._create_semantic_annotation(semantic_diff, merged_content)
                if annotation:
                    annotations.append(annotation)
        
        # Add legal risk annotations
        if legal_risks:
            for legal_risk in legal_risks:
                annotation = self._create_legal_risk_annotation(legal_risk, merged_content)
                if annotation:
                    annotations.append(annotation)
        
        # Sort annotations by position
        annotations.sort(key=lambda a: a.start_pos)
        
        return merged_content, annotations
    
    def _create_annotation_from_diff(self, diff: DiffResult, content: str) -> Optional[InlineAnnotation]:
        """Create annotation from a diff result."""
        self.annotation_counter += 1
        
        if diff.diff_type == DiffType.INSERTION:
            return InlineAnnotation(
                annotation_id=f"diff_{self.annotation_counter}",
                annotation_type=AnnotationType.INSERTION,
                start_pos=diff.new_position[0],
                end_pos=diff.new_position[1],
                content=diff.new_content,
                tooltip=f"Added: {diff.new_content[:50]}{'...' if len(diff.new_content) > 50 else ''}",
                severity="medium",
                metadata={'diff': diff}
            )
        elif diff.diff_type == DiffType.DELETION:
            # For deletions, we need to find a position in the new content
            # This is simplified - in practice would need more sophisticated positioning
            return InlineAnnotation(
                annotation_id=f"diff_{self.annotation_counter}",
                annotation_type=AnnotationType.DELETION,
                start_pos=diff.new_position[0] if diff.new_position[0] < len(content) else len(content),
                end_pos=diff.new_position[0] if diff.new_position[0] < len(content) else len(content),
                content=f"[DELETED: {diff.old_content}]",
                tooltip=f"Deleted: {diff.old_content[:50]}{'...' if len(diff.old_content) > 50 else ''}",
                severity="high",
                metadata={'diff': diff}
            )
        elif diff.diff_type == DiffType.MODIFICATION:
            return InlineAnnotation(
                annotation_id=f"diff_{self.annotation_counter}",
                annotation_type=AnnotationType.MODIFICATION,
                start_pos=diff.new_position[0],
                end_pos=diff.new_position[1],
                content=diff.new_content,
                tooltip=f"Changed from: {diff.old_content[:30]}... to: {diff.new_content[:30]}...",
                severity="medium",
                metadata={'diff': diff, 'old_content': diff.old_content}
            )
        
        return None
    
    def _create_semantic_annotation(self, semantic_diff: SemanticDiff, content: str) -> Optional[InlineAnnotation]:
        """Create annotation from semantic diff."""
        self.annotation_counter += 1
        
        # Map semantic change type to severity
        severity_map = {
            SemanticChangeType.MEANING_NEGATED: "critical",
            SemanticChangeType.MEANING_SHIFTED: "high",
            SemanticChangeType.MEANING_STRENGTHENED: "medium",
            SemanticChangeType.MEANING_WEAKENED: "medium",
            SemanticChangeType.CONCEPT_ADDED: "low",
            SemanticChangeType.CONCEPT_REMOVED: "medium",
            SemanticChangeType.RELATIONSHIP_CHANGED: "high",
            SemanticChangeType.MEANING_PRESERVED: "low"
        }
        
        return InlineAnnotation(
            annotation_id=f"semantic_{self.annotation_counter}",
            annotation_type=AnnotationType.SEMANTIC_CHANGE,
            start_pos=semantic_diff.position[0],
            end_pos=semantic_diff.position[1],
            content=semantic_diff.new_segment,
            tooltip=f"Semantic change: {semantic_diff.change_type.value} - {semantic_diff.explanation}",
            severity=severity_map.get(semantic_diff.change_type, "medium"),
            metadata={
                'semantic_diff': semantic_diff,
                'similarity': semantic_diff.semantic_similarity,
                'intent_change': semantic_diff.intent_change_score
            }
        )
    
    def _create_legal_risk_annotation(self, legal_risk: LegalRisk, content: str) -> Optional[InlineAnnotation]:
        """Create annotation from legal risk."""
        self.annotation_counter += 1
        
        # Map legal risk level to severity
        severity_map = {
            LegalRiskLevel.CRITICAL: "critical",
            LegalRiskLevel.HIGH: "high",
            LegalRiskLevel.MEDIUM: "medium",
            LegalRiskLevel.LOW: "low",
            LegalRiskLevel.INFORMATIONAL: "low"
        }
        
        return InlineAnnotation(
            annotation_id=f"legal_{self.annotation_counter}",
            annotation_type=AnnotationType.LEGAL_RISK,
            start_pos=legal_risk.position[0],
            end_pos=legal_risk.position[1],
            content=legal_risk.new_language,
            tooltip=f"Legal Risk ({legal_risk.risk_level.value}): {legal_risk.description}",
            severity=severity_map.get(legal_risk.risk_level, "medium"),
            metadata={
                'legal_risk': legal_risk,
                'category': legal_risk.category.value,
                'impact': legal_risk.impact.value
            }
        )
    
    def _generate_html_inline_diff(self, content: str, annotations: List[InlineAnnotation],
                                 comparison_result: ComparisonResult, show_context: bool) -> str:
        """Generate HTML for inline diff view."""
        # Create annotated HTML content
        annotated_html = self._create_annotated_html(content, annotations)
        
        # Generate sidebar with annotation details
        sidebar_html = self._generate_annotation_sidebar(annotations)
        
        # Generate statistics
        stats_html = self._generate_inline_stats_html(comparison_result, annotations)
        
        css_styles = self._generate_inline_css()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Inline Document Comparison</title>
            <style>{css_styles}</style>
        </head>
        <body>
            <div class="inline-diff-container">
                <div class="header-section">
                    <h1>Inline Document Comparison</h1>
                    {stats_html}
                </div>
                
                <div class="main-content">
                    <div class="document-section">
                        <div class="document-header">
                            <h2>Document with Changes</h2>
                            <div class="view-controls">
                                <button onclick="toggleAnnotations()">Toggle Annotations</button>
                                <button onclick="filterByType('all')">All Changes</button>
                                <button onclick="filterByType('legal_risk')">Legal Risks Only</button>
                                <button onclick="filterByType('semantic_change')">Semantic Changes Only</button>
                            </div>
                        </div>
                        <div class="document-content">
                            <pre class="annotated-content">{annotated_html}</pre>
                        </div>
                    </div>
                    
                    <div class="sidebar-section">
                        <div class="sidebar-header">
                            <h3>Change Details</h3>
                        </div>
                        <div class="sidebar-content">
                            {sidebar_html}
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                {self._generate_javascript()}
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_annotated_html(self, content: str, annotations: List[InlineAnnotation]) -> str:
        """Create HTML content with inline annotations."""
        import html
        
        # Escape HTML in content
        content = html.escape(content)
        
        # Sort annotations by position (reverse order to maintain positions)
        sorted_annotations = sorted(annotations, key=lambda a: a.start_pos, reverse=True)
        
        # Insert annotation markers
        for annotation in sorted_annotations:
            start_pos = annotation.start_pos
            end_pos = annotation.end_pos
            
            # Ensure positions are within content bounds
            start_pos = max(0, min(start_pos, len(content)))
            end_pos = max(start_pos, min(end_pos, len(content)))
            
            # Create annotation wrapper
            css_class = f"annotation annotation-{annotation.annotation_type.value} severity-{annotation.severity}"
            data_attrs = f'data-id="{annotation.annotation_id}" data-type="{annotation.annotation_type.value}"'
            title_attr = f'title="{html.escape(annotation.tooltip)}"'
            
            if annotation.annotation_type == AnnotationType.DELETION:
                # For deletions, insert the deleted content marker
                marker = f'<span class="{css_class}" {data_attrs} {title_attr}>{html.escape(annotation.content)}</span>'
                content = content[:start_pos] + marker + content[start_pos:]
            else:
                # For other types, wrap the existing content
                original_content = content[start_pos:end_pos]
                wrapped_content = f'<span class="{css_class}" {data_attrs} {title_attr}>{original_content}</span>'
                content = content[:start_pos] + wrapped_content + content[end_pos:]
        
        return content
    
    def _generate_annotation_sidebar(self, annotations: List[InlineAnnotation]) -> str:
        """Generate sidebar with annotation details."""
        sidebar_items = []
        
        for annotation in annotations:
            severity_icon = {
                "critical": "🔴",
                "high": "🟠", 
                "medium": "🟡",
                "low": "🟢"
            }.get(annotation.severity, "⚪")
            
            type_label = annotation.annotation_type.value.replace("_", " ").title()
            
            sidebar_item = f"""
            <div class="sidebar-item" data-annotation-id="{annotation.annotation_id}">
                <div class="sidebar-item-header">
                    <span class="severity-icon">{severity_icon}</span>
                    <span class="type-label">{type_label}</span>
                </div>
                <div class="sidebar-item-content">
                    <div class="tooltip-content">{annotation.tooltip}</div>
                    {self._generate_metadata_display(annotation.metadata)}
                </div>
            </div>
            """
            sidebar_items.append(sidebar_item)
        
        return ''.join(sidebar_items)
    
    def _generate_metadata_display(self, metadata: Dict[str, Any]) -> str:
        """Generate display for annotation metadata."""
        if not metadata:
            return ""
        
        metadata_html = "<div class='metadata-section'>"
        
        # Display key metadata fields
        if 'diff' in metadata:
            diff = metadata['diff']
            metadata_html += f"<div class='metadata-item'><strong>Confidence:</strong> {diff.confidence:.2f}</div>"
        
        if 'similarity' in metadata:
            metadata_html += f"<div class='metadata-item'><strong>Similarity:</strong> {metadata['similarity']:.2f}</div>"
        
        if 'category' in metadata:
            metadata_html += f"<div class='metadata-item'><strong>Category:</strong> {metadata['category']}</div>"
        
        if 'impact' in metadata:
            metadata_html += f"<div class='metadata-item'><strong>Impact:</strong> {metadata['impact']}</div>"
        
        metadata_html += "</div>"
        return metadata_html
    
    def _generate_inline_stats_html(self, comparison_result: ComparisonResult, 
                                  annotations: List[InlineAnnotation]) -> str:
        """Generate statistics HTML for inline view."""
        stats = comparison_result.statistics
        
        annotation_counts = {}
        for annotation in annotations:
            ann_type = annotation.annotation_type.value
            annotation_counts[ann_type] = annotation_counts.get(ann_type, 0) + 1
        
        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{comparison_result.similarity_score:.1%}</div>
                <div class="stat-label">Similarity</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(annotations)}</div>
                <div class="stat-label">Total Changes</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{annotation_counts.get('insertion', 0)}</div>
                <div class="stat-label">Additions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{annotation_counts.get('deletion', 0)}</div>
                <div class="stat-label">Deletions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{annotation_counts.get('legal_risk', 0)}</div>
                <div class="stat-label">Legal Risks</div>
            </div>
        </div>
        """
    
    def _generate_text_inline_diff(self, content: str, annotations: List[InlineAnnotation]) -> str:
        """Generate text version of inline diff."""
        lines = content.split('\n')
        annotated_lines = []
        
        for line_num, line in enumerate(lines):
            line_annotations = [
                ann for ann in annotations
                if self._annotation_affects_line(ann, line_num, content)
            ]
            
            if line_annotations:
                # Add annotation markers to the line
                annotated_line = line
                for ann in reversed(sorted(line_annotations, key=lambda a: a.start_pos)):
                    marker = f"[{ann.annotation_type.value.upper()}]"
                    # This is simplified - would need more precise positioning
                    annotated_line += f" {marker}"
                
                annotated_lines.append(annotated_line)
                
                # Add annotation details as comments
                for ann in line_annotations:
                    annotated_lines.append(f"  // {ann.tooltip}")
            else:
                annotated_lines.append(line)
        
        return '\n'.join(annotated_lines)
    
    def _annotation_affects_line(self, annotation: InlineAnnotation, line_num: int, content: str) -> bool:
        """Check if an annotation affects a specific line."""
        lines_before = content[:annotation.start_pos].count('\n')
        lines_after = content[:annotation.end_pos].count('\n')
        return lines_before <= line_num <= lines_after
    
    def _annotation_to_dict(self, annotation: InlineAnnotation) -> Dict[str, Any]:
        """Convert annotation to dictionary."""
        return {
            'id': annotation.annotation_id,
            'type': annotation.annotation_type.value,
            'start_pos': annotation.start_pos,
            'end_pos': annotation.end_pos,
            'content': annotation.content,
            'tooltip': annotation.tooltip,
            'severity': annotation.severity,
            'metadata': annotation.metadata or {}
        }
    
    def _generate_inline_css(self) -> str:
        """Generate CSS for inline diff view."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            background-color: #f8f9fa;
        }
        
        .inline-diff-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header-section h1 {
            margin: 0 0 15px 0;
            color: #2c3e50;
        }
        
        .stats-grid {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            min-width: 80px;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            font-size: 12px;
            color: #6c757d;
            margin-top: 4px;
        }
        
        .main-content {
            display: flex;
            gap: 20px;
        }
        
        .document-section {
            flex: 2;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .document-header {
            background: #34495e;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .document-header h2 {
            margin: 0;
            font-size: 18px;
        }
        
        .view-controls button {
            background: #3498db;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            margin-left: 8px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .view-controls button:hover {
            background: #2980b9;
        }
        
        .document-content {
            height: calc(100vh - 300px);
            overflow: auto;
        }
        
        .annotated-content {
            margin: 0;
            padding: 20px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .sidebar-section {
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .sidebar-header {
            background: #95a5a6;
            color: white;
            padding: 15px 20px;
        }
        
        .sidebar-header h3 {
            margin: 0;
            font-size: 16px;
        }
        
        .sidebar-content {
            height: calc(100vh - 300px);
            overflow-y: auto;
            padding: 0;
        }
        
        .sidebar-item {
            padding: 15px 20px;
            border-bottom: 1px solid #ecf0f1;
            cursor: pointer;
        }
        
        .sidebar-item:hover {
            background-color: #f8f9fa;
        }
        
        .sidebar-item-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .severity-icon {
            font-size: 14px;
        }
        
        .type-label {
            font-weight: bold;
            color: #2c3e50;
            font-size: 14px;
        }
        
        .tooltip-content {
            color: #6c757d;
            font-size: 13px;
            line-height: 1.4;
        }
        
        .metadata-section {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #ecf0f1;
        }
        
        .metadata-item {
            font-size: 11px;
            color: #6c757d;
            margin-bottom: 4px;
        }
        
        /* Annotation styles */
        .annotation {
            position: relative;
            border-radius: 3px;
            padding: 1px 2px;
        }
        
        .annotation-insertion {
            background-color: #d4edda;
            color: #155724;
        }
        
        .annotation-deletion {
            background-color: #f8d7da;
            color: #721c24;
            text-decoration: line-through;
        }
        
        .annotation-modification {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .annotation-semantic_change {
            background-color: #e2e3ff;
            color: #383d41;
        }
        
        .annotation-legal_risk {
            background-color: #ffebee;
            color: #c62828;
            font-weight: bold;
        }
        
        .severity-critical {
            border: 2px solid #dc3545;
        }
        
        .severity-high {
            border: 1px solid #fd7e14;
        }
        
        .severity-medium {
            border: 1px solid #ffc107;
        }
        
        .severity-low {
            border: 1px solid #28a745;
        }
        """
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive features."""
        return """
        function toggleAnnotations() {
            const annotations = document.querySelectorAll('.annotation');
            annotations.forEach(ann => {
                ann.style.display = ann.style.display === 'none' ? 'inline' : 'none';
            });
        }
        
        function filterByType(type) {
            const annotations = document.querySelectorAll('.annotation');
            const sidebarItems = document.querySelectorAll('.sidebar-item');
            
            annotations.forEach(ann => {
                if (type === 'all' || ann.dataset.type === type) {
                    ann.style.display = 'inline';
                } else {
                    ann.style.display = 'none';
                }
            });
            
            sidebarItems.forEach(item => {
                const annId = item.dataset.annotationId;
                const annotation = document.querySelector(`[data-id="${annId}"]`);
                if (!annotation || annotation.style.display === 'none') {
                    item.style.display = 'none';
                } else {
                    item.style.display = 'block';
                }
            });
        }
        
        // Highlight corresponding annotation when sidebar item is hovered
        document.addEventListener('DOMContentLoaded', function() {
            const sidebarItems = document.querySelectorAll('.sidebar-item');
            sidebarItems.forEach(item => {
                item.addEventListener('mouseenter', function() {
                    const annId = this.dataset.annotationId;
                    const annotation = document.querySelector(`[data-id="${annId}"]`);
                    if (annotation) {
                        annotation.style.backgroundColor = '#ffeb3b';
                    }
                });
                
                item.addEventListener('mouseleave', function() {
                    const annId = this.dataset.annotationId;
                    const annotation = document.querySelector(`[data-id="${annId}"]`);
                    if (annotation) {
                        annotation.style.backgroundColor = '';
                    }
                });
            });
        });
        """


# Utility function
def create_inline_diff(old_content: str, new_content: str,
                      comparison_result: ComparisonResult) -> str:
    """
    Quick utility to create inline diff visualization.
    
    Args:
        old_content: Original document
        new_content: Modified document
        comparison_result: Comparison results
        
    Returns:
        HTML string with inline diff
    """
    visualizer = InlineDiffVisualizer()
    result = visualizer.visualize_inline_diff(old_content, new_content, comparison_result)
    return result['html_output']