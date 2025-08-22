"""
PDF export functionality for document comparison results.
Creates annotated PDF reports with highlighting and analysis.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect
from io import BytesIO
import html
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..diff_engine import ComparisonResult, DiffResult, DiffType
from ..semantic_comparison import SemanticDiff
from ..legal_change_detector import LegalRisk, LegalRiskLevel


class PDFExporter:
    """
    Exports document comparison results to annotated PDF reports.
    Supports various layouts and highlighting styles.
    """
    
    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='DocumentTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#34495e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='ChangeHighlight',
            parent=self.styles['Normal'],
            backgroundColor=colors.HexColor('#fff3cd'),
            borderColor=colors.HexColor('#ffc107'),
            borderWidth=1,
            borderPadding=3
        ))
        
        self.styles.add(ParagraphStyle(
            name='LegalRisk',
            parent=self.styles['Normal'],
            backgroundColor=colors.HexColor('#f8d7da'),
            textColor=colors.HexColor('#721c24'),
            borderColor=colors.HexColor('#dc3545'),
            borderWidth=1,
            borderPadding=3
        ))
    
    def export_comparison_report(self, old_content: str, new_content: str,
                               comparison_result: ComparisonResult,
                               semantic_diffs: List[SemanticDiff] = None,
                               legal_risks: List[LegalRisk] = None,
                               output_path: str = None) -> BytesIO:
        """
        Export complete comparison report as PDF.
        
        Args:
            old_content: Original document content
            new_content: Modified document content
            comparison_result: Comparison results
            semantic_diffs: Optional semantic analysis
            legal_risks: Optional legal risk analysis
            output_path: Optional file path to save PDF
            
        Returns:
            BytesIO buffer with PDF content
        """
        buffer = BytesIO()
        
        # Create document
        if output_path:
            doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
        else:
            doc = SimpleDocTemplate(buffer, pagesize=self.page_size)
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(comparison_result))
        
        # Executive summary
        story.extend(self._create_executive_summary(
            comparison_result, semantic_diffs, legal_risks
        ))
        
        # Detailed analysis
        story.extend(self._create_detailed_analysis(
            comparison_result, semantic_diffs, legal_risks
        ))
        
        # Side-by-side comparison
        story.extend(self._create_side_by_side_content(old_content, new_content, comparison_result))
        
        # Appendices
        if legal_risks:
            story.extend(self._create_legal_risk_appendix(legal_risks))
        
        if semantic_diffs:
            story.extend(self._create_semantic_analysis_appendix(semantic_diffs))
        
        # Build PDF
        doc.build(story)
        
        if not output_path:
            buffer.seek(0)
            return buffer
        
        return None
    
    def export_side_by_side_pdf(self, old_content: str, new_content: str,
                               comparison_result: ComparisonResult,
                               output_path: str = None) -> BytesIO:
        """Export side-by-side comparison as PDF."""
        buffer = BytesIO()
        
        if output_path:
            c = canvas.Canvas(output_path, pagesize=self.page_size)
        else:
            c = canvas.Canvas(buffer, pagesize=self.page_size)
        
        width, height = self.page_size
        
        # Set up layout
        margin = 0.5 * inch
        column_width = (width - 3 * margin) / 2
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, height - margin, "Document Comparison - Side by Side")
        
        # Column headers
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, height - margin - 40, "Original Document")
        c.drawString(margin + column_width + margin, height - margin - 40, "Modified Document")
        
        # Draw column divider
        c.line(margin + column_width + margin/2, height - margin - 60, 
               margin + column_width + margin/2, margin)
        
        # Process content
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        y_pos = height - margin - 70
        line_height = 12
        
        max_lines_per_page = int((height - 2 * margin - 70) / line_height)
        current_line = 0
        
        for i in range(max(len(old_lines), len(new_lines))):
            if current_line >= max_lines_per_page:
                c.showPage()
                y_pos = height - margin
                current_line = 0
            
            # Draw old content
            if i < len(old_lines):
                old_line = old_lines[i][:80]  # Truncate long lines
                c.setFont("Helvetica", 9)
                c.drawString(margin, y_pos, old_line)
            
            # Draw new content
            if i < len(new_lines):
                new_line = new_lines[i][:80]  # Truncate long lines
                c.setFont("Helvetica", 9)
                c.drawString(margin + column_width + margin, y_pos, new_line)
            
            y_pos -= line_height
            current_line += 1
        
        c.save()
        
        if not output_path:
            buffer.seek(0)
            return buffer
        
        return None
    
    def _create_title_page(self, comparison_result: ComparisonResult) -> List:
        """Create title page elements."""
        elements = []
        
        # Title
        title = Paragraph("Document Comparison Report", self.styles['DocumentTitle'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Metadata table
        metadata = [
            ['Analysis Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Document A ID', comparison_result.document_a_id],
            ['Document B ID', comparison_result.document_b_id],
            ['Similarity Score', f"{comparison_result.similarity_score:.1%}"],
            ['Total Changes', str(len(comparison_result.differences))]
        ]
        
        table = Table(metadata, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 40))
        
        return elements
    
    def _create_executive_summary(self, comparison_result: ComparisonResult,
                                semantic_diffs: List[SemanticDiff] = None,
                                legal_risks: List[LegalRisk] = None) -> List:
        """Create executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Summary statistics
        stats = comparison_result.statistics
        summary_text = f"""
        This report analyzes the differences between two document versions with a similarity score of 
        {comparison_result.similarity_score:.1%}. A total of {len(comparison_result.differences)} changes 
        were identified, including {stats.get('insertions', 0)} insertions, {stats.get('deletions', 0)} 
        deletions, and {stats.get('modifications', 0)} modifications.
        """
        
        if semantic_diffs:
            summary_text += f" Additionally, {len(semantic_diffs)} semantic changes were detected."
        
        if legal_risks:
            critical_risks = len([r for r in legal_risks if r.risk_level == LegalRiskLevel.CRITICAL])
            high_risks = len([r for r in legal_risks if r.risk_level == LegalRiskLevel.HIGH])
            summary_text += f" Legal analysis identified {len(legal_risks)} potential risks, including {critical_risks} critical and {high_risks} high-priority issues."
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_detailed_analysis(self, comparison_result: ComparisonResult,
                                semantic_diffs: List[SemanticDiff] = None,
                                legal_risks: List[LegalRisk] = None) -> List:
        """Create detailed analysis section."""
        elements = []
        
        elements.append(Paragraph("Detailed Analysis", self.styles['SectionHeader']))
        
        # Change breakdown
        change_data = [['Change Type', 'Count', 'Percentage']]
        stats = comparison_result.statistics
        total_changes = len(comparison_result.differences)
        
        if total_changes > 0:
            change_types = [
                ('Insertions', stats.get('insertions', 0)),
                ('Deletions', stats.get('deletions', 0)),
                ('Modifications', stats.get('modifications', 0)),
                ('Moves', stats.get('moves', 0))
            ]
            
            for change_type, count in change_types:
                percentage = (count / total_changes) * 100 if total_changes > 0 else 0
                change_data.append([change_type, str(count), f"{percentage:.1f}%"])
        
        change_table = Table(change_data, colWidths=[2*inch, 1*inch, 1*inch])
        change_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        elements.append(change_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_side_by_side_content(self, old_content: str, new_content: str,
                                   comparison_result: ComparisonResult) -> List:
        """Create side-by-side content comparison."""
        elements = []
        
        elements.append(Paragraph("Document Comparison", self.styles['SectionHeader']))
        
        # Split content into chunks for display
        old_chunks = self._split_content_for_pdf(old_content)
        new_chunks = self._split_content_for_pdf(new_content)
        
        # Create comparison table
        comparison_data = [['Original Document', 'Modified Document']]
        
        max_chunks = max(len(old_chunks), len(new_chunks))
        for i in range(max_chunks):
            old_chunk = old_chunks[i] if i < len(old_chunks) else ""
            new_chunk = new_chunks[i] if i < len(new_chunks) else ""
            
            # Highlight changes in chunks
            old_para = Paragraph(self._escape_html(old_chunk), self.styles['Normal'])
            new_para = Paragraph(self._escape_html(new_chunk), self.styles['Normal'])
            
            comparison_data.append([old_para, new_para])
            
            # Limit number of rows to prevent huge PDFs
            if len(comparison_data) > 20:
                comparison_data.append([
                    Paragraph("... (content truncated) ...", self.styles['Normal']),
                    Paragraph("... (content truncated) ...", self.styles['Normal'])
                ])
                break
        
        comparison_table = Table(comparison_data, colWidths=[3.5*inch, 3.5*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        elements.append(comparison_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_legal_risk_appendix(self, legal_risks: List[LegalRisk]) -> List:
        """Create legal risk analysis appendix."""
        elements = []
        
        elements.append(Paragraph("Legal Risk Analysis", self.styles['SectionHeader']))
        
        # Group risks by level
        risks_by_level = {}
        for risk in legal_risks:
            level = risk.risk_level
            if level not in risks_by_level:
                risks_by_level[level] = []
            risks_by_level[level].append(risk)
        
        # Display risks by severity
        severity_order = [
            LegalRiskLevel.CRITICAL,
            LegalRiskLevel.HIGH,
            LegalRiskLevel.MEDIUM,
            LegalRiskLevel.LOW,
            LegalRiskLevel.INFORMATIONAL
        ]
        
        for level in severity_order:
            if level not in risks_by_level:
                continue
            
            level_risks = risks_by_level[level]
            elements.append(Paragraph(f"{level.value.title()} Risk Issues ({len(level_risks)})", 
                                    self.styles['Heading3']))
            
            for i, risk in enumerate(level_risks[:10]):  # Limit to 10 per level
                risk_text = f"""
                <b>Risk {i+1}:</b> {risk.description}<br/>
                <b>Category:</b> {risk.category.value}<br/>
                <b>Impact:</b> {risk.impact.value}<br/>
                <b>Recommendation:</b> {risk.recommendation}<br/>
                <b>Affected Content:</b> {risk.new_language[:100]}{'...' if len(risk.new_language) > 100 else ''}
                """
                
                elements.append(Paragraph(risk_text, self.styles['LegalRisk']))
                elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_semantic_analysis_appendix(self, semantic_diffs: List[SemanticDiff]) -> List:
        """Create semantic analysis appendix."""
        elements = []
        
        elements.append(Paragraph("Semantic Analysis", self.styles['SectionHeader']))
        
        # Group by change type
        changes_by_type = {}
        for diff in semantic_diffs:
            change_type = diff.change_type
            if change_type not in changes_by_type:
                changes_by_type[change_type] = []
            changes_by_type[change_type].append(diff)
        
        for change_type, diffs in changes_by_type.items():
            elements.append(Paragraph(f"{change_type.value.replace('_', ' ').title()} ({len(diffs)})",
                                    self.styles['Heading3']))
            
            for i, diff in enumerate(diffs[:5]):  # Limit to 5 per type
                semantic_text = f"""
                <b>Change {i+1}:</b> {diff.explanation}<br/>
                <b>Semantic Similarity:</b> {diff.semantic_similarity:.2f}<br/>
                <b>Intent Change Score:</b> {diff.intent_change_score:.2f}<br/>
                <b>Confidence:</b> {diff.confidence:.2f}<br/>
                <b>Content:</b> {diff.new_segment[:150]}{'...' if len(diff.new_segment) > 150 else ''}
                """
                
                elements.append(Paragraph(semantic_text, self.styles['ChangeHighlight']))
                elements.append(Spacer(1, 10))
        
        return elements
    
    def _split_content_for_pdf(self, content: str, max_chunk_size: int = 500) -> List[str]:
        """Split content into chunks suitable for PDF display."""
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters for PDF display."""
        return html.escape(text)


# Utility functions
def export_comparison_pdf(old_content: str, new_content: str,
                         comparison_result: ComparisonResult,
                         output_path: str = None) -> BytesIO:
    """
    Quick utility to export comparison as PDF.
    
    Args:
        old_content: Original document
        new_content: Modified document
        comparison_result: Comparison results
        output_path: Optional output file path
        
    Returns:
        BytesIO buffer with PDF (if no output_path)
    """
    exporter = PDFExporter()
    return exporter.export_comparison_report(
        old_content, new_content, comparison_result, output_path=output_path
    )