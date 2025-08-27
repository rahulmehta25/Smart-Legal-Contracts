"""
Document Comparison Demo Workflow
================================

This module provides an interactive demonstration of document comparison
capabilities including semantic analysis, visual diff display, and 
legal change tracking.

Features:
- Side-by-side document comparison
- Semantic similarity analysis
- Legal change detection and classification
- Visual diff rendering with highlighting
- Version tracking and history
- Change impact assessment
"""

import streamlit as st
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import difflib
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from app.comparison.semantic_comparison import SemanticComparator
    from app.comparison.legal_change_detector import LegalChangeDetector
    from app.comparison.diff_engine import DiffEngine
    from app.comparison.version_tracker import VersionTracker
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")


class DocumentComparisonDemo:
    """Interactive document comparison demonstration."""
    
    def __init__(self):
        """Initialize the demo."""
        self.semantic_comparator = None
        self.change_detector = None
        self.diff_engine = None
        self.version_tracker = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize comparison components."""
        try:
            # Note: These would normally be imported from actual backend modules
            # For demo purposes, we'll create mock implementations
            self.semantic_comparator = MockSemanticComparator()
            self.change_detector = MockLegalChangeDetector()
            self.diff_engine = MockDiffEngine()
            self.version_tracker = MockVersionTracker()
        except Exception as e:
            st.error(f"Error initializing components: {e}")
    
    def render_comparison_interface(self):
        """Render the document comparison interface."""
        st.markdown("### 📋 Document Comparison Interface")
        
        # Document input methods
        tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "✏️ Text Input", "📚 Sample Documents"])
        
        with tab1:
            self.render_upload_interface()
        
        with tab2:
            self.render_text_input_interface()
        
        with tab3:
            self.render_sample_documents_interface()
    
    def render_upload_interface(self):
        """Render file upload interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 Document A (Original)")
            doc_a_files = st.file_uploader(
                "Upload original document",
                type=['txt', 'pdf', 'docx'],
                key="doc_a_upload"
            )
        
        with col2:
            st.markdown("#### 📄 Document B (Modified)")
            doc_b_files = st.file_uploader(
                "Upload modified document",
                type=['txt', 'pdf', 'docx'],
                key="doc_b_upload"
            )
        
        if st.button("🔍 Compare Uploaded Documents", type="primary"):
            if doc_a_files and doc_b_files:
                doc_a_content = self.extract_text_from_file(doc_a_files)
                doc_b_content = self.extract_text_from_file(doc_b_files)
                
                self.perform_comparison(
                    doc_a_content, doc_b_content,
                    doc_a_files.name, doc_b_files.name
                )
            else:
                st.warning("Please upload both documents to compare")
    
    def render_text_input_interface(self):
        """Render text input interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Document A (Original)")
            doc_a_text = st.text_area(
                "Original document text",
                height=300,
                placeholder="Paste the original document text here...",
                key="doc_a_text"
            )
        
        with col2:
            st.markdown("#### 📝 Document B (Modified)")
            doc_b_text = st.text_area(
                "Modified document text",
                height=300,
                placeholder="Paste the modified document text here...",
                key="doc_b_text"
            )
        
        if st.button("🔍 Compare Text Documents", type="primary"):
            if doc_a_text.strip() and doc_b_text.strip():
                self.perform_comparison(
                    doc_a_text.strip(), doc_b_text.strip(),
                    "Document A", "Document B"
                )
            else:
                st.warning("Please enter text for both documents")
    
    def render_sample_documents_interface(self):
        """Render sample documents interface."""
        st.markdown("#### 📚 Try with Sample Documents")
        
        sample_pairs = self.get_sample_document_pairs()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Privacy Policy Updates"):
                pair = sample_pairs['privacy_policy']
                self.perform_comparison(
                    pair['original'], pair['modified'],
                    "Privacy Policy v1.0", "Privacy Policy v2.0"
                )
        
        with col2:
            if st.button("📄 Terms of Service Changes"):
                pair = sample_pairs['terms_of_service']
                self.perform_comparison(
                    pair['original'], pair['modified'],
                    "ToS v1.0", "ToS v2.0"
                )
        
        with col3:
            if st.button("📜 License Agreement Updates"):
                pair = sample_pairs['license_agreement']
                self.perform_comparison(
                    pair['original'], pair['modified'],
                    "License v1.0", "License v2.0"
                )
    
    def perform_comparison(self, doc_a: str, doc_b: str, name_a: str, name_b: str):
        """Perform document comparison and display results."""
        with st.spinner("Analyzing documents..."):
            # Perform various types of comparison
            comparison_results = {
                'semantic_similarity': self.semantic_comparator.compare(doc_a, doc_b),
                'legal_changes': self.change_detector.detect_changes(doc_a, doc_b),
                'textual_diff': self.diff_engine.generate_diff(doc_a, doc_b),
                'change_summary': self.generate_change_summary(doc_a, doc_b)
            }
            
            # Store results
            st.session_state.demo_results['comparison'] = {
                'documents': {'a': {'name': name_a, 'content': doc_a}, 'b': {'name': name_b, 'content': doc_b}},
                'results': comparison_results,
                'timestamp': time.time()
            }
        
        # Display results
        self.display_comparison_results(comparison_results, name_a, name_b)
    
    def display_comparison_results(self, results: Dict, name_a: str, name_b: str):
        """Display comparison results."""
        st.markdown("## 📊 Comparison Results")
        
        # Overall similarity metrics
        col1, col2, col3, col4 = st.columns(4)
        
        semantic_score = results['semantic_similarity']['overall_score']
        structural_score = results['semantic_similarity']['structural_similarity']
        legal_impact = results['legal_changes']['impact_score']
        change_count = len(results['legal_changes']['changes'])
        
        with col1:
            st.metric("Semantic Similarity", f"{semantic_score:.1%}")
        with col2:
            st.metric("Structural Similarity", f"{structural_score:.1%}")
        with col3:
            st.metric("Legal Impact Score", f"{legal_impact:.1%}")
        with col4:
            st.metric("Changes Detected", change_count)
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visual Diff", "🔍 Legal Changes", "📈 Semantic Analysis", "📋 Change Summary"
        ])
        
        with tab1:
            self.render_visual_diff(results['textual_diff'], name_a, name_b)
        
        with tab2:
            self.render_legal_changes(results['legal_changes'])
        
        with tab3:
            self.render_semantic_analysis(results['semantic_similarity'])
        
        with tab4:
            self.render_change_summary(results['change_summary'])
    
    def render_visual_diff(self, diff_results: Dict, name_a: str, name_b: str):
        """Render visual diff display."""
        st.markdown("### 📊 Visual Document Comparison")
        
        # Diff view options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            view_mode = st.selectbox(
                "View Mode",
                ["Side by Side", "Unified", "Inline Highlights"]
            )
            
            show_unchanged = st.checkbox("Show Unchanged Text", value=False)
            context_lines = st.slider("Context Lines", 0, 10, 3)
        
        with col1:
            if view_mode == "Side by Side":
                self.render_side_by_side_diff(diff_results, name_a, name_b, show_unchanged, context_lines)
            elif view_mode == "Unified":
                self.render_unified_diff(diff_results, show_unchanged, context_lines)
            else:
                self.render_inline_diff(diff_results)
    
    def render_side_by_side_diff(self, diff_results: Dict, name_a: str, name_b: str, 
                                show_unchanged: bool, context_lines: int):
        """Render side-by-side diff view."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 📄 {name_a}")
            original_lines = diff_results['original_lines']
            self.render_diff_lines(original_lines, 'original', show_unchanged)
        
        with col2:
            st.markdown(f"#### 📄 {name_b}")
            modified_lines = diff_results['modified_lines']
            self.render_diff_lines(modified_lines, 'modified', show_unchanged)
    
    def render_diff_lines(self, lines: List[Dict], doc_type: str, show_unchanged: bool):
        """Render individual diff lines with styling."""
        for line in lines:
            if not show_unchanged and line['type'] == 'unchanged':
                continue
            
            line_type = line['type']
            content = line['content']
            
            if line_type == 'added':
                st.markdown(f'<div style="background-color: #d4edda; padding: 5px; border-left: 4px solid #28a745;">+ {content}</div>', 
                          unsafe_allow_html=True)
            elif line_type == 'removed':
                st.markdown(f'<div style="background-color: #f8d7da; padding: 5px; border-left: 4px solid #dc3545;">- {content}</div>', 
                          unsafe_allow_html=True)
            elif line_type == 'modified':
                st.markdown(f'<div style="background-color: #fff3cd; padding: 5px; border-left: 4px solid #ffc107;">~ {content}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding: 5px; color: #6c757d;">{content}</div>', 
                          unsafe_allow_html=True)
    
    def render_unified_diff(self, diff_results: Dict, show_unchanged: bool, context_lines: int):
        """Render unified diff view."""
        st.markdown("#### 📄 Unified Diff View")
        
        unified_diff = diff_results['unified_diff']
        
        diff_html = "<div style='font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>"
        
        for line in unified_diff:
            line_type = line['type']
            content = line['content']
            
            if not show_unchanged and line_type == 'unchanged':
                continue
            
            if line_type == 'added':
                diff_html += f'<div style="background-color: #d4edda; color: #155724;">+{content}</div>'
            elif line_type == 'removed':
                diff_html += f'<div style="background-color: #f8d7da; color: #721c24;">-{content}</div>'
            elif line_type == 'context':
                diff_html += f'<div style="color: #6c757d;"> {content}</div>'
            else:
                diff_html += f'<div>{content}</div>'
        
        diff_html += "</div>"
        st.markdown(diff_html, unsafe_allow_html=True)
    
    def render_inline_diff(self, diff_results: Dict):
        """Render inline diff with word-level highlighting."""
        st.markdown("#### 📄 Inline Diff with Word Highlighting")
        
        inline_diff = diff_results['inline_diff']
        
        st.markdown(inline_diff, unsafe_allow_html=True)
    
    def render_legal_changes(self, legal_changes: Dict):
        """Render legal changes analysis."""
        st.markdown("### 🔍 Legal Changes Analysis")
        
        changes = legal_changes['changes']
        
        if not changes:
            st.info("No significant legal changes detected.")
            return
        
        # Change category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Change types pie chart
            change_types = [change['category'] for change in changes]
            type_counts = pd.Series(change_types).value_counts()
            
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title="Change Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Impact levels
            impact_levels = [change['impact_level'] for change in changes]
            impact_counts = pd.Series(impact_levels).value_counts()
            
            fig = px.bar(x=impact_counts.index, y=impact_counts.values,
                        title="Impact Levels Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed changes
        st.markdown("#### 📋 Detailed Changes")
        
        for i, change in enumerate(changes):
            with st.expander(f"{change['category']} - {change['impact_level']} Impact", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {change['description']}")
                    st.markdown(f"**Legal Significance:** {change['legal_significance']}")
                    
                    if change.get('original_text'):
                        st.markdown("**Original Text:**")
                        st.markdown(f"```\n{change['original_text']}\n```")
                    
                    if change.get('modified_text'):
                        st.markdown("**Modified Text:**")
                        st.markdown(f"```\n{change['modified_text']}\n```")
                
                with col2:
                    st.markdown(f"**Category:** {change['category']}")
                    st.markdown(f"**Impact Level:** {change['impact_level']}")
                    st.markdown(f"**Confidence:** {change['confidence']:.1%}")
                    
                    if change.get('recommendations'):
                        st.markdown("**Recommendations:**")
                        for rec in change['recommendations']:
                            st.write(f"• {rec}")
    
    def render_semantic_analysis(self, semantic_results: Dict):
        """Render semantic analysis results."""
        st.markdown("### 📈 Semantic Analysis")
        
        # Similarity metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Overall Similarity",
                f"{semantic_results['overall_score']:.1%}",
                help="Overall semantic similarity between documents"
            )
        
        with col2:
            st.metric(
                "Structural Similarity",
                f"{semantic_results['structural_similarity']:.1%}",
                help="Similarity in document structure and organization"
            )
        
        with col3:
            st.metric(
                "Content Similarity",
                f"{semantic_results['content_similarity']:.1%}",
                help="Similarity in actual content and meaning"
            )
        
        # Section-by-section analysis
        if 'section_similarities' in semantic_results:
            st.markdown("#### 📊 Section-by-Section Analysis")
            
            section_data = []
            for section, similarity in semantic_results['section_similarities'].items():
                section_data.append({
                    'Section': section,
                    'Similarity': similarity
                })
            
            if section_data:
                df = pd.DataFrame(section_data)
                
                fig = px.bar(df, x='Section', y='Similarity',
                           title="Similarity by Document Section")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Similarity heatmap
        if 'similarity_matrix' in semantic_results:
            st.markdown("#### 🔥 Similarity Heatmap")
            
            similarity_matrix = semantic_results['similarity_matrix']
            
            fig = px.imshow(similarity_matrix,
                          title="Document Section Similarity Matrix",
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_change_summary(self, change_summary: Dict):
        """Render change summary."""
        st.markdown("### 📋 Change Summary")
        
        # High-level statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Changes", change_summary['total_changes'])
        with col2:
            st.metric("High Impact", change_summary['high_impact_changes'])
        with col3:
            st.metric("Words Added", change_summary['words_added'])
        with col4:
            st.metric("Words Removed", change_summary['words_removed'])
        
        # Change timeline
        if 'change_timeline' in change_summary:
            st.markdown("#### ⏱️ Change Timeline")
            
            timeline_data = change_summary['change_timeline']
            df = pd.DataFrame(timeline_data)
            
            fig = px.timeline(df, x_start="start", x_end="end", y="section",
                            title="Document Changes Timeline")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("#### 💡 Key Insights")
        
        insights = change_summary.get('insights', [])
        for insight in insights:
            st.write(f"• {insight}")
        
        if not insights:
            st.info("No specific insights generated for this comparison.")
    
    def render_version_tracking(self):
        """Render version tracking interface."""
        st.markdown("### 🔄 Document Version Tracking")
        
        # Version history
        if hasattr(st.session_state, 'document_versions'):
            st.markdown("#### 📚 Version History")
            
            versions = st.session_state.document_versions
            
            if versions:
                version_df = pd.DataFrame(versions)
                st.dataframe(version_df, use_container_width=True)
            else:
                st.info("No version history available.")
        
        # Add new version
        st.markdown("#### ➕ Add New Version")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            version_name = st.text_input("Version Name", placeholder="e.g., v2.1, Draft 3")
            version_notes = st.text_area("Version Notes", placeholder="Describe the changes in this version...")
        
        with col2:
            version_type = st.selectbox("Version Type", ["Major", "Minor", "Patch", "Draft"])
            auto_track = st.checkbox("Auto-track Changes", value=True)
        
        if st.button("📝 Save Version"):
            if version_name:
                # This would normally save to the version tracker
                st.success(f"Version '{version_name}' saved successfully!")
            else:
                st.warning("Please enter a version name.")
    
    def render_analytics(self):
        """Render analytics dashboard."""
        st.markdown("### 📈 Comparison Analytics")
        
        if 'comparison' not in st.session_state.demo_results:
            st.info("No comparison data available. Please run a comparison first.")
            return
        
        comparison_data = st.session_state.demo_results['comparison']
        
        # Analytics overview
        st.markdown("#### 📊 Analytics Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Compared", 2)
        with col2:
            st.metric("Analysis Duration", f"{time.time() - comparison_data['timestamp']:.1f}s")
        with col3:
            st.metric("Data Points", len(comparison_data['results']['legal_changes']['changes']))
        
        # Comparison trends (mock data for demo)
        st.markdown("#### 📈 Comparison Trends")
        
        # Generate sample trend data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Comparisons': np.random.randint(5, 25, 30),
            'Legal Changes': np.random.randint(1, 10, 30),
            'High Impact Changes': np.random.randint(0, 5, 30)
        })
        
        fig = px.line(trend_data, x='Date', y=['Comparisons', 'Legal Changes', 'High Impact Changes'],
                     title="Document Comparison Activity")
        st.plotly_chart(fig, use_container_width=True)
    
    def extract_text_from_file(self, file) -> str:
        """Extract text content from uploaded file."""
        if file.type == "text/plain":
            return str(file.read(), "utf-8")
        elif file.type == "application/pdf":
            return f"[PDF content extraction not implemented in demo - filename: {file.name}]"
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return f"[DOCX content extraction not implemented in demo - filename: {file.name}]"
        else:
            return str(file.read(), "utf-8")
    
    def generate_change_summary(self, doc_a: str, doc_b: str) -> Dict:
        """Generate a summary of changes between documents."""
        # Mock implementation for demo
        words_a = len(doc_a.split())
        words_b = len(doc_b.split())
        
        return {
            'total_changes': 15,
            'high_impact_changes': 3,
            'words_added': max(0, words_b - words_a),
            'words_removed': max(0, words_a - words_b),
            'insights': [
                "New arbitration clause added in section 12",
                "Privacy policy strengthened with GDPR compliance",
                "Data retention period reduced from 5 years to 3 years",
                "User rights section expanded with deletion requests"
            ]
        }
    
    def get_sample_document_pairs(self) -> Dict[str, Dict]:
        """Get sample document pairs for demonstration."""
        return {
            'privacy_policy': {
                'original': '''
                PRIVACY POLICY
                
                1. INFORMATION COLLECTION
                We collect personal information that you provide to us directly.
                
                2. DATA USAGE
                We use your information to provide and improve our services.
                
                3. DATA RETENTION
                We retain your personal data for up to 5 years after account closure.
                
                4. YOUR RIGHTS
                You have the right to access and update your personal information.
                ''',
                'modified': '''
                PRIVACY POLICY
                
                1. INFORMATION COLLECTION
                We collect personal information that you provide to us directly, including name, email, and usage data.
                
                2. DATA USAGE
                We use your information to provide and improve our services, and for analytics purposes.
                
                3. DATA RETENTION
                We retain your personal data for up to 3 years after account closure, or as required by law.
                
                4. YOUR RIGHTS
                You have the right to access, update, delete, and port your personal information. You may also object to processing.
                
                5. GDPR COMPLIANCE
                For EU residents, we comply with GDPR requirements including lawful basis for processing.
                '''
            },
            'terms_of_service': {
                'original': '''
                TERMS OF SERVICE
                
                1. ACCEPTANCE
                By using our service, you agree to these terms.
                
                15. DISPUTES
                Any disputes will be resolved through negotiation or litigation.
                
                16. GOVERNING LAW
                These terms are governed by the laws of Delaware.
                ''',
                'modified': '''
                TERMS OF SERVICE
                
                1. ACCEPTANCE
                By using our service, you agree to these terms and conditions.
                
                15. DISPUTE RESOLUTION
                Any disputes, claims, or controversies shall be resolved through binding arbitration administered by the American Arbitration Association. You waive your right to jury trial and class action participation.
                
                16. GOVERNING LAW
                These terms are governed by the laws of Delaware, USA.
                '''
            },
            'license_agreement': {
                'original': '''
                SOFTWARE LICENSE AGREEMENT
                
                1. GRANT OF LICENSE
                We grant you a non-exclusive license to use this software.
                
                2. RESTRICTIONS
                You may not modify or distribute the software.
                
                3. TERMINATION
                This license terminates if you breach these terms.
                ''',
                'modified': '''
                SOFTWARE LICENSE AGREEMENT
                
                1. GRANT OF LICENSE
                We grant you a non-exclusive, non-transferable license to use this software for personal or commercial purposes.
                
                2. RESTRICTIONS
                You may not modify, reverse engineer, or distribute the software without explicit written permission.
                
                3. TERMINATION
                This license terminates automatically if you breach these terms. We may also terminate for convenience with 30 days notice.
                
                4. WARRANTY DISCLAIMER
                The software is provided "as is" without any warranties, express or implied.
                '''
            }
        }


# Mock classes for demonstration (would be replaced with actual backend implementations)
class MockSemanticComparator:
    def compare(self, doc_a: str, doc_b: str) -> Dict:
        import random
        return {
            'overall_score': random.uniform(0.7, 0.95),
            'structural_similarity': random.uniform(0.8, 0.98),
            'content_similarity': random.uniform(0.6, 0.9),
            'section_similarities': {
                'Introduction': random.uniform(0.8, 1.0),
                'Terms': random.uniform(0.5, 0.8),
                'Privacy': random.uniform(0.7, 0.9),
                'Dispute Resolution': random.uniform(0.3, 0.7)
            }
        }


class MockLegalChangeDetector:
    def detect_changes(self, doc_a: str, doc_b: str) -> Dict:
        return {
            'impact_score': 0.75,
            'changes': [
                {
                    'category': 'Arbitration Clause',
                    'impact_level': 'High',
                    'confidence': 0.95,
                    'description': 'New binding arbitration clause added',
                    'legal_significance': 'Significantly limits user legal options',
                    'original_text': 'Any disputes will be resolved through negotiation or litigation.',
                    'modified_text': 'Any disputes shall be resolved through binding arbitration.',
                    'recommendations': ['Consider user notification', 'Review regulatory compliance']
                },
                {
                    'category': 'Data Retention',
                    'impact_level': 'Medium',
                    'confidence': 0.88,
                    'description': 'Data retention period reduced',
                    'legal_significance': 'More user-friendly privacy practice',
                    'original_text': 'We retain your personal data for up to 5 years',
                    'modified_text': 'We retain your personal data for up to 3 years',
                    'recommendations': ['Update data management procedures']
                }
            ]
        }


class MockDiffEngine:
    def generate_diff(self, doc_a: str, doc_b: str) -> Dict:
        # Simple diff implementation for demo
        lines_a = doc_a.split('\n')
        lines_b = doc_b.split('\n')
        
        differ = difflib.unified_diff(lines_a, lines_b, lineterm='')
        unified_diff = []
        
        for line in differ:
            if line.startswith('+++') or line.startswith('---'):
                continue
            elif line.startswith('+'):
                unified_diff.append({'type': 'added', 'content': line[1:]})
            elif line.startswith('-'):
                unified_diff.append({'type': 'removed', 'content': line[1:]})
            elif line.startswith('@@'):
                unified_diff.append({'type': 'context', 'content': line})
            else:
                unified_diff.append({'type': 'unchanged', 'content': line})
        
        return {
            'original_lines': [{'type': 'unchanged', 'content': line} for line in lines_a],
            'modified_lines': [{'type': 'unchanged', 'content': line} for line in lines_b],
            'unified_diff': unified_diff,
            'inline_diff': self.create_inline_diff(doc_a, doc_b)
        }
    
    def create_inline_diff(self, doc_a: str, doc_b: str) -> str:
        # Simple inline diff for demo
        return f"""
        <div style="font-family: monospace;">
        <span style="background-color: #f8d7da; text-decoration: line-through;">{doc_a[:100]}...</span>
        <span style="background-color: #d4edda;">{doc_b[:100]}...</span>
        </div>
        """


class MockVersionTracker:
    def __init__(self):
        pass