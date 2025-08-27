"""
Arbitration Detection Demo Workflow
==================================

This module provides an interactive demonstration of the arbitration detection
capabilities including real-time analysis, confidence scoring, and detailed
clause extraction.

Features:
- Real-time document upload and analysis
- Interactive confidence threshold adjustment
- Multi-language document support
- Detailed clause visualization
- Performance metrics tracking
"""

import streamlit as st
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import io

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from app.rag.arbitration_detector import ArbitrationDetector, DetectionResult
    from app.rag.patterns import ArbitrationPatterns
    from app.nlp.multilingual import MultilingualProcessor
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")


class ArbitrationDetectionDemo:
    """Interactive arbitration detection demonstration."""
    
    def __init__(self):
        """Initialize the demo."""
        self.detector = None
        self.multilingual_processor = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize detection components."""
        try:
            self.detector = ArbitrationDetector()
            # self.multilingual_processor = MultilingualProcessor()
        except Exception as e:
            st.error(f"Error initializing components: {e}")
    
    def render_upload_interface(self):
        """Render the document upload and analysis interface."""
        st.markdown("### 📤 Document Upload & Analysis")
        
        # File upload options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload legal documents",
                type=['txt', 'pdf', 'docx'],
                accept_multiple_files=True,
                help="Supported formats: TXT, PDF, DOCX"
            )
        
        with col2:
            st.markdown("#### ⚙️ Analysis Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Minimum confidence score for clause detection"
            )
            
            language = st.selectbox(
                "Document Language",
                ["auto-detect", "english", "spanish", "french", "german", "chinese"],
                help="Select document language or use auto-detection"
            )
        
        # Text input option
        st.markdown("#### ✏️ Or paste document text:")
        text_input = st.text_area(
            "Document text",
            height=200,
            placeholder="Paste your legal document text here...",
            help="You can directly paste document content for quick analysis"
        )
        
        # Analysis button
        if st.button("🔍 Analyze Documents", type="primary"):
            self.analyze_documents(uploaded_files, text_input, confidence_threshold, language)
        
        # Quick demo button with sample documents
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Try Sample: Terms of Service"):
                self.analyze_sample_document("terms_of_service", confidence_threshold)
        with col2:
            if st.button("📄 Try Sample: Privacy Policy"):
                self.analyze_sample_document("privacy_policy", confidence_threshold)
        with col3:
            if st.button("📜 Try Sample: User Agreement"):
                self.analyze_sample_document("user_agreement", confidence_threshold)
    
    def analyze_documents(self, uploaded_files: List, text_input: str, 
                         confidence_threshold: float, language: str):
        """Analyze uploaded documents or text input."""
        if not self.detector:
            st.error("Detection system not initialized")
            return
        
        documents_to_analyze = []
        
        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                try:
                    content = self.extract_text_from_file(file)
                    documents_to_analyze.append({
                        'name': file.name,
                        'content': content,
                        'source': 'upload'
                    })
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
        
        # Process text input
        if text_input.strip():
            documents_to_analyze.append({
                'name': 'Text Input',
                'content': text_input.strip(),
                'source': 'input'
            })
        
        if not documents_to_analyze:
            st.warning("Please upload files or enter text to analyze")
            return
        
        # Analyze documents
        with st.spinner("Analyzing documents..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, doc in enumerate(documents_to_analyze):
                try:
                    # Update threshold in detector
                    self.detector.arbitration_threshold = confidence_threshold
                    
                    # Perform detection
                    result = self.detector.detect(doc['content'], doc['name'])
                    results.append({
                        'document': doc,
                        'result': result
                    })
                    
                    progress_bar.progress((i + 1) / len(documents_to_analyze))
                    
                except Exception as e:
                    st.error(f"Error analyzing {doc['name']}: {e}")
            
            progress_bar.empty()
        
        # Store results in session state
        st.session_state.demo_results['arbitration'] = results
        
        # Display results
        self.display_analysis_results(results)
    
    def analyze_sample_document(self, sample_type: str, confidence_threshold: float):
        """Analyze a sample document."""
        sample_docs = self.get_sample_documents()
        
        if sample_type not in sample_docs:
            st.error(f"Sample document {sample_type} not found")
            return
        
        doc = sample_docs[sample_type]
        
        with st.spinner(f"Analyzing {doc['name']}..."):
            if self.detector:
                self.detector.arbitration_threshold = confidence_threshold
                result = self.detector.detect(doc['content'], doc['name'])
                
                # Store and display result
                analysis_result = [{
                    'document': {'name': doc['name'], 'content': doc['content'], 'source': 'sample'},
                    'result': result
                }]
                
                st.session_state.demo_results['arbitration'] = analysis_result
                self.display_analysis_results(analysis_result)
    
    def extract_text_from_file(self, file) -> str:
        """Extract text content from uploaded file."""
        if file.type == "text/plain":
            return str(file.read(), "utf-8")
        elif file.type == "application/pdf":
            # For demo purposes, return placeholder
            return f"[PDF content extraction not implemented in demo - filename: {file.name}]"
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # For demo purposes, return placeholder
            return f"[DOCX content extraction not implemented in demo - filename: {file.name}]"
        else:
            return str(file.read(), "utf-8")
    
    def display_analysis_results(self, results: List[Dict]):
        """Display analysis results."""
        st.markdown("## 📊 Analysis Results")
        
        if not results:
            st.info("No analysis results to display")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_docs = len(results)
        docs_with_arbitration = sum(1 for r in results if r['result'].has_arbitration)
        avg_confidence = sum(r['result'].confidence for r in results) / total_docs if total_docs > 0 else 0
        total_clauses = sum(len(r['result'].clauses) for r in results)
        
        with col1:
            st.metric("Documents Analyzed", total_docs)
        with col2:
            st.metric("Arbitration Detected", docs_with_arbitration)
        with col3:
            st.metric("Average Confidence", f"{avg_confidence:.2%}")
        with col4:
            st.metric("Total Clauses Found", total_clauses)
        
        # Individual document results
        for i, analysis in enumerate(results):
            doc = analysis['document']
            result = analysis['result']
            
            with st.expander(f"📄 {doc['name']} - {'✅ Arbitration Detected' if result.has_arbitration else '❌ No Arbitration'}", 
                           expanded=True):
                
                # Result summary
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 📈 Detection Summary")
                    
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result.confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Score"},
                        delta={'reference': 80},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Summary")
                    summary = result.summary
                    
                    st.write(f"**Binding Arbitration:** {'Yes' if summary.get('has_binding_arbitration') else 'No'}")
                    st.write(f"**Class Action Waiver:** {'Yes' if summary.get('has_class_action_waiver') else 'No'}")
                    st.write(f"**Jury Trial Waiver:** {'Yes' if summary.get('has_jury_trial_waiver') else 'No'}")
                    st.write(f"**Opt-out Available:** {'Yes' if summary.get('has_opt_out') else 'No'}")
                    
                    if summary.get('arbitration_provider'):
                        st.write(f"**Provider:** {summary['arbitration_provider']}")
                    
                    st.write(f"**Processing Time:** {result.processing_time:.3f}s")
                
                # Detected clauses
                if result.clauses:
                    st.markdown("#### 📝 Detected Clauses")
                    
                    for j, clause in enumerate(result.clauses):
                        with st.container():
                            st.markdown(f"**Clause {j+1}** (Confidence: {clause.confidence_score:.2%})")
                            
                            # Clause details
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Highlight clause text
                                st.markdown(f"```\n{clause.text}\n```")
                            
                            with col2:
                                st.write(f"**Type:** {clause.arbitration_type.value}")
                                st.write(f"**Categories:** {', '.join([ct.value for ct in clause.clause_types])}")
                                st.write(f"**Location:** {clause.location['start_char']}-{clause.location['end_char']}")
                                
                                # Score breakdown
                                score_data = pd.DataFrame({
                                    'Score Type': ['Pattern', 'Semantic', 'Keyword'],
                                    'Score': [clause.keyword_score, clause.semantic_score, clause.keyword_score]
                                })
                                
                                fig = px.bar(score_data, x='Score Type', y='Score', 
                                           title="Score Breakdown")
                                fig.update_layout(height=200)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("---")
                else:
                    st.info("No arbitration clauses detected in this document")
                
                # Document preview
                if st.checkbox(f"Show document preview - {doc['name']}", key=f"preview_{i}"):
                    st.markdown("#### 📄 Document Preview")
                    preview_text = doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content']
                    st.text_area("Document content", preview_text, height=200, disabled=True)
    
    def render_results_dashboard(self):
        """Render the results dashboard."""
        st.markdown("### 📊 Results Dashboard")
        
        if 'arbitration' not in st.session_state.demo_results:
            st.info("No analysis results available. Please run an analysis first.")
            return
        
        results = st.session_state.demo_results['arbitration']
        
        # Performance metrics
        st.markdown("#### ⚡ Performance Metrics")
        
        processing_times = [r['result'].processing_time for r in results]
        confidence_scores = [r['result'].confidence for r in results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time chart
            fig = px.histogram(x=processing_times, title="Processing Time Distribution",
                             labels={'x': 'Processing Time (seconds)', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = px.histogram(x=confidence_scores, title="Confidence Score Distribution",
                             labels={'x': 'Confidence Score', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Clause type analysis
        st.markdown("#### 📈 Clause Type Analysis")
        
        clause_types_data = []
        for result in results:
            for clause in result['result'].clauses:
                for clause_type in clause.clause_types:
                    clause_types_data.append({
                        'Document': result['document']['name'],
                        'Clause Type': clause_type.value,
                        'Confidence': clause.confidence_score
                    })
        
        if clause_types_data:
            df = pd.DataFrame(clause_types_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Clause type frequency
                type_counts = df['Clause Type'].value_counts()
                fig = px.bar(x=type_counts.index, y=type_counts.values,
                           title="Clause Type Frequency")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average confidence by type
                avg_confidence = df.groupby('Clause Type')['Confidence'].mean()
                fig = px.bar(x=avg_confidence.index, y=avg_confidence.values,
                           title="Average Confidence by Clause Type")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No clause data available for analysis")
    
    def render_configuration(self):
        """Render the configuration interface."""
        st.markdown("### ⚙️ Detection Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Detection Thresholds")
            
            arbitration_threshold = st.slider(
                "Arbitration Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Minimum confidence score to classify as arbitration clause"
            )
            
            high_confidence_threshold = st.slider(
                "High Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Threshold for high-confidence classifications"
            )
            
            clause_merge_threshold = st.number_input(
                "Clause Merge Distance (characters)",
                min_value=0,
                max_value=500,
                value=100,
                help="Maximum distance between clauses to merge them"
            )
        
        with col2:
            st.markdown("#### 🔍 Pattern Matching")
            
            if st.button("View Arbitration Patterns"):
                self.show_arbitration_patterns()
            
            st.markdown("#### 🌐 Language Support")
            
            supported_languages = [
                "English", "Spanish", "French", "German", 
                "Chinese", "Japanese", "Portuguese", "Italian"
            ]
            
            for lang in supported_languages:
                st.write(f"✅ {lang}")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.markdown("#### Embedding Configuration")
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
                help="Choose the embedding model for semantic analysis"
            )
            
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=1000,
                value=512,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=50,
                help="Overlap between consecutive chunks"
            )
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            config = {
                'arbitration_threshold': arbitration_threshold,
                'high_confidence_threshold': high_confidence_threshold,
                'clause_merge_threshold': clause_merge_threshold,
                'embedding_model': embedding_model,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            
            st.session_state.arbitration_config = config
            st.success("Configuration saved successfully!")
    
    def show_arbitration_patterns(self):
        """Display arbitration patterns information."""
        st.markdown("#### 📋 Arbitration Detection Patterns")
        
        patterns_info = {
            "High Confidence Keywords": [
                "binding arbitration", "mandatory arbitration", "arbitration agreement",
                "agree to arbitrate", "submit to arbitration", "resolve by arbitration"
            ],
            "Medium Confidence Keywords": [
                "dispute resolution", "alternative dispute resolution", "adr",
                "arbitrator", "arbitration rules", "arbitration proceedings"
            ],
            "Negative Indicators": [
                "litigation", "court proceedings", "jury trial",
                "class action", "small claims court"
            ]
        }
        
        for category, patterns in patterns_info.items():
            st.markdown(f"**{category}:**")
            for pattern in patterns:
                st.write(f"• {pattern}")
            st.write("")
    
    def get_sample_documents(self) -> Dict[str, Dict]:
        """Get sample documents for demonstration."""
        return {
            'terms_of_service': {
                'name': 'Sample Terms of Service',
                'content': '''
                TERMS OF SERVICE AGREEMENT

                1. ACCEPTANCE OF TERMS
                By accessing and using this service, you accept and agree to be bound by the terms and provision of this agreement.

                15. DISPUTE RESOLUTION
                Any dispute, claim or controversy arising out of or relating to this Agreement or the breach, termination, enforcement, interpretation or validity thereof, including the determination of the scope or applicability of this agreement to arbitrate, shall be determined by arbitration in New York, New York before one arbitrator. The arbitration shall be administered by JAMS pursuant to its Comprehensive Arbitration Rules and Procedures. Judgment on the Award may be entered in any court having jurisdiction. This clause shall not preclude parties from seeking provisional remedies in aid of arbitration from a court of appropriate jurisdiction.

                YOU AGREE THAT BY ENTERING INTO THIS AGREEMENT, YOU AND COMPANY ARE EACH WAIVING THE RIGHT TO A JURY TRIAL OR TO PARTICIPATE IN A CLASS ACTION.

                16. GOVERNING LAW
                This agreement shall be governed by and construed in accordance with the laws of the State of New York.
                '''
            },
            'privacy_policy': {
                'name': 'Sample Privacy Policy',
                'content': '''
                PRIVACY POLICY

                1. INFORMATION WE COLLECT
                We collect information you provide directly to us, such as when you create an account, make a purchase, or contact us.

                2. HOW WE USE YOUR INFORMATION
                We use the information we collect to provide, maintain, and improve our services.

                10. CONTACT US
                If you have any questions about this Privacy Policy, please contact us at privacy@company.com or by mail at:
                Company Name
                123 Main Street
                City, State 12345

                This policy does not contain arbitration clauses as it is focused on privacy practices rather than dispute resolution.
                '''
            },
            'user_agreement': {
                'name': 'Sample User Agreement',
                'content': '''
                USER AGREEMENT

                1. ELIGIBILITY
                You must be at least 18 years old to use this service.

                12. BINDING ARBITRATION AND CLASS ACTION WAIVER
                PLEASE READ THIS SECTION CAREFULLY – IT MAY SIGNIFICANTLY AFFECT YOUR LEGAL RIGHTS, INCLUDING YOUR RIGHT TO FILE A LAWSUIT IN COURT AND TO HAVE A JURY HEAR YOUR CLAIMS.

                All claims and disputes arising under or relating to this Agreement are to be settled by binding arbitration in the state of California. The arbitration shall be conducted on a confidential basis pursuant to the Commercial Arbitration Rules of the American Arbitration Association. Any decision or award as a result of any such arbitration proceeding shall be in writing and shall provide an explanation for all conclusions of law and fact and shall include the assessment of costs, expenses, and reasonable attorneys' fees.

                You and Company agree that each may bring claims against the other only in your or its individual capacity, and not as a plaintiff or class member in any purported class or representative proceeding.

                OPT-OUT: You have the right to opt-out and not be bound by the arbitration and class action waiver provisions set forth above by sending written notice of your decision to opt-out to the following address: Company Legal Department, 456 Oak Street, City, State 67890. The notice must be sent within 30 days of your first use of the service.
                '''
            }
        }