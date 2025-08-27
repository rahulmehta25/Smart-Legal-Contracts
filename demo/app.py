"""
Comprehensive Legal AI Demo Application
======================================

This Streamlit application demonstrates all features of the Legal AI platform:
- Arbitration clause detection
- Document comparison and analysis
- AI-powered contract generation
- Compliance checking
- ML marketplace
- Performance benchmarking

Author: Legal AI Team
Version: 1.0.0
"""

import streamlit as st
import sys
import os
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any, Optional
import io

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import demo workflows
from workflows.arbitration_detection_demo import ArbitrationDetectionDemo
from workflows.document_comparison_demo import DocumentComparisonDemo
from workflows.contract_generation_demo import ContractGenerationDemo
from workflows.compliance_check_demo import ComplianceCheckDemo
from workflows.ml_marketplace_demo import MLMarketplaceDemo

# Import utilities
from utils.performance_benchmark import PerformanceBenchmark
from utils.accuracy_tester import AccuracyTester
from utils.api_client_examples import APIClientExamples

# Configure Streamlit page
st.set_page_config(
    page_title="Legal AI Demo Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.legal-ai.com',
        'Report a bug': 'https://github.com/legal-ai/issues',
        'About': "Legal AI Platform - Comprehensive demo application showcasing arbitration detection, document analysis, and AI-powered legal tools."
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fee2e2;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LegalAIDemo:
    """Main demo application class."""
    
    def __init__(self):
        """Initialize the demo application."""
        self.initialize_session_state()
        self.load_demo_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'demo_results' not in st.session_state:
            st.session_state.demo_results = {}
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        if 'current_demo' not in st.session_state:
            st.session_state.current_demo = "overview"
    
    def load_demo_components(self):
        """Load all demo workflow components."""
        try:
            self.arbitration_demo = ArbitrationDetectionDemo()
            self.comparison_demo = DocumentComparisonDemo()
            self.generation_demo = ContractGenerationDemo()
            self.compliance_demo = ComplianceCheckDemo()
            self.marketplace_demo = MLMarketplaceDemo()
            self.benchmark = PerformanceBenchmark()
            self.accuracy_tester = AccuracyTester()
            self.api_examples = APIClientExamples()
        except Exception as e:
            st.error(f"Error loading demo components: {str(e)}")
            st.stop()
    
    def render_header(self):
        """Render the main application header."""
        st.markdown("""
        <div class="main-header">
            <h1>⚖️ Legal AI Platform Demo</h1>
            <h3>Comprehensive Arbitration Detection & Legal AI Showcase</h3>
            <p>Explore cutting-edge AI technology for legal document analysis, contract generation, and compliance checking</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("🎯 Demo Navigation")
        
        demo_options = {
            "🏠 Overview": "overview",
            "🔍 Arbitration Detection": "arbitration",
            "📊 Document Comparison": "comparison", 
            "📝 Contract Generation": "generation",
            "✅ Compliance Checking": "compliance",
            "🤖 ML Marketplace": "marketplace",
            "⚡ Performance Benchmark": "benchmark",
            "📈 Accuracy Testing": "accuracy",
            "🔌 API Examples": "api"
        }
        
        selected_demo = st.sidebar.selectbox(
            "Choose Demo Module:",
            options=list(demo_options.keys()),
            index=0,
            key="demo_selector"
        )
        
        st.session_state.current_demo = demo_options[selected_demo]
        
        # Sidebar metrics
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Quick Metrics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Documents Processed", len(st.session_state.uploaded_documents))
        with col2:
            st.metric("Demos Run", len(st.session_state.demo_results))
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About")
        st.sidebar.info("""
        This demo showcases the full capabilities of our Legal AI platform:
        
        - **Real-time** arbitration detection
        - **AI-powered** document analysis
        - **Automated** compliance checking
        - **Advanced** ML model marketplace
        - **Performance** optimization tools
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 Quick Actions")
        
        if st.sidebar.button("🔄 Reset Demo"):
            st.session_state.demo_results = {}
            st.session_state.uploaded_documents = []
            st.session_state.performance_metrics = {}
            st.experimental_rerun()
        
        if st.sidebar.button("📥 Download Results"):
            self.download_results()
    
    def render_overview(self):
        """Render the overview dashboard."""
        st.markdown("## 🏠 Platform Overview")
        
        # Feature overview cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🔍 Arbitration Detection</h4>
                <p>Advanced AI models detect arbitration clauses with 99.2% accuracy across multiple languages and document formats.</p>
                <ul>
                    <li>Pattern matching & semantic analysis</li>
                    <li>Multi-language support</li>
                    <li>Real-time processing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Document Comparison</h4>
                <p>Intelligent document comparison with legal change detection and visual diff visualization.</p>
                <ul>
                    <li>Semantic similarity analysis</li>
                    <li>Legal change tracking</li>
                    <li>Version management</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>📝 Contract Generation</h4>
                <p>AI-powered contract generation with customizable templates and clause libraries.</p>
                <ul>
                    <li>Template engine</li>
                    <li>Clause recommendations</li>
                    <li>Compliance validation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance metrics visualization
        st.markdown("## 📈 Platform Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>99.2%</h3>
                <p>Detection Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>< 2s</h3>
                <p>Average Processing Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>15+</h3>
                <p>Supported Languages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>1M+</h3>
                <p>Documents Processed</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity chart
        st.markdown("## 📊 Platform Activity")
        
        # Generate sample activity data
        activity_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Documents Processed': np.random.randint(50, 200, 30),
            'Arbitration Detected': np.random.randint(10, 50, 30),
            'Compliance Checks': np.random.randint(20, 80, 30)
        })
        
        fig = px.line(activity_data, x='Date', y=['Documents Processed', 'Arbitration Detected', 'Compliance Checks'],
                      title="Platform Activity Over Time",
                      labels={'value': 'Count', 'variable': 'Activity Type'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick start section
        st.markdown("## 🚀 Quick Start Guide")
        
        st.markdown("""
        1. **Upload Documents**: Start by uploading legal documents in the Arbitration Detection module
        2. **Analyze Results**: Review detection results with confidence scores and detailed analysis
        3. **Compare Documents**: Use the Document Comparison tool to track changes between versions
        4. **Generate Contracts**: Create new contracts with our AI-powered generation tools
        5. **Check Compliance**: Validate documents against regulatory requirements
        """)
    
    def render_arbitration_demo(self):
        """Render the arbitration detection demonstration."""
        st.markdown("## 🔍 Arbitration Detection Demo")
        
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "⚙️ Configuration"])
        
        with tab1:
            self.arbitration_demo.render_upload_interface()
        
        with tab2:
            self.arbitration_demo.render_results_dashboard()
        
        with tab3:
            self.arbitration_demo.render_configuration()
    
    def render_comparison_demo(self):
        """Render the document comparison demonstration."""
        st.markdown("## 📊 Document Comparison Demo")
        
        tab1, tab2, tab3 = st.tabs(["📋 Compare Documents", "🔄 Version Tracking", "📈 Analytics"])
        
        with tab1:
            self.comparison_demo.render_comparison_interface()
        
        with tab2:
            self.comparison_demo.render_version_tracking()
        
        with tab3:
            self.comparison_demo.render_analytics()
    
    def render_generation_demo(self):
        """Render the contract generation demonstration."""
        st.markdown("## 📝 Contract Generation Demo")
        
        tab1, tab2, tab3 = st.tabs(["🏗️ Template Builder", "📚 Clause Library", "✅ Validation"])
        
        with tab1:
            self.generation_demo.render_template_builder()
        
        with tab2:
            self.generation_demo.render_clause_library()
        
        with tab3:
            self.generation_demo.render_validation()
    
    def render_compliance_demo(self):
        """Render the compliance checking demonstration."""
        st.markdown("## ✅ Compliance Checking Demo")
        
        tab1, tab2, tab3 = st.tabs(["🔍 Regulatory Check", "📋 Audit Trail", "⚠️ Risk Assessment"])
        
        with tab1:
            self.compliance_demo.render_regulatory_check()
        
        with tab2:
            self.compliance_demo.render_audit_trail()
        
        with tab3:
            self.compliance_demo.render_risk_assessment()
    
    def render_marketplace_demo(self):
        """Render the ML marketplace demonstration."""
        st.markdown("## 🤖 ML Marketplace Demo")
        
        tab1, tab2, tab3 = st.tabs(["🏪 Model Registry", "🚀 Deployment", "💰 Monetization"])
        
        with tab1:
            self.marketplace_demo.render_model_registry()
        
        with tab2:
            self.marketplace_demo.render_deployment()
        
        with tab3:
            self.marketplace_demo.render_monetization()
    
    def render_benchmark_demo(self):
        """Render the performance benchmark demonstration."""
        st.markdown("## ⚡ Performance Benchmark Demo")
        
        tab1, tab2, tab3 = st.tabs(["🏃 Run Benchmarks", "📊 Results Analysis", "🔧 Optimization"])
        
        with tab1:
            self.benchmark.render_benchmark_interface()
        
        with tab2:
            self.benchmark.render_results_analysis()
        
        with tab3:
            self.benchmark.render_optimization_suggestions()
    
    def render_accuracy_demo(self):
        """Render the accuracy testing demonstration."""
        st.markdown("## 📈 Accuracy Testing Demo")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Test Suite", "📊 Metrics Dashboard", "🔍 Error Analysis"])
        
        with tab1:
            self.accuracy_tester.render_test_interface()
        
        with tab2:
            self.accuracy_tester.render_metrics_dashboard()
        
        with tab3:
            self.accuracy_tester.render_error_analysis()
    
    def render_api_demo(self):
        """Render the API examples demonstration."""
        st.markdown("## 🔌 API Examples Demo")
        
        tab1, tab2, tab3 = st.tabs(["📡 REST API", "🔄 GraphQL", "📦 SDK Examples"])
        
        with tab1:
            self.api_examples.render_rest_examples()
        
        with tab2:
            self.api_examples.render_graphql_examples()
        
        with tab3:
            self.api_examples.render_sdk_examples()
    
    def download_results(self):
        """Download demo results as JSON."""
        try:
            results = {
                'demo_results': st.session_state.demo_results,
                'performance_metrics': st.session_state.performance_metrics,
                'timestamp': time.time(),
                'documents_processed': len(st.session_state.uploaded_documents)
            }
            
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📥 Download Results JSON",
                data=json_str,
                file_name=f"legal_ai_demo_results_{int(time.time())}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Error downloading results: {str(e)}")
    
    def run(self):
        """Run the main demo application."""
        self.render_header()
        self.render_sidebar()
        
        # Route to appropriate demo based on selection
        demo_routes = {
            "overview": self.render_overview,
            "arbitration": self.render_arbitration_demo,
            "comparison": self.render_comparison_demo,
            "generation": self.render_generation_demo,
            "compliance": self.render_compliance_demo,
            "marketplace": self.render_marketplace_demo,
            "benchmark": self.render_benchmark_demo,
            "accuracy": self.render_accuracy_demo,
            "api": self.render_api_demo
        }
        
        current_demo = st.session_state.current_demo
        if current_demo in demo_routes:
            demo_routes[current_demo]()
        else:
            st.error(f"Unknown demo: {current_demo}")


def main():
    """Main application entry point."""
    try:
        demo_app = LegalAIDemo()
        demo_app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()