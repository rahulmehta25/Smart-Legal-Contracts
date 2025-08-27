"""
Contract Generation Demo Workflow
=================================

This module provides an interactive demonstration of AI-powered contract
generation capabilities including template building, clause libraries,
and automated validation.

Features:
- Interactive template builder with drag-and-drop
- Comprehensive clause library with recommendations
- AI-powered content generation
- Real-time validation and compliance checking
- Multi-format export (PDF, DOCX, HTML)
- Version control and collaboration features
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
import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from app.ai.contract_generator import ContractGenerator
    from app.legal.document_generator import DocumentGenerator
    from app.compliance.auto_checker import ComplianceChecker
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")


class ContractGenerationDemo:
    """Interactive contract generation demonstration."""
    
    def __init__(self):
        """Initialize the demo."""
        self.contract_generator = None
        self.document_generator = None
        self.compliance_checker = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize generation components."""
        try:
            # Mock implementations for demo
            self.contract_generator = MockContractGenerator()
            self.document_generator = MockDocumentGenerator()
            self.compliance_checker = MockComplianceChecker()
        except Exception as e:
            st.error(f"Error initializing components: {e}")
    
    def render_template_builder(self):
        """Render the template builder interface."""
        st.markdown("### 🏗️ Contract Template Builder")
        
        # Template selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📋 Template Selection")
            
            template_categories = ["Terms of Service", "Privacy Policy", "Software License", 
                                 "Employment Agreement", "Service Agreement", "NDA"]
            
            selected_category = st.selectbox(
                "Contract Category",
                template_categories,
                help="Choose the type of contract to generate"
            )
            
            # Get templates for selected category
            templates = self.get_templates_for_category(selected_category)
            
            selected_template = st.selectbox(
                "Base Template",
                templates,
                help="Choose a base template to customize"
            )
        
        with col2:
            st.markdown("#### ⚙️ Generation Settings")
            
            jurisdiction = st.selectbox(
                "Jurisdiction",
                ["United States", "European Union", "United Kingdom", "Canada", "Australia"],
                help="Legal jurisdiction for compliance"
            )
            
            complexity_level = st.selectbox(
                "Complexity Level",
                ["Basic", "Standard", "Advanced", "Enterprise"],
                index=1,
                help="Complexity of legal language and clauses"
            )
            
            language = st.selectbox(
                "Language",
                ["English", "Spanish", "French", "German", "Portuguese"],
                help="Contract language"
            )
        
        # Template customization
        st.markdown("#### 🎨 Template Customization")
        
        tab1, tab2, tab3 = st.tabs(["📝 Basic Info", "🧩 Clause Selection", "🎯 AI Configuration"])
        
        with tab1:
            self.render_basic_info_form()
        
        with tab2:
            self.render_clause_selection()
        
        with tab3:
            self.render_ai_configuration()
        
        # Generate button
        if st.button("🚀 Generate Contract", type="primary", use_container_width=True):
            self.generate_contract(selected_category, selected_template, jurisdiction, 
                                 complexity_level, language)
    
    def render_basic_info_form(self):
        """Render basic contract information form."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Company Name", placeholder="Your Company Inc.")
            st.text_input("Company Address", placeholder="123 Main St, City, State")
            st.text_input("Contact Email", placeholder="legal@company.com")
            st.date_input("Effective Date", datetime.date.today())
        
        with col2:
            st.text_input("Contract Title", placeholder="Terms of Service Agreement")
            st.text_input("Version", placeholder="1.0", value="1.0")
            st.selectbox("Contract Type", ["B2C", "B2B", "Internal"])
            st.date_input("Expiration Date", datetime.date.today() + datetime.timedelta(days=365))
        
        # Additional fields
        st.text_area("Additional Requirements", 
                    placeholder="Specify any additional requirements or special clauses needed...",
                    height=100)
    
    def render_clause_selection(self):
        """Render clause selection interface."""
        st.markdown("##### 🧩 Available Clauses")
        
        clause_categories = {
            "🔐 Data & Privacy": [
                "Data Collection", "Data Usage", "Data Retention", "User Rights", 
                "Cookie Policy", "GDPR Compliance", "CCPA Compliance"
            ],
            "⚖️ Legal & Disputes": [
                "Arbitration Clause", "Jurisdiction", "Governing Law", "Limitation of Liability",
                "Indemnification", "Force Majeure", "Severability"
            ],
            "👤 User Obligations": [
                "Acceptable Use", "User Conduct", "Account Responsibility", 
                "Age Restrictions", "Registration Requirements"
            ],
            "💼 Service Terms": [
                "Service Description", "Availability", "Modifications", "Termination",
                "Refund Policy", "Payment Terms", "Intellectual Property"
            ]
        }
        
        selected_clauses = {}
        
        for category, clauses in clause_categories.items():
            st.markdown(f"**{category}**")
            
            cols = st.columns(3)
            for i, clause in enumerate(clauses):
                with cols[i % 3]:
                    selected = st.checkbox(clause, value=True, key=f"clause_{clause}")
                    selected_clauses[clause] = selected
        
        # Store selected clauses in session state
        st.session_state.selected_clauses = selected_clauses
        
        # Clause recommendations
        st.markdown("##### 💡 AI Recommendations")
        
        recommendations = self.get_clause_recommendations()
        
        for rec in recommendations:
            st.info(f"💡 **Recommended:** {rec['clause']} - {rec['reason']}")
    
    def render_ai_configuration(self):
        """Render AI configuration options."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🤖 AI Generation Settings")
            
            tone = st.selectbox(
                "Contract Tone",
                ["Professional", "Friendly", "Formal", "Conversational"],
                index=0,
                help="Tone of the generated language"
            )
            
            detail_level = st.slider(
                "Detail Level",
                min_value=1,
                max_value=5,
                value=3,
                help="Level of detail in clause explanations"
            )
            
            include_examples = st.checkbox(
                "Include Examples",
                value=True,
                help="Include practical examples in clauses"
            )
        
        with col2:
            st.markdown("##### 🎯 Customization Options")
            
            industry_focus = st.selectbox(
                "Industry Focus",
                ["Technology", "Healthcare", "Finance", "E-commerce", "Education", "General"],
                help="Tailor content to specific industry"
            )
            
            user_type = st.selectbox(
                "Primary User Type",
                ["Consumers", "Businesses", "Developers", "Partners"],
                help="Target audience for the contract"
            )
            
            compliance_strictness = st.slider(
                "Compliance Strictness",
                min_value=1,
                max_value=5,
                value=4,
                help="How strict compliance requirements should be"
            )
    
    def generate_contract(self, category: str, template: str, jurisdiction: str, 
                         complexity: str, language: str):
        """Generate contract based on user inputs."""
        with st.spinner("🤖 Generating contract with AI..."):
            # Simulate AI generation
            time.sleep(2)
            
            # Get selected clauses
            selected_clauses = st.session_state.get('selected_clauses', {})
            
            # Generate contract content
            contract_content = self.contract_generator.generate(
                category=category,
                template=template,
                jurisdiction=jurisdiction,
                complexity=complexity,
                language=language,
                selected_clauses=selected_clauses
            )
            
            # Store generated contract
            st.session_state.generated_contract = {
                'content': contract_content,
                'metadata': {
                    'category': category,
                    'template': template,
                    'jurisdiction': jurisdiction,
                    'complexity': complexity,
                    'language': language,
                    'generated_at': time.time()
                }
            }
        
        # Display generated contract
        self.display_generated_contract(contract_content)
    
    def display_generated_contract(self, contract_content: Dict):
        """Display the generated contract."""
        st.markdown("## 📄 Generated Contract")
        
        # Contract preview
        tab1, tab2, tab3 = st.tabs(["📖 Preview", "📊 Analysis", "⬇️ Export"])
        
        with tab1:
            self.render_contract_preview(contract_content)
        
        with tab2:
            self.render_contract_analysis(contract_content)
        
        with tab3:
            self.render_export_options(contract_content)
    
    def render_contract_preview(self, contract_content: Dict):
        """Render contract preview."""
        st.markdown("### 📖 Contract Preview")
        
        # Contract sections
        for section in contract_content['sections']:
            with st.expander(f"📄 {section['title']}", expanded=True):
                st.markdown(section['content'])
                
                if section.get('ai_generated'):
                    st.info("🤖 This section was AI-generated")
                
                # Edit button for each section
                if st.button(f"✏️ Edit {section['title']}", key=f"edit_{section['title']}"):
                    self.edit_section(section)
        
        # Overall contract metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Word Count", contract_content['word_count'])
        with col2:
            st.metric("Sections", len(contract_content['sections']))
        with col3:
            st.metric("Readability Score", f"{contract_content['readability_score']}/100")
        with col4:
            st.metric("Compliance Score", f"{contract_content['compliance_score']:.1%}")
    
    def render_contract_analysis(self, contract_content: Dict):
        """Render contract analysis."""
        st.markdown("### 📊 Contract Analysis")
        
        # Compliance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Compliance Check")
            
            compliance_results = contract_content['compliance_analysis']
            
            for jurisdiction, result in compliance_results.items():
                status_icon = "✅" if result['compliant'] else "⚠️"
                st.write(f"{status_icon} **{jurisdiction}**: {result['score']:.1%}")
                
                if result['issues']:
                    for issue in result['issues']:
                        st.write(f"  • {issue}")
        
        with col2:
            st.markdown("#### 📈 Quality Metrics")
            
            # Quality metrics visualization
            metrics_data = pd.DataFrame({
                'Metric': ['Clarity', 'Completeness', 'Legal Strength', 'User Friendliness'],
                'Score': [85, 92, 88, 76]
            })
            
            fig = px.bar(metrics_data, x='Metric', y='Score', 
                        title="Contract Quality Metrics")
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        st.markdown("#### ⚠️ Risk Assessment")
        
        risks = contract_content['risk_assessment']
        
        for risk in risks:
            severity_color = {
                'Low': 'green',
                'Medium': 'orange', 
                'High': 'red'
            }[risk['severity']]
            
            st.markdown(f"**{risk['category']}** "
                       f"<span style='color: {severity_color}'>[{risk['severity']}]</span>", 
                       unsafe_allow_html=True)
            st.write(f"• {risk['description']}")
            if risk['mitigation']:
                st.write(f"  💡 *Mitigation*: {risk['mitigation']}")
            st.write("")
    
    def render_export_options(self, contract_content: Dict):
        """Render export options."""
        st.markdown("### ⬇️ Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📄 Document Formats")
            
            if st.button("📄 Export as PDF", use_container_width=True):
                self.export_contract("pdf", contract_content)
            
            if st.button("📝 Export as DOCX", use_container_width=True):
                self.export_contract("docx", contract_content)
            
            if st.button("🌐 Export as HTML", use_container_width=True):
                self.export_contract("html", contract_content)
        
        with col2:
            st.markdown("#### 📋 Data Formats")
            
            if st.button("📊 Export as JSON", use_container_width=True):
                self.export_contract("json", contract_content)
            
            if st.button("📈 Export Analysis", use_container_width=True):
                self.export_analysis(contract_content)
            
            if st.button("📧 Email Contract", use_container_width=True):
                self.email_contract(contract_content)
        
        with col3:
            st.markdown("#### 🔗 Integration")
            
            if st.button("📤 Send to DocuSign", use_container_width=True):
                st.info("DocuSign integration would be implemented here")
            
            if st.button("💾 Save to Cloud", use_container_width=True):
                st.info("Cloud storage integration would be implemented here")
            
            if st.button("🔄 Version Control", use_container_width=True):
                st.info("Version control system would be implemented here")
    
    def render_clause_library(self):
        """Render the clause library interface."""
        st.markdown("### 📚 Clause Library")
        
        # Library navigation
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### 🗂️ Categories")
            
            categories = [
                "All Clauses", "Privacy & Data", "Terms & Conditions", 
                "Liability & Risk", "Dispute Resolution", "Intellectual Property",
                "Payment & Billing", "Termination", "Custom Clauses"
            ]
            
            selected_category = st.selectbox("Category", categories, key="clause_category")
            
            # Search and filters
            search_term = st.text_input("🔍 Search clauses", placeholder="Enter keywords...")
            
            st.markdown("**Filters:**")
            jurisdiction_filter = st.multiselect("Jurisdiction", 
                                                ["US", "EU", "UK", "CA", "AU"])
            complexity_filter = st.selectbox("Complexity", 
                                           ["All", "Basic", "Standard", "Advanced"])
        
        with col2:
            st.markdown("#### 📋 Available Clauses")
            
            # Get clauses for selected category
            clauses = self.get_clauses_for_category(selected_category, search_term, 
                                                  jurisdiction_filter, complexity_filter)
            
            # Display clauses
            for clause in clauses:
                with st.expander(f"📄 {clause['title']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Description:**")
                        st.write(clause['description'])
                        
                        st.markdown("**Sample Text:**")
                        st.text_area("", clause['sample_text'], height=100, disabled=True, 
                                   key=f"sample_{clause['id']}")
                    
                    with col2:
                        st.write(f"**Category:** {clause['category']}")
                        st.write(f"**Complexity:** {clause['complexity']}")
                        st.write(f"**Jurisdictions:** {', '.join(clause['jurisdictions'])}")
                        st.write(f"**Rating:** {'⭐' * clause['rating']}")
                        
                        if st.button("➕ Add to Contract", key=f"add_{clause['id']}"):
                            self.add_clause_to_contract(clause)
                        
                        if st.button("📝 Customize", key=f"customize_{clause['id']}"):
                            self.customize_clause(clause)
        
        # Clause suggestions
        st.markdown("#### 💡 AI-Powered Suggestions")
        
        if st.button("🤖 Get Clause Suggestions"):
            suggestions = self.get_ai_clause_suggestions()
            
            for suggestion in suggestions:
                st.info(f"💡 **Suggested:** {suggestion['title']} - {suggestion['reason']}")
    
    def render_validation(self):
        """Render the validation interface."""
        st.markdown("### ✅ Contract Validation")
        
        if 'generated_contract' not in st.session_state:
            st.info("Please generate a contract first to run validation.")
            return
        
        contract_content = st.session_state.generated_contract['content']
        
        # Validation options
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Validation Settings")
            
            validation_types = st.multiselect(
                "Validation Types",
                ["Legal Compliance", "Grammar & Style", "Readability", "Completeness", "Risk Assessment"],
                default=["Legal Compliance", "Completeness"]
            )
            
            strict_mode = st.checkbox("Strict Mode", value=False,
                                    help="Enable stricter validation rules")
            
            if st.button("🔍 Run Validation", type="primary"):
                self.run_validation(contract_content, validation_types, strict_mode)
        
        with col2:
            if 'validation_results' in st.session_state:
                self.display_validation_results(st.session_state.validation_results)
    
    def run_validation(self, contract_content: Dict, validation_types: List[str], 
                      strict_mode: bool):
        """Run contract validation."""
        with st.spinner("🔍 Running validation..."):
            time.sleep(1)
            
            # Mock validation results
            validation_results = {
                'overall_score': 0.85,
                'passed': True,
                'issues': [
                    {
                        'type': 'Legal Compliance',
                        'severity': 'Medium',
                        'description': 'Consider adding explicit GDPR compliance clause for EU users',
                        'location': 'Section 5: Privacy Policy',
                        'suggestion': 'Add: "For users in the European Union, we comply with GDPR..."'
                    },
                    {
                        'type': 'Readability',
                        'severity': 'Low',
                        'description': 'Some sentences exceed recommended length for legal documents',
                        'location': 'Section 3: Terms of Use',
                        'suggestion': 'Break down complex sentences into shorter, clearer statements'
                    }
                ],
                'summary': {
                    'total_checks': 25,
                    'passed_checks': 21,
                    'warnings': 3,
                    'errors': 1
                }
            }
            
            st.session_state.validation_results = validation_results
    
    def display_validation_results(self, results: Dict):
        """Display validation results."""
        st.markdown("#### ✅ Validation Results")
        
        # Overall score
        score = results['overall_score']
        score_color = 'green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red'
        
        st.markdown(f"**Overall Score:** "
                   f"<span style='color: {score_color}; font-size: 1.5em'>{score:.1%}</span>",
                   unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        summary = results['summary']
        with col1:
            st.metric("Total Checks", summary['total_checks'])
        with col2:
            st.metric("Passed", summary['passed_checks'])
        with col3:
            st.metric("Warnings", summary['warnings'])
        with col4:
            st.metric("Errors", summary['errors'])
        
        # Issues
        if results['issues']:
            st.markdown("#### ⚠️ Issues Found")
            
            for issue in results['issues']:
                severity_color = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }[issue['severity']]
                
                with st.expander(f"{issue['type']} - {issue['severity']} Severity", expanded=True):
                    st.write(f"**Description:** {issue['description']}")
                    st.write(f"**Location:** {issue['location']}")
                    
                    if issue.get('suggestion'):
                        st.info(f"💡 **Suggestion:** {issue['suggestion']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Accept Suggestion", key=f"accept_{issue['type']}"):
                            st.success("Suggestion applied!")
                    with col2:
                        if st.button(f"❌ Dismiss", key=f"dismiss_{issue['type']}"):
                            st.info("Issue dismissed")
        else:
            st.success("🎉 No issues found! Your contract looks great.")
    
    # Helper methods
    def get_templates_for_category(self, category: str) -> List[str]:
        """Get available templates for a category."""
        templates = {
            "Terms of Service": ["Basic ToS", "SaaS ToS", "E-commerce ToS", "Mobile App ToS"],
            "Privacy Policy": ["Basic Privacy", "GDPR Compliant", "CCPA Compliant", "Healthcare Privacy"],
            "Software License": ["MIT License", "GPL License", "Commercial License", "SaaS License"],
            "Employment Agreement": ["Full-time Employee", "Contractor", "Remote Worker", "Executive"],
            "Service Agreement": ["Consulting", "Maintenance", "Support", "Development"],
            "NDA": ["Mutual NDA", "One-way NDA", "Employee NDA", "Vendor NDA"]
        }
        return templates.get(category, ["Basic Template"])
    
    def get_clause_recommendations(self) -> List[Dict]:
        """Get AI-powered clause recommendations."""
        return [
            {
                'clause': 'GDPR Compliance Clause',
                'reason': 'Recommended for international services'
            },
            {
                'clause': 'Force Majeure Clause',
                'reason': 'Important for service availability disclaimers'
            },
            {
                'clause': 'Severability Clause',
                'reason': 'Protects contract validity if parts are invalid'
            }
        ]
    
    def get_clauses_for_category(self, category: str, search_term: str, 
                                jurisdictions: List[str], complexity: str) -> List[Dict]:
        """Get clauses for selected category with filters."""
        # Mock clause data
        sample_clauses = [
            {
                'id': '1',
                'title': 'Data Collection Clause',
                'category': 'Privacy & Data',
                'description': 'Defines what personal data is collected and how',
                'sample_text': 'We collect personal information that you provide to us directly, such as when you create an account, make a purchase, or contact us.',
                'complexity': 'Standard',
                'jurisdictions': ['US', 'EU', 'UK'],
                'rating': 4
            },
            {
                'id': '2',
                'title': 'Binding Arbitration Clause',
                'category': 'Dispute Resolution',
                'description': 'Requires disputes to be resolved through arbitration',
                'sample_text': 'Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration.',
                'complexity': 'Advanced',
                'jurisdictions': ['US'],
                'rating': 5
            },
            {
                'id': '3',
                'title': 'Limitation of Liability',
                'category': 'Liability & Risk',
                'description': 'Limits company liability for damages',
                'sample_text': 'In no event shall the Company be liable for any indirect, incidental, special, or consequential damages.',
                'complexity': 'Standard',
                'jurisdictions': ['US', 'UK', 'CA'],
                'rating': 4
            }
        ]
        
        # Apply filters (simplified for demo)
        filtered_clauses = sample_clauses
        
        if search_term:
            filtered_clauses = [c for c in filtered_clauses 
                              if search_term.lower() in c['title'].lower() 
                              or search_term.lower() in c['description'].lower()]
        
        return filtered_clauses
    
    def add_clause_to_contract(self, clause: Dict):
        """Add clause to current contract."""
        st.success(f"Added '{clause['title']}' to contract!")
    
    def customize_clause(self, clause: Dict):
        """Open clause customization interface."""
        st.info(f"Customization interface for '{clause['title']}' would open here")
    
    def get_ai_clause_suggestions(self) -> List[Dict]:
        """Get AI-powered clause suggestions."""
        return [
            {
                'title': 'Cookie Consent Clause',
                'reason': 'Required for EU compliance based on your privacy settings'
            },
            {
                'title': 'Age Verification Clause',
                'reason': 'Recommended for services with age restrictions'
            }
        ]
    
    def edit_section(self, section: Dict):
        """Edit a contract section."""
        st.info(f"Section editor for '{section['title']}' would open here")
    
    def export_contract(self, format_type: str, contract_content: Dict):
        """Export contract in specified format."""
        st.success(f"Contract exported as {format_type.upper()} (mock export)")
    
    def export_analysis(self, contract_content: Dict):
        """Export contract analysis."""
        st.success("Analysis exported (mock export)")
    
    def email_contract(self, contract_content: Dict):
        """Email contract to stakeholders."""
        st.success("Contract emailed (mock email)")


# Mock classes for demonstration
class MockContractGenerator:
    def generate(self, **kwargs) -> Dict:
        """Generate a mock contract."""
        return {
            'sections': [
                {
                    'title': '1. Acceptance of Terms',
                    'content': '''By accessing and using this service, you accept and agree to be bound by the terms and provision of this agreement. If you do not agree to abide by the above, please do not use this service.''',
                    'ai_generated': True
                },
                {
                    'title': '2. Use License',
                    'content': '''Permission is granted to temporarily download one copy of the materials on our website for personal, non-commercial transitory viewing only. This is the grant of a license, not a transfer of title.''',
                    'ai_generated': True
                },
                {
                    'title': '3. Disclaimer',
                    'content': '''The materials on our website are provided on an 'as is' basis. To the fullest extent permitted by law, this Company excludes all representations, warranties, obligations, and liabilities.''',
                    'ai_generated': False
                },
                {
                    'title': '4. Limitations',
                    'content': '''In no event shall our Company or its suppliers be liable for any damages (including, without limitation, damages for loss of data or profit, or due to business interruption) arising out of the use or inability to use the materials on our website.''',
                    'ai_generated': True
                },
                {
                    'title': '5. Privacy Policy',
                    'content': '''Your privacy is important to us. Our Privacy Policy explains how we collect, use, and protect your information when you use our service. By using our service, you agree to the collection and use of information in accordance with our Privacy Policy.''',
                    'ai_generated': True
                }
            ],
            'word_count': 1247,
            'readability_score': 78,
            'compliance_score': 0.92,
            'compliance_analysis': {
                'United States': {'compliant': True, 'score': 0.95, 'issues': []},
                'European Union': {'compliant': False, 'score': 0.87, 'issues': ['GDPR clause needed']},
                'United Kingdom': {'compliant': True, 'score': 0.91, 'issues': []}
            },
            'risk_assessment': [
                {
                    'category': 'Liability Risk',
                    'severity': 'Medium',
                    'description': 'Broad liability limitations may not be enforceable in all jurisdictions',
                    'mitigation': 'Add jurisdiction-specific liability clauses'
                },
                {
                    'category': 'Privacy Risk',
                    'severity': 'Low',
                    'description': 'Privacy policy reference is present but could be more specific',
                    'mitigation': 'Include detailed privacy practices in the terms'
                }
            ]
        }


class MockDocumentGenerator:
    pass


class MockComplianceChecker:
    pass