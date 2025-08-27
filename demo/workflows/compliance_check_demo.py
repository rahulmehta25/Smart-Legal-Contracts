"""
Compliance Check Demo Workflow
=============================

This module provides an interactive demonstration of automated compliance
checking capabilities including regulatory validation, audit trail generation,
and risk assessment.

Features:
- Multi-jurisdiction regulatory compliance checking
- Real-time validation against legal frameworks
- Automated audit trail generation
- Risk assessment with severity levels
- Compliance dashboard with metrics
- Remediation suggestions and guidance
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
    from app.compliance.auto_checker import ComplianceChecker
    from app.compliance.regulations_db import RegulationsDatabase
    from app.compliance.audit_generator import AuditGenerator
    from app.compliance.risk_matrix import RiskMatrix
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")


class ComplianceCheckDemo:
    """Interactive compliance checking demonstration."""
    
    def __init__(self):
        """Initialize the demo."""
        self.compliance_checker = None
        self.regulations_db = None
        self.audit_generator = None
        self.risk_matrix = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize compliance components."""
        try:
            # Mock implementations for demo
            self.compliance_checker = MockComplianceChecker()
            self.regulations_db = MockRegulationsDatabase()
            self.audit_generator = MockAuditGenerator()
            self.risk_matrix = MockRiskMatrix()
        except Exception as e:
            st.error(f"Error initializing components: {e}")
    
    def render_regulatory_check(self):
        """Render the regulatory compliance check interface."""
        st.markdown("### 🔍 Regulatory Compliance Check")
        
        # Document input
        tab1, tab2 = st.tabs(["📤 Upload Document", "✏️ Paste Text"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload document for compliance check",
                type=['txt', 'pdf', 'docx'],
                help="Upload a legal document to check compliance"
            )
            
            if uploaded_file and st.button("🔍 Check Uploaded Document"):
                document_text = self.extract_text_from_file(uploaded_file)
                self.perform_compliance_check(document_text, uploaded_file.name)
        
        with tab2:
            document_text = st.text_area(
                "Paste document text",
                height=300,
                placeholder="Paste your legal document text here for compliance analysis...",
                help="Enter the text of your legal document"
            )
            
            if document_text.strip() and st.button("🔍 Check Text Document"):
                self.perform_compliance_check(document_text.strip(), "Text Input")
        
        # Jurisdiction and regulation selection
        st.markdown("#### ⚖️ Compliance Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            jurisdictions = st.multiselect(
                "Jurisdictions",
                ["United States", "European Union", "United Kingdom", "Canada", 
                 "Australia", "California (CCPA)", "New York"],
                default=["United States", "European Union"],
                help="Select jurisdictions to check compliance against"
            )
        
        with col2:
            regulation_types = st.multiselect(
                "Regulation Types",
                ["Data Protection", "Consumer Protection", "Financial Services", 
                 "Healthcare", "Employment Law", "Accessibility"],
                default=["Data Protection", "Consumer Protection"],
                help="Types of regulations to validate against"
            )
        
        with col3:
            compliance_level = st.selectbox(
                "Compliance Level",
                ["Basic", "Standard", "Strict", "Enterprise"],
                index=1,
                help="Level of compliance checking strictness"
            )
        
        # Quick compliance test with samples
        st.markdown("#### 🚀 Quick Compliance Tests")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Test: Privacy Policy"):
                sample_doc = self.get_sample_documents()['privacy_policy']
                self.perform_compliance_check(sample_doc, "Sample Privacy Policy")
        
        with col2:
            if st.button("📄 Test: Terms of Service"):
                sample_doc = self.get_sample_documents()['terms_of_service']
                self.perform_compliance_check(sample_doc, "Sample Terms of Service")
        
        with col3:
            if st.button("📜 Test: Cookie Policy"):
                sample_doc = self.get_sample_documents()['cookie_policy']
                self.perform_compliance_check(sample_doc, "Sample Cookie Policy")
    
    def perform_compliance_check(self, document_text: str, document_name: str):
        """Perform comprehensive compliance check."""
        with st.spinner("🔍 Running compliance analysis..."):
            # Simulate compliance checking
            time.sleep(2)
            
            # Generate compliance results
            compliance_results = self.compliance_checker.check_compliance(
                document_text, 
                document_name,
                jurisdictions=st.session_state.get('jurisdictions', ['United States']),
                regulation_types=st.session_state.get('regulation_types', ['Data Protection'])
            )
            
            # Store results
            st.session_state.demo_results['compliance'] = {
                'document_name': document_name,
                'document_text': document_text,
                'results': compliance_results,
                'timestamp': time.time()
            }
        
        # Display results
        self.display_compliance_results(compliance_results, document_name)
    
    def display_compliance_results(self, results: Dict, document_name: str):
        """Display compliance check results."""
        st.markdown(f"## 📊 Compliance Results: {document_name}")
        
        # Overall compliance score
        overall_score = results['overall_score']
        score_color = 'green' if overall_score >= 0.8 else 'orange' if overall_score >= 0.6 else 'red'
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**Overall Score**")
            st.markdown(f"<span style='color: {score_color}; font-size: 2em'>{overall_score:.1%}</span>",
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Compliant Jurisdictions", 
                     f"{results['compliant_jurisdictions']}/{results['total_jurisdictions']}")
        
        with col3:
            st.metric("Issues Found", results['total_issues'])
        
        with col4:
            st.metric("High Priority Issues", results['high_priority_issues'])
        
        # Detailed results by jurisdiction
        st.markdown("### 🌍 Jurisdiction-Specific Results")
        
        for jurisdiction, result in results['jurisdiction_results'].items():
            with st.expander(f"⚖️ {jurisdiction} - {result['status']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Compliance status
                    status_icon = "✅" if result['compliant'] else "⚠️"
                    st.markdown(f"{status_icon} **Status:** {result['status']}")
                    st.markdown(f"**Score:** {result['score']:.1%}")
                    
                    # Issues
                    if result['issues']:
                        st.markdown("**Issues Found:**")
                        for issue in result['issues']:
                            severity_color = {
                                'Low': 'green',
                                'Medium': 'orange',
                                'High': 'red',
                                'Critical': 'darkred'
                            }[issue['severity']]
                            
                            st.markdown(f"• **{issue['category']}** "
                                       f"<span style='color: {severity_color}'>[{issue['severity']}]</span>: "
                                       f"{issue['description']}", 
                                       unsafe_allow_html=True)
                            
                            if issue.get('suggestion'):
                                st.write(f"  💡 *Suggestion*: {issue['suggestion']}")
                    else:
                        st.success("✅ No compliance issues found!")
                
                with col2:
                    # Regulation breakdown
                    st.markdown("**Regulations Checked:**")
                    for reg, status in result['regulations'].items():
                        icon = "✅" if status else "❌"
                        st.write(f"{icon} {reg}")
                    
                    # Compliance score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['score'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{jurisdiction} Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': score_color},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
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
        
        # Recommendations
        if results.get('recommendations'):
            st.markdown("### 💡 Recommendations")
            
            for i, rec in enumerate(results['recommendations']):
                priority_color = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }[rec['priority']]
                
                with st.container():
                    st.markdown(f"**{i+1}. {rec['title']}** "
                               f"<span style='color: {priority_color}'>[{rec['priority']} Priority]</span>",
                               unsafe_allow_html=True)
                    st.write(rec['description'])
                    
                    if rec.get('action_items'):
                        st.write("**Action Items:**")
                        for action in rec['action_items']:
                            st.write(f"• {action}")
                    
                    st.markdown("---")
    
    def render_audit_trail(self):
        """Render the audit trail interface."""
        st.markdown("### 📋 Compliance Audit Trail")
        
        # Audit configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Audit Settings")
            
            audit_scope = st.multiselect(
                "Audit Scope",
                ["Document Changes", "Compliance Checks", "User Actions", 
                 "System Events", "Risk Assessments"],
                default=["Document Changes", "Compliance Checks"]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()],
                help="Select date range for audit trail"
            )
            
            if st.button("📊 Generate Audit Trail"):
                self.generate_audit_trail(audit_scope, date_range)
        
        with col2:
            # Display audit trail
            if 'audit_trail' in st.session_state.demo_results:
                self.display_audit_trail(st.session_state.demo_results['audit_trail'])
            else:
                st.info("Click 'Generate Audit Trail' to view compliance history")
    
    def generate_audit_trail(self, scope: List[str], date_range):
        """Generate compliance audit trail."""
        with st.spinner("📋 Generating audit trail..."):
            time.sleep(1)
            
            # Mock audit trail data
            audit_data = self.audit_generator.generate_trail(scope, date_range)
            
            st.session_state.demo_results['audit_trail'] = audit_data
    
    def display_audit_trail(self, audit_data: Dict):
        """Display audit trail results."""
        st.markdown("#### 📊 Audit Trail Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", audit_data['total_events'])
        with col2:
            st.metric("Compliance Checks", audit_data['compliance_checks'])
        with col3:
            st.metric("Issues Identified", audit_data['issues_identified'])
        with col4:
            st.metric("Actions Taken", audit_data['actions_taken'])
        
        # Audit events timeline
        st.markdown("#### ⏰ Event Timeline")
        
        events_df = pd.DataFrame(audit_data['events'])
        
        # Timeline chart
        fig = px.timeline(events_df, 
                         x_start="timestamp", 
                         x_end="end_time",
                         y="event_type",
                         color="severity",
                         title="Compliance Audit Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
        # Event details table
        st.markdown("#### 📋 Event Details")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_type_filter = st.selectbox(
                "Filter by Event Type",
                ["All"] + list(events_df['event_type'].unique())
            )
        
        with col2:
            severity_filter = st.selectbox(
                "Filter by Severity", 
                ["All"] + list(events_df['severity'].unique())
            )
        
        with col3:
            user_filter = st.selectbox(
                "Filter by User",
                ["All"] + list(events_df['user'].unique())
            )
        
        # Apply filters
        filtered_df = events_df.copy()
        if event_type_filter != "All":
            filtered_df = filtered_df[filtered_df['event_type'] == event_type_filter]
        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df['severity'] == severity_filter]
        if user_filter != "All":
            filtered_df = filtered_df[filtered_df['user'] == user_filter]
        
        # Display filtered events
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as PDF"):
                st.success("Audit trail exported as PDF (mock)")
        
        with col2:
            if st.button("📊 Export as Excel"):
                st.success("Audit trail exported as Excel (mock)")
        
        with col3:
            if st.button("📧 Email Report"):
                st.success("Audit report emailed (mock)")
    
    def render_risk_assessment(self):
        """Render the risk assessment interface."""
        st.markdown("### ⚠️ Compliance Risk Assessment")
        
        # Risk assessment configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Assessment Settings")
            
            risk_categories = st.multiselect(
                "Risk Categories",
                ["Legal Compliance", "Data Privacy", "Financial Regulatory", 
                 "Operational", "Reputational", "Strategic"],
                default=["Legal Compliance", "Data Privacy"]
            )
            
            assessment_depth = st.selectbox(
                "Assessment Depth",
                ["Quick Scan", "Standard Assessment", "Deep Analysis"],
                index=1
            )
            
            include_predictions = st.checkbox(
                "Include Risk Predictions",
                value=True,
                help="Use ML models to predict future risks"
            )
            
            if st.button("🔍 Run Risk Assessment", type="primary"):
                self.run_risk_assessment(risk_categories, assessment_depth, include_predictions)
        
        with col2:
            # Display risk assessment results
            if 'risk_assessment' in st.session_state.demo_results:
                self.display_risk_assessment(st.session_state.demo_results['risk_assessment'])
            else:
                st.info("Click 'Run Risk Assessment' to analyze compliance risks")
    
    def run_risk_assessment(self, categories: List[str], depth: str, predictions: bool):
        """Run compliance risk assessment."""
        with st.spinner("⚠️ Analyzing compliance risks..."):
            time.sleep(2)
            
            # Generate risk assessment
            risk_data = self.risk_matrix.assess_risks(categories, depth, predictions)
            
            st.session_state.demo_results['risk_assessment'] = risk_data
    
    def display_risk_assessment(self, risk_data: Dict):
        """Display risk assessment results."""
        st.markdown("#### ⚠️ Risk Assessment Results")
        
        # Overall risk score
        overall_risk = risk_data['overall_risk_score']
        risk_level = risk_data['risk_level']
        
        risk_color = {
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red',
            'Critical': 'darkred'
        }[risk_level]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**Overall Risk Level**")
            st.markdown(f"<span style='color: {risk_color}; font-size: 1.5em'>{risk_level}</span>",
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Risk Score", f"{overall_risk:.1f}/10")
        
        with col3:
            st.metric("High Risk Items", risk_data['high_risk_count'])
        
        with col4:
            st.metric("Mitigation Actions", risk_data['mitigation_actions'])
        
        # Risk matrix visualization
        st.markdown("#### 🎯 Risk Matrix")
        
        risk_matrix_data = risk_data['risk_matrix']
        
        fig = px.scatter(risk_matrix_data, 
                        x='probability', 
                        y='impact',
                        size='severity_score',
                        color='category',
                        hover_data=['risk_name'],
                        title="Compliance Risk Matrix",
                        labels={'probability': 'Probability', 'impact': 'Impact'})
        
        fig.update_layout(
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual risk details
        st.markdown("#### 📋 Risk Details")
        
        for risk in risk_data['risks']:
            severity_color = {
                'Low': 'green',
                'Medium': 'orange', 
                'High': 'red',
                'Critical': 'darkred'
            }[risk['severity']]
            
            with st.expander(f"⚠️ {risk['name']} - {risk['severity']} Risk", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {risk['description']}")
                    st.write(f"**Impact:** {risk['impact']}")
                    st.write(f"**Likelihood:** {risk['likelihood']}")
                    
                    if risk['mitigation_strategies']:
                        st.write("**Mitigation Strategies:**")
                        for strategy in risk['mitigation_strategies']:
                            st.write(f"• {strategy}")
                
                with col2:
                    st.write(f"**Category:** {risk['category']}")
                    st.write(f"**Probability:** {risk['probability']}/10")
                    st.write(f"**Impact Score:** {risk['impact_score']}/10")
                    st.write(f"**Risk Score:** {risk['risk_score']:.1f}/10")
                    
                    # Risk trend
                    if risk.get('trend_data'):
                        trend_fig = px.line(
                            x=risk['trend_data']['dates'],
                            y=risk['trend_data']['scores'],
                            title=f"{risk['name']} Trend"
                        )
                        trend_fig.update_layout(height=200)
                        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Risk mitigation recommendations
        st.markdown("#### 💡 Mitigation Recommendations")
        
        for rec in risk_data['recommendations']:
            priority_color = {
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red'
            }[rec['priority']]
            
            st.markdown(f"**{rec['title']}** "
                       f"<span style='color: {priority_color}'>[{rec['priority']} Priority]</span>",
                       unsafe_allow_html=True)
            st.write(rec['description'])
            
            if rec.get('timeline'):
                st.write(f"⏰ **Timeline:** {rec['timeline']}")
            
            if rec.get('resources'):
                st.write(f"💰 **Resources Required:** {rec['resources']}")
            
            st.markdown("---")
    
    # Helper methods
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
    
    def get_sample_documents(self) -> Dict[str, str]:
        """Get sample documents for compliance testing."""
        return {
            'privacy_policy': '''
            PRIVACY POLICY
            
            Last updated: January 1, 2024
            
            1. INFORMATION WE COLLECT
            We collect personal information that you provide to us directly, such as when you create an account, make a purchase, or contact us. This may include your name, email address, phone number, and payment information.
            
            2. HOW WE USE YOUR INFORMATION
            We use the information we collect to provide, maintain, and improve our services, process transactions, and communicate with you.
            
            3. DATA RETENTION
            We retain your personal information for as long as necessary to provide our services and comply with legal obligations.
            
            4. YOUR RIGHTS
            You have the right to access, update, or delete your personal information. You may also object to certain processing of your data.
            
            5. CONTACT US
            If you have questions about this Privacy Policy, please contact us at privacy@company.com.
            ''',
            
            'terms_of_service': '''
            TERMS OF SERVICE
            
            1. ACCEPTANCE OF TERMS
            By accessing and using this service, you accept and agree to be bound by the terms and provision of this agreement.
            
            2. DESCRIPTION OF SERVICE
            Our service provides [description of service]. We reserve the right to modify or discontinue the service at any time.
            
            3. USER OBLIGATIONS
            You agree to use the service only for lawful purposes and in accordance with these terms.
            
            4. LIMITATION OF LIABILITY
            In no event shall the company be liable for any indirect, incidental, special, or consequential damages.
            
            5. GOVERNING LAW
            These terms shall be governed by and construed in accordance with the laws of [jurisdiction].
            ''',
            
            'cookie_policy': '''
            COOKIE POLICY
            
            1. WHAT ARE COOKIES
            Cookies are small text files that are placed on your computer or mobile device when you visit our website.
            
            2. HOW WE USE COOKIES
            We use cookies to improve your browsing experience, analyze site traffic, and for marketing purposes.
            
            3. TYPES OF COOKIES
            - Essential cookies: Required for the website to function properly
            - Analytics cookies: Help us understand how visitors use our site
            - Marketing cookies: Used to deliver relevant advertisements
            
            4. YOUR CHOICES
            You can control and/or delete cookies as you wish through your browser settings.
            
            5. UPDATES TO THIS POLICY
            We may update this Cookie Policy from time to time. Please review it periodically.
            '''
        }


# Mock classes for demonstration
class MockComplianceChecker:
    def check_compliance(self, document_text: str, document_name: str, **kwargs) -> Dict:
        """Mock compliance check."""
        import random
        
        # Simulate compliance analysis
        jurisdiction_results = {
            'United States': {
                'compliant': True,
                'status': 'Compliant',
                'score': 0.92,
                'issues': [
                    {
                        'category': 'Accessibility',
                        'severity': 'Medium',
                        'description': 'Consider adding accessibility statement for ADA compliance',
                        'suggestion': 'Include a section detailing accessibility features and contact information'
                    }
                ],
                'regulations': {
                    'FTC Guidelines': True,
                    'CAN-SPAM Act': True,
                    'ADA Compliance': False,
                    'COPPA': True
                }
            },
            'European Union': {
                'compliant': False,
                'status': 'Non-Compliant',
                'score': 0.76,
                'issues': [
                    {
                        'category': 'GDPR Compliance',
                        'severity': 'High',
                        'description': 'Missing explicit consent mechanism for data processing',
                        'suggestion': 'Add clear consent checkboxes and lawful basis for processing'
                    },
                    {
                        'category': 'Data Subject Rights',
                        'severity': 'Medium',
                        'description': 'Incomplete description of user rights under GDPR',
                        'suggestion': 'Include detailed explanation of rights to access, rectify, and erase data'
                    }
                ],
                'regulations': {
                    'GDPR': False,
                    'ePrivacy Directive': True,
                    'Digital Services Act': True
                }
            },
            'United Kingdom': {
                'compliant': True,
                'status': 'Compliant',
                'score': 0.88,
                'issues': [
                    {
                        'category': 'UK GDPR',
                        'severity': 'Low',
                        'description': 'Consider referencing UK GDPR specifically',
                        'suggestion': 'Add reference to UK GDPR alongside EU GDPR'
                    }
                ],
                'regulations': {
                    'UK GDPR': True,
                    'Data Protection Act 2018': True,
                    'Consumer Rights Act': True
                }
            }
        }
        
        total_issues = sum(len(result['issues']) for result in jurisdiction_results.values())
        high_priority_issues = sum(
            len([issue for issue in result['issues'] if issue['severity'] in ['High', 'Critical']])
            for result in jurisdiction_results.values()
        )
        
        return {
            'overall_score': 0.85,
            'total_jurisdictions': len(jurisdiction_results),
            'compliant_jurisdictions': sum(1 for r in jurisdiction_results.values() if r['compliant']),
            'total_issues': total_issues,
            'high_priority_issues': high_priority_issues,
            'jurisdiction_results': jurisdiction_results,
            'recommendations': [
                {
                    'title': 'Implement GDPR Consent Management',
                    'priority': 'High',
                    'description': 'Add comprehensive consent management system for EU users',
                    'action_items': [
                        'Design consent banner with granular options',
                        'Implement consent withdrawal mechanism',
                        'Add cookie preference center'
                    ]
                },
                {
                    'title': 'Enhance Accessibility Compliance',
                    'priority': 'Medium',
                    'description': 'Improve accessibility features for ADA compliance',
                    'action_items': [
                        'Add accessibility statement',
                        'Implement screen reader compatibility',
                        'Provide alternative contact methods'
                    ]
                }
            ]
        }


class MockRegulationsDatabase:
    pass


class MockAuditGenerator:
    def generate_trail(self, scope: List[str], date_range) -> Dict:
        """Generate mock audit trail."""
        import random
        from datetime import datetime, timedelta
        
        # Generate sample events
        events = []
        start_date = datetime.now() - timedelta(days=30)
        
        event_types = ['Compliance Check', 'Document Update', 'User Action', 'System Event', 'Risk Assessment']
        severities = ['Low', 'Medium', 'High']
        users = ['admin@company.com', 'legal@company.com', 'user@company.com', 'system']
        
        for i in range(50):
            event_time = start_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            events.append({
                'id': f'EVENT-{i+1:03d}',
                'timestamp': event_time,
                'end_time': event_time + timedelta(minutes=random.randint(1, 60)),
                'event_type': random.choice(event_types),
                'severity': random.choice(severities),
                'user': random.choice(users),
                'description': f'Sample event {i+1} description',
                'status': random.choice(['Completed', 'In Progress', 'Failed'])
            })
        
        return {
            'total_events': len(events),
            'compliance_checks': len([e for e in events if e['event_type'] == 'Compliance Check']),
            'issues_identified': random.randint(5, 15),
            'actions_taken': random.randint(10, 25),
            'events': events
        }


class MockRiskMatrix:
    def assess_risks(self, categories: List[str], depth: str, predictions: bool) -> Dict:
        """Generate mock risk assessment."""
        import random
        
        risks = [
            {
                'name': 'GDPR Non-Compliance',
                'category': 'Legal Compliance',
                'severity': 'High',
                'description': 'Risk of GDPR violations due to inadequate consent mechanisms',
                'impact': 'Potential fines up to 4% of annual revenue',
                'likelihood': 'Medium probability based on current practices',
                'probability': 7,
                'impact_score': 9,
                'risk_score': 7.8,
                'mitigation_strategies': [
                    'Implement comprehensive consent management system',
                    'Conduct GDPR compliance audit',
                    'Train staff on data protection requirements'
                ]
            },
            {
                'name': 'Data Breach Exposure',
                'category': 'Data Privacy',
                'severity': 'Critical',
                'description': 'Risk of personal data exposure due to security vulnerabilities',
                'impact': 'Regulatory fines, reputation damage, customer loss',
                'likelihood': 'Low but increasing with cyber threats',
                'probability': 4,
                'impact_score': 10,
                'risk_score': 7.0,
                'mitigation_strategies': [
                    'Implement end-to-end encryption',
                    'Regular security audits and penetration testing',
                    'Employee cybersecurity training'
                ]
            },
            {
                'name': 'Accessibility Compliance Gap',
                'category': 'Legal Compliance',
                'severity': 'Medium',
                'description': 'Risk of ADA non-compliance affecting user accessibility',
                'impact': 'Legal action, exclusion of disabled users',
                'likelihood': 'Medium probability without proactive measures',
                'probability': 6,
                'impact_score': 6,
                'risk_score': 6.0,
                'mitigation_strategies': [
                    'Conduct accessibility audit',
                    'Implement WCAG 2.1 AA standards',
                    'Regular accessibility testing'
                ]
            }
        ]
        
        # Risk matrix data for visualization
        risk_matrix_data = []
        for risk in risks:
            risk_matrix_data.append({
                'risk_name': risk['name'],
                'category': risk['category'],
                'probability': risk['probability'],
                'impact': risk['impact_score'],
                'severity_score': risk['risk_score']
            })
        
        return {
            'overall_risk_score': 6.9,
            'risk_level': 'High',
            'high_risk_count': 2,
            'mitigation_actions': 9,
            'risks': risks,
            'risk_matrix': risk_matrix_data,
            'recommendations': [
                {
                    'title': 'Immediate GDPR Compliance Review',
                    'priority': 'High',
                    'description': 'Conduct comprehensive review of GDPR compliance status',
                    'timeline': '2-4 weeks',
                    'resources': 'Legal consultant, IT team'
                },
                {
                    'title': 'Implement Security Monitoring',
                    'priority': 'High',
                    'description': 'Deploy continuous security monitoring and incident response',
                    'timeline': '4-6 weeks',
                    'resources': 'Security team, monitoring tools'
                },
                {
                    'title': 'Accessibility Compliance Program',
                    'priority': 'Medium',
                    'description': 'Establish ongoing accessibility compliance program',
                    'timeline': '6-8 weeks',
                    'resources': 'UX team, accessibility consultant'
                }
            ]
        }