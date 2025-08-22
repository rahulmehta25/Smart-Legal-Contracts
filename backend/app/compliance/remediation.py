from typing import Dict, List, Any
from .regulations_db import RegulationType
from .risk_matrix import ComplianceRiskMatrix, RiskLevel

class ComplianceRemediation:
    def __init__(self, risk_matrix: ComplianceRiskMatrix):
        self.risk_matrix = risk_matrix
        self.remediation_playbooks: Dict[RiskLevel, Dict[str, List[str]]] = {
            RiskLevel.LOW: {
                'general': [
                    'Perform quarterly compliance review',
                    'Update documentation',
                    'Conduct refresher training'
                ]
            },
            RiskLevel.MEDIUM: {
                'data_collection': [
                    'Review and minimize data collection',
                    'Update consent mechanisms',
                    'Enhance privacy notices'
                ],
                'data_storage': [
                    'Implement encryption at rest',
                    'Audit access controls',
                    'Review data retention policies'
                ]
            },
            RiskLevel.HIGH: {
                'data_sharing': [
                    'Conduct comprehensive third-party vendor assessment',
                    'Revise data processing agreements',
                    'Implement stricter data sharing controls'
                ],
                'breach_response': [
                    'Develop comprehensive incident response plan',
                    'Conduct breach simulation exercises',
                    'Establish clear notification procedures'
                ]
            },
            RiskLevel.CRITICAL: {
                'immediate_actions': [
                    'Halt non-essential data processing',
                    'Conduct immediate comprehensive security audit',
                    'Engage legal and compliance experts',
                    'Prepare for potential regulatory reporting'
                ]
            }
        }
    
    def generate_remediation_plan(self, compliance_data: Dict[str, Any], regulation: RegulationType) -> Dict[str, Any]:
        """Generate a detailed remediation plan based on compliance risk assessment"""
        risk_assessment = self.risk_matrix.assess_compliance_risk(compliance_data, regulation)
        risk_level = risk_assessment['risk_level']
        
        remediation_plan = {
            'regulation': regulation.name,
            'risk_level': risk_level.name,
            'total_risk_score': risk_assessment['total_risk_score'],
            'remediation_steps': self._get_remediation_steps(risk_level, risk_assessment['risk_factors']),
            'implementation_guidance': self._provide_implementation_guidance(risk_level)
        }
        
        return remediation_plan
    
    def _get_remediation_steps(self, risk_level: RiskLevel, risk_factors: Dict[str, Any]) -> List[str]:
        """Retrieve remediation steps based on risk level and specific risk factors"""
        steps = []
        
        # Add general and specific remediation steps
        if risk_level in self.remediation_playbooks:
            # Add general steps for the risk level
            steps.extend(self.remediation_playbooks[risk_level].get('general', []))
            
            # Add specific steps for high-risk factors
            for factor_name, factor_data in risk_factors.items():
                if factor_data['level'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    specific_steps = self.remediation_playbooks[risk_level].get(factor_name, [])
                    steps.extend(specific_steps)
        
        return steps
    
    def _provide_implementation_guidance(self, risk_level: RiskLevel) -> Dict[str, Any]:
        """Provide detailed implementation guidance based on risk level"""
        guidance = {
            'recommended_timeline': self._get_timeline(risk_level),
            'resource_allocation': self._get_resource_allocation(risk_level),
            'priority_actions': self._get_priority_actions(risk_level)
        }
        return guidance
    
    def _get_timeline(self, risk_level: RiskLevel) -> str:
        """Determine implementation timeline based on risk level"""
        timelines = {
            RiskLevel.LOW: '3-6 months',
            RiskLevel.MEDIUM: '1-3 months',
            RiskLevel.HIGH: '2-4 weeks',
            RiskLevel.CRITICAL: 'Immediate (within 72 hours)'
        }
        return timelines.get(risk_level, 'Undetermined')
    
    def _get_resource_allocation(self, risk_level: RiskLevel) -> Dict[str, str]:
        """Recommend resource allocation based on risk level"""
        allocations = {
            RiskLevel.LOW: {
                'compliance_team': 'Part-time',
                'legal_support': 'Quarterly consultation',
                'budget_impact': 'Minimal'
            },
            RiskLevel.MEDIUM: {
                'compliance_team': 'Dedicated resource',
                'legal_support': 'Monthly consultation',
                'budget_impact': 'Moderate'
            },
            RiskLevel.HIGH: {
                'compliance_team': 'Full-time',
                'legal_support': 'Bi-weekly consultation',
                'budget_impact': 'Significant'
            },
            RiskLevel.CRITICAL: {
                'compliance_team': 'Emergency response team',
                'legal_support': 'Continuous engagement',
                'budget_impact': 'Major investment'
            }
        }
        return allocations.get(risk_level, {})
    
    def _get_priority_actions(self, risk_level: RiskLevel) -> List[str]:
        """Determine top priority actions based on risk level"""
        priority_actions = {
            RiskLevel.LOW: [
                'Update documentation',
                'Conduct routine training'
            ],
            RiskLevel.MEDIUM: [
                'Review data collection practices',
                'Enhance consent mechanisms'
            ],
            RiskLevel.HIGH: [
                'Audit data sharing processes',
                'Strengthen security controls'
            ],
            RiskLevel.CRITICAL: [
                'Immediate security assessment',
                'Halt risky data processing',
                'Prepare regulatory notification'
            ]
        }
        return priority_actions.get(risk_level, [])

# Instantiate with risk matrix
compliance_remediation = ComplianceRemediation(ComplianceRiskMatrix())