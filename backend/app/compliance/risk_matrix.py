from enum import Enum, auto
from typing import Dict, List, Any
from dataclasses import dataclass
from .regulations_db import RegulationType

class RiskLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class RiskFactor:
    name: str
    description: str
    weight: float
    mitigations: List[str]

class ComplianceRiskMatrix:
    def __init__(self):
        self.risk_factors: Dict[str, RiskFactor] = {
            'data_collection': RiskFactor(
                name='Data Collection',
                description='Risks associated with collecting personal data',
                weight=0.3,
                mitigations=[
                    'Implement minimal data collection',
                    'Obtain explicit consent',
                    'Provide clear privacy notices'
                ]
            ),
            'data_storage': RiskFactor(
                name='Data Storage',
                description='Risks related to data storage and protection',
                weight=0.25,
                mitigations=[
                    'Use encryption at rest',
                    'Implement access controls',
                    'Regularly audit storage practices'
                ]
            ),
            'data_sharing': RiskFactor(
                name='Data Sharing',
                description='Risks of sharing data with third parties',
                weight=0.2,
                mitigations=[
                    'Conduct vendor security assessments',
                    'Use data processing agreements',
                    'Limit data shared'
                ]
            ),
            'breach_response': RiskFactor(
                name='Breach Response',
                description='Capability to respond to data breaches',
                weight=0.25,
                mitigations=[
                    'Develop incident response plan',
                    'Conduct regular breach simulations',
                    'Maintain breach notification procedures'
                ]
            )
        }
    
    def assess_compliance_risk(self, compliance_data: Dict[str, Any], regulation: RegulationType) -> Dict[str, Any]:
        """Assess overall compliance risk"""
        risk_scores = {}
        total_risk_score = 0.0
        
        for factor_name, factor in self.risk_factors.items():
            # Simulate risk assessment based on compliance data
            risk_score = self._calculate_risk_score(compliance_data, factor)
            risk_scores[factor_name] = {
                'score': risk_score,
                'level': self._determine_risk_level(risk_score),
                'mitigations': factor.mitigations
            }
            total_risk_score += risk_score * factor.weight
        
        return {
            'regulation': regulation.name,
            'total_risk_score': total_risk_score,
            'risk_level': self._determine_risk_level(total_risk_score),
            'risk_factors': risk_scores
        }
    
    def _calculate_risk_score(self, compliance_data: Dict[str, Any], factor: RiskFactor) -> float:
        """Calculate risk score for a specific factor"""
        # Placeholder risk calculation logic
        # In a real implementation, this would use actual compliance data
        base_score = 0.5  # Default medium risk
        
        # Example risk modifiers
        if factor.name == 'data_collection':
            base_score -= 0.2 if compliance_data.get('minimal_data_collection', False) else 0
        
        if factor.name == 'data_storage':
            base_score -= 0.3 if compliance_data.get('encryption_enabled', False) else 0
        
        return max(0, min(1, base_score))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score"""
        if risk_score < 0.2:
            return RiskLevel.LOW
        elif risk_score < 0.4:
            return RiskLevel.MEDIUM
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

# Singleton instance
risk_matrix = ComplianceRiskMatrix()