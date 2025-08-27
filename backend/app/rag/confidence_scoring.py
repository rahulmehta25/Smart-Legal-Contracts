"""
Advanced Confidence Scoring and Risk Assessment for Legal Document Analysis
Implements multi-factor confidence scoring with explainable AI
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_text
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Detailed confidence score with explanation"""
    overall_score: float
    component_scores: Dict[str, float]
    factors: List[Dict[str, Any]]
    explanation: str
    reliability_indicator: str
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskScore:
    """Comprehensive risk score with breakdown"""
    overall_risk: float
    risk_category: str
    risk_factors: List[Dict[str, Any]]
    mitigation_priority: List[Dict[str, Any]]
    legal_exposure: Dict[str, Any]
    financial_impact: Dict[str, Any]
    operational_impact: Dict[str, Any]
    recommendations: List[str]
    confidence_in_assessment: float


@dataclass
class ClauseRiskProfile:
    """Risk profile for individual clauses"""
    clause_id: str
    clause_type: str
    risk_level: str
    risk_score: float
    vulnerability_points: List[str]
    enforceability_concerns: List[str]
    negotiation_leverage: str
    fallback_positions: List[str]


class ConfidenceScoringEngine:
    """Advanced confidence scoring with explainable AI"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize confidence scoring engine"""
        self.config = config or self._get_default_config()
        self._initialize_models()
        self._load_scoring_weights()
        self.feature_importance_cache = {}
        
        logger.info("Confidence Scoring Engine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'use_ml_models': True,
            'enable_explainability': True,
            'confidence_threshold': 0.7,
            'min_evidence_count': 3,
            'weight_decay_factor': 0.9,
            'ensemble_method': 'weighted_average'
        }
    
    def _initialize_models(self):
        """Initialize ML models for scoring"""
        if self.config['use_ml_models']:
            # Initialize ensemble models
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            self.scaler = StandardScaler()
            
            # Initialize explainability tools
            if self.config['enable_explainability']:
                self.explainer = None  # Will be initialized when model is trained
    
    def _load_scoring_weights(self):
        """Load scoring weights for different factors"""
        self.scoring_weights = {
            'semantic_similarity': 0.25,
            'keyword_density': 0.15,
            'structural_match': 0.15,
            'entity_recognition': 0.10,
            'pattern_match': 0.15,
            'context_relevance': 0.10,
            'legal_terminology': 0.10
        }
        
        # Risk factor weights
        self.risk_weights = {
            'ambiguity': 0.20,
            'one_sidedness': 0.25,
            'enforceability': 0.20,
            'financial_exposure': 0.15,
            'operational_burden': 0.10,
            'precedent_support': 0.10
        }
    
    def calculate_confidence_score(
        self,
        analysis_result: Dict[str, Any],
        evidence: List[Dict[str, Any]]
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score for analysis
        
        Args:
            analysis_result: Result from analysis pipeline
            evidence: Supporting evidence for the analysis
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        # Extract features for scoring
        features = self._extract_confidence_features(analysis_result, evidence)
        
        # Calculate component scores
        component_scores = self._calculate_component_scores(features)
        
        # Calculate overall score using ensemble method
        overall_score = self._calculate_ensemble_score(component_scores, features)
        
        # Determine confidence interval
        confidence_interval = self._calculate_confidence_interval(
            overall_score, evidence
        )
        
        # Generate factors affecting confidence
        factors = self._identify_confidence_factors(features, component_scores)
        
        # Generate explanation
        explanation = self._generate_confidence_explanation(
            overall_score, component_scores, factors
        )
        
        # Determine reliability indicator
        reliability = self._determine_reliability(overall_score, len(evidence))
        
        return ConfidenceScore(
            overall_score=overall_score,
            component_scores=component_scores,
            factors=factors,
            explanation=explanation,
            reliability_indicator=reliability,
            confidence_interval=confidence_interval,
            metadata={
                'evidence_count': len(evidence),
                'feature_vector': features,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _extract_confidence_features(
        self,
        analysis_result: Dict[str, Any],
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features for confidence scoring"""
        features = {
            'has_arbitration': analysis_result.get('has_arbitration_clause', False),
            'clause_count': len(analysis_result.get('clauses', [])),
            'evidence_count': len(evidence),
            'avg_relevance_score': np.mean([e.get('relevance', 0) for e in evidence]) if evidence else 0,
            'max_relevance_score': max([e.get('relevance', 0) for e in evidence]) if evidence else 0,
            'keyword_matches': sum(e.get('keyword_count', 0) for e in evidence),
            'entity_matches': sum(e.get('entity_count', 0) for e in evidence),
            'pattern_matches': sum(e.get('pattern_match', False) for e in evidence),
            'structural_consistency': self._calculate_structural_consistency(evidence),
            'semantic_coherence': self._calculate_semantic_coherence(evidence),
            'legal_term_density': self._calculate_legal_term_density(analysis_result)
        }
        
        return features
    
    def _calculate_component_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual component scores"""
        scores = {}
        
        # Semantic similarity score
        scores['semantic_similarity'] = min(features['avg_relevance_score'] * 1.2, 1.0)
        
        # Keyword density score
        keyword_ratio = features['keyword_matches'] / max(features['clause_count'], 1)
        scores['keyword_density'] = min(keyword_ratio / 10, 1.0)
        
        # Structural match score
        scores['structural_match'] = features['structural_consistency']
        
        # Entity recognition score
        entity_ratio = features['entity_matches'] / max(features['evidence_count'], 1)
        scores['entity_recognition'] = min(entity_ratio, 1.0)
        
        # Pattern match score
        pattern_ratio = features['pattern_matches'] / max(features['evidence_count'], 1)
        scores['pattern_match'] = min(pattern_ratio * 1.5, 1.0)
        
        # Context relevance score
        scores['context_relevance'] = features['semantic_coherence']
        
        # Legal terminology score
        scores['legal_terminology'] = features['legal_term_density']
        
        return scores
    
    def _calculate_ensemble_score(
        self,
        component_scores: Dict[str, float],
        features: Dict[str, Any]
    ) -> float:
        """Calculate ensemble score using multiple methods"""
        if self.config['ensemble_method'] == 'weighted_average':
            # Weighted average of component scores
            weighted_sum = sum(
                component_scores[component] * self.scoring_weights[component]
                for component in component_scores
            )
            base_score = weighted_sum / sum(self.scoring_weights.values())
        
        elif self.config['ensemble_method'] == 'ml_ensemble':
            # Use ML models if available and trained
            if hasattr(self, 'rf_model') and hasattr(self.rf_model, 'predict_proba'):
                # Prepare feature vector
                feature_vector = self._prepare_feature_vector(features, component_scores)
                
                # Get predictions from models
                rf_pred = self.rf_model.predict_proba(feature_vector.reshape(1, -1))[0, 1]
                gb_pred = self.gb_model.predict(feature_vector.reshape(1, -1))[0]
                
                # Combine predictions
                base_score = (rf_pred * 0.6 + gb_pred * 0.4)
            else:
                # Fallback to weighted average
                weighted_sum = sum(
                    component_scores[component] * self.scoring_weights[component]
                    for component in component_scores
                )
                base_score = weighted_sum / sum(self.scoring_weights.values())
        
        else:
            # Simple average
            base_score = np.mean(list(component_scores.values()))
        
        # Apply confidence adjustments
        if features['evidence_count'] < self.config['min_evidence_count']:
            penalty = (self.config['min_evidence_count'] - features['evidence_count']) * 0.1
            base_score = max(0, base_score - penalty)
        
        return min(max(base_score, 0), 1.0)
    
    def _calculate_confidence_interval(
        self,
        score: float,
        evidence: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the score"""
        # Base standard deviation
        base_std = 0.1
        
        # Adjust based on evidence count
        evidence_factor = 1.0 / (1 + np.log(max(len(evidence), 1)))
        std = base_std * evidence_factor
        
        # Calculate interval (95% confidence)
        lower = max(0, score - 1.96 * std)
        upper = min(1.0, score + 1.96 * std)
        
        return (lower, upper)
    
    def _identify_confidence_factors(
        self,
        features: Dict[str, Any],
        component_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify factors affecting confidence"""
        factors = []
        
        # Positive factors
        if features['evidence_count'] >= self.config['min_evidence_count']:
            factors.append({
                'type': 'positive',
                'factor': 'sufficient_evidence',
                'impact': 0.1,
                'description': f"Found {features['evidence_count']} supporting evidence pieces"
            })
        
        if features['max_relevance_score'] > 0.8:
            factors.append({
                'type': 'positive',
                'factor': 'high_relevance',
                'impact': 0.15,
                'description': "Very high relevance scores in evidence"
            })
        
        if features['pattern_matches'] > 3:
            factors.append({
                'type': 'positive',
                'factor': 'strong_pattern_match',
                'impact': 0.12,
                'description': "Multiple pattern matches found"
            })
        
        # Negative factors
        if features['evidence_count'] < self.config['min_evidence_count']:
            factors.append({
                'type': 'negative',
                'factor': 'insufficient_evidence',
                'impact': -0.15,
                'description': f"Only {features['evidence_count']} evidence pieces found"
            })
        
        if features['semantic_coherence'] < 0.5:
            factors.append({
                'type': 'negative',
                'factor': 'low_coherence',
                'impact': -0.1,
                'description': "Low semantic coherence in evidence"
            })
        
        if component_scores.get('legal_terminology', 0) < 0.3:
            factors.append({
                'type': 'negative',
                'factor': 'weak_legal_language',
                'impact': -0.08,
                'description': "Limited legal terminology detected"
            })
        
        return factors
    
    def _generate_confidence_explanation(
        self,
        overall_score: float,
        component_scores: Dict[str, float],
        factors: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable confidence explanation"""
        # Determine confidence level
        if overall_score >= 0.9:
            level = "Very High"
            base_explanation = "The analysis has very high confidence based on strong evidence."
        elif overall_score >= 0.7:
            level = "High"
            base_explanation = "The analysis has high confidence with good supporting evidence."
        elif overall_score >= 0.5:
            level = "Moderate"
            base_explanation = "The analysis has moderate confidence with mixed evidence."
        elif overall_score >= 0.3:
            level = "Low"
            base_explanation = "The analysis has low confidence due to limited evidence."
        else:
            level = "Very Low"
            base_explanation = "The analysis has very low confidence and should be reviewed."
        
        # Add top contributing factors
        top_components = sorted(
            component_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        component_text = "Key factors: " + ", ".join([
            f"{comp.replace('_', ' ')} ({score:.1%})"
            for comp, score in top_components
        ])
        
        # Add impact factors
        positive_factors = [f for f in factors if f['type'] == 'positive']
        negative_factors = [f for f in factors if f['type'] == 'negative']
        
        factor_text = ""
        if positive_factors:
            factor_text += " Strengths: " + "; ".join([f['description'] for f in positive_factors[:2]])
        if negative_factors:
            factor_text += " Concerns: " + "; ".join([f['description'] for f in negative_factors[:2]])
        
        return f"{level} Confidence ({overall_score:.1%}): {base_explanation} {component_text}.{factor_text}"
    
    def _determine_reliability(self, score: float, evidence_count: int) -> str:
        """Determine reliability indicator"""
        if score >= 0.8 and evidence_count >= 5:
            return "highly_reliable"
        elif score >= 0.6 and evidence_count >= 3:
            return "reliable"
        elif score >= 0.4 or evidence_count >= 2:
            return "moderately_reliable"
        else:
            return "low_reliability"
    
    def _calculate_structural_consistency(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate structural consistency of evidence"""
        if not evidence:
            return 0.0
        
        # Check if evidence comes from similar document structures
        structure_types = [e.get('structure_type', 'unknown') for e in evidence]
        
        # Calculate consistency based on structure type agreement
        if len(set(structure_types)) == 1:
            return 1.0
        else:
            unique_types = len(set(structure_types))
            return max(0, 1.0 - (unique_types - 1) * 0.2)
    
    def _calculate_semantic_coherence(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence of evidence"""
        if not evidence:
            return 0.0
        
        # In a real implementation, this would use embeddings
        # For now, use a simplified approach
        relevance_scores = [e.get('relevance', 0) for e in evidence]
        
        if not relevance_scores:
            return 0.0
        
        # Calculate variance in relevance scores
        variance = np.var(relevance_scores)
        
        # Lower variance means higher coherence
        coherence = 1.0 / (1.0 + variance)
        
        return coherence
    
    def _calculate_legal_term_density(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate density of legal terminology"""
        legal_terms = [
            'shall', 'hereby', 'whereas', 'pursuant', 'notwithstanding',
            'arbitration', 'jurisdiction', 'liability', 'indemnify',
            'covenant', 'warranty', 'breach', 'default', 'termination'
        ]
        
        text = ' '.join([
            clause.get('text', '')
            for clause in analysis_result.get('clauses', [])
        ])
        
        if not text:
            return 0.0
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        term_count = sum(1 for term in legal_terms if term in text_lower)
        density = term_count / word_count * 100  # Terms per 100 words
        
        # Normalize to 0-1 scale
        return min(density / 10, 1.0)
    
    def _prepare_feature_vector(
        self,
        features: Dict[str, Any],
        component_scores: Dict[str, float]
    ) -> np.ndarray:
        """Prepare feature vector for ML models"""
        vector = []
        
        # Add raw features
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, bool):
                vector.append(float(value))
            elif isinstance(value, (int, float)):
                vector.append(value)
            else:
                vector.append(0.0)
        
        # Add component scores
        for key in sorted(component_scores.keys()):
            vector.append(component_scores[key])
        
        return np.array(vector)


class RiskAssessmentEngine:
    """Comprehensive risk assessment for legal documents"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk assessment engine"""
        self.config = config or self._get_default_config()
        self._initialize_risk_models()
        self._load_risk_matrices()
        
        logger.info("Risk Assessment Engine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'risk_categories': ['legal', 'financial', 'operational', 'reputational'],
            'risk_threshold_critical': 0.8,
            'risk_threshold_high': 0.6,
            'risk_threshold_medium': 0.4,
            'enable_predictive_modeling': True,
            'consider_jurisdiction': True
        }
    
    def _initialize_risk_models(self):
        """Initialize risk assessment models"""
        # Risk prediction models
        self.risk_factors_db = self._load_risk_factors_database()
        self.precedent_db = self._load_precedent_database()
        
        # Initialize risk scoring models
        if self.config['enable_predictive_modeling']:
            self.risk_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
    
    def _load_risk_matrices(self):
        """Load risk assessment matrices"""
        self.risk_matrices = {
            'clause_type_risk': {
                'arbitration': {'base_risk': 0.6, 'factors': ['binding', 'class_waiver']},
                'limitation_liability': {'base_risk': 0.7, 'factors': ['unlimited', 'gross_negligence']},
                'indemnification': {'base_risk': 0.65, 'factors': ['broad', 'uncapped']},
                'termination': {'base_risk': 0.5, 'factors': ['immediate', 'without_cause']},
                'confidentiality': {'base_risk': 0.4, 'factors': ['perpetual', 'broad_definition']},
                'non_compete': {'base_risk': 0.75, 'factors': ['geographic_scope', 'duration']}
            },
            'jurisdiction_modifiers': {
                'California': {'arbitration': 0.9, 'non_compete': 0.3},
                'New York': {'arbitration': 0.95, 'non_compete': 0.8},
                'Delaware': {'arbitration': 1.0, 'non_compete': 0.85},
                'Texas': {'arbitration': 0.95, 'non_compete': 0.9}
            }
        }
    
    def _load_risk_factors_database(self) -> Dict[str, Any]:
        """Load database of risk factors"""
        return {
            'ambiguous_terms': {
                'reasonable': 0.3,
                'material': 0.35,
                'substantial': 0.3,
                'appropriate': 0.25,
                'promptly': 0.2,
                'good faith': 0.15
            },
            'high_risk_phrases': {
                'sole discretion': 0.7,
                'without limitation': 0.6,
                'in perpetuity': 0.65,
                'unlimited liability': 0.9,
                'no right to terminate': 0.8,
                'automatic renewal': 0.5
            }
        }
    
    def _load_precedent_database(self) -> Dict[str, Any]:
        """Load precedent cases for risk assessment"""
        return {
            'enforceability_cases': {
                'arbitration_class_waiver': {
                    'AT&T Mobility v. Concepcion': {'year': 2011, 'outcome': 'enforced', 'impact': 0.8},
                    'Epic Systems v. Lewis': {'year': 2018, 'outcome': 'enforced', 'impact': 0.85}
                },
                'non_compete': {
                    'California Business and Professions Code 16600': {
                        'jurisdiction': 'California',
                        'outcome': 'void',
                        'impact': 1.0
                    }
                }
            }
        }
    
    def assess_document_risk(
        self,
        document: Dict[str, Any],
        clauses: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> RiskScore:
        """
        Perform comprehensive risk assessment on document
        
        Args:
            document: Document metadata and content
            clauses: Extracted clauses
            context: Additional context (jurisdiction, party types, etc.)
            
        Returns:
            Comprehensive RiskScore
        """
        # Assess individual clause risks
        clause_risks = [self._assess_clause_risk(clause, context) for clause in clauses]
        
        # Calculate aggregate risk scores
        risk_by_category = self._aggregate_risks_by_category(clause_risks)
        
        # Identify specific risk factors
        risk_factors = self._identify_risk_factors(document, clauses, clause_risks)
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(risk_by_category, risk_factors)
        
        # Determine risk category
        risk_category = self._determine_risk_category(overall_risk)
        
        # Assess legal exposure
        legal_exposure = self._assess_legal_exposure(clauses, context)
        
        # Assess financial impact
        financial_impact = self._assess_financial_impact(clauses, context)
        
        # Assess operational impact
        operational_impact = self._assess_operational_impact(clauses)
        
        # Generate mitigation priorities
        mitigation_priority = self._prioritize_mitigation(risk_factors, clause_risks)
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            risk_factors, clause_risks, context
        )
        
        # Calculate confidence in assessment
        confidence = self._calculate_assessment_confidence(clauses, risk_factors)
        
        return RiskScore(
            overall_risk=overall_risk,
            risk_category=risk_category,
            risk_factors=risk_factors,
            mitigation_priority=mitigation_priority,
            legal_exposure=legal_exposure,
            financial_impact=financial_impact,
            operational_impact=operational_impact,
            recommendations=recommendations,
            confidence_in_assessment=confidence
        )
    
    def _assess_clause_risk(
        self,
        clause: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> ClauseRiskProfile:
        """Assess risk for individual clause"""
        clause_type = clause.get('clause_type', 'general')
        clause_text = clause.get('text', '')
        
        # Get base risk for clause type
        base_risk = self.risk_matrices['clause_type_risk'].get(
            clause_type, {}
        ).get('base_risk', 0.3)
        
        # Apply jurisdiction modifiers if available
        if context and self.config['consider_jurisdiction']:
            jurisdiction = context.get('jurisdiction', 'Delaware')
            if jurisdiction in self.risk_matrices['jurisdiction_modifiers']:
                modifier = self.risk_matrices['jurisdiction_modifiers'][jurisdiction].get(
                    clause_type, 1.0
                )
                base_risk *= modifier
        
        # Check for risk factors in clause
        risk_factors = []
        vulnerability_points = []
        
        # Check for ambiguous terms
        for term, risk_add in self.risk_factors_db['ambiguous_terms'].items():
            if term in clause_text.lower():
                base_risk += risk_add * 0.5
                vulnerability_points.append(f"Contains ambiguous term: '{term}'")
        
        # Check for high-risk phrases
        for phrase, risk_add in self.risk_factors_db['high_risk_phrases'].items():
            if phrase in clause_text.lower():
                base_risk += risk_add * 0.7
                risk_factors.append(f"High-risk phrase: '{phrase}'")
        
        # Determine enforceability concerns
        enforceability_concerns = self._check_enforceability_concerns(
            clause_type, clause_text, context
        )
        
        # Calculate final risk score
        final_risk = min(base_risk, 1.0)
        
        # Determine risk level
        if final_risk >= self.config['risk_threshold_critical']:
            risk_level = 'critical'
        elif final_risk >= self.config['risk_threshold_high']:
            risk_level = 'high'
        elif final_risk >= self.config['risk_threshold_medium']:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Determine negotiation leverage
        negotiation_leverage = self._assess_negotiation_leverage(
            clause_type, risk_level, context
        )
        
        # Generate fallback positions
        fallback_positions = self._generate_fallback_positions(
            clause_type, risk_factors
        )
        
        return ClauseRiskProfile(
            clause_id=clause.get('id', 'unknown'),
            clause_type=clause_type,
            risk_level=risk_level,
            risk_score=final_risk,
            vulnerability_points=vulnerability_points,
            enforceability_concerns=enforceability_concerns,
            negotiation_leverage=negotiation_leverage,
            fallback_positions=fallback_positions
        )
    
    def _check_enforceability_concerns(
        self,
        clause_type: str,
        clause_text: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Check for enforceability concerns"""
        concerns = []
        
        # Check jurisdiction-specific issues
        if context:
            jurisdiction = context.get('jurisdiction', '')
            
            if clause_type == 'non_compete' and jurisdiction == 'California':
                concerns.append("Non-compete clauses are generally unenforceable in California")
            
            if clause_type == 'arbitration':
                if 'class action waiver' in clause_text.lower():
                    if jurisdiction in ['California', 'New York']:
                        concerns.append("Class action waivers face scrutiny in some contexts")
        
        # Check for unconscionability factors
        if 'sole discretion' in clause_text.lower():
            concerns.append("One-sided discretion may raise unconscionability concerns")
        
        if clause_type == 'limitation_liability':
            if 'gross negligence' not in clause_text.lower():
                concerns.append("Limitation without gross negligence exception may be unenforceable")
        
        return concerns
    
    def _assess_negotiation_leverage(
        self,
        clause_type: str,
        risk_level: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Assess negotiation leverage for clause"""
        if risk_level in ['critical', 'high']:
            return 'strong'
        elif risk_level == 'medium':
            if context and context.get('party_type') == 'consumer':
                return 'moderate_to_strong'
            else:
                return 'moderate'
        else:
            return 'limited'
    
    def _generate_fallback_positions(
        self,
        clause_type: str,
        risk_factors: List[str]
    ) -> List[str]:
        """Generate fallback negotiation positions"""
        fallbacks = []
        
        if clause_type == 'arbitration':
            fallbacks.append("Request carve-outs for injunctive relief")
            fallbacks.append("Negotiate for AAA Consumer Rules if applicable")
            fallbacks.append("Add small claims court exception")
        
        elif clause_type == 'limitation_liability':
            fallbacks.append("Request mutual limitation of liability")
            fallbacks.append("Add gross negligence and willful misconduct exceptions")
            fallbacks.append("Negotiate for cap tied to contract value")
        
        elif clause_type == 'indemnification':
            fallbacks.append("Limit to third-party claims only")
            fallbacks.append("Add knowledge qualifier")
            fallbacks.append("Request notice and control of defense provisions")
        
        return fallbacks
    
    def _aggregate_risks_by_category(
        self,
        clause_risks: List[ClauseRiskProfile]
    ) -> Dict[str, float]:
        """Aggregate risks by category"""
        category_risks = defaultdict(list)
        
        for clause_risk in clause_risks:
            # Map clause types to risk categories
            if clause_risk.clause_type in ['arbitration', 'governing_law', 'jurisdiction']:
                category_risks['legal'].append(clause_risk.risk_score)
            
            if clause_risk.clause_type in ['payment', 'limitation_liability', 'indemnification']:
                category_risks['financial'].append(clause_risk.risk_score)
            
            if clause_risk.clause_type in ['termination', 'performance', 'delivery']:
                category_risks['operational'].append(clause_risk.risk_score)
            
            if clause_risk.clause_type in ['confidentiality', 'intellectual_property']:
                category_risks['reputational'].append(clause_risk.risk_score)
        
        # Calculate average risk per category
        aggregated = {}
        for category in self.config['risk_categories']:
            if category in category_risks and category_risks[category]:
                aggregated[category] = np.mean(category_risks[category])
            else:
                aggregated[category] = 0.0
        
        return aggregated
    
    def _identify_risk_factors(
        self,
        document: Dict[str, Any],
        clauses: List[Dict[str, Any]],
        clause_risks: List[ClauseRiskProfile]
    ) -> List[Dict[str, Any]]:
        """Identify specific risk factors in document"""
        risk_factors = []
        
        # Check for missing critical clauses
        expected_clauses = {
            'governing_law', 'termination', 'limitation_liability',
            'confidentiality', 'dispute_resolution'
        }
        found_clauses = {c.get('clause_type') for c in clauses}
        missing = expected_clauses - found_clauses
        
        if missing:
            risk_factors.append({
                'type': 'missing_clauses',
                'severity': 'medium',
                'description': f"Missing critical clauses: {', '.join(missing)}",
                'impact': 0.3,
                'mitigation': "Add standard clauses to provide legal certainty"
            })
        
        # Check for high-risk clauses
        critical_clauses = [cr for cr in clause_risks if cr.risk_level == 'critical']
        if critical_clauses:
            risk_factors.append({
                'type': 'critical_clauses',
                'severity': 'high',
                'description': f"{len(critical_clauses)} critical risk clauses identified",
                'impact': 0.5,
                'mitigation': "Priority review and renegotiation required"
            })
        
        # Check for one-sided provisions
        one_sided_count = sum(
            1 for cr in clause_risks
            if any('one-sided' in v.lower() for v in cr.vulnerability_points)
        )
        
        if one_sided_count > 2:
            risk_factors.append({
                'type': 'imbalanced_agreement',
                'severity': 'high',
                'description': "Multiple one-sided provisions favor other party",
                'impact': 0.4,
                'mitigation': "Negotiate for more balanced terms"
            })
        
        # Check for enforceability issues
        enforceability_issues = sum(
            len(cr.enforceability_concerns) for cr in clause_risks
        )
        
        if enforceability_issues > 0:
            risk_factors.append({
                'type': 'enforceability_concerns',
                'severity': 'medium',
                'description': f"{enforceability_issues} enforceability concerns identified",
                'impact': 0.35,
                'mitigation': "Review with local counsel for jurisdiction-specific issues"
            })
        
        return risk_factors
    
    def _calculate_overall_risk(
        self,
        risk_by_category: Dict[str, float],
        risk_factors: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall risk score"""
        # Base risk from categories
        if risk_by_category:
            category_risk = np.mean(list(risk_by_category.values()))
        else:
            category_risk = 0.3
        
        # Add risk from specific factors
        factor_risk = sum(f.get('impact', 0) for f in risk_factors)
        
        # Combine with weights
        overall = (category_risk * 0.6 + min(factor_risk, 1.0) * 0.4)
        
        return min(overall, 1.0)
    
    def _determine_risk_category(self, risk_score: float) -> str:
        """Determine overall risk category"""
        if risk_score >= self.config['risk_threshold_critical']:
            return 'Critical Risk - Immediate Action Required'
        elif risk_score >= self.config['risk_threshold_high']:
            return 'High Risk - Significant Concerns'
        elif risk_score >= self.config['risk_threshold_medium']:
            return 'Medium Risk - Moderate Concerns'
        else:
            return 'Low Risk - Standard Precautions'
    
    def _assess_legal_exposure(
        self,
        clauses: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess legal exposure from document"""
        exposure = {
            'litigation_risk': 'low',
            'regulatory_risk': 'low',
            'compliance_risk': 'low',
            'key_concerns': []
        }
        
        # Check for litigation triggers
        arbitration_clauses = [c for c in clauses if c.get('clause_type') == 'arbitration']
        if not arbitration_clauses:
            exposure['litigation_risk'] = 'medium'
            exposure['key_concerns'].append("No arbitration clause increases litigation exposure")
        
        # Check for regulatory compliance
        if context and context.get('industry') in ['healthcare', 'finance']:
            exposure['regulatory_risk'] = 'medium'
            exposure['key_concerns'].append(f"Heightened regulatory requirements in {context['industry']}")
        
        return exposure
    
    def _assess_financial_impact(
        self,
        clauses: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess potential financial impact"""
        impact = {
            'maximum_liability': 'uncapped',
            'payment_obligations': [],
            'penalty_provisions': [],
            'estimated_exposure': 'undefined'
        }
        
        # Check for liability limitations
        limitation_clauses = [c for c in clauses if c.get('clause_type') == 'limitation_liability']
        if limitation_clauses:
            # Parse for caps
            for clause in limitation_clauses:
                if 'cap' in clause.get('text', '').lower():
                    impact['maximum_liability'] = 'capped'
                    break
        
        # Check for payment terms
        payment_clauses = [c for c in clauses if c.get('clause_type') == 'payment']
        for clause in payment_clauses:
            impact['payment_obligations'].append({
                'type': 'payment',
                'description': clause.get('text', '')[:100] + '...'
            })
        
        return impact
    
    def _assess_operational_impact(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess operational impact of contract terms"""
        impact = {
            'compliance_burden': 'low',
            'reporting_requirements': [],
            'performance_obligations': [],
            'resource_requirements': 'standard'
        }
        
        # Check for compliance requirements
        compliance_keywords = ['audit', 'report', 'comply', 'maintain records']
        for clause in clauses:
            clause_text = clause.get('text', '').lower()
            if any(keyword in clause_text for keyword in compliance_keywords):
                impact['compliance_burden'] = 'medium'
                impact['reporting_requirements'].append(clause.get('clause_type', 'unknown'))
        
        return impact
    
    def _prioritize_mitigation(
        self,
        risk_factors: List[Dict[str, Any]],
        clause_risks: List[ClauseRiskProfile]
    ) -> List[Dict[str, Any]]:
        """Prioritize risk mitigation actions"""
        priorities = []
        
        # Sort risk factors by severity and impact
        sorted_factors = sorted(
            risk_factors,
            key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}.get(x.get('severity', 'low'), 0),
                x.get('impact', 0)
            ),
            reverse=True
        )
        
        for i, factor in enumerate(sorted_factors[:5]):  # Top 5 priorities
            priorities.append({
                'priority': i + 1,
                'risk_type': factor['type'],
                'severity': factor['severity'],
                'action': factor.get('mitigation', 'Review required'),
                'urgency': 'immediate' if factor['severity'] == 'high' else 'short-term'
            })
        
        # Add clause-specific priorities
        critical_clauses = [cr for cr in clause_risks if cr.risk_level == 'critical']
        for clause in critical_clauses[:3]:  # Top 3 critical clauses
            priorities.append({
                'priority': len(priorities) + 1,
                'risk_type': f"{clause.clause_type}_clause",
                'severity': 'high',
                'action': f"Renegotiate {clause.clause_type} clause",
                'urgency': 'immediate'
            })
        
        return priorities
    
    def _generate_risk_recommendations(
        self,
        risk_factors: List[Dict[str, Any]],
        clause_risks: List[ClauseRiskProfile],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable risk recommendations"""
        recommendations = []
        
        # General recommendations based on risk factors
        for factor in risk_factors:
            if factor['type'] == 'missing_clauses':
                recommendations.append(
                    "Add missing standard clauses to reduce legal uncertainty"
                )
            elif factor['type'] == 'critical_clauses':
                recommendations.append(
                    "Schedule legal review for critical risk clauses before signing"
                )
            elif factor['type'] == 'imbalanced_agreement':
                recommendations.append(
                    "Negotiate for more balanced terms to reduce one-sided risk"
                )
        
        # Specific recommendations for high-risk clauses
        for clause_risk in clause_risks:
            if clause_risk.risk_level in ['critical', 'high']:
                if clause_risk.fallback_positions:
                    recommendations.append(
                        f"For {clause_risk.clause_type}: {clause_risk.fallback_positions[0]}"
                    )
        
        # Context-specific recommendations
        if context:
            if context.get('party_type') == 'startup':
                recommendations.append(
                    "Consider shorter term with renewal options to maintain flexibility"
                )
            
            if context.get('jurisdiction') == 'California':
                recommendations.append(
                    "Review employment-related clauses for California compliance"
                )
        
        # Limit to top recommendations
        return recommendations[:7]
    
    def _calculate_assessment_confidence(
        self,
        clauses: List[Dict[str, Any]],
        risk_factors: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in risk assessment"""
        base_confidence = 0.7
        
        # Adjust based on clause coverage
        if len(clauses) > 10:
            base_confidence += 0.1
        elif len(clauses) < 5:
            base_confidence -= 0.1
        
        # Adjust based on identified risk factors
        if len(risk_factors) > 5:
            base_confidence += 0.05
        
        # Adjust based on clause confidence scores
        clause_confidences = [c.get('confidence_score', 0.5) for c in clauses]
        if clause_confidences:
            avg_confidence = np.mean(clause_confidences)
            base_confidence = (base_confidence + avg_confidence) / 2
        
        return min(max(base_confidence, 0.3), 0.95)