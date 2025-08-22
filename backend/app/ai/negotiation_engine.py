"""
GPT-4 powered negotiation engine for clause negotiation and strategy recommendations.
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from openai import AsyncOpenAI
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


@dataclass
class NegotiationSuggestion:
    """Represents a negotiation suggestion for a clause."""
    original_clause: str
    suggested_alternatives: List[str]
    negotiation_points: List[str]
    leverage_score: float  # 0-1 scale
    win_probability: float  # 0-1 scale
    strategy: str
    reasoning: str
    risk_level: str
    confidence: float


@dataclass
class NegotiationStrategy:
    """Comprehensive negotiation strategy."""
    approach: str  # aggressive, balanced, conservative
    key_points: List[str]
    concessions: List[str]
    red_lines: List[str]
    expected_outcome: str
    success_probability: float
    timeline: str


class NegotiationEngine:
    """Advanced negotiation engine using GPT-4 and ensemble methods."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the negotiation engine."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided. GPT-4 features will be limited.")
        
        # Initialize local models for fallback and ensemble
        self._initialize_local_models()
        
        # Pre-defined negotiation templates
        self.negotiation_templates = self._load_negotiation_templates()
        
        # Historical negotiation data for win probability
        self.win_probability_model = None
        self.feature_scaler = StandardScaler()
        
    def _initialize_local_models(self):
        """Initialize local transformer models for negotiation analysis."""
        try:
            # Sentiment analyzer for tone assessment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            
            # Zero-shot classifier for clause categorization
            self.clause_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Text generation for alternative suggestions (smaller model for speed)
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                max_length=200
            )
        except Exception as e:
            logger.error(f"Error initializing local models: {e}")
            self.sentiment_analyzer = None
            self.clause_classifier = None
            self.text_generator = None
    
    def _load_negotiation_templates(self) -> Dict[str, List[str]]:
        """Load pre-defined negotiation templates."""
        return {
            "arbitration": [
                "We propose maintaining judicial recourse with arbitration as a voluntary option",
                "Consider mediation as the first step before binding arbitration",
                "Request carve-outs for claims under $10,000 or injunctive relief",
                "Suggest mutual agreement on arbitrator selection process",
                "Propose cost-shifting provisions favorable to consumers"
            ],
            "limitation_of_liability": [
                "Request removal of consequential damages waiver",
                "Propose liability cap at contract value rather than fixed amount",
                "Include exceptions for gross negligence and willful misconduct",
                "Request mutual limitation rather than one-sided protection",
                "Suggest graduated liability based on service tiers"
            ],
            "termination": [
                "Request notice period extension to 30 days",
                "Include cure period for breach before termination",
                "Add data portability and transition assistance provisions",
                "Request pro-rated refunds for pre-paid services",
                "Include termination for convenience with reasonable notice"
            ],
            "intellectual_property": [
                "Clarify ownership of pre-existing IP remains with creator",
                "Request license-back provisions for improvements",
                "Limit IP assignment to work specifically paid for",
                "Include moral rights retention where applicable",
                "Request attribution requirements for derivative works"
            ]
        }
    
    async def analyze_clause_negotiability(
        self,
        clause: str,
        context: Optional[str] = None,
        party_position: str = "consumer"
    ) -> NegotiationSuggestion:
        """
        Analyze a clause and generate negotiation suggestions.
        
        Args:
            clause: The clause to analyze
            context: Additional contract context
            party_position: Position of the negotiating party
            
        Returns:
            NegotiationSuggestion with alternatives and strategy
        """
        # Extract clause features
        features = await self._extract_clause_features(clause)
        
        # Generate alternatives using ensemble approach
        alternatives = await self._generate_alternatives(clause, context, party_position)
        
        # Calculate win probability
        win_probability = self._calculate_win_probability(features)
        
        # Determine negotiation strategy
        strategy = self._determine_strategy(features, win_probability, party_position)
        
        # Generate negotiation points
        negotiation_points = await self._generate_negotiation_points(
            clause, alternatives, strategy
        )
        
        # Assess leverage and risk
        leverage_score = self._calculate_leverage(features, party_position)
        risk_level = self._assess_risk_level(features, leverage_score)
        
        return NegotiationSuggestion(
            original_clause=clause,
            suggested_alternatives=alternatives,
            negotiation_points=negotiation_points,
            leverage_score=leverage_score,
            win_probability=win_probability,
            strategy=strategy['approach'],
            reasoning=strategy['reasoning'],
            risk_level=risk_level,
            confidence=features.get('confidence', 0.7)
        )
    
    async def _extract_clause_features(self, clause: str) -> Dict[str, Any]:
        """Extract features from clause for analysis."""
        features = {
            'length': len(clause.split()),
            'complexity': self._calculate_complexity(clause),
            'sentiment': 0.0,
            'enforceability': 0.0,
            'one_sided': 0.0,
            'ambiguity': 0.0,
            'confidence': 0.0
        }
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(clause[:512])[0]
                features['sentiment'] = float(sentiment['label'].split()[0]) / 5.0
            except:
                pass
        
        # Classify clause type
        if self.clause_classifier:
            try:
                labels = ["fair", "unfair", "one-sided", "balanced", "ambiguous"]
                result = self.clause_classifier(clause, labels)
                features['one_sided'] = result['scores'][labels.index('one-sided')]
                features['ambiguity'] = result['scores'][labels.index('ambiguous')]
                features['confidence'] = max(result['scores'])
            except:
                pass
        
        # GPT-4 analysis if available
        if self.client:
            try:
                gpt_features = await self._gpt4_feature_extraction(clause)
                features.update(gpt_features)
            except:
                pass
        
        return features
    
    async def _gpt4_feature_extraction(self, clause: str) -> Dict[str, float]:
        """Use GPT-4 for advanced feature extraction."""
        prompt = f"""
        Analyze this legal clause and provide scores (0-1) for:
        1. Enforceability (how likely to hold up in court)
        2. Fairness (balance between parties)
        3. Clarity (how clear and unambiguous)
        4. Negotiability (how open to negotiation)
        
        Clause: {clause}
        
        Return JSON format: {{"enforceability": 0.0, "fairness": 0.0, "clarity": 0.0, "negotiability": 0.0}}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a legal contract analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"GPT-4 feature extraction error: {e}")
            return {}
    
    async def _generate_alternatives(
        self,
        clause: str,
        context: Optional[str],
        party_position: str
    ) -> List[str]:
        """Generate alternative clause formulations."""
        alternatives = []
        
        # Use templates if available
        clause_type = self._identify_clause_type(clause)
        if clause_type in self.negotiation_templates:
            alternatives.extend(self.negotiation_templates[clause_type][:2])
        
        # GPT-4 generation
        if self.client:
            try:
                gpt_alternatives = await self._gpt4_generate_alternatives(
                    clause, context, party_position
                )
                alternatives.extend(gpt_alternatives)
            except:
                pass
        
        # Local model generation as fallback
        if self.text_generator and len(alternatives) < 3:
            try:
                prompt = f"Alternative to: {clause[:100]}... Suggestion:"
                local_alt = self.text_generator(prompt, max_length=150)[0]['generated_text']
                alternatives.append(local_alt.split("Suggestion:")[1].strip())
            except:
                pass
        
        return alternatives[:5]  # Return top 5 alternatives
    
    async def _gpt4_generate_alternatives(
        self,
        clause: str,
        context: Optional[str],
        party_position: str
    ) -> List[str]:
        """Use GPT-4 to generate alternative clauses."""
        prompt = f"""
        Generate 3 alternative formulations for this clause that are more favorable to the {party_position}.
        Maintain legal validity while improving fairness and clarity.
        
        Original clause: {clause}
        
        Context: {context or 'Standard commercial agreement'}
        
        Return as JSON array of strings.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a contract negotiation expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('alternatives', [])
        except Exception as e:
            logger.error(f"GPT-4 alternative generation error: {e}")
            return []
    
    def _calculate_win_probability(self, features: Dict[str, Any]) -> float:
        """Calculate probability of successful negotiation."""
        if self.win_probability_model:
            try:
                feature_vector = np.array([
                    features.get('enforceability', 0.5),
                    features.get('fairness', 0.5),
                    features.get('clarity', 0.5),
                    features.get('negotiability', 0.5),
                    features.get('one_sided', 0.5),
                    features.get('ambiguity', 0.5)
                ]).reshape(1, -1)
                
                scaled_features = self.feature_scaler.transform(feature_vector)
                probability = self.win_probability_model.predict_proba(scaled_features)[0, 1]
                return float(probability)
            except:
                pass
        
        # Heuristic calculation if no model
        base_prob = 0.5
        
        # Adjust based on features
        if features.get('fairness', 0.5) < 0.3:
            base_prob += 0.2  # Unfair clauses more negotiable
        if features.get('ambiguity', 0.5) > 0.6:
            base_prob += 0.15  # Ambiguous clauses open to interpretation
        if features.get('one_sided', 0.5) > 0.7:
            base_prob += 0.1  # One-sided clauses often negotiable
        if features.get('enforceability', 0.5) < 0.4:
            base_prob += 0.1  # Weak enforceability increases negotiation leverage
        
        return min(base_prob, 0.95)
    
    def _determine_strategy(
        self,
        features: Dict[str, Any],
        win_probability: float,
        party_position: str
    ) -> Dict[str, str]:
        """Determine optimal negotiation strategy."""
        if win_probability > 0.7:
            approach = "aggressive"
            reasoning = "High success probability justifies firm negotiation stance"
        elif win_probability > 0.4:
            approach = "balanced"
            reasoning = "Moderate success probability suggests collaborative approach"
        else:
            approach = "conservative"
            reasoning = "Lower success probability requires careful concessions"
        
        # Adjust for party position
        if party_position == "consumer" and features.get('one_sided', 0) > 0.6:
            approach = "aggressive"
            reasoning = "One-sided clause against consumer warrants strong pushback"
        
        return {
            'approach': approach,
            'reasoning': reasoning
        }
    
    async def _generate_negotiation_points(
        self,
        clause: str,
        alternatives: List[str],
        strategy: Dict[str, str]
    ) -> List[str]:
        """Generate specific negotiation talking points."""
        points = []
        
        # Standard negotiation points based on strategy
        if strategy['approach'] == "aggressive":
            points.extend([
                "This clause is fundamentally unfair and requires substantial revision",
                "Industry standards support more balanced terms",
                "We cannot accept this provision without significant modifications"
            ])
        elif strategy['approach'] == "balanced":
            points.extend([
                "We appreciate the intent but suggest modifications for mutual benefit",
                "Let's find middle ground that protects both parties",
                "Minor adjustments would make this acceptable to both sides"
            ])
        else:
            points.extend([
                "We understand your position and propose modest changes",
                "Can we explore alternatives that address your concerns?",
                "We're flexible but need some accommodation on key points"
            ])
        
        # GPT-4 specific points if available
        if self.client and clause:
            try:
                gpt_points = await self._gpt4_negotiation_points(clause, strategy)
                points.extend(gpt_points)
            except:
                pass
        
        return points[:5]
    
    async def _gpt4_negotiation_points(
        self,
        clause: str,
        strategy: Dict[str, str]
    ) -> List[str]:
        """Generate negotiation points using GPT-4."""
        prompt = f"""
        Generate 3 specific negotiation talking points for this clause.
        Strategy: {strategy['approach']}
        
        Clause: {clause[:500]}
        
        Return as JSON array of concise, professional negotiation points.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a skilled contract negotiator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.6
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('points', [])
        except:
            return []
    
    def _calculate_leverage(self, features: Dict[str, Any], party_position: str) -> float:
        """Calculate negotiation leverage score."""
        leverage = 0.5  # Base leverage
        
        # Adjust based on clause characteristics
        if features.get('enforceability', 0.5) < 0.4:
            leverage += 0.2  # Weak clauses give more leverage
        if features.get('ambiguity', 0.5) > 0.6:
            leverage += 0.15  # Ambiguous clauses are negotiable
        if features.get('one_sided', 0.5) > 0.7:
            leverage += 0.15  # Extremely one-sided clauses vulnerable
        
        # Party position adjustment
        if party_position == "enterprise":
            leverage += 0.1  # Enterprises typically have more leverage
        
        return min(leverage, 1.0)
    
    def _assess_risk_level(self, features: Dict[str, Any], leverage: float) -> str:
        """Assess risk level of negotiation."""
        risk_score = 0.0
        
        # Calculate risk factors
        risk_score += (1 - features.get('clarity', 0.5)) * 0.3
        risk_score += features.get('enforceability', 0.5) * 0.3
        risk_score += (1 - leverage) * 0.4
        
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        else:
            return "high"
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity metrics
        avg_word_length = sum(len(w) for w in words) / len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Normalize to 0-1 scale
        complexity = min((avg_word_length / 10 + avg_sentence_length / 50) / 2, 1.0)
        return complexity
    
    def _identify_clause_type(self, clause: str) -> str:
        """Identify the type of clause for template matching."""
        clause_lower = clause.lower()
        
        if any(term in clause_lower for term in ['arbitration', 'arbitrator', 'binding']):
            return "arbitration"
        elif any(term in clause_lower for term in ['liability', 'damages', 'limitation']):
            return "limitation_of_liability"
        elif any(term in clause_lower for term in ['termination', 'terminate', 'cancel']):
            return "termination"
        elif any(term in clause_lower for term in ['intellectual property', 'copyright', 'patent']):
            return "intellectual_property"
        else:
            return "general"
    
    async def generate_negotiation_strategy(
        self,
        contract_text: str,
        party_position: str = "consumer",
        priorities: Optional[List[str]] = None
    ) -> NegotiationStrategy:
        """
        Generate comprehensive negotiation strategy for entire contract.
        
        Args:
            contract_text: Full contract text
            party_position: Negotiating party's position
            priorities: List of priority areas for negotiation
            
        Returns:
            Complete negotiation strategy
        """
        # Extract and analyze key clauses
        key_clauses = self._extract_key_clauses(contract_text)
        
        # Analyze each clause
        clause_analyses = []
        for clause in key_clauses:
            analysis = await self.analyze_clause_negotiability(
                clause, contract_text[:1000], party_position
            )
            clause_analyses.append(analysis)
        
        # Determine overall strategy
        avg_win_prob = np.mean([a.win_probability for a in clause_analyses])
        avg_leverage = np.mean([a.leverage_score for a in clause_analyses])
        
        if avg_win_prob > 0.65 and avg_leverage > 0.6:
            approach = "aggressive"
            timeline = "Push for quick resolution (1-2 weeks)"
        elif avg_win_prob > 0.45:
            approach = "balanced"
            timeline = "Standard negotiation timeline (2-4 weeks)"
        else:
            approach = "conservative"
            timeline = "Extended negotiation expected (4-6 weeks)"
        
        # Identify key negotiation points
        key_points = []
        concessions = []
        red_lines = []
        
        for analysis in clause_analyses:
            if analysis.win_probability > 0.7:
                key_points.extend(analysis.negotiation_points[:2])
            elif analysis.win_probability < 0.3:
                concessions.append(f"May need to accept: {analysis.original_clause[:100]}...")
            
            if analysis.risk_level == "high":
                red_lines.append(f"Cannot accept without modification: {analysis.original_clause[:100]}...")
        
        # Apply priorities if provided
        if priorities:
            key_points = [p for p in key_points if any(
                prio.lower() in p.lower() for prio in priorities
            )] + key_points
        
        return NegotiationStrategy(
            approach=approach,
            key_points=key_points[:5],
            concessions=concessions[:3],
            red_lines=red_lines[:3],
            expected_outcome=f"Favorable modifications to {len(key_points)} key provisions",
            success_probability=avg_win_prob,
            timeline=timeline
        )
    
    def _extract_key_clauses(self, contract_text: str) -> List[str]:
        """Extract key negotiable clauses from contract."""
        # Simple clause extraction - split by common section markers
        sections = contract_text.split('\n\n')
        
        key_terms = [
            'arbitration', 'liability', 'termination', 'payment',
            'intellectual property', 'confidentiality', 'warranty',
            'indemnification', 'dispute', 'governing law'
        ]
        
        key_clauses = []
        for section in sections:
            if any(term in section.lower() for term in key_terms):
                key_clauses.append(section[:1000])  # Limit length
        
        return key_clauses[:10]  # Return top 10 key clauses
    
    def train_win_probability_model(
        self,
        training_data: List[Tuple[Dict[str, float], bool]]
    ):
        """
        Train the win probability model on historical negotiation data.
        
        Args:
            training_data: List of (features, success) tuples
        """
        if not training_data:
            return
        
        X = np.array([features for features, _ in training_data])
        y = np.array([1 if success else 0 for _, success in training_data])
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train ensemble model
        from sklearn.ensemble import GradientBoostingClassifier
        self.win_probability_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.win_probability_model.fit(X_scaled, y)
        
        logger.info("Win probability model trained successfully")
    
    def save_models(self, path: str):
        """Save trained models to disk."""
        if self.win_probability_model:
            joblib.dump(self.win_probability_model, f"{path}/win_probability_model.pkl")
            joblib.dump(self.feature_scaler, f"{path}/feature_scaler.pkl")
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        try:
            self.win_probability_model = joblib.load(f"{path}/win_probability_model.pkl")
            self.feature_scaler = joblib.load(f"{path}/feature_scaler.pkl")
        except:
            logger.warning("Could not load pre-trained models")