"""
AI-powered precedent analyzer for case law outcome prediction and similar case retrieval.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import faiss
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class CasePrecedent:
    """Represents a legal precedent case."""
    case_id: str
    case_name: str
    jurisdiction: str
    year: int
    court_level: str  # supreme, appellate, district
    clause_type: str
    outcome: str  # upheld, struck_down, modified
    key_facts: List[str]
    legal_reasoning: str
    similarity_score: float
    relevance_score: float


@dataclass
class OutcomePrediction:
    """Prediction of case outcome based on precedents."""
    predicted_outcome: str
    confidence: float
    supporting_cases: List[CasePrecedent]
    opposing_cases: List[CasePrecedent]
    key_factors: Dict[str, float]
    judge_bias_factor: float
    jurisdiction_factor: float
    recommendation: str


@dataclass
class JudgeBias:
    """Analysis of judge or arbitrator bias."""
    name: str
    historical_rulings: Dict[str, float]  # clause_type -> plaintiff_win_rate
    ideological_leaning: str  # conservative, moderate, liberal
    corporate_favorability: float  # 0-1 scale
    consumer_favorability: float  # 0-1 scale
    reversal_rate: float
    years_experience: int


class LegalTransformer(nn.Module):
    """Transformer model for legal text understanding."""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, num_layers: int = 6):
        super(LegalTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads for different predictions
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # upheld, struck_down, modified
        )
        
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask=None):
        """Forward pass through transformer."""
        # Apply transformer
        transformed = self.transformer(x, src_key_padding_mask=mask)
        
        # Pool for classification
        pooled = transformed.mean(dim=1)
        
        # Predict outcome
        outcome = self.outcome_head(pooled)
        
        return outcome, transformed
    
    def compute_similarity(self, x1, x2):
        """Compute similarity between two legal texts."""
        # Concatenate representations
        combined = torch.cat([x1, x2], dim=-1)
        similarity = self.similarity_head(combined)
        return similarity


class PrecedentAnalyzer:
    """Analyzes legal precedents for outcome prediction and case retrieval."""
    
    def __init__(self, precedent_db_path: Optional[str] = None):
        """Initialize precedent analyzer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transformer models
        self._initialize_models()
        
        # Load precedent database
        self.precedent_database = self._load_precedent_database(precedent_db_path)
        
        # Initialize vector index for similarity search
        self._initialize_vector_index()
        
        # Judge bias database
        self.judge_database = self._load_judge_database()
        
        # Outcome prediction models
        self.outcome_predictor = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7
        )
        self.feature_scaler = StandardScaler()
        
        # Case law patterns
        self.legal_patterns = self._compile_legal_patterns()
    
    def _initialize_models(self):
        """Initialize transformer models for legal analysis."""
        try:
            # Legal-BERT for embeddings
            self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased').to(self.device)
            
            # Custom legal transformer
            self.legal_transformer = LegalTransformer().to(self.device)
            
        except Exception as e:
            logger.warning(f"Could not load Legal-BERT, using fallback: {e}")
            # Fallback to standard BERT
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            self.legal_transformer = LegalTransformer().to(self.device)
        
        # TF-IDF for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
    
    def _load_precedent_database(self, path: Optional[str]) -> List[Dict]:
        """Load database of legal precedents."""
        # Simulated precedent database
        precedents = [
            {
                "case_id": "2023-SC-001",
                "case_name": "Consumer v. TechCorp",
                "jurisdiction": "US Supreme Court",
                "year": 2023,
                "court_level": "supreme",
                "clause_type": "arbitration",
                "outcome": "struck_down",
                "key_facts": [
                    "Mandatory arbitration clause in consumer contract",
                    "Class action waiver included",
                    "No opt-out provision"
                ],
                "legal_reasoning": "Unconscionable due to lack of mutuality and excessive cost burden on consumers",
                "keywords": ["arbitration", "class action", "unconscionable", "consumer protection"]
            },
            {
                "case_id": "2022-CA-047",
                "case_name": "SmallBiz v. Enterprise",
                "jurisdiction": "9th Circuit",
                "year": 2022,
                "court_level": "appellate",
                "clause_type": "liability_limitation",
                "outcome": "modified",
                "key_facts": [
                    "Complete liability waiver",
                    "B2B contract",
                    "Gross negligence involved"
                ],
                "legal_reasoning": "Limitation valid except for gross negligence and willful misconduct",
                "keywords": ["liability", "limitation", "gross negligence", "B2B"]
            },
            {
                "case_id": "2023-NY-112",
                "case_name": "Employee v. FinanceInc",
                "jurisdiction": "NY State Court",
                "year": 2023,
                "court_level": "district",
                "clause_type": "non_compete",
                "outcome": "struck_down",
                "key_facts": [
                    "2-year non-compete period",
                    "Nationwide geographic scope",
                    "Entry-level employee"
                ],
                "legal_reasoning": "Overly broad in temporal and geographic scope for employee level",
                "keywords": ["non-compete", "employment", "reasonable scope", "restraint of trade"]
            }
        ]
        
        # Load from file if provided
        if path:
            try:
                with open(path, 'r') as f:
                    precedents = json.load(f)
            except:
                logger.warning(f"Could not load precedent database from {path}")
        
        return precedents
    
    def _initialize_vector_index(self):
        """Initialize FAISS index for similarity search."""
        if not self.precedent_database:
            self.vector_index = None
            return
        
        # Generate embeddings for all precedents
        embeddings = []
        for precedent in self.precedent_database:
            text = f"{precedent['case_name']} {precedent['legal_reasoning']} {' '.join(precedent['key_facts'])}"
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)
        
        # Create FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        dimension = embeddings_np.shape[1]
        
        # Use L2 distance
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings_np)
        
        logger.info(f"Initialized vector index with {len(embeddings)} precedents")
    
    def _load_judge_database(self) -> Dict[str, JudgeBias]:
        """Load database of judge/arbitrator bias information."""
        # Simulated judge database
        judges = {
            "Hon. Sarah Mitchell": JudgeBias(
                name="Hon. Sarah Mitchell",
                historical_rulings={
                    "arbitration": 0.3,  # Rules for plaintiff 30% of time
                    "liability_limitation": 0.4,
                    "non_compete": 0.6,
                    "ip_assignment": 0.5
                },
                ideological_leaning="liberal",
                corporate_favorability=0.3,
                consumer_favorability=0.7,
                reversal_rate=0.15,
                years_experience=15
            ),
            "Hon. Robert Stevens": JudgeBias(
                name="Hon. Robert Stevens",
                historical_rulings={
                    "arbitration": 0.7,
                    "liability_limitation": 0.6,
                    "non_compete": 0.4,
                    "ip_assignment": 0.6
                },
                ideological_leaning="conservative",
                corporate_favorability=0.7,
                consumer_favorability=0.3,
                reversal_rate=0.20,
                years_experience=20
            ),
            "Arbitrator John Chen": JudgeBias(
                name="Arbitrator John Chen",
                historical_rulings={
                    "arbitration": 0.5,
                    "liability_limitation": 0.5,
                    "non_compete": 0.5,
                    "ip_assignment": 0.5
                },
                ideological_leaning="moderate",
                corporate_favorability=0.5,
                consumer_favorability=0.5,
                reversal_rate=0.10,
                years_experience=10
            )
        }
        
        return judges
    
    def _compile_legal_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for legal concept extraction."""
        return {
            "unconscionability": re.compile(
                r"(unconscionable|procedural unconscionability|substantive unconscionability|"
                r"adhesion contract|unequal bargaining|oppressive)",
                re.IGNORECASE
            ),
            "public_policy": re.compile(
                r"(public policy|against public policy|void as against|public interest|"
                r"statutory violation|illegal)",
                re.IGNORECASE
            ),
            "mutuality": re.compile(
                r"(mutuality|lack of mutuality|one-sided|unilateral|bilateral)",
                re.IGNORECASE
            ),
            "consideration": re.compile(
                r"(consideration|adequate consideration|illusory|nominal consideration)",
                re.IGNORECASE
            ),
            "severability": re.compile(
                r"(severability|severable|blue pencil|strike|reformation)",
                re.IGNORECASE
            )
        }
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for legal text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()
    
    async def find_similar_cases(
        self,
        clause: str,
        jurisdiction: Optional[str] = None,
        limit: int = 5
    ) -> List[CasePrecedent]:
        """
        Find similar legal precedents for a clause.
        
        Args:
            clause: Clause text to analyze
            jurisdiction: Preferred jurisdiction
            limit: Maximum number of cases to return
            
        Returns:
            List of similar precedent cases
        """
        if not self.vector_index:
            return []
        
        # Generate embedding for query clause
        query_embedding = self._generate_embedding(clause)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search similar cases
        distances, indices = self.vector_index.search(query_embedding, limit * 2)
        
        # Convert to CasePrecedent objects
        similar_cases = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.precedent_database):
                precedent = self.precedent_database[idx]
                
                # Calculate relevance score
                relevance = self._calculate_relevance(clause, precedent, jurisdiction)
                
                case = CasePrecedent(
                    case_id=precedent['case_id'],
                    case_name=precedent['case_name'],
                    jurisdiction=precedent['jurisdiction'],
                    year=precedent['year'],
                    court_level=precedent['court_level'],
                    clause_type=precedent['clause_type'],
                    outcome=precedent['outcome'],
                    key_facts=precedent['key_facts'],
                    legal_reasoning=precedent['legal_reasoning'],
                    similarity_score=1.0 / (1.0 + distance),  # Convert distance to similarity
                    relevance_score=relevance
                )
                similar_cases.append(case)
        
        # Sort by combined score and limit
        similar_cases.sort(
            key=lambda x: (x.similarity_score * 0.6 + x.relevance_score * 0.4),
            reverse=True
        )
        
        return similar_cases[:limit]
    
    def _calculate_relevance(
        self,
        clause: str,
        precedent: Dict,
        jurisdiction: Optional[str]
    ) -> float:
        """Calculate relevance score for a precedent."""
        relevance = 0.5  # Base relevance
        
        # Jurisdiction match
        if jurisdiction and jurisdiction in precedent.get('jurisdiction', ''):
            relevance += 0.2
        
        # Court level weight
        court_weights = {"supreme": 0.3, "appellate": 0.2, "district": 0.1}
        relevance += court_weights.get(precedent.get('court_level', ''), 0)
        
        # Recency factor
        current_year = datetime.now().year
        years_old = current_year - precedent.get('year', current_year)
        if years_old < 2:
            relevance += 0.15
        elif years_old < 5:
            relevance += 0.1
        elif years_old > 10:
            relevance -= 0.1
        
        # Keyword overlap
        clause_lower = clause.lower()
        keyword_matches = sum(
            1 for keyword in precedent.get('keywords', [])
            if keyword in clause_lower
        )
        relevance += min(keyword_matches * 0.05, 0.2)
        
        return min(relevance, 1.0)
    
    async def predict_outcome(
        self,
        clause: str,
        case_context: Dict[str, Any],
        judge_name: Optional[str] = None
    ) -> OutcomePrediction:
        """
        Predict likely outcome for a clause challenge.
        
        Args:
            clause: Clause text being challenged
            case_context: Context including parties, jurisdiction, etc.
            judge_name: Name of judge/arbitrator if known
            
        Returns:
            Outcome prediction with confidence and supporting cases
        """
        # Find similar cases
        similar_cases = await self.find_similar_cases(
            clause,
            case_context.get('jurisdiction'),
            limit=10
        )
        
        # Extract features for prediction
        features = self._extract_prediction_features(clause, case_context, similar_cases)
        
        # Get judge bias factor
        judge_bias = 0.5  # Neutral default
        if judge_name and judge_name in self.judge_database:
            judge_info = self.judge_database[judge_name]
            clause_type = self._identify_clause_type(clause)
            judge_bias = judge_info.historical_rulings.get(clause_type, 0.5)
        
        # Predict outcome using ensemble
        prediction = self._ensemble_prediction(features, similar_cases, judge_bias)
        
        # Separate supporting and opposing cases
        supporting_cases = [c for c in similar_cases if c.outcome == prediction['outcome']]
        opposing_cases = [c for c in similar_cases if c.outcome != prediction['outcome']]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            prediction,
            supporting_cases,
            opposing_cases,
            judge_bias
        )
        
        return OutcomePrediction(
            predicted_outcome=prediction['outcome'],
            confidence=prediction['confidence'],
            supporting_cases=supporting_cases[:3],
            opposing_cases=opposing_cases[:2],
            key_factors=prediction['factors'],
            judge_bias_factor=judge_bias,
            jurisdiction_factor=features.get('jurisdiction_factor', 0.5),
            recommendation=recommendation
        )
    
    def _extract_prediction_features(
        self,
        clause: str,
        context: Dict[str, Any],
        similar_cases: List[CasePrecedent]
    ) -> Dict[str, float]:
        """Extract features for outcome prediction."""
        features = {}
        
        # Text-based features
        clause_lower = clause.lower()
        features['length'] = len(clause.split())
        features['complexity'] = self._calculate_complexity(clause)
        
        # Legal concept presence
        for concept, pattern in self.legal_patterns.items():
            features[f'has_{concept}'] = 1.0 if pattern.search(clause) else 0.0
        
        # Context features
        features['is_consumer'] = 1.0 if context.get('party_type') == 'consumer' else 0.0
        features['is_employment'] = 1.0 if context.get('contract_type') == 'employment' else 0.0
        features['is_b2b'] = 1.0 if context.get('contract_type') == 'b2b' else 0.0
        
        # Jurisdiction features
        jurisdiction = context.get('jurisdiction', '')
        features['jurisdiction_factor'] = self._get_jurisdiction_factor(jurisdiction)
        
        # Similar case statistics
        if similar_cases:
            outcomes = [c.outcome for c in similar_cases]
            features['similar_upheld_rate'] = outcomes.count('upheld') / len(outcomes)
            features['similar_struck_rate'] = outcomes.count('struck_down') / len(outcomes)
            features['similar_modified_rate'] = outcomes.count('modified') / len(outcomes)
            
            # Average similarity and relevance
            features['avg_similarity'] = np.mean([c.similarity_score for c in similar_cases])
            features['avg_relevance'] = np.mean([c.relevance_score for c in similar_cases])
            
            # Court level distribution
            court_levels = [c.court_level for c in similar_cases]
            features['supreme_court_ratio'] = court_levels.count('supreme') / len(court_levels)
        
        return features
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Various complexity metrics
        avg_word_length = sum(len(w) for w in words) / len(words)
        unique_words = len(set(words)) / len(words)
        
        # Legal jargon density
        legal_terms = [
            'whereas', 'herein', 'thereof', 'notwithstanding',
            'heretofore', 'hereinafter', 'aforementioned'
        ]
        jargon_density = sum(1 for w in words if w.lower() in legal_terms) / len(words)
        
        complexity = (avg_word_length / 10 + (1 - unique_words) + jargon_density * 5) / 3
        return min(complexity, 1.0)
    
    def _identify_clause_type(self, clause: str) -> str:
        """Identify the type of legal clause."""
        clause_lower = clause.lower()
        
        if 'arbitration' in clause_lower or 'arbitrator' in clause_lower:
            return 'arbitration'
        elif 'liability' in clause_lower or 'damages' in clause_lower:
            return 'liability_limitation'
        elif 'non-compete' in clause_lower or 'competition' in clause_lower:
            return 'non_compete'
        elif 'intellectual property' in clause_lower or 'copyright' in clause_lower:
            return 'ip_assignment'
        elif 'confidential' in clause_lower or 'nda' in clause_lower:
            return 'confidentiality'
        else:
            return 'general'
    
    def _get_jurisdiction_factor(self, jurisdiction: str) -> float:
        """Get jurisdiction favorability factor."""
        # Simplified jurisdiction favorability scores
        favorable_jurisdictions = {
            'california': 0.7,  # Consumer-friendly
            'new york': 0.5,    # Balanced
            'delaware': 0.3,    # Corporate-friendly
            'texas': 0.4,
            'illinois': 0.5,
            '9th circuit': 0.6,
            '2nd circuit': 0.5,
            'supreme court': 0.5
        }
        
        jurisdiction_lower = jurisdiction.lower() if jurisdiction else ''
        
        for key, factor in favorable_jurisdictions.items():
            if key in jurisdiction_lower:
                return factor
        
        return 0.5  # Default neutral
    
    def _ensemble_prediction(
        self,
        features: Dict[str, float],
        similar_cases: List[CasePrecedent],
        judge_bias: float
    ) -> Dict[str, Any]:
        """Ensemble prediction using multiple methods."""
        predictions = []
        confidences = []
        
        # Method 1: Similar case voting
        if similar_cases:
            outcomes = [c.outcome for c in similar_cases[:5]]
            weights = [c.similarity_score * c.relevance_score for c in similar_cases[:5]]
            
            outcome_scores = defaultdict(float)
            for outcome, weight in zip(outcomes, weights):
                outcome_scores[outcome] += weight
            
            vote_prediction = max(outcome_scores.items(), key=lambda x: x[1])[0]
            vote_confidence = outcome_scores[vote_prediction] / sum(outcome_scores.values())
            
            predictions.append(vote_prediction)
            confidences.append(vote_confidence)
        
        # Method 2: Feature-based prediction
        feature_prediction = self._feature_based_prediction(features)
        predictions.append(feature_prediction['outcome'])
        confidences.append(feature_prediction['confidence'])
        
        # Method 3: Judge bias adjustment
        if judge_bias != 0.5:
            if judge_bias < 0.4:
                bias_prediction = 'struck_down'
                bias_confidence = 1.0 - judge_bias
            elif judge_bias > 0.6:
                bias_prediction = 'upheld'
                bias_confidence = judge_bias
            else:
                bias_prediction = 'modified'
                bias_confidence = 0.5
            
            predictions.append(bias_prediction)
            confidences.append(bias_confidence * 0.7)  # Weight judge bias less
        
        # Combine predictions
        outcome_counts = defaultdict(float)
        for pred, conf in zip(predictions, confidences):
            outcome_counts[pred] += conf
        
        final_outcome = max(outcome_counts.items(), key=lambda x: x[1])[0]
        final_confidence = outcome_counts[final_outcome] / sum(outcome_counts.values())
        
        # Key factors
        key_factors = {
            'similar_cases': confidences[0] if confidences else 0.0,
            'legal_patterns': features.get('has_unconscionability', 0.0),
            'judge_bias': judge_bias,
            'jurisdiction': features.get('jurisdiction_factor', 0.5),
            'complexity': features.get('complexity', 0.5)
        }
        
        return {
            'outcome': final_outcome,
            'confidence': final_confidence,
            'factors': key_factors
        }
    
    def _feature_based_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict outcome based on features."""
        score = 0.5  # Base score
        
        # Adjust based on legal concepts
        if features.get('has_unconscionability', 0) > 0:
            score -= 0.3
        if features.get('has_public_policy', 0) > 0:
            score -= 0.2
        if features.get('has_mutuality', 0) == 0:
            score -= 0.15
        
        # Party type adjustments
        if features.get('is_consumer', 0) > 0:
            score -= 0.1
        if features.get('is_b2b', 0) > 0:
            score += 0.1
        
        # Complexity factor
        if features.get('complexity', 0) > 0.7:
            score -= 0.1
        
        # Jurisdiction factor
        score += (features.get('jurisdiction_factor', 0.5) - 0.5) * 0.3
        
        # Determine outcome
        if score < 0.35:
            outcome = 'struck_down'
        elif score > 0.65:
            outcome = 'upheld'
        else:
            outcome = 'modified'
        
        confidence = abs(score - 0.5) * 2  # Convert to confidence
        
        return {'outcome': outcome, 'confidence': min(confidence, 0.9)}
    
    def _generate_recommendation(
        self,
        prediction: Dict[str, Any],
        supporting_cases: List[CasePrecedent],
        opposing_cases: List[CasePrecedent],
        judge_bias: float
    ) -> str:
        """Generate strategic recommendation based on prediction."""
        recommendations = []
        
        outcome = prediction['outcome']
        confidence = prediction['confidence']
        
        # High confidence recommendations
        if confidence > 0.7:
            if outcome == 'struck_down':
                recommendations.append(
                    "Strong position to challenge this clause. Focus on unconscionability and lack of mutuality."
                )
            elif outcome == 'upheld':
                recommendations.append(
                    "Clause likely to be upheld. Consider negotiation rather than litigation."
                )
            else:
                recommendations.append(
                    "Clause may be modified. Propose specific reasonable alternatives."
                )
        else:
            recommendations.append(
                "Outcome uncertain. Strengthen arguments with additional precedents."
            )
        
        # Judge-specific strategy
        if judge_bias < 0.4:
            recommendations.append(
                "Judge historically favors challengers. Emphasize consumer/employee protection."
            )
        elif judge_bias > 0.6:
            recommendations.append(
                "Judge tends to uphold contracts. Focus on clear statutory violations or unconscionability."
            )
        
        # Case strategy
        if supporting_cases:
            top_case = supporting_cases[0]
            recommendations.append(
                f"Cite {top_case.case_name} ({top_case.year}) as primary authority."
            )
        
        if opposing_cases and len(opposing_cases) > len(supporting_cases):
            recommendations.append(
                "Distinguish opposing precedents based on factual differences."
            )
        
        return " ".join(recommendations[:3])
    
    def analyze_judge_bias(
        self,
        judge_name: str,
        clause_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze judge or arbitrator bias patterns.
        
        Args:
            judge_name: Name of judge/arbitrator
            clause_type: Specific clause type to analyze
            
        Returns:
            Detailed bias analysis
        """
        if judge_name not in self.judge_database:
            return {
                'error': 'Judge not found in database',
                'recommendation': 'Conduct independent research on judge background'
            }
        
        judge = self.judge_database[judge_name]
        
        analysis = {
            'name': judge.name,
            'ideological_leaning': judge.ideological_leaning,
            'years_experience': judge.years_experience,
            'reversal_rate': judge.reversal_rate,
            'corporate_favorability': judge.corporate_favorability,
            'consumer_favorability': judge.consumer_favorability
        }
        
        if clause_type:
            analysis['clause_specific_rate'] = judge.historical_rulings.get(clause_type, 0.5)
            
            if analysis['clause_specific_rate'] < 0.4:
                analysis['strategy'] = f"Judge favors challengers in {clause_type} cases"
            elif analysis['clause_specific_rate'] > 0.6:
                analysis['strategy'] = f"Judge tends to uphold {clause_type} clauses"
            else:
                analysis['strategy'] = f"Judge shows balanced approach to {clause_type}"
        
        # Overall recommendations
        recommendations = []
        
        if judge.ideological_leaning == 'liberal':
            recommendations.append("Emphasize fairness and power imbalance")
        elif judge.ideological_leaning == 'conservative':
            recommendations.append("Focus on contract terms and business necessity")
        
        if judge.reversal_rate > 0.15:
            recommendations.append("Consider appeal prospects if unfavorable ruling")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def calculate_success_rate(
        self,
        clause: str,
        context: Dict[str, Any],
        historical_outcomes: Optional[List[Tuple[str, str]]] = None
    ) -> float:
        """
        Calculate success rate for challenging a clause.
        
        Args:
            clause: Clause text
            context: Case context
            historical_outcomes: List of (similar_clause, outcome) tuples
            
        Returns:
            Estimated success rate (0-1)
        """
        base_rate = 0.5
        
        # Find similar cases
        similar_cases = asyncio.run(
            self.find_similar_cases(clause, context.get('jurisdiction'), limit=20)
        )
        
        if similar_cases:
            # Calculate success rate from precedents
            struck_down = sum(1 for c in similar_cases if c.outcome == 'struck_down')
            modified = sum(1 for c in similar_cases if c.outcome == 'modified')
            total = len(similar_cases)
            
            # Consider modified as partial success
            success_rate = (struck_down + modified * 0.5) / total
            
            # Weight by similarity and relevance
            weighted_rate = sum(
                (1 if c.outcome == 'struck_down' else 0.5 if c.outcome == 'modified' else 0) *
                c.similarity_score * c.relevance_score
                for c in similar_cases
            ) / sum(c.similarity_score * c.relevance_score for c in similar_cases)
            
            base_rate = (success_rate + weighted_rate) / 2
        
        # Adjust for context factors
        if context.get('party_type') == 'consumer':
            base_rate += 0.1
        if context.get('jurisdiction') and 'california' in context['jurisdiction'].lower():
            base_rate += 0.05
        
        # Historical outcomes adjustment
        if historical_outcomes:
            historical_success = sum(
                1 for _, outcome in historical_outcomes
                if outcome in ['struck_down', 'modified']
            ) / len(historical_outcomes)
            base_rate = (base_rate + historical_success) / 2
        
        return min(max(base_rate, 0.05), 0.95)  # Bound between 5% and 95%