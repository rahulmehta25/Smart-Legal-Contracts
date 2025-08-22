"""
Advanced anomaly detection for unusual clauses, hidden terms, and deceptive patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import re
import logging
from collections import Counter, defaultdict
import spacy
from textstat import flesch_reading_ease, gunning_fog
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


@dataclass
class AnomalyReport:
    """Comprehensive anomaly detection report."""
    clause_text: str
    anomaly_score: float  # 0-1, higher is more anomalous
    anomaly_types: List[str]
    hidden_terms: List[Dict[str, Any]]
    deceptive_patterns: List[str]
    unusual_features: Dict[str, float]
    fairness_score: float  # 0-1, higher is fairer
    explanations: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class HiddenTerm:
    """Represents a potentially hidden or obscured term."""
    text: str
    location: Tuple[int, int]  # start, end position
    obscurity_method: str  # buried, small_print, complex_language, cross_reference
    severity: str  # low, medium, high
    explanation: str


@dataclass
class FairnessAnalysis:
    """Analysis of clause fairness."""
    overall_score: float
    balance_score: float  # Balance between parties
    transparency_score: float
    reasonableness_score: float
    factors: Dict[str, float]
    issues: List[str]


class AnomalyDetectorNet(nn.Module):
    """Deep learning model for anomaly detection."""
    
    def __init__(self, input_dim: int = 768, latent_dim: int = 128):
        super(AnomalyDetectorNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder (for reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )
        
        # Anomaly classifier
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        anomaly_score = self.anomaly_classifier(latent)
        return reconstructed, latent, anomaly_score
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection."""
        reconstructed, _, _ = self.forward(x)
        error = F.mse_loss(reconstructed, x, reduction='none').mean(dim=1)
        return error


class DeceptivePatternDetector:
    """Detects deceptive and manipulative patterns in legal text."""
    
    def __init__(self):
        self.deceptive_patterns = self._compile_deceptive_patterns()
        self.dark_patterns = self._load_dark_patterns()
        
        # Load spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found, some features will be limited")
            self.nlp = None
    
    def _compile_deceptive_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for deceptive language."""
        return {
            "buried_waiver": re.compile(
                r"(?:notwithstanding|subject to|except as).{20,200}waiv",
                re.IGNORECASE | re.DOTALL
            ),
            "disguised_arbitration": re.compile(
                r"(?:dispute resolution|mediation).{0,50}(?:binding|final)",
                re.IGNORECASE
            ),
            "hidden_auto_renewal": re.compile(
                r"(?:automatically|unless.{0,30}notice).{0,50}renew",
                re.IGNORECASE
            ),
            "obscured_liability": re.compile(
                r"(?:to the extent|as permitted).{0,100}(?:liability|damages)",
                re.IGNORECASE
            ),
            "complex_negation": re.compile(
                r"(?:not\s+un|neither.{0,20}nor|without.{0,20}not)",
                re.IGNORECASE
            ),
            "circular_reference": re.compile(
                r"(?:as defined|pursuant to|subject to).{0,30}(?:Section|Exhibit|Schedule)",
                re.IGNORECASE
            ),
            "vague_modifier": re.compile(
                r"(?:substantially|materially|reasonably|approximately)",
                re.IGNORECASE
            ),
            "escape_clause": re.compile(
                r"(?:unless otherwise|except as|notwithstanding anything)",
                re.IGNORECASE
            )
        }
    
    def _load_dark_patterns(self) -> Dict[str, List[str]]:
        """Load dark patterns commonly used in contracts."""
        return {
            "roach_motel": [
                "easy to enter, hard to exit",
                "automatic renewal with difficult cancellation",
                "free trial converting to paid without clear notice"
            ],
            "bait_and_switch": [
                "advertised terms modified in fine print",
                "key benefits subject to undisclosed limitations",
                "pricing changes after commitment"
            ],
            "forced_continuity": [
                "silent auto-renewal",
                "difficult cancellation process",
                "penalties for early termination"
            ],
            "hidden_costs": [
                "fees disclosed separately from main terms",
                "additional charges in different sections",
                "cost escalation clauses"
            ],
            "misdirection": [
                "important terms in unexpected locations",
                "critical information in footnotes",
                "key provisions in exhibits or schedules"
            ]
        }
    
    def detect_deceptive_patterns(self, text: str) -> List[str]:
        """Detect deceptive patterns in text."""
        patterns_found = []
        
        for pattern_name, pattern_regex in self.deceptive_patterns.items():
            if pattern_regex.search(text):
                patterns_found.append(pattern_name)
        
        # Check for dark patterns
        text_lower = text.lower()
        for dark_pattern, indicators in self.dark_patterns.items():
            if any(indicator in text_lower for indicator in indicators):
                patterns_found.append(f"dark_pattern_{dark_pattern}")
        
        return patterns_found
    
    def analyze_complexity(self, text: str) -> Dict[str, float]:
        """Analyze text complexity to detect obfuscation."""
        metrics = {}
        
        # Reading ease scores
        try:
            metrics['flesch_reading_ease'] = flesch_reading_ease(text)
            metrics['gunning_fog'] = gunning_fog(text)
        except:
            metrics['flesch_reading_ease'] = 0
            metrics['gunning_fog'] = 20
        
        # Sentence complexity
        sentences = sent_tokenize(text)
        if sentences:
            metrics['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences])
            metrics['max_sentence_length'] = max([len(s.split()) for s in sentences])
        
        # Nesting depth (parentheses, commas)
        metrics['nesting_depth'] = self._calculate_nesting_depth(text)
        
        # Legal jargon density
        metrics['jargon_density'] = self._calculate_jargon_density(text)
        
        return metrics
    
    def _calculate_nesting_depth(self, text: str) -> int:
        """Calculate maximum nesting depth of parentheses and clauses."""
        max_depth = 0
        current_depth = 0
        
        for char in text:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth = max(0, current_depth - 1)
        
        # Also consider comma-separated subclauses
        comma_chains = re.findall(r'[^.!?]+(?:,\s*[^,]+){3,}', text)
        if comma_chains:
            max_comma_depth = max(len(chain.split(',')) for chain in comma_chains)
            max_depth = max(max_depth, max_comma_depth // 2)
        
        return max_depth
    
    def _calculate_jargon_density(self, text: str) -> float:
        """Calculate density of legal jargon."""
        jargon_terms = {
            'heretofore', 'hereinafter', 'whereas', 'aforementioned',
            'notwithstanding', 'pursuant', 'thereof', 'therein',
            'hereby', 'hereunder', 'foregoing', 'aforesaid',
            'wherein', 'whereof', 'thereunder', 'herewith'
        }
        
        words = word_tokenize(text.lower())
        jargon_count = sum(1 for word in words if word in jargon_terms)
        
        return jargon_count / max(len(words), 1)


class AnomalyDetector:
    """Comprehensive anomaly detection system for contract clauses."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize anomaly detector."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._initialize_models()
        
        # Load training data if available
        if model_path:
            self.load_models(model_path)
        
        # Initialize detectors
        self.deceptive_detector = DeceptivePatternDetector()
        
        # Statistical anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Clustering for pattern detection
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Fairness analyzer
        self.fairness_analyzer = FairnessAnalyzer()
    
    def _initialize_models(self):
        """Initialize deep learning models."""
        try:
            # BERT for embeddings
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            
            # Anomaly detection network
            self.anomaly_net = AnomalyDetectorNet().to(self.device)
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.tokenizer = None
            self.bert_model = None
            self.anomaly_net = None
    
    async def detect_anomalies(
        self,
        clause: str,
        context: Optional[Dict[str, Any]] = None,
        reference_clauses: Optional[List[str]] = None
    ) -> AnomalyReport:
        """
        Detect anomalies in a contract clause.
        
        Args:
            clause: Clause text to analyze
            context: Additional context
            reference_clauses: Normal clauses for comparison
            
        Returns:
            Comprehensive anomaly report
        """
        # Extract features
        features = self._extract_features(clause)
        
        # Deep learning anomaly detection
        dl_anomaly_score = self._deep_learning_detection(clause)
        
        # Statistical anomaly detection
        stat_anomaly_score = self._statistical_detection(features, reference_clauses)
        
        # Detect hidden terms
        hidden_terms = self._detect_hidden_terms(clause)
        
        # Detect deceptive patterns
        deceptive_patterns = self.deceptive_detector.detect_deceptive_patterns(clause)
        
        # Analyze unusual features
        unusual_features = self._analyze_unusual_features(clause, features)
        
        # Calculate fairness score
        fairness_analysis = self.fairness_analyzer.analyze(clause, context)
        
        # Combine anomaly scores
        overall_anomaly_score = self._combine_anomaly_scores(
            dl_anomaly_score, stat_anomaly_score, len(deceptive_patterns)
        )
        
        # Determine anomaly types
        anomaly_types = self._determine_anomaly_types(
            overall_anomaly_score, hidden_terms, deceptive_patterns, unusual_features
        )
        
        # Generate explanations
        explanations = self._generate_explanations(
            anomaly_types, hidden_terms, deceptive_patterns, fairness_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_anomaly_score, anomaly_types, fairness_analysis
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(features, reference_clauses)
        
        return AnomalyReport(
            clause_text=clause,
            anomaly_score=overall_anomaly_score,
            anomaly_types=anomaly_types,
            hidden_terms=[self._hidden_term_to_dict(ht) for ht in hidden_terms],
            deceptive_patterns=deceptive_patterns,
            unusual_features=unusual_features,
            fairness_score=fairness_analysis.overall_score,
            explanations=explanations,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def _extract_features(self, clause: str) -> np.ndarray:
        """Extract features from clause for anomaly detection."""
        features = []
        
        # Text statistics
        features.extend([
            len(clause),  # Length
            len(clause.split()),  # Word count
            len(sent_tokenize(clause)),  # Sentence count
            clause.count(','),  # Comma count
            clause.count('('),  # Parenthesis count
            clause.count(';'),  # Semicolon count
        ])
        
        # Complexity metrics
        complexity = self.deceptive_detector.analyze_complexity(clause)
        features.extend(list(complexity.values()))
        
        # Pattern counts
        pattern_counts = self._count_patterns(clause)
        features.extend(list(pattern_counts.values()))
        
        # Linguistic features
        if self.deceptive_detector.nlp:
            doc = self.deceptive_detector.nlp(clause)
            features.extend([
                len([ent for ent in doc.ents]),  # Entity count
                len([token for token in doc if token.pos_ == 'VERB']),  # Verb count
                len([token for token in doc if token.dep_ == 'neg']),  # Negation count
            ])
        
        return np.array(features)
    
    def _count_patterns(self, text: str) -> Dict[str, int]:
        """Count various patterns in text."""
        patterns = {
            'cross_references': len(re.findall(r'Section \d+|Exhibit [A-Z]', text)),
            'qualifiers': len(re.findall(r'\b(may|might|could|possibly)\b', text, re.I)),
            'absolutes': len(re.findall(r'\b(must|shall|will|always|never)\b', text, re.I)),
            'exceptions': len(re.findall(r'\b(except|unless|excluding)\b', text, re.I)),
            'vague_terms': len(re.findall(r'\b(reasonable|appropriate|adequate)\b', text, re.I)),
        }
        return patterns
    
    def _deep_learning_detection(self, clause: str) -> float:
        """Use deep learning for anomaly detection."""
        if not self.bert_model or not self.anomaly_net:
            return 0.5  # Default score if models not available
        
        try:
            # Generate embeddings
            inputs = self.tokenizer(
                clause,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Get reconstruction error
                error = self.anomaly_net.get_reconstruction_error(embeddings)
                
                # Get anomaly score
                _, _, anomaly_score = self.anomaly_net(embeddings)
            
            # Combine error and score
            combined_score = (error.item() * 0.4 + anomaly_score.item() * 0.6)
            return float(min(combined_score, 1.0))
            
        except Exception as e:
            logger.error(f"Deep learning detection error: {e}")
            return 0.5
    
    def _statistical_detection(
        self,
        features: np.ndarray,
        reference_clauses: Optional[List[str]]
    ) -> float:
        """Statistical anomaly detection."""
        if reference_clauses and len(reference_clauses) > 10:
            # Extract features from reference clauses
            ref_features = [self._extract_features(ref) for ref in reference_clauses]
            ref_features_array = np.array(ref_features)
            
            # Fit isolation forest
            scaled_refs = self.scaler.fit_transform(ref_features_array)
            self.isolation_forest.fit(scaled_refs)
            
            # Score the target clause
            scaled_target = self.scaler.transform(features.reshape(1, -1))
            anomaly_score = self.isolation_forest.decision_function(scaled_target)[0]
            
            # Convert to 0-1 scale (lower decision function = more anomalous)
            normalized_score = 1.0 / (1.0 + np.exp(anomaly_score))
            return float(normalized_score)
        else:
            # Use heuristic scoring if no references
            return self._heuristic_anomaly_score(features)
    
    def _heuristic_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate heuristic anomaly score."""
        score = 0.0
        
        # Check for extreme values
        if features[0] > 1000:  # Very long clause
            score += 0.2
        if features[4] > 5:  # Many parentheses
            score += 0.15
        if features[2] > 10:  # Many sentences
            score += 0.1
        
        # Complexity features (assuming they're in positions 6-10)
        if len(features) > 10:
            if features[6] < 30:  # Very low reading ease
                score += 0.2
            if features[7] > 15:  # High fog index
                score += 0.15
        
        return min(score, 1.0)
    
    def _detect_hidden_terms(self, clause: str) -> List[HiddenTerm]:
        """Detect potentially hidden or obscured terms."""
        hidden_terms = []
        
        # Check for buried important terms
        important_patterns = [
            (r'waive?\w*\s+\w+\s+rights?', 'waiver'),
            (r'arbitrat\w+', 'arbitration'),
            (r'class\s+action', 'class_action'),
            (r'liquidated\s+damages', 'liquidated_damages'),
            (r'indemnif\w+', 'indemnification'),
            (r'confidential\w*', 'confidentiality'),
            (r'non-compete|non-competition', 'non_compete')
        ]
        
        for pattern, term_type in important_patterns:
            matches = re.finditer(pattern, clause, re.IGNORECASE)
            for match in matches:
                # Check if buried (far from beginning, in middle of long sentence)
                position = match.start()
                if position > 200:  # Not near beginning
                    # Find sentence boundaries
                    sentence_start = clause.rfind('.', 0, position)
                    sentence_end = clause.find('.', position)
                    
                    if sentence_end - sentence_start > 150:  # Long sentence
                        hidden_terms.append(HiddenTerm(
                            text=match.group(),
                            location=(match.start(), match.end()),
                            obscurity_method='buried',
                            severity='high' if term_type in ['waiver', 'arbitration'] else 'medium',
                            explanation=f"{term_type} term buried in long text"
                        ))
        
        # Check for complex language obscurity
        complex_sections = self._find_complex_sections(clause)
        for section_start, section_end in complex_sections:
            section_text = clause[section_start:section_end]
            for pattern, term_type in important_patterns:
                if re.search(pattern, section_text, re.IGNORECASE):
                    hidden_terms.append(HiddenTerm(
                        text=section_text[:50] + "...",
                        location=(section_start, section_end),
                        obscurity_method='complex_language',
                        severity='medium',
                        explanation=f"{term_type} obscured by complex language"
                    ))
        
        return hidden_terms
    
    def _find_complex_sections(self, text: str) -> List[Tuple[int, int]]:
        """Find sections with complex language."""
        complex_sections = []
        sentences = sent_tokenize(text)
        
        current_pos = 0
        for sentence in sentences:
            # Check complexity
            if len(sentence.split()) > 40:  # Long sentence
                start = text.find(sentence, current_pos)
                end = start + len(sentence)
                complex_sections.append((start, end))
            current_pos += len(sentence)
        
        return complex_sections
    
    def _analyze_unusual_features(
        self,
        clause: str,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Analyze unusual features in the clause."""
        unusual = {}
        
        # Length anomaly
        word_count = len(clause.split())
        if word_count > 500:
            unusual['excessive_length'] = min(word_count / 1000, 1.0)
        elif word_count < 20:
            unusual['suspiciously_short'] = 1.0 - (word_count / 20)
        
        # Complexity anomaly
        complexity = self.deceptive_detector.analyze_complexity(clause)
        if complexity.get('flesch_reading_ease', 50) < 20:
            unusual['extreme_complexity'] = 1.0 - (complexity['flesch_reading_ease'] / 100)
        
        # Pattern anomalies
        patterns = self._count_patterns(clause)
        if patterns['cross_references'] > 5:
            unusual['excessive_cross_references'] = min(patterns['cross_references'] / 10, 1.0)
        if patterns['vague_terms'] > 10:
            unusual['excessive_vagueness'] = min(patterns['vague_terms'] / 20, 1.0)
        
        # Negation complexity
        negation_count = clause.lower().count('not') + clause.lower().count("n't")
        if negation_count > 5:
            unusual['complex_negation'] = min(negation_count / 10, 1.0)
        
        return unusual
    
    def _combine_anomaly_scores(
        self,
        dl_score: float,
        stat_score: float,
        deceptive_count: int
    ) -> float:
        """Combine different anomaly scores."""
        # Weight the scores
        deceptive_score = min(deceptive_count * 0.15, 1.0)
        
        # Weighted average
        combined = (
            dl_score * 0.4 +
            stat_score * 0.3 +
            deceptive_score * 0.3
        )
        
        return float(min(combined, 1.0))
    
    def _determine_anomaly_types(
        self,
        score: float,
        hidden_terms: List[HiddenTerm],
        deceptive_patterns: List[str],
        unusual_features: Dict[str, float]
    ) -> List[str]:
        """Determine types of anomalies present."""
        types = []
        
        if score > 0.7:
            types.append("high_anomaly_score")
        
        if hidden_terms:
            types.append("hidden_terms_detected")
        
        if deceptive_patterns:
            types.append("deceptive_patterns_found")
        
        for feature, value in unusual_features.items():
            if value > 0.5:
                types.append(f"unusual_{feature}")
        
        if not types and score > 0.5:
            types.append("moderate_anomaly")
        
        return types
    
    def _hidden_term_to_dict(self, ht: HiddenTerm) -> Dict[str, Any]:
        """Convert HiddenTerm to dictionary."""
        return {
            'text': ht.text,
            'location': ht.location,
            'obscurity_method': ht.obscurity_method,
            'severity': ht.severity,
            'explanation': ht.explanation
        }
    
    def _generate_explanations(
        self,
        anomaly_types: List[str],
        hidden_terms: List[HiddenTerm],
        deceptive_patterns: List[str],
        fairness_analysis: 'FairnessAnalysis'
    ) -> List[str]:
        """Generate explanations for detected anomalies."""
        explanations = []
        
        if "high_anomaly_score" in anomaly_types:
            explanations.append(
                "This clause shows significant deviation from typical contract language"
            )
        
        if hidden_terms:
            explanations.append(
                f"Found {len(hidden_terms)} potentially hidden or obscured terms"
            )
        
        if deceptive_patterns:
            pattern_desc = ", ".join(deceptive_patterns[:3])
            explanations.append(
                f"Detected deceptive patterns: {pattern_desc}"
            )
        
        if fairness_analysis.overall_score < 0.4:
            explanations.append(
                "Clause appears significantly unbalanced against one party"
            )
        
        if "unusual_excessive_length" in anomaly_types:
            explanations.append(
                "Clause is unusually long, potentially hiding important terms"
            )
        
        if "unusual_extreme_complexity" in anomaly_types:
            explanations.append(
                "Language complexity makes clause difficult to understand"
            )
        
        return explanations
    
    def _generate_recommendations(
        self,
        score: float,
        anomaly_types: List[str],
        fairness_analysis: 'FairnessAnalysis'
    ) -> List[str]:
        """Generate recommendations based on anomalies."""
        recommendations = []
        
        if score > 0.7:
            recommendations.append(
                "High anomaly score - careful review recommended"
            )
            recommendations.append(
                "Consider requesting simplified language"
            )
        
        if "hidden_terms_detected" in anomaly_types:
            recommendations.append(
                "Request all important terms be clearly highlighted"
            )
        
        if "deceptive_patterns_found" in anomaly_types:
            recommendations.append(
                "Ask for removal of deceptive language patterns"
            )
        
        if fairness_analysis.overall_score < 0.4:
            recommendations.append(
                "Negotiate for more balanced terms"
            )
        
        if "unusual_excessive_cross_references" in anomaly_types:
            recommendations.append(
                "Request consolidation of related terms"
            )
        
        return recommendations
    
    def _calculate_confidence(
        self,
        features: np.ndarray,
        reference_clauses: Optional[List[str]]
    ) -> float:
        """Calculate confidence in anomaly detection."""
        base_confidence = 0.7
        
        # Adjust based on reference data
        if reference_clauses:
            if len(reference_clauses) > 50:
                base_confidence += 0.2
            elif len(reference_clauses) > 20:
                base_confidence += 0.1
        
        # Adjust based on feature clarity
        if len(features) > 15:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)


class FairnessAnalyzer:
    """Analyzes fairness and balance in contract clauses."""
    
    def __init__(self):
        self.fairness_indicators = self._load_fairness_indicators()
        self.balance_patterns = self._compile_balance_patterns()
    
    def _load_fairness_indicators(self) -> Dict[str, List[str]]:
        """Load indicators of fair and unfair terms."""
        return {
            'unfair': [
                'sole discretion', 'absolute right', 'without cause',
                'immediately terminate', 'no liability', 'waive all rights',
                'non-refundable', 'automatic renewal', 'unilateral'
            ],
            'fair': [
                'mutual agreement', 'reasonable notice', 'good faith',
                'commercially reasonable', 'pro rata', 'equitable',
                'mutual consent', 'shared equally', 'balanced'
            ]
        }
    
    def _compile_balance_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for detecting balance/imbalance."""
        return {
            'one_sided_obligation': re.compile(
                r'(party a|provider|company) shall.{0,100}(party b|customer|user) shall not',
                re.IGNORECASE
            ),
            'mutual_obligation': re.compile(
                r'(both parties|each party|parties mutually)',
                re.IGNORECASE
            ),
            'unilateral_right': re.compile(
                r'(sole|exclusive|absolute) (right|discretion|option)',
                re.IGNORECASE
            ),
            'balanced_right': re.compile(
                r'(mutual|reciprocal|equal) (right|option|ability)',
                re.IGNORECASE
            )
        }
    
    def analyze(self, clause: str, context: Optional[Dict[str, Any]] = None) -> FairnessAnalysis:
        """Analyze fairness of a clause."""
        # Calculate component scores
        balance_score = self._calculate_balance_score(clause)
        transparency_score = self._calculate_transparency_score(clause)
        reasonableness_score = self._calculate_reasonableness_score(clause)
        
        # Factor analysis
        factors = {
            'balance': balance_score,
            'transparency': transparency_score,
            'reasonableness': reasonableness_score,
            'mutuality': self._assess_mutuality(clause),
            'proportionality': self._assess_proportionality(clause)
        }
        
        # Overall score
        overall_score = np.mean(list(factors.values()))
        
        # Identify issues
        issues = []
        if balance_score < 0.4:
            issues.append("Clause appears one-sided")
        if transparency_score < 0.4:
            issues.append("Terms lack transparency")
        if reasonableness_score < 0.4:
            issues.append("Terms may be unreasonable")
        
        return FairnessAnalysis(
            overall_score=float(overall_score),
            balance_score=balance_score,
            transparency_score=transparency_score,
            reasonableness_score=reasonableness_score,
            factors=factors,
            issues=issues
        )
    
    def _calculate_balance_score(self, clause: str) -> float:
        """Calculate balance between parties."""
        score = 0.5  # Start neutral
        
        clause_lower = clause.lower()
        
        # Check for unfair indicators
        unfair_count = sum(1 for term in self.fairness_indicators['unfair'] 
                          if term in clause_lower)
        
        # Check for fair indicators
        fair_count = sum(1 for term in self.fairness_indicators['fair'] 
                        if term in clause_lower)
        
        # Check patterns
        if self.balance_patterns['one_sided_obligation'].search(clause):
            score -= 0.3
        if self.balance_patterns['mutual_obligation'].search(clause):
            score += 0.2
        if self.balance_patterns['unilateral_right'].search(clause):
            score -= 0.2
        if self.balance_patterns['balanced_right'].search(clause):
            score += 0.2
        
        # Adjust for counts
        score -= unfair_count * 0.1
        score += fair_count * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_transparency_score(self, clause: str) -> float:
        """Calculate transparency of terms."""
        score = 1.0  # Start with full transparency
        
        # Deduct for complexity
        if len(clause.split()) > 200:
            score -= 0.2
        
        # Deduct for vague terms
        vague_terms = ['reasonable', 'appropriate', 'adequate', 'material', 'substantial']
        vague_count = sum(1 for term in vague_terms if term in clause.lower())
        score -= vague_count * 0.05
        
        # Deduct for cross-references
        cross_refs = len(re.findall(r'Section \d+|Exhibit [A-Z]', clause))
        score -= cross_refs * 0.05
        
        # Deduct for complex sentence structure
        sentences = sent_tokenize(clause)
        long_sentences = [s for s in sentences if len(s.split()) > 40]
        score -= len(long_sentences) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_reasonableness_score(self, clause: str) -> float:
        """Calculate reasonableness of terms."""
        score = 0.7  # Start with reasonable assumption
        
        clause_lower = clause.lower()
        
        # Check for unreasonable terms
        unreasonable_patterns = [
            'immediately', 'without notice', 'sole discretion',
            'unlimited liability', 'perpetual', 'irrevocable',
            'non-negotiable', 'absolute', 'unconditional'
        ]
        
        for pattern in unreasonable_patterns:
            if pattern in clause_lower:
                score -= 0.15
        
        # Check for reasonable modifiers
        reasonable_patterns = [
            'reasonable notice', 'good faith', 'commercially reasonable',
            'mutual agreement', 'upon consent', 'with notice'
        ]
        
        for pattern in reasonable_patterns:
            if pattern in clause_lower:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_mutuality(self, clause: str) -> float:
        """Assess mutuality of obligations and rights."""
        # Count party references
        party_a_obligations = len(re.findall(r'(Party A|Provider|Company) shall', clause, re.I))
        party_b_obligations = len(re.findall(r'(Party B|Customer|User) shall', clause, re.I))
        
        if party_a_obligations + party_b_obligations == 0:
            return 0.5  # No clear obligations
        
        # Calculate balance
        total = party_a_obligations + party_b_obligations
        balance = 1.0 - abs(party_a_obligations - party_b_obligations) / total
        
        return balance
    
    def _assess_proportionality(self, clause: str) -> float:
        """Assess proportionality of remedies and penalties."""
        score = 0.7  # Start with proportional assumption
        
        clause_lower = clause.lower()
        
        # Check for disproportionate remedies
        if 'liquidated damages' in clause_lower and 'actual damages' not in clause_lower:
            score -= 0.2
        
        if 'immediate termination' in clause_lower and 'cure period' not in clause_lower:
            score -= 0.2
        
        if 'unlimited' in clause_lower or 'all damages' in clause_lower:
            score -= 0.3
        
        # Check for proportionate terms
        if 'pro rata' in clause_lower or 'proportionate' in clause_lower:
            score += 0.2
        
        if 'reasonable damages' in clause_lower or 'actual losses' in clause_lower:
            score += 0.15
        
        return max(0.0, min(1.0, score))