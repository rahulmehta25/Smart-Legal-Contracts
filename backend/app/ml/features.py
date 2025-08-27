"""
Feature Extraction Module for Arbitration Clause Detection

This module provides comprehensive feature extraction capabilities including:
- TF-IDF vectorization
- N-gram features  
- Legal keyword presence
- Sentence structure analysis
- Document statistics
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from scipy.sparse import hstack, csr_matrix
import joblib
from pathlib import Path

class LegalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature extractor for legal text analysis
    """
    
    def __init__(self, 
                 max_features_tfidf: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 use_legal_keywords: bool = True,
                 use_structure_features: bool = True,
                 use_statistical_features: bool = True):
        
        self.max_features_tfidf = max_features_tfidf
        self.ngram_range = ngram_range
        self.use_legal_keywords = use_legal_keywords
        self.use_structure_features = use_structure_features
        self.use_statistical_features = use_statistical_features
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.ngram_vectorizer = None
        
        # Legal keyword dictionaries
        self.arbitration_keywords = {
            'strong': [
                'binding arbitration', 'mandatory arbitration', 'individual arbitration',
                'arbitration agreement', 'arbitration clause', 'arbitrable dispute',
                'arbitral tribunal', 'arbitral award', 'arbitral proceedings'
            ],
            'medium': [
                'arbitration', 'arbitrator', 'arbitral', 'arbitrate', 'arbitrated',
                'dispute resolution', 'alternative dispute resolution', 'ADR',
                'JAMS', 'AAA', 'American Arbitration Association', 
                'International Chamber of Commerce', 'ICC'
            ],
            'weak': [
                'mediation', 'settlement', 'resolution', 'neutral', 'facilitate'
            ]
        }
        
        self.waiver_keywords = {
            'jury_trial': [
                'jury trial waiver', 'waive jury trial', 'right to jury trial',
                'trial by jury', 'jury trial right', 'waive right to jury'
            ],
            'class_action': [
                'class action waiver', 'class action prohibition', 'class proceedings',
                'representative action', 'collective action', 'class lawsuit',
                'class action suit', 'class or representative'
            ],
            'court': [
                'court proceedings', 'judicial proceedings', 'litigation',
                'right to court', 'court action', 'legal action'
            ]
        }
        
        self.legal_entities = [
            'Delaware', 'California', 'New York', 'federal court', 'state court',
            'district court', 'circuit court', 'supreme court', 'jurisdiction',
            'venue', 'governing law', 'applicable law'
        ]
        
        self.enforcement_terms = [
            'binding', 'mandatory', 'final', 'conclusive', 'exclusive',
            'sole remedy', 'only remedy', 'shall', 'must', 'required',
            'compelled', 'enforced', 'enforceable'
        ]
        
        # Negation patterns
        self.negation_patterns = [
            r'\b(?:not|no|never|without|except|excluding|other than)\b',
            r'\b(?:may not|cannot|shall not|will not|won\'t|can\'t)\b'
        ]
        
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Some features will be limited.")
            self.nlp = None
    
    def fit(self, X: List[str], y=None):
        """Fit the feature extractors"""
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in X]
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features_tfidf,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',
            max_df=0.95,
            min_df=2
        )
        self.tfidf_vectorizer.fit(processed_texts)
        
        # Fit n-gram vectorizer for specific patterns
        self.ngram_vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            max_features=1000,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',
            binary=True  # Just presence/absence
        )
        self.ngram_vectorizer.fit(processed_texts)
        
        return self
    
    def transform(self, X: List[str]) -> csr_matrix:
        """Transform texts into feature matrix"""
        
        feature_matrices = []
        
        # TF-IDF features
        processed_texts = [self._preprocess_text(text) for text in X]
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        feature_matrices.append(tfidf_features)
        
        # N-gram features
        if self.ngram_vectorizer:
            ngram_features = self.ngram_vectorizer.transform(processed_texts)
            feature_matrices.append(ngram_features)
        
        # Legal keyword features
        if self.use_legal_keywords:
            keyword_features = self._extract_keyword_features(X)
            feature_matrices.append(csr_matrix(keyword_features))
        
        # Structure features
        if self.use_structure_features:
            structure_features = self._extract_structure_features(X)
            feature_matrices.append(csr_matrix(structure_features))
        
        # Statistical features
        if self.use_statistical_features:
            statistical_features = self._extract_statistical_features(X)
            feature_matrices.append(csr_matrix(statistical_features))
        
        # Combine all features
        combined_features = hstack(feature_matrices)
        return combined_features
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for feature extraction"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize some legal terms
        text = re.sub(r'\b(american arbitration association|aaa)\b', 'american_arbitration_association', text)
        text = re.sub(r'\b(international chamber of commerce|icc)\b', 'international_chamber_commerce', text)
        text = re.sub(r'\bjams\b', 'jams_arbitration', text)
        
        return text.strip()
    
    def _extract_keyword_features(self, texts: List[str]) -> np.ndarray:
        """Extract legal keyword features"""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            text_features = []
            
            # Arbitration keyword counts by strength
            for strength, keywords in self.arbitration_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                text_features.append(count)
            
            # Waiver keyword features
            for category, keywords in self.waiver_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                text_features.append(count)
                
                # Binary presence feature
                has_waiver = any(keyword in text_lower for keyword in keywords)
                text_features.append(1 if has_waiver else 0)
            
            # Legal entity mentions
            entity_count = sum(1 for entity in self.legal_entities if entity.lower() in text_lower)
            text_features.append(entity_count)
            
            # Enforcement terms
            enforcement_count = sum(1 for term in self.enforcement_terms if term.lower() in text_lower)
            text_features.append(enforcement_count)
            
            # Specific arbitration patterns
            arbitration_patterns = [
                r'\barbitration\s+(?:shall|will|must)\s+be\b',
                r'\bbinding\s+arbitration\b',
                r'\bmandatory\s+arbitration\b',
                r'\bwaive.*jury\s+trial\b',
                r'\bclass\s+action\s+waiver\b',
                r'\bdispute.*arbitration\b',
                r'\barbitration.*dispute\b'
            ]
            
            for pattern in arbitration_patterns:
                matches = len(re.findall(pattern, text_lower))
                text_features.append(matches)
            
            # Negation context (important for understanding opt-out clauses)
            negation_score = 0
            for negation_pattern in self.negation_patterns:
                negation_matches = re.findall(negation_pattern, text_lower)
                negation_score += len(negation_matches)
            text_features.append(negation_score)
            
            features.append(text_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_structure_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentence and document structure features"""
        features = []
        
        for text in texts:
            text_features = []
            
            # Basic structure
            sentence_count = len(re.split(r'[.!?]+', text))
            text_features.append(sentence_count)
            
            # Capitalization patterns (legal docs often have caps for emphasis)
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            text_features.append(caps_ratio)
            
            all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
            text_features.append(all_caps_words)
            
            # Punctuation patterns
            exclamation_count = text.count('!')
            question_count = text.count('?')
            semicolon_count = text.count(';')
            colon_count = text.count(':')
            
            text_features.extend([exclamation_count, question_count, semicolon_count, colon_count])
            
            # Parentheses (often used for citations or clarifications)
            paren_count = text.count('(')
            text_features.append(paren_count)
            
            # Legal formatting patterns
            has_section_numbering = 1 if re.search(r'\b\(\w+\)\b', text) else 0
            has_legal_citations = 1 if re.search(r'\b\d+\s+[A-Z][a-z]+\s+\d+\b', text) else 0
            
            text_features.extend([has_section_numbering, has_legal_citations])
            
            # Sentence complexity (using spaCy if available)
            if self.nlp:
                try:
                    doc = self.nlp(text[:1000])  # Limit length for performance
                    avg_sent_length = np.mean([len(sent.text.split()) for sent in doc.sents]) if doc.sents else 0
                    text_features.append(avg_sent_length)
                except:
                    text_features.append(0)
            else:
                # Fallback: estimate complexity
                words = text.split()
                avg_word_length = np.mean([len(word) for word in words]) if words else 0
                text_features.append(avg_word_length)
            
            features.append(text_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features about the text"""
        features = []
        
        for text in texts:
            text_features = []
            
            # Basic statistics
            char_count = len(text)
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            
            text_features.extend([char_count, word_count, unique_words])
            
            # Ratios
            avg_word_length = char_count / max(word_count, 1)
            unique_word_ratio = unique_words / max(word_count, 1)
            
            text_features.extend([avg_word_length, unique_word_ratio])
            
            # Vocabulary diversity
            words = text.lower().split()
            word_freq = Counter(words)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
            vocabulary_diversity = len(word_freq) / max(word_count, 1)
            
            text_features.extend([most_common_freq, vocabulary_diversity])
            
            # Readability approximation (Flesch-like)
            sentences = len(re.split(r'[.!?]+', text))
            avg_sentence_length = word_count / max(sentences, 1)
            
            # Count syllables (rough approximation)
            syllable_count = sum(max(1, len(re.findall(r'[aeiouy]+', word.lower()))) for word in words)
            avg_syllables_per_word = syllable_count / max(word_count, 1)
            
            readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            text_features.extend([avg_sentence_length, avg_syllables_per_word, readability_score])
            
            # Legal complexity indicators
            complex_legal_terms = [
                'notwithstanding', 'whereas', 'heretofore', 'hereinafter', 'pursuant',
                'therefor', 'thereof', 'therein', 'aforesaid', 'aforementioned'
            ]
            
            complex_term_count = sum(1 for term in complex_legal_terms if term in text.lower())
            text_features.append(complex_term_count)
            
            features.append(text_features)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        feature_names = []
        
        # TF-IDF feature names
        if self.tfidf_vectorizer:
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            feature_names.extend(tfidf_names)
        
        # N-gram feature names
        if self.ngram_vectorizer:
            ngram_names = [f"ngram_{name}" for name in self.ngram_vectorizer.get_feature_names_out()]
            feature_names.extend(ngram_names)
        
        # Keyword feature names
        if self.use_legal_keywords:
            keyword_names = [
                'arb_strong_count', 'arb_medium_count', 'arb_weak_count',
                'jury_waiver_count', 'jury_waiver_present',
                'class_waiver_count', 'class_waiver_present',
                'court_waiver_count', 'court_waiver_present',
                'legal_entity_count', 'enforcement_count'
            ]
            
            # Arbitration pattern names
            pattern_names = [
                'pattern_arb_shall_be', 'pattern_binding_arb', 'pattern_mandatory_arb',
                'pattern_waive_jury', 'pattern_class_waiver', 'pattern_dispute_arb',
                'pattern_arb_dispute', 'negation_score'
            ]
            keyword_names.extend(pattern_names)
            feature_names.extend(keyword_names)
        
        # Structure feature names
        if self.use_structure_features:
            structure_names = [
                'sentence_count', 'caps_ratio', 'all_caps_words',
                'exclamation_count', 'question_count', 'semicolon_count', 'colon_count',
                'paren_count', 'has_section_numbering', 'has_legal_citations',
                'avg_sent_length_or_word_length'
            ]
            feature_names.extend(structure_names)
        
        # Statistical feature names
        if self.use_statistical_features:
            statistical_names = [
                'char_count', 'word_count', 'unique_words',
                'avg_word_length', 'unique_word_ratio',
                'most_common_freq', 'vocabulary_diversity',
                'avg_sentence_length', 'avg_syllables_per_word', 'readability_score',
                'complex_legal_terms'
            ]
            feature_names.extend(statistical_names)
        
        return feature_names
    
    def save(self, filepath: str):
        """Save the fitted feature extractor"""
        joblib.dump(self, filepath)
        print(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted feature extractor"""
        return joblib.load(filepath)


class ArbitrationFeatureAnalyzer:
    """
    Analyzer for understanding which features are most important for arbitration detection
    """
    
    def __init__(self, feature_extractor: LegalFeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def analyze_feature_importance(self, texts: List[str], labels: List[int], 
                                 model=None) -> Dict[str, Any]:
        """Analyze which features are most important"""
        
        # Extract features
        X = self.feature_extractor.transform(texts)
        feature_names = self.feature_extractor.get_feature_names()
        
        analysis = {
            'feature_statistics': {},
            'feature_importance': {},
            'feature_correlations': {}
        }
        
        # Basic feature statistics
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        for i, name in enumerate(feature_names):
            if i < X_dense.shape[1]:
                feature_values = X_dense[:, i]
                analysis['feature_statistics'][name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'non_zero_ratio': float(np.sum(feature_values > 0) / len(feature_values))
                }
        
        # Feature importance from model if provided
        if model and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    analysis['feature_importance'][name] = float(importances[i])
        
        # Correlation with labels
        labels_array = np.array(labels)
        for i, name in enumerate(feature_names):
            if i < X_dense.shape[1]:
                feature_values = X_dense[:, i]
                if np.std(feature_values) > 0:  # Avoid division by zero
                    correlation = np.corrcoef(feature_values, labels_array)[0, 1]
                    analysis['feature_correlations'][name] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return analysis
    
    def get_top_features(self, analysis: Dict[str, Any], 
                        criterion: str = 'correlation', 
                        top_k: int = 20) -> List[Tuple[str, float]]:
        """Get top features by specified criterion"""
        
        if criterion not in analysis:
            raise ValueError(f"Criterion {criterion} not found in analysis")
        
        feature_scores = analysis[criterion]
        
        if criterion == 'feature_correlations':
            # Sort by absolute correlation
            sorted_features = sorted(feature_scores.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        else:
            sorted_features = sorted(feature_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_k]


def main():
    """Test the feature extractor"""
    
    # Sample texts for testing
    sample_texts = [
        "Any dispute arising out of these terms shall be resolved through binding arbitration in Delaware.",
        "You waive your right to jury trial and class action participation for all disputes.",
        "Legal action may be brought in federal or state court in California.",
        "This privacy policy explains how we collect and use your personal information.",
        "BINDING ARBITRATION: All disputes MUST be resolved through individual arbitration with AAA."
    ]
    
    sample_labels = [1, 1, 0, 0, 1]
    
    # Create and fit feature extractor
    extractor = LegalFeatureExtractor()
    extractor.fit(sample_texts)
    
    # Transform texts
    features = extractor.transform(sample_texts)
    print(f"Feature matrix shape: {features.shape}")
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"Number of features: {len(feature_names)}")
    
    # Analyze features
    analyzer = ArbitrationFeatureAnalyzer(extractor)
    analysis = analyzer.analyze_feature_importance(sample_texts, sample_labels)
    
    # Get top correlated features
    top_features = analyzer.get_top_features(analysis, 'feature_correlations', 10)
    print("\nTop features by correlation:")
    for name, score in top_features:
        print(f"{name}: {score:.4f}")
    
    return extractor, features, analysis


if __name__ == "__main__":
    extractor, features, analysis = main()