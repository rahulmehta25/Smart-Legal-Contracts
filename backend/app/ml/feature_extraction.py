"""
Feature extraction for legal text analysis and arbitration clause detection
Combines linguistic, semantic, and domain-specific features
"""
import logging
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter, defaultdict
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import textstat
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class LegalFeatureExtractor:
    """
    Comprehensive feature extraction for legal text analysis
    Specializes in arbitration clause detection
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm",
                 cache_dir: str = "backend/models/feature_cache"):
        self.embedding_model_name = embedding_model
        self.spacy_model_name = spacy_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.nlp = self._load_spacy_model()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.scaler = StandardScaler()
        
        # Legal-specific patterns and vocabularies
        self.arbitration_keywords = self._create_arbitration_keywords()
        self.legal_patterns = self._create_legal_patterns()
        self.negation_patterns = self._create_negation_patterns()
        
        logger.info("Legal feature extractor initialized")
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            return spacy.load(self.spacy_model_name)
        except OSError:
            logger.warning(f"spaCy model {self.spacy_model_name} not found. Using en_core_web_sm")
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model found. Please install: python -m spacy download en_core_web_sm")
                return None
    
    def _create_arbitration_keywords(self) -> Dict[str, List[str]]:
        """Create categorized arbitration-related keywords"""
        return {
            "arbitration_core": [
                "arbitration", "arbitrate", "arbitrated", "arbitrating", "arbitrator",
                "arbitral", "arbitrability", "arbitrable"
            ],
            "binding_terms": [
                "binding", "final", "conclusive", "definitive", "irrevocable",
                "non-appealable", "enforceable", "mandatory", "compulsory"
            ],
            "dispute_resolution": [
                "dispute", "controversy", "claim", "disagreement", "conflict",
                "resolution", "settlement", "mediation", "adr", "alternative dispute resolution"
            ],
            "tribunal_terms": [
                "tribunal", "panel", "arbitrator", "neutral", "umpire", "adjudicator"
            ],
            "procedure_terms": [
                "proceedings", "hearing", "award", "decision", "ruling", "determination"
            ],
            "waiver_terms": [
                "waive", "waiver", "waiving", "relinquish", "forfeit", "abandon",
                "jury trial", "class action", "collective action"
            ],
            "institutional": [
                "aaa", "american arbitration association", "jams", "icc", 
                "international chamber of commerce", "lcia", "siac", "uncitral"
            ]
        }
    
    def _create_legal_patterns(self) -> List[Dict]:
        """Create regex patterns for legal concepts"""
        return [
            {
                "name": "arbitration_clause",
                "pattern": r"\b(?:any|all)?\s*(?:dispute|controversy|claim|disagreement)s?\s+(?:arising|relating|pertaining)\s+(?:out\s+of|under|from|to)\s+(?:this|the)\s+(?:agreement|contract)",
                "weight": 3.0
            },
            {
                "name": "binding_arbitration",
                "pattern": r"\bbinding\s+arbitration\b",
                "weight": 5.0
            },
            {
                "name": "final_arbitration",
                "pattern": r"\bfinal\s+(?:and\s+)?(?:binding\s+)?arbitration\b",
                "weight": 4.0
            },
            {
                "name": "arbitration_rules",
                "pattern": r"\b(?:under|pursuant\s+to|in\s+accordance\s+with|governed\s+by)\s+(?:the\s+)?(?:rules|procedures)\s+of\s+(?:the\s+)?(?:aaa|jams|icc|lcia|siac|uncitral)",
                "weight": 4.0
            },
            {
                "name": "tribunal_reference",
                "pattern": r"\b(?:arbitral\s+)?tribunal\b",
                "weight": 3.0
            },
            {
                "name": "jury_waiver",
                "pattern": r"\bwaiv(?:e|er|ing)\s+(?:their|the)\s+right\s+to\s+(?:a\s+)?jury\s+trial\b",
                "weight": 3.0
            },
            {
                "name": "class_action_waiver",
                "pattern": r"\bclass\s+action\s+waiver\b",
                "weight": 3.0
            },
            {
                "name": "exclusive_remedy",
                "pattern": r"\b(?:sole|exclusive|only)\s+(?:remedy|means|method)\b",
                "weight": 2.0
            }
        ]
    
    def _create_negation_patterns(self) -> List[str]:
        """Create patterns that negate arbitration clauses"""
        return [
            r"\bno\s+arbitration\b",
            r"\bnot\s+subject\s+to\s+arbitration\b",
            r"\bexcept\s+for\s+arbitration\b",
            r"\bother\s+than\s+arbitration\b",
            r"\bresolved\s+in\s+court\b",
            r"\bjudicial\s+proceedings\b",
            r"\bcourt\s+of\s+competent\s+jurisdiction\b"
        ]
    
    def extract_keyword_features(self, text: str) -> Dict[str, float]:
        """Extract keyword-based features"""
        text_lower = text.lower()
        features = {}
        
        # Count keywords by category
        for category, keywords in self.arbitration_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f"keyword_{category}_count"] = count
            features[f"keyword_{category}_density"] = count / len(text.split()) if text.split() else 0
        
        # Total keyword features
        total_arbitration_keywords = sum(features[k] for k in features if k.endswith("_count"))
        features["total_arbitration_keywords"] = total_arbitration_keywords
        features["arbitration_keyword_density"] = total_arbitration_keywords / len(text.split()) if text.split() else 0
        
        return features
    
    def extract_pattern_features(self, text: str) -> Dict[str, float]:
        """Extract regex pattern-based features"""
        features = {}
        
        # Arbitration patterns
        for pattern_info in self.legal_patterns:
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]
            name = pattern_info["name"]
            
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features[f"pattern_{name}_count"] = matches
            features[f"pattern_{name}_weighted"] = matches * weight
        
        # Negation patterns
        negation_count = 0
        for pattern in self.negation_patterns:
            negation_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        features["negation_patterns"] = negation_count
        features["negation_ratio"] = negation_count / (features["total_arbitration_keywords"] + 1)
        
        # Overall pattern score
        pattern_score = sum(v for k, v in features.items() if k.endswith("_weighted"))
        features["total_pattern_score"] = pattern_score
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and readability features"""
        features = {}
        
        if not text.strip():
            return {f"linguistic_{k}": 0.0 for k in [
                "word_count", "sentence_count", "avg_word_length", "flesch_reading_ease",
                "flesch_kincaid_grade", "gunning_fog", "automated_readability_index"
            ]}
        
        # Basic statistics
        words = text.split()
        sentences = text.split('.')
        
        features["linguistic_word_count"] = len(words)
        features["linguistic_sentence_count"] = len(sentences)
        features["linguistic_avg_word_length"] = np.mean([len(word) for word in words]) if words else 0
        features["linguistic_avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
        
        # Readability scores
        try:
            features["linguistic_flesch_reading_ease"] = textstat.flesch_reading_ease(text)
            features["linguistic_flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
            features["linguistic_gunning_fog"] = textstat.gunning_fog(text)
            features["linguistic_automated_readability_index"] = textstat.automated_readability_index(text)
        except:
            features.update({
                "linguistic_flesch_reading_ease": 0,
                "linguistic_flesch_kincaid_grade": 0,
                "linguistic_gunning_fog": 0,
                "linguistic_automated_readability_index": 0
            })
        
        return features
    
    def extract_spacy_features(self, text: str) -> Dict[str, float]:
        """Extract spaCy-based linguistic features"""
        if self.nlp is None:
            return {}
        
        doc = self.nlp(text)
        features = {}
        
        # POS tag distribution
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        for pos_tag in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
            features[f"spacy_pos_{pos_tag.lower()}_ratio"] = pos_counts.get(pos_tag, 0) / total_tokens if total_tokens > 0 else 0
        
        # Named entities
        entity_counts = Counter([ent.label_ for ent in doc.ents])
        for entity_type in ["ORG", "PERSON", "GPE", "LAW"]:
            features[f"spacy_entity_{entity_type.lower()}_count"] = entity_counts.get(entity_type, 0)
        
        # Dependency features
        dep_counts = Counter([token.dep_ for token in doc])
        features["spacy_complex_dependencies"] = sum(dep_counts.get(dep, 0) for dep in ["ccomp", "xcomp", "csubj"])
        
        # Legal-specific entity patterns
        legal_entities = 0
        for ent in doc.ents:
            if any(keyword in ent.text.lower() for keyword in ["arbitration", "court", "tribunal", "association"]):
                legal_entities += 1
        
        features["spacy_legal_entities"] = legal_entities
        
        return features
    
    def extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic embeddings and similarity features"""
        features = {}
        
        # Generate embeddings
        embedding = self.embedding_model.encode([text])[0]
        
        # Basic embedding statistics
        features["semantic_embedding_mean"] = np.mean(embedding)
        features["semantic_embedding_std"] = np.std(embedding)
        features["semantic_embedding_min"] = np.min(embedding)
        features["semantic_embedding_max"] = np.max(embedding)
        
        # Similarity to known arbitration clauses
        arbitration_examples = [
            "Any dispute arising under this agreement shall be resolved through binding arbitration.",
            "All controversies shall be settled by final and binding arbitration.",
            "The parties agree to submit disputes to arbitration under AAA rules."
        ]
        
        arbitration_embeddings = self.embedding_model.encode(arbitration_examples)
        similarities = [np.dot(embedding, arb_emb) / (np.linalg.norm(embedding) * np.linalg.norm(arb_emb)) 
                       for arb_emb in arbitration_embeddings]
        
        features["semantic_max_arbitration_similarity"] = max(similarities)
        features["semantic_avg_arbitration_similarity"] = np.mean(similarities)
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        for key, value in sentiment_scores.items():
            features[f"sentiment_{key}"] = value
        
        return features, embedding
    
    def extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract document structure features"""
        features = {}
        
        # Punctuation analysis
        features["structural_period_count"] = text.count('.')
        features["structural_comma_count"] = text.count(',')
        features["structural_semicolon_count"] = text.count(';')
        features["structural_parentheses_count"] = text.count('(') + text.count(')')
        
        # Capitalization patterns
        words = text.split()
        if words:
            features["structural_capitalized_ratio"] = sum(1 for word in words if word[0].isupper()) / len(words)
            features["structural_all_caps_count"] = sum(1 for word in words if word.isupper() and len(word) > 1)
        else:
            features["structural_capitalized_ratio"] = 0
            features["structural_all_caps_count"] = 0
        
        # Quoted text
        features["structural_quoted_segments"] = text.count('"') // 2
        
        # Legal formatting patterns
        features["structural_section_references"] = len(re.findall(r'\bsection\s+\d+', text, re.IGNORECASE))
        features["structural_subsection_references"] = len(re.findall(r'\b\d+\.\d+', text))
        
        return features
    
    def extract_all_features(self, text: str) -> Tuple[Dict[str, float], np.ndarray]:
        """Extract all feature types for a given text"""
        all_features = {}
        
        # Extract each feature type
        keyword_features = self.extract_keyword_features(text)
        pattern_features = self.extract_pattern_features(text)
        linguistic_features = self.extract_linguistic_features(text)
        spacy_features = self.extract_spacy_features(text)
        semantic_features, embedding = self.extract_semantic_features(text)
        structural_features = self.extract_structural_features(text)
        
        # Combine all features
        all_features.update(keyword_features)
        all_features.update(pattern_features)
        all_features.update(linguistic_features)
        all_features.update(spacy_features)
        all_features.update(semantic_features)
        all_features.update(structural_features)
        
        return all_features, embedding
    
    def extract_features_batch(self, texts: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract features for a batch of texts efficiently"""
        all_features = []
        all_embeddings = []
        
        logger.info(f"Extracting features for {len(texts)} texts...")
        
        # Batch processing for embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processing text {i+1}/{len(texts)}")
            
            features, _ = self.extract_all_features(text)
            all_features.append(features)
            all_embeddings.append(embeddings[i])
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        embeddings_array = np.array(all_embeddings)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        logger.info(f"Extracted {len(features_df.columns)} features for {len(texts)} texts")
        return features_df, embeddings_array
    
    def fit_vectorizers(self, texts: List[str], max_features: int = 1000):
        """Fit TF-IDF and count vectorizers on training data"""
        logger.info("Fitting vectorizers...")
        
        # TF-IDF vectorizer for general terms
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Count vectorizer for legal-specific terms
        legal_vocabulary = []
        for category, keywords in self.arbitration_keywords.items():
            legal_vocabulary.extend(keywords)
        
        self.count_vectorizer = CountVectorizer(
            vocabulary=legal_vocabulary,
            ngram_range=(1, 3)
        )
        
        # Fit vectorizers
        self.tfidf_vectorizer.fit(texts)
        self.count_vectorizer.fit(texts)
        
        logger.info("Vectorizers fitted successfully")
    
    def get_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Get TF-IDF features for texts"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_vectorizers first.")
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def get_count_features(self, texts: List[str]) -> np.ndarray:
        """Get count-based features for legal terms"""
        if self.count_vectorizer is None:
            raise ValueError("Count vectorizer not fitted. Call fit_vectorizers first.")
        
        return self.count_vectorizer.transform(texts).toarray()
    
    def scale_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale numerical features"""
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)
    
    def save_extractors(self, filepath: str):
        """Save fitted extractors to disk"""
        extractors = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'scaler': self.scaler,
            'embedding_model_name': self.embedding_model_name,
            'spacy_model_name': self.spacy_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(extractors, f)
        
        logger.info(f"Feature extractors saved to {filepath}")
    
    def load_extractors(self, filepath: str):
        """Load fitted extractors from disk"""
        with open(filepath, 'rb') as f:
            extractors = pickle.load(f)
        
        self.tfidf_vectorizer = extractors['tfidf_vectorizer']
        self.count_vectorizer = extractors['count_vectorizer']
        self.scaler = extractors['scaler']
        
        logger.info(f"Feature extractors loaded from {filepath}")
    
    def create_feature_importance_report(self, 
                                       features_df: pd.DataFrame,
                                       target: np.ndarray,
                                       method: str = 'correlation') -> pd.DataFrame:
        """Create feature importance analysis"""
        if method == 'correlation':
            # Calculate correlation with target
            correlations = []
            for column in features_df.columns:
                corr = np.corrcoef(features_df[column], target)[0, 1]
                correlations.append({
                    'feature': column,
                    'correlation': abs(corr) if not np.isnan(corr) else 0,
                    'direction': 'positive' if corr > 0 else 'negative'
                })
            
            importance_df = pd.DataFrame(correlations)
            importance_df = importance_df.sort_values('correlation', ascending=False)
            
        return importance_df


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities"""
    extractor = LegalFeatureExtractor()
    
    # Example arbitration clause
    arbitration_text = """
    Any dispute, controversy or claim arising out of or relating to this Agreement, 
    or the breach, termination or validity thereof, shall be settled by final and 
    binding arbitration administered by the American Arbitration Association under 
    its Commercial Arbitration Rules, and judgment on the award rendered by the 
    arbitrator(s) may be entered in any court having jurisdiction thereof.
    """
    
    # Example non-arbitration clause
    non_arbitration_text = """
    Any disputes arising under this agreement shall be resolved in the federal courts 
    of the State of New York. The parties hereby consent to the jurisdiction of such 
    courts and waive any objection to venue therein.
    """
    
    # Extract features
    print("Extracting features for arbitration clause...")
    arb_features, arb_embedding = extractor.extract_all_features(arbitration_text)
    
    print("Extracting features for non-arbitration clause...")
    non_arb_features, non_arb_embedding = extractor.extract_all_features(non_arbitration_text)
    
    # Compare key features
    key_features = [
        "total_arbitration_keywords", "total_pattern_score", 
        "semantic_max_arbitration_similarity", "pattern_binding_arbitration_count"
    ]
    
    print("\nFeature Comparison:")
    print("-" * 50)
    for feature in key_features:
        arb_val = arb_features.get(feature, 0)
        non_arb_val = non_arb_features.get(feature, 0)
        print(f"{feature:35}: {arb_val:8.3f} | {non_arb_val:8.3f}")


if __name__ == "__main__":
    demo_feature_extraction()