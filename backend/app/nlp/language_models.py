"""
Language models for multilingual arbitration clause detection.

This module provides multilingual embeddings, cross-lingual similarity scoring,
and language-agnostic pattern detection for arbitration clauses across
different languages and legal systems.
"""

import logging
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import pickle
import json
from pathlib import Path
import hashlib
from functools import lru_cache
import time

# Transformers and embeddings
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    XLMRobertaModel, XLMRobertaTokenizer,
    BertModel, BertTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss

# Scientific computing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.spatial.distance as distance

# Configuration
import yaml


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of language models available."""
    MULTILINGUAL_BERT = "multilingual_bert"
    XLM_ROBERTA = "xlm_roberta"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    LEGAL_BERT = "legal_bert"
    CUSTOM_LEGAL = "custom_legal"


@dataclass
class EmbeddingResult:
    """Result of text embedding operation."""
    text: str
    embeddings: np.ndarray
    model_used: str
    language: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    text1: str
    text2: str
    similarity_score: float
    cross_lingual: bool
    language1: str
    language2: str
    method_used: str
    confidence: float


@dataclass
class PatternDetectionResult:
    """Result of arbitration pattern detection."""
    text: str
    patterns_detected: List[str]
    confidence_scores: Dict[str, float]
    language: str
    legal_domain: Optional[str]
    arbitration_probability: float
    key_phrases: List[str]


class MultilingualEmbeddingEngine:
    """Engine for generating multilingual embeddings with multiple models."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.models = {}
        self.tokenizers = {}
        self.embedding_cache = {}
        self.cache_max_size = 5000
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._initialize_models()
        
        # FAISS index for fast similarity search
        self.faiss_indices = {}
        self.index_metadata = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        return {
            'models': {
                'multilingual_bert': 'bert-base-multilingual-cased',
                'xlm_roberta': 'xlm-roberta-base',
                'sentence_transformer': 'paraphrase-multilingual-MiniLM-L12-v2',
                'legal_sentence_transformer': 'nlpaueb/legal-bert-base-uncased'
            },
            'embedding_dim': 768,
            'max_sequence_length': 512,
            'batch_size': 16,
            'enable_caching': True,
            'faiss_index_type': 'flat'
        }
    
    def _initialize_models(self):
        """Initialize multilingual models."""
        model_configs = self.config['models']
        
        # Multilingual BERT
        try:
            model_name = model_configs['multilingual_bert']
            self.models[ModelType.MULTILINGUAL_BERT] = AutoModel.from_pretrained(model_name)
            self.tokenizers[ModelType.MULTILINGUAL_BERT] = AutoTokenizer.from_pretrained(model_name)
            self.models[ModelType.MULTILINGUAL_BERT].to(self.device)
            logger.info(f"Loaded Multilingual BERT: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Multilingual BERT: {e}")
        
        # XLM-RoBERTa
        try:
            model_name = model_configs['xlm_roberta']
            self.models[ModelType.XLM_ROBERTA] = XLMRobertaModel.from_pretrained(model_name)
            self.tokenizers[ModelType.XLM_ROBERTA] = XLMRobertaTokenizer.from_pretrained(model_name)
            self.models[ModelType.XLM_ROBERTA].to(self.device)
            logger.info(f"Loaded XLM-RoBERTa: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load XLM-RoBERTa: {e}")
        
        # Sentence Transformer
        try:
            model_name = model_configs['sentence_transformer']
            self.models[ModelType.SENTENCE_TRANSFORMER] = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                self.models[ModelType.SENTENCE_TRANSFORMER] = self.models[ModelType.SENTENCE_TRANSFORMER].cuda()
            logger.info(f"Loaded Sentence Transformer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
        
        # Legal domain specific model (if available)
        try:
            if 'legal_sentence_transformer' in model_configs:
                model_name = model_configs['legal_sentence_transformer']
                self.models[ModelType.LEGAL_BERT] = SentenceTransformer(model_name)
                logger.info(f"Loaded Legal Sentence Transformer: {model_name}")
        except Exception as e:
            logger.warning(f"Legal model not available: {e}")
    
    def _cache_key(self, text: str, model_type: ModelType, language: str) -> str:
        """Generate cache key for embeddings."""
        content = f"{text}_{model_type.value}_{language}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def get_embeddings(
        self,
        text: str,
        model_type: ModelType = ModelType.SENTENCE_TRANSFORMER,
        language: Optional[str] = None,
        normalize: bool = True
    ) -> EmbeddingResult:
        """
        Generate embeddings for text using specified model.
        
        Args:
            text: Input text
            model_type: Type of model to use
            language: Language of the text (for optimization)
            normalize: Whether to normalize embeddings
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        
        # Check cache first
        if self.config.get('enable_caching', True):
            cache_key = self._cache_key(text, model_type, language or 'unknown')
            if cache_key in self.embedding_cache:
                cached_result = self.embedding_cache[cache_key]
                cached_result.processing_time = time.time() - start_time
                return cached_result
        
        # Clean up cache if too large
        if len(self.embedding_cache) > self.cache_max_size:
            keys_to_remove = list(self.embedding_cache.keys())[:500]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        # Generate embeddings based on model type
        try:
            if model_type == ModelType.SENTENCE_TRANSFORMER:
                embeddings = await self._get_sentence_transformer_embeddings(text)
            elif model_type == ModelType.MULTILINGUAL_BERT:
                embeddings = await self._get_bert_embeddings(text, ModelType.MULTILINGUAL_BERT)
            elif model_type == ModelType.XLM_ROBERTA:
                embeddings = await self._get_bert_embeddings(text, ModelType.XLM_ROBERTA)
            elif model_type == ModelType.LEGAL_BERT:
                embeddings = await self._get_legal_embeddings(text)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings)
            
            result = EmbeddingResult(
                text=text,
                embeddings=embeddings,
                model_used=model_type.value,
                language=language or 'unknown',
                confidence=0.8,  # Default confidence
                processing_time=time.time() - start_time,
                metadata={
                    'embedding_dim': len(embeddings),
                    'normalized': normalize,
                    'device': str(self.device)
                }
            )
            
            # Cache result
            if self.config.get('enable_caching', True):
                self.embedding_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero embeddings as fallback
            dim = self.config.get('embedding_dim', 768)
            return EmbeddingResult(
                text=text,
                embeddings=np.zeros(dim),
                model_used=f"{model_type.value}_fallback",
                language=language or 'unknown',
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _get_sentence_transformer_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using Sentence Transformer."""
        model = self.models.get(ModelType.SENTENCE_TRANSFORMER)
        if not model:
            raise ValueError("Sentence Transformer model not available")
        
        embeddings = model.encode([text])
        return embeddings[0]
    
    async def _get_bert_embeddings(self, text: str, model_type: ModelType) -> np.ndarray:
        """Get embeddings using BERT-based models."""
        model = self.models.get(model_type)
        tokenizer = self.tokenizers.get(model_type)
        
        if not model or not tokenizer:
            raise ValueError(f"Model {model_type} not available")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_sequence_length', 512)
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token or mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs.pooler_output
            
        return embeddings.cpu().numpy().flatten()
    
    async def _get_legal_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using legal domain-specific model."""
        model = self.models.get(ModelType.LEGAL_BERT)
        if not model:
            # Fallback to sentence transformer
            return await self._get_sentence_transformer_embeddings(text)
        
        embeddings = model.encode([text])
        return embeddings[0]
    
    async def get_batch_embeddings(
        self,
        texts: List[str],
        model_type: ModelType = ModelType.SENTENCE_TRANSFORMER,
        language: Optional[str] = None
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to process
            model_type: Type of model to use
            language: Language of the texts
            
        Returns:
            List of EmbeddingResult objects
        """
        batch_size = self.config.get('batch_size', 16)
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tasks = [
                self.get_embeddings(text, model_type, language)
                for text in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def build_faiss_index(
        self,
        embeddings: List[np.ndarray],
        index_name: str,
        metadata: List[Dict[str, Any]]
    ):
        """
        Build FAISS index for fast similarity search.
        
        Args:
            embeddings: List of embedding vectors
            index_name: Name for the index
            metadata: Metadata for each embedding
        """
        if not embeddings:
            return
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embedding_matrix.shape[1]
        
        if self.config.get('faiss_index_type') == 'ivf':
            # IVF index for larger datasets
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embedding_matrix)
        else:
            # Flat index for smaller datasets
            index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embedding_matrix)
        
        # Store index and metadata
        self.faiss_indices[index_name] = index
        self.index_metadata[index_name] = metadata
        
        logger.info(f"Built FAISS index '{index_name}' with {len(embeddings)} embeddings")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        index_name: str,
        k: int = 5
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for similar embeddings using FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            index_name: Name of the index to search
            k: Number of similar items to return
            
        Returns:
            List of (distance, metadata) tuples
        """
        if index_name not in self.faiss_indices:
            return []
        
        index = self.faiss_indices[index_name]
        metadata = self.index_metadata[index_name]
        
        # Search
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vector, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):
                results.append((float(distance), metadata[idx]))
        
        return results


class CrossLingualSimilarityEngine:
    """Engine for computing cross-lingual similarity scores."""
    
    def __init__(self, embedding_engine: MultilingualEmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.similarity_cache = {}
        self.cache_max_size = 2000
    
    def _cache_key(self, text1: str, text2: str, method: str) -> str:
        """Generate cache key for similarity computation."""
        # Sort texts to ensure consistency
        sorted_texts = sorted([text1, text2])
        content = f"{sorted_texts[0]}_{sorted_texts[1]}_{method}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        language1: Optional[str] = None,
        language2: Optional[str] = None,
        method: str = "cosine",
        model_type: ModelType = ModelType.SENTENCE_TRANSFORMER
    ) -> SimilarityResult:
        """
        Compute cross-lingual similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            language1: Language of first text
            language2: Language of second text
            method: Similarity computation method
            model_type: Model to use for embeddings
            
        Returns:
            SimilarityResult with similarity score and metadata
        """
        # Check cache
        cache_key = self._cache_key(text1, text2, f"{method}_{model_type.value}")
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Clean up cache if too large
        if len(self.similarity_cache) > self.cache_max_size:
            keys_to_remove = list(self.similarity_cache.keys())[:200]
            for key in keys_to_remove:
                del self.similarity_cache[key]
        
        # Get embeddings for both texts
        embedding1_result = await self.embedding_engine.get_embeddings(
            text1, model_type, language1
        )
        embedding2_result = await self.embedding_engine.get_embeddings(
            text2, model_type, language2
        )
        
        # Compute similarity
        if method == "cosine":
            similarity_score = self._cosine_similarity(
                embedding1_result.embeddings,
                embedding2_result.embeddings
            )
        elif method == "euclidean":
            similarity_score = self._euclidean_similarity(
                embedding1_result.embeddings,
                embedding2_result.embeddings
            )
        elif method == "dot_product":
            similarity_score = self._dot_product_similarity(
                embedding1_result.embeddings,
                embedding2_result.embeddings
            )
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        # Determine if cross-lingual
        cross_lingual = (
            language1 and language2 and 
            language1 != language2 and
            language1 != 'unknown' and language2 != 'unknown'
        )
        
        # Calculate confidence based on embedding confidence
        confidence = min(
            embedding1_result.confidence,
            embedding2_result.confidence
        )
        
        result = SimilarityResult(
            text1=text1,
            text2=text2,
            similarity_score=similarity_score,
            cross_lingual=cross_lingual,
            language1=language1 or 'unknown',
            language2=language2 or 'unknown',
            method_used=method,
            confidence=confidence
        )
        
        # Cache result
        self.similarity_cache[cache_key] = result
        
        return result
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(cosine_similarity([vec1], [vec2])[0][0])
    
    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute euclidean similarity (converted to [0,1] range)."""
        euclidean_dist = distance.euclidean(vec1, vec2)
        # Convert to similarity score (closer to 1 means more similar)
        return 1.0 / (1.0 + euclidean_dist)
    
    def _dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute normalized dot product similarity."""
        # Normalize vectors first
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1_norm, vec2_norm))
    
    async def compute_batch_similarity(
        self,
        text_pairs: List[Tuple[str, str]],
        language_pairs: Optional[List[Tuple[str, str]]] = None,
        method: str = "cosine",
        model_type: ModelType = ModelType.SENTENCE_TRANSFORMER
    ) -> List[SimilarityResult]:
        """
        Compute similarity for multiple text pairs in batch.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            language_pairs: List of (lang1, lang2) tuples
            method: Similarity computation method
            model_type: Model to use for embeddings
            
        Returns:
            List of SimilarityResult objects
        """
        tasks = []
        for i, (text1, text2) in enumerate(text_pairs):
            lang1, lang2 = None, None
            if language_pairs and i < len(language_pairs):
                lang1, lang2 = language_pairs[i]
            
            tasks.append(
                self.compute_similarity(text1, text2, lang1, lang2, method, model_type)
            )
        
        return await asyncio.gather(*tasks)


class ArbitrationPatternDetector:
    """Language-agnostic arbitration pattern detector."""
    
    def __init__(
        self,
        embedding_engine: MultilingualEmbeddingEngine,
        similarity_engine: CrossLingualSimilarityEngine
    ):
        self.embedding_engine = embedding_engine
        self.similarity_engine = similarity_engine
        self.pattern_templates = {}
        self.classification_threshold = 0.7
        self._initialize_pattern_templates()
    
    def _initialize_pattern_templates(self):
        """Initialize arbitration pattern templates in multiple languages."""
        self.pattern_templates = {
            'binding_arbitration': {
                'en': [
                    "any dispute shall be settled by binding arbitration",
                    "disputes arising under this agreement shall be resolved through arbitration",
                    "all claims must be submitted to binding arbitration"
                ],
                'es': [
                    "cualquier disputa será resuelta por arbitraje vinculante",
                    "las disputas que surjan bajo este acuerdo serán resueltas a través de arbitraje",
                    "todas las reclamaciones deben ser sometidas a arbitraje vinculante"
                ],
                'fr': [
                    "tout différend sera réglé par arbitrage contraignant",
                    "les différends découlant de cet accord seront résolus par arbitrage",
                    "toutes les réclamations doivent être soumises à un arbitrage contraignant"
                ],
                'de': [
                    "jede Streitigkeit wird durch bindendes Schiedsverfahren beigelegt",
                    "Streitigkeiten aus diesem Vertrag werden durch Schiedsverfahren gelöst",
                    "alle Ansprüche müssen einem bindenden Schiedsverfahren unterworfen werden"
                ]
            },
            'arbitration_clause': {
                'en': [
                    "arbitration clause",
                    "dispute resolution through arbitration",
                    "agreement to arbitrate"
                ],
                'es': [
                    "cláusula de arbitraje",
                    "resolución de disputas a través de arbitraje",
                    "acuerdo de arbitraje"
                ],
                'fr': [
                    "clause d'arbitrage",
                    "résolution des différends par arbitrage",
                    "accord d'arbitrage"
                ],
                'de': [
                    "Schiedsklausel",
                    "Streitbeilegung durch Schiedsverfahren",
                    "Schiedsvereinbarung"
                ]
            },
            'waiver_clauses': {
                'en': [
                    "waive the right to jury trial",
                    "waiver of class action rights",
                    "no right to participate in class action"
                ],
                'es': [
                    "renuncia al derecho de juicio por jurado",
                    "renuncia a los derechos de acción de clase",
                    "sin derecho a participar en acción de clase"
                ],
                'fr': [
                    "renonciation au droit au procès devant jury",
                    "renonciation aux droits d'action collective",
                    "aucun droit de participer à une action collective"
                ],
                'de': [
                    "Verzicht auf das Recht auf Geschworenenprozess",
                    "Verzicht auf Sammelklagerechte",
                    "kein Recht zur Teilnahme an Sammelklagen"
                ]
            }
        }
    
    async def detect_patterns(
        self,
        text: str,
        language: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> PatternDetectionResult:
        """
        Detect arbitration patterns in text using cross-lingual similarity.
        
        Args:
            text: Text to analyze
            language: Language of the text
            threshold: Similarity threshold for pattern detection
            
        Returns:
            PatternDetectionResult with detected patterns and scores
        """
        threshold = threshold or self.classification_threshold
        
        # Get text embedding
        text_embedding = await self.embedding_engine.get_embeddings(
            text, ModelType.SENTENCE_TRANSFORMER, language
        )
        
        patterns_detected = []
        confidence_scores = {}
        key_phrases = []
        
        # Check against all pattern templates
        for pattern_type, lang_templates in self.pattern_templates.items():
            max_similarity = 0.0
            best_match = None
            
            # Check against all languages (cross-lingual detection)
            for template_lang, templates in lang_templates.items():
                for template in templates:
                    similarity_result = await self.similarity_engine.compute_similarity(
                        text, template, language, template_lang
                    )
                    
                    if similarity_result.similarity_score > max_similarity:
                        max_similarity = similarity_result.similarity_score
                        best_match = template
            
            # Record if above threshold
            if max_similarity >= threshold:
                patterns_detected.append(pattern_type)
                confidence_scores[pattern_type] = max_similarity
                if best_match:
                    key_phrases.append(best_match)
        
        # Calculate overall arbitration probability
        arbitration_probability = self._calculate_arbitration_probability(
            confidence_scores, patterns_detected
        )
        
        # Detect legal domain
        legal_domain = self._detect_legal_domain(patterns_detected)
        
        return PatternDetectionResult(
            text=text,
            patterns_detected=patterns_detected,
            confidence_scores=confidence_scores,
            language=language or 'unknown',
            legal_domain=legal_domain,
            arbitration_probability=arbitration_probability,
            key_phrases=key_phrases
        )
    
    def _calculate_arbitration_probability(
        self,
        confidence_scores: Dict[str, float],
        patterns_detected: List[str]
    ) -> float:
        """Calculate overall probability of arbitration clause presence."""
        if not confidence_scores:
            return 0.0
        
        # Weight different pattern types
        pattern_weights = {
            'binding_arbitration': 0.8,
            'arbitration_clause': 0.7,
            'waiver_clauses': 0.6
        }
        
        weighted_scores = []
        for pattern, score in confidence_scores.items():
            weight = pattern_weights.get(pattern, 0.5)
            weighted_scores.append(score * weight)
        
        if weighted_scores:
            return min(sum(weighted_scores) / len(weighted_scores), 1.0)
        
        return 0.0
    
    def _detect_legal_domain(self, patterns_detected: List[str]) -> Optional[str]:
        """Detect the legal domain based on detected patterns."""
        if any('arbitration' in pattern for pattern in patterns_detected):
            return 'arbitration'
        elif any('waiver' in pattern for pattern in patterns_detected):
            return 'dispute_resolution'
        else:
            return 'contract'
    
    async def batch_detect_patterns(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None
    ) -> List[PatternDetectionResult]:
        """
        Detect patterns in multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            languages: List of languages for each text
            
        Returns:
            List of PatternDetectionResult objects
        """
        tasks = []
        for i, text in enumerate(texts):
            language = languages[i] if languages and i < len(languages) else None
            tasks.append(self.detect_patterns(text, language))
        
        return await asyncio.gather(*tasks)


class LanguageModelManager:
    """Main manager class coordinating all language model functionality."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.embedding_engine = MultilingualEmbeddingEngine(config_path)
        self.similarity_engine = CrossLingualSimilarityEngine(self.embedding_engine)
        self.pattern_detector = ArbitrationPatternDetector(
            self.embedding_engine, self.similarity_engine
        )
        
        # Performance monitoring
        self.performance_stats = {
            'embeddings_generated': 0,
            'similarities_computed': 0,
            'patterns_detected': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def analyze_document(
        self,
        text: str,
        language: Optional[str] = None,
        include_embeddings: bool = False,
        include_similarity: bool = False,
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a document for arbitration content.
        
        Args:
            text: Document text to analyze
            language: Language of the document
            include_embeddings: Whether to include embeddings in result
            include_similarity: Whether to compute similarity with reference texts
            reference_texts: Reference texts for similarity comparison
            
        Returns:
            Comprehensive analysis results
        """
        results = {}
        
        # Pattern detection
        pattern_result = await self.pattern_detector.detect_patterns(text, language)
        results['pattern_detection'] = {
            'patterns_detected': pattern_result.patterns_detected,
            'confidence_scores': pattern_result.confidence_scores,
            'arbitration_probability': pattern_result.arbitration_probability,
            'key_phrases': pattern_result.key_phrases,
            'legal_domain': pattern_result.legal_domain
        }
        
        # Embeddings (optional)
        if include_embeddings:
            embedding_result = await self.embedding_engine.get_embeddings(
                text, ModelType.SENTENCE_TRANSFORMER, language
            )
            results['embeddings'] = {
                'model_used': embedding_result.model_used,
                'embedding_dim': len(embedding_result.embeddings),
                'confidence': embedding_result.confidence,
                'processing_time': embedding_result.processing_time
            }
            if include_embeddings == 'full':
                results['embeddings']['vectors'] = embedding_result.embeddings.tolist()
        
        # Similarity analysis (optional)
        if include_similarity and reference_texts:
            similarity_results = []
            for ref_text in reference_texts:
                sim_result = await self.similarity_engine.compute_similarity(
                    text, ref_text, language, None
                )
                similarity_results.append({
                    'reference_text': ref_text[:100] + "..." if len(ref_text) > 100 else ref_text,
                    'similarity_score': sim_result.similarity_score,
                    'cross_lingual': sim_result.cross_lingual,
                    'confidence': sim_result.confidence
                })
            results['similarity_analysis'] = similarity_results
        
        # Update performance stats
        self.performance_stats['patterns_detected'] += 1
        
        return results
    
    async def build_document_index(
        self,
        documents: List[Dict[str, Any]],
        index_name: str = "arbitration_docs"
    ):
        """
        Build searchable index of documents.
        
        Args:
            documents: List of documents with 'text' and optional 'metadata'
            index_name: Name for the index
        """
        texts = [doc['text'] for doc in documents]
        metadata = [doc.get('metadata', {}) for doc in documents]
        
        # Generate embeddings for all documents
        embedding_results = await self.embedding_engine.get_batch_embeddings(texts)
        embeddings = [result.embeddings for result in embedding_results]
        
        # Build FAISS index
        self.embedding_engine.build_faiss_index(embeddings, index_name, metadata)
        
        logger.info(f"Built document index '{index_name}' with {len(documents)} documents")
    
    async def search_similar_documents(
        self,
        query_text: str,
        index_name: str = "arbitration_docs",
        k: int = 5,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the index.
        
        Args:
            query_text: Query text
            index_name: Name of the index to search
            k: Number of results to return
            language: Language of the query
            
        Returns:
            List of similar documents with metadata
        """
        # Get query embedding
        query_embedding = await self.embedding_engine.get_embeddings(
            query_text, ModelType.SENTENCE_TRANSFORMER, language
        )
        
        # Search in index
        results = self.embedding_engine.search_similar(
            query_embedding.embeddings, index_name, k
        )
        
        # Format results
        formatted_results = []
        for distance, metadata in results:
            formatted_results.append({
                'similarity_score': 1.0 - (distance / 2.0),  # Convert distance to similarity
                'metadata': metadata,
                'distance': distance
            })
        
        return formatted_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self.embedding_engine.embedding_cache.clear()
        self.similarity_engine.similarity_cache.clear()
        logger.info("Cleared all caches")
    
    async def benchmark_models(
        self,
        test_texts: List[str],
        languages: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different models on test texts.
        
        Args:
            test_texts: List of test texts
            languages: List of languages for test texts
            
        Returns:
            Benchmark results with timing and performance metrics
        """
        results = {}
        
        for model_type in [ModelType.MULTILINGUAL_BERT, ModelType.XLM_ROBERTA, ModelType.SENTENCE_TRANSFORMER]:
            if model_type not in self.embedding_engine.models:
                continue
                
            start_time = time.time()
            
            # Generate embeddings for all test texts
            embedding_results = await self.embedding_engine.get_batch_embeddings(
                test_texts, model_type
            )
            
            total_time = time.time() - start_time
            avg_time = total_time / len(test_texts)
            
            results[model_type.value] = {
                'total_time': total_time,
                'average_time_per_text': avg_time,
                'texts_processed': len(test_texts),
                'average_confidence': np.mean([r.confidence for r in embedding_results])
            }
        
        return results