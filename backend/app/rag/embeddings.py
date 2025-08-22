"""
Embedding generation and management utilities for arbitration detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
import os
from pathlib import Path
import pickle
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 256
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = "/backend/data/embedding_cache"


class EmbeddingGenerator:
    """Manages embedding generation and caching for text documents."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding generator."""
        self.config = config or EmbeddingConfig()
        self.model = None
        self.cache = {}
        self._initialize_model()
        self._setup_cache_dir()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            self.model.max_seq_length = self.config.max_seq_length
            logger.info(f"Loaded embedding model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _setup_cache_dir(self):
        """Setup cache directory for embeddings."""
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(f"{text}_{self.config.model_name}".encode()).hexdigest()
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Check disk cache
            cache_file = self.cache_path / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                        self.cache[cache_key] = embedding
                        return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
        
        # Generate new embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        # Cache the embedding
        if use_cache:
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = embedding
            
            # Save to disk
            try:
                cache_file = self.cache_path / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        batch_size = batch_size or self.config.batch_size
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=show_progress and i == 0
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Compute similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Array of document embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(idx, similarities[idx]) for idx in top_indices]
        return results
    
    def create_embeddings_index(self, texts: List[str]) -> Dict[str, Any]:
        """
        Create an embeddings index for a collection of texts.
        
        Args:
            texts: List of texts to index
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        embeddings = self.generate_embeddings_batch(texts, show_progress=True)
        
        index = {
            'embeddings': embeddings,
            'texts': texts,
            'model_name': self.config.model_name,
            'embedding_dim': self.config.embedding_dim,
            'num_documents': len(texts)
        }
        
        return index
    
    def save_index(self, index: Dict[str, Any], filepath: str):
        """Save embeddings index to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(index, f)
        logger.info(f"Saved embeddings index to {filepath}")
    
    def load_index(self, filepath: str) -> Dict[str, Any]:
        """Load embeddings index from file."""
        with open(filepath, 'rb') as f:
            index = pickle.load(f)
        logger.info(f"Loaded embeddings index from {filepath}")
        return index


class SemanticChunker:
    """Chunk documents semantically for better embedding quality."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, 
                 chunk_size: int = 512, overlap: int = 128):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_generator: Embedding generator instance
            chunk_size: Target size for chunks in characters
            overlap: Overlap between chunks
        """
        self.embedding_generator = embedding_generator
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text into semantically meaningful segments.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_char': len(' '.join(chunks)) if chunks else 0,
                    'end_char': len(' '.join(chunks)) + len(chunk_text) if chunks else len(chunk_text),
                    'chunk_id': len(chunks)
                })
                
                # Handle overlap
                if self.overlap > 0:
                    overlap_sentences = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        if overlap_size >= self.overlap:
                            break
                        overlap_sentences.insert(0, sent)
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_char': len(' '.join([c['text'] for c in chunks])) if chunks else 0,
                'end_char': len(text),
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_and_embed(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text and generate embeddings for each chunk.
        
        Args:
            text: Input text
            
        Returns:
            List of chunks with embeddings
        """
        chunks = self.chunk_text(text)
        
        for chunk in chunks:
            chunk['embedding'] = self.embedding_generator.generate_embedding(chunk['text'])
        
        return chunks


class EmbeddingSimilarityScorer:
    """Score text similarity using embeddings."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """Initialize the similarity scorer."""
        self.embedding_generator = embedding_generator
        self.arbitration_examples = []
        self.non_arbitration_examples = []
        self._load_training_examples()
    
    def _load_training_examples(self):
        """Load training examples for similarity scoring."""
        # These would normally be loaded from files
        self.arbitration_examples = [
            "Any dispute arising out of or relating to these Terms shall be resolved through binding arbitration.",
            "You agree to resolve all disputes through mandatory arbitration administered by JAMS.",
            "By using this service, you waive your right to a jury trial and agree to binding arbitration.",
            "All claims must be resolved through individual arbitration, not class action.",
            "Disputes will be settled by binding arbitration under the Federal Arbitration Act.",
        ]
        
        self.non_arbitration_examples = [
            "These terms are governed by the laws of the State of California.",
            "You may contact our customer service team for any questions.",
            "We reserve the right to modify these terms at any time.",
            "Your privacy is important to us. Please review our Privacy Policy.",
            "This service is provided as-is without any warranties.",
        ]
    
    def compute_arbitration_similarity(self, text: str) -> float:
        """
        Compute similarity to known arbitration clause examples.
        
        Args:
            text: Input text to score
            
        Returns:
            Similarity score (0-1)
        """
        text_embedding = self.embedding_generator.generate_embedding(text)
        
        # Get embeddings for examples
        arb_embeddings = np.array([
            self.embedding_generator.generate_embedding(ex) 
            for ex in self.arbitration_examples
        ])
        
        non_arb_embeddings = np.array([
            self.embedding_generator.generate_embedding(ex)
            for ex in self.non_arbitration_examples
        ])
        
        # Compute similarities
        arb_similarities = cosine_similarity(
            text_embedding.reshape(1, -1), 
            arb_embeddings
        )[0]
        
        non_arb_similarities = cosine_similarity(
            text_embedding.reshape(1, -1),
            non_arb_embeddings
        )[0]
        
        # Calculate weighted score
        avg_arb_sim = np.mean(arb_similarities)
        avg_non_arb_sim = np.mean(non_arb_similarities)
        max_arb_sim = np.max(arb_similarities)
        
        # Weighted scoring formula
        score = (0.5 * avg_arb_sim + 0.3 * max_arb_sim - 0.2 * avg_non_arb_sim)
        
        # Normalize to 0-1 range
        return max(0, min(1, score))
    
    def find_similar_clauses(
        self, 
        query_text: str, 
        document_chunks: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find chunks most similar to query text.
        
        Args:
            query_text: Query text
            document_chunks: List of document chunks with embeddings
            top_k: Number of results to return
            
        Returns:
            Top k similar chunks with scores
        """
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        chunk_embeddings = np.array([chunk['embedding'] for chunk in document_chunks])
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embeddings
        )[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = document_chunks[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)
        
        return results