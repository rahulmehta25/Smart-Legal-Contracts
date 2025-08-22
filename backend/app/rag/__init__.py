"""
RAG system for arbitration clause detection.
"""

from .arbitration_detector import (
    ArbitrationDetector,
    ArbitrationType,
    ClauseType,
    ArbitrationClause,
    DetectionResult
)
from .embeddings import (
    EmbeddingGenerator,
    EmbeddingConfig,
    SemanticChunker,
    EmbeddingSimilarityScorer
)
from .retriever import (
    DocumentRetriever,
    HybridRetriever,
    RetrievalConfig,
    DocumentProcessor
)
from .patterns import (
    ArbitrationPatterns,
    ArbitrationPattern
)

__all__ = [
    # Main detector
    'ArbitrationDetector',
    'ArbitrationType',
    'ClauseType',
    'ArbitrationClause',
    'DetectionResult',
    
    # Embeddings
    'EmbeddingGenerator',
    'EmbeddingConfig',
    'SemanticChunker',
    'EmbeddingSimilarityScorer',
    
    # Retriever
    'DocumentRetriever',
    'HybridRetriever',
    'RetrievalConfig',
    'DocumentProcessor',
    
    # Patterns
    'ArbitrationPatterns',
    'ArbitrationPattern',
]

__version__ = '1.0.0'