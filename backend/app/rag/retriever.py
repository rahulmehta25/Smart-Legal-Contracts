"""
Document retrieval system for arbitration clause detection.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import logging
import re
from collections import defaultdict

from .embeddings import EmbeddingGenerator, SemanticChunker, EmbeddingSimilarityScorer
from .patterns import ArbitrationPatterns

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k_retrieval: int = 10
    similarity_threshold: float = 0.6
    rerank_top_k: int = 5
    use_hybrid_search: bool = True
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7


class DocumentProcessor:
    """Process and prepare documents for retrieval."""
    
    def __init__(self, config: RetrievalConfig = None):
        """Initialize the document processor."""
        self.config = config or RetrievalConfig()
        self.patterns = ArbitrationPatterns()
    
    def preprocess_document(self, text: str) -> str:
        """
        Preprocess document text for better retrieval.
        
        Args:
            text: Raw document text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[•·■□▪]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`´]', "'", text)
        
        # Remove page numbers and headers/footers if present
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract logical sections from document.
        
        Args:
            text: Document text
            
        Returns:
            List of sections with titles and content
        """
        sections = []
        
        # Common section headers in Terms of Use
        section_patterns = [
            r'^\d+\.?\s+[A-Z][A-Za-z\s]+',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # All caps headers
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:',  # Title case with colon
        ]
        
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': ''}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    # Save current section if it has content
                    if current_section['content']:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {'title': line, 'content': ''}
                    is_header = True
                    break
            
            if not is_header:
                current_section['content'] += line + ' '
        
        # Add last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def identify_arbitration_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Identify sections likely to contain arbitration clauses.
        
        Args:
            sections: List of document sections
            
        Returns:
            Sections with arbitration likelihood scores
        """
        arbitration_sections = []
        
        # Keywords that suggest arbitration content in section titles
        title_keywords = [
            'arbitration', 'dispute', 'resolution', 'legal', 'claims',
            'litigation', 'binding', 'waiver', 'jury', 'class action'
        ]
        
        for section in sections:
            title_lower = section['title'].lower()
            content_lower = section['content'].lower()
            
            # Score based on title
            title_score = 0
            for keyword in title_keywords:
                if keyword in title_lower:
                    title_score += 0.3
            
            # Score based on content keywords
            content_score = 0
            for pattern in ArbitrationPatterns.HIGH_CONFIDENCE_KEYWORDS:
                if pattern[0] in content_lower:
                    content_score += pattern[1]
            
            for pattern in ArbitrationPatterns.MEDIUM_CONFIDENCE_KEYWORDS:
                if pattern[0] in content_lower:
                    content_score += pattern[1] * 0.5
            
            # Combine scores
            total_score = min(1.0, title_score + content_score * 0.3)
            
            if total_score > 0.2:  # Threshold for potential arbitration content
                arbitration_sections.append({
                    'title': section['title'],
                    'content': section['content'],
                    'score': total_score
                })
        
        return sorted(arbitration_sections, key=lambda x: x['score'], reverse=True)


class HybridRetriever:
    """Hybrid retrieval combining keyword and semantic search."""
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 config: RetrievalConfig = None):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_generator: Embedding generator instance
            config: Retrieval configuration
        """
        self.embedding_generator = embedding_generator
        self.config = config or RetrievalConfig()
        self.chunker = SemanticChunker(
            embedding_generator,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        self.similarity_scorer = EmbeddingSimilarityScorer(embedding_generator)
        self.patterns = ArbitrationPatterns()
        self.document_processor = DocumentProcessor(config)
        self.document_index = {}
    
    def index_document(self, document_id: str, text: str) -> Dict[str, Any]:
        """
        Index a document for retrieval.
        
        Args:
            document_id: Unique document identifier
            text: Document text
            
        Returns:
            Indexing results
        """
        # Preprocess document
        processed_text = self.document_processor.preprocess_document(text)
        
        # Extract sections
        sections = self.document_processor.extract_sections(processed_text)
        
        # Chunk and embed document
        chunks = self.chunker.chunk_and_embed(processed_text)
        
        # Create inverted index for keyword search
        inverted_index = self._create_inverted_index(chunks)
        
        # Store in document index
        self.document_index[document_id] = {
            'chunks': chunks,
            'sections': sections,
            'inverted_index': inverted_index,
            'original_text': text,
            'processed_text': processed_text
        }
        
        return {
            'document_id': document_id,
            'num_chunks': len(chunks),
            'num_sections': len(sections),
            'indexed': True
        }
    
    def _create_inverted_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Create inverted index for keyword search.
        
        Args:
            chunks: Document chunks
            
        Returns:
            Inverted index mapping terms to chunk indices
        """
        inverted_index = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            # Tokenize chunk text
            tokens = re.findall(r'\b\w+\b', chunk['text'].lower())
            
            # Add to inverted index
            for token in set(tokens):
                inverted_index[token].append(i)
        
        return dict(inverted_index)
    
    def retrieve(self, 
                 query: str, 
                 document_id: Optional[str] = None,
                 top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            document_id: Optional specific document to search
            top_k: Number of results to return
            
        Returns:
            Retrieved chunks with scores
        """
        top_k = top_k or self.config.top_k_retrieval
        
        if document_id and document_id not in self.document_index:
            logger.warning(f"Document {document_id} not found in index")
            return []
        
        # Get documents to search
        if document_id:
            documents = {document_id: self.document_index[document_id]}
        else:
            documents = self.document_index
        
        all_results = []
        
        for doc_id, doc_data in documents.items():
            chunks = doc_data['chunks']
            inverted_index = doc_data['inverted_index']
            
            if self.config.use_hybrid_search:
                # Hybrid search
                results = self._hybrid_search(query, chunks, inverted_index, top_k)
            else:
                # Semantic search only
                results = self._semantic_search(query, chunks, top_k)
            
            # Add document ID to results
            for result in results:
                result['document_id'] = doc_id
            
            all_results.extend(results)
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _semantic_search(self, 
                        query: str, 
                        chunks: List[Dict[str, Any]], 
                        top_k: int) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            chunks: Document chunks with embeddings
            top_k: Number of results
            
        Returns:
            Retrieved chunks with scores
        """
        query_embedding = self.embedding_generator.generate_embedding(query)
        chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embeddings
        )[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.config.similarity_threshold:
                result = chunks[idx].copy()
                result['score'] = float(similarities[idx])
                result['retrieval_type'] = 'semantic'
                results.append(result)
        
        return results
    
    def _keyword_search(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]], 
                       inverted_index: Dict[str, List[int]],
                       top_k: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query
            chunks: Document chunks
            inverted_index: Inverted index for chunks
            top_k: Number of results
            
        Returns:
            Retrieved chunks with scores
        """
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Score chunks based on keyword matches
        chunk_scores = defaultdict(float)
        
        for token in query_tokens:
            if token in inverted_index:
                for chunk_idx in inverted_index[token]:
                    # TF-IDF-like scoring
                    tf = chunks[chunk_idx]['text'].lower().count(token)
                    idf = np.log(len(chunks) / len(inverted_index[token]))
                    chunk_scores[chunk_idx] += tf * idf
        
        # Check for pattern matches
        for i, chunk in enumerate(chunks):
            chunk_text_lower = chunk['text'].lower()
            
            # Check high confidence patterns
            for pattern, weight, _ in ArbitrationPatterns.HIGH_CONFIDENCE_KEYWORDS:
                if pattern in chunk_text_lower:
                    chunk_scores[i] += weight * 2
            
            # Check regex patterns
            for pattern, weight, _ in ArbitrationPatterns.REGEX_PATTERNS:
                if re.search(pattern, chunk_text_lower):
                    chunk_scores[i] += weight * 1.5
        
        # Get top k chunks
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_idx, score in sorted_chunks:
            if score > 0:
                result = chunks[chunk_idx].copy()
                result['score'] = float(score) / (max(chunk_scores.values()) + 1e-6)  # Normalize
                result['retrieval_type'] = 'keyword'
                results.append(result)
        
        return results
    
    def _hybrid_search(self,
                      query: str,
                      chunks: List[Dict[str, Any]],
                      inverted_index: Dict[str, List[int]],
                      top_k: int) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining keyword and semantic search.
        
        Args:
            query: Search query
            chunks: Document chunks
            inverted_index: Inverted index
            top_k: Number of results
            
        Returns:
            Retrieved chunks with combined scores
        """
        # Get semantic search results
        semantic_results = self._semantic_search(query, chunks, top_k * 2)
        semantic_scores = {r['chunk_id']: r['score'] for r in semantic_results}
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, chunks, inverted_index, top_k * 2)
        keyword_scores = {r['chunk_id']: r['score'] for r in keyword_results}
        
        # Combine scores
        all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        for chunk_id in all_chunk_ids:
            semantic_score = semantic_scores.get(chunk_id, 0)
            keyword_score = keyword_scores.get(chunk_id, 0)
            
            # Weighted combination
            combined_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score
            )
            
            result = chunks[chunk_id].copy()
            result['score'] = combined_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            result['retrieval_type'] = 'hybrid'
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_results[:top_k]
    
    def rerank(self, 
               query: str, 
               results: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank retrieved results for better relevance.
        
        Args:
            query: Original query
            results: Retrieved results to rerank
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        top_k = top_k or self.config.rerank_top_k
        
        for result in results:
            # Compute arbitration-specific similarity
            arb_similarity = self.similarity_scorer.compute_arbitration_similarity(
                result['text']
            )
            
            # Check for specific pattern matches
            pattern_score = 0
            text_lower = result['text'].lower()
            
            for pattern in ArbitrationPatterns.get_all_patterns():
                if pattern.pattern_type == 'regex':
                    if re.search(pattern.pattern, text_lower):
                        pattern_score += pattern.weight
                else:
                    if pattern.pattern in text_lower:
                        pattern_score += pattern.weight
            
            # Combine scores for reranking
            result['rerank_score'] = (
                0.4 * result['score'] +
                0.3 * arb_similarity +
                0.3 * min(1.0, pattern_score / 3)
            )
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results[:top_k]


class DocumentRetriever:
    """High-level document retrieval interface."""
    
    def __init__(self, config: RetrievalConfig = None):
        """Initialize the document retriever."""
        self.config = config or RetrievalConfig()
        self.embedding_generator = EmbeddingGenerator()
        self.retriever = HybridRetriever(self.embedding_generator, config)
        self.patterns = ArbitrationPatterns()
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process and index a document.
        
        Args:
            document_path: Path to document file
            
        Returns:
            Processing results
        """
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get document ID from filename
        document_id = Path(document_path).stem
        
        # Index document
        index_result = self.retriever.index_document(document_id, text)
        
        return index_result
    
    def search_arbitration_clauses(self, 
                                   document_id: Optional[str] = None,
                                   custom_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for arbitration clauses in indexed documents.
        
        Args:
            document_id: Optional specific document to search
            custom_query: Optional custom search query
            
        Returns:
            Retrieved arbitration clause candidates
        """
        # Default query for arbitration clauses
        if custom_query:
            query = custom_query
        else:
            query = (
                "binding arbitration mandatory arbitration dispute resolution "
                "waive jury trial class action waiver arbitration agreement "
                "Federal Arbitration Act FAA JAMS AAA arbitrator"
            )
        
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, document_id)
        
        # Rerank for arbitration relevance
        reranked_results = self.retriever.rerank(query, results)
        
        # Extract arbitration details for top results
        for result in reranked_results:
            result['arbitration_details'] = ArbitrationPatterns.extract_arbitration_details(
                result['text']
            )
        
        return reranked_results
    
    def get_document_sections(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get sections from an indexed document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document sections
        """
        if document_id not in self.retriever.document_index:
            return []
        
        doc_data = self.retriever.document_index[document_id]
        return doc_data.get('sections', [])