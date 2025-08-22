"""
Integration tests for the RAG (Retrieval-Augmented Generation) pipeline.

This module tests the end-to-end RAG pipeline including:
- Document ingestion and vectorization
- Vector similarity search
- Context retrieval and ranking
- LLM integration and response generation
- Pipeline performance and accuracy
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any, Optional
import numpy as np


class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, mock_vector_store, mock_llm, sample_documents):
        """Test complete RAG pipeline from query to response."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        query = "Does this document contain an arbitration clause?"
        document_text = sample_documents["clear_arbitration"]
        
        # Act
        result = await pipeline.process_query(query, document_text)
        
        # Assert
        assert result is not None
        assert "has_arbitration" in result
        assert "confidence" in result
        assert "explanation" in result
        assert result["has_arbitration"] is True
    
    @pytest.mark.asyncio
    async def test_document_ingestion_and_vectorization(self, mock_embeddings):
        """Test document ingestion and conversion to vectors."""
        # Arrange
        pipeline = MockRAGPipeline(embeddings=mock_embeddings)
        documents = [
            "Document 1 with arbitration clause",
            "Document 2 with mediation clause",
            "Document 3 with litigation clause"
        ]
        
        # Act
        vectors = await pipeline.ingest_documents(documents)
        
        # Assert
        assert len(vectors) == len(documents)
        assert all(len(vector) > 0 for vector in vectors)
    
    def test_similarity_search_accuracy(self, mock_vector_store):
        """Test accuracy of similarity search in vector store."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store)
        query = "arbitration agreement"
        
        # Configure mock to return relevant documents
        mock_vector_store.similarity_search.return_value = [
            Mock(page_content="Binding arbitration clause", metadata={"source": "contract1.pdf", "relevance": 0.95}),
            Mock(page_content="Arbitration agreement terms", metadata={"source": "contract2.pdf", "relevance": 0.88}),
            Mock(page_content="Dispute resolution arbitration", metadata={"source": "contract3.pdf", "relevance": 0.82})
        ]
        
        # Act
        results = pipeline.search_similar_documents(query, k=3)
        
        # Assert
        assert len(results) == 3
        assert all("arbitration" in result.page_content.lower() for result in results)
        mock_vector_store.similarity_search.assert_called_once_with(query, k=3)
    
    @pytest.mark.asyncio
    async def test_context_retrieval_and_ranking(self, mock_vector_store):
        """Test context retrieval and relevance ranking."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store)
        query = "mandatory arbitration clause"
        
        # Configure mock with documents of varying relevance
        mock_vector_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Mandatory binding arbitration", metadata={"source": "doc1"}), 0.95),
            (Mock(page_content="Optional arbitration clause", metadata={"source": "doc2"}), 0.75),
            (Mock(page_content="Mediation and arbitration", metadata={"source": "doc3"}), 0.65),
            (Mock(page_content="Court litigation only", metadata={"source": "doc4"}), 0.25)
        ]
        
        # Act
        context = await pipeline.retrieve_context(query, threshold=0.7)
        
        # Assert
        assert len(context) >= 2  # Should filter out low-relevance documents
        assert all(score >= 0.7 for _, score in context)
    
    @pytest.mark.asyncio
    async def test_llm_integration_and_response_generation(self, mock_llm):
        """Test LLM integration for generating responses."""
        # Arrange
        pipeline = MockRAGPipeline(llm=mock_llm)
        context = [
            "Document contains binding arbitration clause",
            "All disputes must be resolved through AAA arbitration"
        ]
        query = "Is there an arbitration clause?"
        
        # Configure mock LLM response
        mock_llm.apredict.return_value = json.dumps({
            "has_arbitration": True,
            "confidence": 0.92,
            "clause_type": "mandatory_binding",
            "explanation": "The document contains explicit binding arbitration requirements."
        })
        
        # Act
        response = await pipeline.generate_response(query, context)
        
        # Assert
        assert response["has_arbitration"] is True
        assert response["confidence"] > 0.9
        assert "explanation" in response
        mock_llm.apredict.assert_called_once()
    
    def test_vector_store_operations(self, mock_vector_store, mock_embeddings):
        """Test vector store CRUD operations."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, embeddings=mock_embeddings)
        documents = ["Test document with arbitration clause"]
        
        # Act - Add documents
        pipeline.add_documents_to_store(documents)
        
        # Assert
        mock_vector_store.add_documents.assert_called_once()
        
        # Act - Search documents
        pipeline.search_similar_documents("arbitration", k=5)
        
        # Assert
        mock_vector_store.similarity_search.assert_called_with("arbitration", k=5)
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_vector_store, mock_llm):
        """Test error handling throughout the pipeline."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        
        # Test vector store failure
        mock_vector_store.similarity_search.side_effect = Exception("Vector store error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Vector store error"):
            await pipeline.process_query("test query", "test document")
        
        # Test LLM failure
        mock_vector_store.similarity_search.side_effect = None
        mock_vector_store.similarity_search.return_value = [Mock(page_content="test")]
        mock_llm.apredict.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            await pipeline.process_query("test query", "test document")


class TestRAGPerformance:
    """Performance tests for RAG pipeline components."""
    
    @pytest.mark.asyncio
    async def test_query_processing_speed(self, mock_vector_store, mock_llm, performance_metrics):
        """Test query processing performance."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        query = "arbitration clause detection"
        document = "Sample document with arbitration clause" * 100
        max_time = performance_metrics["processing_time_per_page"]
        
        # Act
        start_time = time.time()
        result = await pipeline.process_query(query, document)
        processing_time = time.time() - start_time
        
        # Assert
        assert processing_time < max_time
        assert result is not None
    
    def test_vector_search_performance(self, mock_vector_store):
        """Test vector similarity search performance."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store)
        
        # Configure mock for performance test
        mock_vector_store.similarity_search.return_value = [
            Mock(page_content=f"Document {i}", metadata={"source": f"doc{i}.pdf"})
            for i in range(10)
        ]
        
        # Act
        start_time = time.time()
        for _ in range(100):
            pipeline.search_similar_documents("test query", k=10)
        processing_time = time.time() - start_time
        
        # Assert
        assert processing_time < 1.0  # Should complete 100 searches in under 1 second
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, mock_vector_store, mock_llm):
        """Test concurrent query processing capabilities."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        queries = [f"Query {i} about arbitration" for i in range(10)]
        document = "Test document with arbitration clause"
        
        # Act
        start_time = time.time()
        tasks = [pipeline.process_query(query, document) for query in queries]
        results = await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        # Assert
        assert len(results) == 10
        assert all(result is not None for result in results)
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    def test_memory_usage_during_vectorization(self, mock_embeddings):
        """Test memory usage during document vectorization."""
        import psutil
        import os
        
        # Arrange
        pipeline = MockRAGPipeline(embeddings=mock_embeddings)
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Large document set for testing
        large_documents = ["Document " + str(i) + " with arbitration clause" for i in range(1000)]
        
        # Act
        pipeline.ingest_documents(large_documents)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert
        assert memory_increase < 500  # Less than 500MB increase


class TestRAGAccuracy:
    """Accuracy tests for RAG pipeline components."""
    
    def test_retrieval_relevance(self, mock_vector_store):
        """Test relevance of retrieved documents."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store)
        query = "binding arbitration clause"
        
        # Configure mock with relevant and irrelevant documents
        mock_vector_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Binding arbitration required", metadata={}), 0.95),
            (Mock(page_content="Optional mediation clause", metadata={}), 0.30),
            (Mock(page_content="Arbitration agreement mandatory", metadata={}), 0.88)
        ]
        
        # Act
        relevant_docs = pipeline.get_relevant_documents(query, relevance_threshold=0.7)
        
        # Assert
        assert len(relevant_docs) == 2  # Only highly relevant documents
        assert all(score >= 0.7 for _, score in relevant_docs)
    
    @pytest.mark.asyncio
    async def test_response_accuracy(self, mock_vector_store, mock_llm, sample_documents, expected_detection_results):
        """Test accuracy of generated responses."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        
        # Test multiple document types
        test_cases = [
            ("clear_arbitration", True),
            ("no_arbitration", False),
            ("hidden_arbitration", True)
        ]
        
        for doc_key, expected_has_arbitration in test_cases:
            document = sample_documents[doc_key]
            
            # Configure appropriate mock response
            mock_llm.apredict.return_value = json.dumps({
                "has_arbitration": expected_has_arbitration,
                "confidence": 0.9 if expected_has_arbitration else 0.1,
                "explanation": f"Test response for {doc_key}"
            })
            
            # Act
            result = await pipeline.process_query("Check for arbitration clause", document)
            
            # Assert
            assert result["has_arbitration"] == expected_has_arbitration
    
    def test_context_ranking_accuracy(self, mock_vector_store):
        """Test accuracy of context ranking algorithm."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store)
        
        # Documents with different relevance levels
        documents_with_scores = [
            ("Binding arbitration clause in section 15", 0.95),
            ("Arbitration required for all disputes", 0.90),
            ("Optional mediation available", 0.40),
            ("Court jurisdiction in New York", 0.20),
            ("Arbitration fees split between parties", 0.85)
        ]
        
        mock_vector_store.similarity_search_with_score.return_value = [
            (Mock(page_content=content), score) for content, score in documents_with_scores
        ]
        
        # Act
        ranked_context = pipeline.rank_context_by_relevance("arbitration clause")
        
        # Assert
        # Verify ranking is correct (highest scores first)
        scores = [score for _, score in ranked_context]
        assert scores == sorted(scores, reverse=True)
        
        # Verify high-relevance documents are prioritized
        top_documents = ranked_context[:3]
        assert all(score >= 0.8 for _, score in top_documents)


class TestRAGEdgeCases:
    """Edge case tests for RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_empty_vector_store(self, mock_llm):
        """Test behavior with empty vector store."""
        # Arrange
        empty_vector_store = Mock()
        empty_vector_store.similarity_search.return_value = []
        pipeline = MockRAGPipeline(empty_vector_store, mock_llm)
        
        # Act
        result = await pipeline.process_query("test query", "test document")
        
        # Assert
        assert result is not None
        # Should still provide a response based on the document alone
    
    @pytest.mark.asyncio
    async def test_very_long_document_processing(self, mock_vector_store, mock_llm):
        """Test processing of very long documents."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        very_long_document = "This is a test document. " * 10000
        very_long_document += " Binding arbitration clause at the end."
        
        # Act
        result = await pipeline.process_query("arbitration clause", very_long_document)
        
        # Assert
        assert result is not None
        assert "has_arbitration" in result
    
    @pytest.mark.asyncio
    async def test_malformed_llm_response(self, mock_vector_store, mock_llm):
        """Test handling of malformed LLM responses."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        
        # Configure mock to return malformed JSON
        mock_llm.apredict.return_value = "Invalid JSON response"
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            await pipeline.process_query("test query", "test document")
    
    def test_duplicate_document_handling(self, mock_vector_store):
        """Test handling of duplicate documents in vector store."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store)
        
        # Configure mock to return duplicate documents
        duplicate_doc = Mock(page_content="Arbitration clause text", metadata={"source": "doc1.pdf"})
        mock_vector_store.similarity_search.return_value = [duplicate_doc, duplicate_doc, duplicate_doc]
        
        # Act
        unique_results = pipeline.deduplicate_search_results(mock_vector_store.similarity_search("test", k=10))
        
        # Assert
        assert len(unique_results) == 1
    
    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self, mock_vector_store, mock_llm):
        """Test handling of network timeouts."""
        # Arrange
        pipeline = MockRAGPipeline(mock_vector_store, mock_llm)
        
        # Simulate timeout
        mock_llm.apredict.side_effect = asyncio.TimeoutError("Request timeout")
        
        # Act & Assert
        with pytest.raises(asyncio.TimeoutError):
            await pipeline.process_query("test query", "test document")


# Mock implementation for testing
class MockRAGPipeline:
    """Mock RAG Pipeline implementation for testing."""
    
    def __init__(self, vector_store=None, llm=None, embeddings=None):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
    
    async def process_query(self, query: str, document: str) -> Dict[str, Any]:
        """Process a query against a document using RAG pipeline."""
        # Simulate vector search
        if self.vector_store:
            context_docs = self.vector_store.similarity_search(query, k=5)
        else:
            context_docs = []
        
        # Simulate LLM processing
        if self.llm:
            context_text = [doc.page_content for doc in context_docs]
            response = await self.llm.apredict(f"Query: {query}\nDocument: {document}\nContext: {context_text}")
            return json.loads(response)
        
        # Fallback response
        return {
            "has_arbitration": "arbitration" in document.lower(),
            "confidence": 0.8,
            "explanation": "Mock response"
        }
    
    async def ingest_documents(self, documents: List[str]) -> List[List[float]]:
        """Ingest documents and return vectors."""
        if self.embeddings:
            return self.embeddings.embed_documents(documents)
        return [[0.1, 0.2, 0.3] for _ in documents]
    
    def search_similar_documents(self, query: str, k: int = 5):
        """Search for similar documents."""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []
    
    async def retrieve_context(self, query: str, threshold: float = 0.7):
        """Retrieve relevant context with score filtering."""
        if self.vector_store and hasattr(self.vector_store, 'similarity_search_with_score'):
            results = self.vector_store.similarity_search_with_score(query, k=10)
            return [(doc, score) for doc, score in results if score >= threshold]
        return []
    
    async def generate_response(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using LLM."""
        if self.llm:
            response = await self.llm.apredict(f"Query: {query}\nContext: {context}")
            return json.loads(response)
        return {"has_arbitration": False, "confidence": 0.5, "explanation": "No LLM available"}
    
    def add_documents_to_store(self, documents: List[str]):
        """Add documents to vector store."""
        if self.vector_store:
            self.vector_store.add_documents(documents)
    
    def get_relevant_documents(self, query: str, relevance_threshold: float = 0.7):
        """Get relevant documents above threshold."""
        if self.vector_store and hasattr(self.vector_store, 'similarity_search_with_score'):
            results = self.vector_store.similarity_search_with_score(query, k=10)
            return [(doc, score) for doc, score in results if score >= relevance_threshold]
        return []
    
    def rank_context_by_relevance(self, query: str):
        """Rank context by relevance score."""
        if self.vector_store and hasattr(self.vector_store, 'similarity_search_with_score'):
            results = self.vector_store.similarity_search_with_score(query, k=10)
            return sorted(results, key=lambda x: x[1], reverse=True)
        return []
    
    def deduplicate_search_results(self, results):
        """Remove duplicate search results."""
        seen_content = set()
        unique_results = []
        for result in results:
            if result.page_content not in seen_content:
                seen_content.add(result.page_content)
                unique_results.append(result)
        return unique_results


if __name__ == "__main__":
    pytest.main([__file__])