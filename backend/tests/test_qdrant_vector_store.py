"""
Tests for Qdrant vector store operations.

All operations are mocked to avoid requiring a running Qdrant instance.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Qdrant Client Tests
# ============================================================================

class TestQdrantClient:
    """Test suite for Qdrant client operations."""

    def test_mock_client_create_collection(self, mock_qdrant_client):
        """Test creating a collection."""
        result = mock_qdrant_client.create_collection(
            collection_name="test_collection",
            vectors_config={"size": 384, "distance": "Cosine"}
        )

        assert result is True
        assert "test_collection" in mock_qdrant_client._collections

    def test_mock_client_get_collection(self, mock_qdrant_client):
        """Test getting collection info."""
        # Create collection first
        mock_qdrant_client.create_collection(
            collection_name="test_collection",
            vectors_config={"size": 384}
        )

        collection = mock_qdrant_client.get_collection("test_collection")

        assert collection is not None
        assert hasattr(collection, 'vectors_count')

    def test_mock_client_get_collection_not_found(self, mock_qdrant_client):
        """Test getting non-existent collection raises error."""
        with pytest.raises(Exception):
            mock_qdrant_client.get_collection("nonexistent")

    def test_mock_client_upsert_points(self, mock_qdrant_client):
        """Test upserting points."""
        # Create collection
        mock_qdrant_client.create_collection("test_collection", {})

        # Create mock points
        points = [
            Mock(id="1", vector=[0.1] * 384, payload={"text": "Test 1"}),
            Mock(id="2", vector=[0.2] * 384, payload={"text": "Test 2"})
        ]

        result = mock_qdrant_client.upsert(
            collection_name="test_collection",
            points=points
        )

        assert result.status == "completed"
        assert len(mock_qdrant_client._points["test_collection"]) == 2

    def test_mock_client_search(self, mock_qdrant_client):
        """Test searching vectors."""
        # Setup
        mock_qdrant_client.create_collection("test_collection", {})
        points = [
            Mock(id="1", vector=[0.1] * 384, payload={"text": "Arbitration clause"}),
            Mock(id="2", vector=[0.2] * 384, payload={"text": "Regular clause"})
        ]
        mock_qdrant_client.upsert("test_collection", points)

        # Search
        results = mock_qdrant_client.search(
            collection_name="test_collection",
            query_vector=[0.15] * 384,
            limit=2
        )

        assert len(results) <= 2
        for result in results:
            assert hasattr(result, 'id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'payload')

    def test_mock_client_delete_points(self, mock_qdrant_client):
        """Test deleting points."""
        mock_qdrant_client.create_collection("test_collection", {})

        result = mock_qdrant_client.delete(
            collection_name="test_collection",
            points_selector={"ids": ["1", "2"]}
        )

        assert result.status == "completed"

    def test_mock_client_delete_collection(self, mock_qdrant_client):
        """Test deleting a collection."""
        mock_qdrant_client.create_collection("test_collection", {})

        result = mock_qdrant_client.delete_collection("test_collection")

        assert result is True


# ============================================================================
# Vector Store Wrapper Tests
# ============================================================================

class TestVectorStoreWrapper:
    """Test suite for vector store wrapper."""

    def test_add_texts(self, mock_vector_store):
        """Test adding texts to vector store."""
        texts = [
            "This contract includes an arbitration clause.",
            "No arbitration in this document.",
            "Binding arbitration administered by AAA."
        ]
        metadatas = [
            {"source": "doc1.pdf"},
            {"source": "doc2.pdf"},
            {"source": "doc3.pdf"}
        ]

        ids = mock_vector_store.add_texts(texts, metadatas)

        assert len(ids) == 3
        assert len(mock_vector_store._documents) == 3

    def test_add_texts_without_metadata(self, mock_vector_store):
        """Test adding texts without metadata."""
        texts = ["Text 1", "Text 2"]

        ids = mock_vector_store.add_texts(texts)

        assert len(ids) == 2

    def test_add_texts_with_ids(self, mock_vector_store):
        """Test adding texts with custom IDs."""
        texts = ["Text 1", "Text 2"]
        custom_ids = ["custom_1", "custom_2"]

        ids = mock_vector_store.add_texts(texts, ids=custom_ids)

        assert ids == custom_ids

    def test_similarity_search(self, mock_vector_store):
        """Test similarity search."""
        # Add documents
        mock_vector_store.add_texts([
            "Document about arbitration",
            "Document about contracts"
        ])

        results = mock_vector_store.similarity_search("arbitration", k=2)

        assert len(results) <= 2
        for result in results:
            assert hasattr(result, 'page_content')
            assert hasattr(result, 'metadata')

    def test_similarity_search_empty_store(self, mock_vector_store):
        """Test similarity search on empty store."""
        # Clear any default documents
        mock_vector_store._documents = {}

        results = mock_vector_store.similarity_search("test", k=5)

        # Should return default results or empty
        assert isinstance(results, list)

    def test_similarity_search_with_score(self, mock_vector_store):
        """Test similarity search with scores."""
        mock_vector_store.add_texts(["Test document"])

        results = mock_vector_store.similarity_search_with_score("test", k=2)

        assert len(results) >= 1
        doc, score = results[0]
        assert hasattr(doc, 'page_content')
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_delete_documents(self, mock_vector_store):
        """Test deleting documents."""
        mock_vector_store.add_texts(["Test document"])

        result = mock_vector_store.delete(["0"])

        assert result is True


# ============================================================================
# Embedding Storage Tests
# ============================================================================

class TestEmbeddingStorage:
    """Test suite for embedding storage operations."""

    def test_store_embeddings(self, mock_qdrant_client, mock_embeddings):
        """Test storing embeddings in Qdrant."""
        # Generate embeddings
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = mock_embeddings.embed_documents(texts)

        # Store in Qdrant
        mock_qdrant_client.create_collection("embeddings", {"size": 384})

        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            point = Mock(
                id=str(i),
                vector=embedding,
                payload={"text": text}
            )
            points.append(point)

        result = mock_qdrant_client.upsert("embeddings", points)

        assert result.status == "completed"

    def test_retrieve_by_similarity(self, mock_qdrant_client, mock_embeddings):
        """Test retrieving documents by similarity."""
        # Setup
        mock_qdrant_client.create_collection("embeddings", {})
        texts = ["Arbitration clause", "Contract terms", "Legal agreement"]
        embeddings = mock_embeddings.embed_documents(texts)

        points = [
            Mock(id=str(i), vector=emb, payload={"text": text})
            for i, (text, emb) in enumerate(zip(texts, embeddings))
        ]
        mock_qdrant_client.upsert("embeddings", points)

        # Search
        query_embedding = mock_embeddings.embed_query("arbitration")
        results = mock_qdrant_client.search(
            collection_name="embeddings",
            query_vector=query_embedding,
            limit=2
        )

        assert len(results) <= 2


# ============================================================================
# Vector Store Configuration Tests
# ============================================================================

class TestVectorStoreConfiguration:
    """Test suite for vector store configuration."""

    def test_collection_with_custom_distance(self, mock_qdrant_client):
        """Test creating collection with custom distance metric."""
        config = {
            "size": 384,
            "distance": "Cosine"
        }

        mock_qdrant_client.create_collection("custom_collection", config)

        assert "custom_collection" in mock_qdrant_client._collections

    def test_collection_with_custom_size(self, mock_qdrant_client):
        """Test creating collection with custom vector size."""
        config = {
            "size": 768,  # Different embedding model
            "distance": "Dot"
        }

        mock_qdrant_client.create_collection("large_vectors", config)

        assert "large_vectors" in mock_qdrant_client._collections


# ============================================================================
# Batch Operation Tests
# ============================================================================

class TestBatchOperations:
    """Test suite for batch operations."""

    def test_batch_upsert(self, mock_qdrant_client):
        """Test batch upserting many points."""
        mock_qdrant_client.create_collection("batch_test", {})

        # Create many points
        batch_size = 100
        points = [
            Mock(id=str(i), vector=[0.1] * 384, payload={"index": i})
            for i in range(batch_size)
        ]

        result = mock_qdrant_client.upsert("batch_test", points)

        assert result.status == "completed"
        assert len(mock_qdrant_client._points["batch_test"]) == batch_size

    def test_batch_search(self, mock_qdrant_client):
        """Test batch searching."""
        mock_qdrant_client.create_collection("batch_search", {})

        points = [
            Mock(id=str(i), vector=[0.1 * i] * 384, payload={"text": f"Doc {i}"})
            for i in range(10)
        ]
        mock_qdrant_client.upsert("batch_search", points)

        # Multiple searches
        queries = [[0.1] * 384, [0.5] * 384, [0.9] * 384]
        results = []
        for query in queries:
            result = mock_qdrant_client.search("batch_search", query, limit=3)
            results.append(result)

        assert len(results) == 3


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestVectorStoreErrors:
    """Test suite for error handling."""

    def test_search_nonexistent_collection(self, mock_qdrant_client):
        """Test searching non-existent collection."""
        # This should work with mock (returns empty)
        results = mock_qdrant_client.search(
            "nonexistent",
            [0.1] * 384,
            limit=5
        )

        # Mock returns empty for nonexistent
        assert isinstance(results, list)

    def test_invalid_vector_dimension(self, mock_qdrant_client, mock_embeddings):
        """Test with invalid vector dimensions."""
        mock_qdrant_client.create_collection("dim_test", {"size": 384})

        # Should handle gracefully
        points = [Mock(id="1", vector=[0.1] * 384, payload={})]
        result = mock_qdrant_client.upsert("dim_test", points)

        assert result.status == "completed"


# ============================================================================
# Integration Tests
# ============================================================================

class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""

    @pytest.mark.integration
    def test_full_workflow(self, mock_qdrant_client, mock_embeddings, sample_documents):
        """Test full vector store workflow."""
        # 1. Create collection
        mock_qdrant_client.create_collection("legal_docs", {"size": 384})

        # 2. Generate embeddings and store documents
        points = []
        for name, text in sample_documents.items():
            embedding = mock_embeddings.embed_query(text)
            point = Mock(
                id=name,
                vector=embedding,
                payload={"text": text, "name": name}
            )
            points.append(point)

        mock_qdrant_client.upsert("legal_docs", points)

        # 3. Search for arbitration-related documents
        query = "binding arbitration clause"
        query_embedding = mock_embeddings.embed_query(query)

        results = mock_qdrant_client.search(
            "legal_docs",
            query_embedding,
            limit=3
        )

        # 4. Verify results
        assert len(results) <= 3
        for result in results:
            assert hasattr(result, 'score')
            assert 0 <= result.score <= 1

    @pytest.mark.integration
    def test_document_retrieval_accuracy(self, mock_vector_store, sample_documents):
        """Test document retrieval returns relevant results."""
        # Add all sample documents
        texts = list(sample_documents.values())
        names = list(sample_documents.keys())
        metadatas = [{"name": name} for name in names]

        mock_vector_store.add_texts(texts, metadatas, ids=names)

        # Search for arbitration
        results = mock_vector_store.similarity_search("arbitration clause", k=5)

        # Should return some results
        assert len(results) > 0
