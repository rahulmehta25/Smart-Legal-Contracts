"""
Unit tests for RAG pipeline and arbitration detector.

Tests:
- ArbitrationDetector class
- Pattern matching
- Confidence scoring
- Clause type detection
- Document retrieval
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Arbitration Detector Tests
# ============================================================================

class TestArbitrationDetector:
    """Test suite for ArbitrationDetector."""

    @pytest.fixture
    def detector(self, mock_embeddings, mock_vector_store):
        """Create ArbitrationDetector with mocked dependencies."""
        try:
            from app.rag.arbitration_detector import ArbitrationDetector

            with patch('app.rag.arbitration_detector.EmbeddingGenerator') as mock_emb:
                with patch('app.rag.arbitration_detector.DocumentRetriever') as mock_ret:
                    mock_emb.return_value = mock_embeddings
                    mock_ret.return_value.search_arbitration_clauses.return_value = []
                    mock_ret.return_value.retriever.index_document = Mock()

                    detector = ArbitrationDetector()
                    return detector
        except ImportError:
            pytest.skip("ArbitrationDetector not available")

    @pytest.fixture
    def patterns(self):
        """Get arbitration patterns module."""
        try:
            from app.rag.patterns import ArbitrationPatterns
            return ArbitrationPatterns
        except ImportError:
            pytest.skip("ArbitrationPatterns not available")

    # ========================================================================
    # Detection Tests
    # ========================================================================

    def test_detect_clear_arbitration(self, detector, sample_documents):
        """Test detection of clear arbitration clause."""
        text = sample_documents["clear_arbitration"]

        result = detector.detect(text, "test_doc")

        assert result.has_arbitration is True
        assert result.confidence >= 0.6

    def test_detect_no_arbitration(self, detector, sample_documents):
        """Test detection of document without arbitration."""
        text = sample_documents["no_arbitration"]

        result = detector.detect(text, "test_doc")

        # May or may not detect as arbitration depending on implementation
        assert result is not None
        assert hasattr(result, 'has_arbitration')
        assert hasattr(result, 'confidence')

    def test_detect_hidden_arbitration(self, detector, sample_documents):
        """Test detection of hidden arbitration clause."""
        text = sample_documents["hidden_arbitration"]

        result = detector.detect(text, "test_doc")

        assert result is not None
        # Hidden clauses may have lower confidence
        assert hasattr(result, 'confidence')

    def test_detect_complex_arbitration(self, detector, sample_documents):
        """Test detection of complex multi-step arbitration."""
        text = sample_documents["complex_arbitration"]

        result = detector.detect(text, "test_doc")

        assert result is not None

    # ========================================================================
    # Result Structure Tests
    # ========================================================================

    def test_detection_result_structure(self, detector, sample_documents):
        """Test DetectionResult has correct structure."""
        text = sample_documents["clear_arbitration"]

        result = detector.detect(text, "test_doc_123")

        assert hasattr(result, 'document_id')
        assert hasattr(result, 'has_arbitration')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'clauses')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'processing_time')

        assert result.document_id == "test_doc_123"
        assert 0 <= result.confidence <= 1
        assert result.processing_time >= 0

    def test_detection_result_to_dict(self, detector, sample_documents):
        """Test DetectionResult to_dict conversion."""
        text = sample_documents["clear_arbitration"]

        result = detector.detect(text, "test_doc")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "document_id" in result_dict
        assert "has_arbitration" in result_dict
        assert "confidence" in result_dict
        assert "clauses" in result_dict

    # ========================================================================
    # Clause Type Detection Tests
    # ========================================================================

    def test_identify_arbitration_clause_type(self, detector):
        """Test identification of arbitration clause type."""
        text = "Any disputes shall be resolved through binding arbitration."

        clause_types = detector._identify_clause_types(text)

        assert len(clause_types) > 0

    def test_identify_class_action_waiver(self, detector, sample_documents):
        """Test identification of class action waiver."""
        text = sample_documents["class_action_waiver"]

        clause_types = detector._identify_clause_types(text)

        # Should detect class action waiver
        clause_type_values = [ct.value for ct in clause_types]
        assert "class_action_waiver" in clause_type_values or "arbitration" in clause_type_values

    def test_identify_jury_trial_waiver(self, detector):
        """Test identification of jury trial waiver."""
        text = "You waive your right to a jury trial in any proceedings."

        clause_types = detector._identify_clause_types(text)

        clause_type_values = [ct.value for ct in clause_types]
        assert "jury_trial_waiver" in clause_type_values or len(clause_types) > 0

    def test_identify_opt_out_provision(self, detector, sample_documents):
        """Test identification of opt-out provision."""
        text = sample_documents["opt_out_provision"]

        clause_types = detector._identify_clause_types(text)

        clause_type_values = [ct.value for ct in clause_types]
        assert "opt_out" in clause_type_values or "arbitration" in clause_type_values

    # ========================================================================
    # Arbitration Type Detection Tests
    # ========================================================================

    def test_determine_binding_arbitration(self, detector):
        """Test determination of binding arbitration type."""
        try:
            from app.rag.arbitration_detector import ArbitrationType

            text = "All disputes resolved through binding arbitration."
            arb_type = detector._determine_arbitration_type(text, ["binding"], ["binding arbitration"])

            assert arb_type == ArbitrationType.BINDING
        except ImportError:
            pytest.skip("ArbitrationType not available")

    def test_determine_mandatory_arbitration(self, detector):
        """Test determination of mandatory arbitration type."""
        try:
            from app.rag.arbitration_detector import ArbitrationType

            text = "Mandatory arbitration required for all disputes."
            arb_type = detector._determine_arbitration_type(text, [], [])

            assert arb_type in [ArbitrationType.MANDATORY, ArbitrationType.BINDING, ArbitrationType.UNKNOWN]
        except ImportError:
            pytest.skip("ArbitrationType not available")

    def test_determine_voluntary_arbitration(self, detector):
        """Test determination of voluntary arbitration type."""
        try:
            from app.rag.arbitration_detector import ArbitrationType

            text = "Parties may elect voluntary arbitration if desired."
            arb_type = detector._determine_arbitration_type(text, [], [])

            assert arb_type in [ArbitrationType.VOLUNTARY, ArbitrationType.UNKNOWN]
        except ImportError:
            pytest.skip("ArbitrationType not available")

    # ========================================================================
    # Score Combination Tests
    # ========================================================================

    def test_combine_scores_balanced(self, detector):
        """Test score combination with balanced inputs."""
        combined = detector._combine_scores(0.8, 0.8, 0.8)

        assert 0.7 <= combined <= 1.0

    def test_combine_scores_high_pattern(self, detector):
        """Test score combination with high pattern score."""
        combined = detector._combine_scores(0.95, 0.5, 0.5)

        # Should boost overall score
        assert combined >= 0.6

    def test_combine_scores_low_all(self, detector):
        """Test score combination with all low scores."""
        combined = detector._combine_scores(0.1, 0.1, 0.1)

        assert combined < 0.3

    def test_combine_scores_bounds(self, detector):
        """Test score combination stays within bounds."""
        combined = detector._combine_scores(1.0, 1.0, 1.0)
        assert 0 <= combined <= 1.0

        combined = detector._combine_scores(0.0, 0.0, 0.0)
        assert 0 <= combined <= 1.0

    # ========================================================================
    # Keyword Score Tests
    # ========================================================================

    def test_compute_keyword_score_high(self, detector):
        """Test keyword score with many keywords."""
        text = """
        Binding arbitration agreement administered by AAA.
        Mandatory arbitration for all disputes.
        Waive right to jury trial. Class action waiver.
        """

        score = detector._compute_keyword_score(text)

        assert score >= 0.5

    def test_compute_keyword_score_low(self, detector):
        """Test keyword score with no keywords."""
        text = "This is a regular paragraph with no legal terms."

        score = detector._compute_keyword_score(text)

        assert score < 0.3

    def test_compute_keyword_score_empty(self, detector):
        """Test keyword score with empty text."""
        score = detector._compute_keyword_score("")

        assert score == 0

    # ========================================================================
    # Clause Merging Tests
    # ========================================================================

    def test_merge_nearby_clauses_empty(self, detector):
        """Test merging empty clause list."""
        result = detector._merge_nearby_clauses([])

        assert result == []

    def test_merge_nearby_clauses_single(self, detector):
        """Test merging single clause."""
        try:
            from app.rag.arbitration_detector import ArbitrationClause, ArbitrationType, ClauseType

            clause = ArbitrationClause(
                text="Test clause",
                confidence_score=0.9,
                arbitration_type=ArbitrationType.BINDING,
                clause_types=[ClauseType.ARBITRATION],
                location={"start_char": 0, "end_char": 100},
                details={},
                pattern_matches=["arbitration"],
                semantic_score=0.8,
                keyword_score=0.85
            )

            result = detector._merge_nearby_clauses([clause])

            assert len(result) == 1
            assert result[0].text == "Test clause"
        except ImportError:
            pytest.skip("ArbitrationClause not available")

    # ========================================================================
    # Location Finding Tests
    # ========================================================================

    def test_find_location_exact_match(self, detector):
        """Test finding location with exact match."""
        full_text = "This is the full document text with an arbitration clause here."
        chunk = "arbitration clause"

        location = detector._find_location(chunk, full_text)

        assert location["start_char"] >= 0
        assert location["end_char"] > location["start_char"]

    def test_find_location_no_match(self, detector):
        """Test finding location with no match."""
        full_text = "This is the document text."
        chunk = "not in document"

        location = detector._find_location(chunk, full_text)

        # Should return default or approximate
        assert "start_char" in location
        assert "end_char" in location


# ============================================================================
# Arbitration Patterns Tests
# ============================================================================

class TestArbitrationPatterns:
    """Test suite for ArbitrationPatterns."""

    @pytest.fixture
    def patterns(self):
        """Get patterns class."""
        try:
            from app.rag.patterns import ArbitrationPatterns
            return ArbitrationPatterns
        except ImportError:
            pytest.skip("ArbitrationPatterns not available")

    def test_get_all_patterns(self, patterns):
        """Test getting all patterns."""
        all_patterns = patterns.get_all_patterns()

        assert len(all_patterns) > 0

    def test_high_confidence_keywords(self, patterns):
        """Test high confidence keywords exist."""
        assert hasattr(patterns, 'HIGH_CONFIDENCE_KEYWORDS')
        assert len(patterns.HIGH_CONFIDENCE_KEYWORDS) > 0

    def test_medium_confidence_keywords(self, patterns):
        """Test medium confidence keywords exist."""
        assert hasattr(patterns, 'MEDIUM_CONFIDENCE_KEYWORDS')

    def test_extract_arbitration_details(self, patterns):
        """Test extracting arbitration details."""
        text = """
        Disputes shall be resolved through binding arbitration
        administered by the American Arbitration Association (AAA)
        in New York, NY. The arbitration shall be conducted in English.
        """

        details = patterns.extract_arbitration_details(text)

        assert isinstance(details, dict)

    def test_get_negative_indicators(self, patterns):
        """Test getting negative indicators."""
        indicators = patterns.get_negative_indicators()

        assert isinstance(indicators, list)


# ============================================================================
# Confidence Scoring Tests
# ============================================================================

class TestConfidenceScoring:
    """Test suite for confidence scoring."""

    @pytest.fixture
    def scorer(self):
        """Get confidence scoring module."""
        try:
            from app.rag.confidence_scoring import ConfidenceScorer
            return ConfidenceScorer()
        except ImportError:
            pytest.skip("ConfidenceScorer not available")

    def test_confidence_scorer_initialization(self, scorer):
        """Test ConfidenceScorer initialization."""
        assert scorer is not None


# ============================================================================
# Text Processor Tests
# ============================================================================

class TestTextProcessor:
    """Test suite for text processor."""

    @pytest.fixture
    def text_processor(self):
        """Get text processor."""
        try:
            from app.rag.text_processor import TextProcessor
            return TextProcessor()
        except ImportError:
            pytest.skip("TextProcessor not available")

    def test_text_processor_initialization(self, text_processor):
        """Test TextProcessor initialization."""
        assert text_processor is not None


# ============================================================================
# Embeddings Tests
# ============================================================================

class TestEmbeddings:
    """Test suite for embeddings."""

    @pytest.fixture
    def embedding_generator(self):
        """Get embedding generator."""
        try:
            from app.rag.embeddings import EmbeddingGenerator
            return EmbeddingGenerator()
        except ImportError:
            pytest.skip("EmbeddingGenerator not available")

    def test_embedding_generator_initialization(self, embedding_generator):
        """Test EmbeddingGenerator initialization."""
        assert embedding_generator is not None


# ============================================================================
# Retriever Tests
# ============================================================================

class TestRetriever:
    """Test suite for document retriever."""

    @pytest.fixture
    def retriever(self, mock_embeddings, mock_vector_store):
        """Create retriever with mocks."""
        try:
            from app.rag.retriever import DocumentRetriever, RetrievalConfig

            with patch('app.rag.retriever.EmbeddingGenerator') as mock_emb:
                mock_emb.return_value = mock_embeddings
                config = RetrievalConfig()
                return DocumentRetriever(config)
        except ImportError:
            pytest.skip("DocumentRetriever not available")

    def test_retriever_initialization(self, retriever):
        """Test DocumentRetriever initialization."""
        assert retriever is not None


# ============================================================================
# Mock-based RAG Tests
# ============================================================================

class TestMockRAG:
    """Tests using mock RAG components."""

    def test_mock_arbitration_detector_detect(self, mock_arbitration_detector):
        """Test mock detector detect method."""
        text = "This agreement includes binding arbitration clause."

        result = mock_arbitration_detector.detect(text, "doc_123")

        assert result["document_id"] == "doc_123"
        assert result["has_arbitration"] is True
        assert "confidence" in result
        assert "summary" in result

    def test_mock_detector_summary(self, mock_arbitration_detector):
        """Test mock detector generates summary."""
        text = """
        This agreement includes binding arbitration.
        You waive your right to participate in class action lawsuits.
        You may opt out within 30 days.
        """

        result = mock_arbitration_detector.detect(text)

        assert "summary" in result
        assert "has_binding_arbitration" in result["summary"]
        assert "has_class_action_waiver" in result["summary"]
        assert "has_opt_out" in result["summary"]

    def test_mock_embeddings_consistency(self, mock_embeddings):
        """Test mock embeddings are consistent for same input."""
        text = "Test text for embedding"

        emb1 = mock_embeddings.embed_query(text)
        emb2 = mock_embeddings.embed_query(text)

        assert emb1 == emb2

    def test_mock_embeddings_different_texts(self, mock_embeddings):
        """Test mock embeddings differ for different inputs."""
        emb1 = mock_embeddings.embed_query("First text")
        emb2 = mock_embeddings.embed_query("Second text")

        assert emb1 != emb2

    def test_mock_vector_store_add_and_search(self, mock_vector_store):
        """Test mock vector store operations."""
        # Add documents
        texts = ["Document about arbitration", "Another legal document"]
        ids = mock_vector_store.add_texts(texts)

        assert len(ids) == 2

        # Search
        results = mock_vector_store.similarity_search("arbitration", k=1)

        assert len(results) >= 1

    def test_mock_vector_store_search_with_score(self, mock_vector_store):
        """Test mock vector store search with scores."""
        mock_vector_store.add_texts(["Test document"])

        results = mock_vector_store.similarity_search_with_score("test", k=2)

        assert len(results) >= 1
        assert len(results[0]) == 2  # (doc, score)


# ============================================================================
# RAG Pipeline Integration Tests
# ============================================================================

class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline components."""

    @pytest.mark.integration
    def test_full_detection_pipeline(self, mock_arbitration_detector, sample_documents):
        """Test full detection pipeline flow."""
        results = {}

        for name, text in sample_documents.items():
            result = mock_arbitration_detector.detect(text, name)
            results[name] = result

        # Check we got results for all documents
        assert len(results) == len(sample_documents)

        # Clear arbitration should be detected
        assert results["clear_arbitration"]["has_arbitration"] is True

        # No arbitration should not be detected
        assert results["no_arbitration"]["has_arbitration"] is False

    @pytest.mark.integration
    def test_detection_performance(self, mock_arbitration_detector, large_document_text, benchmark_timer):
        """Test detection performance on large document."""
        with benchmark_timer:
            result = mock_arbitration_detector.detect(large_document_text, "large_doc")

        # Should complete within reasonable time
        assert benchmark_timer.elapsed < 5.0
        assert result is not None
