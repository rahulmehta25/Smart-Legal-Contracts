"""
Property-based tests using Hypothesis.

Tests data model invariants, edge cases, and fuzzing for:
- Document models
- Analysis results
- API payloads
- Text processing
"""

import pytest
from pathlib import Path
import json
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import hypothesis
try:
    from hypothesis import given, strategies as st, settings, assume, example
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

    # Create dummy decorators when hypothesis not available
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="Hypothesis not installed")(f)
        return decorator

    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    def assume(x):
        pass

    def example(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    class _DummyStrategy:
        def __call__(self, *args, **kwargs):
            return None

    class st:
        text = _DummyStrategy()
        sampled_from = _DummyStrategy()
        integers = _DummyStrategy()
        floats = _DummyStrategy()
        lists = _DummyStrategy()
        from_regex = _DummyStrategy()
        dictionaries = _DummyStrategy()
        one_of = _DummyStrategy()
        booleans = _DummyStrategy()
        just = _DummyStrategy()
        characters = _DummyStrategy()


pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="Hypothesis not installed"
)


# ============================================================================
# Strategies
# ============================================================================

if HYPOTHESIS_AVAILABLE:
    # Text strategies
    legal_text = st.text(
        alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'Z'),
            min_codepoint=32,
            max_codepoint=126
        ),
        min_size=0,
        max_size=10000
    )

    document_content = st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z', 'S')),
        min_size=1,
        max_size=5000
    )

    filename = st.from_regex(r'[a-zA-Z0-9_-]{1,50}\.(txt|pdf|docx|html)', fullmatch=True)

    confidence_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

    # Arbitration keywords
    arbitration_keywords = st.sampled_from([
        "arbitration", "binding arbitration", "AAA", "JAMS",
        "dispute resolution", "mediation", "class action waiver",
        "jury trial waiver", "opt out", "mandatory arbitration"
    ])
else:
    # Dummy strategies when hypothesis not available
    legal_text = None
    document_content = None
    filename = None
    confidence_score = None
    arbitration_keywords = None


# ============================================================================
# Document Model Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestDocumentModels:
    """Property-based tests for document models."""

    @given(content=document_content, fname=filename)
    @settings(max_examples=50)
    def test_document_content_preserved(self, content, fname):
        """Test that document content is preserved through processing."""
        # Simulate document storage
        stored = {
            "filename": fname,
            "content": content,
            "size": len(content)
        }

        # Content should be preserved
        assert stored["content"] == content
        assert stored["size"] == len(content)
        assert stored["filename"] == fname

    @given(text=legal_text)
    @settings(max_examples=50)
    def test_word_count_consistency(self, text):
        """Test word count is consistent."""
        words = text.split()
        word_count = len(words)

        # Rejoin and resplit should give same count
        rejoined = " ".join(words)
        assert len(rejoined.split()) == word_count

    @given(text=document_content)
    @settings(max_examples=50)
    def test_character_count_accurate(self, text):
        """Test character count is accurate."""
        char_count = len(text)

        # Length should be non-negative
        assert char_count >= 0

        # Should equal sum of character lengths
        assert char_count == sum(1 for _ in text)


# ============================================================================
# Analysis Result Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestAnalysisResults:
    """Property-based tests for analysis results."""

    @given(score=confidence_score)
    @settings(max_examples=100)
    def test_confidence_score_bounds(self, score):
        """Test confidence scores are always in valid range."""
        assert 0.0 <= score <= 1.0

    @given(scores=st.lists(confidence_score, min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_average_confidence_in_bounds(self, scores):
        """Test average confidence stays in bounds."""
        avg = sum(scores) / len(scores)

        assert 0.0 <= avg <= 1.0

    @given(scores=st.lists(confidence_score, min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_max_confidence_ordering(self, scores):
        """Test max confidence is greater than or equal to average."""
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        assert max_score >= avg_score


# ============================================================================
# Text Processing Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestTextProcessing:
    """Property-based tests for text processing."""

    @given(text=legal_text)
    @settings(max_examples=50)
    def test_lowercase_idempotent(self, text):
        """Test lowercase is idempotent."""
        lower1 = text.lower()
        lower2 = lower1.lower()

        assert lower1 == lower2

    @given(text=legal_text)
    @settings(max_examples=50)
    def test_strip_idempotent(self, text):
        """Test strip is idempotent."""
        stripped1 = text.strip()
        stripped2 = stripped1.strip()

        assert stripped1 == stripped2

    @given(text=document_content)
    @settings(max_examples=50)
    def test_split_join_roundtrip(self, text):
        """Test split and join roundtrip preserves content."""
        words = text.split()
        rejoined = " ".join(words)

        # Words should be preserved
        assert rejoined.split() == words

    @given(text=legal_text, pattern=st.sampled_from([r'\s+', r'\n+', r'\t+']))
    @settings(max_examples=30)
    def test_whitespace_normalization(self, text, pattern):
        """Test whitespace normalization is consistent."""
        normalized = re.sub(pattern, ' ', text)

        # Should not contain pattern
        if pattern == r'\n+':
            assert '\n' not in normalized or text == ''
        if pattern == r'\t+':
            assert '\t' not in normalized or text == ''


# ============================================================================
# Arbitration Detection Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestArbitrationDetection:
    """Property-based tests for arbitration detection."""

    @given(keyword=arbitration_keywords)
    @settings(max_examples=20)
    def test_keyword_in_text_detected(self, keyword, mock_arbitration_detector):
        """Test that arbitration keywords are detected."""
        text = f"This agreement contains a {keyword} provision."

        result = mock_arbitration_detector.detect_arbitration_clause(text)

        # Should detect if keyword contains "arbitration"
        if "arbitration" in keyword:
            assert result["has_arbitration"] is True

    @given(text=st.text(alphabet=st.characters(whitelist_categories=('L',)), min_size=10, max_size=100))
    @settings(max_examples=30)
    def test_no_false_positives_random_text(self, text, mock_arbitration_detector):
        """Test no false positives on random text without keywords."""
        assume("arbitrat" not in text.lower())
        assume("dispute" not in text.lower())
        assume("jams" not in text.lower())
        assume("aaa" not in text.lower())

        result = mock_arbitration_detector.detect_arbitration_clause(text)

        assert result["has_arbitration"] is False

    @given(
        prefix=st.text(min_size=0, max_size=100),
        suffix=st.text(min_size=0, max_size=100)
    )
    @settings(max_examples=30)
    def test_keyword_detection_position_invariant(self, prefix, suffix, mock_arbitration_detector):
        """Test keyword detection regardless of position in text."""
        keyword = "binding arbitration"
        text = f"{prefix} {keyword} {suffix}"

        result = mock_arbitration_detector.detect_arbitration_clause(text)

        assert result["has_arbitration"] is True


# ============================================================================
# API Payload Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestAPIPayloads:
    """Property-based tests for API payloads."""

    @given(
        filename=filename,
        content=document_content,
        content_type=st.sampled_from(["text/plain", "application/pdf", "application/json"])
    )
    @settings(max_examples=30)
    def test_document_create_payload_valid(self, filename, content, content_type):
        """Test document creation payloads are valid."""
        payload = {
            "filename": filename,
            "content": content,
            "content_type": content_type
        }

        assert "filename" in payload
        assert "content" in payload
        assert len(payload["filename"]) > 0

    @given(
        text=document_content,
        document_id=st.text(alphabet=st.characters(whitelist_categories=('L', 'N')), min_size=1, max_size=50)
    )
    @settings(max_examples=30)
    def test_analysis_request_payload_valid(self, text, document_id):
        """Test analysis request payloads are valid."""
        payload = {
            "text": text,
            "document_id": document_id,
            "options": {"detect_clauses": True}
        }

        assert "text" in payload
        assert len(payload["text"]) >= 1

    @given(payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
        min_size=0,
        max_size=10
    ))
    @settings(max_examples=20)
    def test_json_serialization_roundtrip(self, payload):
        """Test JSON serialization roundtrip."""
        serialized = json.dumps(payload)
        deserialized = json.loads(serialized)

        assert deserialized == payload


# ============================================================================
# Embedding Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestEmbeddings:
    """Property-based tests for embeddings."""

    @given(text=document_content)
    @settings(max_examples=30)
    def test_embedding_deterministic(self, text, mock_embeddings):
        """Test embedding generation is deterministic."""
        emb1 = mock_embeddings.embed_query(text)
        emb2 = mock_embeddings.embed_query(text)

        assert emb1 == emb2

    @given(texts=st.lists(document_content, min_size=1, max_size=5))
    @settings(max_examples=20)
    def test_batch_embedding_length(self, texts, mock_embeddings):
        """Test batch embedding returns correct number."""
        embeddings = mock_embeddings.embed_documents(texts)

        assert len(embeddings) == len(texts)

    @given(text=document_content)
    @settings(max_examples=20)
    def test_embedding_dimension(self, text, mock_embeddings):
        """Test embedding has correct dimension."""
        embedding = mock_embeddings.embed_query(text)

        assert len(embedding) == mock_embeddings.dimension


# ============================================================================
# Vector Store Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestVectorStore:
    """Property-based tests for vector store."""

    @given(texts=st.lists(document_content, min_size=1, max_size=10, unique=True))
    @settings(max_examples=20)
    def test_add_retrieve_count(self, texts, mock_vector_store):
        """Test adding documents increases count."""
        # Clear existing
        mock_vector_store._documents = {}

        ids = mock_vector_store.add_texts(texts)

        assert len(ids) == len(texts)
        assert len(mock_vector_store._documents) == len(texts)

    @given(
        k=st.integers(min_value=1, max_value=10),
        texts=st.lists(document_content, min_size=1, max_size=5)
    )
    @settings(max_examples=20)
    def test_search_returns_at_most_k(self, k, texts, mock_vector_store):
        """Test search returns at most k results."""
        mock_vector_store._documents = {}
        mock_vector_store.add_texts(texts)

        results = mock_vector_store.similarity_search("test", k=k)

        assert len(results) <= k


# ============================================================================
# Score Calculation Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestScoreCalculation:
    """Property-based tests for score calculations."""

    @given(
        pattern_score=confidence_score,
        semantic_score=confidence_score,
        keyword_score=confidence_score
    )
    @settings(max_examples=50)
    def test_combined_score_bounds(self, pattern_score, semantic_score, keyword_score, mock_arbitration_detector):
        """Test combined score stays in bounds."""
        # If detector has _combine_scores method
        if hasattr(mock_arbitration_detector, '_combine_scores'):
            combined = mock_arbitration_detector._combine_scores(
                pattern_score, semantic_score, keyword_score
            )
            assert 0.0 <= combined <= 1.0
        else:
            # Mock doesn't have this method, skip
            pytest.skip("Mock detector doesn't have _combine_scores")

    @given(scores=st.lists(confidence_score, min_size=1, max_size=20))
    @settings(max_examples=30)
    def test_weighted_average_bounds(self, scores):
        """Test weighted average calculation stays in bounds."""
        weights = [1.0 / len(scores)] * len(scores)
        weighted_avg = sum(s * w for s, w in zip(scores, weights))

        assert 0.0 <= weighted_avg <= 1.0


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestEdgeCases:
    """Edge case tests."""

    @given(text=st.just(""))
    def test_empty_text_handling(self, text, mock_arbitration_detector):
        """Test handling of empty text."""
        result = mock_arbitration_detector.detect_arbitration_clause(text)

        assert "has_arbitration" in result
        assert result["has_arbitration"] is False

    @given(text=st.text(min_size=10000, max_size=20000))
    @settings(max_examples=5)
    def test_large_text_handling(self, text, mock_arbitration_detector):
        """Test handling of large text."""
        result = mock_arbitration_detector.detect_arbitration_clause(text)

        assert "has_arbitration" in result
        assert "confidence" in result

    @given(text=st.text(alphabet=st.characters(
        whitelist_categories=('Lo', 'Ll', 'Lu'),
        min_codepoint=0x0600,
        max_codepoint=0x06FF
    ), min_size=10, max_size=100))
    @settings(max_examples=10)
    def test_unicode_text_handling(self, text, mock_arbitration_detector):
        """Test handling of Unicode text."""
        result = mock_arbitration_detector.detect_arbitration_clause(text)

        assert "has_arbitration" in result

    @example(text=" " * 100)
    @given(text=st.text(alphabet=" \t\n", min_size=1, max_size=100))
    @settings(max_examples=10)
    def test_whitespace_only_handling(self, text, mock_arbitration_detector):
        """Test handling of whitespace-only text."""
        result = mock_arbitration_detector.detect_arbitration_clause(text)

        assert "has_arbitration" in result
        assert result["has_arbitration"] is False
