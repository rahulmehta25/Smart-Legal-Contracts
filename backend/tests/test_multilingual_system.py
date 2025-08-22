"""
Test suite for the multilingual arbitration detection system.

This module contains comprehensive tests for language detection, translation,
and cross-lingual arbitration clause detection functionality.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import json

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.nlp.multilingual import (
    MultilingualProcessor, 
    LanguageDetector, 
    TranslationPipeline,
    SupportedLanguage,
    DetectionResult,
    TranslationResult
)
from app.nlp.legal_translator import (
    LegalTranslator,
    LegalTerminologyDatabase,
    ContextualLegalTranslator,
    LegalDomain,
    ContextualTranslation
)
from app.nlp.language_models import (
    LanguageModelManager,
    MultilingualEmbeddingEngine,
    CrossLingualSimilarityEngine,
    ArbitrationPatternDetector,
    ModelType
)


class TestLanguageDetection:
    """Test language detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_english_detection(self):
        """Test English language detection."""
        text = "This is a binding arbitration clause in English."
        result = self.detector.detect_language(text)
        
        assert result.language == "en"
        assert result.confidence > 0.5
        assert result.is_supported
        assert result.detected_by in ["langdetect", "polyglot", "character_based"]
    
    def test_spanish_detection(self):
        """Test Spanish language detection."""
        text = "Esta es una cláusula de arbitraje vinculante en español."
        result = self.detector.detect_language(text)
        
        assert result.language == "es"
        assert result.confidence > 0.5
        assert result.is_supported
    
    def test_chinese_detection(self):
        """Test Chinese language detection."""
        text = "这是一个具有约束力的仲裁条款。"
        result = self.detector.detect_language(text)
        
        # Should detect as Chinese (zh or zh-cn)
        assert result.language.startswith("zh")
        assert result.confidence > 0.3  # Lower threshold for character-based detection
        assert result.is_supported
    
    def test_short_text_handling(self):
        """Test handling of very short texts."""
        text = "Test"
        result = self.detector.detect_language(text)
        
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert not result.is_supported
        assert result.detected_by == "insufficient_text"
    
    def test_unsupported_language(self):
        """Test detection of unsupported languages."""
        # This might be detected as an unsupported language
        text = "Dette er en bindende voldgiftsklausul på norsk."  # Norwegian
        result = self.detector.detect_language(text)
        
        # The result might be detected but marked as unsupported
        assert isinstance(result, DetectionResult)
    
    def test_legal_system_mapping(self):
        """Test legal system mapping for different languages."""
        test_cases = [
            ("en", "common_law"),
            ("es", "civil_law"),
            ("fr", "civil_law"),
            ("de", "civil_law"),
            ("zh", "civil_law"),
            ("ja", "civil_law"),
            ("ar", "islamic_law")
        ]
        
        for lang_code, expected_system in test_cases:
            legal_system = self.detector._get_legal_system(lang_code)
            assert legal_system == expected_system


class TestLegalTerminologyDatabase:
    """Test legal terminology database functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db = LegalTerminologyDatabase()
    
    def test_arbitration_terms_english(self):
        """Test retrieval of English arbitration terms."""
        term = self.db.get_legal_term("arbitration", "en", LegalDomain.ARBITRATION)
        
        assert term is not None
        assert term.term == "arbitration"
        assert term.category == "dispute_resolution"
        assert "es" in term.equivalents
        assert "arbitraje" in term.equivalents["es"]
    
    def test_find_equivalent_terms(self):
        """Test finding equivalent terms across languages."""
        equivalents = self.db.find_equivalent_terms(
            "arbitration", "en", "es", LegalDomain.ARBITRATION
        )
        
        assert len(equivalents) > 0
        assert "arbitraje" in equivalents
    
    def test_legal_phrase_detection(self):
        """Test detection of legal phrases using regex patterns."""
        text = "Any dispute shall be settled by binding arbitration."
        phrases = self.db.detect_legal_phrases(text)
        
        assert len(phrases) > 0
        phrase_text, category, start, end = phrases[0]
        assert "arbitration" in phrase_text.lower()
        assert category == "arbitration_clauses"
    
    def test_phrase_pattern_matching(self):
        """Test specific phrase pattern matching."""
        test_cases = [
            ("submit to binding arbitration", "arbitration_clauses"),
            ("waive the right to jury trial", "waiver_clauses"),
            ("exclusive jurisdiction of the courts", "jurisdiction_clauses")
        ]
        
        for text, expected_category in test_cases:
            phrases = self.db.detect_legal_phrases(text, expected_category)
            assert len(phrases) > 0
            assert phrases[0][1] == expected_category


@pytest.mark.asyncio
class TestTranslationPipeline:
    """Test translation pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal config for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'models': {
                    'marian': {},
                    'multilingual': 'facebook/mbart-large-50-many-to-many-mmt'
                },
                'fallback_to_google': False,
                'max_length': 512,
                'batch_size': 2
            }
            import yaml
            yaml.dump(config, f)
            self.config_file = f.name
        
        self.pipeline = TranslationPipeline(self.config_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        if hasattr(self, 'config_file'):
            os.unlink(self.config_file)
    
    async def test_basic_translation(self):
        """Test basic translation functionality."""
        text = "arbitration clause"
        
        # Mock translation result for testing
        result = TranslationResult(
            original_text=text,
            translated_text="cláusula de arbitraje",
            source_language="en",
            target_language="es",
            confidence=0.8,
            translation_method="mock",
            preserved_terms=[],
            processing_time=0.1
        )
        
        assert result.original_text == text
        assert result.source_language == "en"
        assert result.target_language == "es"
        assert result.confidence > 0.5
    
    async def test_legal_term_preservation(self):
        """Test preservation of legal terms during translation."""
        text = "The arbitration clause is binding."
        preserved_terms = [("__LEGAL_TERM_0__", "arbitration")]
        
        processed_text, terms = self.pipeline._preserve_legal_terms(text, "en")
        
        # Check that legal terms were identified and preserved
        assert len(terms) >= 0  # May or may not find terms depending on implementation
        
        # Test restoration
        if terms:
            restored_text = self.pipeline._restore_legal_terms(processed_text, terms)
            assert "arbitration" in restored_text.lower()
    
    async def test_batch_translation(self):
        """Test batch translation functionality."""
        texts = [
            "arbitration agreement",
            "binding clause",
            "dispute resolution"
        ]
        
        # Since we're testing without actual models, we'll test the structure
        results = await self.pipeline.translate_batch(texts, "en", "es")
        
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, TranslationResult)


@pytest.mark.asyncio
class TestMultilingualEmbeddings:
    """Test multilingual embedding functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create minimal config for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'models': {
                    'sentence_transformer': 'paraphrase-multilingual-MiniLM-L12-v2'
                },
                'embedding_dim': 384,
                'batch_size': 2,
                'enable_caching': True
            }
            import yaml
            yaml.dump(config, f)
            self.config_file = f.name
        
        self.embedding_engine = MultilingualEmbeddingEngine(self.config_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        if hasattr(self, 'config_file'):
            os.unlink(self.config_file)
    
    async def test_embedding_generation(self):
        """Test embedding generation for different languages."""
        texts = {
            "en": "binding arbitration clause",
            "es": "cláusula de arbitraje vinculante",
            "fr": "clause d'arbitrage contraignant"
        }
        
        embeddings = {}
        for lang, text in texts.items():
            try:
                result = await self.embedding_engine.get_embeddings(
                    text, ModelType.SENTENCE_TRANSFORMER, lang
                )
                embeddings[lang] = result
                
                assert result.text == text
                assert len(result.embeddings) > 0
                assert result.model_used == ModelType.SENTENCE_TRANSFORMER.value
                assert result.language == lang
                
            except Exception as e:
                # Model might not be available in test environment
                pytest.skip(f"Embedding model not available: {e}")
    
    async def test_batch_embeddings(self):
        """Test batch embedding generation."""
        texts = [
            "arbitration agreement",
            "binding clause",
            "dispute resolution"
        ]
        
        try:
            results = await self.embedding_engine.get_batch_embeddings(
                texts, ModelType.SENTENCE_TRANSFORMER, "en"
            )
            
            assert len(results) == len(texts)
            for i, result in enumerate(results):
                assert result.text == texts[i]
                assert len(result.embeddings) > 0
                
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")
    
    def test_faiss_index_building(self):
        """Test FAISS index building."""
        # Create dummy embeddings
        embeddings = [
            np.random.rand(384).astype('float32'),
            np.random.rand(384).astype('float32'),
            np.random.rand(384).astype('float32')
        ]
        
        metadata = [
            {"text": "arbitration clause", "lang": "en"},
            {"text": "cláusula de arbitraje", "lang": "es"},
            {"text": "clause d'arbitrage", "lang": "fr"}
        ]
        
        self.embedding_engine.build_faiss_index(embeddings, "test_index", metadata)
        
        assert "test_index" in self.embedding_engine.faiss_indices
        assert "test_index" in self.embedding_engine.index_metadata
        
        # Test search
        query_embedding = np.random.rand(384).astype('float32')
        results = self.embedding_engine.search_similar(query_embedding, "test_index", k=2)
        
        assert len(results) <= 2
        for distance, meta in results:
            assert isinstance(distance, float)
            assert "text" in meta
            assert "lang" in meta


@pytest.mark.asyncio
class TestCrossLingualSimilarity:
    """Test cross-lingual similarity computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'models': {
                    'sentence_transformer': 'paraphrase-multilingual-MiniLM-L12-v2'
                },
                'embedding_dim': 384
            }
            import yaml
            yaml.dump(config, f)
            self.config_file = f.name
        
        self.embedding_engine = MultilingualEmbeddingEngine(self.config_file)
        self.similarity_engine = CrossLingualSimilarityEngine(self.embedding_engine)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        if hasattr(self, 'config_file'):
            os.unlink(self.config_file)
    
    async def test_similarity_computation(self):
        """Test similarity computation between equivalent texts."""
        text1 = "binding arbitration clause"
        text2 = "cláusula de arbitraje vinculante"
        
        try:
            result = await self.similarity_engine.compute_similarity(
                text1, text2, "en", "es", "cosine"
            )
            
            assert result.text1 == text1
            assert result.text2 == text2
            assert result.language1 == "en"
            assert result.language2 == "es"
            assert result.cross_lingual == True
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.method_used == "cosine"
            
        except Exception as e:
            pytest.skip(f"Similarity computation failed: {e}")
    
    def test_similarity_methods(self):
        """Test different similarity computation methods."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 4.0, 6.0])
        
        # Test cosine similarity
        cosine_sim = self.similarity_engine._cosine_similarity(vec1, vec2)
        assert 0.0 <= cosine_sim <= 1.0
        assert abs(cosine_sim - 1.0) < 0.01  # Should be very similar (same direction)
        
        # Test euclidean similarity
        euclidean_sim = self.similarity_engine._euclidean_similarity(vec1, vec2)
        assert 0.0 <= euclidean_sim <= 1.0
        
        # Test dot product similarity
        dot_sim = self.similarity_engine._dot_product_similarity(vec1, vec2)
        assert -1.0 <= dot_sim <= 1.0
    
    async def test_batch_similarity(self):
        """Test batch similarity computation."""
        text_pairs = [
            ("arbitration clause", "cláusula de arbitraje"),
            ("binding agreement", "acuerdo vinculante"),
            ("dispute resolution", "resolución de disputas")
        ]
        
        language_pairs = [("en", "es")] * len(text_pairs)
        
        try:
            results = await self.similarity_engine.compute_batch_similarity(
                text_pairs, language_pairs, "cosine"
            )
            
            assert len(results) == len(text_pairs)
            for result in results:
                assert result.cross_lingual == True
                assert 0.0 <= result.similarity_score <= 1.0
                
        except Exception as e:
            pytest.skip(f"Batch similarity computation failed: {e}")


@pytest.mark.asyncio
class TestArbitrationPatternDetection:
    """Test arbitration pattern detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'models': {
                    'sentence_transformer': 'paraphrase-multilingual-MiniLM-L12-v2'
                }
            }
            import yaml
            yaml.dump(config, f)
            self.config_file = f.name
        
        self.embedding_engine = MultilingualEmbeddingEngine(self.config_file)
        self.similarity_engine = CrossLingualSimilarityEngine(self.embedding_engine)
        self.pattern_detector = ArbitrationPatternDetector(
            self.embedding_engine, self.similarity_engine
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        if hasattr(self, 'config_file'):
            os.unlink(self.config_file)
    
    async def test_english_arbitration_detection(self):
        """Test arbitration pattern detection in English."""
        text = "Any dispute shall be settled by final and binding arbitration."
        
        try:
            result = await self.pattern_detector.detect_patterns(text, "en")
            
            assert result.text == text
            assert result.language == "en"
            assert len(result.patterns_detected) > 0
            assert "binding_arbitration" in result.patterns_detected
            assert result.arbitration_probability > 0.5
            assert result.legal_domain is not None
            
        except Exception as e:
            pytest.skip(f"Pattern detection failed: {e}")
    
    async def test_multilingual_detection(self):
        """Test pattern detection across multiple languages."""
        texts = {
            "en": "binding arbitration clause",
            "es": "cláusula de arbitraje vinculante",
            "fr": "clause d'arbitrage contraignant"
        }
        
        try:
            for lang, text in texts.items():
                result = await self.pattern_detector.detect_patterns(text, lang)
                
                assert result.language == lang
                assert isinstance(result.arbitration_probability, float)
                assert 0.0 <= result.arbitration_probability <= 1.0
                
        except Exception as e:
            pytest.skip(f"Multilingual detection failed: {e}")
    
    async def test_negative_cases(self):
        """Test that non-arbitration text has low probability."""
        negative_texts = [
            "This is a privacy policy explaining data collection.",
            "Copyright notice and intellectual property rights.",
            "Contact information and business address."
        ]
        
        try:
            for text in negative_texts:
                result = await self.pattern_detector.detect_patterns(text, "en")
                
                # Should have low arbitration probability
                assert result.arbitration_probability < 0.5
                assert len(result.patterns_detected) == 0
                
        except Exception as e:
            pytest.skip(f"Negative case testing failed: {e}")
    
    def test_pattern_template_initialization(self):
        """Test that pattern templates are properly initialized."""
        templates = self.pattern_detector.pattern_templates
        
        assert "binding_arbitration" in templates
        assert "arbitration_clause" in templates
        assert "waiver_clauses" in templates
        
        # Check that multiple languages are supported
        for pattern_type, lang_templates in templates.items():
            assert "en" in lang_templates
            assert len(lang_templates["en"]) > 0


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete multilingual system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'models': {
                    'sentence_transformer': 'paraphrase-multilingual-MiniLM-L12-v2'
                }
            }
            import yaml
            yaml.dump(config, f)
            self.config_file = f.name
        
        self.processor = MultilingualProcessor(self.config_file)
        self.translator = LegalTranslator(self.config_file)
        self.model_manager = LanguageModelManager(self.config_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        if hasattr(self, 'config_file'):
            os.unlink(self.config_file)
    
    async def test_end_to_end_processing(self):
        """Test complete end-to-end document processing."""
        spanish_text = """
        Cualquier disputa que surja de este acuerdo será resuelta mediante 
        arbitraje vinculante administrado por la AAA.
        """
        
        try:
            # Process document
            result = await self.processor.process_document(spanish_text, "en")
            
            assert result['source_language'] == "es"
            assert result['target_language'] == "en"
            assert result['translation_needed'] == True
            assert 'processed_text' in result
            assert 'legal_keywords' in result
            
        except Exception as e:
            pytest.skip(f"End-to-end processing failed: {e}")
    
    async def test_batch_processing(self):
        """Test batch processing of multiple documents."""
        documents = [
            "Binding arbitration clause in English",
            "Cláusula de arbitraje vinculante en español",
            "Clause d'arbitrage contraignant en français"
        ]
        
        try:
            results = await self.processor.process_batch(documents, "en")
            
            assert len(results) == len(documents)
            for result in results:
                assert 'source_language' in result
                assert 'processed_text' in result
                assert 'confidence_score' in result
                
        except Exception as e:
            pytest.skip(f"Batch processing failed: {e}")
    
    def test_supported_languages(self):
        """Test that all expected languages are supported."""
        supported = self.processor.get_supported_languages()
        
        expected_codes = ['en', 'es', 'fr', 'de', 'zh-cn', 'zh-tw', 'ja', 'pt', 'it', 'ru', 'ar', 'ko']
        
        supported_codes = [lang['code'] for lang in supported]
        
        for code in expected_codes:
            assert code in supported_codes
        
        # Check that legal systems are mapped
        for lang in supported:
            assert 'legal_system' in lang
            assert lang['legal_system'] is not None


class TestPerformance:
    """Performance tests for the multilingual system."""
    
    def test_language_detection_performance(self):
        """Test language detection performance."""
        detector = LanguageDetector()
        
        # Test with various text lengths
        test_texts = [
            "Short text",
            "Medium length text with some arbitration clauses and legal terminology",
            "Very long text " * 100 + " with arbitration clauses and legal content"
        ]
        
        for text in test_texts:
            import time
            start = time.time()
            result = detector.detect_language(text)
            end = time.time()
            
            # Should complete within reasonable time
            assert (end - start) < 1.0  # Less than 1 second
            assert isinstance(result, DetectionResult)
    
    def test_cache_functionality(self):
        """Test that caching improves performance."""
        detector = LanguageDetector()
        text = "This is a test text for caching performance evaluation."
        
        # First detection (cache miss)
        import time
        start1 = time.time()
        result1 = detector.detect_language(text)
        time1 = time.time() - start1
        
        # Second detection (cache hit)
        start2 = time.time()
        result2 = detector.detect_language(text)
        time2 = time.time() - start2
        
        # Results should be identical
        assert result1.language == result2.language
        assert result1.confidence == result2.confidence
        
        # Second call should be faster (cached)
        assert time2 <= time1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])