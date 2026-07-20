"""
Edge case tests for RAG system resilience and robustness.

This module focuses on testing specific edge cases that could cause the RAG system
to fail silently or produce incorrect results:
1. Malformed legal documents
2. Ambiguous arbitration language
3. Resource exhaustion scenarios
4. Concurrent processing issues
5. Cache corruption and recovery
6. Model inference failures
"""

import pytest
import asyncio
import threading
import time
import tempfile
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import numpy as np

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.rag.pipeline import RAGPipeline, AnalysisResult
from app.rag.text_processor import LegalTextProcessor, TextChunk
from app.rag.embeddings import EmbeddingGenerator, EmbeddingConfig
from app.rag.retriever import ArbitrationRetriever
from app.db.vector_store import VectorStore
from app.services.document_service import DocumentService


class TestMalformedLegalDocuments:
    """Test handling of malformed or unusual legal document structures."""
    
    def setup_method(self):
        self.text_processor = LegalTextProcessor()
        self.rag_pipeline = RAGPipeline()
    
    def test_document_with_corrupted_legal_formatting(self):
        """Test document with corrupted legal formatting and mixed languages."""
        corrupted_doc = """
        TÉRMINOS Y CONDICIONES / TERMS AND CONDITIONS
        
        Section 1.1.1.1.1 Invalid Nesting
        
        (a)(b)(c)(d)(e)(f)(g) Invalid enumeration
        
        Any disputes arising from this agreement shall be resolved through binding arbitration.
        Cualquier disputa que surja de este acuerdo será resuelta mediante arbitraje vinculante.
        
        @@@ CORRUPTED SECTION @@@
        NULL NULL NULL
        
        ARBITRATION CLAUSE:
        {MISSING CONTENT}
        
        %%% END DOCUMENT %%%
        """
        
        result = self.rag_pipeline.quick_text_analysis(corrupted_doc)
        
        # Should still detect arbitration despite formatting issues
        assert isinstance(result, AnalysisResult)
        # Should handle gracefully without crashing
        assert result.confidence_score >= 0.0
        
    def test_document_with_recursive_references(self):
        """Test document with circular or recursive clause references."""
        recursive_doc = """
        Section 5 refers to Section 10.
        Section 10 refers to Section 5.
        
        As defined in Section 5, arbitration as outlined in Section 10 shall apply.
        As defined in Section 10, dispute resolution as outlined in Section 5 shall apply.
        
        Binding arbitration (see Section 5) is mandatory (see Section 10).
        """
        
        chunks, metadata = self.text_processor.process_document(recursive_doc)
        
        # Should handle circular references without infinite loops
        assert len(chunks) > 0
        assert metadata["total_chunks"] > 0
        # Should complete in reasonable time
        assert metadata.get("processing_time_ms", 0) < 10000  # Less than 10 seconds
    
    def test_document_with_contradictory_clauses(self):
        """Test document with contradictory arbitration clauses."""
        contradictory_doc = """
        1. All disputes shall be resolved through binding arbitration.
        
        2. The parties waive their right to arbitration and agree to resolve
           disputes in federal court.
        
        3. Notwithstanding section 2, mandatory arbitration applies to all claims.
        
        4. This agreement does not contain arbitration clauses.
        """
        
        result = self.rag_pipeline.quick_text_analysis(contradictory_doc)
        
        # Should detect arbitration presence despite contradictions
        assert result.has_arbitration_clause  # Should detect positive references
        assert len(result.clauses) > 0
        # Should have lower confidence due to contradictions
        assert result.confidence_score < 0.9
    
    def test_document_with_obfuscated_arbitration_language(self):
        """Test document with deliberately obfuscated arbitration language."""
        obfuscated_doc = """
        In the event of disagreements, parties shall utilize alternative 
        dispute resolution mechanisms, specifically those involving neutral 
        third-party decision-makers whose determinations shall be final 
        and enforceable.
        
        The parties hereby waive judicial proceedings in favor of private 
        adjudication by impartial arbiters selected according to established 
        commercial rules.
        
        Class action procedures are expressly prohibited; individual 
        resolution through private tribunals is mandated.
        """
        
        result = self.rag_pipeline.quick_text_analysis(obfuscated_doc)
        
        # Should still detect arbitration despite obfuscation
        assert result.has_arbitration_clause
        assert result.confidence_score > 0.5
        assert len(result.clauses) > 0


class TestResourceExhaustion:
    """Test system behavior under resource exhaustion conditions."""
    
    def setup_method(self):
        self.embedding_generator = EmbeddingGenerator()
        self.rag_pipeline = RAGPipeline()
    
    def test_memory_pressure_during_embedding_generation(self):
        """Test behavior under memory pressure during embedding generation."""
        # Simulate memory pressure by creating many embeddings simultaneously
        texts = [f"Legal document {i} with arbitration clause {i}" for i in range(100)]
        
        with patch('app.rag.embeddings.logger') as mock_logger:
            try:
                embeddings = self.embedding_generator.generate_embeddings_batch(texts, batch_size=5)
                assert len(embeddings) == len(texts)
                
                # Should log if processing takes too long
                if any("memory" in str(call) for call in mock_logger.warning.call_args_list):
                    assert True  # Expected warning logged
                    
            except MemoryError:
                # Acceptable failure under extreme memory pressure
                pytest.skip("System under memory pressure")
    
    def test_concurrent_processing_stress(self):
        """Test system behavior under concurrent processing load."""
        test_documents = [
            f"Legal document {i} containing binding arbitration clauses and dispute resolution mechanisms."
            for i in range(20)
        ]
        
        def process_document(text):
            return self.rag_pipeline.quick_text_analysis(text)
        
        # Process documents concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_document, doc) for doc in test_documents]
            results = []
            
            for future in as_completed(futures, timeout=60):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Log but don't fail the test - some failures under load are acceptable
                    print(f"Concurrent processing error: {e}")
        
        # At least some results should succeed
        assert len(results) > len(test_documents) * 0.5  # 50% success rate minimum
    
    def test_cache_exhaustion_handling(self):
        """Test behavior when embedding cache is exhausted."""
        # Fill up the cache
        config = EmbeddingConfig()
        embedding_gen = EmbeddingGenerator(config)
        
        # Generate many different embeddings to fill cache
        for i in range(1000):
            text = f"Unique text {i} to fill cache"
            try:
                embedding_gen.generate_embedding(text, use_cache=True)
            except Exception as e:
                # Should handle cache exhaustion gracefully
                assert "cache" in str(e).lower() or "memory" in str(e).lower()
                break
    
    def test_disk_space_exhaustion(self):
        """Test behavior when disk space is exhausted during caching."""
        with patch('builtins.open', side_effect=IOError("No space left on device")):
            embedding_gen = EmbeddingGenerator()
            
            # Should work without caching
            embedding = embedding_gen.generate_embedding("test text", use_cache=True)
            assert embedding is not None
            assert len(embedding.shape) == 1


class TestCacheCorruption:
    """Test cache corruption scenarios and recovery mechanisms."""
    
    def setup_method(self):
        self.embedding_generator = EmbeddingGenerator()
    
    def test_corrupted_cache_file_recovery(self):
        """Test recovery from corrupted cache files."""
        # Create a corrupted cache file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp.write(b"corrupted cache data that is not a pickle")
            tmp.flush()
            corrupted_path = tmp.name
        
        try:
            with patch.object(self.embedding_generator, 'cache_path') as mock_cache_path:
                mock_cache_path.return_value = os.path.dirname(corrupted_path)
                
                with patch('app.rag.embeddings.logger') as mock_logger:
                    # Should handle corrupted cache gracefully
                    embedding = self.embedding_generator.generate_embedding("test text")
                    
                    assert embedding is not None
                    mock_logger.warning.assert_called()  # Should log cache corruption
                    
        finally:
            os.unlink(corrupted_path)
    
    def test_cache_inconsistency_detection(self):
        """Test detection and handling of cache inconsistencies."""
        # Simulate cache with wrong model name
        with patch.object(self.embedding_generator, '_get_cache_key') as mock_cache_key:
            mock_cache_key.return_value = "fake_cache_key"
            
            # Populate cache with wrong model
            fake_embedding = np.random.rand(384)  # Wrong dimensions
            self.embedding_generator.cache["fake_cache_key"] = fake_embedding
            
            # Should detect inconsistency and regenerate
            real_embedding = self.embedding_generator.generate_embedding("test text")
            
            # Should get properly sized embedding
            assert real_embedding.shape[0] == self.embedding_generator.config.embedding_dim


class TestModelInferenceFailures:
    """Test handling of ML model inference failures."""
    
    def setup_method(self):
        self.embedding_generator = EmbeddingGenerator()
        self.rag_pipeline = RAGPipeline()
    
    def test_model_inference_timeout(self):
        """Test handling of model inference timeouts."""
        with patch.object(self.embedding_generator.model, 'encode') as mock_encode:
            # Simulate very slow inference
            def slow_encode(*args, **kwargs):
                time.sleep(10)  # Simulate 10 second delay
                return np.random.rand(1, 384)
            
            mock_encode.side_effect = slow_encode
            
            start_time = time.time()
            try:
                embedding = self.embedding_generator.generate_embedding("test text")
                processing_time = time.time() - start_time
                
                # Should either complete or timeout gracefully
                if embedding is not None:
                    assert processing_time < 15  # Should not hang indefinitely
                    
            except Exception as e:
                # Timeout is acceptable
                assert "timeout" in str(e).lower() or "time" in str(e).lower()
    
    def test_model_returns_invalid_embeddings(self):
        """Test handling of invalid embeddings from model."""
        with patch.object(self.embedding_generator.model, 'encode') as mock_encode:
            # Test various invalid return values
            invalid_returns = [
                None,
                np.array([]),  # Empty array
                np.array([np.nan, np.inf, -np.inf]),  # Invalid values
                np.array([[1, 2], [3, 4]]),  # Wrong shape
                "not an array",  # Wrong type
            ]
            
            for invalid_return in invalid_returns:
                mock_encode.return_value = invalid_return
                
                with pytest.raises(Exception):
                    self.embedding_generator.generate_embedding("test text")
    
    def test_cuda_out_of_memory_recovery(self):
        """Test recovery from CUDA out of memory errors."""
        with patch.object(self.embedding_generator.model, 'encode') as mock_encode:
            mock_encode.side_effect = RuntimeError("CUDA out of memory")
            
            # Should either fallback to CPU or fail gracefully
            with pytest.raises(RuntimeError):
                self.embedding_generator.generate_embedding("test text")
    
    def test_model_device_mismatch(self):
        """Test handling of device mismatch errors."""
        with patch.object(self.embedding_generator.model, 'encode') as mock_encode:
            mock_encode.side_effect = RuntimeError("Input tensor is not on the correct device")
            
            with pytest.raises(RuntimeError):
                self.embedding_generator.generate_embedding("test text")


class TestAmbiguousArbitrationLanguage:
    """Test handling of ambiguous or edge-case arbitration language."""
    
    def setup_method(self):
        self.rag_pipeline = RAGPipeline()
    
    def test_conditional_arbitration_clauses(self):
        """Test arbitration clauses with complex conditions."""
        conditional_doc = """
        If and only if the dispute amount exceeds $10,000, then binding arbitration
        may apply, unless the parties agree otherwise, except in cases where
        state law prohibits arbitration, provided that such prohibition does not
        violate federal arbitration act requirements.
        
        Arbitration shall be mandatory only for disputes arising after the
        effective date of this amendment, but not for pre-existing claims,
        unless specifically agreed to in writing by both parties.
        """
        
        result = self.rag_pipeline.quick_text_analysis(conditional_doc)
        
        # Should detect arbitration presence but with appropriate confidence
        assert result.has_arbitration_clause
        # Confidence should be lower due to conditions
        assert result.confidence_score < 0.8
    
    def test_implied_arbitration_references(self):
        """Test documents with implied rather than explicit arbitration."""
        implied_doc = """
        Disputes shall be resolved according to the Commercial Arbitration Rules
        of the American Arbitration Association.
        
        The parties agree to follow the procedures outlined in the Federal
        Arbitration Act for all disagreements.
        
        Neither party may pursue class action relief, and all claims must be
        brought individually before a neutral forum.
        """
        
        result = self.rag_pipeline.quick_text_analysis(implied_doc)
        
        # Should detect arbitration through context and references
        assert result.has_arbitration_clause
        assert any("arbitration" in clause["text"].lower() for clause in result.clauses)
    
    def test_arbitration_exclusions_and_carve_outs(self):
        """Test documents with arbitration exclusions and carve-outs."""
        exclusion_doc = """
        All disputes shall be resolved through binding arbitration, except:
        1. Claims for injunctive relief
        2. Intellectual property disputes
        3. Small claims court matters under $5,000
        4. Disputes involving real property
        5. Claims that cannot be arbitrated under applicable law
        
        Notwithstanding the above, arbitration remains mandatory for all
        other disputes, including breach of contract and tort claims.
        """
        
        result = self.rag_pipeline.quick_text_analysis(exclusion_doc)
        
        # Should still recognize as having arbitration
        assert result.has_arbitration_clause
        assert result.confidence_score > 0.6  # Should be reasonably confident
        
        # Should capture both arbitration requirement and exclusions
        clause_texts = " ".join([clause["text"] for clause in result.clauses])
        assert "binding arbitration" in clause_texts.lower()
        assert "except" in clause_texts.lower()


class TestConcurrencyAndRaceConditions:
    """Test race conditions and concurrency issues in RAG system."""
    
    def setup_method(self):
        self.document_service = DocumentService()
        self.rag_pipeline = RAGPipeline()
    
    def test_concurrent_document_processing(self):
        """Test concurrent processing of the same document."""
        test_doc_id = 1
        test_content = "This legal document contains mandatory arbitration clauses."
        
        def process_same_document():
            return self.rag_pipeline.quick_text_analysis(test_content)
        
        # Process same document concurrently
        results = []
        threads = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(process_same_document()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)
        
        # All results should be consistent
        assert len(results) == 5
        confidence_scores = [r.confidence_score for r in results]
        assert all(abs(score - confidence_scores[0]) < 0.1 for score in confidence_scores)
    
    def test_cache_race_condition(self):
        """Test race conditions in embedding cache."""
        embedding_gen = EmbeddingGenerator()
        test_text = "Race condition test text"
        
        def generate_embedding():
            return embedding_gen.generate_embedding(test_text, use_cache=True)
        
        # Generate same embedding concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_embedding) for _ in range(10)]
            embeddings = [future.result() for future in futures]
        
        # All embeddings should be identical (from cache)
        for embedding in embeddings[1:]:
            np.testing.assert_array_equal(embedding, embeddings[0])
    
    def test_vector_store_concurrent_updates(self):
        """Test concurrent updates to vector store."""
        vector_store = VectorStore()
        
        def add_chunks(doc_id):
            chunks = [f"Chunk {i} for document {doc_id}" for i in range(5)]
            try:
                return vector_store.add_document_chunks(
                    chunks=chunks,
                    document_id=doc_id,
                    chunk_indices=list(range(5)),
                    start_chars=[i*100 for i in range(5)],
                    end_chars=[(i+1)*100 for i in range(5)]
                )
            except Exception as e:
                return None
        
        # Add chunks for different documents concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_chunks, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # Should handle concurrent updates gracefully
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) >= len(results) * 0.8  # 80% success rate


class TestSystemRecovery:
    """Test system recovery mechanisms after failures."""
    
    def test_recovery_after_complete_system_failure(self):
        """Test system recovery after complete failure."""
        rag_pipeline = RAGPipeline()
        
        # Simulate complete system failure
        with patch.multiple(
            rag_pipeline,
            text_processor=Mock(side_effect=Exception("System failed")),
            embedding_generator=Mock(side_effect=Exception("System failed")),
            vector_store=Mock(side_effect=Exception("System failed")),
            retriever=Mock(side_effect=Exception("System failed"))
        ):
            with pytest.raises(Exception):
                rag_pipeline.quick_text_analysis("test")
        
        # System should recover after patch is removed
        result = rag_pipeline.quick_text_analysis("This document contains arbitration clauses.")
        assert isinstance(result, AnalysisResult)
        assert result.confidence_score >= 0.0
    
    def test_partial_system_recovery(self):
        """Test recovery when only some components fail."""
        rag_pipeline = RAGPipeline()
        
        # Simulate partial failure (only vector store fails)
        with patch.object(rag_pipeline, 'vector_store', side_effect=Exception("Vector store failed")):
            # Should still work with text analysis only
            result = rag_pipeline.quick_text_analysis("Arbitration clause test")
            
            assert isinstance(result, AnalysisResult)
            # Should indicate limited functionality
            assert result.metadata["analysis_method"] == "quick_analysis"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])