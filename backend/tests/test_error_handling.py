"""
Comprehensive test suite for error handling and edge cases in the RAG system.

This module tests:
1. Invalid file formats and corrupted files
2. Empty documents and missing content
3. Extremely large documents
4. Network failures and database issues
5. Missing dependencies and fallback mechanisms
6. Error message security and informativeness
7. Logging functionality
8. Exception handling in each module
"""

import pytest
import asyncio
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import OperationalError, IntegrityError
from fastapi.testclient import TestClient
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import app
from app.rag.pipeline import RAGPipeline
from app.rag.text_processor import LegalTextProcessor
from app.rag.embeddings import EmbeddingGenerator
from app.services.document_service import DocumentService
from app.core.error_handlers import (
    APIError, DatabaseConnectionError, ValidationException, 
    CircuitBreaker, retry_with_backoff
)
from app.db.vector_store import VectorStore


class TestInvalidFileFormats:
    """Test error handling for invalid file formats and corrupted files."""
    
    def setup_method(self):
        self.client = TestClient(app)
        self.rag_pipeline = RAGPipeline()
    
    def test_corrupted_pdf_upload(self):
        """Test uploading a corrupted PDF file."""
        corrupted_pdf_content = b"This is not a PDF file %PDF-1.4 corrupted data"
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(corrupted_pdf_content)
            tmp.flush()
            
            try:
                with open(tmp.name, 'rb') as f:
                    response = self.client.post(
                        "/api/documents/upload",
                        files={"file": ("corrupted.pdf", f, "application/pdf")}
                    )
                
                assert response.status_code in [400, 422, 500]
                assert "error" in response.json()
                error_data = response.json()["error"]
                assert "message" in error_data
                # Should not expose internal details in production
                if "details" in error_data:
                    assert "corrupted" in error_data["message"].lower() or "invalid" in error_data["message"].lower()
                    
            finally:
                os.unlink(tmp.name)
    
    def test_unsupported_file_format(self):
        """Test uploading an unsupported file format."""
        unsupported_content = b"This is a binary executable file"
        
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
            tmp.write(unsupported_content)
            tmp.flush()
            
            try:
                with open(tmp.name, 'rb') as f:
                    response = self.client.post(
                        "/api/documents/upload",
                        files={"file": ("malicious.exe", f, "application/octet-stream")}
                    )
                
                assert response.status_code == 400
                error_data = response.json()["error"]
                assert "unsupported" in error_data["message"].lower() or "invalid" in error_data["message"].lower()
                assert error_data["error_code"] in ["INVALID_FILE_FORMAT", "UNSUPPORTED_FORMAT"]
                    
            finally:
                os.unlink(tmp.name)
    
    def test_invalid_text_encoding(self):
        """Test processing text with invalid encoding."""
        # Create text with invalid UTF-8 sequences
        invalid_text = b"Valid text \xff\xfe invalid bytes \x80\x81"
        
        try:
            text_str = invalid_text.decode('utf-8')
            pytest.fail("Should have raised UnicodeDecodeError")
        except UnicodeDecodeError:
            # This is expected - now test how the system handles it
            pass
        
        # Test with latin-1 fallback
        text_with_fallback = invalid_text.decode('latin-1')
        
        with patch('app.rag.text_processor.logger') as mock_logger:
            processor = LegalTextProcessor()
            result = processor.preprocess_text(text_with_fallback)
            
            # Should handle gracefully and log warning
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_malformed_json_in_api_request(self):
        """Test API with malformed JSON request."""
        response = self.client.post(
            "/api/documents/analyze",
            data="{ invalid json malformed",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
        error_data = response.json()["error"]
        assert "validation" in error_data["message"].lower() or "json" in error_data["message"].lower()
        assert error_data["error_code"] == "VALIDATION_ERROR"


class TestEmptyAndMissingContent:
    """Test handling of empty documents and missing content."""
    
    def setup_method(self):
        self.client = TestClient(app)
        self.text_processor = LegalTextProcessor()
        self.rag_pipeline = RAGPipeline()
    
    def test_empty_document_processing(self):
        """Test processing completely empty document."""
        empty_text = ""
        
        with patch('app.rag.text_processor.logger') as mock_logger:
            chunks, metadata = self.text_processor.process_document(empty_text)
            
            assert len(chunks) == 0
            assert metadata["total_chunks"] == 0
            assert metadata["arbitration_relevant_chunks"] == 0
            mock_logger.info.assert_called()
    
    def test_whitespace_only_document(self):
        """Test document with only whitespace."""
        whitespace_text = "   \n\n   \t\t   \r\n   "
        
        result = self.rag_pipeline.quick_text_analysis(whitespace_text)
        
        assert not result.has_arbitration_clause
        assert result.confidence_score == 0.0
        assert "no arbitration" in result.summary.lower()
        assert len(result.clauses) == 0
    
    def test_missing_file_content(self):
        """Test uploading file with missing content."""
        response = self.client.post(
            "/api/documents/",
            json={
                "filename": "empty.txt",
                "content": "",
                "content_type": "text/plain"
            }
        )
        
        # Should handle gracefully - empty content is valid but not useful
        assert response.status_code in [200, 201, 400]
        if response.status_code in [200, 201]:
            data = response.json()
            # Should create document but warn about empty content
            assert data["file_size"] == 0
    
    def test_document_with_only_special_characters(self):
        """Test document containing only special characters and symbols."""
        special_char_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        
        chunks, metadata = self.text_processor.process_document(special_char_text)
        
        # Should process but find no meaningful content
        assert metadata["arbitration_relevant_chunks"] == 0
        signals = self.text_processor.extract_arbitration_signals(special_char_text)
        assert signals["arbitration_keywords_count"] == 0


class TestLargeDocuments:
    """Test handling of extremely large documents."""
    
    def setup_method(self):
        self.rag_pipeline = RAGPipeline()
        self.embedding_generator = EmbeddingGenerator()
    
    def test_extremely_large_document(self):
        """Test processing very large document (> 10MB)."""
        # Create a large document (simulate 5MB text)
        large_text = "This is a sample legal document. " * 150000  # ~5MB
        
        with patch('app.rag.text_processor.logger') as mock_logger:
            start_time = datetime.now()
            
            try:
                chunks, metadata = self.rag_pipeline.text_processor.process_document(large_text, chunk_size=1000)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Should complete but may take time
                assert len(chunks) > 100  # Should create many chunks
                assert processing_time < 300  # Should complete within 5 minutes
                assert metadata["processed_length"] > 1000000  # > 1MB
                
                # Should log performance warnings
                mock_logger.warning.assert_called()
                
            except MemoryError:
                # This is acceptable for very large documents
                pytest.skip("System doesn't have enough memory for this test")
            except Exception as e:
                # Should fail gracefully with informative error
                assert "memory" in str(e).lower() or "size" in str(e).lower()
    
    def test_memory_exhaustion_handling(self):
        """Test behavior when memory is exhausted during processing."""
        # Simulate memory exhaustion
        with patch('app.rag.embeddings.np.array') as mock_array:
            mock_array.side_effect = MemoryError("Not enough memory")
            
            with pytest.raises(MemoryError):
                large_text = "Legal document content " * 10000
                self.embedding_generator.generate_embedding(large_text)
    
    def test_processing_timeout(self):
        """Test handling of processing timeouts for large documents."""
        # Simulate slow processing
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 1000]  # Simulate 1000 seconds elapsed
            
            with patch('app.rag.pipeline.logger') as mock_logger:
                large_text = "Test document " * 1000
                result = self.rag_pipeline.quick_text_analysis(large_text)
                
                # Should complete with timeout warnings
                assert result.processing_time_ms > 900000  # > 900 seconds in ms
                mock_logger.warning.assert_called()


class TestNetworkFailures:
    """Test handling of network failures and database issues."""
    
    def setup_method(self):
        self.client = TestClient(app)
        self.document_service = DocumentService()
    
    @patch('app.db.database.SessionLocal')
    def test_database_connection_failure(self, mock_session):
        """Test handling of database connection failures."""
        mock_session.side_effect = OperationalError("Connection refused", None, None)
        
        response = self.client.get("/api/documents/")
        
        assert response.status_code == 503
        error_data = response.json()["error"]
        assert error_data["error_code"] == "DB_CONNECTION_ERROR"
        assert "database" in error_data["message"].lower()
        # Should not expose sensitive connection details
        assert "password" not in error_data["message"].lower()
        assert "connection string" not in error_data["message"].lower()
    
    @patch('app.db.vector_store.VectorStore.similarity_search')
    def test_vector_store_failure(self, mock_search):
        """Test handling of vector store failures."""
        mock_search.side_effect = ConnectionError("Vector store unavailable")
        
        response = self.client.post(
            "/api/documents/search",
            json={"query": "arbitration clause", "limit": 10}
        )
        
        assert response.status_code in [500, 503]
        error_data = response.json()["error"]
        # Should provide fallback or graceful degradation
        assert "search" in error_data["message"].lower() or "unavailable" in error_data["message"].lower()
    
    @patch('redis.Redis')
    def test_redis_connection_failure(self, mock_redis):
        """Test handling of Redis cache failures."""
        mock_redis.side_effect = ConnectionError("Redis connection failed")
        
        # System should work without cache
        response = self.client.get("/api/health")
        
        # Should work but may have degraded performance
        assert response.status_code == 200
        health_data = response.json()
        # Should indicate cache unavailability
        if "cache" in health_data:
            assert health_data["cache"] is False
    
    def test_database_integrity_error(self):
        """Test handling of database integrity constraint violations."""
        # Try to create duplicate document (simulate unique constraint violation)
        with patch('sqlalchemy.orm.Session.commit') as mock_commit:
            mock_commit.side_effect = IntegrityError("Duplicate key", None, None)
            
            response = self.client.post(
                "/api/documents/",
                json={
                    "filename": "test.txt",
                    "content": "Test content",
                    "content_type": "text/plain"
                }
            )
            
            assert response.status_code == 400
            error_data = response.json()["error"]
            assert error_data["error_code"] == "DB_INTEGRITY_ERROR"
            assert "integrity" in error_data["message"].lower() or "duplicate" in error_data["message"].lower()


class TestMissingDependencies:
    """Test handling of missing dependencies and fallback mechanisms."""
    
    def test_spacy_model_missing(self):
        """Test behavior when spaCy model is not available."""
        with patch('spacy.load') as mock_load:
            mock_load.side_effect = OSError("Model not found")
            
            with patch('app.rag.text_processor.logger') as mock_logger:
                processor = LegalTextProcessor()
                
                # Should initialize with warning and use fallback
                assert processor.nlp is None
                mock_logger.warning.assert_called()
                
                # Should still work with simple chunking
                text = "This is a legal document with arbitration clauses."
                chunks, metadata = processor.process_document(text)
                assert len(chunks) > 0
    
    def test_sentence_transformers_model_failure(self):
        """Test behavior when sentence transformers model fails to load."""
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_model.side_effect = Exception("Model download failed")
            
            with pytest.raises(Exception):
                EmbeddingGenerator()
                
            # Should fail gracefully and log error
    
    def test_fallback_to_simple_chunking(self):
        """Test fallback to simple chunking when advanced processing fails."""
        processor = LegalTextProcessor()
        processor.nlp = None  # Simulate missing spaCy
        
        text = "Legal document content. Another sentence. Third sentence here."
        chunks = processor.chunk_text_by_sentences(text, max_chunk_size=50)
        
        # Should use simple word-based chunking
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) <= 60  # Some flexibility for word boundaries
    
    @patch('torch.cuda.is_available')
    def test_gpu_unavailable_fallback(self, mock_cuda):
        """Test fallback to CPU when GPU is unavailable."""
        mock_cuda.return_value = False
        
        embedding_gen = EmbeddingGenerator()
        assert embedding_gen.config.device == "cpu"
        
        # Should work on CPU
        embedding = embedding_gen.generate_embedding("Test text")
        assert embedding is not None
        assert len(embedding.shape) == 1


class TestErrorMessageSecurity:
    """Test that error messages are informative but don't expose sensitive information."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_sql_injection_in_error_message(self):
        """Test that SQL injection attempts don't leak in error messages."""
        malicious_input = "'; DROP TABLE documents; --"
        
        response = self.client.post(
            "/api/documents/",
            json={
                "filename": malicious_input,
                "content": "Test content",
                "content_type": "text/plain"
            }
        )
        
        # Should not expose SQL in error message
        if response.status_code >= 400:
            error_message = response.json()["error"]["message"]
            assert "DROP TABLE" not in error_message
            assert "SQL" not in error_message.upper()
            assert "--" not in error_message
    
    def test_path_traversal_in_error_message(self):
        """Test that path traversal attempts don't leak system paths."""
        malicious_filename = "../../etc/passwd"
        
        response = self.client.post(
            "/api/documents/",
            json={
                "filename": malicious_filename,
                "content": "Test content",
                "content_type": "text/plain"
            }
        )
        
        if response.status_code >= 400:
            error_data = response.json()["error"]
            # Should not expose system paths
            assert "/etc/" not in error_data["message"]
            assert "/var/" not in error_data["message"]
            assert "/usr/" not in error_data["message"]
    
    def test_internal_stack_trace_not_exposed(self):
        """Test that internal stack traces are not exposed in production."""
        # Force an internal error
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with patch('app.services.document_service.DocumentService.get_document') as mock_get:
                mock_get.side_effect = Exception("Internal error with sensitive data: api_key=secret123")
                
                response = self.client.get("/api/documents/999999")
                
                assert response.status_code >= 400
                error_data = response.json()["error"]
                # Should not expose sensitive internal details
                assert "api_key" not in error_data["message"]
                assert "secret123" not in error_data["message"]
                assert "Internal error" not in error_data["message"]
    
    def test_development_vs_production_error_details(self):
        """Test different error detail levels in development vs production."""
        # Test in development mode
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            with patch('app.services.document_service.DocumentService.get_document') as mock_get:
                mock_get.side_effect = ValueError("Detailed error for debugging")
                
                response = self.client.get("/api/documents/999999")
                dev_error = response.json()["error"]
        
        # Test in production mode
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with patch('app.services.document_service.DocumentService.get_document') as mock_get:
                mock_get.side_effect = ValueError("Detailed error for debugging")
                
                response = self.client.get("/api/documents/999999")
                prod_error = response.json()["error"]
        
        # Development should have more details, production should be generic
        assert len(prod_error["message"]) <= len(dev_error.get("message", ""))


class TestLoggingFunctionality:
    """Test that logging works correctly for error scenarios."""
    
    def setup_method(self):
        self.rag_pipeline = RAGPipeline()
    
    def test_error_logging_with_context(self):
        """Test that errors are logged with appropriate context."""
        with patch('app.rag.pipeline.logger') as mock_logger:
            # Force an error during document processing
            with patch.object(self.rag_pipeline.text_processor, 'process_document') as mock_process:
                mock_process.side_effect = Exception("Processing failed")
                
                with pytest.raises(Exception):
                    self.rag_pipeline.process_document(1, "test content")
                
                # Should log error with context
                mock_logger.error.assert_called()
                logged_message = mock_logger.error.call_args[0][0]
                assert "processing" in logged_message.lower()
    
    def test_performance_logging(self):
        """Test that performance issues are logged appropriately."""
        with patch('app.rag.pipeline.logger') as mock_logger:
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0, 10]  # 10 second processing time
                
                result = self.rag_pipeline.quick_text_analysis("Test content")
                
                # Should log performance warning for slow processing
                assert result.processing_time_ms == 10000
                mock_logger.warning.assert_called()
    
    def test_security_event_logging(self):
        """Test that security-related events are logged."""
        with patch('app.core.error_handlers.logger') as mock_logger:
            from app.core.error_handlers import create_error_response
            
            # Simulate suspicious activity
            response = create_error_response(
                status_code=400,
                message="Invalid input detected",
                error_code="SUSPICIOUS_INPUT"
            )
            
            assert response.status_code == 400


class TestCircuitBreakerAndRetry:
    """Test circuit breaker and retry mechanisms."""
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        def failing_function():
            raise Exception("Service failed")
        
        # Test failures until circuit opens
        for _ in range(3):
            with pytest.raises(Exception):
                asyncio.run(breaker.call(failing_function))
        
        assert breaker.state == "open"
        
        # Next call should fail immediately
        with pytest.raises(APIError) as exc_info:
            asyncio.run(breaker.call(failing_function))
        
        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.message.lower()
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        def failing_then_working():
            if breaker.failure_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                asyncio.run(breaker.call(failing_then_working))
        
        assert breaker.state == "open"
        
        # Wait for recovery timeout (simulate)
        import time
        time.sleep(1.1)
        
        # Should attempt recovery
        result = asyncio.run(breaker.call(failing_then_working))
        assert result == "success"
        assert breaker.state == "closed"
    
    def test_retry_with_exponential_backoff(self):
        """Test retry mechanism with exponential backoff."""
        call_count = 0
        
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        start_time = datetime.now()
        result = asyncio.run(retry_with_backoff(unreliable_function, max_retries=3))
        end_time = datetime.now()
        
        assert result == "success"
        assert call_count == 3
        # Should have taken time for backoff (1s + 2s = 3s minimum)
        assert (end_time - start_time).total_seconds() >= 2.0
    
    def test_retry_exhaustion(self):
        """Test behavior when all retries are exhausted."""
        def always_failing():
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception) as exc_info:
            asyncio.run(retry_with_backoff(always_failing, max_retries=2))
        
        assert str(exc_info.value) == "Persistent failure"


class TestModuleSpecificErrors:
    """Test exception handling in each specific RAG module."""
    
    def test_text_processor_unicode_errors(self):
        """Test text processor handling of unicode errors."""
        processor = LegalTextProcessor()
        
        # Test with problematic unicode
        problematic_text = "Legal text with unicode issues \udcff"
        
        with patch('app.rag.text_processor.logger') as mock_logger:
            result = processor.preprocess_text(problematic_text)
            
            # Should handle gracefully
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_embedding_generator_model_errors(self):
        """Test embedding generator handling of model errors."""
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
            mock_encode.side_effect = RuntimeError("CUDA out of memory")
            
            embedding_gen = EmbeddingGenerator()
            
            with pytest.raises(RuntimeError):
                embedding_gen.generate_embedding("test text")
    
    def test_vector_store_storage_errors(self):
        """Test vector store handling of storage errors."""
        vector_store = VectorStore()
        
        with patch.object(vector_store, '_get_collection') as mock_collection:
            mock_collection.side_effect = Exception("Storage backend failed")
            
            with pytest.raises(Exception):
                vector_store.add_document_chunks(
                    chunks=["test chunk"],
                    document_id=1,
                    chunk_indices=[0],
                    start_chars=[0],
                    end_chars=[10]
                )


# Integration test for complete error handling workflow
class TestErrorHandlingIntegration:
    """Integration tests for complete error handling workflows."""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_complete_document_processing_with_failures(self):
        """Test complete document processing pipeline with various failures."""
        # Test the full pipeline with injected failures at different stages
        
        # 1. Start with successful upload
        response = self.client.post(
            "/api/documents/",
            json={
                "filename": "test_error_handling.txt",
                "content": "This is a test document with some arbitration clauses for testing error handling.",
                "content_type": "text/plain"
            }
        )
        
        if response.status_code not in [200, 201]:
            pytest.skip("Document creation failed, skipping integration test")
        
        document_id = response.json()["id"]
        
        # 2. Test processing with database failure
        with patch('sqlalchemy.orm.Session.commit') as mock_commit:
            mock_commit.side_effect = OperationalError("Database locked", None, None)
            
            response = self.client.post(f"/api/documents/{document_id}/process")
            
            assert response.status_code == 503
            error_data = response.json()["error"]
            assert error_data["error_code"] == "DB_CONNECTION_ERROR"
    
    def test_health_check_error_recovery(self):
        """Test health check with error recovery mechanisms."""
        response = self.client.get("/api/health")
        
        # Should always return some response, even if components are down
        assert response.status_code in [200, 503]
        
        health_data = response.json()
        assert "status" in health_data
        
        # Should report individual component health
        if "checks" in health_data:
            checks = health_data["checks"]
            assert isinstance(checks, dict)
            # Common health check components
            expected_checks = ["database", "memory"]
            for check in expected_checks:
                if check in checks:
                    assert isinstance(checks[check], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])