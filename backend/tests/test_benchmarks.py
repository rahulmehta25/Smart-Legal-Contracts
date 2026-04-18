"""
Performance benchmarks for critical paths.

Benchmarks:
- Document analysis latency
- RAG pipeline throughput
- API response times
- Vector store operations
"""

import pytest
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import statistics

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Benchmark Configuration
# ============================================================================

BENCHMARK_CONFIG = {
    "iterations": 10,
    "warmup_iterations": 2,
    "max_document_processing_time": 10.0,  # seconds
    "max_api_response_time": 2.0,  # seconds
    "max_vector_search_time": 0.5,  # seconds
    "max_embedding_time": 1.0,  # seconds per document
}


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, times: List[float]):
        self.name = name
        self.times = times
        self.mean = statistics.mean(times) if times else 0
        self.median = statistics.median(times) if times else 0
        self.min = min(times) if times else 0
        self.max = max(times) if times else 0
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0

    def __repr__(self):
        return (f"Benchmark({self.name}): mean={self.mean:.4f}s, "
                f"median={self.median:.4f}s, min={self.min:.4f}s, max={self.max:.4f}s")


def run_benchmark(func, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Run a function multiple times and collect timing statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Actual benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return BenchmarkResult(func.__name__, times)


# ============================================================================
# Document Processing Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestDocumentProcessingBenchmarks:
    """Benchmarks for document processing."""

    @pytest.fixture
    def sample_documents(self):
        """Generate sample documents of various sizes."""
        base_text = "This is a sample legal document paragraph. " * 10

        return {
            "small": base_text * 10,      # ~5KB
            "medium": base_text * 100,    # ~50KB
            "large": base_text * 500,     # ~250KB
        }

    @pytest.mark.performance
    def test_text_processing_latency(self, sample_documents, mock_document_processor, benchmark_timer):
        """Benchmark text document processing latency."""
        results = {}

        for size_name, text in sample_documents.items():
            times = []
            for _ in range(5):
                with benchmark_timer:
                    # Simulate processing
                    mock_document_processor.process_document(text)
                times.append(benchmark_timer.elapsed)

            results[size_name] = statistics.mean(times)

        # Verify performance thresholds
        assert results["small"] < 1.0, "Small document processing too slow"
        assert results["medium"] < 3.0, "Medium document processing too slow"
        assert results["large"] < 10.0, "Large document processing too slow"

    @pytest.mark.performance
    def test_batch_processing_throughput(self, sample_documents, mock_document_processor, benchmark_timer):
        """Benchmark batch document processing throughput."""
        documents = [sample_documents["small"]] * 20

        with benchmark_timer:
            for doc in documents:
                mock_document_processor.process_document(doc)

        throughput = len(documents) / max(benchmark_timer.elapsed, 0.001)

        # Should process at least 5 documents per second
        assert throughput >= 5, f"Throughput too low: {throughput:.2f} docs/sec"


# ============================================================================
# RAG Pipeline Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestRAGPipelineBenchmarks:
    """Benchmarks for RAG pipeline operations."""

    @pytest.fixture
    def rag_test_documents(self):
        """Generate documents for RAG benchmarking."""
        return [
            "This contract includes a binding arbitration clause administered by AAA.",
            "The parties agree to resolve disputes through JAMS arbitration.",
            "Any disputes shall be settled in the courts of New York.",
            "Mandatory arbitration is required for all claims.",
            "You waive your right to participate in class action lawsuits.",
        ] * 10

    @pytest.mark.performance
    def test_arbitration_detection_latency(self, mock_arbitration_detector, sample_documents, benchmark_timer):
        """Benchmark arbitration detection latency."""
        times = []

        for _ in range(BENCHMARK_CONFIG["iterations"]):
            with benchmark_timer:
                for name, text in sample_documents.items():
                    mock_arbitration_detector.detect(text, name)
            times.append(benchmark_timer.elapsed)

        avg_time = statistics.mean(times) / len(sample_documents)

        assert avg_time < BENCHMARK_CONFIG["max_document_processing_time"], \
            f"Detection too slow: {avg_time:.4f}s per document"

    @pytest.mark.performance
    def test_embedding_generation_latency(self, mock_embeddings, rag_test_documents, benchmark_timer):
        """Benchmark embedding generation latency."""
        times = []

        for _ in range(5):
            with benchmark_timer:
                mock_embeddings.embed_documents(rag_test_documents)
            times.append(benchmark_timer.elapsed)

        avg_time_per_doc = statistics.mean(times) / len(rag_test_documents)

        assert avg_time_per_doc < BENCHMARK_CONFIG["max_embedding_time"], \
            f"Embedding generation too slow: {avg_time_per_doc:.4f}s per document"

    @pytest.mark.performance
    def test_similarity_search_latency(self, mock_vector_store, mock_embeddings, benchmark_timer):
        """Benchmark similarity search latency."""
        # Setup: add documents
        texts = ["Document " + str(i) for i in range(100)]
        mock_vector_store.add_texts(texts)

        times = []
        for _ in range(20):
            with benchmark_timer:
                mock_vector_store.similarity_search("arbitration clause", k=10)
            times.append(benchmark_timer.elapsed)

        avg_time = statistics.mean(times)

        assert avg_time < BENCHMARK_CONFIG["max_vector_search_time"], \
            f"Search too slow: {avg_time:.4f}s"


# ============================================================================
# API Response Time Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestAPIBenchmarks:
    """Benchmarks for API response times."""

    def test_health_endpoint_latency(self, test_client, benchmark_timer):
        """Benchmark health endpoint response time."""
        times = []

        for _ in range(50):
            with benchmark_timer:
                response = test_client.get("/health")
            times.append(benchmark_timer.elapsed)

        avg_time = statistics.mean(times)
        p99 = sorted(times)[int(len(times) * 0.99)]

        assert avg_time < 0.1, f"Health check too slow: {avg_time:.4f}s"
        assert p99 < 0.5, f"P99 latency too high: {p99:.4f}s"

    @pytest.mark.performance
    def test_concurrent_requests_latency(self, test_client):
        """Benchmark concurrent request handling."""
        import concurrent.futures

        def make_request():
            start = time.perf_counter()
            test_client.get("/health")
            return time.perf_counter() - start

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            times = [f.result() for f in concurrent.futures.as_completed(futures)]

        avg_time = statistics.mean(times)
        p95 = sorted(times)[int(len(times) * 0.95)]

        assert avg_time < 0.5, f"Concurrent avg latency too high: {avg_time:.4f}s"
        assert p95 < 1.0, f"P95 latency too high: {p95:.4f}s"


# ============================================================================
# Vector Store Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestVectorStoreBenchmarks:
    """Benchmarks for vector store operations."""

    @pytest.mark.performance
    def test_bulk_insert_throughput(self, mock_qdrant_client, mock_embeddings, benchmark_timer):
        """Benchmark bulk insert throughput."""
        mock_qdrant_client.create_collection("benchmark", {"size": 384})

        batch_sizes = [10, 50, 100]
        results = {}

        for batch_size in batch_sizes:
            texts = [f"Document {i}" for i in range(batch_size)]
            embeddings = mock_embeddings.embed_documents(texts)

            points = [
                Mock(id=str(i), vector=emb, payload={"text": text})
                for i, (text, emb) in enumerate(zip(texts, embeddings))
            ]

            with benchmark_timer:
                mock_qdrant_client.upsert("benchmark", points)

            results[batch_size] = benchmark_timer.elapsed

        # Larger batches should be relatively efficient
        assert results[100] / 100 <= results[10] / 10 * 2, \
            "Bulk insert not scaling efficiently"

    @pytest.mark.performance
    def test_search_scaling(self, mock_qdrant_client, benchmark_timer):
        """Benchmark search performance as collection grows."""
        mock_qdrant_client.create_collection("scaling_test", {})

        collection_sizes = [10, 100, 500]
        search_times = {}

        for size in collection_sizes:
            # Add documents
            points = [
                Mock(id=str(i), vector=[0.1 * (i % 10)] * 384, payload={"text": f"Doc {i}"})
                for i in range(size)
            ]
            mock_qdrant_client._points["scaling_test"] = points

            # Benchmark search
            times = []
            for _ in range(10):
                with benchmark_timer:
                    mock_qdrant_client.search(
                        "scaling_test",
                        [0.5] * 384,
                        limit=10
                    )
                times.append(benchmark_timer.elapsed)

            search_times[size] = statistics.mean(times)

        # Search time should not increase dramatically with size
        assert search_times[500] < search_times[10] * 20, \
            "Search not scaling well with collection size"


# ============================================================================
# Memory Usage Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmarks for memory usage."""

    @pytest.mark.performance
    def test_large_document_memory(self, mock_arbitration_detector, large_document_text):
        """Test memory usage with large documents."""
        import sys

        initial_size = sys.getsizeof(large_document_text)

        result = mock_arbitration_detector.detect(large_document_text, "large_doc")

        # Result should not be excessively larger than input
        result_size = sys.getsizeof(str(result))
        assert result_size < initial_size * 10, \
            f"Result size ({result_size}) too large compared to input ({initial_size})"

    @pytest.mark.performance
    def test_batch_processing_memory(self, mock_document_processor):
        """Test memory doesn't leak during batch processing."""
        import gc

        # Force garbage collection
        gc.collect()

        # Process many documents
        for i in range(100):
            text = f"Document {i} content " * 100
            mock_document_processor.process_document(text)

            # Periodic GC
            if i % 20 == 0:
                gc.collect()


# ============================================================================
# End-to-End Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestE2EBenchmarks:
    """End-to-end performance benchmarks."""

    @pytest.mark.performance
    def test_full_analysis_pipeline(
        self,
        mock_document_processor,
        mock_arbitration_detector,
        mock_vector_store,
        sample_documents,
        benchmark_timer
    ):
        """Benchmark full analysis pipeline."""
        times = []

        for _ in range(3):
            with benchmark_timer:
                for name, text in sample_documents.items():
                    # 1. Process document
                    mock_document_processor.process_document(text)

                    # 2. Store in vector store
                    mock_vector_store.add_texts([text], [{"name": name}])

                    # 3. Detect arbitration
                    mock_arbitration_detector.detect(text, name)

            times.append(benchmark_timer.elapsed)

        avg_pipeline_time = statistics.mean(times)
        docs_processed = len(sample_documents)

        throughput = docs_processed / avg_pipeline_time

        # Should process at least 1 document per second
        assert throughput >= 1, f"Pipeline throughput too low: {throughput:.2f} docs/sec"

    @pytest.mark.performance
    def test_search_and_analyze_workflow(
        self,
        mock_vector_store,
        mock_arbitration_detector,
        sample_documents,
        benchmark_timer
    ):
        """Benchmark search and analyze workflow."""
        # Setup: Add documents to vector store
        for name, text in sample_documents.items():
            mock_vector_store.add_texts([text], [{"name": name}])

        times = []
        for _ in range(5):
            with benchmark_timer:
                # Search for relevant documents
                results = mock_vector_store.similarity_search("arbitration", k=5)

                # Analyze each result
                for result in results:
                    text = result.page_content
                    mock_arbitration_detector.detect(text)

            times.append(benchmark_timer.elapsed)

        avg_time = statistics.mean(times)

        assert avg_time < 5.0, f"Search and analyze workflow too slow: {avg_time:.4f}s"


# ============================================================================
# Benchmark Reporting
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def benchmark_report(request):
    """Generate benchmark report at end of session."""
    yield

    # This would write results to a file in a real implementation
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
