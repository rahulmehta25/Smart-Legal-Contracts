"""
Performance benchmarks for arbitration clause detection system.

This module provides comprehensive performance testing including:
- Processing speed benchmarks
- Memory usage profiling
- Throughput testing
- Latency measurements
- Scalability testing
"""

import pytest
import time
import psutil
import json
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from unittest.mock import Mock
import numpy as np
from memory_profiler import profile


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    @pytest.fixture
    def performance_data(self):
        """Generate various sizes of test data for benchmarking."""
        return {
            "small": "Arbitration clause: " + "test " * 50,  # ~250 chars
            "medium": "Arbitration clause: " + "test " * 500,  # ~2.5K chars  
            "large": "Arbitration clause: " + "test " * 2000,  # ~10K chars
            "xlarge": "Arbitration clause: " + "test " * 10000,  # ~50K chars
            "xxlarge": "Arbitration clause: " + "test " * 50000,  # ~250K chars
        }
    
    @pytest.fixture
    def mock_detector(self):
        """Mock arbitration detector with realistic processing times."""
        detector = Mock()
        
        def mock_detect(text: str):
            # Simulate processing time based on text length
            base_time = 0.1  # 100ms base processing time
            time_per_char = 0.000001  # 1μs per character
            processing_time = base_time + (len(text) * time_per_char)
            
            time.sleep(processing_time)
            
            return {
                "has_arbitration": "arbitration" in text.lower(),
                "confidence": 0.85,
                "processing_time": processing_time
            }
        
        detector.detect_arbitration_clause = mock_detect
        return detector

    @pytest.mark.benchmark(group="text_processing_speed")
    def test_small_document_processing_speed(self, benchmark, mock_detector, performance_data):
        """Benchmark processing speed for small documents."""
        result = benchmark(mock_detector.detect_arbitration_clause, performance_data["small"])
        
        # Assertions for small document performance
        assert result["processing_time"] < 0.5  # Should process in under 500ms
        
        # Benchmark should complete quickly
        assert benchmark.stats.mean < 0.5

    @pytest.mark.benchmark(group="text_processing_speed")
    def test_medium_document_processing_speed(self, benchmark, mock_detector, performance_data):
        """Benchmark processing speed for medium documents."""
        result = benchmark(mock_detector.detect_arbitration_clause, performance_data["medium"])
        
        assert result["processing_time"] < 1.0  # Should process in under 1s
        assert benchmark.stats.mean < 1.0

    @pytest.mark.benchmark(group="text_processing_speed")
    def test_large_document_processing_speed(self, benchmark, mock_detector, performance_data):
        """Benchmark processing speed for large documents."""
        result = benchmark(mock_detector.detect_arbitration_clause, performance_data["large"])
        
        assert result["processing_time"] < 2.0  # Should process in under 2s
        assert benchmark.stats.mean < 2.0

    @pytest.mark.benchmark(group="text_processing_speed")
    def test_xlarge_document_processing_speed(self, benchmark, mock_detector, performance_data):
        """Benchmark processing speed for extra large documents."""
        result = benchmark(mock_detector.detect_arbitration_clause, performance_data["xlarge"])
        
        assert result["processing_time"] < 5.0  # Should process in under 5s
        assert benchmark.stats.mean < 5.0

    @pytest.mark.benchmark(group="memory_usage")
    def test_memory_usage_small_documents(self, mock_detector, performance_data):
        """Test memory usage for small document processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process multiple small documents
        for _ in range(100):
            mock_detector.detect_arbitration_clause(performance_data["small"])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 50  # Should not increase memory by more than 50MB

    @pytest.mark.benchmark(group="memory_usage")
    def test_memory_usage_large_documents(self, mock_detector, performance_data):
        """Test memory usage for large document processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process large documents
        for _ in range(10):
            mock_detector.detect_arbitration_clause(performance_data["large"])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 100  # Should not increase memory by more than 100MB

    @pytest.mark.benchmark(group="concurrent_processing")
    def test_concurrent_processing_throughput(self, mock_detector, performance_data):
        """Test throughput with concurrent processing."""
        start_time = time.time()
        
        def process_document():
            return mock_detector.detect_arbitration_clause(performance_data["medium"])
        
        # Process documents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_document) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        throughput = len(results) / total_time  # documents per second
        
        assert throughput > 5  # Should process at least 5 documents per second
        assert all(result["has_arbitration"] for result in results)

    @pytest.mark.benchmark(group="batch_processing")
    def test_batch_processing_performance(self, benchmark, mock_detector, performance_data):
        """Benchmark batch processing performance."""
        documents = [performance_data["small"]] * 20
        
        def batch_process():
            return [mock_detector.detect_arbitration_clause(doc) for doc in documents]
        
        results = benchmark(batch_process)
        
        assert len(results) == 20
        assert benchmark.stats.mean < 5.0  # Should process batch in under 5s

    @pytest.mark.asyncio
    async def test_async_processing_performance(self, performance_data):
        """Test asynchronous processing performance."""
        
        async def async_detect(text: str):
            # Simulate async processing
            await asyncio.sleep(0.1)
            return {
                "has_arbitration": "arbitration" in text.lower(),
                "confidence": 0.85
            }
        
        start_time = time.time()
        
        # Process documents asynchronously
        tasks = [async_detect(performance_data["medium"]) for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        assert len(results) == 20
        assert total_time < 1.0  # Async processing should be much faster
        assert all(result["has_arbitration"] for result in results)

    def test_memory_leak_detection(self, mock_detector, performance_data):
        """Test for memory leaks during extended processing."""
        process = psutil.Process()
        memory_readings = []
        
        # Take memory readings during processing
        for i in range(100):
            mock_detector.detect_arbitration_clause(performance_data["medium"])
            
            if i % 10 == 0:  # Record memory every 10 iterations
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_readings.append(memory_mb)
        
        # Check for consistent memory increase (potential leak)
        memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        assert memory_trend < 1  # Memory increase should be less than 1MB per 10 iterations

    @pytest.mark.benchmark(group="scalability")
    def test_load_scalability(self, benchmark, mock_detector, performance_data):
        """Test system scalability under increasing load."""
        
        def increasing_load_test():
            results = []
            load_sizes = [1, 5, 10, 20, 50]
            
            for load_size in load_sizes:
                start_time = time.time()
                
                # Process documents in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=load_size) as executor:
                    futures = [
                        executor.submit(mock_detector.detect_arbitration_clause, performance_data["medium"])
                        for _ in range(load_size)
                    ]
                    batch_results = [future.result() for future in futures]
                
                batch_time = time.time() - start_time
                throughput = load_size / batch_time
                
                results.append({
                    "load_size": load_size,
                    "batch_time": batch_time,
                    "throughput": throughput
                })
            
            return results
        
        results = benchmark(increasing_load_test)
        
        # Verify scalability characteristics
        throughputs = [r["throughput"] for r in results]
        
        # Throughput should generally increase or stay stable with load
        # (until system limits are reached)
        assert max(throughputs) > min(throughputs) * 0.5

    @pytest.mark.benchmark(group="response_time_percentiles")
    def test_response_time_distribution(self, mock_detector, performance_data):
        """Test response time distribution and percentiles."""
        response_times = []
        
        # Collect response times from multiple requests
        for _ in range(100):
            start_time = time.time()
            mock_detector.detect_arbitration_clause(performance_data["medium"])
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
        
        # Calculate percentiles
        p50 = np.percentile(response_times, 50)
        p95 = np.percentile(response_times, 95)
        p99 = np.percentile(response_times, 99)
        
        # Performance assertions
        assert p50 < 500  # 50th percentile should be under 500ms
        assert p95 < 1000  # 95th percentile should be under 1s
        assert p99 < 2000  # 99th percentile should be under 2s
        
        # Log percentiles for monitoring
        print(f"Response time percentiles - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")

    def test_cpu_usage_monitoring(self, mock_detector, performance_data):
        """Monitor CPU usage during processing."""
        # Start CPU monitoring
        cpu_percent_start = psutil.cpu_percent(interval=None)
        
        start_time = time.time()
        
        # Process documents while monitoring CPU
        for _ in range(50):
            mock_detector.detect_arbitration_clause(performance_data["medium"])
        
        processing_time = time.time() - start_time
        cpu_percent_end = psutil.cpu_percent(interval=None)
        
        # Calculate CPU efficiency
        cpu_time_ratio = processing_time / (processing_time + 1)  # Rough estimation
        
        # Assertions
        assert processing_time < 30  # Should complete in reasonable time
        print(f"CPU usage: {cpu_percent_end:.1f}%, Processing time: {processing_time:.2f}s")

    @pytest.mark.benchmark(group="cache_performance")
    def test_caching_performance_impact(self, mock_detector, performance_data):
        """Test performance impact of caching mechanisms."""
        
        # Simulate cache miss (first request)
        start_time = time.time()
        result1 = mock_detector.detect_arbitration_clause(performance_data["medium"])
        cache_miss_time = time.time() - start_time
        
        # Simulate cache hit (repeated request)
        # In real implementation, this would use actual caching
        start_time = time.time()
        result2 = mock_detector.detect_arbitration_clause(performance_data["medium"])
        cache_hit_time = time.time() - start_time
        
        # Cache should provide some performance benefit
        # (In this mock, times will be similar, but real cache would be faster)
        assert cache_hit_time <= cache_miss_time * 1.1  # Allow for some variance
        assert result1 == result2  # Results should be identical

    def test_garbage_collection_impact(self, mock_detector, performance_data):
        """Test impact of garbage collection on performance."""
        import gc
        
        # Disable garbage collection temporarily
        gc.disable()
        
        start_time = time.time()
        for _ in range(20):
            mock_detector.detect_arbitration_clause(performance_data["large"])
        gc_disabled_time = time.time() - start_time
        
        # Re-enable garbage collection
        gc.enable()
        gc.collect()  # Force garbage collection
        
        start_time = time.time()
        for _ in range(20):
            mock_detector.detect_arbitration_clause(performance_data["large"])
        gc_enabled_time = time.time() - start_time
        
        # Performance should not degrade significantly with GC enabled
        performance_ratio = gc_enabled_time / gc_disabled_time
        assert performance_ratio < 1.5  # GC overhead should be less than 50%

    def test_warm_up_performance(self, mock_detector, performance_data):
        """Test system warm-up time and performance stabilization."""
        response_times = []
        
        # Measure response times for initial requests (warm-up period)
        for i in range(50):
            start_time = time.time()
            mock_detector.detect_arbitration_clause(performance_data["medium"])
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        # Check if performance stabilizes after warm-up
        initial_times = response_times[:10]  # First 10 requests
        stable_times = response_times[-10:]  # Last 10 requests
        
        initial_avg = np.mean(initial_times)
        stable_avg = np.mean(stable_times)
        
        # Performance should improve or stabilize after warm-up
        improvement_ratio = initial_avg / stable_avg
        assert improvement_ratio >= 0.8  # Should not degrade significantly


class TestResourceUtilization:
    """Test resource utilization patterns."""
    
    def test_memory_efficiency_by_document_size(self, mock_detector):
        """Test memory efficiency across different document sizes."""
        sizes = ["small", "medium", "large", "xlarge"]
        memory_usage = {}
        
        for size in sizes:
            # Generate test data
            if size == "small":
                text = "arbitration " * 100
            elif size == "medium":
                text = "arbitration " * 1000
            elif size == "large":
                text = "arbitration " * 5000
            else:  # xlarge
                text = "arbitration " * 20000
            
            # Measure memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            mock_detector.detect_arbitration_clause(text)
            
            peak_memory = process.memory_info().rss
            memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
            
            memory_usage[size] = {
                "text_size_chars": len(text),
                "memory_increase_mb": memory_increase,
                "efficiency_ratio": len(text) / max(memory_increase, 0.1)  # chars per MB
            }
        
        # Log memory efficiency
        for size, stats in memory_usage.items():
            print(f"{size}: {stats['text_size_chars']} chars, "
                  f"{stats['memory_increase_mb']:.2f} MB, "
                  f"efficiency: {stats['efficiency_ratio']:.0f} chars/MB")
        
        # Efficiency should not degrade dramatically with size
        small_efficiency = memory_usage["small"]["efficiency_ratio"]
        large_efficiency = memory_usage["large"]["efficiency_ratio"]
        efficiency_retention = large_efficiency / small_efficiency
        
        assert efficiency_retention > 0.5  # Should retain at least 50% efficiency

    def test_disk_io_performance(self, tmp_path, mock_detector):
        """Test disk I/O performance impact."""
        
        # Create test files
        test_files = []
        for i in range(10):
            file_path = tmp_path / f"test_doc_{i}.txt"
            file_path.write_text(f"Test document {i} with arbitration clause content.")
            test_files.append(file_path)
        
        # Measure file processing performance
        start_time = time.time()
        
        for file_path in test_files:
            content = file_path.read_text()
            mock_detector.detect_arbitration_clause(content)
        
        total_time = time.time() - start_time
        files_per_second = len(test_files) / total_time
        
        assert files_per_second > 5  # Should process at least 5 files per second
        assert total_time < 5  # Should complete within 5 seconds


class PerformanceProfiler:
    """Utility class for detailed performance profiling."""
    
    @staticmethod
    def profile_function_calls(func, *args, **kwargs):
        """Profile function calls with detailed timing."""
        import cProfile
        import pstats
        from io import StringIO
        
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        
        # Get profiling results
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        profile_output = s.getvalue()
        
        return result, profile_output
    
    @staticmethod
    def generate_performance_report(benchmark_results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("ARBITRATION DETECTION SYSTEM - PERFORMANCE REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Summary statistics
        if "processing_times" in benchmark_results:
            times = benchmark_results["processing_times"]
            report.append(f"Processing Time Statistics:")
            report.append(f"  Mean: {np.mean(times):.3f}s")
            report.append(f"  Median: {np.median(times):.3f}s")
            report.append(f"  95th percentile: {np.percentile(times, 95):.3f}s")
            report.append(f"  99th percentile: {np.percentile(times, 99):.3f}s")
            report.append("")
        
        # Throughput metrics
        if "throughput" in benchmark_results:
            report.append(f"Throughput: {benchmark_results['throughput']:.2f} documents/second")
            report.append("")
        
        # Memory usage
        if "memory_usage" in benchmark_results:
            report.append(f"Memory Usage:")
            report.append(f"  Peak: {benchmark_results['memory_usage']['peak']:.2f} MB")
            report.append(f"  Average: {benchmark_results['memory_usage']['average']:.2f} MB")
            report.append("")
        
        # Performance recommendations
        report.append("Performance Recommendations:")
        if np.mean(times) > 2.0:
            report.append("  - Consider optimizing text processing algorithms")
        if benchmark_results.get("memory_usage", {}).get("peak", 0) > 500:
            report.append("  - Monitor memory usage for potential optimizations")
        if benchmark_results.get("throughput", 0) < 10:
            report.append("  - Consider implementing parallel processing")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([
        __file__,
        "--benchmark-only",
        "--benchmark-json=performance_benchmark_results.json",
        "--benchmark-save=arbitration_detection_benchmarks"
    ])