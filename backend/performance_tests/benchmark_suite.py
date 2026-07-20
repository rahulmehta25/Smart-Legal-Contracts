#!/usr/bin/env python3
"""
Comprehensive Performance Testing Suite for RAG Legal Analysis System

This script benchmarks:
1. Document processing speed (PDF, TXT, DOCX)
2. Legal-BERT inference time
3. Pattern matching performance 
4. Vector similarity search speed
5. Database query performance
6. Redis caching effectiveness
7. End-to-end pipeline throughput
"""

import os
import sys
import time
import json
import psutil
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
import asyncio
from pathlib import Path
import tempfile
import random
import string

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag.pipeline import RAGPipeline
from app.rag.arbitration_detector import ArbitrationDetector
from app.services.pdf_service import PDFService
from app.db.vector_store import VectorStore
from app.core.config import get_settings

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    test_name: str
    document_size: str
    num_documents: int
    total_time_seconds: float
    avg_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    p50_time_seconds: float
    p95_time_seconds: float
    p99_time_seconds: float
    throughput_docs_per_sec: float
    memory_used_mb: float
    cpu_percent: float
    timestamp: str


class DocumentGenerator:
    """Generate test documents of various sizes"""
    
    @staticmethod
    def generate_text_document(size: str = "medium") -> str:
        """Generate a text document with arbitration content"""
        
        # Base arbitration clauses
        arbitration_clauses = [
            "Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration.",
            "The parties agree to submit to mandatory arbitration all disputes arising under this contract.",
            "By accepting these terms, you waive your right to trial by jury and agree to individual arbitration.",
            "All claims must be resolved through final and binding arbitration instead of in court.",
            "This arbitration agreement survives termination of the agreement between parties.",
        ]
        
        # Filler legal text
        legal_filler = [
            "The parties hereby acknowledge and agree to the terms and conditions set forth herein.",
            "This agreement shall be governed by the laws of the State of California.",
            "Neither party shall be liable for indirect, incidental, or consequential damages.",
            "All intellectual property rights remain with their respective owners.",
            "This agreement constitutes the entire understanding between the parties.",
        ]
        
        # Size configurations (approximate character counts)
        sizes = {
            "small": 1000,      # ~1KB - 1-2 pages
            "medium": 10000,    # ~10KB - 10-15 pages
            "large": 50000,     # ~50KB - 50-60 pages
            "xlarge": 100000,   # ~100KB - 100+ pages
        }
        
        target_size = sizes.get(size, sizes["medium"])
        
        # Build document
        document_parts = []
        current_size = 0
        
        # Add title
        document_parts.append("TERMS OF SERVICE AGREEMENT\n\n")
        current_size += len(document_parts[-1])
        
        # Add arbitration clauses
        for clause in arbitration_clauses:
            if current_size < target_size * 0.3:  # 30% arbitration content
                document_parts.append(f"\n{clause}\n")
                current_size += len(document_parts[-1])
        
        # Fill with legal text
        while current_size < target_size:
            filler = random.choice(legal_filler)
            document_parts.append(f"\n{filler}")
            current_size += len(filler)
            
            # Occasionally add arbitration keywords
            if random.random() < 0.1:
                keyword = random.choice(["arbitration", "dispute resolution", "binding", "waiver"])
                document_parts.append(f" The {keyword} provisions shall apply.")
                current_size += len(document_parts[-1])
        
        return "".join(document_parts)[:target_size]
    
    @staticmethod
    def generate_pdf_document(size: str = "medium", output_path: str = None) -> str:
        """Generate a PDF document for testing"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            print("Warning: reportlab not installed. Using text file instead.")
            text_content = DocumentGenerator.generate_text_document(size)
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(text_content)
            return output_path or text_content
        
        # Generate text content
        text_content = DocumentGenerator.generate_text_document(size)
        
        # Create PDF
        if not output_path:
            output_path = tempfile.mktemp(suffix='.pdf')
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Split content into paragraphs
        paragraphs = text_content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para, styles['Normal']))
        
        doc.build(story)
        return output_path


class PerformanceBenchmark:
    """Main performance benchmarking class"""
    
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.detector = ArbitrationDetector()
        self.vector_store = VectorStore()
        self.results: List[BenchmarkResult] = []
        
    def benchmark_text_processing(self, sizes: List[str] = None) -> List[BenchmarkResult]:
        """Benchmark text document processing speed"""
        if sizes is None:
            sizes = ["small", "medium", "large"]
        
        results = []
        
        for size in sizes:
            print(f"\nBenchmarking text processing - {size} documents...")
            
            # Generate test documents
            documents = [DocumentGenerator.generate_text_document(size) for _ in range(10)]
            
            # Track metrics
            times = []
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percents = []
            
            for doc in documents:
                # Monitor CPU
                cpu_start = psutil.cpu_percent(interval=None)
                
                # Time the processing
                start_time = time.time()
                result = self.detector.detect_arbitration_clause(doc)
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Record CPU
                cpu_end = psutil.cpu_percent(interval=None)
                cpu_percents.append(cpu_end)
            
            # Calculate statistics
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            benchmark_result = BenchmarkResult(
                test_name="text_processing",
                document_size=size,
                num_documents=len(documents),
                total_time_seconds=sum(times),
                avg_time_seconds=np.mean(times),
                min_time_seconds=min(times),
                max_time_seconds=max(times),
                p50_time_seconds=np.percentile(times, 50),
                p95_time_seconds=np.percentile(times, 95),
                p99_time_seconds=np.percentile(times, 99),
                throughput_docs_per_sec=len(documents) / sum(times),
                memory_used_mb=memory_end - memory_start,
                cpu_percent=np.mean(cpu_percents),
                timestamp=datetime.now().isoformat()
            )
            
            results.append(benchmark_result)
            self.results.append(benchmark_result)
            
            # Print summary
            print(f"  Processed {len(documents)} {size} documents")
            print(f"  Average time: {benchmark_result.avg_time_seconds:.3f}s")
            print(f"  Throughput: {benchmark_result.throughput_docs_per_sec:.2f} docs/sec")
            print(f"  P95 latency: {benchmark_result.p95_time_seconds:.3f}s")
        
        return results
    
    def benchmark_pdf_processing(self, sizes: List[str] = None) -> List[BenchmarkResult]:
        """Benchmark PDF document processing speed"""
        if sizes is None:
            sizes = ["small", "medium"]
        
        results = []
        pdf_service = PDFService()
        
        for size in sizes:
            print(f"\nBenchmarking PDF processing - {size} documents...")
            
            # Generate test PDFs
            pdf_files = []
            temp_dir = tempfile.mkdtemp()
            
            for i in range(5):  # Fewer PDFs due to slower processing
                pdf_path = os.path.join(temp_dir, f"test_{i}.pdf")
                DocumentGenerator.generate_pdf_document(size, pdf_path)
                pdf_files.append(pdf_path)
            
            # Track metrics
            times = []
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            for pdf_path in pdf_files:
                start_time = time.time()
                
                # Process PDF
                with open(pdf_path, 'rb') as f:
                    text = pdf_service.extract_text_from_pdf(f)
                    result = self.detector.detect_arbitration_clause(text)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            # Calculate statistics
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            benchmark_result = BenchmarkResult(
                test_name="pdf_processing",
                document_size=size,
                num_documents=len(pdf_files),
                total_time_seconds=sum(times),
                avg_time_seconds=np.mean(times),
                min_time_seconds=min(times),
                max_time_seconds=max(times),
                p50_time_seconds=np.percentile(times, 50),
                p95_time_seconds=np.percentile(times, 95),
                p99_time_seconds=np.percentile(times, 99),
                throughput_docs_per_sec=len(pdf_files) / sum(times),
                memory_used_mb=memory_end - memory_start,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                timestamp=datetime.now().isoformat()
            )
            
            results.append(benchmark_result)
            self.results.append(benchmark_result)
            
            # Cleanup
            for pdf_path in pdf_files:
                os.remove(pdf_path)
            os.rmdir(temp_dir)
            
            # Print summary
            print(f"  Processed {len(pdf_files)} {size} PDFs")
            print(f"  Average time: {benchmark_result.avg_time_seconds:.3f}s")
            print(f"  Throughput: {benchmark_result.throughput_docs_per_sec:.2f} docs/sec")
        
        return results
    
    def benchmark_vector_search(self, num_queries: int = 100) -> BenchmarkResult:
        """Benchmark vector similarity search performance"""
        print(f"\nBenchmarking vector search with {num_queries} queries...")
        
        # First, populate vector store with test data
        print("  Populating vector store with test embeddings...")
        for i in range(100):
            text = f"Test document {i} with arbitration clause content."
            chunks = [text[j:j+500] for j in range(0, len(text), 500)]
            self.vector_store.add_document_chunks(
                chunks=chunks,
                document_id=i,
                chunk_indices=list(range(len(chunks))),
                start_chars=[j for j in range(0, len(text), 500)],
                end_chars=[min(j+500, len(text)) for j in range(0, len(text), 500)]
            )
        
        # Benchmark searches
        times = []
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        for _ in range(num_queries):
            query = "arbitration clause dispute resolution"
            
            start_time = time.time()
            results = self.vector_store.search_similar_chunks(
                query_text=query,
                document_id=None,
                top_k=10
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        benchmark_result = BenchmarkResult(
            test_name="vector_search",
            document_size="N/A",
            num_documents=num_queries,
            total_time_seconds=sum(times),
            avg_time_seconds=np.mean(times),
            min_time_seconds=min(times),
            max_time_seconds=max(times),
            p50_time_seconds=np.percentile(times, 50),
            p95_time_seconds=np.percentile(times, 95),
            p99_time_seconds=np.percentile(times, 99),
            throughput_docs_per_sec=num_queries / sum(times),
            memory_used_mb=memory_end - memory_start,
            cpu_percent=psutil.cpu_percent(interval=0.1),
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(benchmark_result)
        
        print(f"  Executed {num_queries} vector searches")
        print(f"  Average time: {benchmark_result.avg_time_seconds*1000:.2f}ms")
        print(f"  Throughput: {benchmark_result.throughput_docs_per_sec:.2f} queries/sec")
        
        return benchmark_result
    
    def benchmark_pattern_matching(self, sizes: List[str] = None) -> List[BenchmarkResult]:
        """Benchmark pattern matching performance on large texts"""
        if sizes is None:
            sizes = ["small", "medium", "large"]
        
        results = []
        
        from app.models.pattern import PatternMatcher
        pattern_matcher = PatternMatcher()
        
        for size in sizes:
            print(f"\nBenchmarking pattern matching - {size} documents...")
            
            # Generate test documents
            documents = [DocumentGenerator.generate_text_document(size) for _ in range(10)]
            
            times = []
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            for doc in documents:
                start_time = time.time()
                matches = pattern_matcher.find_arbitration_patterns(doc)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            benchmark_result = BenchmarkResult(
                test_name="pattern_matching",
                document_size=size,
                num_documents=len(documents),
                total_time_seconds=sum(times),
                avg_time_seconds=np.mean(times),
                min_time_seconds=min(times),
                max_time_seconds=max(times),
                p50_time_seconds=np.percentile(times, 50),
                p95_time_seconds=np.percentile(times, 95),
                p99_time_seconds=np.percentile(times, 99),
                throughput_docs_per_sec=len(documents) / sum(times),
                memory_used_mb=memory_end - memory_start,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                timestamp=datetime.now().isoformat()
            )
            
            results.append(benchmark_result)
            self.results.append(benchmark_result)
            
            print(f"  Processed {len(documents)} documents")
            print(f"  Average time: {benchmark_result.avg_time_seconds:.3f}s")
            print(f"  Throughput: {benchmark_result.throughput_docs_per_sec:.2f} docs/sec")
        
        return results
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = None) -> List[BenchmarkResult]:
        """Benchmark batch processing performance"""
        if batch_sizes is None:
            batch_sizes = [1, 5, 10, 20]
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking batch processing - batch size {batch_size}...")
            
            # Generate documents
            documents = [
                DocumentGenerator.generate_text_document("medium") 
                for _ in range(batch_size)
            ]
            
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Time batch processing
            start_time = time.time()
            
            # Process batch
            batch_results = []
            for doc in documents:
                result = self.detector.detect_arbitration_clause(doc)
                batch_results.append(result)
            
            total_time = time.time() - start_time
            
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            benchmark_result = BenchmarkResult(
                test_name=f"batch_processing_size_{batch_size}",
                document_size="medium",
                num_documents=batch_size,
                total_time_seconds=total_time,
                avg_time_seconds=total_time / batch_size,
                min_time_seconds=total_time / batch_size,
                max_time_seconds=total_time / batch_size,
                p50_time_seconds=total_time / batch_size,
                p95_time_seconds=total_time / batch_size,
                p99_time_seconds=total_time / batch_size,
                throughput_docs_per_sec=batch_size / total_time,
                memory_used_mb=memory_end - memory_start,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                timestamp=datetime.now().isoformat()
            )
            
            results.append(benchmark_result)
            self.results.append(benchmark_result)
            
            print(f"  Processed batch of {batch_size} documents")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {benchmark_result.throughput_docs_per_sec:.2f} docs/sec")
        
        return results
    
    def benchmark_concurrent_processing(self, num_workers: List[int] = None) -> List[BenchmarkResult]:
        """Benchmark concurrent processing with different worker counts"""
        if num_workers is None:
            num_workers = [1, 2, 4, 8]
        
        results = []
        
        for workers in num_workers:
            print(f"\nBenchmarking concurrent processing - {workers} workers...")
            
            # Generate documents
            documents = [
                DocumentGenerator.generate_text_document("medium") 
                for _ in range(20)
            ]
            
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            
            # Process concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.detector.detect_arbitration_clause, doc)
                    for doc in documents
                ]
                results_concurrent = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024
            
            benchmark_result = BenchmarkResult(
                test_name=f"concurrent_processing_{workers}_workers",
                document_size="medium",
                num_documents=len(documents),
                total_time_seconds=total_time,
                avg_time_seconds=total_time / len(documents),
                min_time_seconds=total_time / len(documents),
                max_time_seconds=total_time / len(documents),
                p50_time_seconds=total_time / len(documents),
                p95_time_seconds=total_time / len(documents),
                p99_time_seconds=total_time / len(documents),
                throughput_docs_per_sec=len(documents) / total_time,
                memory_used_mb=memory_end - memory_start,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                timestamp=datetime.now().isoformat()
            )
            
            results.append(benchmark_result)
            self.results.append(benchmark_result)
            
            print(f"  Processed {len(documents)} documents with {workers} workers")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {benchmark_result.throughput_docs_per_sec:.2f} docs/sec")
        
        return results
    
    def benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark Redis cache performance if available"""
        print("\nBenchmarking cache performance...")
        
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
        except:
            print("  Redis not available - skipping cache benchmarks")
            return None
        
        # Test documents
        documents = [DocumentGenerator.generate_text_document("small") for _ in range(20)]
        
        times_uncached = []
        times_cached = []
        
        # First pass - uncached
        for doc in documents:
            start_time = time.time()
            result = self.detector.detect_arbitration_clause(doc)
            elapsed = time.time() - start_time
            times_uncached.append(elapsed)
            
            # Store in cache
            doc_hash = hash(doc)
            r.set(f"doc:{doc_hash}", json.dumps(result), ex=300)
        
        # Second pass - cached
        for doc in documents:
            doc_hash = hash(doc)
            
            start_time = time.time()
            cached_result = r.get(f"doc:{doc_hash}")
            if cached_result:
                result = json.loads(cached_result)
            else:
                result = self.detector.detect_arbitration_clause(doc)
            elapsed = time.time() - start_time
            times_cached.append(elapsed)
        
        # Calculate improvement
        avg_uncached = np.mean(times_uncached)
        avg_cached = np.mean(times_cached)
        cache_speedup = avg_uncached / avg_cached if avg_cached > 0 else 1
        
        benchmark_result = BenchmarkResult(
            test_name="cache_performance",
            document_size="small",
            num_documents=len(documents),
            total_time_seconds=sum(times_cached),
            avg_time_seconds=avg_cached,
            min_time_seconds=min(times_cached),
            max_time_seconds=max(times_cached),
            p50_time_seconds=np.percentile(times_cached, 50),
            p95_time_seconds=np.percentile(times_cached, 95),
            p99_time_seconds=np.percentile(times_cached, 99),
            throughput_docs_per_sec=len(documents) / sum(times_cached),
            memory_used_mb=0,
            cpu_percent=psutil.cpu_percent(interval=0.1),
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(benchmark_result)
        
        print(f"  Uncached avg time: {avg_uncached*1000:.2f}ms")
        print(f"  Cached avg time: {avg_cached*1000:.2f}ms")
        print(f"  Cache speedup: {cache_speedup:.2f}x")
        
        return benchmark_result
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage patterns"""
        print("\nBenchmarking memory usage...")
        
        # Enable memory tracking
        tracemalloc.start()
        
        memory_stats = {}
        
        # Baseline memory
        baseline = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_stats['baseline_mb'] = baseline
        
        # Process small documents
        for _ in range(10):
            doc = DocumentGenerator.generate_text_document("small")
            self.detector.detect_arbitration_clause(doc)
        
        small_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_stats['after_small_docs_mb'] = small_memory
        
        # Process large documents
        for _ in range(5):
            doc = DocumentGenerator.generate_text_document("large")
            self.detector.detect_arbitration_clause(doc)
        
        large_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_stats['after_large_docs_mb'] = large_memory
        
        # Get peak memory
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        memory_stats['peak_memory_mb'] = peak_memory
        
        tracemalloc.stop()
        
        print(f"  Baseline memory: {baseline:.2f} MB")
        print(f"  After small docs: {small_memory:.2f} MB")
        print(f"  After large docs: {large_memory:.2f} MB")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        
        return memory_stats
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("="*80)
        report.append("RAG LEGAL ANALYSIS SYSTEM - PERFORMANCE BENCHMARK REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Group results by test type
        test_groups = {}
        for result in self.results:
            test_type = result.test_name.split('_')[0]
            if test_type not in test_groups:
                test_groups[test_type] = []
            test_groups[test_type].append(result)
        
        # Document Processing Performance
        if 'text' in test_groups:
            report.append("\n## TEXT DOCUMENT PROCESSING")
            report.append("-"*40)
            for result in test_groups['text']:
                report.append(f"\n### Document Size: {result.document_size}")
                report.append(f"  Documents processed: {result.num_documents}")
                report.append(f"  Average time: {result.avg_time_seconds:.3f}s")
                report.append(f"  Min/Max time: {result.min_time_seconds:.3f}s / {result.max_time_seconds:.3f}s")
                report.append(f"  P50/P95/P99: {result.p50_time_seconds:.3f}s / {result.p95_time_seconds:.3f}s / {result.p99_time_seconds:.3f}s")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} docs/sec")
                report.append(f"  Memory used: {result.memory_used_mb:.2f} MB")
        
        # PDF Processing Performance
        if 'pdf' in test_groups:
            report.append("\n## PDF DOCUMENT PROCESSING")
            report.append("-"*40)
            for result in test_groups['pdf']:
                report.append(f"\n### Document Size: {result.document_size}")
                report.append(f"  PDFs processed: {result.num_documents}")
                report.append(f"  Average time: {result.avg_time_seconds:.3f}s")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} docs/sec")
        
        # Vector Search Performance
        if 'vector' in test_groups:
            report.append("\n## VECTOR SIMILARITY SEARCH")
            report.append("-"*40)
            for result in test_groups['vector']:
                report.append(f"  Queries executed: {result.num_documents}")
                report.append(f"  Average time: {result.avg_time_seconds*1000:.2f}ms")
                report.append(f"  P95 latency: {result.p95_time_seconds*1000:.2f}ms")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} queries/sec")
        
        # Pattern Matching Performance
        if 'pattern' in test_groups:
            report.append("\n## PATTERN MATCHING")
            report.append("-"*40)
            for result in test_groups['pattern']:
                report.append(f"\n### Document Size: {result.document_size}")
                report.append(f"  Average time: {result.avg_time_seconds:.3f}s")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} docs/sec")
        
        # Batch Processing Performance
        if 'batch' in test_groups:
            report.append("\n## BATCH PROCESSING")
            report.append("-"*40)
            for result in test_groups['batch']:
                batch_size = result.test_name.split('_')[-1]
                report.append(f"\n### Batch Size: {batch_size}")
                report.append(f"  Total time: {result.total_time_seconds:.3f}s")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} docs/sec")
        
        # Concurrent Processing Performance
        if 'concurrent' in test_groups:
            report.append("\n## CONCURRENT PROCESSING")
            report.append("-"*40)
            for result in test_groups['concurrent']:
                workers = result.test_name.split('_')[-2]
                report.append(f"\n### Workers: {workers}")
                report.append(f"  Documents: {result.num_documents}")
                report.append(f"  Total time: {result.total_time_seconds:.3f}s")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} docs/sec")
        
        # Cache Performance
        if 'cache' in test_groups:
            report.append("\n## CACHE PERFORMANCE")
            report.append("-"*40)
            for result in test_groups['cache']:
                report.append(f"  Cached response time: {result.avg_time_seconds*1000:.2f}ms")
                report.append(f"  Throughput: {result.throughput_docs_per_sec:.2f} docs/sec")
        
        # Performance Summary
        report.append("\n## PERFORMANCE SUMMARY")
        report.append("-"*40)
        
        # Calculate overall statistics
        all_throughputs = [r.throughput_docs_per_sec for r in self.results]
        all_latencies = [r.avg_time_seconds for r in self.results]
        
        report.append(f"  Average throughput: {np.mean(all_throughputs):.2f} docs/sec")
        report.append(f"  Average latency: {np.mean(all_latencies):.3f}s")
        report.append(f"  Total tests run: {len(self.results)}")
        
        # Recommendations
        report.append("\n## RECOMMENDATIONS")
        report.append("-"*40)
        
        # Check for performance issues
        slow_tests = [r for r in self.results if r.avg_time_seconds > 2.0]
        if slow_tests:
            report.append("  ⚠️  Some tests showed high latency (>2s):")
            for test in slow_tests[:3]:
                report.append(f"     - {test.test_name}: {test.avg_time_seconds:.3f}s")
            report.append("     Consider optimizing these components")
        
        # Check memory usage
        high_memory = [r for r in self.results if r.memory_used_mb > 100]
        if high_memory:
            report.append("  ⚠️  High memory usage detected in some tests")
            report.append("     Consider implementing memory optimization strategies")
        
        # Check throughput
        low_throughput = [r for r in self.results if r.throughput_docs_per_sec < 1.0]
        if low_throughput:
            report.append("  ⚠️  Low throughput detected in some scenarios")
            report.append("     Consider implementing parallel processing or caching")
        
        if not (slow_tests or high_memory or low_throughput):
            report.append("  ✅ All performance metrics within acceptable ranges")
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            
            # Also save raw results as JSON
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)
        
        return report_text


def main():
    """Run complete benchmark suite"""
    
    print("Starting RAG Legal Analysis System Performance Benchmarks...")
    print("="*60)
    
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    print("\n1. Testing Text Document Processing...")
    benchmark.benchmark_text_processing(["small", "medium", "large"])
    
    print("\n2. Testing PDF Processing...")
    benchmark.benchmark_pdf_processing(["small", "medium"])
    
    print("\n3. Testing Vector Search...")
    benchmark.benchmark_vector_search(100)
    
    print("\n4. Testing Pattern Matching...")
    benchmark.benchmark_pattern_matching(["small", "medium", "large"])
    
    print("\n5. Testing Batch Processing...")
    benchmark.benchmark_batch_processing([1, 5, 10, 20])
    
    print("\n6. Testing Concurrent Processing...")
    benchmark.benchmark_concurrent_processing([1, 2, 4, 8])
    
    print("\n7. Testing Cache Performance...")
    benchmark.benchmark_cache_performance()
    
    print("\n8. Testing Memory Usage...")
    memory_stats = benchmark.benchmark_memory_usage()
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_report_{timestamp}.txt"
    
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE REPORT...")
    print("="*60)
    
    report = benchmark.generate_report(report_file)
    print(report)
    
    print(f"\n✅ Performance benchmark complete!")
    print(f"   Report saved to: {report_file}")
    print(f"   JSON data saved to: {report_file.replace('.txt', '.json')}")


if __name__ == "__main__":
    main()