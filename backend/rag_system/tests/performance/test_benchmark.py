"""Performance benchmarking script for the RAG system components."""
import os
import sys
import time
import json
import psutil
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.arbitration_detector import ArbitrationDetectionPipeline
from src.models.legal_bert_detector import LegalBERTDetector
from src.document.section_detector import DocumentStructureAnalyzer
from src.comparison.comparison_engine import ClauseComparisonEngine

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for RAG system."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pipeline = ArbitrationDetectionPipeline(cache_enabled=True)
        self.bert_detector = LegalBERTDetector()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.comparison_engine = ClauseComparisonEngine()
        
        # Results storage
        self.results = {
            'document_processing': [],
            'api_response': [],
            'database_queries': [],
            'vector_search': [],
            'cache_performance': [],
            'memory_usage': [],
            'cpu_usage': [],
            'inference_time': []
        }
        
    def measure_time_and_memory(self, func, *args, **kwargs):
        """Measure execution time and memory usage of a function."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        
        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used': peak / 1024 / 1024,  # Convert to MB
            'memory_delta': mem_after - mem_before,
            'cpu_percent': process.cpu_percent()
        }
    
    def benchmark_document_processing(self, num_docs: int = 10):
        """Benchmark document processing speed for various file sizes."""
        print("\n" + "="*60)
        print("BENCHMARKING DOCUMENT PROCESSING")
        print("="*60)
        
        # Create test documents of various sizes
        test_docs = self._create_test_documents(num_docs)
        
        for doc_path, doc_size in test_docs:
            print(f"\nProcessing {doc_size/1024:.1f}KB document...")
            
            # Test PDF extraction
            metrics = self.measure_time_and_memory(
                self.structure_analyzer.extract_full_text,
                doc_path
            )
            
            self.results['document_processing'].append({
                'document_size_kb': doc_size / 1024,
                'extraction_time': metrics['execution_time'],
                'memory_used_mb': metrics['memory_used'],
                'type': 'pdf_extraction'
            })
            
            # Test full pipeline detection
            metrics = self.measure_time_and_memory(
                self.pipeline.detect_arbitration_clause,
                doc_path
            )
            
            self.results['document_processing'].append({
                'document_size_kb': doc_size / 1024,
                'detection_time': metrics['execution_time'],
                'memory_used_mb': metrics['memory_used'],
                'type': 'full_detection'
            })
            
            # Clean up
            os.unlink(doc_path)
        
        # Calculate statistics
        df = pd.DataFrame(self.results['document_processing'])
        
        print("\n" + "-"*40)
        print("Document Processing Summary:")
        print("-"*40)
        
        for doc_type in df['type'].unique():
            type_data = df[df['type'] == doc_type]
            print(f"\n{doc_type}:")
            print(f"  Average time: {type_data['extraction_time'].mean():.3f}s")
            print(f"  95th percentile: {type_data['extraction_time'].quantile(0.95):.3f}s")
            print(f"  Max memory: {type_data['memory_used_mb'].max():.1f}MB")
    
    def benchmark_bert_inference(self, num_samples: int = 100):
        """Benchmark Legal-BERT inference time."""
        print("\n" + "="*60)
        print("BENCHMARKING LEGAL-BERT INFERENCE")
        print("="*60)
        
        # Generate test texts of various lengths
        test_texts = self._generate_test_texts(num_samples)
        
        inference_times = []
        text_lengths = []
        
        for i, text in enumerate(test_texts):
            if i % 20 == 0:
                print(f"Processing sample {i+1}/{num_samples}...")
            
            # Measure inference time
            metrics = self.measure_time_and_memory(
                self.bert_detector.detect,
                text
            )
            
            inference_times.append(metrics['execution_time'])
            text_lengths.append(len(text))
            
            self.results['inference_time'].append({
                'text_length': len(text),
                'inference_time': metrics['execution_time'],
                'memory_used_mb': metrics['memory_used']
            })
        
        # Calculate statistics
        print("\n" + "-"*40)
        print("BERT Inference Summary:")
        print("-"*40)
        print(f"  Average inference time: {np.mean(inference_times)*1000:.2f}ms")
        print(f"  Median inference time: {np.median(inference_times)*1000:.2f}ms")
        print(f"  95th percentile: {np.percentile(inference_times, 95)*1000:.2f}ms")
        print(f"  99th percentile: {np.percentile(inference_times, 99)*1000:.2f}ms")
        print(f"  Throughput: {num_samples/sum(inference_times):.1f} samples/second")
    
    def benchmark_vector_search(self, num_queries: int = 50):
        """Benchmark vector similarity search performance."""
        print("\n" + "="*60)
        print("BENCHMARKING VECTOR SIMILARITY SEARCH")
        print("="*60)
        
        # Generate test queries
        test_queries = self._generate_test_texts(num_queries, max_length=500)
        
        search_times = []
        result_counts = []
        
        for i, query in enumerate(test_queries):
            if i % 10 == 0:
                print(f"Processing query {i+1}/{num_queries}...")
            
            # Measure vector search time
            start = time.perf_counter()
            try:
                results = self.comparison_engine.compare_clause(query, top_k=10)
                end = time.perf_counter()
                
                search_times.append(end - start)
                result_counts.append(len(results.get('similar_clauses', [])))
                
                self.results['vector_search'].append({
                    'query_length': len(query),
                    'search_time': end - start,
                    'num_results': len(results.get('similar_clauses', []))
                })
            except Exception as e:
                print(f"  Error in vector search: {e}")
        
        if search_times:
            # Calculate statistics
            print("\n" + "-"*40)
            print("Vector Search Summary:")
            print("-"*40)
            print(f"  Average search time: {np.mean(search_times)*1000:.2f}ms")
            print(f"  Median search time: {np.median(search_times)*1000:.2f}ms")
            print(f"  95th percentile: {np.percentile(search_times, 95)*1000:.2f}ms")
            print(f"  Average results returned: {np.mean(result_counts):.1f}")
    
    def benchmark_cache_performance(self, num_operations: int = 100):
        """Benchmark cache hit/miss performance."""
        print("\n" + "="*60)
        print("BENCHMARKING CACHE PERFORMANCE")
        print("="*60)
        
        if not self.pipeline.cache_enabled:
            print("Cache is not enabled. Skipping cache benchmarks.")
            return
        
        # Create a test document
        test_doc = self._create_test_documents(1)[0][0]
        
        # Test cache misses (first access)
        cache_miss_times = []
        for i in range(5):
            # Clear cache
            self.pipeline.cache.flushall()
            
            start = time.perf_counter()
            self.pipeline.detect_arbitration_clause(test_doc)
            end = time.perf_counter()
            
            cache_miss_times.append(end - start)
            
            self.results['cache_performance'].append({
                'operation': 'miss',
                'time': end - start
            })
        
        # Test cache hits (subsequent accesses)
        cache_hit_times = []
        
        # Ensure document is in cache
        self.pipeline.detect_arbitration_clause(test_doc)
        
        for i in range(20):
            start = time.perf_counter()
            self.pipeline.detect_arbitration_clause(test_doc)
            end = time.perf_counter()
            
            cache_hit_times.append(end - start)
            
            self.results['cache_performance'].append({
                'operation': 'hit',
                'time': end - start
            })
        
        # Clean up
        os.unlink(test_doc)
        
        # Calculate statistics
        print("\n" + "-"*40)
        print("Cache Performance Summary:")
        print("-"*40)
        print(f"  Average cache miss time: {np.mean(cache_miss_times)*1000:.2f}ms")
        print(f"  Average cache hit time: {np.mean(cache_hit_times)*1000:.2f}ms")
        print(f"  Cache speedup: {np.mean(cache_miss_times)/np.mean(cache_hit_times):.1f}x")
        print(f"  Hit ratio benefit: {(1 - np.mean(cache_hit_times)/np.mean(cache_miss_times))*100:.1f}%")
    
    def benchmark_concurrent_load(self, num_concurrent: int = 10):
        """Benchmark system under concurrent load."""
        print("\n" + "="*60)
        print("BENCHMARKING CONCURRENT LOAD")
        print("="*60)
        
        # Create test data
        test_texts = self._generate_test_texts(num_concurrent)
        
        def process_text(text):
            """Process a single text."""
            start = time.perf_counter()
            result = self.pipeline.detect_from_text(text)
            end = time.perf_counter()
            return end - start
        
        # Test sequential processing
        print(f"\nSequential processing ({num_concurrent} requests)...")
        sequential_start = time.perf_counter()
        sequential_times = [process_text(text) for text in test_texts]
        sequential_end = time.perf_counter()
        
        # Test concurrent processing
        print(f"Concurrent processing ({num_concurrent} requests)...")
        concurrent_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            concurrent_times = list(executor.map(process_text, test_texts))
        concurrent_end = time.perf_counter()
        
        # Calculate statistics
        print("\n" + "-"*40)
        print("Concurrent Load Summary:")
        print("-"*40)
        print(f"  Sequential total time: {sequential_end - sequential_start:.2f}s")
        print(f"  Concurrent total time: {concurrent_end - concurrent_start:.2f}s")
        print(f"  Speedup: {(sequential_end - sequential_start)/(concurrent_end - concurrent_start):.2f}x")
        print(f"  Sequential avg per request: {np.mean(sequential_times)*1000:.2f}ms")
        print(f"  Concurrent avg per request: {np.mean(concurrent_times)*1000:.2f}ms")
        print(f"  Throughput (sequential): {num_concurrent/(sequential_end - sequential_start):.1f} req/s")
        print(f"  Throughput (concurrent): {num_concurrent/(concurrent_end - concurrent_start):.1f} req/s")
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = [1, 5, 10, 20, 50]):
        """Benchmark batch processing performance."""
        print("\n" + "="*60)
        print("BENCHMARKING BATCH PROCESSING")
        print("="*60)
        
        batch_results = []
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create test documents
            test_docs = self._create_test_documents(batch_size)
            
            # Process batch
            start = time.perf_counter()
            results = []
            for doc_path, _ in test_docs:
                result = self.pipeline.detect_arbitration_clause(doc_path)
                results.append(result)
            end = time.perf_counter()
            
            total_time = end - start
            avg_time_per_doc = total_time / batch_size
            throughput = batch_size / total_time
            
            batch_results.append({
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_doc': avg_time_per_doc,
                'throughput': throughput
            })
            
            # Clean up
            for doc_path, _ in test_docs:
                os.unlink(doc_path)
            
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg per document: {avg_time_per_doc:.3f}s")
            print(f"  Throughput: {throughput:.1f} docs/s")
        
        self.results['batch_processing'] = batch_results
        
        # Plot batch processing performance
        self._plot_batch_performance(batch_results)
    
    def benchmark_memory_usage(self, num_iterations: int = 20):
        """Benchmark memory usage patterns."""
        print("\n" + "="*60)
        print("BENCHMARKING MEMORY USAGE")
        print("="*60)
        
        process = psutil.Process()
        memory_samples = []
        
        # Create test data
        test_texts = self._generate_test_texts(num_iterations)
        
        print("Monitoring memory usage during processing...")
        
        for i, text in enumerate(test_texts):
            # Get memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process text
            result = self.pipeline.detect_from_text(text)
            
            # Get memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_samples.append({
                'iteration': i + 1,
                'memory_before': mem_before,
                'memory_after': mem_after,
                'memory_delta': mem_after - mem_before,
                'text_length': len(text)
            })
            
            self.results['memory_usage'].append(memory_samples[-1])
        
        # Calculate statistics
        df = pd.DataFrame(memory_samples)
        
        print("\n" + "-"*40)
        print("Memory Usage Summary:")
        print("-"*40)
        print(f"  Starting memory: {df['memory_before'].iloc[0]:.1f}MB")
        print(f"  Ending memory: {df['memory_after'].iloc[-1]:.1f}MB")
        print(f"  Total increase: {df['memory_after'].iloc[-1] - df['memory_before'].iloc[0]:.1f}MB")
        print(f"  Average per operation: {df['memory_delta'].mean():.2f}MB")
        print(f"  Max spike: {df['memory_delta'].max():.2f}MB")
        
        # Check for memory leaks
        if df['memory_after'].iloc[-1] - df['memory_before'].iloc[0] > 50:
            print("  WARNING: Potential memory leak detected!")
    
    def _create_test_documents(self, num_docs: int) -> List[Tuple[str, int]]:
        """Create test PDF documents of various sizes."""
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        docs = []
        temp_dir = tempfile.mkdtemp()
        
        # Different document sizes (in KB)
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        
        for i in range(num_docs):
            size = sizes[i % len(sizes)]
            
            # Create PDF
            pdf_path = os.path.join(temp_dir, f"test_doc_{i}_{size}kb.pdf")
            c = canvas.Canvas(pdf_path, pagesize=letter)
            
            # Add content to reach approximate size
            text = self._generate_legal_text(size * 100)  # Rough approximation
            
            # Split text into pages
            lines = text.split('\n')
            page_lines = 50
            
            for j in range(0, len(lines), page_lines):
                if j > 0:
                    c.showPage()
                
                y_position = 750
                for line in lines[j:j+page_lines]:
                    c.drawString(50, y_position, line[:100])  # Truncate long lines
                    y_position -= 15
            
            c.save()
            
            # Get actual file size
            actual_size = os.path.getsize(pdf_path)
            docs.append((pdf_path, actual_size))
        
        return docs
    
    def _generate_test_texts(self, num_texts: int, max_length: int = 2000) -> List[str]:
        """Generate test texts of various lengths."""
        texts = []
        
        for i in range(num_texts):
            length = np.random.randint(100, max_length)
            text = self._generate_legal_text(length)
            texts.append(text)
        
        return texts
    
    def _generate_legal_text(self, approx_length: int) -> str:
        """Generate sample legal text."""
        templates = [
            "This Agreement contains provisions regarding {topic}. ",
            "The parties agree that {action} shall be {condition}. ",
            "All disputes arising from {context} must be resolved through {method}. ",
            "In the event of {event}, the {party} shall {obligation}. ",
            "Arbitration proceedings shall be conducted in accordance with {rules}. ",
            "The arbitrator's decision shall be {binding} and {finality}. ",
            "Class action lawsuits are {status} under this agreement. ",
            "Any claim must be brought within {timeframe} of the dispute arising. "
        ]
        
        topics = ["arbitration", "mediation", "litigation", "dispute resolution", "claims"]
        actions = ["disputes", "claims", "controversies", "disagreements", "conflicts"]
        conditions = ["binding", "mandatory", "optional", "required", "enforceable"]
        
        text = ""
        while len(text) < approx_length:
            template = np.random.choice(templates)
            text += template.format(
                topic=np.random.choice(topics),
                action=np.random.choice(actions),
                condition=np.random.choice(conditions),
                context="this agreement",
                method="binding arbitration",
                event="a dispute",
                party="affected party",
                obligation="provide notice",
                rules="AAA rules",
                binding="final",
                finality="binding",
                status="waived",
                timeframe="30 days"
            )
        
        return text[:approx_length]
    
    def _plot_batch_performance(self, batch_results: List[Dict]):
        """Plot batch processing performance."""
        df = pd.DataFrame(batch_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Throughput vs batch size
        axes[0].plot(df['batch_size'], df['throughput'], 'bo-')
        axes[0].set_xlabel('Batch Size')
        axes[0].set_ylabel('Throughput (docs/s)')
        axes[0].set_title('Throughput vs Batch Size')
        axes[0].grid(True, alpha=0.3)
        
        # Average time per document vs batch size
        axes[1].plot(df['batch_size'], df['avg_time_per_doc'], 'ro-')
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Avg Time per Document (s)')
        axes[1].set_title('Processing Time vs Batch Size')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_performance.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("GENERATING PERFORMANCE REPORT")
        print("="*60)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            clean_results = {}
            for key, value in self.results.items():
                if isinstance(value, list):
                    clean_results[key] = value
                else:
                    clean_results[key] = str(value)
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate visualizations if data exists
        if self.results['document_processing']:
            self._generate_visualizations()
        
        # Generate markdown report
        self._generate_markdown_report(timestamp)
    
    def _generate_visualizations(self):
        """Generate performance visualization plots."""
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Document processing times
        if self.results['document_processing']:
            df = pd.DataFrame(self.results['document_processing'])
            if 'document_size_kb' in df.columns:
                for doc_type in df['type'].unique():
                    type_data = df[df['type'] == doc_type]
                    if 'extraction_time' in type_data.columns:
                        axes[0, 0].scatter(type_data['document_size_kb'], 
                                         type_data['extraction_time'], 
                                         label=doc_type, alpha=0.6)
                    elif 'detection_time' in type_data.columns:
                        axes[0, 0].scatter(type_data['document_size_kb'], 
                                         type_data['detection_time'], 
                                         label=doc_type, alpha=0.6)
                
                axes[0, 0].set_xlabel('Document Size (KB)')
                axes[0, 0].set_ylabel('Processing Time (s)')
                axes[0, 0].set_title('Document Processing Performance')
                axes[0, 0].legend()
        
        # 2. BERT inference times
        if self.results['inference_time']:
            df = pd.DataFrame(self.results['inference_time'])
            axes[0, 1].scatter(df['text_length'], df['inference_time'] * 1000, alpha=0.5)
            axes[0, 1].set_xlabel('Text Length (characters)')
            axes[0, 1].set_ylabel('Inference Time (ms)')
            axes[0, 1].set_title('Legal-BERT Inference Performance')
        
        # 3. Cache performance
        if self.results['cache_performance']:
            df = pd.DataFrame(self.results['cache_performance'])
            cache_data = df.groupby('operation')['time'].apply(list).to_dict()
            
            if 'hit' in cache_data and 'miss' in cache_data:
                data = [cache_data['hit'], cache_data['miss']]
                axes[1, 0].boxplot(data, labels=['Cache Hit', 'Cache Miss'])
                axes[1, 0].set_ylabel('Time (s)')
                axes[1, 0].set_title('Cache Performance')
        
        # 4. Memory usage over time
        if self.results['memory_usage']:
            df = pd.DataFrame(self.results['memory_usage'])
            axes[1, 1].plot(df['iteration'], df['memory_after'], 'b-', label='Memory Used')
            axes[1, 1].fill_between(df['iteration'], 
                                   df['memory_before'], 
                                   df['memory_after'],
                                   alpha=0.3)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].set_title('Memory Usage Pattern')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {self.output_dir / 'performance_metrics.png'}")
    
    def _generate_markdown_report(self, timestamp: str):
        """Generate markdown report with all results."""
        report = []
        report.append("# RAG System Performance Benchmark Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n---\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        
        if self.results['document_processing']:
            df = pd.DataFrame(self.results['document_processing'])
            if 'detection_time' in df.columns:
                avg_time = df['detection_time'].mean()
                report.append(f"- **Average Document Processing:** {avg_time:.3f}s")
        
        if self.results['inference_time']:
            df = pd.DataFrame(self.results['inference_time'])
            avg_inference = df['inference_time'].mean() * 1000
            p95_inference = df['inference_time'].quantile(0.95) * 1000
            report.append(f"- **Average BERT Inference:** {avg_inference:.2f}ms")
            report.append(f"- **P95 BERT Inference:** {p95_inference:.2f}ms")
        
        if self.results['cache_performance']:
            df = pd.DataFrame(self.results['cache_performance'])
            hit_times = df[df['operation'] == 'hit']['time'].values
            miss_times = df[df['operation'] == 'miss']['time'].values
            
            if len(hit_times) > 0 and len(miss_times) > 0:
                speedup = np.mean(miss_times) / np.mean(hit_times)
                report.append(f"- **Cache Speedup:** {speedup:.1f}x")
        
        report.append("\n---\n")
        
        # Detailed Results
        report.append("## Detailed Results\n")
        
        # Add sections for each benchmark
        sections = [
            ('Document Processing', 'document_processing'),
            ('BERT Inference', 'inference_time'),
            ('Vector Search', 'vector_search'),
            ('Cache Performance', 'cache_performance'),
            ('Memory Usage', 'memory_usage'),
            ('Batch Processing', 'batch_processing')
        ]
        
        for title, key in sections:
            if key in self.results and self.results[key]:
                report.append(f"### {title}\n")
                
                df = pd.DataFrame(self.results[key])
                
                # Create summary table
                if not df.empty:
                    report.append("| Metric | Value |")
                    report.append("|--------|-------|")
                    
                    # Add relevant metrics based on data
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols:
                        if col != 'iteration':
                            mean_val = df[col].mean()
                            max_val = df[col].max()
                            min_val = df[col].min()
                            
                            report.append(f"| {col.replace('_', ' ').title()} (avg) | {mean_val:.3f} |")
                            if max_val != min_val:
                                report.append(f"| {col.replace('_', ' ').title()} (max) | {max_val:.3f} |")
                
                report.append("\n")
        
        report.append("\n---\n")
        
        # Recommendations
        report.append("## Performance Recommendations\n")
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")
        
        # Save report
        report_file = self.output_dir / f"performance_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Markdown report saved to: {report_file}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        # Check BERT inference performance
        if self.results['inference_time']:
            df = pd.DataFrame(self.results['inference_time'])
            avg_inference = df['inference_time'].mean() * 1000
            
            if avg_inference > 100:
                recommendations.append("Consider using GPU acceleration for BERT inference (currently >100ms)")
            if avg_inference > 50:
                recommendations.append("Consider implementing request batching for better throughput")
        
        # Check memory usage
        if self.results['memory_usage']:
            df = pd.DataFrame(self.results['memory_usage'])
            memory_growth = df['memory_after'].iloc[-1] - df['memory_before'].iloc[0]
            
            if memory_growth > 100:
                recommendations.append(f"High memory growth detected ({memory_growth:.1f}MB) - investigate potential memory leaks")
        
        # Check cache performance
        if self.results['cache_performance']:
            df = pd.DataFrame(self.results['cache_performance'])
            hit_times = df[df['operation'] == 'hit']['time'].values
            miss_times = df[df['operation'] == 'miss']['time'].values
            
            if len(hit_times) > 0 and len(miss_times) > 0:
                speedup = np.mean(miss_times) / np.mean(hit_times)
                
                if speedup < 2:
                    recommendations.append("Cache speedup is low (<2x) - consider optimizing cache implementation")
                else:
                    recommendations.append(f"Cache is providing {speedup:.1f}x speedup - ensure high cache hit ratio")
        
        # Check document processing
        if self.results['document_processing']:
            df = pd.DataFrame(self.results['document_processing'])
            if 'detection_time' in df.columns:
                avg_time = df['detection_time'].mean()
                
                if avg_time > 5:
                    recommendations.append("Document processing is slow (>5s) - consider parallel processing for large documents")
        
        if not recommendations:
            recommendations.append("System is performing within acceptable parameters")
        
        return recommendations

def main():
    """Run complete benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RAG system performance")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer iterations)")
    parser.add_argument("--components", nargs="+", 
                       choices=["document", "bert", "vector", "cache", "concurrent", "batch", "memory"],
                       help="Specific components to benchmark")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(output_dir=args.output_dir)
    
    # Determine which benchmarks to run
    if args.components:
        components = args.components
    else:
        components = ["document", "bert", "vector", "cache", "concurrent", "batch", "memory"]
    
    # Adjust iterations for quick mode
    if args.quick:
        doc_count = 3
        sample_count = 20
        query_count = 10
        concurrent_count = 5
    else:
        doc_count = 10
        sample_count = 100
        query_count = 50
        concurrent_count = 10
    
    # Run selected benchmarks
    try:
        if "document" in components:
            benchmark.benchmark_document_processing(doc_count)
        
        if "bert" in components:
            benchmark.benchmark_bert_inference(sample_count)
        
        if "vector" in components:
            benchmark.benchmark_vector_search(query_count)
        
        if "cache" in components:
            benchmark.benchmark_cache_performance()
        
        if "concurrent" in components:
            benchmark.benchmark_concurrent_load(concurrent_count)
        
        if "batch" in components:
            batch_sizes = [1, 5, 10] if args.quick else [1, 5, 10, 20, 50]
            benchmark.benchmark_batch_processing(batch_sizes)
        
        if "memory" in components:
            iterations = 10 if args.quick else 20
            benchmark.benchmark_memory_usage(iterations)
        
        # Generate report
        benchmark.generate_report()
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        print(f"Results saved to: {benchmark.output_dir}")
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())