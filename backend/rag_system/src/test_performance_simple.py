#!/usr/bin/env python3
"""
Simplified performance testing for RAG system database and vector store operations.
Uses only standard library and commonly available packages.
"""

import os
import sys
import time
import logging
import traceback
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import sqlite3
import json
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplePerformanceMonitor:
    """Simple performance monitor using basic memory tracking."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def log_operation(self, operation: str):
        """Log operation completion."""
        elapsed = time.time() - self.start_time
        logger.info(f"Completed {operation} in {elapsed:.3f} seconds")
        
    def get_elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time

class DatabaseTester:
    """Simplified database testing."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.db_path = os.path.join(temp_dir, "test_arbitration.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.monitor = SimplePerformanceMonitor()
        
        try:
            from database.schema import DatabaseManager
            self.db_manager = DatabaseManager(self.db_url)
        except Exception as e:
            logger.error(f"Could not initialize DatabaseManager: {e}")
            self.db_manager = None
        
    def generate_sample_clauses(self, count: int) -> List[Dict]:
        """Generate sample arbitration clauses for testing."""
        sample_clauses = []
        companies = ["TechCorp", "FinanceInc", "HealthCare LLC", "RetailCorp", "ManufacturingCo"]
        industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
        doc_types = ["TOS", "Employment", "Service", "Privacy", "Contract"]
        jurisdictions = ["US", "UK", "EU", "CA", "AU"]
        
        base_clauses = [
            "Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration in accordance with the Commercial Arbitration Rules of the American Arbitration Association.",
            "The parties agree that all disputes must be submitted to final and binding arbitration under JAMS rules. You waive any right to a class action or jury trial.",
            "Disputes will be resolved by mandatory arbitration on an individual basis only. No class actions are permitted under this agreement.",
            "Either party may elect to resolve disputes through arbitration administered by the International Chamber of Commerce in accordance with its rules.",
            "All claims shall be arbitrated in accordance with the Federal Arbitration Act. You have 30 days to opt out of this arbitration provision."
        ]
        
        for i in range(count):
            company = companies[i % len(companies)]
            industry = industries[i % len(industries)]
            doc_type = doc_types[i % len(doc_types)]
            jurisdiction = jurisdictions[i % len(jurisdictions)]
            base_text = base_clauses[i % len(base_clauses)]
            
            clause_text = f"{base_text} This clause was added on {datetime.now().strftime('%Y-%m-%d')} for {company}."
            
            clause = {
                "company": company,
                "industry": industry,
                "document_type": doc_type,
                "clause_text": clause_text,
                "summary": clause_text[:100] + "...",
                "key_provisions": [f"binding_arbitration", f"{doc_type.lower()}_specific"],
                "enforceability": 0.5 + (i % 5) * 0.1,  # Simple variation
                "risk_score": 0.3 + (i % 7) * 0.1,  # Simple variation
                "jurisdiction": jurisdiction,
                "metadata": {"test_id": i, "generated": True}
            }
            sample_clauses.append(clause)
            
        return sample_clauses
    
    def test_basic_database_operations(self) -> Dict:
        """Test basic CRUD operations."""
        logger.info("Testing basic database operations...")
        results = {}
        
        if not self.db_manager:
            results['error'] = 'Database manager not initialized'
            return results
        
        try:
            # Test single insert
            start_time = time.time()
            sample_clause = self.generate_sample_clauses(1)[0]
            
            clause_id = self.db_manager.add_clause({
                'company_name': sample_clause['company'],
                'industry': sample_clause['industry'],
                'document_type': sample_clause['document_type'],
                'clause_text': sample_clause['clause_text'],
                'clause_summary': sample_clause['summary'],
                'key_provisions': sample_clause['key_provisions'],
                'enforceability_score': sample_clause['enforceability'],
                'risk_score': sample_clause['risk_score'],
                'jurisdiction': sample_clause['jurisdiction'],
                'vector_id': f"test_vector_{clause_id}",
                'metadata': sample_clause['metadata']
            })
            
            insert_time = time.time() - start_time
            results['single_insert_time'] = insert_time
            results['single_insert_success'] = clause_id > 0
            
            # Test retrieval
            start_time = time.time()
            retrieved_clause = self.db_manager.get_clause(clause_id)
            retrieve_time = time.time() - start_time
            
            results['single_retrieve_time'] = retrieve_time
            results['single_retrieve_success'] = retrieved_clause is not None
            results['data_integrity'] = (
                retrieved_clause and
                retrieved_clause['company_name'] == sample_clause['company'] and
                retrieved_clause['industry'] == sample_clause['industry']
            )
            
            # Test search with filters
            start_time = time.time()
            search_results = self.db_manager.search_clauses({
                'company_name': sample_clause['company']
            })
            search_time = time.time() - start_time
            
            results['search_time'] = search_time
            results['search_success'] = len(search_results) > 0
            
            logger.info(f"Basic operations completed: {results}")
            
        except Exception as e:
            logger.error(f"Error in basic database operations: {e}")
            results['error'] = str(e)
            traceback.print_exc()
            
        return results
    
    def test_bulk_operations(self, batch_sizes: List[int] = [10, 50, 100]) -> Dict:
        """Test bulk insert and query operations."""
        logger.info("Testing bulk database operations...")
        results = {}
        
        if not self.db_manager:
            results['error'] = 'Database manager not initialized'
            return results
        
        for batch_size in batch_sizes:
            try:
                logger.info(f"Testing batch size: {batch_size}")
                
                # Generate test data
                sample_clauses = self.generate_sample_clauses(batch_size)
                
                # Test bulk insert
                start_time = time.time()
                
                inserted_ids = []
                for i, clause in enumerate(sample_clauses):
                    try:
                        clause_id = self.db_manager.add_clause({
                            'company_name': clause['company'],
                            'industry': clause['industry'],
                            'document_type': clause['document_type'],
                            'clause_text': clause['clause_text'],
                            'clause_summary': clause['summary'],
                            'key_provisions': clause['key_provisions'],
                            'enforceability_score': clause['enforceability'],
                            'risk_score': clause['risk_score'],
                            'jurisdiction': clause['jurisdiction'],
                            'vector_id': f"bulk_test_{batch_size}_{i}",
                            'metadata': clause['metadata']
                        })
                        inserted_ids.append(clause_id)
                    except Exception as e:
                        logger.error(f"Error inserting clause {i}: {e}")
                
                insert_time = time.time() - start_time
                
                results[f'batch_{batch_size}'] = {
                    'insert_time': insert_time,
                    'insert_rate': len(inserted_ids) / insert_time if insert_time > 0 else 0,
                    'success_rate': len(inserted_ids) / batch_size,
                    'inserted_count': len(inserted_ids)
                }
                
                # Test bulk search
                start_time = time.time()
                search_results = self.db_manager.search_clauses({'industry': 'Technology'})
                search_time = time.time() - start_time
                
                results[f'batch_{batch_size}']['search_time'] = search_time
                results[f'batch_{batch_size}']['search_results'] = len(search_results)
                
                logger.info(f"Batch {batch_size}: {results[f'batch_{batch_size}']}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_size}: {e}")
                results[f'batch_{batch_size}'] = {'error': str(e)}
        
        return results
    
    def test_index_performance(self) -> Dict:
        """Test database index performance."""
        logger.info("Testing database index performance...")
        results = {}
        
        if not self.db_manager:
            results['error'] = 'Database manager not initialized'
            return results
        
        try:
            # Add test data if needed
            sample_clauses = self.generate_sample_clauses(50)
            for i, clause in enumerate(sample_clauses):
                self.db_manager.add_clause({
                    'company_name': clause['company'],
                    'industry': clause['industry'],
                    'document_type': clause['document_type'],
                    'clause_text': clause['clause_text'],
                    'clause_summary': clause['summary'],
                    'key_provisions': clause['key_provisions'],
                    'enforceability_score': clause['enforceability'],
                    'risk_score': clause['risk_score'],
                    'jurisdiction': clause['jurisdiction'],
                    'vector_id': f"index_test_{i}",
                    'metadata': clause['metadata']
                })
            
            # Test different query patterns
            test_queries = [
                {'company_name': 'TechCorp'},
                {'industry': 'Technology'},
                {'document_type': 'TOS'},
                {'jurisdiction': 'US'},
                {'min_risk': 0.5},
                {'max_risk': 0.7}
            ]
            
            for i, query in enumerate(test_queries):
                start_time = time.time()
                search_results = self.db_manager.search_clauses(query)
                query_time = time.time() - start_time
                
                results[f'query_{i}'] = {
                    'query': str(query),  # Convert to string for JSON serialization
                    'time': query_time,
                    'results_count': len(search_results)
                }
                
        except Exception as e:
            logger.error(f"Error testing index performance: {e}")
            results['error'] = str(e)
            
        return results

class VectorStoreTester:
    """Simplified vector store testing."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.monitor = SimplePerformanceMonitor()
        
        try:
            from database.schema import VectorStore
            self.vector_store = VectorStore(dimension=768)
        except Exception as e:
            logger.warning(f"Could not initialize VectorStore: {e}")
            self.vector_store = None
        
    def generate_test_embeddings(self, count: int, dimension: int = 768) -> List[Tuple[str, any]]:
        """Generate test embeddings."""
        embeddings = []
        
        try:
            import numpy as np
            for i in range(count):
                # Generate random normalized embedding
                vector = np.random.randn(dimension).astype(np.float32)
                vector = vector / np.linalg.norm(vector)  # Normalize
                embeddings.append((f"test_clause_{i}", vector))
        except ImportError:
            logger.warning("NumPy not available, skipping vector tests")
            
        return embeddings
    
    def test_basic_vector_operations(self) -> Dict:
        """Test basic vector store operations."""
        logger.info("Testing basic vector store operations...")
        results = {}
        
        if not self.vector_store:
            results['error'] = 'Vector store not initialized'
            return results
        
        try:
            # Clear store for clean test
            self.vector_store.clear()
            
            # Test single vector addition
            start_time = time.time()
            test_embeddings = self.generate_test_embeddings(1)
            
            if not test_embeddings:
                results['error'] = 'Could not generate test embeddings'
                return results
            
            clause_id, embedding = test_embeddings[0]
            success = self.vector_store.add_clause(clause_id, embedding)
            add_time = time.time() - start_time
            
            results['single_add_time'] = add_time
            results['single_add_success'] = success
            
            # Test search
            start_time = time.time()
            search_results = self.vector_store.search_similar(embedding, k=1)
            search_time = time.time() - start_time
            
            results['single_search_time'] = search_time
            results['search_results_count'] = len(search_results)
            results['search_accuracy'] = (
                len(search_results) > 0 and search_results[0][0] == clause_id
            )
            
            # Test stats
            stats = self.vector_store.get_stats()
            results['stats'] = stats
            results['stats_consistency'] = stats['total_vectors'] == 1
            
        except Exception as e:
            logger.error(f"Error in basic vector operations: {e}")
            results['error'] = str(e)
            traceback.print_exc()
            
        return results
    
    def test_bulk_vector_operations(self, sizes: List[int] = [10, 50, 100]) -> Dict:
        """Test bulk vector operations."""
        logger.info("Testing bulk vector operations...")
        results = {}
        
        if not self.vector_store:
            results['error'] = 'Vector store not initialized'
            return results
        
        for size in sizes:
            try:
                logger.info(f"Testing vector bulk size: {size}")
                
                # Clear store
                self.vector_store.clear()
                
                # Generate test embeddings
                test_embeddings = self.generate_test_embeddings(size)
                
                if not test_embeddings:
                    results[f'size_{size}'] = {'error': 'Could not generate test embeddings'}
                    continue
                
                # Test bulk addition
                start_time = time.time()
                
                success_count = 0
                for clause_id, embedding in test_embeddings:
                    if self.vector_store.add_clause(clause_id, embedding):
                        success_count += 1
                
                add_time = time.time() - start_time
                
                # Test bulk search
                query_embedding = test_embeddings[0][1]  # Use first embedding as query
                
                # Test different k values
                search_times = {}
                for k in [1, 5, min(10, size)]:
                    if k <= size:
                        start_time = time.time()
                        search_results = self.vector_store.search_similar(query_embedding, k=k)
                        search_time = time.time() - start_time
                        search_times[f'k_{k}'] = {
                            'time': search_time,
                            'results_count': len(search_results),
                            'top_similarity': float(search_results[0][1]) if search_results else 0.0
                        }
                
                results[f'size_{size}'] = {
                    'add_time': add_time,
                    'add_rate': success_count / add_time if add_time > 0 else 0,
                    'success_rate': success_count / size,
                    'search_times': search_times,
                    'final_vector_count': self.vector_store.get_stats()['total_vectors']
                }
                
                logger.info(f"Size {size}: {results[f'size_{size}']}")
                
            except Exception as e:
                logger.error(f"Error in bulk size {size}: {e}")
                results[f'size_{size}'] = {'error': str(e)}
                traceback.print_exc()
        
        return results

class ConcurrencyTester:
    """Test concurrent access to database."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.db_path = os.path.join(temp_dir, "concurrency_test.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.monitor = SimplePerformanceMonitor()
        
    def worker_function(self, worker_id: int, operation_count: int) -> Dict:
        """Worker function for concurrent testing."""
        results = {
            'worker_id': worker_id,
            'operations_completed': 0,
            'errors': [],
            'start_time': time.time()
        }
        
        try:
            from database.schema import DatabaseManager
            # Create separate database manager for this worker
            db_manager = DatabaseManager(self.db_url)
            
            for i in range(operation_count):
                try:
                    # Perform database operation
                    clause_data = {
                        'company_name': f'Worker{worker_id}Corp',
                        'industry': f'Industry{worker_id}',
                        'document_type': 'TOS',
                        'clause_text': f'Test clause from worker {worker_id}, operation {i}',
                        'clause_summary': f'Summary from worker {worker_id}',
                        'key_provisions': [f'worker_{worker_id}_provision'],
                        'enforceability_score': 0.5,
                        'risk_score': 0.5,
                        'jurisdiction': 'US',
                        'vector_id': f'worker_{worker_id}_op_{i}',
                        'metadata': {'worker_id': worker_id, 'operation': i}
                    }
                    
                    clause_id = db_manager.add_clause(clause_data)
                    results['operations_completed'] += 1
                    
                    # Occasionally read data
                    if i % 5 == 0:
                        retrieved = db_manager.get_clause(clause_id)
                        if not retrieved:
                            results['errors'].append(f"Failed to retrieve clause {clause_id}")
                    
                except Exception as e:
                    results['errors'].append(f"Operation {i}: {str(e)}")
                    
        except Exception as e:
            results['errors'].append(f"Worker initialization error: {str(e)}")
            
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        
        return results
    
    def test_concurrent_database_access(self, num_workers: int = 3, operations_per_worker: int = 10) -> Dict:
        """Test concurrent database access."""
        logger.info(f"Testing concurrent database access with {num_workers} workers...")
        results = {}
        
        try:
            start_time = time.time()
            
            # Use ThreadPoolExecutor for concurrent access
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for worker_id in range(num_workers):
                    future = executor.submit(self.worker_function, worker_id, operations_per_worker)
                    futures.append(future)
                
                # Collect results
                worker_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        worker_results.append(result)
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                        worker_results.append({'error': str(e)})
            
            total_time = time.time() - start_time
            
            # Analyze results
            total_operations = sum(r.get('operations_completed', 0) for r in worker_results)
            total_errors = sum(len(r.get('errors', [])) for r in worker_results)
            
            results['total_time'] = total_time
            results['num_workers'] = num_workers
            results['operations_per_worker'] = operations_per_worker
            results['total_operations_completed'] = total_operations
            results['total_errors'] = total_errors
            results['success_rate'] = total_operations / (num_workers * operations_per_worker)
            results['operations_per_second'] = total_operations / total_time if total_time > 0 else 0
            
            # Limit worker results details for readability
            results['worker_summary'] = [
                {
                    'worker_id': r.get('worker_id'),
                    'operations_completed': r.get('operations_completed', 0),
                    'error_count': len(r.get('errors', [])),
                    'duration': r.get('duration', 0)
                }
                for r in worker_results
            ]
            
            # Check database consistency
            try:
                from database.schema import DatabaseManager
                db_manager = DatabaseManager(self.db_url)
                final_count = len(db_manager.search_clauses({}))
                results['final_database_count'] = final_count
                results['data_consistency'] = final_count == total_operations
            except Exception as e:
                results['consistency_check_error'] = str(e)
                
        except Exception as e:
            logger.error(f"Error in concurrent test: {e}")
            results['error'] = str(e)
            
        return results

def main():
    """Main testing function."""
    logger.info("Starting simplified RAG system performance tests...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    all_results = {
        'test_start_time': datetime.now().isoformat(),
        'temp_directory': temp_dir,
        'python_version': sys.version
    }
    
    try:
        # Test 1: Database Operations
        logger.info("=" * 60)
        logger.info("TESTING DATABASE OPERATIONS")
        logger.info("=" * 60)
        
        db_tester = DatabaseTester(temp_dir)
        
        all_results['database_basic'] = db_tester.test_basic_database_operations()
        all_results['database_bulk'] = db_tester.test_bulk_operations()
        all_results['database_indexes'] = db_tester.test_index_performance()
        
        # Test 2: Vector Store Operations
        logger.info("=" * 60)
        logger.info("TESTING VECTOR STORE OPERATIONS")
        logger.info("=" * 60)
        
        vector_tester = VectorStoreTester(temp_dir)
        
        all_results['vector_basic'] = vector_tester.test_basic_vector_operations()
        all_results['vector_bulk'] = vector_tester.test_bulk_vector_operations()
        
        # Test 3: Concurrent Access
        logger.info("=" * 60)
        logger.info("TESTING CONCURRENT ACCESS")
        logger.info("=" * 60)
        
        concurrency_tester = ConcurrencyTester(temp_dir)
        all_results['concurrency'] = concurrency_tester.test_concurrent_database_access()
        
        # Performance Summary
        logger.info("=" * 60)
        logger.info("PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        
        all_results['performance_summary'] = calculate_performance_summary(all_results)
        
    except Exception as e:
        logger.error(f"Error in main testing: {e}")
        all_results['main_error'] = str(e)
        traceback.print_exc()
    
    finally:
        all_results['test_end_time'] = datetime.now().isoformat()
        
        # Generate performance report
        generate_performance_report(all_results, temp_dir)

def calculate_performance_summary(results: Dict) -> Dict:
    """Calculate overall performance summary."""
    summary = {}
    
    try:
        # Database performance
        db_bulk = results.get('database_bulk', {})
        if db_bulk:
            max_insert_rate = 0
            for key, value in db_bulk.items():
                if key.startswith('batch_') and isinstance(value, dict):
                    insert_rate = value.get('insert_rate', 0)
                    max_insert_rate = max(max_insert_rate, insert_rate)
            summary['max_database_insert_rate'] = max_insert_rate
        
        # Vector store performance
        vector_bulk = results.get('vector_bulk', {})
        if vector_bulk:
            max_vector_rate = 0
            for key, value in vector_bulk.items():
                if key.startswith('size_') and isinstance(value, dict):
                    add_rate = value.get('add_rate', 0)
                    max_vector_rate = max(max_vector_rate, add_rate)
            summary['max_vector_add_rate'] = max_vector_rate
        
        # Concurrency performance
        concurrency = results.get('concurrency', {})
        if concurrency:
            summary['concurrent_operations_per_second'] = concurrency.get('operations_per_second', 0)
            summary['concurrent_success_rate'] = concurrency.get('success_rate', 0)
        
        # Overall health score (0-100)
        health_factors = []
        
        # Database health
        db_basic = results.get('database_basic', {})
        if db_basic.get('single_insert_success') and db_basic.get('data_integrity'):
            health_factors.append(100)
        elif db_basic.get('single_insert_success'):
            health_factors.append(75)
        else:
            health_factors.append(25)
        
        # Vector store health
        vector_basic = results.get('vector_basic', {})
        if vector_basic.get('single_add_success') and vector_basic.get('search_accuracy'):
            health_factors.append(100)
        elif vector_basic.get('single_add_success'):
            health_factors.append(75)
        else:
            health_factors.append(25)
        
        # Concurrency health
        if concurrency and concurrency.get('success_rate', 0) > 0.9:
            health_factors.append(100)
        elif concurrency and concurrency.get('success_rate', 0) > 0.5:
            health_factors.append(75)
        else:
            health_factors.append(25)
        
        if health_factors:
            summary['overall_health_score'] = sum(health_factors) / len(health_factors)
        
    except Exception as e:
        logger.error(f"Error calculating performance summary: {e}")
        summary['calculation_error'] = str(e)
    
    return summary

def generate_performance_report(results: Dict, output_dir: str):
    """Generate performance report."""
    report_path = os.path.join(output_dir, "performance_report.json")
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    
    try:
        # Save detailed JSON report
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        with open(summary_path, 'w') as f:
            f.write("RAG SYSTEM PERFORMANCE TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Start: {results.get('test_start_time', 'Unknown')}\n")
            f.write(f"Test End: {results.get('test_end_time', 'Unknown')}\n")
            f.write(f"Python Version: {results.get('python_version', 'Unknown')}\n\n")
            
            # Performance summary
            summary = results.get('performance_summary', {})
            f.write("PERFORMANCE SUMMARY:\n")
            f.write(f"  Overall Health Score: {summary.get('overall_health_score', 0):.1f}/100\n")
            f.write(f"  Max Database Insert Rate: {summary.get('max_database_insert_rate', 0):.2f} records/sec\n")
            f.write(f"  Max Vector Add Rate: {summary.get('max_vector_add_rate', 0):.2f} vectors/sec\n")
            f.write(f"  Concurrent Operations/sec: {summary.get('concurrent_operations_per_second', 0):.2f}\n")
            f.write(f"  Concurrent Success Rate: {summary.get('concurrent_success_rate', 0):.2%}\n\n")
            
            # Test results summary
            f.write("TEST RESULTS SUMMARY:\n")
            
            # Database tests
            db_basic = results.get('database_basic', {})
            f.write(f"  Database Basic Operations: {'PASS' if db_basic.get('single_insert_success') else 'FAIL'}\n")
            if 'error' in db_basic:
                f.write(f"    Error: {db_basic['error']}\n")
            
            vector_basic = results.get('vector_basic', {})
            f.write(f"  Vector Store Basic Operations: {'PASS' if vector_basic.get('single_add_success') else 'FAIL'}\n")
            if 'error' in vector_basic:
                f.write(f"    Error: {vector_basic['error']}\n")
            
            concurrency = results.get('concurrency', {})
            f.write(f"  Concurrent Access: {'PASS' if concurrency.get('success_rate', 0) > 0.9 else 'PARTIAL' if concurrency.get('success_rate', 0) > 0.5 else 'FAIL'}\n")
            if 'error' in concurrency:
                f.write(f"    Error: {concurrency['error']}\n")
            
            f.write("\n")
            
            # Optimization recommendations
            f.write("OPTIMIZATION RECOMMENDATIONS:\n")
            
            max_insert_rate = summary.get('max_database_insert_rate', 0)
            if max_insert_rate < 20:
                f.write("  - Database: Performance is low, check for locking issues or add connection pooling\n")
            elif max_insert_rate < 50:
                f.write("  - Database: Consider adding connection pooling for better insert performance\n")
            
            max_vector_rate = summary.get('max_vector_add_rate', 0)
            if max_vector_rate < 50:
                f.write("  - Vector Store: Consider batch operations for better performance\n")
            
            concurrent_success = summary.get('concurrent_success_rate', 0)
            if concurrent_success < 0.9:
                f.write("  - Concurrency: Consider implementing connection pooling\n")
                f.write("  - Concurrency: Check for database locking contention\n")
            
            if not any([
                max_insert_rate < 50,
                max_vector_rate < 50,
                concurrent_success < 0.9
            ]):
                f.write("  - All systems performing well!\n")
        
        # Print summary to console
        logger.info("=" * 60)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)
        
        summary = results.get('performance_summary', {})
        logger.info(f"Overall Health Score: {summary.get('overall_health_score', 0):.1f}/100")
        logger.info(f"Max Database Insert Rate: {summary.get('max_database_insert_rate', 0):.2f} records/sec")
        logger.info(f"Max Vector Add Rate: {summary.get('max_vector_add_rate', 0):.2f} vectors/sec")
        logger.info(f"Concurrent Success Rate: {summary.get('concurrent_success_rate', 0):.2%}")
        
        logger.info(f"\nReports generated:")
        logger.info(f"  Detailed: {report_path}")
        logger.info(f"  Summary: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")

if __name__ == "__main__":
    main()