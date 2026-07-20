#!/usr/bin/env python3
"""
Comprehensive performance testing for RAG system database and vector store operations.
Tests database operations, vector store performance, comparison engine, and concurrent access.
"""

import os
import sys
import time
import logging
import traceback
import threading
import multiprocessing
import psutil
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import sqlite3
import json
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.schema import DatabaseManager, VectorStore, ArbitrationClauseDB
from comparison.comparison_engine import ClauseComparisonEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/rag_performance_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance during testing."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory
        self.start_time = time.time()
        
    def log_memory_usage(self, operation: str):
        """Log current memory usage."""
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
        memory_mb = current_memory / 1024 / 1024
        logger.info(f"Memory usage after {operation}: {memory_mb:.2f} MB")
        return memory_mb
        
    def get_memory_delta(self):
        """Get memory increase since start."""
        current_memory = self.process.memory_info().rss
        delta_mb = (current_memory - self.start_memory) / 1024 / 1024
        return delta_mb
        
    def get_elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time

class DatabaseTester:
    """Comprehensive database testing."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.db_path = os.path.join(temp_dir, "test_arbitration.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.db_manager = DatabaseManager(self.db_url)
        self.monitor = PerformanceMonitor()
        
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
            "All claims shall be arbitrated in accordance with the Federal Arbitration Act. You have 30 days to opt out of this arbitration provision.",
            "Binding arbitration shall be the exclusive remedy for all disputes. The arbitrator's decision shall be final and non-appealable.",
            "Any controversy arising out of this contract shall be settled by arbitration under UNCITRAL rules with the seat in New York.",
            "Disputes must be resolved through individual arbitration. Class action waivers apply and jury trial rights are waived.",
            "The parties agree to submit all disputes to binding arbitration administered by LCIA under English law.",
            "Mandatory arbitration applies to all claims. Proceedings shall be confidential and conducted by a single arbitrator."
        ]
        
        for i in range(count):
            company = companies[i % len(companies)]
            industry = industries[i % len(industries)]
            doc_type = doc_types[i % len(doc_types)]
            jurisdiction = jurisdictions[i % len(jurisdictions)]
            base_text = base_clauses[i % len(base_clauses)]
            
            # Add variation to the text
            clause_text = f"{base_text} This clause was added on {datetime.now().strftime('%Y-%m-%d')} for {company}."
            
            clause = {
                "company": company,
                "industry": industry,
                "document_type": doc_type,
                "clause_text": clause_text,
                "summary": clause_text[:100] + "...",
                "key_provisions": [f"binding_arbitration", f"{doc_type.lower()}_specific"],
                "enforceability": np.random.uniform(0.3, 0.9),
                "risk_score": np.random.uniform(0.2, 0.8),
                "jurisdiction": jurisdiction,
                "metadata": {"test_id": i, "generated": True}
            }
            sample_clauses.append(clause)
            
        return sample_clauses
    
    def test_basic_database_operations(self) -> Dict:
        """Test basic CRUD operations."""
        logger.info("Testing basic database operations...")
        results = {}
        
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
            
            self.monitor.log_memory_usage("single insert")
            
            # Test retrieval
            start_time = time.time()
            retrieved_clause = self.db_manager.get_clause(clause_id)
            retrieve_time = time.time() - start_time
            
            results['single_retrieve_time'] = retrieve_time
            results['single_retrieve_success'] = retrieved_clause is not None
            results['data_integrity'] = (
                retrieved_clause['company_name'] == sample_clause['company'] and
                retrieved_clause['industry'] == sample_clause['industry']
            )
            
            # Test search with filters
            start_time = time.time()
            search_results = self.db_manager.search_clauses({
                'company_name': sample_clause['company'],
                'industry': sample_clause['industry']
            })
            search_time = time.time() - start_time
            
            results['search_time'] = search_time
            results['search_success'] = len(search_results) > 0
            
            logger.info(f"Basic operations completed: {results}")
            
        except Exception as e:
            logger.error(f"Error in basic database operations: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_bulk_operations(self, batch_sizes: List[int] = [10, 50, 100, 500]) -> Dict:
        """Test bulk insert and query operations."""
        logger.info("Testing bulk database operations...")
        results = {}
        
        for batch_size in batch_sizes:
            try:
                logger.info(f"Testing batch size: {batch_size}")
                
                # Generate test data
                sample_clauses = self.generate_sample_clauses(batch_size)
                
                # Test bulk insert
                start_time = time.time()
                start_memory = self.monitor.process.memory_info().rss
                
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
                end_memory = self.monitor.process.memory_info().rss
                memory_used = (end_memory - start_memory) / 1024 / 1024
                
                results[f'batch_{batch_size}'] = {
                    'insert_time': insert_time,
                    'insert_rate': len(inserted_ids) / insert_time if insert_time > 0 else 0,
                    'success_rate': len(inserted_ids) / batch_size,
                    'memory_used_mb': memory_used,
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
        
        try:
            # Add test data if needed
            sample_clauses = self.generate_sample_clauses(100)
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
                {'company_name': 'TechCorp', 'industry': 'Technology'},
                {'min_risk': 0.5},
                {'max_risk': 0.7},
                {'min_risk': 0.3, 'max_risk': 0.8}
            ]
            
            for i, query in enumerate(test_queries):
                start_time = time.time()
                search_results = self.db_manager.search_clauses(query)
                query_time = time.time() - start_time
                
                results[f'query_{i}'] = {
                    'query': query,
                    'time': query_time,
                    'results_count': len(search_results)
                }
                
            # Test EXPLAIN QUERY PLAN for SQLite
            session = self.db_manager.get_session()
            try:
                # Get the raw connection to execute SQLite-specific commands
                raw_connection = session.connection().connection
                cursor = raw_connection.cursor()
                
                # Test query plan for indexed query
                cursor.execute("""
                    EXPLAIN QUERY PLAN 
                    SELECT * FROM arbitration_clauses 
                    WHERE company_name = 'TechCorp' AND industry = 'Technology'
                """)
                
                query_plan = cursor.fetchall()
                results['query_plan_indexed'] = [dict(zip([d[0] for d in cursor.description], row)) for row in query_plan]
                
                # Test query plan for non-indexed query
                cursor.execute("""
                    EXPLAIN QUERY PLAN 
                    SELECT * FROM arbitration_clauses 
                    WHERE clause_text LIKE '%arbitration%'
                """)
                
                query_plan = cursor.fetchall()
                results['query_plan_text_search'] = [dict(zip([d[0] for d in cursor.description], row)) for row in query_plan]
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error testing index performance: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_data_integrity(self) -> Dict:
        """Test data integrity and consistency."""
        logger.info("Testing data integrity...")
        results = {}
        
        try:
            # Test with various data types and edge cases
            test_cases = [
                {
                    'name': 'unicode_text',
                    'clause': {
                        'company_name': 'Test™ Company',
                        'industry': 'Tech & Finance',
                        'document_type': 'TOS',
                        'clause_text': 'Arbitration clause with unicode: © ® ™ éñ 中文',
                        'clause_summary': 'Unicode test',
                        'key_provisions': ['unicode', 'special_chars'],
                        'enforceability_score': 0.75,
                        'risk_score': 0.5,
                        'jurisdiction': 'US',
                        'vector_id': 'unicode_test',
                        'metadata': {'test': 'unicode', 'special': '™©®'}
                    }
                },
                {
                    'name': 'long_text',
                    'clause': {
                        'company_name': 'Long Text Corp',
                        'industry': 'Testing',
                        'document_type': 'Contract',
                        'clause_text': 'A' * 10000,  # Very long text
                        'clause_summary': 'Long text test',
                        'key_provisions': ['long_text'] * 100,  # Large JSON array
                        'enforceability_score': 0.9,
                        'risk_score': 0.1,
                        'jurisdiction': 'EU',
                        'vector_id': 'long_text_test',
                        'metadata': {'length': 10000, 'type': 'stress_test'}
                    }
                },
                {
                    'name': 'edge_values',
                    'clause': {
                        'company_name': '',  # Empty string
                        'industry': None,  # None value
                        'document_type': 'Edge',
                        'clause_text': 'Edge case test',
                        'clause_summary': None,
                        'key_provisions': [],  # Empty array
                        'enforceability_score': 0.0,  # Min value
                        'risk_score': 1.0,  # Max value
                        'jurisdiction': 'XX',
                        'vector_id': 'edge_test',
                        'metadata': {}  # Empty object
                    }
                }
            ]
            
            for test_case in test_cases:
                try:
                    # Insert test case
                    clause_id = self.db_manager.add_clause(test_case['clause'])
                    
                    # Retrieve and verify
                    retrieved = self.db_manager.get_clause(clause_id)
                    
                    # Check data integrity
                    integrity_checks = {
                        'id_matches': retrieved['id'] == clause_id,
                        'text_preserved': retrieved['clause_text'] == test_case['clause']['clause_text'],
                        'json_preserved': retrieved['key_provisions'] == test_case['clause']['key_provisions'],
                        'metadata_preserved': retrieved['metadata'] == test_case['clause']['metadata'],
                        'scores_preserved': (
                            abs(retrieved['enforceability_score'] - test_case['clause']['enforceability_score']) < 0.001 and
                            abs(retrieved['risk_score'] - test_case['clause']['risk_score']) < 0.001
                        )
                    }
                    
                    results[test_case['name']] = {
                        'insert_success': clause_id > 0,
                        'retrieve_success': retrieved is not None,
                        'integrity_checks': integrity_checks,
                        'all_checks_passed': all(integrity_checks.values())
                    }
                    
                except Exception as e:
                    results[test_case['name']] = {'error': str(e)}
                    logger.error(f"Error in {test_case['name']}: {e}")
            
        except Exception as e:
            logger.error(f"Error in data integrity test: {e}")
            results['error'] = str(e)
            
        return results

class VectorStoreTester:
    """Comprehensive vector store testing."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.vector_store = VectorStore(dimension=768)
        self.monitor = PerformanceMonitor()
        
    def generate_test_embeddings(self, count: int, dimension: int = 768) -> List[Tuple[str, np.ndarray]]:
        """Generate test embeddings."""
        embeddings = []
        for i in range(count):
            # Generate random normalized embedding
            vector = np.random.randn(dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize
            embeddings.append((f"test_clause_{i}", vector))
        return embeddings
    
    def test_basic_vector_operations(self) -> Dict:
        """Test basic vector store operations."""
        logger.info("Testing basic vector store operations...")
        results = {}
        
        try:
            # Clear store for clean test
            self.vector_store.clear()
            
            # Test single vector addition
            start_time = time.time()
            test_embeddings = self.generate_test_embeddings(1)
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
            
            self.monitor.log_memory_usage("basic vector operations")
            
        except Exception as e:
            logger.error(f"Error in basic vector operations: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_bulk_vector_operations(self, sizes: List[int] = [10, 50, 100, 500, 1000]) -> Dict:
        """Test bulk vector operations."""
        logger.info("Testing bulk vector operations...")
        results = {}
        
        for size in sizes:
            try:
                logger.info(f"Testing vector bulk size: {size}")
                
                # Clear store
                self.vector_store.clear()
                
                # Generate test embeddings
                test_embeddings = self.generate_test_embeddings(size)
                
                # Test bulk addition
                start_time = time.time()
                start_memory = self.monitor.process.memory_info().rss
                
                success_count = 0
                for clause_id, embedding in test_embeddings:
                    if self.vector_store.add_clause(clause_id, embedding):
                        success_count += 1
                
                add_time = time.time() - start_time
                end_memory = self.monitor.process.memory_info().rss
                memory_used = (end_memory - start_memory) / 1024 / 1024
                
                # Test bulk search
                query_embedding = test_embeddings[0][1]  # Use first embedding as query
                
                # Test different k values
                search_times = {}
                for k in [1, 5, 10, min(50, size)]:
                    if k <= size:
                        start_time = time.time()
                        search_results = self.vector_store.search_similar(query_embedding, k=k)
                        search_time = time.time() - start_time
                        search_times[f'k_{k}'] = {
                            'time': search_time,
                            'results_count': len(search_results),
                            'top_similarity': search_results[0][1] if search_results else 0.0
                        }
                
                results[f'size_{size}'] = {
                    'add_time': add_time,
                    'add_rate': success_count / add_time if add_time > 0 else 0,
                    'success_rate': success_count / size,
                    'memory_used_mb': memory_used,
                    'search_times': search_times,
                    'final_vector_count': self.vector_store.get_stats()['total_vectors']
                }
                
                logger.info(f"Size {size}: {results[f'size_{size}']}")
                
            except Exception as e:
                logger.error(f"Error in bulk size {size}: {e}")
                results[f'size_{size}'] = {'error': str(e)}
        
        return results
    
    def test_similarity_accuracy(self) -> Dict:
        """Test similarity search accuracy."""
        logger.info("Testing similarity search accuracy...")
        results = {}
        
        try:
            # Clear store
            self.vector_store.clear()
            
            # Create embeddings with known similarities
            base_embedding = np.random.randn(768).astype(np.float32)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            # Add base embedding
            self.vector_store.add_clause("base", base_embedding)
            
            # Create similar and dissimilar embeddings
            similar_embeddings = []
            dissimilar_embeddings = []
            
            # Similar embeddings (add small noise)
            for i in range(5):
                noise = np.random.randn(768).astype(np.float32) * 0.1
                similar_emb = base_embedding + noise
                similar_emb = similar_emb / np.linalg.norm(similar_emb)
                similar_embeddings.append(similar_emb)
                self.vector_store.add_clause(f"similar_{i}", similar_emb)
            
            # Dissimilar embeddings (random)
            for i in range(5):
                dissimilar_emb = np.random.randn(768).astype(np.float32)
                dissimilar_emb = dissimilar_emb / np.linalg.norm(dissimilar_emb)
                dissimilar_embeddings.append(dissimilar_emb)
                self.vector_store.add_clause(f"dissimilar_{i}", dissimilar_emb)
            
            # Test similarity search
            search_results = self.vector_store.search_similar(base_embedding, k=11)
            
            # Analyze results
            similar_count = sum(1 for clause_id, _ in search_results[:6] if clause_id.startswith('similar') or clause_id == 'base')
            dissimilar_count = sum(1 for clause_id, _ in search_results[:6] if clause_id.startswith('dissimilar'))
            
            results['total_results'] = len(search_results)
            results['similar_in_top6'] = similar_count
            results['dissimilar_in_top6'] = dissimilar_count
            results['accuracy'] = similar_count / 6 if similar_count > 0 else 0
            results['similarity_scores'] = [(clause_id, float(score)) for clause_id, score in search_results]
            
            # Test cosine similarity manually
            manual_similarities = []
            for i, emb in enumerate(similar_embeddings):
                similarity = np.dot(base_embedding, emb)
                manual_similarities.append(similarity)
            
            results['manual_similarities'] = manual_similarities
            results['average_manual_similarity'] = np.mean(manual_similarities)
            
        except Exception as e:
            logger.error(f"Error in similarity accuracy test: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_persistence(self) -> Dict:
        """Test vector store persistence."""
        logger.info("Testing vector store persistence...")
        results = {}
        
        try:
            # Clear and add test data
            self.vector_store.clear()
            test_embeddings = self.generate_test_embeddings(50)
            
            for clause_id, embedding in test_embeddings:
                self.vector_store.add_clause(clause_id, embedding)
            
            original_stats = self.vector_store.get_stats()
            
            # Test save
            start_time = time.time()
            save_path = os.path.join(self.temp_dir, "test_vectors")
            self.vector_store.save(save_path)
            save_time = time.time() - start_time
            
            results['save_time'] = save_time
            results['save_files_exist'] = (
                os.path.exists(f"{save_path}.faiss") and
                os.path.exists(f"{save_path}.map")
            )
            
            # Test load with new instance
            start_time = time.time()
            new_vector_store = VectorStore(dimension=768)
            new_vector_store.load(save_path)
            load_time = time.time() - start_time
            
            loaded_stats = new_vector_store.get_stats()
            
            results['load_time'] = load_time
            results['stats_match'] = original_stats == loaded_stats
            results['original_stats'] = original_stats
            results['loaded_stats'] = loaded_stats
            
            # Test search consistency
            query_embedding = test_embeddings[0][1]
            original_results = self.vector_store.search_similar(query_embedding, k=5)
            loaded_results = new_vector_store.search_similar(query_embedding, k=5)
            
            results['search_consistency'] = len(original_results) == len(loaded_results)
            results['top_result_match'] = (
                len(original_results) > 0 and len(loaded_results) > 0 and
                original_results[0][0] == loaded_results[0][0]
            )
            
        except Exception as e:
            logger.error(f"Error in persistence test: {e}")
            results['error'] = str(e)
            
        return results

class ComparisonEngineTester:
    """Test the comparison engine functionality."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.db_path = os.path.join(temp_dir, "comparison_test.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.monitor = PerformanceMonitor()
        
        try:
            # Initialize comparison engine with test database
            self.comparison_engine = ClauseComparisonEngine(self.db_url)
        except Exception as e:
            logger.warning(f"Could not initialize comparison engine: {e}")
            self.comparison_engine = None
    
    def test_clause_addition_and_comparison(self) -> Dict:
        """Test adding clauses and performing comparisons."""
        logger.info("Testing clause addition and comparison...")
        results = {}
        
        if not self.comparison_engine:
            results['error'] = "Comparison engine not initialized"
            return results
        
        try:
            # Add sample clauses to database
            sample_clauses = [
                {
                    "company": "TechCorp",
                    "industry": "Technology",
                    "document_type": "TOS",
                    "clause_text": "Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration in accordance with the Commercial Arbitration Rules of the American Arbitration Association.",
                    "summary": "Binding arbitration with AAA rules",
                    "key_provisions": ["binding_arbitration", "aaa_rules"],
                    "enforceability": 0.85,
                    "risk_score": 0.6,
                    "jurisdiction": "US"
                },
                {
                    "company": "FinanceInc",
                    "industry": "Finance", 
                    "document_type": "Service",
                    "clause_text": "The parties agree that all disputes must be submitted to final and binding arbitration under JAMS rules. You waive any right to a class action or jury trial.",
                    "summary": "JAMS arbitration with class action waiver",
                    "key_provisions": ["binding_arbitration", "jams_rules", "class_action_waiver"],
                    "enforceability": 0.75,
                    "risk_score": 0.8,
                    "jurisdiction": "US"
                },
                {
                    "company": "HealthCare LLC",
                    "industry": "Healthcare",
                    "document_type": "Employment",
                    "clause_text": "Disputes will be resolved by mandatory arbitration on an individual basis only. No class actions are permitted under this agreement.",
                    "summary": "Mandatory individual arbitration",
                    "key_provisions": ["mandatory_arbitration", "individual_basis", "no_class_action"],
                    "enforceability": 0.9,
                    "risk_score": 0.7,
                    "jurisdiction": "US"
                }
            ]
            
            # Add clauses to database
            start_time = time.time()
            added_clauses = []
            for i, clause in enumerate(sample_clauses):
                try:
                    clause_id = self.comparison_engine.add_clause_to_database(clause)
                    added_clauses.append(clause_id)
                except Exception as e:
                    logger.error(f"Error adding clause {i}: {e}")
            
            add_time = time.time() - start_time
            
            results['clauses_added'] = len(added_clauses)
            results['add_time'] = add_time
            results['add_success_rate'] = len(added_clauses) / len(sample_clauses)
            
            # Test comparison with new clause
            test_clause = "All claims and disputes arising under this agreement shall be resolved by binding arbitration administered by the American Arbitration Association."
            
            start_time = time.time()
            comparison_result = self.comparison_engine.compare_clause(test_clause, top_k=5)
            comparison_time = time.time() - start_time
            
            results['comparison_time'] = comparison_time
            results['comparison_success'] = 'similar_clauses' in comparison_result
            results['similar_clauses_found'] = len(comparison_result.get('similar_clauses', []))
            results['analysis_provided'] = 'analysis' in comparison_result
            results['statistics_provided'] = 'statistics' in comparison_result
            
            # Analyze comparison quality
            if comparison_result.get('similar_clauses'):
                similarities = [clause['similarity'] for clause in comparison_result['similar_clauses']]
                results['similarity_scores'] = similarities
                results['avg_similarity'] = np.mean(similarities)
                results['max_similarity'] = max(similarities)
                results['min_similarity'] = min(similarities)
            
            self.monitor.log_memory_usage("comparison test")
            
        except Exception as e:
            logger.error(f"Error in comparison test: {e}")
            results['error'] = str(e)
            traceback.print_exc()
            
        return results
    
    def test_bulk_import(self) -> Dict:
        """Test bulk import functionality."""
        logger.info("Testing bulk import...")
        results = {}
        
        if not self.comparison_engine:
            results['error'] = "Comparison engine not initialized"
            return results
        
        try:
            # Generate test clauses for bulk import
            companies = ["Corp1", "Corp2", "Corp3", "Corp4", "Corp5"]
            industries = ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing"]
            
            bulk_clauses = []
            for i in range(20):
                clause = {
                    "company": companies[i % len(companies)],
                    "industry": industries[i % len(industries)],
                    "document_type": "TOS",
                    "clause_text": f"Test arbitration clause number {i}. This clause requires binding arbitration for all disputes.",
                    "summary": f"Test clause {i}",
                    "key_provisions": ["binding_arbitration", "test_clause"],
                    "enforceability": np.random.uniform(0.3, 0.9),
                    "risk_score": np.random.uniform(0.2, 0.8),
                    "jurisdiction": "US"
                }
                bulk_clauses.append(clause)
            
            # Test bulk import
            start_time = time.time()
            import_result = self.comparison_engine.bulk_import_clauses(bulk_clauses)
            import_time = time.time() - start_time
            
            results['import_time'] = import_time
            results['import_result'] = import_result
            results['success_rate'] = import_result['success'] / import_result['total']
            results['import_rate'] = import_result['total'] / import_time if import_time > 0 else 0
            
            # Test database stats after import
            stats = self.comparison_engine.get_database_stats()
            results['database_stats'] = stats
            
        except Exception as e:
            logger.error(f"Error in bulk import test: {e}")
            results['error'] = str(e)
            
        return results

class ConcurrencyTester:
    """Test concurrent access to database and vector store."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.db_path = os.path.join(temp_dir, "concurrency_test.db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.monitor = PerformanceMonitor()
        
    def worker_function(self, worker_id: int, operation_count: int) -> Dict:
        """Worker function for concurrent testing."""
        results = {
            'worker_id': worker_id,
            'operations_completed': 0,
            'errors': [],
            'start_time': time.time()
        }
        
        try:
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
                    if i % 10 == 0:
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
    
    def test_concurrent_database_access(self, num_workers: int = 5, operations_per_worker: int = 20) -> Dict:
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
            results['worker_results'] = worker_results
            
            # Check database consistency
            db_manager = DatabaseManager(self.db_url)
            final_count = len(db_manager.search_clauses({}))
            results['final_database_count'] = final_count
            results['data_consistency'] = final_count == total_operations
            
            self.monitor.log_memory_usage("concurrent test")
            
        except Exception as e:
            logger.error(f"Error in concurrent test: {e}")
            results['error'] = str(e)
            
        return results

def main():
    """Main testing function."""
    logger.info("Starting comprehensive RAG system performance tests...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    all_results = {
        'test_start_time': datetime.now().isoformat(),
        'temp_directory': temp_dir
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
        all_results['database_integrity'] = db_tester.test_data_integrity()
        
        # Test 2: Vector Store Operations
        logger.info("=" * 60)
        logger.info("TESTING VECTOR STORE OPERATIONS")
        logger.info("=" * 60)
        
        vector_tester = VectorStoreTester(temp_dir)
        
        all_results['vector_basic'] = vector_tester.test_basic_vector_operations()
        all_results['vector_bulk'] = vector_tester.test_bulk_vector_operations()
        all_results['vector_similarity'] = vector_tester.test_similarity_accuracy()
        all_results['vector_persistence'] = vector_tester.test_persistence()
        
        # Test 3: Comparison Engine
        logger.info("=" * 60)
        logger.info("TESTING COMPARISON ENGINE")
        logger.info("=" * 60)
        
        comparison_tester = ComparisonEngineTester(temp_dir)
        
        all_results['comparison_basic'] = comparison_tester.test_clause_addition_and_comparison()
        all_results['comparison_bulk'] = comparison_tester.test_bulk_import()
        
        # Test 4: Concurrent Access
        logger.info("=" * 60)
        logger.info("TESTING CONCURRENT ACCESS")
        logger.info("=" * 60)
        
        concurrency_tester = ConcurrencyTester(temp_dir)
        all_results['concurrency'] = concurrency_tester.test_concurrent_database_access()
        
        # Test 5: Memory and Performance Analysis
        logger.info("=" * 60)
        logger.info("MEMORY AND PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        
        # Get system information
        all_results['system_info'] = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # Calculate overall performance metrics
        all_results['performance_summary'] = calculate_performance_summary(all_results)
        
    except Exception as e:
        logger.error(f"Error in main testing: {e}")
        all_results['main_error'] = str(e)
        traceback.print_exc()
    
    finally:
        all_results['test_end_time'] = datetime.now().isoformat()
        
        # Generate performance report
        generate_performance_report(all_results, temp_dir)
        
        # Cleanup (optional - keep for debugging)
        # shutil.rmtree(temp_dir)
        logger.info(f"Test files preserved in: {temp_dir}")

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
        
        # Similarity accuracy
        vector_similarity = results.get('vector_similarity', {})
        if vector_similarity:
            summary['similarity_accuracy'] = vector_similarity.get('accuracy', 0)
        
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
        else:
            health_factors.append(50)
        
        # Vector store health
        vector_basic = results.get('vector_basic', {})
        if vector_basic.get('single_add_success') and vector_basic.get('search_accuracy'):
            health_factors.append(100)
        else:
            health_factors.append(50)
        
        # Comparison engine health
        comparison_basic = results.get('comparison_basic', {})
        if comparison_basic.get('comparison_success'):
            health_factors.append(100)
        else:
            health_factors.append(50)
        
        if health_factors:
            summary['overall_health_score'] = sum(health_factors) / len(health_factors)
        
    except Exception as e:
        logger.error(f"Error calculating performance summary: {e}")
        summary['calculation_error'] = str(e)
    
    return summary

def generate_performance_report(results: Dict, output_dir: str):
    """Generate comprehensive performance report."""
    report_path = os.path.join(output_dir, "performance_report.json")
    
    try:
        # Save detailed JSON report
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = os.path.join(output_dir, "performance_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("RAG SYSTEM PERFORMANCE TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Start: {results.get('test_start_time', 'Unknown')}\n")
            f.write(f"Test End: {results.get('test_end_time', 'Unknown')}\n\n")
            
            # System info
            sys_info = results.get('system_info', {})
            f.write("SYSTEM INFORMATION:\n")
            f.write(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"  Memory: {sys_info.get('memory_gb', 0):.2f} GB\n")
            f.write(f"  Platform: {sys_info.get('platform', 'Unknown')}\n\n")
            
            # Performance summary
            summary = results.get('performance_summary', {})
            f.write("PERFORMANCE SUMMARY:\n")
            f.write(f"  Overall Health Score: {summary.get('overall_health_score', 0):.1f}/100\n")
            f.write(f"  Max Database Insert Rate: {summary.get('max_database_insert_rate', 0):.2f} records/sec\n")
            f.write(f"  Max Vector Add Rate: {summary.get('max_vector_add_rate', 0):.2f} vectors/sec\n")
            f.write(f"  Similarity Search Accuracy: {summary.get('similarity_accuracy', 0):.2%}\n")
            f.write(f"  Concurrent Operations/sec: {summary.get('concurrent_operations_per_second', 0):.2f}\n")
            f.write(f"  Concurrent Success Rate: {summary.get('concurrent_success_rate', 0):.2%}\n\n")
            
            # Test results summary
            f.write("TEST RESULTS SUMMARY:\n")
            
            # Database tests
            db_basic = results.get('database_basic', {})
            f.write(f"  Database Basic Operations: {'PASS' if db_basic.get('single_insert_success') else 'FAIL'}\n")
            
            vector_basic = results.get('vector_basic', {})
            f.write(f"  Vector Store Basic Operations: {'PASS' if vector_basic.get('single_add_success') else 'FAIL'}\n")
            
            comparison_basic = results.get('comparison_basic', {})
            f.write(f"  Comparison Engine: {'PASS' if comparison_basic.get('comparison_success') else 'FAIL'}\n")
            
            concurrency = results.get('concurrency', {})
            f.write(f"  Concurrent Access: {'PASS' if concurrency.get('success_rate', 0) > 0.9 else 'FAIL'}\n\n")
            
            # Recommendations
            f.write("OPTIMIZATION RECOMMENDATIONS:\n")
            
            # Database recommendations
            db_bulk = results.get('database_bulk', {})
            max_insert_rate = summary.get('max_database_insert_rate', 0)
            if max_insert_rate < 50:
                f.write("  - Database: Consider adding connection pooling for better insert performance\n")
            if max_insert_rate < 10:
                f.write("  - Database: Performance is very low, check for locking issues\n")
            
            # Vector store recommendations
            max_vector_rate = summary.get('max_vector_add_rate', 0)
            if max_vector_rate < 100:
                f.write("  - Vector Store: Consider batch operations for better performance\n")
            
            similarity_accuracy = summary.get('similarity_accuracy', 0)
            if similarity_accuracy < 0.8:
                f.write("  - Vector Store: Similarity accuracy is low, check embedding quality\n")
            
            # Concurrency recommendations
            concurrent_success = summary.get('concurrent_success_rate', 0)
            if concurrent_success < 0.95:
                f.write("  - Concurrency: Consider implementing connection pooling\n")
                f.write("  - Concurrency: Check for database locking contention\n")
            
            if not any([
                max_insert_rate < 50,
                max_vector_rate < 100,
                similarity_accuracy < 0.8,
                concurrent_success < 0.95
            ]):
                f.write("  - All systems performing well! No immediate optimizations needed.\n")
        
        logger.info(f"Performance reports generated:")
        logger.info(f"  Detailed: {report_path}")
        logger.info(f"  Summary: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")

if __name__ == "__main__":
    main()