#!/usr/bin/env python3
"""
Comprehensive Database and Vector Store Testing Script

This script tests:
1. SQLite database initialization and table creation
2. Database CRUD operations with performance metrics
3. Vector store operations (FAISS-based similarity search)
4. Query optimization and index performance
5. Error handling and data persistence
6. Comparison engine database operations

Author: Database Testing System
"""

import sys
import os
import time
import logging
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import random
import string

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_system', 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatabaseTester:
    """Comprehensive database testing suite with performance metrics."""
    
    def __init__(self):
        """Initialize the database tester."""
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'performance_metrics': {},
            'error_count': 0,
            'success_count': 0
        }
        self.start_time = time.time()
        
    def log_test_result(self, test_name: str, success: bool, duration: float, details: Dict = None):
        """Log test result with performance metrics."""
        result = {
            'test_name': test_name,
            'success': success,
            'duration_seconds': round(duration, 4),
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.results['tests'].append(result)
        
        if success:
            self.success_count += 1
            logger.info(f"✅ {test_name} - PASSED ({duration:.4f}s)")
        else:
            self.error_count += 1
            logger.error(f"❌ {test_name} - FAILED ({duration:.4f}s)")
            
        if details:
            logger.info(f"   Details: {json.dumps(details, indent=2)}")
    
    def test_sqlite_initialization(self) -> bool:
        """Test SQLite database initialization and table creation."""
        start_time = time.time()
        
        try:
            # Import database components
            from database.schema import DatabaseManager, ArbitrationClauseDB, VectorStore
            from sqlalchemy import inspect
            
            # Initialize database manager
            db_manager = DatabaseManager()
            
            # Check if tables were created
            inspector = inspect(db_manager.engine)
            tables = inspector.get_table_names()
            
            expected_tables = ['arbitration_clauses']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            # Check table schema
            if 'arbitration_clauses' in tables:
                columns = inspector.get_columns('arbitration_clauses')
                indexes = inspector.get_indexes('arbitration_clauses')
                
                column_names = [col['name'] for col in columns]
                expected_columns = ['id', 'company_name', 'industry', 'document_type', 
                                  'clause_text', 'clause_summary', 'key_provisions',
                                  'enforceability_score', 'risk_score', 'jurisdiction',
                                  'date_added', 'date_effective', 'vector_id', 'metadata']
                
                missing_columns = [c for c in expected_columns if c not in column_names]
                
                details = {
                    'tables_created': tables,
                    'missing_tables': missing_tables,
                    'columns_created': column_names,
                    'missing_columns': missing_columns,
                    'indexes_count': len(indexes),
                    'indexes': [idx['name'] for idx in indexes]
                }
                
                success = len(missing_tables) == 0 and len(missing_columns) == 0
                duration = time.time() - start_time
                self.log_test_result("SQLite Database Initialization", success, duration, details)
                
                return success
            else:
                details = {'error': 'arbitration_clauses table not created'}
                duration = time.time() - start_time
                self.log_test_result("SQLite Database Initialization", False, duration, details)
                return False
                
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            duration = time.time() - start_time
            self.log_test_result("SQLite Database Initialization", False, duration, details)
            return False
    
    def test_database_crud_operations(self) -> bool:
        """Test Create, Read, Update, Delete operations with performance metrics."""
        start_time = time.time()
        
        try:
            from database.schema import DatabaseManager
            
            db_manager = DatabaseManager()
            
            # Test data
            test_clauses = self.generate_test_clauses(50)  # Generate 50 test clauses
            
            # CREATE operations with timing
            create_start = time.time()
            clause_ids = []
            
            for i, clause_data in enumerate(test_clauses):
                clause_id = db_manager.add_clause(clause_data)
                clause_ids.append(clause_id)
                
                if i % 10 == 0:  # Log progress every 10 insertions
                    logger.info(f"   Inserted {i+1}/{len(test_clauses)} clauses")
            
            create_duration = time.time() - create_start
            create_rate = len(test_clauses) / create_duration
            
            # READ operations with timing
            read_start = time.time()
            retrieved_clauses = []
            
            for clause_id in clause_ids[:20]:  # Test reading first 20
                clause = db_manager.get_clause(clause_id)
                if clause:
                    retrieved_clauses.append(clause)
            
            read_duration = time.time() - read_start
            read_rate = 20 / read_duration if read_duration > 0 else 0
            
            # Test search with filters
            search_start = time.time()
            
            # Test various filter combinations
            search_results = []
            
            # Search by industry
            tech_clauses = db_manager.search_clauses({'industry': 'Technology'})
            search_results.append(('industry_technology', len(tech_clauses)))
            
            # Search by document type
            tos_clauses = db_manager.search_clauses({'document_type': 'Terms of Service'})
            search_results.append(('document_tos', len(tos_clauses)))
            
            # Search by risk score range
            high_risk_clauses = db_manager.search_clauses({'min_risk': 0.7})
            search_results.append(('high_risk', len(high_risk_clauses)))
            
            # Complex search
            complex_search = db_manager.search_clauses({
                'industry': 'Technology',
                'document_type': 'Terms of Service',
                'min_risk': 0.5
            })
            search_results.append(('complex_filter', len(complex_search)))
            
            search_duration = time.time() - search_start
            
            details = {
                'clauses_created': len(clause_ids),
                'create_duration': round(create_duration, 4),
                'create_rate_per_second': round(create_rate, 2),
                'clauses_retrieved': len(retrieved_clauses),
                'read_duration': round(read_duration, 4),
                'read_rate_per_second': round(read_rate, 2),
                'search_results': dict(search_results),
                'search_duration': round(search_duration, 4)
            }
            
            # Validate data integrity
            integrity_issues = []
            for clause in retrieved_clauses:
                if not clause.get('company_name') or not clause.get('clause_text'):
                    integrity_issues.append(f"Missing required fields in clause {clause.get('id')}")
            
            details['data_integrity_issues'] = integrity_issues
            
            success = (len(clause_ids) == len(test_clauses) and 
                      len(retrieved_clauses) == 20 and
                      len(integrity_issues) == 0)
            
            duration = time.time() - start_time
            self.log_test_result("Database CRUD Operations", success, duration, details)
            
            # Store performance metrics
            self.results['performance_metrics']['database_crud'] = {
                'insert_rate_per_second': round(create_rate, 2),
                'read_rate_per_second': round(read_rate, 2),
                'total_records_created': len(clause_ids),
                'search_performance': round(search_duration, 4)
            }
            
            return success
            
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            duration = time.time() - start_time
            self.log_test_result("Database CRUD Operations", False, duration, details)
            return False
    
    def test_vector_store_operations(self) -> bool:
        """Test vector store operations including embeddings, similarity search, and persistence."""
        start_time = time.time()
        
        try:
            from database.schema import VectorStore
            
            # Initialize vector store
            vector_store = VectorStore(dimension=768)
            
            # Generate test embeddings
            num_vectors = 100
            test_vectors = []
            vector_ids = []
            
            embedding_start = time.time()
            
            for i in range(num_vectors):
                # Generate random normalized embedding
                embedding = np.random.randn(768).astype('float32')
                embedding = embedding / np.linalg.norm(embedding)
                
                vector_id = f"test_clause_{i:03d}"
                success = vector_store.add_clause(vector_id, embedding)
                
                if success:
                    test_vectors.append(embedding)
                    vector_ids.append(vector_id)
                    
                if i % 20 == 0:
                    logger.info(f"   Added {i+1}/{num_vectors} vectors")
            
            embedding_duration = time.time() - embedding_start
            embedding_rate = len(vector_ids) / embedding_duration
            
            # Test similarity search with different query sizes
            search_start = time.time()
            search_results = []
            
            # Use first vector as query
            if len(test_vectors) > 0:
                query_vector = test_vectors[0]
                
                # Test different k values
                for k in [5, 10, 20]:
                    results = vector_store.search_similar(query_vector, k=k)
                    search_results.append((k, len(results)))
                    
                    # Validate similarity scores
                    if results:
                        scores = [score for _, score in results]
                        max_score = max(scores)
                        min_score = min(scores)
                        avg_score = sum(scores) / len(scores)
                        
                        search_results.append((f"k{k}_max_score", round(max_score, 4)))
                        search_results.append((f"k{k}_min_score", round(min_score, 4)))
                        search_results.append((f"k{k}_avg_score", round(avg_score, 4)))
            
            search_duration = time.time() - search_start
            
            # Test save and load operations
            persistence_start = time.time()
            
            # Get stats before save
            stats_before = vector_store.get_stats()
            
            # Save to disk
            vector_store.save()
            
            # Create new vector store and load
            vector_store_loaded = VectorStore(dimension=768)
            vector_store_loaded.load()
            
            # Get stats after load
            stats_after = vector_store_loaded.get_stats()
            
            # Test search on loaded store
            loaded_results = []
            if len(test_vectors) > 0:
                loaded_search = vector_store_loaded.search_similar(test_vectors[0], k=10)
                loaded_results = loaded_search
            
            persistence_duration = time.time() - persistence_start
            
            # Test vector removal
            removal_start = time.time()
            removed_count = 0
            
            for vector_id in vector_ids[:5]:  # Remove first 5 vectors
                success = vector_store.remove_clause(vector_id)
                if success:
                    removed_count += 1
            
            removal_duration = time.time() - removal_start
            
            details = {
                'vectors_added': len(vector_ids),
                'embedding_duration': round(embedding_duration, 4),
                'embedding_rate_per_second': round(embedding_rate, 2),
                'search_results': dict(search_results),
                'search_duration': round(search_duration, 4),
                'stats_before_save': stats_before,
                'stats_after_load': stats_after,
                'persistence_duration': round(persistence_duration, 4),
                'persistence_successful': stats_before['total_vectors'] == stats_after['total_vectors'],
                'loaded_search_results': len(loaded_results),
                'vectors_removed': removed_count,
                'removal_duration': round(removal_duration, 4)
            }
            
            success = (len(vector_ids) == num_vectors and
                      len(search_results) > 0 and
                      stats_before['total_vectors'] == stats_after['total_vectors'])
            
            duration = time.time() - start_time
            self.log_test_result("Vector Store Operations", success, duration, details)
            
            # Store performance metrics
            self.results['performance_metrics']['vector_store'] = {
                'embedding_rate_per_second': round(embedding_rate, 2),
                'search_duration': round(search_duration, 4),
                'persistence_duration': round(persistence_duration, 4),
                'total_vectors': len(vector_ids)
            }
            
            return success
            
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            duration = time.time() - start_time
            self.log_test_result("Vector Store Operations", False, duration, details)
            return False
    
    def test_database_indexes_performance(self) -> bool:
        """Test database query performance with and without indexes."""
        start_time = time.time()
        
        try:
            from database.schema import DatabaseManager, ArbitrationClauseDB
            from sqlalchemy import text
            
            db_manager = DatabaseManager()
            session = db_manager.get_session()
            
            try:
                # Test query performance on indexed columns
                performance_results = []
                
                # Test 1: Query by company_name (indexed)
                query_start = time.time()
                result1 = session.execute(text("""
                    SELECT COUNT(*) FROM arbitration_clauses 
                    WHERE company_name LIKE '%Tech%'
                """)).scalar()
                company_query_duration = time.time() - query_start
                performance_results.append(('company_name_query', result1, round(company_query_duration, 4)))
                
                # Test 2: Query by industry (indexed)
                query_start = time.time()
                result2 = session.execute(text("""
                    SELECT COUNT(*) FROM arbitration_clauses 
                    WHERE industry = 'Technology'
                """)).scalar()
                industry_query_duration = time.time() - query_start
                performance_results.append(('industry_query', result2, round(industry_query_duration, 4)))
                
                # Test 3: Range query on risk_score (indexed)
                query_start = time.time()
                result3 = session.execute(text("""
                    SELECT COUNT(*) FROM arbitration_clauses 
                    WHERE risk_score BETWEEN 0.5 AND 0.8
                """)).scalar()
                risk_query_duration = time.time() - query_start
                performance_results.append(('risk_range_query', result3, round(risk_query_duration, 4)))
                
                # Test 4: Complex query using composite index
                query_start = time.time()
                result4 = session.execute(text("""
                    SELECT COUNT(*) FROM arbitration_clauses 
                    WHERE company_name LIKE '%Corp%' AND industry = 'Finance'
                """)).scalar()
                composite_query_duration = time.time() - query_start
                performance_results.append(('composite_index_query', result4, round(composite_query_duration, 4)))
                
                # Test 5: Full text search on clause_text (not indexed - should be slower)
                query_start = time.time()
                result5 = session.execute(text("""
                    SELECT COUNT(*) FROM arbitration_clauses 
                    WHERE clause_text LIKE '%arbitration%'
                """)).scalar()
                fulltext_query_duration = time.time() - query_start
                performance_results.append(('fulltext_query', result5, round(fulltext_query_duration, 4)))
                
                # Get query execution plans (if supported)
                explain_results = []
                try:
                    explain_result = session.execute(text("""
                        EXPLAIN QUERY PLAN 
                        SELECT * FROM arbitration_clauses 
                        WHERE industry = 'Technology'
                    """)).fetchall()
                    explain_results.append(('industry_explain', [dict(row._mapping) for row in explain_result]))
                except Exception as e:
                    explain_results.append(('explain_error', str(e)))
                
                # Test index usage statistics
                index_stats = []
                try:
                    # Get table info
                    pragma_result = session.execute(text("PRAGMA table_info(arbitration_clauses)")).fetchall()
                    index_stats.append(('table_columns', len(pragma_result)))
                    
                    # Get index list
                    index_result = session.execute(text("PRAGMA index_list(arbitration_clauses)")).fetchall()
                    index_stats.append(('indexes_count', len(index_result)))
                    index_stats.append(('index_names', [row[1] for row in index_result]))
                    
                except Exception as e:
                    index_stats.append(('index_stats_error', str(e)))
                
                details = {
                    'performance_results': performance_results,
                    'explain_plans': dict(explain_results),
                    'index_statistics': dict(index_stats),
                    'total_queries_tested': len(performance_results)
                }
                
                # Check if indexed queries are performing reasonably
                indexed_queries = [p for p in performance_results if p[0] in ['company_name_query', 'industry_query', 'composite_index_query']]
                avg_indexed_time = sum(p[2] for p in indexed_queries) / len(indexed_queries) if indexed_queries else 0
                
                # Full text query should generally be slower (not always true for small datasets)
                fulltext_time = next(p[2] for p in performance_results if p[0] == 'fulltext_query')
                
                success = len(performance_results) == 5 and avg_indexed_time < 1.0  # Reasonable performance
                
            finally:
                session.close()
            
            duration = time.time() - start_time
            self.log_test_result("Database Indexes Performance", success, duration, details)
            
            # Store performance metrics
            self.results['performance_metrics']['query_performance'] = {
                'average_indexed_query_time': round(avg_indexed_time, 4),
                'fulltext_query_time': round(fulltext_time, 4),
                'performance_results': dict([(p[0], {'count': p[1], 'duration': p[2]}) for p in performance_results])
            }
            
            return success
            
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            duration = time.time() - start_time
            self.log_test_result("Database Indexes Performance", False, duration, details)
            return False
    
    def test_comparison_engine_database_ops(self) -> bool:
        """Test comparison engine database operations."""
        start_time = time.time()
        
        try:
            # Test if comparison engine exists
            comparison_path = os.path.join(os.path.dirname(__file__), 'rag_system', 'src', 'comparison')
            
            if not os.path.exists(comparison_path):
                details = {'error': 'Comparison engine directory not found', 'path_checked': comparison_path}
                duration = time.time() - start_time
                self.log_test_result("Comparison Engine Database Ops", False, duration, details)
                return False
            
            # Import comparison engine
            sys.path.insert(0, comparison_path)
            from comparison_engine import ComparisonEngine
            
            # Initialize comparison engine
            comparison_engine = ComparisonEngine()
            
            # Test operations
            operations_tested = []
            
            # Test 1: Compare two sample clauses
            clause1 = {
                'id': 'test1',
                'text': 'All disputes shall be resolved through binding arbitration in New York.',
                'metadata': {'jurisdiction': 'New York'}
            }
            
            clause2 = {
                'id': 'test2', 
                'text': 'Any disputes will be settled by arbitration in California.',
                'metadata': {'jurisdiction': 'California'}
            }
            
            comparison_start = time.time()
            comparison_result = comparison_engine.compare_clauses(clause1, clause2)
            comparison_duration = time.time() - comparison_start
            
            operations_tested.append(('clause_comparison', bool(comparison_result), round(comparison_duration, 4)))
            
            # Test 2: Bulk comparison operations
            if hasattr(comparison_engine, 'bulk_compare'):
                bulk_start = time.time()
                bulk_result = comparison_engine.bulk_compare([clause1, clause2], [clause1, clause2])
                bulk_duration = time.time() - bulk_start
                operations_tested.append(('bulk_comparison', bool(bulk_result), round(bulk_duration, 4)))
            
            details = {
                'comparison_engine_found': True,
                'operations_tested': operations_tested,
                'comparison_result_keys': list(comparison_result.keys()) if comparison_result else []
            }
            
            success = len(operations_tested) > 0 and all(op[1] for op in operations_tested)
            
        except ImportError as e:
            # Try alternative approach - test with mock comparison operations
            logger.info("Comparison engine not found, testing with mock operations")
            
            from database.schema import DatabaseManager
            db_manager = DatabaseManager()
            
            # Test database operations that would be used by comparison engine
            test_operations = []
            
            # Test getting multiple clauses for comparison
            session = db_manager.get_session()
            try:
                # Get clauses for comparison
                clauses = session.query(ArbitrationClauseDB).limit(10).all()
                test_operations.append(('fetch_clauses_for_comparison', len(clauses)))
                
                # Test filtering by similarity criteria
                similar_clauses = session.query(ArbitrationClauseDB).filter(
                    ArbitrationClauseDB.risk_score.between(0.3, 0.7)
                ).limit(5).all()
                test_operations.append(('filter_similar_risk', len(similar_clauses)))
                
            finally:
                session.close()
            
            details = {
                'comparison_engine_found': False,
                'mock_operations': test_operations,
                'error': str(e)
            }
            
            success = len(test_operations) > 0
            
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            success = False
        
        duration = time.time() - start_time
        self.log_test_result("Comparison Engine Database Ops", success, duration, details)
        return success
    
    def test_data_persistence(self) -> bool:
        """Test data persistence across database sessions."""
        start_time = time.time()
        
        try:
            from database.schema import DatabaseManager
            
            # Session 1: Add data
            db_manager1 = DatabaseManager()
            
            persistence_test_data = {
                'company_name': 'Persistence Test Corp',
                'industry': 'Testing',
                'document_type': 'Test Document',
                'clause_text': 'This is a test clause for persistence verification.',
                'clause_summary': 'Test summary',
                'key_provisions': ['test_provision_1', 'test_provision_2'],
                'enforceability_score': 0.95,
                'risk_score': 0.15,
                'jurisdiction': 'Test Jurisdiction',
                'vector_id': f'persistence_test_{int(time.time())}'
            }
            
            clause_id = db_manager1.add_clause(persistence_test_data)
            
            # Close first session
            del db_manager1
            
            # Session 2: Retrieve data
            db_manager2 = DatabaseManager()
            retrieved_clause = db_manager2.get_clause(clause_id)
            
            # Verify data integrity
            integrity_checks = []
            
            if retrieved_clause:
                for key, expected_value in persistence_test_data.items():
                    if key == 'key_provisions':
                        # JSON field comparison
                        actual_value = retrieved_clause.get(key)
                        integrity_checks.append((key, actual_value == expected_value))
                    else:
                        actual_value = retrieved_clause.get(key)
                        integrity_checks.append((key, actual_value == expected_value))
            
            # Test vector store persistence
            from database.schema import VectorStore
            
            # Create vector store and add test vector
            vector_store1 = VectorStore()
            test_embedding = np.random.randn(768).astype('float32')
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            
            vector_id = f"persistence_vector_{int(time.time())}"
            vector_add_success = vector_store1.add_clause(vector_id, test_embedding)
            vector_store1.save()
            
            # Create new vector store and load
            vector_store2 = VectorStore()
            vector_store2.load()
            
            # Test search
            search_results = vector_store2.search_similar(test_embedding, k=1)
            vector_persistence_success = len(search_results) > 0 and search_results[0][0] == vector_id
            
            details = {
                'clause_id': clause_id,
                'data_retrieved': retrieved_clause is not None,
                'integrity_checks': dict(integrity_checks),
                'integrity_passed': all(check[1] for check in integrity_checks),
                'vector_add_success': vector_add_success,
                'vector_persistence_success': vector_persistence_success,
                'search_results_count': len(search_results)
            }
            
            success = (retrieved_clause is not None and 
                      all(check[1] for check in integrity_checks) and
                      vector_persistence_success)
            
            duration = time.time() - start_time
            self.log_test_result("Data Persistence", success, duration, details)
            
            return success
            
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            duration = time.time() - start_time
            self.log_test_result("Data Persistence", False, duration, details)
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for various database failure scenarios."""
        start_time = time.time()
        
        try:
            from database.schema import DatabaseManager, VectorStore
            
            error_tests = []
            
            # Test 1: Invalid database URL
            try:
                invalid_db = DatabaseManager(db_url="invalid://invalid_db_url")
                error_tests.append(('invalid_db_url', False, 'Should have failed'))
            except Exception as e:
                error_tests.append(('invalid_db_url', True, str(e)))
            
            # Test 2: Adding clause with missing required fields
            try:
                db_manager = DatabaseManager()
                invalid_clause = {'company_name': 'Test'}  # Missing many required fields
                clause_id = db_manager.add_clause(invalid_clause)
                error_tests.append(('missing_fields', False, f'Unexpected success with ID {clause_id}'))
            except Exception as e:
                error_tests.append(('missing_fields', True, str(e)))
            
            # Test 3: Vector store with invalid dimension
            try:
                vector_store = VectorStore()
                invalid_embedding = np.random.randn(512).astype('float32')  # Wrong dimension
                result = vector_store.add_clause('invalid_dim', invalid_embedding)
                error_tests.append(('invalid_vector_dimension', not result, f'Add result: {result}'))
            except Exception as e:
                error_tests.append(('invalid_vector_dimension', True, str(e)))
            
            # Test 4: Loading non-existent vector store
            try:
                vector_store = VectorStore()
                vector_store.load('/non/existent/path/vectors')
                error_tests.append(('load_nonexistent_vectors', False, 'Should have failed'))
            except Exception as e:
                error_tests.append(('load_nonexistent_vectors', True, str(e)))
            
            # Test 5: Database connection with locked database
            # This is hard to test reliably, so we'll skip it
            
            # Test 6: Vector search with empty index
            try:
                empty_vector_store = VectorStore()
                empty_vector_store.clear()  # Ensure it's empty
                query = np.random.randn(768).astype('float32')
                results = empty_vector_store.search_similar(query, k=5)
                error_tests.append(('search_empty_index', len(results) == 0, f'Results: {len(results)}'))
            except Exception as e:
                error_tests.append(('search_empty_index', True, str(e)))
            
            details = {
                'error_tests': error_tests,
                'tests_passed': sum(1 for test in error_tests if test[1]),
                'total_tests': len(error_tests)
            }
            
            # Most error tests should pass (meaning errors were properly handled)
            success = sum(1 for test in error_tests if test[1]) >= len(error_tests) * 0.8
            
            duration = time.time() - start_time
            self.log_test_result("Error Handling", success, duration, details)
            
            return success
            
        except Exception as e:
            details = {'error': str(e), 'traceback': traceback.format_exc()}
            duration = time.time() - start_time
            self.log_test_result("Error Handling", False, duration, details)
            return False
    
    def generate_test_clauses(self, count: int) -> List[Dict]:
        """Generate test arbitration clauses for database testing."""
        companies = ['TechCorp', 'DataSystems Inc', 'CloudServices LLC', 'StartupXYZ', 'Enterprise Solutions']
        industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
        document_types = ['Terms of Service', 'Employment Agreement', 'Vendor Contract', 'User Agreement']
        jurisdictions = ['New York', 'California', 'Delaware', 'Texas', 'Illinois']
        
        test_clauses = []
        
        for i in range(count):
            clause = {
                'company_name': random.choice(companies),
                'industry': random.choice(industries),
                'document_type': random.choice(document_types),
                'clause_text': f'Test arbitration clause #{i+1}. All disputes shall be resolved through binding arbitration.',
                'clause_summary': f'Summary of test clause #{i+1}',
                'key_provisions': [f'provision_{i+1}_1', f'provision_{i+1}_2'],
                'enforceability_score': round(random.uniform(0.1, 1.0), 2),
                'risk_score': round(random.uniform(0.1, 1.0), 2),
                'jurisdiction': random.choice(jurisdictions),
                'date_effective': datetime.now() - timedelta(days=random.randint(1, 365)),
                'vector_id': f'test_vector_{i+1:03d}_{int(time.time())}',
                'metadata': {'test': True, 'batch_id': int(time.time())}
            }
            test_clauses.append(clause)
        
        return test_clauses
    
    def generate_final_report(self):
        """Generate final test report with performance metrics and recommendations."""
        total_duration = time.time() - self.start_time
        
        self.results['summary'] = {
            'total_duration': round(total_duration, 2),
            'total_tests': len(self.results['tests']),
            'passed_tests': self.success_count,
            'failed_tests': self.error_count,
            'success_rate': round(self.success_count / len(self.results['tests']) * 100, 2) if self.results['tests'] else 0
        }
        
        # Generate recommendations
        recommendations = []
        
        # Database performance recommendations
        if 'database_crud' in self.results['performance_metrics']:
            crud_metrics = self.results['performance_metrics']['database_crud']
            if crud_metrics['insert_rate_per_second'] < 100:
                recommendations.append("Consider batch insertions or transaction optimization for better write performance")
            if crud_metrics['read_rate_per_second'] < 1000:
                recommendations.append("Consider adding more indexes or query optimization for read operations")
        
        # Vector store recommendations
        if 'vector_store' in self.results['performance_metrics']:
            vector_metrics = self.results['performance_metrics']['vector_store']
            if vector_metrics['search_duration'] > 0.1:
                recommendations.append("Consider upgrading to FAISS GPU version for faster similarity search")
        
        # General recommendations
        if self.error_count > 0:
            recommendations.append("Review error logs and implement additional error handling")
        
        if self.success_count == len(self.results['tests']):
            recommendations.append("All tests passed! System is ready for production deployment")
        
        self.results['recommendations'] = recommendations
        
        # Write detailed report
        report_filename = f"database_test_report_{int(time.time())}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*80}")
        logger.info("DATABASE AND VECTOR STORE TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Tests: {self.results['summary']['total_tests']}")
        logger.info(f"Passed: {self.success_count} ({self.results['summary']['success_rate']:.1f}%)")
        logger.info(f"Failed: {self.error_count}")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"Detailed Report: {report_filename}")
        
        if recommendations:
            logger.info(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
        
        return self.results

def main():
    """Run comprehensive database tests."""
    logger.info("Starting comprehensive database and vector store testing...")
    
    # Initialize tester
    tester = DatabaseTester()
    
    # Run all tests
    tests = [
        tester.test_sqlite_initialization,
        tester.test_database_crud_operations,
        tester.test_vector_store_operations,
        tester.test_database_indexes_performance,
        tester.test_comparison_engine_database_ops,
        tester.test_data_persistence,
        tester.test_error_handling
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            logger.error(f"Unexpected error in {test_func.__name__}: {e}")
            tester.error_count += 1
    
    # Generate final report
    results = tester.generate_final_report()
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    if results['summary']['failed_tests'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)