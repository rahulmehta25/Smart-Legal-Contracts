#!/usr/bin/env python3
"""
Standalone database and vector store performance testing for RAG system.
Uses only standard library components and simulates vector operations where needed.
"""

import os
import sys
import time
import logging
import sqlite3
import json
import tempfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLitePerformanceTester:
    """Test SQLite database performance directly."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.db_path = os.path.join(temp_dir, "test_arbitration.db")
        self.init_database()
        
    def init_database(self):
        """Initialize the database with the schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create the arbitration_clauses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS arbitration_clauses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT,
                    industry TEXT,
                    document_type TEXT,
                    clause_text TEXT,
                    clause_summary TEXT,
                    key_provisions TEXT,  -- JSON as TEXT
                    enforceability_score REAL,
                    risk_score REAL,
                    jurisdiction TEXT,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    date_effective TIMESTAMP,
                    vector_id TEXT UNIQUE,
                    metadata TEXT  -- JSON as TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_company_name ON arbitration_clauses(company_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_industry ON arbitration_clauses(industry)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_type ON arbitration_clauses(document_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jurisdiction ON arbitration_clauses(jurisdiction)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_company_industry ON arbitration_clauses(company_name, industry)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_score ON arbitration_clauses(risk_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_vector_id ON arbitration_clauses(vector_id)')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
        finally:
            conn.close()
    
    def generate_sample_data(self, count: int) -> List[Dict]:
        """Generate sample arbitration clause data."""
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
        
        data = []
        for i in range(count):
            clause_data = {
                'company_name': companies[i % len(companies)],
                'industry': industries[i % len(industries)],
                'document_type': doc_types[i % len(doc_types)],
                'clause_text': base_clauses[i % len(base_clauses)] + f" [Test clause {i}]",
                'clause_summary': f"Summary of arbitration clause {i}",
                'key_provisions': json.dumps([f"provision_{i}", "binding_arbitration"]),
                'enforceability_score': 0.3 + (i % 7) * 0.1,
                'risk_score': 0.2 + (i % 8) * 0.1,
                'jurisdiction': jurisdictions[i % len(jurisdictions)],
                'vector_id': f"vec_{i:06d}",
                'metadata': json.dumps({"test_id": i, "batch": "performance_test"})
            }
            data.append(clause_data)
        
        return data
    
    def test_basic_operations(self) -> Dict:
        """Test basic database CRUD operations."""
        logger.info("Testing basic database operations...")
        results = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Test single insert
            start_time = time.time()
            sample_data = self.generate_sample_data(1)[0]
            
            cursor.execute('''
                INSERT INTO arbitration_clauses 
                (company_name, industry, document_type, clause_text, clause_summary, 
                 key_provisions, enforceability_score, risk_score, jurisdiction, 
                 vector_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sample_data['company_name'], sample_data['industry'], sample_data['document_type'],
                sample_data['clause_text'], sample_data['clause_summary'], sample_data['key_provisions'],
                sample_data['enforceability_score'], sample_data['risk_score'], sample_data['jurisdiction'],
                sample_data['vector_id'], sample_data['metadata']
            ))
            
            conn.commit()
            clause_id = cursor.lastrowid
            insert_time = time.time() - start_time
            
            results['single_insert_time'] = insert_time
            results['single_insert_success'] = clause_id > 0
            
            # Test retrieval
            start_time = time.time()
            cursor.execute('SELECT * FROM arbitration_clauses WHERE id = ?', (clause_id,))
            result = cursor.fetchone()
            retrieve_time = time.time() - start_time
            
            results['single_retrieve_time'] = retrieve_time
            results['single_retrieve_success'] = result is not None
            results['data_integrity'] = (
                result and result[1] == sample_data['company_name'] and result[2] == sample_data['industry']
            )
            
            # Test search with index
            start_time = time.time()
            cursor.execute('SELECT * FROM arbitration_clauses WHERE company_name = ?', (sample_data['company_name'],))
            search_results = cursor.fetchall()
            search_time = time.time() - start_time
            
            results['indexed_search_time'] = search_time
            results['search_results_count'] = len(search_results)
            
            # Test search without index (text search)
            start_time = time.time()
            cursor.execute('SELECT * FROM arbitration_clauses WHERE clause_text LIKE ?', ('%arbitration%',))
            text_search_results = cursor.fetchall()
            text_search_time = time.time() - start_time
            
            results['text_search_time'] = text_search_time
            results['text_search_results_count'] = len(text_search_results)
            
            conn.close()
            logger.info(f"Basic operations results: {results}")
            
        except Exception as e:
            logger.error(f"Error in basic operations: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_bulk_operations(self, batch_sizes: List[int] = [10, 50, 100, 500]) -> Dict:
        """Test bulk insert and query performance."""
        logger.info("Testing bulk database operations...")
        results = {}
        
        for batch_size in batch_sizes:
            try:
                logger.info(f"Testing batch size: {batch_size}")
                
                # Generate test data
                test_data = self.generate_sample_data(batch_size)
                
                # Test bulk insert
                start_time = time.time()
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                insert_data = [
                    (d['company_name'], d['industry'], d['document_type'], d['clause_text'],
                     d['clause_summary'], d['key_provisions'], d['enforceability_score'],
                     d['risk_score'], d['jurisdiction'], d['vector_id'], d['metadata'])
                    for d in test_data
                ]
                
                cursor.executemany('''
                    INSERT INTO arbitration_clauses 
                    (company_name, industry, document_type, clause_text, clause_summary, 
                     key_provisions, enforceability_score, risk_score, jurisdiction, 
                     vector_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', insert_data)
                
                conn.commit()
                insert_time = time.time() - start_time
                
                # Test bulk query
                start_time = time.time()
                cursor.execute('SELECT COUNT(*) FROM arbitration_clauses')
                total_count = cursor.fetchone()[0]
                count_time = time.time() - start_time
                
                # Test filtered query
                start_time = time.time()
                cursor.execute('SELECT * FROM arbitration_clauses WHERE industry = ?', ('Technology',))
                filtered_results = cursor.fetchall()
                filtered_time = time.time() - start_time
                
                # Test complex query
                start_time = time.time()
                cursor.execute('''
                    SELECT industry, AVG(risk_score), COUNT(*) 
                    FROM arbitration_clauses 
                    GROUP BY industry 
                    ORDER BY AVG(risk_score) DESC
                ''')
                complex_results = cursor.fetchall()
                complex_time = time.time() - start_time
                
                conn.close()
                
                results[f'batch_{batch_size}'] = {
                    'insert_time': insert_time,
                    'insert_rate': batch_size / insert_time if insert_time > 0 else 0,
                    'count_time': count_time,
                    'filtered_query_time': filtered_time,
                    'filtered_results_count': len(filtered_results),
                    'complex_query_time': complex_time,
                    'complex_results_count': len(complex_results),
                    'total_records_after_insert': total_count
                }
                
                logger.info(f"Batch {batch_size} results: {results[f'batch_{batch_size}']}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_size}: {e}")
                results[f'batch_{batch_size}'] = {'error': str(e)}
                traceback.print_exc()
        
        return results
    
    def test_query_performance(self) -> Dict:
        """Test various query patterns and their performance."""
        logger.info("Testing query performance patterns...")
        results = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure we have data
            cursor.execute('SELECT COUNT(*) FROM arbitration_clauses')
            record_count = cursor.fetchone()[0]
            
            if record_count < 50:
                # Add some test data
                test_data = self.generate_sample_data(100)
                insert_data = [
                    (d['company_name'], d['industry'], d['document_type'], d['clause_text'],
                     d['clause_summary'], d['key_provisions'], d['enforceability_score'],
                     d['risk_score'], d['jurisdiction'], d['vector_id'], d['metadata'])
                    for d in test_data
                ]
                cursor.executemany('''
                    INSERT INTO arbitration_clauses 
                    (company_name, industry, document_type, clause_text, clause_summary, 
                     key_provisions, enforceability_score, risk_score, jurisdiction, 
                     vector_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', insert_data)
                conn.commit()
            
            # Test different query patterns
            test_queries = [
                ("Primary key lookup", "SELECT * FROM arbitration_clauses WHERE id = 1"),
                ("Indexed company search", "SELECT * FROM arbitration_clauses WHERE company_name = 'TechCorp'"),
                ("Indexed industry search", "SELECT * FROM arbitration_clauses WHERE industry = 'Technology'"),
                ("Composite index search", "SELECT * FROM arbitration_clauses WHERE company_name = 'TechCorp' AND industry = 'Technology'"),
                ("Range query on risk", "SELECT * FROM arbitration_clauses WHERE risk_score BETWEEN 0.4 AND 0.7"),
                ("Text search (no index)", "SELECT * FROM arbitration_clauses WHERE clause_text LIKE '%arbitration%'"),
                ("JSON search simulation", "SELECT * FROM arbitration_clauses WHERE key_provisions LIKE '%binding_arbitration%'"),
                ("Aggregation query", "SELECT industry, COUNT(*), AVG(risk_score) FROM arbitration_clauses GROUP BY industry"),
                ("ORDER BY query", "SELECT * FROM arbitration_clauses ORDER BY risk_score DESC LIMIT 10"),
                ("Complex JOIN simulation", "SELECT company_name, COUNT(*) as clause_count FROM arbitration_clauses GROUP BY company_name HAVING COUNT(*) > 1")
            ]
            
            for query_name, sql in test_queries:
                start_time = time.time()
                cursor.execute(sql)
                query_results = cursor.fetchall()
                query_time = time.time() - start_time
                
                results[query_name.lower().replace(' ', '_')] = {
                    'time': query_time,
                    'results_count': len(query_results),
                    'sql': sql
                }
            
            # Test query plan analysis
            cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM arbitration_clauses WHERE company_name = 'TechCorp'")
            indexed_plan = cursor.fetchall()
            
            cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM arbitration_clauses WHERE clause_text LIKE '%arbitration%'")
            text_search_plan = cursor.fetchall()
            
            results['query_plans'] = {
                'indexed_search': str(indexed_plan),
                'text_search': str(text_search_plan)
            }
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error in query performance test: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_concurrent_access(self, num_workers: int = 3, operations_per_worker: int = 10) -> Dict:
        """Test concurrent database access."""
        logger.info(f"Testing concurrent access with {num_workers} workers...")
        results = {}
        
        def worker_function(worker_id: int) -> Dict:
            worker_results = {
                'worker_id': worker_id,
                'operations_completed': 0,
                'errors': []
            }
            
            try:
                # Each worker gets its own connection
                conn = sqlite3.connect(self.db_path, timeout=10.0)  # 10 second timeout
                cursor = conn.cursor()
                
                for i in range(operations_per_worker):
                    try:
                        # Insert operation
                        cursor.execute('''
                            INSERT INTO arbitration_clauses 
                            (company_name, industry, document_type, clause_text, clause_summary, 
                             key_provisions, enforceability_score, risk_score, jurisdiction, 
                             vector_id, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            f'Worker{worker_id}Corp',
                            f'Industry{worker_id}',
                            'TOS',
                            f'Concurrent test clause from worker {worker_id}, op {i}',
                            f'Summary from worker {worker_id}',
                            json.dumps([f'worker_{worker_id}']),
                            0.5,
                            0.5,
                            'US',
                            f'concurrent_{worker_id}_{i}',
                            json.dumps({'worker': worker_id, 'operation': i})
                        ))
                        
                        # Occasional read operation
                        if i % 3 == 0:
                            cursor.execute('SELECT COUNT(*) FROM arbitration_clauses WHERE company_name = ?', 
                                         (f'Worker{worker_id}Corp',))
                            count = cursor.fetchone()[0]
                        
                        conn.commit()
                        worker_results['operations_completed'] += 1
                        
                    except Exception as e:
                        worker_results['errors'].append(f"Op {i}: {str(e)}")
                        try:
                            conn.rollback()
                        except:
                            pass
                
                conn.close()
                
            except Exception as e:
                worker_results['errors'].append(f"Worker error: {str(e)}")
            
            return worker_results
        
        try:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_function, i) for i in range(num_workers)]
                worker_results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # Analyze results
            total_operations = sum(r['operations_completed'] for r in worker_results)
            total_errors = sum(len(r['errors']) for r in worker_results)
            
            results = {
                'total_time': total_time,
                'num_workers': num_workers,
                'operations_per_worker': operations_per_worker,
                'total_operations_completed': total_operations,
                'total_errors': total_errors,
                'success_rate': total_operations / (num_workers * operations_per_worker) if (num_workers * operations_per_worker) > 0 else 0,
                'operations_per_second': total_operations / total_time if total_time > 0 else 0,
                'worker_summaries': [
                    {
                        'worker_id': r['worker_id'],
                        'completed': r['operations_completed'],
                        'errors': len(r['errors'])
                    } for r in worker_results
                ]
            }
            
            # Check final database state
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM arbitration_clauses')
            final_count = cursor.fetchone()[0]
            conn.close()
            
            results['final_database_count'] = final_count
            
        except Exception as e:
            logger.error(f"Error in concurrent access test: {e}")
            results['error'] = str(e)
            
        return results

class MockVectorStoreTest:
    """Mock vector store testing to simulate FAISS operations."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.vectors = {}  # Simple in-memory store
        self.dimension = 768
        
    def test_vector_operations(self) -> Dict:
        """Test simulated vector operations."""
        logger.info("Testing mock vector store operations...")
        results = {}
        
        try:
            import random
            import math
            
            # Simulate adding vectors
            start_time = time.time()
            for i in range(100):
                vector_id = f"vec_{i}"
                # Simulate normalized vector
                vector = [random.gauss(0, 1) for _ in range(self.dimension)]
                norm = math.sqrt(sum(x*x for x in vector))
                vector = [x/norm for x in vector]
                self.vectors[vector_id] = vector
            
            add_time = time.time() - start_time
            
            # Simulate similarity search
            query_vector = [random.gauss(0, 1) for _ in range(self.dimension)]
            norm = math.sqrt(sum(x*x for x in query_vector))
            query_vector = [x/norm for x in query_vector]
            
            start_time = time.time()
            similarities = []
            for vec_id, vec in self.vectors.items():
                # Cosine similarity
                dot_product = sum(a*b for a, b in zip(query_vector, vec))
                similarities.append((vec_id, dot_product))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            search_time = time.time() - start_time
            
            results = {
                'vectors_added': len(self.vectors),
                'add_time': add_time,
                'add_rate': len(self.vectors) / add_time if add_time > 0 else 0,
                'search_time': search_time,
                'top_5_similarities': similarities[:5],
                'dimension': self.dimension
            }
            
            # Test persistence simulation
            import pickle
            persistence_file = os.path.join(self.temp_dir, "mock_vectors.pkl")
            
            start_time = time.time()
            with open(persistence_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            save_time = time.time() - start_time
            
            # Test loading
            start_time = time.time()
            with open(persistence_file, 'rb') as f:
                loaded_vectors = pickle.load(f)
            load_time = time.time() - start_time
            
            results['save_time'] = save_time
            results['load_time'] = load_time
            results['persistence_integrity'] = len(loaded_vectors) == len(self.vectors)
            
        except Exception as e:
            logger.error(f"Error in vector operations: {e}")
            results['error'] = str(e)
            
        return results

def analyze_database_schema(db_path: str) -> Dict:
    """Analyze the database schema and indexes."""
    results = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(arbitration_clauses)")
        table_info = cursor.fetchall()
        
        # Get index info
        cursor.execute("PRAGMA index_list(arbitration_clauses)")
        index_list = cursor.fetchall()
        
        # Get database stats
        cursor.execute("SELECT COUNT(*) FROM arbitration_clauses")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT company_name) FROM arbitration_clauses")
        unique_companies = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT industry) FROM arbitration_clauses")
        unique_industries = cursor.fetchone()[0]
        
        # Get database size
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        db_size = page_count * page_size
        
        results = {
            'table_columns': len(table_info),
            'indexes_count': len(index_list),
            'total_records': total_records,
            'unique_companies': unique_companies,
            'unique_industries': unique_industries,
            'database_size_bytes': db_size,
            'database_size_mb': db_size / (1024 * 1024),
            'avg_record_size_bytes': db_size / total_records if total_records > 0 else 0
        }
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error analyzing schema: {e}")
        results['error'] = str(e)
        
    return results

def main():
    """Main testing function."""
    logger.info("Starting standalone RAG system database performance tests...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="rag_standalone_test_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    test_results = {
        'test_start_time': datetime.now().isoformat(),
        'temp_directory': temp_dir,
        'python_version': sys.version
    }
    
    try:
        # Test 1: SQLite Database Operations
        logger.info("=" * 60)
        logger.info("TESTING SQLITE DATABASE OPERATIONS")
        logger.info("=" * 60)
        
        db_tester = SQLitePerformanceTester(temp_dir)
        
        test_results['database_basic'] = db_tester.test_basic_operations()
        test_results['database_bulk'] = db_tester.test_bulk_operations()
        test_results['database_queries'] = db_tester.test_query_performance()
        test_results['database_concurrent'] = db_tester.test_concurrent_access()
        test_results['database_schema'] = analyze_database_schema(db_tester.db_path)
        
        # Test 2: Mock Vector Store Operations
        logger.info("=" * 60)
        logger.info("TESTING MOCK VECTOR STORE OPERATIONS")
        logger.info("=" * 60)
        
        vector_tester = MockVectorStoreTest(temp_dir)
        test_results['vector_mock'] = vector_tester.test_vector_operations()
        
        # Performance Analysis
        logger.info("=" * 60)
        logger.info("PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        
        test_results['performance_summary'] = calculate_performance_summary(test_results)
        
    except Exception as e:
        logger.error(f"Error in main testing: {e}")
        test_results['main_error'] = str(e)
        traceback.print_exc()
    
    finally:
        test_results['test_end_time'] = datetime.now().isoformat()
        
        # Generate reports
        generate_performance_report(test_results, temp_dir)

def calculate_performance_summary(results: Dict) -> Dict:
    """Calculate performance summary and recommendations."""
    summary = {}
    
    try:
        # Database performance metrics
        db_bulk = results.get('database_bulk', {})
        max_insert_rate = 0
        for key, value in db_bulk.items():
            if key.startswith('batch_') and isinstance(value, dict):
                rate = value.get('insert_rate', 0)
                max_insert_rate = max(max_insert_rate, rate)
        
        summary['max_database_insert_rate'] = max_insert_rate
        
        # Query performance
        db_queries = results.get('database_queries', {})
        if db_queries:
            indexed_search_time = db_queries.get('indexed_company_search', {}).get('time', 0)
            text_search_time = db_queries.get('text_search_(no_index)', {}).get('time', 0)
            
            summary['indexed_search_time'] = indexed_search_time
            summary['text_search_time'] = text_search_time
            summary['search_performance_ratio'] = text_search_time / indexed_search_time if indexed_search_time > 0 else 0
        
        # Concurrent performance
        concurrent = results.get('database_concurrent', {})
        if concurrent:
            summary['concurrent_success_rate'] = concurrent.get('success_rate', 0)
            summary['concurrent_ops_per_second'] = concurrent.get('operations_per_second', 0)
        
        # Vector store performance (mock)
        vector_mock = results.get('vector_mock', {})
        if vector_mock:
            summary['mock_vector_add_rate'] = vector_mock.get('add_rate', 0)
            summary['mock_vector_search_time'] = vector_mock.get('search_time', 0)
        
        # Database efficiency metrics
        schema = results.get('database_schema', {})
        if schema:
            summary['database_size_mb'] = schema.get('database_size_mb', 0)
            summary['records_per_mb'] = schema.get('total_records', 0) / schema.get('database_size_mb', 1) if schema.get('database_size_mb', 0) > 0 else 0
        
        # Overall health score
        health_factors = []
        
        # Database health
        if max_insert_rate > 100:
            health_factors.append(100)
        elif max_insert_rate > 50:
            health_factors.append(80)
        elif max_insert_rate > 10:
            health_factors.append(60)
        else:
            health_factors.append(30)
        
        # Concurrent health
        success_rate = concurrent.get('success_rate', 0) if concurrent else 0
        if success_rate > 0.95:
            health_factors.append(100)
        elif success_rate > 0.8:
            health_factors.append(80)
        elif success_rate > 0.5:
            health_factors.append(60)
        else:
            health_factors.append(30)
        
        if health_factors:
            summary['overall_health_score'] = sum(health_factors) / len(health_factors)
        
    except Exception as e:
        logger.error(f"Error calculating performance summary: {e}")
        summary['calculation_error'] = str(e)
        
    return summary

def generate_performance_report(results: Dict, output_dir: str):
    """Generate comprehensive performance report."""
    try:
        # Save JSON report
        json_report_path = os.path.join(output_dir, "standalone_performance_report.json")
        with open(json_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate text summary
        summary_path = os.path.join(output_dir, "standalone_performance_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("RAG SYSTEM STANDALONE PERFORMANCE TEST REPORT\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Test Period: {results.get('test_start_time', 'Unknown')} to {results.get('test_end_time', 'Unknown')}\n")
            f.write(f"Python Version: {results.get('python_version', 'Unknown')}\n")
            f.write(f"Test Directory: {results.get('temp_directory', 'Unknown')}\n\n")
            
            summary = results.get('performance_summary', {})
            f.write("PERFORMANCE SUMMARY:\n")
            f.write(f"  Overall Health Score: {summary.get('overall_health_score', 0):.1f}/100\n")
            f.write(f"  Max Database Insert Rate: {summary.get('max_database_insert_rate', 0):.1f} records/sec\n")
            f.write(f"  Indexed Search Time: {summary.get('indexed_search_time', 0):.4f} seconds\n")
            f.write(f"  Text Search Time: {summary.get('text_search_time', 0):.4f} seconds\n")
            f.write(f"  Search Performance Ratio: {summary.get('search_performance_ratio', 0):.1f}x slower for text search\n")
            f.write(f"  Concurrent Success Rate: {summary.get('concurrent_success_rate', 0):.2%}\n")
            f.write(f"  Concurrent Ops/Second: {summary.get('concurrent_ops_per_second', 0):.1f}\n\n")
            
            # Database analysis
            schema = results.get('database_schema', {})
            f.write("DATABASE ANALYSIS:\n")
            f.write(f"  Total Records: {schema.get('total_records', 0)}\n")
            f.write(f"  Database Size: {schema.get('database_size_mb', 0):.2f} MB\n")
            f.write(f"  Records per MB: {summary.get('records_per_mb', 0):.0f}\n")
            f.write(f"  Indexes Count: {schema.get('indexes_count', 0)}\n")
            f.write(f"  Unique Companies: {schema.get('unique_companies', 0)}\n")
            f.write(f"  Unique Industries: {schema.get('unique_industries', 0)}\n\n")
            
            # Test results overview
            f.write("TEST RESULTS:\n")
            
            db_basic = results.get('database_basic', {})
            f.write(f"  Basic Operations: {'PASS' if db_basic.get('single_insert_success') and db_basic.get('data_integrity') else 'FAIL'}\n")
            
            bulk_tests = results.get('database_bulk', {})
            bulk_success = any(
                isinstance(v, dict) and v.get('insert_rate', 0) > 0 
                for k, v in bulk_tests.items() 
                if k.startswith('batch_')
            )
            f.write(f"  Bulk Operations: {'PASS' if bulk_success else 'FAIL'}\n")
            
            concurrent = results.get('database_concurrent', {})
            f.write(f"  Concurrent Access: {'PASS' if concurrent.get('success_rate', 0) > 0.8 else 'FAIL'}\n")
            
            vector_mock = results.get('vector_mock', {})
            f.write(f"  Mock Vector Store: {'PASS' if vector_mock.get('vectors_added', 0) > 0 else 'FAIL'}\n\n")
            
            # Recommendations
            f.write("OPTIMIZATION RECOMMENDATIONS:\n")
            
            max_insert_rate = summary.get('max_database_insert_rate', 0)
            if max_insert_rate < 50:
                f.write("  - Database: Insert performance is low. Consider using transactions and batch operations\n")
            
            search_ratio = summary.get('search_performance_ratio', 0)
            if search_ratio > 50:
                f.write("  - Database: Text search is very slow. Consider implementing full-text search indexes\n")
            elif search_ratio > 10:
                f.write("  - Database: Consider adding text search indexes for better performance\n")
            
            concurrent_success = summary.get('concurrent_success_rate', 0)
            if concurrent_success < 0.9:
                f.write("  - Concurrency: Consider increasing SQLite timeout settings\n")
                f.write("  - Concurrency: Implement retry logic for locked database scenarios\n")
            
            db_size_mb = summary.get('database_size_mb', 0)
            if db_size_mb > 100:
                f.write("  - Database: Consider database partitioning for large datasets\n")
            
            if all([max_insert_rate >= 50, search_ratio <= 10, concurrent_success >= 0.9]):
                f.write("  - All database operations performing well!\n")
                f.write("  - Ready for production use with real FAISS vector store\n")
        
        # Print summary to console
        logger.info("=" * 60)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)
        
        summary = results.get('performance_summary', {})
        logger.info(f"Overall Health Score: {summary.get('overall_health_score', 0):.1f}/100")
        logger.info(f"Max Database Insert Rate: {summary.get('max_database_insert_rate', 0):.1f} records/sec")
        logger.info(f"Concurrent Success Rate: {summary.get('concurrent_success_rate', 0):.2%}")
        logger.info(f"Database Size: {results.get('database_schema', {}).get('database_size_mb', 0):.2f} MB")
        
        logger.info(f"\nDetailed reports generated:")
        logger.info(f"  JSON Report: {json_report_path}")
        logger.info(f"  Summary Report: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")

if __name__ == "__main__":
    main()