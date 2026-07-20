#!/usr/bin/env python3
"""
Comprehensive debug and test script for RAG system integration.

This script tests:
1. Import functionality
2. RAGIntegration class initialization and methods
3. Backward compatibility with RAGWrapper
4. Integration with existing backend structure
5. Error handling and logging
6. Database connections
7. Circular dependencies check
"""

import os
import sys
import traceback
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'errors': [],
    'warnings': []
}

def log_test_result(test_name: str, status: str, message: str = "", error: str = ""):
    """Log test result and add to tracking."""
    result = {
        'test': test_name,
        'status': status,
        'message': message,
        'error': error
    }
    
    test_results[status].append(result)
    
    if status == 'passed':
        logger.info(f"✅ {test_name}: {message}")
    elif status == 'failed':
        logger.error(f"❌ {test_name}: {message}")
        if error:
            logger.error(f"   Error: {error}")
    elif status == 'warnings':
        logger.warning(f"⚠️  {test_name}: {message}")
    else:  # errors
        logger.error(f"💥 {test_name}: {message}")
        if error:
            logger.error(f"   Error: {error}")

def test_imports():
    """Test 1: Basic imports and path resolution."""
    print("\n" + "="*60)
    print("TEST 1: Testing imports and path resolution")
    print("="*60)
    
    try:
        # Test Python path
        current_dir = Path(__file__).parent
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Python path: {sys.path}")
        
        # Try importing integration_rag
        try:
            import integration_rag
            log_test_result('import_integration_rag', 'passed', 'Integration module imported successfully')
            
            # Check if RAG system path is accessible
            rag_path = current_dir / "rag_system" / "src"
            logger.info(f"RAG system path: {rag_path}")
            logger.info(f"RAG system exists: {rag_path.exists()}")
            
            if rag_path.exists():
                log_test_result('rag_system_path', 'passed', f'RAG system path exists: {rag_path}')
            else:
                log_test_result('rag_system_path', 'failed', f'RAG system path not found: {rag_path}')
            
        except ImportError as e:
            log_test_result('import_integration_rag', 'failed', 'Failed to import integration_rag', str(e))
            return False
            
    except Exception as e:
        log_test_result('test_imports', 'errors', 'Import test failed', str(e))
        return False
    
    return True

def test_rag_integration_class():
    """Test 2: RAGIntegration class functionality."""
    print("\n" + "="*60)
    print("TEST 2: Testing RAGIntegration class")
    print("="*60)
    
    try:
        from integration_rag import RAGIntegration
        
        # Test initialization
        try:
            rag = RAGIntegration()
            log_test_result('rag_init_default', 'passed', 'RAGIntegration initialized with default config')
        except Exception as e:
            log_test_result('rag_init_default', 'failed', 'Failed to initialize RAGIntegration', str(e))
            return False
        
        # Test initialization with config
        try:
            config = {
                'cache_enabled': False,
                'database_url': 'sqlite:///test.db'
            }
            rag_config = RAGIntegration(config)
            log_test_result('rag_init_config', 'passed', 'RAGIntegration initialized with custom config')
        except Exception as e:
            log_test_result('rag_init_config', 'failed', 'Failed to initialize RAGIntegration with config', str(e))
        
        # Test method availability
        methods = ['detect_arbitration', 'detect_from_text', 'compare_clause', 'get_database_stats']
        for method in methods:
            if hasattr(rag, method):
                log_test_result(f'method_{method}', 'passed', f'Method {method} exists')
            else:
                log_test_result(f'method_{method}', 'failed', f'Method {method} missing')
        
        return True
        
    except ImportError as e:
        log_test_result('rag_class_import', 'failed', 'Failed to import RAGIntegration', str(e))
        return False
    except Exception as e:
        log_test_result('rag_class_test', 'errors', 'RAGIntegration class test failed', str(e))
        return False

def test_detect_methods():
    """Test 3: Detection method functionality."""
    print("\n" + "="*60)
    print("TEST 3: Testing detection methods")
    print("="*60)
    
    try:
        from integration_rag import RAGIntegration
        rag = RAGIntegration()
        
        # Test detect_from_text with sample text
        sample_text = """
        Any dispute arising from this agreement shall be resolved through binding arbitration
        in accordance with the rules of the American Arbitration Association.
        """
        
        try:
            result = rag.detect_from_text(sample_text)
            log_test_result('detect_from_text', 'passed', f'detect_from_text returned: {type(result)}')
            logger.info(f"Detection result keys: {result.keys()}")
            
            # Check result structure
            expected_keys = ['detected', 'confidence']
            for key in expected_keys:
                if key in result:
                    log_test_result(f'result_key_{key}', 'passed', f'Result contains {key}: {result.get(key)}')
                else:
                    log_test_result(f'result_key_{key}', 'warnings', f'Result missing expected key: {key}')
                    
        except Exception as e:
            log_test_result('detect_from_text', 'failed', 'detect_from_text method failed', str(e))
        
        # Test detect_arbitration with temporary file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(sample_text)
                tmp_path = tmp_file.name
            
            try:
                result = rag.detect_arbitration(tmp_path)
                log_test_result('detect_arbitration', 'passed', f'detect_arbitration returned: {type(result)}')
                logger.info(f"File detection result keys: {result.keys()}")
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            log_test_result('detect_arbitration', 'failed', 'detect_arbitration method failed', str(e))
        
        return True
        
    except Exception as e:
        log_test_result('detect_methods', 'errors', 'Detection methods test failed', str(e))
        return False

def test_backward_compatibility():
    """Test 4: RAGWrapper backward compatibility."""
    print("\n" + "="*60)
    print("TEST 4: Testing RAGWrapper backward compatibility")
    print("="*60)
    
    try:
        from integration_rag import RAGWrapper
        
        # Test initialization
        try:
            wrapper = RAGWrapper()
            log_test_result('wrapper_init', 'passed', 'RAGWrapper initialized successfully')
        except Exception as e:
            log_test_result('wrapper_init', 'failed', 'Failed to initialize RAGWrapper', str(e))
            return False
        
        # Test backward compatible methods
        methods = ['analyze_document', 'analyze_text']
        for method in methods:
            if hasattr(wrapper, method):
                log_test_result(f'wrapper_method_{method}', 'passed', f'Wrapper method {method} exists')
            else:
                log_test_result(f'wrapper_method_{method}', 'failed', f'Wrapper method {method} missing')
        
        # Test analyze_text functionality
        sample_text = "This agreement contains an arbitration clause."
        try:
            result = wrapper.analyze_text(sample_text)
            log_test_result('wrapper_analyze_text', 'passed', f'analyze_text returned: {type(result)}')
            
            # Check backward compatible format
            expected_keys = ['has_arbitration_clause', 'confidence_score']
            for key in expected_keys:
                if key in result:
                    log_test_result(f'wrapper_key_{key}', 'passed', f'Wrapper result contains {key}: {result.get(key)}')
                else:
                    log_test_result(f'wrapper_key_{key}', 'warnings', f'Wrapper result missing expected key: {key}')
                    
        except Exception as e:
            log_test_result('wrapper_analyze_text', 'failed', 'Wrapper analyze_text failed', str(e))
        
        return True
        
    except ImportError as e:
        log_test_result('wrapper_import', 'failed', 'Failed to import RAGWrapper', str(e))
        return False
    except Exception as e:
        log_test_result('wrapper_test', 'errors', 'RAGWrapper test failed', str(e))
        return False

def test_existing_backend_compatibility():
    """Test 5: Integration with existing backend structure."""
    print("\n" + "="*60)
    print("TEST 5: Testing existing backend compatibility")
    print("="*60)
    
    try:
        # Check if existing backend modules can be imported
        backend_modules = [
            'app.models',
            'app.api',
            'app.services',
            'app.core'
        ]
        
        for module in backend_modules:
            try:
                __import__(module)
                log_test_result(f'backend_import_{module}', 'passed', f'Successfully imported {module}')
            except ImportError as e:
                log_test_result(f'backend_import_{module}', 'warnings', f'Could not import {module}', str(e))
            except Exception as e:
                log_test_result(f'backend_import_{module}', 'failed', f'Error importing {module}', str(e))
        
        # Check for potential conflicts
        try:
            import integration_rag
            import app.rag.arbitration_detector  # Existing RAG module
            log_test_result('rag_conflict_check', 'passed', 'Both RAG systems can coexist')
        except ImportError as e:
            log_test_result('rag_conflict_check', 'warnings', 'Existing RAG module not found', str(e))
        except Exception as e:
            log_test_result('rag_conflict_check', 'failed', 'RAG conflict detected', str(e))
        
        return True
        
    except Exception as e:
        log_test_result('backend_compatibility', 'errors', 'Backend compatibility test failed', str(e))
        return False

def test_database_connections():
    """Test 6: Database connection testing."""
    print("\n" + "="*60)
    print("TEST 6: Testing database connections")
    print("="*60)
    
    try:
        from integration_rag import RAGIntegration
        
        # Test with different database configurations
        configs = [
            {'database_url': None},  # Default
            {'database_url': 'sqlite:///test_rag.db'},  # SQLite
        ]
        
        for i, config in enumerate(configs):
            try:
                rag = RAGIntegration(config)
                stats = rag.get_database_stats()
                log_test_result(f'db_config_{i}', 'passed', f'Database config {i} works: {type(stats)}')
            except Exception as e:
                log_test_result(f'db_config_{i}', 'failed', f'Database config {i} failed', str(e))
        
        return True
        
    except Exception as e:
        log_test_result('database_test', 'errors', 'Database connection test failed', str(e))
        return False

def test_error_handling():
    """Test 7: Error handling and logging."""
    print("\n" + "="*60)
    print("TEST 7: Testing error handling and logging")
    print("="*60)
    
    try:
        from integration_rag import RAGIntegration
        rag = RAGIntegration()
        
        # Test with invalid inputs
        test_cases = [
            ('detect_arbitration', '/nonexistent/file.txt'),
            ('detect_from_text', None),
            ('compare_clause', ''),
        ]
        
        for method_name, invalid_input in test_cases:
            try:
                method = getattr(rag, method_name)
                result = method(invalid_input) if invalid_input is not None else method()
                
                # Check if error is handled gracefully
                if isinstance(result, dict) and ('error' in result or 'detected' in result):
                    log_test_result(f'error_handling_{method_name}', 'passed', f'{method_name} handles errors gracefully')
                else:
                    log_test_result(f'error_handling_{method_name}', 'warnings', f'{method_name} error handling unclear')
                    
            except Exception as e:
                log_test_result(f'error_handling_{method_name}', 'failed', f'{method_name} error handling failed', str(e))
        
        return True
        
    except Exception as e:
        log_test_result('error_handling_test', 'errors', 'Error handling test failed', str(e))
        return False

def test_circular_dependencies():
    """Test 8: Check for circular dependencies."""
    print("\n" + "="*60)
    print("TEST 8: Testing for circular dependencies")
    print("="*60)
    
    try:
        import sys
        initial_modules = set(sys.modules.keys())
        
        # Import integration module
        import integration_rag
        
        # Check what new modules were loaded
        new_modules = set(sys.modules.keys()) - initial_modules
        logger.info(f"New modules loaded: {len(new_modules)}")
        
        # Look for potential circular imports
        problematic_modules = [m for m in new_modules if 'integration_rag' in str(sys.modules.get(m, ''))]
        
        if problematic_modules:
            log_test_result('circular_dependencies', 'warnings', f'Potential circular dependencies: {problematic_modules}')
        else:
            log_test_result('circular_dependencies', 'passed', 'No circular dependencies detected')
        
        return True
        
    except Exception as e:
        log_test_result('circular_dependencies_test', 'errors', 'Circular dependency test failed', str(e))
        return False

def test_existing_documents():
    """Test 9: Test with existing test documents."""
    print("\n" + "="*60)
    print("TEST 9: Testing with existing test documents")
    print("="*60)
    
    try:
        from integration_rag import RAGIntegration
        rag = RAGIntegration()
        
        # Check for existing test documents
        test_docs_dir = Path(__file__).parent / "data" / "test_documents"
        logger.info(f"Looking for test documents in: {test_docs_dir}")
        
        if test_docs_dir.exists():
            log_test_result('test_docs_exist', 'passed', f'Test documents directory exists: {test_docs_dir}')
            
            # Test with existing documents
            test_files = list(test_docs_dir.glob("*.txt"))
            logger.info(f"Found {len(test_files)} test files")
            
            for test_file in test_files[:3]:  # Test first 3 files
                try:
                    result = rag.detect_arbitration(str(test_file))
                    log_test_result(f'existing_doc_{test_file.name}', 'passed', f'Processed {test_file.name}: detected={result.get("detected")}')
                except Exception as e:
                    log_test_result(f'existing_doc_{test_file.name}', 'failed', f'Failed to process {test_file.name}', str(e))
        else:
            log_test_result('test_docs_exist', 'warnings', f'Test documents directory not found: {test_docs_dir}')
        
        return True
        
    except Exception as e:
        log_test_result('existing_docs_test', 'errors', 'Existing documents test failed', str(e))
        return False

def generate_integration_report():
    """Generate comprehensive integration status report."""
    print("\n" + "="*60)
    print("INTEGRATION STATUS REPORT")
    print("="*60)
    
    total_tests = sum(len(results) for results in test_results.values())
    passed = len(test_results['passed'])
    failed = len(test_results['failed'])
    errors = len(test_results['errors'])
    warnings = len(test_results['warnings'])
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"Success Rate: {(passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
    
    if failed > 0:
        print("\n🔴 FAILED TESTS:")
        for result in test_results['failed']:
            print(f"   - {result['test']}: {result['message']}")
    
    if errors > 0:
        print("\n💥 ERROR TESTS:")
        for result in test_results['errors']:
            print(f"   - {result['test']}: {result['message']}")
    
    if warnings > 0:
        print("\n⚠️  WARNING TESTS:")
        for result in test_results['warnings']:
            print(f"   - {result['test']}: {result['message']}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if failed > 0 or errors > 0:
        print("🔴 CRITICAL ISSUES FOUND:")
        print("   - Review failed tests and fix underlying issues")
        print("   - Check import paths and dependencies")
        print("   - Verify RAG system installation")
    elif warnings > 0:
        print("🟡 MINOR ISSUES FOUND:")
        print("   - Review warnings for potential improvements")
        print("   - Consider addressing non-critical issues")
    else:
        print("🟢 INTEGRATION LOOKS GOOD:")
        print("   - All tests passed successfully")
        print("   - RAG system is properly integrated")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Run additional testing with real documents")
    print("   2. Performance benchmarking")
    print("   3. Production deployment verification")
    print("   4. Monitor system logs for issues")
    
    return test_results

def main():
    """Run all integration tests."""
    print("🚀 Starting RAG Integration Debug and Test Suite")
    print("="*60)
    
    # Set up working directory
    os.chdir(Path(__file__).parent)
    
    # Run all tests
    test_functions = [
        test_imports,
        test_rag_integration_class,
        test_detect_methods,
        test_backward_compatibility,
        test_existing_backend_compatibility,
        test_database_connections,
        test_error_handling,
        test_circular_dependencies,
        test_existing_documents,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except KeyboardInterrupt:
            print("\n⚠️  Test suite interrupted by user")
            break
        except Exception as e:
            test_name = test_func.__name__
            log_test_result(test_name, 'errors', f'Test function {test_name} crashed', str(e))
            logger.error(f"Test function {test_name} crashed: {e}")
            logger.error(traceback.format_exc())
    
    # Generate final report
    report = generate_integration_report()
    
    # Save results to file
    try:
        import json
        with open('rag_integration_test_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Results saved to: rag_integration_test_results.json")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    return report

if __name__ == "__main__":
    results = main()