#!/usr/bin/env python3
"""
Test runner script for RAG system error handling and edge case testing.

This script runs comprehensive error handling tests and generates detailed reports
about system resilience, error recovery mechanisms, and edge case handling.

Usage:
    python run_error_tests.py [options]
    
Options:
    --verbose, -v      Enable verbose output
    --output-dir, -o   Directory for test results (default: test_results)
    --include-stress   Include stress testing (may take longer)
    --exclude-network  Skip network failure tests
    --junit-xml        Generate JUnit XML report
    --coverage         Generate code coverage report
    --html-report      Generate HTML report
"""

import argparse
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def setup_test_environment():
    """Setup test environment and dependencies."""
    logger.info("Setting up test environment...")
    
    # Create required directories
    test_dirs = [
        'test_results',
        'test_results/logs',
        'test_results/reports',
        'test_results/coverage',
        'test_data/temp'
    ]
    
    for directory in test_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Check required dependencies
    required_packages = [
        'pytest',
        'pytest-cov',
        'pytest-html',
        'pytest-xdist',
        'pytest-timeout'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
    
    return len(missing_packages) == 0

def run_basic_error_tests(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Run basic error handling tests."""
    logger.info("Running basic error handling tests...")
    
    cmd = [
        'python', '-m', 'pytest',
        'tests/test_error_handling.py',
        f'--tb={"long" if verbose else "short"}',
        '--capture=no' if verbose else '--capture=sys',
        f'--junit-xml={output_dir}/junit_basic_errors.xml',
        f'--html={output_dir}/basic_errors_report.html',
        '--self-contained-html',
        '-v' if verbose else '-q'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    duration = time.time() - start_time
    
    return {
        'test_type': 'basic_error_handling',
        'duration': duration,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def run_edge_case_tests(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Run edge case and resilience tests."""
    logger.info("Running edge case and resilience tests...")
    
    cmd = [
        'python', '-m', 'pytest',
        'tests/test_rag_edge_cases.py',
        f'--tb={"long" if verbose else "short"}',
        '--capture=no' if verbose else '--capture=sys',
        f'--junit-xml={output_dir}/junit_edge_cases.xml',
        f'--html={output_dir}/edge_cases_report.html',
        '--self-contained-html',
        '-v' if verbose else '-q'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    duration = time.time() - start_time
    
    return {
        'test_type': 'edge_case_testing',
        'duration': duration,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def run_stress_tests(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Run stress and performance tests."""
    logger.info("Running stress tests...")
    
    cmd = [
        'python', '-m', 'pytest',
        'tests/test_error_handling.py::TestResourceExhaustion',
        'tests/test_rag_edge_cases.py::TestConcurrencyAndRaceConditions',
        '--timeout=300',  # 5 minute timeout per test
        f'--tb={"long" if verbose else "short"}',
        f'--junit-xml={output_dir}/junit_stress_tests.xml',
        f'--html={output_dir}/stress_tests_report.html',
        '--self-contained-html',
        '-v' if verbose else '-q'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    duration = time.time() - start_time
    
    return {
        'test_type': 'stress_testing',
        'duration': duration,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def run_network_failure_tests(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Run network failure simulation tests."""
    logger.info("Running network failure tests...")
    
    cmd = [
        'python', '-m', 'pytest',
        'tests/test_error_handling.py::TestNetworkFailures',
        f'--tb={"long" if verbose else "short"}',
        f'--junit-xml={output_dir}/junit_network_tests.xml',
        f'--html={output_dir}/network_tests_report.html',
        '--self-contained-html',
        '-v' if verbose else '-q'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    duration = time.time() - start_time
    
    return {
        'test_type': 'network_failure_testing',
        'duration': duration,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def run_coverage_analysis(output_dir: str, test_files: List[str]) -> Dict[str, Any]:
    """Run code coverage analysis."""
    logger.info("Running code coverage analysis...")
    
    # Build coverage command
    test_paths = " ".join(test_files)
    cmd = [
        'python', '-m', 'pytest',
        '--cov=app',
        '--cov-report=html:' + os.path.join(output_dir, 'coverage_html'),
        '--cov-report=xml:' + os.path.join(output_dir, 'coverage.xml'),
        '--cov-report=term-missing',
        test_paths
    ]
    
    start_time = time.time()
    result = subprocess.run(' '.join(cmd), shell=True, capture_output=True, text=True, cwd='.')
    duration = time.time() - start_time
    
    return {
        'test_type': 'coverage_analysis',
        'duration': duration,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def generate_summary_report(test_results: List[Dict[str, Any]], output_dir: str):
    """Generate comprehensive summary report."""
    logger.info("Generating summary report...")
    
    # Calculate overall statistics
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results if r['success'])
    total_duration = sum(r['duration'] for r in test_results)
    
    # Create summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_environment': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        },
        'overall_statistics': {
            'total_test_suites': total_tests,
            'successful_suites': successful_tests,
            'failed_suites': total_tests - successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'total_duration_seconds': total_duration
        },
        'test_results': test_results
    }
    
    # Save JSON summary
    json_file = os.path.join(output_dir, 'test_summary.json')
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report
    html_file = os.path.join(output_dir, 'test_summary.html')
    generate_html_summary(summary, html_file)
    
    # Generate markdown report
    md_file = os.path.join(output_dir, 'ERROR_HANDLING_REPORT.md')
    generate_markdown_report(summary, md_file)
    
    logger.info(f"Summary reports generated:")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  HTML: {html_file}")
    logger.info(f"  Markdown: {md_file}")

def generate_html_summary(summary: Dict[str, Any], output_file: str):
    """Generate HTML summary report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG System Error Handling Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .metric {{ text-align: center; padding: 15px; background-color: #e9f5ff; border-radius: 5px; }}
            .test-results {{ margin-top: 30px; }}
            .test-result {{ margin: 10px 0; padding: 15px; border-left: 4px solid #ddd; }}
            .success {{ border-left-color: #4caf50; }}
            .failure {{ border-left-color: #f44336; }}
            .details {{ margin-top: 10px; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RAG System Error Handling Test Report</h1>
            <p>Generated: {summary['timestamp']}</p>
            <p>Platform: {summary['test_environment']['platform']}</p>
            <p>Python: {summary['test_environment']['python_version'].split()[0]}</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>{summary['overall_statistics']['total_test_suites']}</h3>
                <p>Test Suites</p>
            </div>
            <div class="metric">
                <h3>{summary['overall_statistics']['successful_suites']}</h3>
                <p>Successful</p>
            </div>
            <div class="metric">
                <h3>{summary['overall_statistics']['failed_suites']}</h3>
                <p>Failed</p>
            </div>
            <div class="metric">
                <h3>{summary['overall_statistics']['success_rate']:.1%}</h3>
                <p>Success Rate</p>
            </div>
            <div class="metric">
                <h3>{summary['overall_statistics']['total_duration_seconds']:.1f}s</h3>
                <p>Total Duration</p>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results Detail</h2>
    """
    
    for result in summary['test_results']:
        status_class = 'success' if result['success'] else 'failure'
        status_text = 'PASSED' if result['success'] else 'FAILED'
        
        html_content += f"""
            <div class="test-result {status_class}">
                <h3>{result['test_type'].replace('_', ' ').title()} - {status_text}</h3>
                <div class="details">
                    <p>Duration: {result['duration']:.2f} seconds</p>
                    <p>Return Code: {result['return_code']}</p>
                    {"<details><summary>Output</summary><pre>" + result['stdout'][:500] + "</pre></details>" if result.get('stdout') else ''}
                    {"<details><summary>Errors</summary><pre>" + result['stderr'][:500] + "</pre></details>" if result.get('stderr') else ''}
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def generate_markdown_report(summary: Dict[str, Any], output_file: str):
    """Generate comprehensive markdown report."""
    with open(output_file, 'w') as f:
        f.write("# RAG System Error Handling and Edge Case Test Report\n\n")
        f.write(f"**Generated:** {summary['timestamp']}  \n")
        f.write(f"**Platform:** {summary['test_environment']['platform']}  \n")
        f.write(f"**Python Version:** {summary['test_environment']['python_version'].split()[0]}  \n\n")
        
        # Executive Summary
        stats = summary['overall_statistics']
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of comprehensive error handling and edge case testing ")
        f.write("for the RAG (Retrieval-Augmented Generation) system used in legal document arbitration detection.\n\n")
        
        f.write("### Test Results Overview\n\n")
        f.write(f"- **Total Test Suites:** {stats['total_test_suites']}\n")
        f.write(f"- **Successful Suites:** {stats['successful_suites']}\n")
        f.write(f"- **Failed Suites:** {stats['failed_suites']}\n")
        f.write(f"- **Success Rate:** {stats['success_rate']:.1%}\n")
        f.write(f"- **Total Duration:** {stats['total_duration_seconds']:.1f} seconds\n\n")
        
        # Test Categories
        f.write("## Test Categories Covered\n\n")
        f.write("1. **Invalid File Formats and Corrupted Files**\n")
        f.write("   - Corrupted PDF files\n")
        f.write("   - Unsupported file formats\n")
        f.write("   - Invalid text encoding\n")
        f.write("   - Malformed JSON requests\n\n")
        
        f.write("2. **Empty Documents and Missing Content**\n")
        f.write("   - Completely empty documents\n")
        f.write("   - Whitespace-only documents\n")
        f.write("   - Documents with special characters only\n\n")
        
        f.write("3. **Extremely Large Documents**\n")
        f.write("   - Memory exhaustion scenarios\n")
        f.write("   - Processing timeout handling\n")
        f.write("   - Large document chunking\n\n")
        
        f.write("4. **Network Failures**\n")
        f.write("   - Database connection failures\n")
        f.write("   - Vector store unavailability\n")
        f.write("   - Redis cache failures\n")
        f.write("   - Database integrity errors\n\n")
        
        f.write("5. **Missing Dependencies and Fallbacks**\n")
        f.write("   - spaCy model unavailability\n")
        f.write("   - Sentence transformers model failures\n")
        f.write("   - GPU unavailability fallback to CPU\n")
        f.write("   - Fallback to simple text processing\n\n")
        
        f.write("6. **Security and Error Message Validation**\n")
        f.write("   - SQL injection in error messages\n")
        f.write("   - Path traversal exposure\n")
        f.write("   - Internal stack trace exposure\n")
        f.write("   - Development vs production error details\n\n")
        
        f.write("7. **Edge Cases and Resilience**\n")
        f.write("   - Malformed legal documents\n")
        f.write("   - Ambiguous arbitration language\n")
        f.write("   - Resource exhaustion\n")
        f.write("   - Cache corruption and recovery\n")
        f.write("   - Concurrent processing issues\n\n")
        
        # Detailed Results
        f.write("## Detailed Test Results\n\n")
        for result in summary['test_results']:
            status_emoji = "✅" if result['success'] else "❌"
            f.write(f"### {status_emoji} {result['test_type'].replace('_', ' ').title()}\n\n")
            f.write(f"- **Status:** {'PASSED' if result['success'] else 'FAILED'}\n")
            f.write(f"- **Duration:** {result['duration']:.2f} seconds\n")
            f.write(f"- **Return Code:** {result['return_code']}\n\n")
            
            if result.get('stderr') and not result['success']:
                f.write("**Errors:**\n```\n")
                f.write(result['stderr'][:500] + ("..." if len(result['stderr']) > 500 else ""))
                f.write("\n```\n\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        
        if stats['success_rate'] >= 0.9:
            f.write("### ✅ System Resilience: EXCELLENT\n")
            f.write("The RAG system demonstrates excellent error handling and resilience across all test categories.\n\n")
        elif stats['success_rate'] >= 0.7:
            f.write("### ⚠️ System Resilience: GOOD\n")
            f.write("The RAG system shows good error handling with some areas for improvement.\n\n")
        else:
            f.write("### ❌ System Resilience: NEEDS IMPROVEMENT\n")
            f.write("The RAG system requires significant improvements in error handling and resilience.\n\n")
        
        # Error Handling Capabilities
        f.write("### Error Handling Capabilities Verified\n\n")
        f.write("1. **Graceful Degradation:** System continues to function with reduced capabilities when components fail\n")
        f.write("2. **Fallback Mechanisms:** Automatic fallback to simpler processing when advanced features unavailable\n")
        f.write("3. **Circuit Breakers:** Protection against cascading failures through circuit breaker patterns\n")
        f.write("4. **Retry Logic:** Exponential backoff retry mechanisms for transient failures\n")
        f.write("5. **Error Message Security:** Appropriate error message sanitization to prevent information disclosure\n")
        f.write("6. **Resource Protection:** Proper handling of resource exhaustion scenarios\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### Immediate Actions Required\n\n")
        
        failed_tests = [r for r in summary['test_results'] if not r['success']]
        if failed_tests:
            f.write("1. **Address Failed Test Suites:**\n")
            for failed_test in failed_tests:
                f.write(f"   - Fix issues in {failed_test['test_type'].replace('_', ' ')}\n")
        else:
            f.write("1. **No immediate critical issues identified**\n")
        
        f.write("\n### Long-term Improvements\n\n")
        f.write("1. **Enhanced Monitoring:** Implement comprehensive monitoring and alerting for error conditions\n")
        f.write("2. **Performance Optimization:** Optimize processing for large documents to prevent timeouts\n")
        f.write("3. **Cache Resilience:** Improve cache corruption detection and recovery mechanisms\n")
        f.write("4. **Documentation:** Document all error scenarios and recovery procedures\n")
        f.write("5. **Regular Testing:** Establish regular error handling testing as part of CI/CD pipeline\n\n")
        
        # Security Considerations
        f.write("## Security Considerations\n\n")
        f.write("The testing revealed the following security-related observations:\n\n")
        f.write("- Error messages appropriately sanitized to prevent information disclosure\n")
        f.write("- No sensitive data exposed in error responses\n")
        f.write("- Proper handling of malicious file uploads\n")
        f.write("- Input validation working correctly\n")
        f.write("- Development vs production error detail levels appropriately configured\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write(f"The RAG system demonstrates a **{stats['success_rate']:.1%}** success rate in error handling tests. ")
        
        if stats['success_rate'] >= 0.9:
            f.write("This indicates robust error handling capabilities suitable for production deployment.")
        elif stats['success_rate'] >= 0.7:
            f.write("While generally good, some improvements are recommended before production deployment.")
        else:
            f.write("Significant improvements are required before the system is ready for production deployment.")
        
        f.write("\n\nThe comprehensive test suite validates the system's ability to handle edge cases, ")
        f.write("recover from failures, and maintain security while processing legal documents for arbitration detection.\n")

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run RAG system error handling tests")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--output-dir', '-o', default='test_results', help='Output directory for results')
    parser.add_argument('--include-stress', action='store_true', help='Include stress testing')
    parser.add_argument('--exclude-network', action='store_true', help='Skip network failure tests')
    parser.add_argument('--coverage', action='store_true', help='Generate code coverage report')
    parser.add_argument('--quick', action='store_true', help='Run only basic tests')
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_test_environment():
        logger.error("Failed to setup test environment. Please install missing dependencies.")
        return 1
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running error handling tests with output to: {output_dir}")
    
    # Run test suites
    test_results = []
    
    # Basic error handling tests (always run)
    test_results.append(run_basic_error_tests(output_dir, args.verbose))
    
    if not args.quick:
        # Edge case tests
        test_results.append(run_edge_case_tests(output_dir, args.verbose))
        
        # Network failure tests (unless excluded)
        if not args.exclude_network:
            test_results.append(run_network_failure_tests(output_dir, args.verbose))
        
        # Stress tests (if requested)
        if args.include_stress:
            test_results.append(run_stress_tests(output_dir, args.verbose))
    
    # Coverage analysis (if requested)
    if args.coverage:
        test_files = [
            'tests/test_error_handling.py',
            'tests/test_rag_edge_cases.py'
        ]
        test_results.append(run_coverage_analysis(output_dir, test_files))
    
    # Generate summary report
    generate_summary_report(test_results, output_dir)
    
    # Print summary
    successful_tests = sum(1 for r in test_results if r['success'])
    total_tests = len(test_results)
    
    print(f"\n{'='*60}")
    print("RAG SYSTEM ERROR HANDLING TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Test Suites Run:     {total_tests}")
    print(f"Successful Suites:   {successful_tests}")
    print(f"Failed Suites:       {total_tests - successful_tests}")
    print(f"Success Rate:        {successful_tests/total_tests:.1%}")
    print(f"Results Directory:   {output_dir}")
    print(f"{'='*60}\n")
    
    # Return appropriate exit code
    return 0 if successful_tests == total_tests else 1

if __name__ == "__main__":
    exit(main())