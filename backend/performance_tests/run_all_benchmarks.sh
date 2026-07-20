#!/bin/bash

# RAG Legal Analysis System - Complete Performance Testing Suite
# This script runs all performance benchmarks and generates reports

set -e

echo "========================================"
echo "RAG SYSTEM PERFORMANCE TESTING SUITE"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Create results directory
RESULTS_DIR="performance_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to check if service is running
check_service() {
    SERVICE=$1
    PORT=$2
    nc -z localhost $PORT 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ $SERVICE is running on port $PORT"
        return 0
    else
        echo "✗ $SERVICE is not running on port $PORT"
        return 1
    fi
}

# Check prerequisites
echo "Checking prerequisites..."
echo "------------------------"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is not installed"
    exit 1
fi
echo "✓ Python 3 is available"

# Check if Redis is running
if check_service "Redis" 6379; then
    REDIS_AVAILABLE=true
else
    REDIS_AVAILABLE=false
    echo "  Warning: Redis benchmarks will be skipped"
fi

# Check if API server is running
if check_service "API Server" 8000; then
    API_AVAILABLE=true
else
    API_AVAILABLE=false
    echo "  Warning: API load tests will be skipped"
fi

echo ""

# Install required packages if needed
echo "Installing required packages..."
pip3 install -q pytest pytest-benchmark locust psutil numpy pandas matplotlib seaborn reportlab redis memory_profiler

echo ""

# 1. Run basic performance benchmarks
echo "========================================"
echo "1. RUNNING BASIC PERFORMANCE BENCHMARKS"
echo "========================================"
echo ""

python3 benchmark_suite.py > "$RESULTS_DIR/benchmark_suite.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Basic benchmarks completed"
    mv performance_report_*.txt "$RESULTS_DIR/" 2>/dev/null || true
    mv performance_report_*.json "$RESULTS_DIR/" 2>/dev/null || true
else
    echo "✗ Basic benchmarks failed (see $RESULTS_DIR/benchmark_suite.log)"
fi

echo ""

# 2. Run cache benchmarks if Redis is available
if [ "$REDIS_AVAILABLE" = true ]; then
    echo "========================================"
    echo "2. RUNNING CACHE PERFORMANCE BENCHMARKS"
    echo "========================================"
    echo ""
    
    python3 cache_benchmark.py > "$RESULTS_DIR/cache_benchmark.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Cache benchmarks completed"
        mv cache_performance_report_*.txt "$RESULTS_DIR/" 2>/dev/null || true
        mv cache_performance_report_*.json "$RESULTS_DIR/" 2>/dev/null || true
        mv cache_performance_metrics.png "$RESULTS_DIR/" 2>/dev/null || true
    else
        echo "✗ Cache benchmarks failed (see $RESULTS_DIR/cache_benchmark.log)"
    fi
else
    echo "Skipping cache benchmarks (Redis not available)"
fi

echo ""

# 3. Run pytest benchmarks
echo "========================================"
echo "3. RUNNING PYTEST PERFORMANCE TESTS"
echo "========================================"
echo ""

cd ..
pytest tests/test_performance_benchmarks.py \
    --benchmark-only \
    --benchmark-json="$RESULTS_DIR/pytest_benchmark.json" \
    --benchmark-save="rag_benchmarks" \
    --benchmark-autosave \
    -v > "performance_tests/$RESULTS_DIR/pytest_benchmark.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Pytest benchmarks completed"
else
    echo "✗ Some pytest benchmarks failed (see $RESULTS_DIR/pytest_benchmark.log)"
fi

cd performance_tests
echo ""

# 4. Run load tests if API is available
if [ "$API_AVAILABLE" = true ]; then
    echo "========================================"
    echo "4. RUNNING LOAD TESTS (30 seconds)"
    echo "========================================"
    echo ""
    
    # Run a quick load test
    locust -f locustfile.py \
        --host http://localhost:8000 \
        --users 10 \
        --spawn-rate 2 \
        --time 30s \
        --headless \
        --html "$RESULTS_DIR/locust_report.html" \
        --csv "$RESULTS_DIR/locust" \
        > "$RESULTS_DIR/locust.log" 2>&1 &
    
    LOCUST_PID=$!
    
    # Wait for load test to complete
    echo "Running load test for 30 seconds..."
    wait $LOCUST_PID
    
    if [ $? -eq 0 ]; then
        echo "✓ Load tests completed"
    else
        echo "✗ Load tests failed (see $RESULTS_DIR/locust.log)"
    fi
else
    echo "Skipping load tests (API server not available)"
fi

echo ""

# 5. Generate combined report
echo "========================================"
echo "5. GENERATING COMBINED REPORT"
echo "========================================"
echo ""

# Create summary report
cat > "$RESULTS_DIR/PERFORMANCE_SUMMARY.md" << EOF
# RAG Legal Analysis System - Performance Test Results

Generated: $(date)

## Test Environment
- Python Version: $(python3 --version)
- Redis Available: $REDIS_AVAILABLE
- API Server Available: $API_AVAILABLE
- Results Directory: $RESULTS_DIR

## Test Results

### 1. Basic Performance Benchmarks
- Status: Completed
- Log: benchmark_suite.log
- Report: performance_report_*.txt
- Data: performance_report_*.json

### 2. Cache Performance (Redis)
- Status: $(if [ "$REDIS_AVAILABLE" = true ]; then echo "Completed"; else echo "Skipped"; fi)
- Log: cache_benchmark.log
- Report: cache_performance_report_*.txt
- Data: cache_performance_report_*.json

### 3. Pytest Benchmarks
- Status: Completed
- Log: pytest_benchmark.log
- Data: pytest_benchmark.json

### 4. Load Testing
- Status: $(if [ "$API_AVAILABLE" = true ]; then echo "Completed"; else echo "Skipped"; fi)
- Log: locust.log
- Report: locust_report.html
- Data: locust_*.csv

## Key Metrics

EOF

# Extract key metrics from reports if available
if [ -f "$RESULTS_DIR"/performance_report_*.txt ]; then
    echo "### Document Processing Performance" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
    grep -E "Average throughput:|Average latency:" "$RESULTS_DIR"/performance_report_*.txt >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md" 2>/dev/null || true
    echo "" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
fi

if [ -f "$RESULTS_DIR"/cache_performance_report_*.txt ]; then
    echo "### Cache Performance" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
    grep -E "Cache Hit Rate:|Speedup Factor:" "$RESULTS_DIR"/cache_performance_report_*.txt >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md" 2>/dev/null || true
    echo "" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
fi

echo "## Recommendations" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
echo "" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"

# Add recommendations based on results
if [ "$REDIS_AVAILABLE" = false ]; then
    echo "- ⚠️ Enable Redis for improved performance through caching" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
fi

if [ "$API_AVAILABLE" = false ]; then
    echo "- ⚠️ API server was not available for load testing" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"
fi

echo "- Review detailed reports in the results directory for specific optimizations" >> "$RESULTS_DIR/PERFORMANCE_SUMMARY.md"

echo "✓ Summary report generated: $RESULTS_DIR/PERFORMANCE_SUMMARY.md"

echo ""
echo "========================================"
echo "PERFORMANCE TESTING COMPLETE"
echo "========================================"
echo ""
echo "All results saved to: $RESULTS_DIR/"
echo ""
echo "Key files:"
echo "  - Summary: $RESULTS_DIR/PERFORMANCE_SUMMARY.md"
echo "  - Benchmark Report: $RESULTS_DIR/performance_report_*.txt"
echo "  - Cache Report: $RESULTS_DIR/cache_performance_report_*.txt"
echo "  - Load Test Report: $RESULTS_DIR/locust_report.html"
echo ""
echo "Completed at: $(date)"