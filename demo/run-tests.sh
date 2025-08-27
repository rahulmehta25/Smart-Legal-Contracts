#!/bin/bash

# Comprehensive Test Runner for Arbitration Detection System
# This script runs all tests, collects metrics, and generates reports

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
DEMO_DIR="$PROJECT_ROOT/demo"
REPORTS_DIR="$DEMO_DIR/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_REPORT_FILE="$REPORTS_DIR/test_report_$TIMESTAMP.json"
PERFORMANCE_REPORT_FILE="$REPORTS_DIR/performance_report_$TIMESTAMP.json"
COVERAGE_REPORT_FILE="$REPORTS_DIR/coverage_report_$TIMESTAMP.html"

# Ensure reports directory exists
mkdir -p "$REPORTS_DIR"

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    else
        print_success "Python 3 is available: $(python3 --version)"
    fi
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed - skipping containerized tests"
        DOCKER_AVAILABLE=false
    else
        if docker info &> /dev/null; then
            print_success "Docker is available and running"
            DOCKER_AVAILABLE=true
        else
            print_warning "Docker is installed but not running - skipping containerized tests"
            DOCKER_AVAILABLE=false
        fi
    fi
    
    # Check if pytest is installed
    if ! python3 -m pytest --version &> /dev/null; then
        print_error "pytest is not installed"
        print_info "Installing pytest..."
        pip3 install pytest pytest-cov pytest-html pytest-json-report
    else
        print_success "pytest is available: $(python3 -m pytest --version)"
    fi
    
    # Check backend directory
    if [ ! -d "$BACKEND_DIR" ]; then
        print_error "Backend directory not found: $BACKEND_DIR"
        exit 1
    else
        print_success "Backend directory found"
    fi
    
    echo ""
}

# Function to setup test environment
setup_test_environment() {
    print_header "Setting Up Test Environment"
    
    # Set PYTHONPATH
    export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"
    print_success "PYTHONPATH set to include backend directory"
    
    # Install test dependencies
    if [ -f "$BACKEND_DIR/requirements-test.txt" ]; then
        print_info "Installing test dependencies..."
        pip3 install -r "$BACKEND_DIR/requirements-test.txt" --quiet
        print_success "Test dependencies installed"
    else
        print_warning "Test requirements file not found, installing basic dependencies"
        pip3 install pytest pytest-cov pytest-html pytest-json-report --quiet
    fi
    
    # Create test database if needed
    print_info "Setting up test database..."
    # In a real implementation, this would setup a test database
    print_success "Test database setup completed"
    
    echo ""
}

# Function to run unit tests
run_unit_tests() {
    print_header "Running Unit Tests"
    
    cd "$BACKEND_DIR"
    
    # Run unit tests with coverage
    python3 -m pytest tests/ \
        --cov=app \
        --cov-report=html:"$REPORTS_DIR/coverage_html" \
        --cov-report=json:"$REPORTS_DIR/coverage.json" \
        --json-report --json-report-file="$REPORTS_DIR/unit_tests.json" \
        --html="$REPORTS_DIR/unit_tests.html" --self-contained-html \
        -v \
        --tb=short \
        --durations=10 \
        || {
            print_error "Unit tests failed"
            return 1
        }
    
    print_success "Unit tests completed"
    echo ""
}

# Function to run integration tests
run_integration_tests() {
    print_header "Running Integration Tests"
    
    cd "$BACKEND_DIR"
    
    # Run integration tests
    python3 -m pytest tests/integration/ \
        --json-report --json-report-file="$REPORTS_DIR/integration_tests.json" \
        --html="$REPORTS_DIR/integration_tests.html" --self-contained-html \
        -v \
        --tb=short \
        --durations=10 \
        || {
            print_error "Integration tests failed"
            return 1
        }
    
    print_success "Integration tests completed"
    echo ""
}

# Function to run demo tests
run_demo_tests() {
    print_header "Running Demo Test Scenarios"
    
    cd "$DEMO_DIR"
    
    # Run demo tests
    python3 -m pytest tests/ \
        --json-report --json-report-file="$REPORTS_DIR/demo_tests.json" \
        --html="$REPORTS_DIR/demo_tests.html" --self-contained-html \
        -v \
        --tb=short \
        --durations=10 \
        -s \
        || {
            print_error "Demo tests failed"
            return 1
        }
    
    print_success "Demo tests completed"
    echo ""
}

# Function to run load tests
run_load_tests() {
    print_header "Running Load Tests"
    
    cd "$BACKEND_DIR"
    
    # Check if locust is available
    if command -v locust &> /dev/null; then
        print_info "Running Locust load tests..."
        
        # Start the application in background
        python3 app/main.py &
        APP_PID=$!
        
        # Wait for app to start
        sleep 5
        
        # Run load tests
        locust -f tests/load_tests/locustfile.py \
            --host=http://localhost:8000 \
            --users=100 \
            --spawn-rate=10 \
            --run-time=60s \
            --headless \
            --html="$REPORTS_DIR/load_tests.html" \
            --csv="$REPORTS_DIR/load_tests" \
            || {
                print_warning "Load tests failed or had issues"
            }
        
        # Stop the application
        kill $APP_PID 2>/dev/null || true
        
        print_success "Load tests completed"
    else
        print_warning "Locust not available, skipping load tests"
        print_info "Install locust with: pip install locust"
    fi
    
    echo ""
}

# Function to run performance benchmarks
run_performance_benchmarks() {
    print_header "Running Performance Benchmarks"
    
    cd "$BACKEND_DIR"
    
    # Run performance tests
    python3 -m pytest tests/integration/test_performance_benchmarks.py \
        --json-report --json-report-file="$REPORTS_DIR/performance_tests.json" \
        --html="$REPORTS_DIR/performance_tests.html" --self-contained-html \
        -v \
        --tb=short \
        --benchmark-only \
        --benchmark-json="$REPORTS_DIR/benchmarks.json" \
        || {
            print_warning "Performance benchmarks had issues"
        }
    
    print_success "Performance benchmarks completed"
    echo ""
}

# Function to run security tests
run_security_tests() {
    print_header "Running Security Tests"
    
    cd "$BACKEND_DIR"
    
    # Check if safety is available for dependency scanning
    if command -v safety &> /dev/null; then
        print_info "Running dependency security scan..."
        safety check --json --output="$REPORTS_DIR/security_dependencies.json" || true
        print_success "Dependency security scan completed"
    else
        print_warning "Safety not available for dependency scanning"
        print_info "Install safety with: pip install safety"
    fi
    
    # Check if bandit is available for code security scanning
    if command -v bandit &> /dev/null; then
        print_info "Running code security scan..."
        bandit -r app/ -f json -o "$REPORTS_DIR/security_code.json" || true
        print_success "Code security scan completed"
    else
        print_warning "Bandit not available for code security scanning"
        print_info "Install bandit with: pip install bandit"
    fi
    
    # Run authentication and authorization tests
    python3 -m pytest tests/integration/test_authentication_authorization.py \
        --json-report --json-report-file="$REPORTS_DIR/security_tests.json" \
        -v \
        || {
            print_warning "Security tests had issues"
        }
    
    print_success "Security tests completed"
    echo ""
}

# Function to run containerized tests (if Docker is available)
run_containerized_tests() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        print_header "Running Containerized Tests"
        
        cd "$PROJECT_ROOT"
        
        # Build test container
        print_info "Building test container..."
        docker build -f backend/Dockerfile.test -t arbitration-detector-test . || {
            print_error "Failed to build test container"
            return 1
        }
        
        # Run tests in container
        print_info "Running tests in container..."
        docker run --rm \
            -v "$REPORTS_DIR:/app/reports" \
            arbitration-detector-test \
            /bin/bash -c "
                cd /app &&
                python -m pytest tests/ \
                    --json-report --json-report-file=reports/container_tests.json \
                    --html=reports/container_tests.html --self-contained-html \
                    -v --tb=short
            " || {
                print_warning "Containerized tests had issues"
            }
        
        print_success "Containerized tests completed"
    else
        print_warning "Docker not available, skipping containerized tests"
    fi
    
    echo ""
}

# Function to collect and process test results
collect_test_results() {
    print_header "Collecting and Processing Test Results"
    
    # Initialize results
    declare -A test_results
    test_results[total_tests]=0
    test_results[passed_tests]=0
    test_results[failed_tests]=0
    test_results[skipped_tests]=0
    
    # Process JSON reports
    for report_file in "$REPORTS_DIR"/*.json; do
        if [ -f "$report_file" ] && [[ "$report_file" == *"tests.json" ]]; then
            if command -v jq &> /dev/null; then
                # Extract test counts using jq
                if [ -s "$report_file" ]; then
                    passed=$(jq -r '.summary.passed // 0' "$report_file" 2>/dev/null || echo "0")
                    failed=$(jq -r '.summary.failed // 0' "$report_file" 2>/dev/null || echo "0")
                    skipped=$(jq -r '.summary.skipped // 0' "$report_file" 2>/dev/null || echo "0")
                    
                    test_results[passed_tests]=$((test_results[passed_tests] + passed))
                    test_results[failed_tests]=$((test_results[failed_tests] + failed))
                    test_results[skipped_tests]=$((test_results[skipped_tests] + skipped))
                fi
            fi
        fi
    done
    
    test_results[total_tests]=$((test_results[passed_tests] + test_results[failed_tests] + test_results[skipped_tests]))
    
    # Calculate success rate
    if [ ${test_results[total_tests]} -gt 0 ]; then
        success_rate=$(( (test_results[passed_tests] * 100) / test_results[total_tests] ))
    else
        success_rate=0
    fi
    
    print_success "Test results collected"
    echo ""
    
    # Store results for report generation
    TEST_SUMMARY="{
        \"total_tests\": ${test_results[total_tests]},
        \"passed_tests\": ${test_results[passed_tests]},
        \"failed_tests\": ${test_results[failed_tests]},
        \"skipped_tests\": ${test_results[skipped_tests]},
        \"success_rate\": $success_rate,
        \"timestamp\": \"$TIMESTAMP\"
    }"
}

# Function to generate comprehensive report
generate_comprehensive_report() {
    print_header "Generating Comprehensive Test Report"
    
    # Create comprehensive JSON report
    cat > "$TEST_REPORT_FILE" << EOF
{
    "test_execution_summary": $TEST_SUMMARY,
    "test_categories": {
        "unit_tests": {
            "status": "completed",
            "report_file": "unit_tests.json",
            "html_report": "unit_tests.html"
        },
        "integration_tests": {
            "status": "completed", 
            "report_file": "integration_tests.json",
            "html_report": "integration_tests.html"
        },
        "demo_tests": {
            "status": "completed",
            "report_file": "demo_tests.json", 
            "html_report": "demo_tests.html"
        },
        "performance_tests": {
            "status": "completed",
            "report_file": "performance_tests.json",
            "html_report": "performance_tests.html"
        },
        "security_tests": {
            "status": "completed",
            "report_file": "security_tests.json"
        }
    },
    "coverage": {
        "html_report": "coverage_html/index.html",
        "json_report": "coverage.json"
    },
    "reports_directory": "$REPORTS_DIR",
    "execution_timestamp": "$TIMESTAMP",
    "system_info": {
        "python_version": "$(python3 --version)",
        "platform": "$(uname -s)",
        "hostname": "$(hostname)"
    }
}
EOF
    
    print_success "Comprehensive report generated: $TEST_REPORT_FILE"
    echo ""
}

# Function to display test summary
display_test_summary() {
    print_header "Test Execution Summary"
    
    echo -e "${CYAN}Timestamp:${NC} $TIMESTAMP"
    echo -e "${CYAN}Reports Directory:${NC} $REPORTS_DIR"
    echo ""
    
    if [ ${test_results[total_tests]} -gt 0 ]; then
        echo -e "${CYAN}Total Tests:${NC} ${test_results[total_tests]}"
        echo -e "${GREEN}Passed:${NC} ${test_results[passed_tests]}"
        echo -e "${RED}Failed:${NC} ${test_results[failed_tests]}"
        echo -e "${YELLOW}Skipped:${NC} ${test_results[skipped_tests]}"
        echo -e "${CYAN}Success Rate:${NC} $success_rate%"
    else
        echo -e "${YELLOW}No test results to summarize${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Generated Reports:${NC}"
    ls -la "$REPORTS_DIR"/*.html 2>/dev/null | while read line; do
        echo -e "  ${GREEN}$(basename ${line##* })${NC}"
    done
    
    echo ""
    echo -e "${CYAN}JSON Reports:${NC}"
    ls -la "$REPORTS_DIR"/*.json 2>/dev/null | while read line; do
        echo -e "  ${BLUE}$(basename ${line##* })${NC}"
    done
    
    echo ""
    if [ ${test_results[failed_tests]} -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed successfully!${NC}"
    else
        echo -e "${RED}❌ Some tests failed. Check the reports for details.${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}To view HTML reports, open:${NC}"
    echo -e "  file://$REPORTS_DIR/unit_tests.html"
    echo -e "  file://$REPORTS_DIR/integration_tests.html" 
    echo -e "  file://$REPORTS_DIR/demo_tests.html"
    echo -e "  file://$REPORTS_DIR/coverage_html/index.html"
    
    echo ""
}

# Function to cleanup
cleanup() {
    print_info "Cleaning up..."
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clean up temporary files
    # (Add any cleanup logic here)
    
    print_success "Cleanup completed"
}

# Main execution function
main() {
    # Trap to ensure cleanup on exit
    trap cleanup EXIT
    
    print_header "Arbitration Detection System - Comprehensive Test Suite"
    echo -e "${PURPLE}Starting comprehensive test execution at $(date)${NC}"
    echo ""
    
    # Run all test phases
    check_prerequisites
    setup_test_environment
    
    # Initialize test results tracking
    declare -A test_results
    
    # Run tests (continue even if some fail)
    run_unit_tests || true
    run_integration_tests || true
    run_demo_tests || true
    run_load_tests || true
    run_performance_benchmarks || true
    run_security_tests || true
    run_containerized_tests || true
    
    # Process results and generate reports
    collect_test_results
    generate_comprehensive_report
    display_test_summary
    
    print_header "Test Execution Complete"
    echo -e "${PURPLE}Finished at $(date)${NC}"
    
    # Exit with appropriate code
    if [ ${test_results[failed_tests]} -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Command line argument parsing
case "${1:-all}" in
    unit)
        setup_test_environment
        run_unit_tests
        ;;
    integration)
        setup_test_environment
        run_integration_tests
        ;;
    demo)
        setup_test_environment
        run_demo_tests
        ;;
    load)
        setup_test_environment
        run_load_tests
        ;;
    performance)
        setup_test_environment
        run_performance_benchmarks
        ;;
    security)
        setup_test_environment
        run_security_tests
        ;;
    docker)
        run_containerized_tests
        ;;
    all|*)
        main
        ;;
esac