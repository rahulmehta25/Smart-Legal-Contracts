#!/bin/bash

# Test Summary Report Generator
# Aggregates test results from multiple sources and generates a comprehensive report

set -e

REPORTS_DIR="/reports"
SUMMARY_FILE="$REPORTS_DIR/test_summary.html"

echo "Generating comprehensive test summary report..."

# Create HTML report structure
cat > $SUMMARY_FILE << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arbitration Detection System - Test Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { padding: 20px; border-radius: 8px; text-align: center; }
        .status-pass { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .status-fail { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .status-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .metric { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 14px; margin-top: 5px; }
        .section { margin-bottom: 30px; }
        .section h2 { border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .warning { color: #ffc107; font-weight: bold; }
        .timestamp { color: #6c757d; font-size: 12px; }
        .chart-container { margin: 20px 0; }
        .progress-bar { width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #28a745; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Arbitration Detection System</h1>
            <h2>Test Suite Summary Report</h2>
            <div class="timestamp">Generated on: $(date)</div>
        </div>
EOF

# Add test overview section
echo "        <div class=\"section\">" >> $SUMMARY_FILE
echo "            <h2>📊 Test Overview</h2>" >> $SUMMARY_FILE
echo "            <div class=\"status-grid\">" >> $SUMMARY_FILE

# Function to check if tests passed based on exit codes and report files
check_test_status() {
    local test_type=$1
    local report_file=$2
    
    if [[ -f "$REPORTS_DIR/$report_file" ]]; then
        # Check for specific patterns in the report
        if grep -q "FAILED\|ERROR\|failed" "$REPORTS_DIR/$report_file" 2>/dev/null; then
            echo "FAIL"
        else
            echo "PASS"
        fi
    else
        echo "UNKNOWN"
    fi
}

# Function to extract metrics from test reports
extract_metric() {
    local file=$1
    local pattern=$2
    local default=$3
    
    if [[ -f "$REPORTS_DIR/$file" ]]; then
        grep -o "$pattern" "$REPORTS_DIR/$file" | head -1 | grep -o '[0-9.]*' || echo "$default"
    else
        echo "$default"
    fi
}

# Backend Tests Status
BACKEND_STATUS=$(check_test_status "backend" "junit.xml")
BACKEND_COVERAGE=$(extract_metric "coverage.xml" 'line-rate="[0-9.]*"' "0")
BACKEND_COVERAGE_PERCENT=$(echo "$BACKEND_COVERAGE * 100" | bc -l 2>/dev/null | cut -d. -f1 || echo "0")

cat >> $SUMMARY_FILE << EOF
                <div class="status-card status-$(echo $BACKEND_STATUS | tr 'A-Z' 'a-z')">
                    <div class="metric">$BACKEND_STATUS</div>
                    <div class="metric-label">Backend Tests</div>
                    <div style="margin-top: 10px;">Coverage: ${BACKEND_COVERAGE_PERCENT}%</div>
                </div>
EOF

# Frontend Tests Status
FRONTEND_STATUS=$(check_test_status "frontend" "coverage-final.json")
FRONTEND_COVERAGE=$(extract_metric "coverage-final.json" '"lines":{"pct":[0-9.]*' "0")

cat >> $SUMMARY_FILE << EOF
                <div class="status-card status-$(echo $FRONTEND_STATUS | tr 'A-Z' 'a-z')">
                    <div class="metric">$FRONTEND_STATUS</div>
                    <div class="metric-label">Frontend Tests</div>
                    <div style="margin-top: 10px;">Coverage: ${FRONTEND_COVERAGE}%</div>
                </div>
EOF

# E2E Tests Status
E2E_STATUS=$(check_test_status "e2e" "cypress-results.xml")
E2E_TESTS_RUN=$(extract_metric "cypress-results.xml" 'tests="[0-9]*"' "0")

cat >> $SUMMARY_FILE << EOF
                <div class="status-card status-$(echo $E2E_STATUS | tr 'A-Z' 'a-z')">
                    <div class="metric">$E2E_STATUS</div>
                    <div class="metric-label">E2E Tests</div>
                    <div style="margin-top: 10px;">Tests: ${E2E_TESTS_RUN}</div>
                </div>
EOF

# Load Tests Status
LOAD_STATUS=$(check_test_status "load" "load-test-report.html")
LOAD_RPS=$(extract_metric "load-test-report.html" 'Total Requests per Second</td><td>[0-9.]*' "0")

cat >> $SUMMARY_FILE << EOF
                <div class="status-card status-$(echo $LOAD_STATUS | tr 'A-Z' 'a-z')">
                    <div class="metric">$LOAD_STATUS</div>
                    <div class="metric-label">Load Tests</div>
                    <div style="margin-top: 10px;">RPS: ${LOAD_RPS}</div>
                </div>
            </div>
        </div>
EOF

# Detailed Results Section
cat >> $SUMMARY_FILE << 'EOF'
        <div class="section">
            <h2>📋 Detailed Results</h2>
            
            <h3>Backend Test Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>Test Suite</th>
                        <th>Status</th>
                        <th>Tests Run</th>
                        <th>Failures</th>
                        <th>Duration</th>
                        <th>Coverage</th>
                    </tr>
                </thead>
                <tbody>
EOF

# Add backend test details if available
if [[ -f "$REPORTS_DIR/junit.xml" ]]; then
    BACKEND_TESTS=$(extract_metric "junit.xml" 'tests="[0-9]*"' "0")
    BACKEND_FAILURES=$(extract_metric "junit.xml" 'failures="[0-9]*"' "0")
    BACKEND_TIME=$(extract_metric "junit.xml" 'time="[0-9.]*"' "0")
    
    cat >> $SUMMARY_FILE << EOF
                    <tr>
                        <td>Unit Tests</td>
                        <td class="$(echo $BACKEND_STATUS | tr 'A-Z' 'a-z')">$BACKEND_STATUS</td>
                        <td>$BACKEND_TESTS</td>
                        <td>$BACKEND_FAILURES</td>
                        <td>${BACKEND_TIME}s</td>
                        <td>${BACKEND_COVERAGE_PERCENT}%</td>
                    </tr>
EOF
fi

cat >> $SUMMARY_FILE << 'EOF'
                </tbody>
            </table>
            
            <h3>Frontend Test Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>Test Suite</th>
                        <th>Status</th>
                        <th>Tests Run</th>
                        <th>Failures</th>
                        <th>Coverage</th>
                    </tr>
                </thead>
                <tbody>
EOF

# Add frontend test details if available
if [[ -f "$REPORTS_DIR/coverage-final.json" ]]; then
    cat >> $SUMMARY_FILE << EOF
                    <tr>
                        <td>React Components</td>
                        <td class="$(echo $FRONTEND_STATUS | tr 'A-Z' 'a-z')">$FRONTEND_STATUS</td>
                        <td>-</td>
                        <td>-</td>
                        <td>${FRONTEND_COVERAGE}%</td>
                    </tr>
EOF
fi

cat >> $SUMMARY_FILE << 'EOF'
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>⚡ Performance Metrics</h2>
EOF

# Add performance metrics if available
if [[ -f "$REPORTS_DIR/benchmark.json" ]]; then
    echo "            <h3>Backend Performance Benchmarks</h3>" >> $SUMMARY_FILE
    echo "            <div class=\"chart-container\">" >> $SUMMARY_FILE
    # Extract benchmark data and create simple visualization
    python3 -c "
import json
import sys
try:
    with open('$REPORTS_DIR/benchmark.json', 'r') as f:
        data = json.load(f)
    if 'benchmarks' in data:
        print('<table>')
        print('<thead><tr><th>Test</th><th>Mean Time</th><th>Min Time</th><th>Max Time</th></tr></thead>')
        print('<tbody>')
        for benchmark in data['benchmarks'][:10]:  # Show top 10
            name = benchmark.get('name', 'Unknown')
            mean = benchmark.get('stats', {}).get('mean', 0)
            min_time = benchmark.get('stats', {}).get('min', 0)
            max_time = benchmark.get('stats', {}).get('max', 0)
            print(f'<tr><td>{name}</td><td>{mean:.3f}s</td><td>{min_time:.3f}s</td><td>{max_time:.3f}s</td></tr>')
        print('</tbody></table>')
except Exception as e:
    print(f'<p>Error parsing benchmark data: {e}</p>')
" >> $SUMMARY_FILE 2>/dev/null || echo "<p>No benchmark data available</p>" >> $SUMMARY_FILE
    echo "            </div>" >> $SUMMARY_FILE
fi

# Add load test results if available
if [[ -f "$REPORTS_DIR/load-test-report.html" ]]; then
    echo "            <h3>Load Test Results</h3>" >> $SUMMARY_FILE
    echo "            <div class=\"chart-container\">" >> $SUMMARY_FILE
    
    # Extract key metrics from load test report
    TOTAL_REQUESTS=$(grep -o 'Total Requests</td><td>[0-9]*' "$REPORTS_DIR/load-test-report.html" | grep -o '[0-9]*' || echo "0")
    FAILED_REQUESTS=$(grep -o 'Failed Requests</td><td>[0-9]*' "$REPORTS_DIR/load-test-report.html" | grep -o '[0-9]*' || echo "0")
    AVG_RESPONSE_TIME=$(grep -o 'Average Response Time</td><td>[0-9.]*' "$REPORTS_DIR/load-test-report.html" | grep -o '[0-9.]*' || echo "0")
    
    cat >> $SUMMARY_FILE << EOF
            <table>
                <tr><td>Total Requests</td><td>$TOTAL_REQUESTS</td></tr>
                <tr><td>Failed Requests</td><td>$FAILED_REQUESTS</td></tr>
                <tr><td>Average Response Time</td><td>${AVG_RESPONSE_TIME}ms</td></tr>
                <tr><td>Requests per Second</td><td>$LOAD_RPS</td></tr>
            </table>
EOF
    echo "            </div>" >> $SUMMARY_FILE
fi

# Quality Gates Section
cat >> $SUMMARY_FILE << 'EOF'
        </div>

        <div class="section">
            <h2>🚦 Quality Gates</h2>
            <table>
                <thead>
                    <tr>
                        <th>Quality Gate</th>
                        <th>Threshold</th>
                        <th>Actual</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
EOF

# Define quality gate checks
check_quality_gate() {
    local name=$1
    local threshold=$2
    local actual=$3
    local comparison=$4  # "gt" for greater than, "lt" for less than
    
    local status="PASS"
    local status_class="pass"
    
    if [[ $comparison == "gt" ]]; then
        if (( $(echo "$actual >= $threshold" | bc -l 2>/dev/null || echo 0) )); then
            status="PASS"
            status_class="pass"
        else
            status="FAIL"
            status_class="fail"
        fi
    elif [[ $comparison == "lt" ]]; then
        if (( $(echo "$actual <= $threshold" | bc -l 2>/dev/null || echo 1) )); then
            status="PASS"
            status_class="pass"
        else
            status="FAIL"
            status_class="fail"
        fi
    fi
    
    echo "<tr><td>$name</td><td>$threshold</td><td>$actual</td><td class=\"$status_class\">$status</td></tr>" >> $SUMMARY_FILE
}

# Quality gate checks
check_quality_gate "Backend Code Coverage" "80%" "${BACKEND_COVERAGE_PERCENT}%" "gt"
check_quality_gate "Frontend Code Coverage" "80%" "${FRONTEND_COVERAGE}%" "gt"
check_quality_gate "Load Test Failure Rate" "5%" "$(echo "scale=2; $FAILED_REQUESTS * 100 / $TOTAL_REQUESTS" | bc -l 2>/dev/null || echo 0)%" "lt"
check_quality_gate "Average Response Time" "2000ms" "${AVG_RESPONSE_TIME}ms" "lt"

cat >> $SUMMARY_FILE << 'EOF'
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>📁 Test Artifacts</h2>
            <ul>
EOF

# List available test artifacts
for file in $REPORTS_DIR/*; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | cut -f1)
        echo "                <li><a href=\"$filename\">$filename</a> ($filesize)</li>" >> $SUMMARY_FILE
    fi
done

cat >> $SUMMARY_FILE << 'EOF'
            </ul>
        </div>

        <div class="section">
            <h2>🔧 Recommendations</h2>
            <div id="recommendations">
EOF

# Generate recommendations based on test results
if [[ $BACKEND_COVERAGE_PERCENT -lt 80 ]]; then
    echo "                <p>⚠️ <strong>Backend code coverage is below 80%.</strong> Consider adding more unit tests to improve coverage.</p>" >> $SUMMARY_FILE
fi

if [[ $FRONTEND_COVERAGE -lt 80 ]]; then
    echo "                <p>⚠️ <strong>Frontend code coverage is below 80%.</strong> Add more component and integration tests.</p>" >> $SUMMARY_FILE
fi

if [[ $BACKEND_STATUS == "FAIL" || $FRONTEND_STATUS == "FAIL" || $E2E_STATUS == "FAIL" ]]; then
    echo "                <p>❌ <strong>Some tests are failing.</strong> Review test failures and fix issues before deployment.</p>" >> $SUMMARY_FILE
fi

if (( $(echo "$AVG_RESPONSE_TIME > 1000" | bc -l 2>/dev/null || echo 0) )); then
    echo "                <p>🐌 <strong>Average response time is above 1 second.</strong> Consider performance optimizations.</p>" >> $SUMMARY_FILE
fi

# Add success message if all tests pass
if [[ $BACKEND_STATUS == "PASS" && $FRONTEND_STATUS == "PASS" && $E2E_STATUS == "PASS" ]]; then
    echo "                <p>✅ <strong>All tests are passing!</strong> The system is ready for deployment.</p>" >> $SUMMARY_FILE
fi

cat >> $SUMMARY_FILE << 'EOF'
            </div>
        </div>
        
        <div class="section">
            <div class="timestamp" style="text-align: center; margin-top: 30px;">
                Report generated by Arbitration Detection System Test Suite
            </div>
        </div>
    </div>
</body>
</html>
EOF

echo "✅ Test summary report generated: $SUMMARY_FILE"

# Also create a simple text summary
TEXT_SUMMARY="$REPORTS_DIR/test_summary.txt"
cat > $TEXT_SUMMARY << EOF
ARBITRATION DETECTION SYSTEM - TEST SUMMARY
===========================================

Test Execution Date: $(date)

OVERALL STATUS:
- Backend Tests: $BACKEND_STATUS (Coverage: ${BACKEND_COVERAGE_PERCENT}%)
- Frontend Tests: $FRONTEND_STATUS (Coverage: ${FRONTEND_COVERAGE}%)
- E2E Tests: $E2E_STATUS (Tests: $E2E_TESTS_RUN)
- Load Tests: $LOAD_STATUS (RPS: $LOAD_RPS)

QUALITY GATES:
- Backend Coverage: ${BACKEND_COVERAGE_PERCENT}% (Threshold: 80%)
- Frontend Coverage: ${FRONTEND_COVERAGE}% (Threshold: 80%)
- Average Response Time: ${AVG_RESPONSE_TIME}ms (Threshold: 2000ms)

ARTIFACTS GENERATED:
$(ls -la $REPORTS_DIR | grep -v '^d' | awk '{print "- " $9 " (" $5 " bytes)"}')

EOF

echo "✅ Text summary generated: $TEXT_SUMMARY"

# Exit with error if any critical tests failed
if [[ $BACKEND_STATUS == "FAIL" || $E2E_STATUS == "FAIL" ]]; then
    echo "❌ Critical tests failed. Exiting with error code 1."
    exit 1
else
    echo "✅ All critical tests passed."
    exit 0
fi