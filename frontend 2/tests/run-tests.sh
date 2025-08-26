#!/bin/bash

# Comprehensive Playwright Test Runner Script
# This script manages the complete test execution pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_PORT=5173
BACKEND_PORT=8000
FRONTEND_URL="http://localhost:$FRONTEND_PORT"
BACKEND_URL="http://localhost:$BACKEND_PORT"
TEST_ENV_FILE=".env.test"

echo -e "${BLUE}🚀 Starting Playwright Test Suite${NC}"
echo "=================================="

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}⏳ Waiting for $name to be ready...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f --silent --max-time 2 "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is ready${NC}"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
    done
    
    echo -e "${RED}❌ $name failed to start within timeout${NC}"
    return 1
}

# Function to start backend if not running
start_backend() {
    if ! check_port $BACKEND_PORT; then
        echo -e "${BLUE}🔧 Starting backend server...${NC}"
        cd ../backend
        
        # Check if virtual environment exists
        if [ -d "venv" ]; then
            source venv/bin/activate
        elif [ -d "../venv" ]; then
            source ../venv/bin/activate
        fi
        
        # Start backend in background
        python -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT &
        BACKEND_PID=$!
        cd "../frontend 2"
        
        # Wait for backend to be ready
        if wait_for_service "$BACKEND_URL/health" "Backend"; then
            echo "Backend PID: $BACKEND_PID"
            echo $BACKEND_PID > .backend.pid
        else
            echo -e "${RED}❌ Failed to start backend${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Backend already running on port $BACKEND_PORT${NC}"
    fi
}

# Function to start frontend if not running
start_frontend() {
    if ! check_port $FRONTEND_PORT; then
        echo -e "${BLUE}🔧 Starting frontend server...${NC}"
        
        # Copy test environment
        if [ -f "$TEST_ENV_FILE" ]; then
            cp "$TEST_ENV_FILE" .env.local
        fi
        
        npm run dev &
        FRONTEND_PID=$!
        echo "Frontend PID: $FRONTEND_PID"
        echo $FRONTEND_PID > .frontend.pid
        
        # Wait for frontend to be ready
        if wait_for_service "$FRONTEND_URL" "Frontend"; then
            echo -e "${GREEN}✅ Frontend ready${NC}"
        else
            echo -e "${RED}❌ Failed to start frontend${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Frontend already running on port $FRONTEND_PORT${NC}"
    fi
}

# Function to cleanup processes
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up test processes...${NC}"
    
    if [ -f .frontend.pid ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm .frontend.pid
    fi
    
    if [ -f .backend.pid ]; then
        BACKEND_PID=$(cat .backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm .backend.pid
    fi
    
    # Clean up any remaining processes
    pkill -f "vite" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    # Parse command line arguments
    local test_type=${1:-"all"}
    local browser=${2:-"chromium"}
    local headed=${3:-"false"}
    
    echo "Test type: $test_type"
    echo "Browser: $browser"
    echo "Headed mode: $headed"
    echo ""
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}📦 Installing dependencies...${NC}"
        npm install
    fi
    
    # Install Playwright browsers if needed
    if [ ! -d "node_modules/.bin/playwright" ]; then
        echo -e "${BLUE}🎭 Installing Playwright browsers...${NC}"
        npx playwright install
    fi
    
    # Start services
    start_backend
    start_frontend
    
    # Verify both services are ready
    echo -e "${BLUE}🔍 Verifying services...${NC}"
    if ! wait_for_service "$BACKEND_URL/health" "Backend final check"; then
        exit 1
    fi
    
    if ! wait_for_service "$FRONTEND_URL" "Frontend final check"; then
        exit 1
    fi
    
    echo -e "${GREEN}✅ All services ready, starting tests...${NC}"
    echo ""
    
    # Run tests based on type
    case $test_type in
        "api")
            echo -e "${BLUE}🔌 Running API tests...${NC}"
            npx playwright test --project=api
            ;;
        "websocket")
            echo -e "${BLUE}🌐 Running WebSocket tests...${NC}"
            npx playwright test --project=websocket
            ;;
        "visual")
            echo -e "${BLUE}🎨 Running visual regression tests...${NC}"
            npx playwright test --project=visual
            ;;
        "mobile")
            echo -e "${BLUE}📱 Running mobile tests...${NC}"
            npx playwright test --project="Mobile Chrome"
            ;;
        "accessibility")
            echo -e "${BLUE}♿ Running accessibility tests...${NC}"
            npx playwright test accessibility
            ;;
        "performance")
            echo -e "${BLUE}⚡ Running performance tests...${NC}"
            npx playwright test performance
            ;;
        "smoke")
            echo -e "${BLUE}💨 Running smoke tests...${NC}"
            npx playwright test api-health frontend-integration --project=chromium
            ;;
        "full")
            echo -e "${BLUE}🎯 Running full test suite...${NC}"
            if [ "$headed" = "true" ]; then
                npx playwright test --headed
            else
                npx playwright test
            fi
            ;;
        *)
            echo -e "${BLUE}🔄 Running default test suite...${NC}"
            if [ "$headed" = "true" ]; then
                npx playwright test --headed --project=$browser
            else
                npx playwright test --project=$browser
            fi
            ;;
    esac
    
    local test_exit_code=$?
    
    # Generate test report
    echo ""
    echo -e "${BLUE}📊 Generating test report...${NC}"
    npx playwright show-report --open=false
    
    # Show summary
    echo ""
    echo "=================================="
    if [ $test_exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "${RED}❌ Some tests failed${NC}"
    fi
    
    echo -e "${BLUE}📁 Test artifacts:${NC}"
    echo "   - HTML Report: playwright-report/index.html"
    echo "   - Test Results: test-results/"
    echo "   - Screenshots: test-results/screenshots/"
    
    exit $test_exit_code
}

# Help function
show_help() {
    echo "Playwright Test Runner"
    echo ""
    echo "Usage: $0 [TEST_TYPE] [BROWSER] [HEADED]"
    echo ""
    echo "TEST_TYPE options:"
    echo "  all        - Run all tests (default)"
    echo "  api        - Run API tests only"
    echo "  websocket  - Run WebSocket tests only"
    echo "  visual     - Run visual regression tests"
    echo "  mobile     - Run mobile tests"
    echo "  accessibility - Run accessibility tests"
    echo "  performance - Run performance tests"
    echo "  smoke      - Run smoke tests (quick validation)"
    echo "  full       - Run complete test suite"
    echo ""
    echo "BROWSER options:"
    echo "  chromium   - Google Chrome (default)"
    echo "  firefox    - Mozilla Firefox"
    echo "  webkit     - Safari"
    echo ""
    echo "HEADED options:"
    echo "  false      - Headless mode (default)"
    echo "  true       - Show browser window"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests in chromium headless"
    echo "  $0 api                       # Run API tests only"
    echo "  $0 visual firefox            # Run visual tests in Firefox"
    echo "  $0 smoke chromium true       # Run smoke tests with browser visible"
    echo "  $0 full                      # Run complete test suite"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"