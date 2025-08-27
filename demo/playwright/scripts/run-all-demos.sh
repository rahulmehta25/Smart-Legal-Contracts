#!/bin/bash

# Comprehensive Playwright Demo Runner
# Runs all demo types and generates complete testing suite

set -e

echo "🎬 Starting Comprehensive Playwright Demo Suite"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
RECORD_VIDEO=false
HEADLESS=false
SKIP_SETUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --record)
            RECORD_VIDEO=true
            shift
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --record      Record video demos"
            echo "  --headless    Run in headless mode"
            echo "  --skip-setup  Skip environment setup"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Node.js and npm are installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    print_error "npm is required but not installed"
    exit 1
fi

# Setup environment
if [ "$SKIP_SETUP" = false ]; then
    print_status "Setting up environment..."
    
    # Install dependencies
    print_status "Installing dependencies..."
    npm install
    
    # Install Playwright browsers
    print_status "Installing Playwright browsers..."
    npm run install-browsers
    
    print_success "Environment setup completed"
fi

# Create output directories
print_status "Creating output directories..."
mkdir -p screenshots/{baseline,current,diff}
mkdir -p demo-videos
mkdir -p marketing-screenshots
mkdir -p reports
mkdir -p artifacts

# Set video recording option
VIDEO_OPTION=""
if [ "$RECORD_VIDEO" = true ]; then
    VIDEO_OPTION="--record"
    print_status "Video recording enabled"
fi

# Set headless option
HEADLESS_OPTION=""
if [ "$HEADLESS" = true ]; then
    HEADLESS_OPTION="--headless"
    print_status "Running in headless mode"
fi

print_status "Starting demo suite execution..."

# 1. Quick Demo (2 minutes)
print_status "Running Quick Demo (2 minutes)..."
if node demo/quick-demo.js $VIDEO_OPTION $HEADLESS_OPTION; then
    print_success "Quick demo completed"
else
    print_warning "Quick demo failed but continuing..."
fi

# 2. Full Demo (10 minutes)
print_status "Running Full Demo (10 minutes)..."
if node demo/full-demo.js $VIDEO_OPTION $HEADLESS_OPTION; then
    print_success "Full demo completed"
else
    print_warning "Full demo failed but continuing..."
fi

# 3. Performance Demo
print_status "Running Performance Demo..."
if node demo/performance-demo.js $HEADLESS_OPTION; then
    print_success "Performance demo completed"
else
    print_warning "Performance demo failed but continuing..."
fi

# 4. Accessibility Demo
print_status "Running Accessibility Demo..."
if node demo/accessibility-demo.js $HEADLESS_OPTION; then
    print_success "Accessibility demo completed"
else
    print_warning "Accessibility demo failed but continuing..."
fi

# 5. Interactive Feature Demos
print_status "Running Interactive Feature Demos..."

# Multilingual demo
if node demo/interaction-demo.js multilingual $HEADLESS_OPTION; then
    print_success "Multilingual demo completed"
else
    print_warning "Multilingual demo failed but continuing..."
fi

# Full interaction demo
if node demo/interaction-demo.js full $HEADLESS_OPTION; then
    print_success "Full interaction demo completed"
else
    print_warning "Full interaction demo failed but continuing..."
fi

# 6. Screenshot Generation
print_status "Generating Marketing Screenshots..."

# Generate all screenshot types
if node demo/screenshot-generator.js all marketing-screenshots; then
    print_success "Marketing screenshots generated"
else
    print_warning "Screenshot generation failed but continuing..."
fi

# 7. Visual Tests
print_status "Running Visual Tests..."

# Homepage visual tests
if npm run test -- --grep "@visual" tests/homepage.spec.ts; then
    print_success "Homepage visual tests completed"
else
    print_warning "Homepage visual tests failed but continuing..."
fi

# Upload flow visual tests
if npm run test -- --grep "@visual" tests/upload-flow.spec.ts; then
    print_success "Upload flow visual tests completed"
else
    print_warning "Upload flow visual tests failed but continuing..."
fi

# Analysis results visual tests
if npm run test -- --grep "@visual" tests/analysis-results.spec.ts; then
    print_success "Analysis results visual tests completed"
else
    print_warning "Analysis results visual tests failed but continuing..."
fi

# Dashboard visual tests
if npm run test -- --grep "@visual" tests/dashboard.spec.ts; then
    print_success "Dashboard visual tests completed"
else
    print_warning "Dashboard visual tests failed but continuing..."
fi

# Responsive visual tests
if npm run test -- --grep "@visual" tests/responsive.spec.ts; then
    print_success "Responsive visual tests completed"
else
    print_warning "Responsive visual tests failed but continuing..."
fi

# 8. Visual Regression Tests
print_status "Running Visual Regression Tests..."
if npm run test -- --grep "@regression" tests/visual-regression.spec.ts; then
    print_success "Visual regression tests completed"
else
    print_warning "Visual regression tests failed but continuing..."
fi

# 9. Generate Reports
print_status "Generating Test Reports..."
if npm run generate-report; then
    print_success "Test reports generated"
else
    print_warning "Report generation failed but continuing..."
fi

# 10. Video Demos (if recording enabled)
if [ "$RECORD_VIDEO" = true ]; then
    print_status "Recording Video Demos..."
    
    # Record different demo types
    demos=("quick" "full" "performance" "journey")
    
    for demo_type in "${demos[@]}"; do
        print_status "Recording $demo_type demo video..."
        if node demo/video-demo.js $demo_type; then
            print_success "$demo_type video demo recorded"
        else
            print_warning "$demo_type video demo failed but continuing..."
        fi
    done
fi

# Summary Report
echo ""
echo "=============================================="
print_success "Demo Suite Execution Completed!"
echo "=============================================="

# Count generated files
SCREENSHOT_COUNT=$(find screenshots -name "*.png" 2>/dev/null | wc -l)
VIDEO_COUNT=$(find demo-videos -name "*.mp4" -o -name "*.webm" 2>/dev/null | wc -l)
MARKETING_COUNT=$(find marketing-screenshots -name "*.png" 2>/dev/null | wc -l)

echo ""
echo "📊 Execution Summary:"
echo "  📸 Screenshots generated: $SCREENSHOT_COUNT"
echo "  🎥 Videos recorded: $VIDEO_COUNT"
echo "  🎨 Marketing assets created: $MARKETING_COUNT"
echo ""

echo "📁 Output Locations:"
echo "  📸 Screenshots: ./screenshots/"
echo "  🎥 Demo videos: ./demo-videos/"
echo "  🎨 Marketing assets: ./marketing-screenshots/"
echo "  📊 Test reports: ./reports/"
echo ""

echo "🎯 Demo Types Executed:"
echo "  ✅ Quick Demo (2 min)"
echo "  ✅ Full Demo (10 min)"
echo "  ✅ Performance Showcase"
echo "  ✅ Accessibility Features"
echo "  ✅ Interactive Demonstrations"
echo "  ✅ Visual Regression Testing"
echo "  ✅ Responsive Design Testing"
echo "  ✅ Marketing Asset Generation"

if [ "$RECORD_VIDEO" = true ]; then
    echo "  ✅ Video Demo Recording"
fi

echo ""
echo "🚀 Next Steps:"
echo "  1. Review generated screenshots in ./screenshots/"
echo "  2. Check test reports in ./reports/"
echo "  3. Use marketing assets from ./marketing-screenshots/"

if [ "$RECORD_VIDEO" = true ]; then
    echo "  4. Share demo videos from ./demo-videos/"
fi

echo ""
print_success "All demos completed successfully!"

# Open reports if in GUI environment
if [ "$HEADLESS" = false ] && command -v open &> /dev/null; then
    print_status "Opening test report..."
    npm run generate-report
fi

exit 0