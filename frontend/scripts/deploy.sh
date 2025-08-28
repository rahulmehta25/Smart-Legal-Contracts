#!/bin/bash

# Vercel Deployment Script for Arbitration Detector Frontend
# This script prepares and deploys the frontend to Vercel with proper configuration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="arbitration-detector-frontend"
BUILD_DIR=".next"
VERCEL_TOKEN=${VERCEL_TOKEN:-""}
BACKEND_URL=${BACKEND_URL:-""}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js first."
        exit 1
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed. Please install npm first."
        exit 1
    fi
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        log_warning "Vercel CLI is not installed. Installing it now..."
        npm install -g vercel
    fi
    
    log_success "Prerequisites check passed"
}

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    if [ -z "$BACKEND_URL" ]; then
        log_error "BACKEND_URL environment variable is not set."
        log_error "Please set it with: export BACKEND_URL=https://your-backend-api.com"
        exit 1
    fi
    
    if [ -z "$VERCEL_TOKEN" ]; then
        log_warning "VERCEL_TOKEN is not set. You'll need to authenticate manually."
    fi
    
    # Check if .env.example exists
    if [ ! -f ".env.example" ]; then
        log_error ".env.example file not found. Please create it first."
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if vercel.json exists and is valid
    if [ ! -f "vercel.json" ]; then
        log_error "vercel.json not found. Please create it first."
        exit 1
    fi
    
    # Validate vercel.json syntax
    if ! node -e "JSON.parse(require('fs').readFileSync('vercel.json', 'utf8'))"; then
        log_error "vercel.json has invalid JSON syntax"
        exit 1
    fi
    
    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        log_error "package.json not found"
        exit 1
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build the application
build_application() {
    log_info "Building the application..."
    
    # Install dependencies
    log_info "Installing dependencies..."
    npm ci --production=false
    
    # Run type checking
    log_info "Running type checking..."
    npm run type-check || {
        log_error "Type checking failed. Please fix TypeScript errors before deployment."
        exit 1
    }
    
    # Run linting
    log_info "Running linting..."
    npm run lint || {
        log_warning "Linting issues found. Consider fixing them before deployment."
    }
    
    # Build the application
    log_info "Building Next.js application..."
    npm run build
    
    if [ ! -d "$BUILD_DIR" ]; then
        log_error "Build failed. No build directory found."
        exit 1
    fi
    
    log_success "Application built successfully"
}

# Deploy to Vercel
deploy_to_vercel() {
    local environment=${1:-"production"}
    
    log_info "Deploying to Vercel (environment: $environment)..."
    
    # Set environment variables for deployment
    export NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL"
    export BACKEND_URL="$BACKEND_URL"
    export NEXT_PUBLIC_API_URL="$BACKEND_URL"
    export NEXT_PUBLIC_WS_URL="${BACKEND_URL/https:/wss:}/ws"
    export NEXT_PUBLIC_ENV="$environment"
    
    # Deployment command based on environment
    if [ "$environment" = "production" ]; then
        vercel --prod --confirm
    else
        vercel --confirm
    fi
    
    log_success "Deployment completed"
}

# Set environment variables in Vercel
set_vercel_environment() {
    log_info "Setting environment variables in Vercel..."
    
    # Required environment variables
    declare -a env_vars=(
        "NEXT_PUBLIC_BACKEND_URL=$BACKEND_URL"
        "NEXT_PUBLIC_API_URL=$BACKEND_URL"
        "NEXT_PUBLIC_WS_URL=${BACKEND_URL/https:/wss:}/ws"
        "BACKEND_URL=$BACKEND_URL"
        "NEXT_PUBLIC_ENV=production"
        "NEXT_PUBLIC_APP_NAME=Arbitration Detector"
        "NEXT_PUBLIC_APP_VERSION=1.0.0"
        "NEXT_TELEMETRY_DISABLED=1"
    )
    
    for env_var in "${env_vars[@]}"; do
        IFS='=' read -r key value <<< "$env_var"
        log_info "Setting $key..."
        vercel env add "$key" production <<< "$value" 2>/dev/null || log_warning "Failed to set $key (may already exist)"
    done
    
    log_success "Environment variables configured"
}

# Post-deployment verification
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Get the deployment URL
    local deployment_url=$(vercel ls --scope=personal --meta | grep "$PROJECT_NAME" | head -1 | awk '{print $2}')
    
    if [ -n "$deployment_url" ]; then
        log_info "Deployment URL: https://$deployment_url"
        
        # Basic health check
        if curl -f -s "https://$deployment_url" > /dev/null; then
            log_success "Deployment is accessible"
        else
            log_warning "Deployment may not be fully ready yet"
        fi
    else
        log_warning "Could not determine deployment URL"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Remove any temporary files if created
    # Clean up build artifacts if needed for space
}

# Main deployment flow
main() {
    echo "=================================="
    echo "Vercel Deployment Script"
    echo "Project: $PROJECT_NAME"
    echo "Backend URL: $BACKEND_URL"
    echo "=================================="
    
    # Parse command line arguments
    local environment="production"
    local skip_build=false
    local setup_env=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --preview)
                environment="preview"
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --setup-env)
                setup_env=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --preview      Deploy to preview environment"
                echo "  --skip-build   Skip the build step"
                echo "  --setup-env    Set up environment variables only"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    validate_environment
    
    if [ "$setup_env" = true ]; then
        set_vercel_environment
        exit 0
    fi
    
    pre_deployment_checks
    
    if [ "$skip_build" = false ]; then
        build_application
    fi
    
    deploy_to_vercel "$environment"
    verify_deployment
    
    log_success "Deployment process completed successfully!"
    echo "=================================="
}

# Run main function with all arguments
main "$@"