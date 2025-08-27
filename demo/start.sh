#!/bin/bash

# Arbitration Detection System - One-Click Demo Startup Script
# This script starts the complete system with all services

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/deploy"
LOG_DIR="${SCRIPT_DIR}/logs"
BACKUP_DIR="${SCRIPT_DIR}/backups"

# Create necessary directories
mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_DIR}/startup.log"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "${LOG_DIR}/startup.log"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "${LOG_DIR}/startup.log"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "${LOG_DIR}/startup.log"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker and Docker Compose
check_dependencies() {
    log "Checking system dependencies..."
    
    if ! command_exists docker; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running. Please start Docker first."
    fi
    
    # Check available system resources
    local total_mem=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
    local available_disk=$(df -h "${SCRIPT_DIR}" | awk 'NR==2{print $4}')
    
    info "System Resources:"
    info "  Memory: ${total_mem}GB"
    info "  Available Disk: ${available_disk}"
    
    # Check if we have enough resources
    if (( $(echo "${total_mem} < 8.0" | bc -l) )); then
        warn "System has less than 8GB RAM. Performance may be degraded."
    fi
    
    log "Dependencies check completed successfully"
}

# Function to setup environment
setup_environment() {
    log "Setting up environment..."
    
    cd "${DEPLOY_DIR}"
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            log "Creating .env file from .env.example..."
            cp .env.example .env
            warn "Please update the .env file with your actual configuration before running in production"
        else
            error ".env.example file not found. Cannot create environment configuration."
        fi
    fi
    
    # Create SSL directory and self-signed certificates for development
    mkdir -p ssl
    if [[ ! -f ssl/cert.pem || ! -f ssl/key.pem ]]; then
        log "Generating self-signed SSL certificates for development..."
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=CA/L=SF/O=ArbitrationDetector/CN=localhost" \
            2>/dev/null || warn "Could not generate SSL certificates. HTTPS may not work."
    fi
    
    # Create configuration files if they don't exist
    mkdir -p config/{grafana/dashboards,grafana/datasources,prometheus,fluentd,nginx/conf.d}
    mkdir -p init/postgres
    
    log "Environment setup completed"
}

# Function to build custom images
build_images() {
    log "Building custom Docker images..."
    
    cd "${SCRIPT_DIR}"
    
    # Build backend image
    if [[ -f Dockerfile ]]; then
        log "Building backend image..."
        docker build -t arbitration-backend:latest . || error "Failed to build backend image"
    else
        warn "Dockerfile not found. Using pre-built images."
    fi
    
    # Build frontend image
    if [[ -d frontend && -f frontend/Dockerfile ]]; then
        log "Building frontend image..."
        docker build -t arbitration-frontend:latest ./frontend || warn "Failed to build frontend image"
    else
        warn "Frontend Dockerfile not found. Using pre-built images."
    fi
    
    # Build integration services image
    if [[ -d integrations && -f integrations/Dockerfile ]]; then
        log "Building integration services image..."
        docker build -t arbitration-integrations:latest ./integrations || warn "Failed to build integrations image"
    fi
    
    log "Image building completed"
}

# Function to start services in order
start_services() {
    log "Starting Arbitration Detection System services..."
    
    cd "${DEPLOY_DIR}"
    
    # Start infrastructure services first
    log "Starting infrastructure services..."
    docker-compose up -d postgres redis chroma elasticsearch rabbitmq
    
    # Wait for infrastructure services to be ready
    log "Waiting for infrastructure services to be ready..."
    
    # Wait for PostgreSQL
    info "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U arbitration_user -d arbitration_db >/dev/null 2>&1; then
            log "PostgreSQL is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "PostgreSQL failed to start within 30 attempts"
        fi
        sleep 2
    done
    
    # Wait for Redis
    info "Waiting for Redis..."
    for i in {1..15}; do
        if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
            log "Redis is ready"
            break
        fi
        if [[ $i -eq 15 ]]; then
            error "Redis failed to start within 15 attempts"
        fi
        sleep 2
    done
    
    # Wait for Elasticsearch
    info "Waiting for Elasticsearch..."
    for i in {1..30}; do
        if curl -s http://localhost:9200/_cluster/health >/dev/null 2>&1; then
            log "Elasticsearch is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            warn "Elasticsearch may not be ready, continuing anyway"
        fi
        sleep 3
    done
    
    # Wait for Chroma
    info "Waiting for Chroma Vector Database..."
    for i in {1..20}; do
        if curl -s http://localhost:8001/api/v1/heartbeat >/dev/null 2>&1; then
            log "Chroma is ready"
            break
        fi
        if [[ $i -eq 20 ]]; then
            warn "Chroma may not be ready, continuing anyway"
        fi
        sleep 3
    done
    
    # Run database migrations
    log "Running database migrations..."
    if docker-compose exec -T backend python migrations/migration_runner.py 2>/dev/null; then
        log "Database migrations completed successfully"
    else
        warn "Database migrations may have failed or migration script not found"
    fi
    
    # Start application services
    log "Starting application services..."
    docker-compose up -d backend graphql mobile-sync blockchain-service integration-service
    
    # Wait for backend to be ready
    info "Waiting for backend API..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log "Backend API is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "Backend API failed to start within 30 attempts"
        fi
        sleep 3
    done
    
    # Start frontend and proxy
    log "Starting frontend and proxy services..."
    docker-compose up -d frontend nginx
    
    # Wait for frontend
    info "Waiting for frontend..."
    for i in {1..20}; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            log "Frontend is ready"
            break
        fi
        if [[ $i -eq 20 ]]; then
            warn "Frontend may not be ready, continuing anyway"
        fi
        sleep 3
    done
    
    # Start monitoring services
    log "Starting monitoring services..."
    docker-compose up -d prometheus grafana jaeger fluentd kibana
    
    # Start additional services
    log "Starting additional services..."
    docker-compose up -d spark-master spark-worker health-checker
    
    log "All services started successfully!"
}

# Function to show system status and URLs
show_status() {
    log "System Status and Access URLs:"
    echo ""
    echo -e "${BLUE}=== Arbitration Detection System - Running Services ===${NC}"
    echo ""
    echo -e "${GREEN}Core Application:${NC}"
    echo -e "  Frontend:           ${BLUE}http://localhost:3000${NC}"
    echo -e "  Backend API:        ${BLUE}http://localhost:8000${NC}"
    echo -e "  GraphQL Playground: ${BLUE}http://localhost:4000/graphql${NC}"
    echo -e "  API Documentation:  ${BLUE}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${GREEN}Monitoring & Observability:${NC}"
    echo -e "  Grafana Dashboard:  ${BLUE}http://localhost:3001${NC} (admin/admin)"
    echo -e "  Prometheus:         ${BLUE}http://localhost:9090${NC}"
    echo -e "  Jaeger Tracing:     ${BLUE}http://localhost:16686${NC}"
    echo -e "  Kibana Logs:        ${BLUE}http://localhost:5601${NC}"
    echo ""
    echo -e "${GREEN}Data & Search:${NC}"
    echo -e "  Elasticsearch:      ${BLUE}http://localhost:9200${NC}"
    echo -e "  Chroma Vector DB:   ${BLUE}http://localhost:8001${NC}"
    echo -e "  RabbitMQ Management: ${BLUE}http://localhost:15672${NC} (arbitration_user/arbitration_pass)"
    echo ""
    echo -e "${GREEN}Analytics & Processing:${NC}"
    echo -e "  Spark Master UI:    ${BLUE}http://localhost:8080${NC}"
    echo ""
    echo -e "${GREEN}Integration Services:${NC}"
    echo -e "  Integration API:    ${BLUE}http://localhost:8003${NC}"
    echo -e "  Blockchain Service: ${BLUE}http://localhost:8004${NC}"
    echo -e "  Mobile Sync:        ${BLUE}http://localhost:8002${NC}"
    echo ""
    echo -e "${YELLOW}Logs Location:${NC} ${LOG_DIR}"
    echo -e "${YELLOW}Configuration:${NC} ${DEPLOY_DIR}/.env"
    echo ""
    
    # Check service health
    echo -e "${BLUE}=== Service Health Check ===${NC}"
    services=(
        "postgres:5432"
        "redis:6379"
        "elasticsearch:9200"
        "chroma:8001"
        "backend:8000"
        "frontend:3000"
        "grafana:3001"
        "prometheus:9090"
    )
    
    for service in "${services[@]}"; do
        name="${service%:*}"
        port="${service#*:}"
        if nc -z localhost "${port}" 2>/dev/null; then
            echo -e "  ✅ ${name^} - ${GREEN}Running${NC}"
        else
            echo -e "  ❌ ${name^} - ${RED}Not responding${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}System started successfully! 🎉${NC}"
    echo -e "${YELLOW}Note: First startup may take a few minutes to download and initialize all services.${NC}"
    echo ""
    echo -e "To stop the system, run: ${BLUE}./stop.sh${NC}"
    echo -e "To check status, run: ${BLUE}./status.sh${NC}"
    echo -e "To reset the system, run: ${BLUE}./reset.sh${NC}"
}

# Function to create initial demo data
create_demo_data() {
    log "Creating demo data..."
    
    # Wait a bit for backend to be fully ready
    sleep 10
    
    # Create demo data via API calls
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        log "Backend is ready, creating demo data..."
        
        # You can add API calls here to create demo data
        # Example:
        # curl -X POST http://localhost:8000/api/demo/create-sample-data
        
        log "Demo data creation completed (or skipped if not implemented)"
    else
        warn "Backend not ready, skipping demo data creation"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║          Arbitration Detection System - Demo Startup         ║"
    echo "║                                                               ║"
    echo "║  Starting complete system with all services...               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_dependencies
    setup_environment
    build_images
    start_services
    create_demo_data
    show_status
}

# Handle script interruption
trap 'error "Script interrupted by user"' INT TERM

# Run main function
main "$@"