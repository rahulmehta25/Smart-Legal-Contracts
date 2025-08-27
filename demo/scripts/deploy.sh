#!/bin/bash

# Production Deployment Script for Arbitration Detection System
# This script handles zero-downtime deployment with rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMPOSE_FILE="$PROJECT_ROOT/demo/production/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"
BACKUP_DIR="$PROJECT_ROOT/backups"
LOG_FILE="$PROJECT_ROOT/logs/deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
SERVICE=""
TAG="latest"
SKIP_BACKUP=false
SKIP_TESTS=false
FORCE_DEPLOY=false
DRY_RUN=false

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    log "INFO" "$1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING" "$1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR" "$1"
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
    log "DEBUG" "$1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the Arbitration Detection System to production.

OPTIONS:
    -e, --environment ENV    Target environment (default: production)
    -s, --service SERVICE    Deploy specific service only (backend|frontend|nginx)
    -t, --tag TAG           Docker image tag to deploy (default: latest)
    --skip-backup           Skip database backup before deployment
    --skip-tests            Skip pre-deployment tests
    --force                 Force deployment even if health checks fail
    --dry-run              Show what would be deployed without executing
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Full production deployment
    $0 -s backend -t v1.2.3              # Deploy only backend with specific tag
    $0 --skip-backup --force              # Quick deployment without backup
    $0 --dry-run                          # Preview deployment actions

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        exit 1
    fi

    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi

    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        print_error "Environment file not found: $ENV_FILE"
        print_info "Please create the environment file with required variables"
        exit 1
    fi

    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_error "Docker compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    # Create necessary directories
    mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

    print_info "Prerequisites check completed"
}

# Load environment variables
load_environment() {
    print_info "Loading environment variables..."
    set -a
    source "$ENV_FILE"
    set +a
    
    # Validate required environment variables
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            print_error "Required environment variable not set: $var"
            exit 1
        fi
    done
}

# Health check function
health_check() {
    local service=$1
    local max_attempts=${2:-30}
    local attempt=0

    print_info "Performing health check for $service..."

    case $service in
        "backend")
            local url="http://localhost:8000/health"
            ;;
        "frontend")
            local url="http://localhost:3000/api/health"
            ;;
        *)
            print_warning "Unknown service for health check: $service"
            return 0
            ;;
    esac

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s "$url" &> /dev/null; then
            print_info "$service health check passed"
            return 0
        fi
        
        attempt=$((attempt + 1))
        print_debug "Health check attempt $attempt/$max_attempts for $service"
        sleep 10
    done

    print_error "$service health check failed after $max_attempts attempts"
    return 1
}

# Backup database
backup_database() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        print_warning "Skipping database backup as requested"
        return 0
    fi

    print_info "Creating database backup..."
    
    local backup_file="$BACKUP_DIR/backup-$(date +%Y%m%d-%H%M%S).sql"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would create backup: $backup_file"
        return 0
    fi

    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        print_info "Database backup created: $backup_file"
        # Keep only last 10 backups
        ls -t "$BACKUP_DIR"/backup-*.sql | tail -n +11 | xargs -r rm
    else
        print_error "Database backup failed"
        return 1
    fi
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_warning "Skipping pre-deployment tests as requested"
        return 0
    fi

    print_info "Running pre-deployment tests..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would run pre-deployment tests"
        return 0
    fi

    # Run backend tests
    if [[ -z "$SERVICE" || "$SERVICE" == "backend" ]]; then
        print_info "Running backend tests..."
        docker-compose -f "$COMPOSE_FILE" exec -T backend python -m pytest tests/ --maxfail=5
        if [[ $? -ne 0 ]]; then
            print_error "Backend tests failed"
            return 1
        fi
    fi

    # Run frontend tests
    if [[ -z "$SERVICE" || "$SERVICE" == "frontend" ]]; then
        print_info "Running frontend tests..."
        docker-compose -f "$COMPOSE_FILE" exec -T frontend npm run test:ci
        if [[ $? -ne 0 ]]; then
            print_error "Frontend tests failed"
            return 1
        fi
    fi

    print_info "Pre-deployment tests passed"
}

# Deploy service
deploy_service() {
    local service_name=$1
    
    print_info "Deploying $service_name..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would deploy $service_name with tag: $TAG"
        return 0
    fi

    # Pull latest images
    print_info "Pulling latest images for $service_name..."
    docker-compose -f "$COMPOSE_FILE" pull "$service_name"

    # Stop and remove old containers
    print_info "Stopping old $service_name containers..."
    docker-compose -f "$COMPOSE_FILE" stop "$service_name"
    docker-compose -f "$COMPOSE_FILE" rm -f "$service_name"

    # Start new containers
    print_info "Starting new $service_name containers..."
    docker-compose -f "$COMPOSE_FILE" up -d "$service_name"

    # Wait for service to be ready
    sleep 30

    # Health check
    if ! health_check "$service_name"; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            print_error "Health check failed for $service_name. Deployment aborted."
            print_info "Use --force to override health check failures"
            return 1
        else
            print_warning "Health check failed but continuing due to --force flag"
        fi
    fi

    print_info "$service_name deployed successfully"
}

# Full deployment
deploy_all() {
    print_info "Starting full deployment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would perform full deployment with tag: $TAG"
        return 0
    fi

    # Deploy infrastructure services first
    print_info "Deploying infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis

    # Wait for infrastructure to be ready
    print_info "Waiting for infrastructure services..."
    sleep 60

    # Deploy application services
    local services=("backend" "frontend" "nginx")
    for service in "${services[@]}"; do
        deploy_service "$service"
    done

    # Deploy monitoring services
    print_info "Deploying monitoring services..."
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana loki promtail

    print_info "Full deployment completed successfully"
}

# Rollback function
rollback() {
    local backup_file=$1
    
    print_error "Deployment failed. Initiating rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would rollback using backup: $backup_file"
        return 0
    fi

    # Restore database if backup exists
    if [[ -n "$backup_file" && -f "$backup_file" ]]; then
        print_info "Restoring database from backup..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$backup_file"
    fi

    # Restart with previous version
    print_info "Restarting services with previous version..."
    docker-compose -f "$COMPOSE_FILE" down
    docker-compose -f "$COMPOSE_FILE" up -d

    print_info "Rollback completed"
}

# Main deployment process
main() {
    local start_time=$(date +%s)
    local backup_file=""

    print_info "=== Starting Arbitration Detection System Deployment ==="
    print_info "Environment: $ENVIRONMENT"
    print_info "Service: ${SERVICE:-all}"
    print_info "Tag: $TAG"
    print_info "Dry Run: $DRY_RUN"

    # Check prerequisites
    check_prerequisites

    # Load environment
    load_environment

    # Create backup
    if backup_database; then
        backup_file="$BACKUP_DIR/backup-$(date +%Y%m%d-%H%M%S).sql"
    else
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            print_error "Backup failed. Deployment aborted."
            exit 1
        fi
    fi

    # Run tests
    if ! run_tests; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            print_error "Tests failed. Deployment aborted."
            exit 1
        fi
    fi

    # Deploy services
    if [[ -n "$SERVICE" ]]; then
        if ! deploy_service "$SERVICE"; then
            rollback "$backup_file"
            exit 1
        fi
    else
        if ! deploy_all; then
            rollback "$backup_file"
            exit 1
        fi
    fi

    # Final health checks
    print_info "Performing final system health checks..."
    local services_to_check=("backend" "frontend")
    if [[ -n "$SERVICE" && "$SERVICE" != "nginx" ]]; then
        services_to_check=("$SERVICE")
    fi

    for service in "${services_to_check[@]}"; do
        if ! health_check "$service" 10; then
            print_warning "Final health check failed for $service"
            if [[ "$FORCE_DEPLOY" != "true" ]]; then
                rollback "$backup_file"
                exit 1
            fi
        fi
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    print_info "=== Deployment Completed Successfully ==="
    print_info "Duration: ${duration}s"
    print_info "Backup created: ${backup_file:-N/A}"
    print_info "Services deployed: ${SERVICE:-all}"
    
    # Show running containers
    print_info "Running containers:"
    docker-compose -f "$COMPOSE_FILE" ps
}

# Handle script interruption
cleanup() {
    print_warning "Deployment interrupted. Cleaning up..."
    exit 130
}

trap cleanup INT TERM

# Parse arguments and run main function
parse_args "$@"
main