#!/bin/bash

# Arbitration Detection System - Clean Shutdown Script
# This script gracefully stops all services and cleans up resources

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

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_DIR}/shutdown.log"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "${LOG_DIR}/shutdown.log"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "${LOG_DIR}/shutdown.log"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "${LOG_DIR}/shutdown.log"
}

# Function to stop services gracefully
stop_services() {
    log "Stopping Arbitration Detection System services..."
    
    cd "${DEPLOY_DIR}"
    
    # Check if docker-compose.yml exists
    if [[ ! -f docker-compose.yml ]]; then
        error "docker-compose.yml not found in ${DEPLOY_DIR}"
    fi
    
    # Get list of running containers
    local running_services
    running_services=$(docker-compose ps --services --filter "status=running" 2>/dev/null || echo "")
    
    if [[ -z "${running_services}" ]]; then
        log "No services are currently running"
        return 0
    fi
    
    info "Running services: $(echo "${running_services}" | tr '\n' ' ')"
    
    # Stop services in reverse order of startup for graceful shutdown
    log "Stopping application services..."
    
    # Stop frontend and proxy first (stop accepting new requests)
    info "Stopping frontend and proxy services..."
    docker-compose stop nginx frontend 2>/dev/null || warn "Failed to stop frontend/proxy services"
    
    # Stop application services
    info "Stopping application services..."
    docker-compose stop backend graphql mobile-sync blockchain-service integration-service 2>/dev/null || warn "Failed to stop application services"
    
    # Stop processing services
    info "Stopping processing services..."
    docker-compose stop spark-worker spark-master health-checker 2>/dev/null || warn "Failed to stop processing services"
    
    # Stop monitoring services
    info "Stopping monitoring services..."
    docker-compose stop prometheus grafana jaeger fluentd kibana 2>/dev/null || warn "Failed to stop monitoring services"
    
    # Finally stop infrastructure services
    info "Stopping infrastructure services..."
    docker-compose stop elasticsearch rabbitmq chroma redis postgres 2>/dev/null || warn "Failed to stop infrastructure services"
    
    # Give containers time to shutdown gracefully
    sleep 5
    
    # Force stop any remaining containers
    local still_running
    still_running=$(docker-compose ps --services --filter "status=running" 2>/dev/null || echo "")
    
    if [[ -n "${still_running}" ]]; then
        warn "Some services are still running, forcing stop..."
        docker-compose kill 2>/dev/null || warn "Failed to force kill remaining services"
    fi
    
    log "All services stopped successfully"
}

# Function to clean up containers and networks
cleanup_containers() {
    log "Cleaning up containers and networks..."
    
    cd "${DEPLOY_DIR}"
    
    # Remove containers
    info "Removing containers..."
    docker-compose rm -f -v 2>/dev/null || warn "Failed to remove some containers"
    
    # Remove anonymous volumes
    info "Removing anonymous volumes..."
    docker volume prune -f 2>/dev/null || warn "Failed to prune volumes"
    
    # Remove network
    info "Removing network..."
    docker-compose down --remove-orphans 2>/dev/null || warn "Failed to remove network"
    
    log "Cleanup completed"
}

# Function to backup important data
backup_data() {
    log "Creating backup of important data..."
    
    local backup_dir="${SCRIPT_DIR}/backups/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "${backup_dir}"
    
    cd "${DEPLOY_DIR}"
    
    # Backup database if it's running
    if docker-compose ps postgres --format json 2>/dev/null | grep -q '"State":"running"'; then
        info "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U arbitration_user arbitration_db > "${backup_dir}/postgres_backup.sql" 2>/dev/null || warn "Failed to backup PostgreSQL"
    fi
    
    # Backup Redis data if it's running
    if docker-compose ps redis --format json 2>/dev/null | grep -q '"State":"running"'; then
        info "Backing up Redis data..."
        docker-compose exec -T redis redis-cli BGSAVE >/dev/null 2>&1 || warn "Failed to backup Redis"
        sleep 2
        docker-compose exec -T redis cat /data/dump.rdb > "${backup_dir}/redis_backup.rdb" 2>/dev/null || warn "Failed to export Redis backup"
    fi
    
    # Backup configuration files
    info "Backing up configuration files..."
    cp -r config "${backup_dir}/" 2>/dev/null || warn "Failed to backup config files"
    cp .env "${backup_dir}/" 2>/dev/null || warn "Failed to backup .env file"
    
    # Backup logs
    info "Backing up logs..."
    cp -r "${LOG_DIR}" "${backup_dir}/" 2>/dev/null || warn "Failed to backup logs"
    
    log "Backup completed: ${backup_dir}"
}

# Function to show cleanup options
show_cleanup_options() {
    echo ""
    echo -e "${BLUE}=== Cleanup Options ===${NC}"
    echo ""
    echo "The system has been stopped. Choose additional cleanup options:"
    echo ""
    echo -e "${GREEN}1)${NC} Keep all data (volumes and images) - Quick restart possible"
    echo -e "${YELLOW}2)${NC} Remove containers only - Keep volumes and images"
    echo -e "${RED}3)${NC} Full cleanup - Remove everything (containers, volumes, images)"
    echo -e "${BLUE}4)${NC} Exit without additional cleanup"
    echo ""
    
    read -p "Select option (1-4): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            log "Keeping all data for quick restart"
            ;;
        2)
            log "Removing containers only..."
            cleanup_containers
            ;;
        3)
            log "Performing full cleanup..."
            cleanup_containers
            
            # Remove images
            info "Removing custom images..."
            docker rmi arbitration-backend:latest 2>/dev/null || warn "Backend image not found"
            docker rmi arbitration-frontend:latest 2>/dev/null || warn "Frontend image not found"
            docker rmi arbitration-integrations:latest 2>/dev/null || warn "Integrations image not found"
            
            # Remove volumes
            info "Removing volumes..."
            docker-compose down -v 2>/dev/null || warn "Failed to remove volumes"
            
            log "Full cleanup completed"
            ;;
        4)
            log "Exiting without additional cleanup"
            ;;
        *)
            warn "Invalid option selected, no additional cleanup performed"
            ;;
    esac
}

# Function to display system status after shutdown
show_status() {
    echo ""
    echo -e "${BLUE}=== Shutdown Status ===${NC}"
    echo ""
    
    cd "${DEPLOY_DIR}"
    
    # Check if any services are still running
    local running_services
    running_services=$(docker-compose ps --services --filter "status=running" 2>/dev/null || echo "")
    
    if [[ -n "${running_services}" ]]; then
        echo -e "${RED}⚠️  Some services are still running:${NC}"
        echo "${running_services}"
        echo ""
        echo -e "To force stop all services, run: ${YELLOW}docker-compose kill${NC}"
    else
        echo -e "${GREEN}✅ All services stopped successfully${NC}"
    fi
    
    # Show Docker resource usage
    echo ""
    echo -e "${BLUE}Docker Resources:${NC}"
    
    local containers_count
    containers_count=$(docker ps -a --format "table {{.Names}}" | grep -E "arbitration|chroma|postgres|redis|elasticsearch" | wc -l)
    echo -e "  Containers: ${containers_count}"
    
    local images_count
    images_count=$(docker images --format "table {{.Repository}}" | grep -E "arbitration|chroma|postgres|redis|elasticsearch" | wc -l)
    echo -e "  Images: ${images_count}"
    
    local volumes_count
    volumes_count=$(docker volume ls --format "table {{.Name}}" | grep -E "deploy|arbitration" | wc -l)
    echo -e "  Volumes: ${volumes_count}"
    
    echo ""
    echo -e "${YELLOW}Logs saved to:${NC} ${LOG_DIR}/shutdown.log"
    echo -e "${YELLOW}Backups location:${NC} ${SCRIPT_DIR}/backups/"
    echo ""
    echo -e "To restart the system, run: ${GREEN}./start.sh${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║         Arbitration Detection System - Clean Shutdown        ║"
    echo "║                                                               ║"
    echo "║  Gracefully stopping all services...                         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Ask user if they want to backup data first
    echo ""
    read -p "Do you want to backup data before shutdown? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        backup_data
    fi
    
    stop_services
    show_cleanup_options
    show_status
    
    echo ""
    echo -e "${GREEN}Shutdown completed successfully! 👋${NC}"
}

# Handle script interruption
trap 'error "Script interrupted by user"' INT TERM

# Run main function
main "$@"