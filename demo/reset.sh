#!/bin/bash

# Arbitration Detection System - Reset to Clean State Script
# This script completely resets the system to a clean initial state

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

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_DIR}/reset.log"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "${LOG_DIR}/reset.log"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "${LOG_DIR}/reset.log"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "${LOG_DIR}/reset.log"
}

# Function to confirm reset operation
confirm_reset() {
    echo -e "${RED}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                        ⚠️  WARNING ⚠️                          ║"
    echo "║                                                               ║"
    echo "║  This will COMPLETELY RESET the system to clean state:       ║"
    echo "║                                                               ║"
    echo "║  • Stop all running services                                  ║"
    echo "║  • Remove all containers and networks                         ║"
    echo "║  • Delete all volumes and data                                ║"
    echo "║  • Remove custom Docker images                                ║"
    echo "║  • Clear all logs and temporary files                         ║"
    echo "║  • Reset configuration to defaults                            ║"
    echo "║                                                               ║"
    echo "║  ALL DATA WILL BE PERMANENTLY LOST!                          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    read -p "Are you absolutely sure you want to reset? Type 'RESET' to confirm: " -r
    echo ""
    
    if [[ $REPLY != "RESET" ]]; then
        log "Reset cancelled by user"
        exit 0
    fi
    
    echo ""
    read -p "Do you want to create a backup before reset? (Y/n): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        return 0  # Create backup
    else
        return 1  # Skip backup
    fi
}

# Function to create emergency backup
create_emergency_backup() {
    log "Creating emergency backup before reset..."
    
    local backup_dir="${BACKUP_DIR}/emergency_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "${backup_dir}"
    
    cd "${DEPLOY_DIR}" 2>/dev/null || warn "Deploy directory not found"
    
    # Backup database if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "arbitration-db"; then
        info "Creating database backup..."
        if docker ps --format "{{.Names}}" | grep -q "arbitration-db"; then
            docker exec arbitration-db pg_dump -U arbitration_user arbitration_db > "${backup_dir}/postgres_backup.sql" 2>/dev/null || warn "Failed to backup database"
        else
            warn "Database container exists but is not running, cannot backup"
        fi
    fi
    
    # Backup Redis data if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "arbitration-redis"; then
        info "Creating Redis backup..."
        if docker ps --format "{{.Names}}" | grep -q "arbitration-redis"; then
            docker exec arbitration-redis redis-cli BGSAVE >/dev/null 2>&1 || warn "Failed to trigger Redis backup"
            sleep 2
            docker exec arbitration-redis cat /data/dump.rdb > "${backup_dir}/redis_backup.rdb" 2>/dev/null || warn "Failed to export Redis backup"
        else
            warn "Redis container exists but is not running, cannot backup"
        fi
    fi
    
    # Backup configuration files
    info "Backing up configuration..."
    [[ -d config ]] && cp -r config "${backup_dir}/" 2>/dev/null || warn "Config directory not found"
    [[ -f .env ]] && cp .env "${backup_dir}/" 2>/dev/null || warn ".env file not found"
    [[ -f .env.local ]] && cp .env.local "${backup_dir}/" 2>/dev/null
    
    # Backup custom certificates
    [[ -d ssl ]] && cp -r ssl "${backup_dir}/" 2>/dev/null
    
    # Backup logs
    [[ -d "${LOG_DIR}" ]] && cp -r "${LOG_DIR}" "${backup_dir}/" 2>/dev/null
    
    # Create inventory of what was running
    info "Creating system inventory..."
    {
        echo "# System Inventory - $(date)"
        echo "## Running Containers"
        docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"
        echo ""
        echo "## All Containers"
        docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null || echo "No containers found"
        echo ""
        echo "## Images"
        docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null || echo "No images found"
        echo ""
        echo "## Volumes"
        docker volume ls --format "table {{.Name}}\t{{.Driver}}" 2>/dev/null || echo "No volumes found"
        echo ""
        echo "## Networks"
        docker network ls --format "table {{.Name}}\t{{.Driver}}" 2>/dev/null || echo "No networks found"
    } > "${backup_dir}/system_inventory.txt"
    
    log "Emergency backup created: ${backup_dir}"
}

# Function to stop all services
stop_all_services() {
    log "Stopping all services..."
    
    cd "${DEPLOY_DIR}" 2>/dev/null || warn "Deploy directory not found"
    
    # Try graceful shutdown first
    if [[ -f docker-compose.yml ]]; then
        info "Attempting graceful shutdown..."
        docker-compose down --timeout 30 2>/dev/null || warn "Graceful shutdown failed"
    fi
    
    # Force stop any remaining arbitration-related containers
    info "Force stopping any remaining containers..."
    local containers
    containers=$(docker ps --format "{{.Names}}" | grep -E "arbitration|chroma" 2>/dev/null || echo "")
    
    if [[ -n "${containers}" ]]; then
        echo "${containers}" | xargs -r docker kill 2>/dev/null || warn "Failed to kill some containers"
        echo "${containers}" | xargs -r docker rm -f 2>/dev/null || warn "Failed to remove some containers"
    fi
    
    log "All services stopped"
}

# Function to remove all containers
remove_containers() {
    log "Removing all containers..."
    
    # Remove all arbitration-related containers
    local all_containers
    all_containers=$(docker ps -a --format "{{.Names}}" | grep -E "arbitration|chroma" 2>/dev/null || echo "")
    
    if [[ -n "${all_containers}" ]]; then
        info "Removing containers: $(echo "${all_containers}" | tr '\n' ' ')"
        echo "${all_containers}" | xargs -r docker rm -f -v 2>/dev/null || warn "Failed to remove some containers"
    else
        info "No arbitration-related containers found"
    fi
    
    # Remove containers from docker-compose if file exists
    cd "${DEPLOY_DIR}" 2>/dev/null || return 0
    if [[ -f docker-compose.yml ]]; then
        docker-compose rm -f -v 2>/dev/null || warn "Failed to remove compose containers"
    fi
    
    log "Container removal completed"
}

# Function to remove all images
remove_images() {
    log "Removing custom Docker images..."
    
    # List of custom images to remove
    local custom_images=(
        "arbitration-backend:latest"
        "arbitration-frontend:latest"
        "arbitration-integrations:latest"
        "arbitration-blockchain:latest"
        "arbitration-mobile-sync:latest"
    )
    
    for image in "${custom_images[@]}"; do
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image}$"; then
            info "Removing image: ${image}"
            docker rmi "${image}" 2>/dev/null || warn "Failed to remove image: ${image}"
        fi
    done
    
    # Remove dangling images
    info "Removing dangling images..."
    docker image prune -f 2>/dev/null || warn "Failed to prune dangling images"
    
    log "Image removal completed"
}

# Function to remove all volumes
remove_volumes() {
    log "Removing all volumes..."
    
    # Remove named volumes from docker-compose
    cd "${DEPLOY_DIR}" 2>/dev/null || return 0
    if [[ -f docker-compose.yml ]]; then
        docker-compose down -v 2>/dev/null || warn "Failed to remove compose volumes"
    fi
    
    # Remove any remaining arbitration-related volumes
    local volumes
    volumes=$(docker volume ls --format "{{.Name}}" | grep -E "deploy|arbitration" 2>/dev/null || echo "")
    
    if [[ -n "${volumes}" ]]; then
        info "Removing volumes: $(echo "${volumes}" | tr '\n' ' ')"
        echo "${volumes}" | xargs -r docker volume rm -f 2>/dev/null || warn "Failed to remove some volumes"
    fi
    
    # Remove unused volumes
    info "Removing unused volumes..."
    docker volume prune -f 2>/dev/null || warn "Failed to prune unused volumes"
    
    log "Volume removal completed"
}

# Function to remove networks
remove_networks() {
    log "Removing networks..."
    
    # Remove docker-compose networks
    cd "${DEPLOY_DIR}" 2>/dev/null || return 0
    if [[ -f docker-compose.yml ]]; then
        docker-compose down --remove-orphans 2>/dev/null || warn "Failed to remove compose networks"
    fi
    
    # Remove any remaining arbitration-related networks
    local networks
    networks=$(docker network ls --format "{{.Name}}" | grep -E "arbitration|deploy" 2>/dev/null || echo "")
    
    if [[ -n "${networks}" ]]; then
        info "Removing networks: $(echo "${networks}" | tr '\n' ' ')"
        echo "${networks}" | xargs -r docker network rm 2>/dev/null || warn "Failed to remove some networks"
    fi
    
    # Prune unused networks
    info "Removing unused networks..."
    docker network prune -f 2>/dev/null || warn "Failed to prune unused networks"
    
    log "Network removal completed"
}

# Function to clean file system
clean_filesystem() {
    log "Cleaning file system..."
    
    # Clean logs
    if [[ -d "${LOG_DIR}" ]]; then
        info "Cleaning logs..."
        rm -rf "${LOG_DIR}"/* 2>/dev/null || warn "Failed to clean some logs"
    fi
    
    # Clean temporary files
    info "Cleaning temporary files..."
    [[ -d "${SCRIPT_DIR}/tmp" ]] && rm -rf "${SCRIPT_DIR}/tmp" 2>/dev/null
    [[ -d "${DEPLOY_DIR}/tmp" ]] && rm -rf "${DEPLOY_DIR}/tmp" 2>/dev/null
    
    # Reset configuration to defaults (keep backups)
    cd "${DEPLOY_DIR}" 2>/dev/null || return 0
    
    if [[ -f .env && -f .env.example ]]; then
        info "Resetting .env to defaults..."
        cp .env.example .env 2>/dev/null || warn "Failed to reset .env file"
    fi
    
    # Remove generated SSL certificates (will be regenerated)
    if [[ -d ssl ]]; then
        info "Removing generated SSL certificates..."
        rm -f ssl/cert.pem ssl/key.pem 2>/dev/null || warn "Failed to remove SSL certificates"
    fi
    
    # Clean up any lock files or PIDs
    find "${SCRIPT_DIR}" -name "*.lock" -delete 2>/dev/null || true
    find "${SCRIPT_DIR}" -name "*.pid" -delete 2>/dev/null || true
    
    log "File system cleaning completed"
}

# Function to verify clean state
verify_clean_state() {
    log "Verifying clean state..."
    
    local issues=0
    
    # Check for remaining containers
    local containers
    containers=$(docker ps -a --format "{{.Names}}" | grep -E "arbitration|chroma" 2>/dev/null || echo "")
    if [[ -n "${containers}" ]]; then
        warn "Remaining containers found: ${containers}"
        ((issues++))
    fi
    
    # Check for remaining images
    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "arbitration" 2>/dev/null || echo "")
    if [[ -n "${images}" ]]; then
        warn "Remaining custom images found: ${images}"
        ((issues++))
    fi
    
    # Check for remaining volumes
    local volumes
    volumes=$(docker volume ls --format "{{.Name}}" | grep -E "deploy|arbitration" 2>/dev/null || echo "")
    if [[ -n "${volumes}" ]]; then
        warn "Remaining volumes found: ${volumes}"
        ((issues++))
    fi
    
    # Check for remaining networks
    local networks
    networks=$(docker network ls --format "{{.Name}}" | grep -E "arbitration|deploy" 2>/dev/null || echo "")
    if [[ -n "${networks}" ]]; then
        warn "Remaining networks found: ${networks}"
        ((issues++))
    fi
    
    if [[ $issues -eq 0 ]]; then
        log "✅ System successfully reset to clean state"
        return 0
    else
        warn "⚠️  System reset completed with ${issues} remaining issues"
        return 1
    fi
}

# Function to show final status
show_final_status() {
    echo ""
    echo -e "${BLUE}=== Reset Complete ===${NC}"
    echo ""
    
    verify_clean_state
    
    echo ""
    echo -e "${GREEN}System Reset Summary:${NC}"
    echo -e "  ✅ All services stopped"
    echo -e "  ✅ All containers removed"
    echo -e "  ✅ All volumes deleted"
    echo -e "  ✅ All networks removed"
    echo -e "  ✅ Custom images removed"
    echo -e "  ✅ File system cleaned"
    echo -e "  ✅ Configuration reset to defaults"
    echo ""
    
    [[ -d "${BACKUP_DIR}" ]] && echo -e "${YELLOW}Backups preserved in:${NC} ${BACKUP_DIR}"
    echo -e "${YELLOW}Reset log saved to:${NC} ${LOG_DIR}/reset.log"
    echo ""
    echo -e "To start fresh, run: ${GREEN}./start.sh${NC}"
    echo ""
    
    # Show current Docker state
    echo -e "${BLUE}Current Docker State:${NC}"
    echo -e "  Containers: $(docker ps -a --format "{{.Names}}" | wc -l) total"
    echo -e "  Images: $(docker images --format "{{.Repository}}" | wc -l) total"
    echo -e "  Volumes: $(docker volume ls --format "{{.Name}}" | wc -l) total"
    echo -e "  Networks: $(docker network ls --format "{{.Name}}" | tail -n +2 | wc -l) total"
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║        Arbitration Detection System - Complete Reset         ║"
    echo "║                                                               ║"
    echo "║  This will reset the system to a completely clean state      ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    if confirm_reset; then
        create_emergency_backup
    fi
    
    stop_all_services
    remove_containers
    remove_images
    remove_volumes
    remove_networks
    clean_filesystem
    show_final_status
    
    echo ""
    echo -e "${GREEN}System reset completed successfully! 🧹${NC}"
    echo -e "${BLUE}You can now start fresh with ./start.sh${NC}"
}

# Handle script interruption
trap 'error "Script interrupted by user"' INT TERM

# Run main function
main "$@"