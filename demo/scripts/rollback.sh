#!/bin/bash

# Rollback Script for Arbitration Detection System
# Handles automated rollback to previous stable version with validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMPOSE_FILE="$PROJECT_ROOT/demo/production/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"
LOG_FILE="$PROJECT_ROOT/logs/rollback.log"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Default settings
ROLLBACK_TYPE="full"  # full, database, application
TARGET_VERSION=""
TARGET_BACKUP=""
SKIP_VALIDATION=false
FORCE_ROLLBACK=false
DRY_RUN=false
AUTO_APPROVE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

Rollback the Arbitration Detection System to a previous stable state.

OPTIONS:
    -t, --type TYPE           Rollback type: full, database, application (default: full)
    -v, --version VERSION     Target version to rollback to
    -b, --backup BACKUP       Specific backup file to restore from
    --skip-validation         Skip post-rollback validation
    --force                   Force rollback without confirmation prompts
    --auto-approve            Automatically approve rollback without user interaction
    --dry-run                Show what would be rolled back without executing
    -h, --help               Show this help message

EXAMPLES:
    $0                                    # Interactive rollback to previous version
    $0 -v v1.2.0                        # Rollback to specific version
    $0 -t database                       # Rollback only database
    $0 -b backup_20240101_120000.sql     # Rollback using specific backup
    $0 --dry-run                         # Preview rollback actions

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                ROLLBACK_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            -b|--backup)
                TARGET_BACKUP="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --force)
                FORCE_ROLLBACK=true
                shift
                ;;
            --auto-approve)
                AUTO_APPROVE=true
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

    # Check required commands
    local required_commands=("docker" "docker-compose" "git")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done

    # Load environment
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
    else
        print_error "Environment file not found: $ENV_FILE"
        exit 1
    fi

    # Create directories
    mkdir -p "$(dirname "$LOG_FILE")"

    print_info "Prerequisites check completed"
}

# Get current version
get_current_version() {
    cd "$PROJECT_ROOT"
    local current_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local current_tag=$(git describe --tags --exact-match 2>/dev/null || echo "")
    local current_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    
    if [[ -n "$current_tag" ]]; then
        echo "$current_tag"
    else
        echo "$current_branch-$(echo "$current_commit" | cut -c1-8)"
    fi
}

# List available versions
list_available_versions() {
    print_info "Available versions:"
    
    cd "$PROJECT_ROOT"
    
    # Show recent tags
    echo "Recent tags:"
    git tag --sort=-version:refname | head -10 | while read -r tag; do
        local commit=$(git rev-list -n 1 "$tag" 2>/dev/null || echo "")
        local date=$(git log -1 --format="%ai" "$commit" 2>/dev/null || echo "Unknown date")
        echo "  $tag (commit: ${commit:0:8}, date: ${date:0:19})"
    done
    
    echo
    echo "Recent commits:"
    git log --oneline --max-count=10 | while read -r line; do
        echo "  $line"
    done
}

# List available backups
list_available_backups() {
    print_info "Available backups:"
    
    if [[ -d "$BACKUP_DIR" ]]; then
        find "$BACKUP_DIR" -name "backup_*.sql*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -10 | while read -r timestamp filepath; do
            local date=$(date -d "@$timestamp" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "Unknown date")
            local size=$(du -h "$filepath" 2>/dev/null | cut -f1)
            echo "  $(basename "$filepath") ($date, $size)"
        done
    else
        print_warning "Backup directory not found: $BACKUP_DIR"
    fi
}

# Interactive version selection
select_target_version() {
    if [[ -n "$TARGET_VERSION" ]]; then
        print_info "Target version specified: $TARGET_VERSION"
        return 0
    fi

    echo
    list_available_versions
    echo
    
    if [[ "$AUTO_APPROVE" != "true" ]]; then
        read -p "Enter target version (tag or commit hash): " TARGET_VERSION
        
        if [[ -z "$TARGET_VERSION" ]]; then
            print_error "No target version specified"
            exit 1
        fi
    else
        print_error "Auto-approve mode requires explicit version specification"
        exit 1
    fi
}

# Interactive backup selection
select_target_backup() {
    if [[ -n "$TARGET_BACKUP" ]]; then
        print_info "Target backup specified: $TARGET_BACKUP"
        
        if [[ ! -f "$TARGET_BACKUP" ]]; then
            # Try to find it in backup directory
            if [[ -f "$BACKUP_DIR/$TARGET_BACKUP" ]]; then
                TARGET_BACKUP="$BACKUP_DIR/$TARGET_BACKUP"
            else
                print_error "Backup file not found: $TARGET_BACKUP"
                exit 1
            fi
        fi
        
        return 0
    fi

    echo
    list_available_backups
    echo
    
    if [[ "$AUTO_APPROVE" != "true" ]]; then
        read -p "Enter backup filename (or press Enter to use latest): " TARGET_BACKUP
        
        if [[ -z "$TARGET_BACKUP" ]]; then
            # Use latest backup
            TARGET_BACKUP=$(find "$BACKUP_DIR" -name "backup_*.sql*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
            
            if [[ -z "$TARGET_BACKUP" ]]; then
                print_error "No backups found"
                exit 1
            fi
            
            print_info "Using latest backup: $(basename "$TARGET_BACKUP")"
        else
            if [[ ! -f "$TARGET_BACKUP" ]]; then
                TARGET_BACKUP="$BACKUP_DIR/$TARGET_BACKUP"
                
                if [[ ! -f "$TARGET_BACKUP" ]]; then
                    print_error "Backup file not found: $TARGET_BACKUP"
                    exit 1
                fi
            fi
        fi
    else
        # Use latest backup in auto-approve mode
        TARGET_BACKUP=$(find "$BACKUP_DIR" -name "backup_*.sql*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        
        if [[ -z "$TARGET_BACKUP" ]]; then
            print_error "No backups found for auto-approve mode"
            exit 1
        fi
        
        print_info "Auto-selected latest backup: $(basename "$TARGET_BACKUP")"
    fi
}

# Confirm rollback
confirm_rollback() {
    if [[ "$FORCE_ROLLBACK" == "true" || "$AUTO_APPROVE" == "true" ]]; then
        return 0
    fi

    local current_version=$(get_current_version)
    
    echo
    echo "=== ROLLBACK CONFIRMATION ==="
    echo "Current version: $current_version"
    echo "Rollback type: $ROLLBACK_TYPE"
    
    if [[ "$ROLLBACK_TYPE" == "full" || "$ROLLBACK_TYPE" == "application" ]]; then
        echo "Target version: ${TARGET_VERSION:-latest}"
    fi
    
    if [[ "$ROLLBACK_TYPE" == "full" || "$ROLLBACK_TYPE" == "database" ]]; then
        echo "Target backup: ${TARGET_BACKUP:-none}"
    fi
    
    echo "Dry run: $DRY_RUN"
    echo "============================="
    echo
    
    read -p "Are you sure you want to proceed with this rollback? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        print_info "Rollback cancelled by user"
        exit 0
    fi
}

# Create emergency backup
create_emergency_backup() {
    print_info "Creating emergency backup before rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would create emergency backup"
        return 0
    fi

    local emergency_backup="$BACKUP_DIR/emergency-rollback-$(date +%Y%m%d_%H%M%S).sql"
    
    # Use backup script
    if "$SCRIPT_DIR/backup.sh" -t full -d "$BACKUP_DIR" --no-uploads; then
        print_info "Emergency backup created successfully"
        return 0
    else
        print_error "Emergency backup failed"
        if [[ "$FORCE_ROLLBACK" != "true" ]]; then
            exit 1
        else
            print_warning "Continuing with rollback despite backup failure (--force enabled)"
            return 0
        fi
    fi
}

# Rollback application code
rollback_application() {
    if [[ "$ROLLBACK_TYPE" != "full" && "$ROLLBACK_TYPE" != "application" ]]; then
        return 0
    fi

    print_info "Rolling back application code to version: $TARGET_VERSION"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would rollback application to: $TARGET_VERSION"
        return 0
    fi

    cd "$PROJECT_ROOT"
    
    # Verify target version exists
    if ! git rev-parse --verify "$TARGET_VERSION" &>/dev/null; then
        print_error "Target version not found: $TARGET_VERSION"
        return 1
    fi

    # Stop services
    print_info "Stopping application services..."
    docker-compose -f "$COMPOSE_FILE" stop backend frontend nginx

    # Checkout target version
    print_info "Checking out target version..."
    git checkout "$TARGET_VERSION"

    # Rebuild and restart services
    print_info "Rebuilding and starting services..."
    docker-compose -f "$COMPOSE_FILE" build backend frontend
    docker-compose -f "$COMPOSE_FILE" up -d backend frontend nginx

    print_info "Application rollback completed"
    return 0
}

# Rollback database
rollback_database() {
    if [[ "$ROLLBACK_TYPE" != "full" && "$ROLLBACK_TYPE" != "database" ]]; then
        return 0
    fi

    if [[ -z "$TARGET_BACKUP" ]]; then
        print_warning "No backup specified for database rollback"
        return 0
    fi

    print_info "Rolling back database using backup: $(basename "$TARGET_BACKUP")"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would rollback database using: $(basename "$TARGET_BACKUP")"
        return 0
    fi

    # Use restore script
    if "$SCRIPT_DIR/restore.sh" --database-only --no-validation --force "$TARGET_BACKUP"; then
        print_info "Database rollback completed"
        return 0
    else
        print_error "Database rollback failed"
        return 1
    fi
}

# Validate rollback
validate_rollback() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        print_info "Skipping rollback validation as requested"
        return 0
    fi

    print_info "Validating rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY RUN] Would validate rollback"
        return 0
    fi

    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 60

    # Use health check script
    if "$SCRIPT_DIR/health-check.sh" -v; then
        print_info "Rollback validation passed"
        return 0
    else
        print_error "Rollback validation failed"
        return 1
    fi
}

# Generate rollback report
generate_report() {
    local start_time="$1"
    local end_time="$2"
    local success="$3"
    
    local duration=$((end_time - start_time))
    local current_version=$(get_current_version)
    
    cat << EOF

=== ROLLBACK REPORT ===
Date: $(date)
Type: $ROLLBACK_TYPE
Duration: ${duration}s
Target Version: ${TARGET_VERSION:-N/A}
Target Backup: ${TARGET_BACKUP:-N/A}
Current Version After Rollback: $current_version
Status: $([ "$success" = "true" ] && echo "SUCCESS" || echo "FAILED")

EOF
}

# Main rollback process
main() {
    local start_time=$(date +%s)
    local success="false"
    
    print_info "=== Starting Arbitration Detection System Rollback ==="
    print_info "Type: $ROLLBACK_TYPE"
    print_info "Dry Run: $DRY_RUN"

    # Check prerequisites
    check_prerequisites

    # Show current state
    local current_version=$(get_current_version)
    print_info "Current version: $current_version"

    # Select targets based on rollback type
    if [[ "$ROLLBACK_TYPE" == "full" || "$ROLLBACK_TYPE" == "application" ]]; then
        select_target_version
    fi

    if [[ "$ROLLBACK_TYPE" == "full" || "$ROLLBACK_TYPE" == "database" ]]; then
        select_target_backup
    fi

    # Confirm rollback
    confirm_rollback

    # Create emergency backup
    if ! create_emergency_backup; then
        if [[ "$FORCE_ROLLBACK" != "true" ]]; then
            print_error "Emergency backup failed. Aborting rollback."
            exit 1
        fi
    fi

    # Perform rollback
    local rollback_failed=false

    # Rollback application
    if ! rollback_application; then
        print_error "Application rollback failed"
        rollback_failed=true
    fi

    # Rollback database
    if ! rollback_database; then
        print_error "Database rollback failed"
        rollback_failed=true
    fi

    if [[ "$rollback_failed" == "true" ]]; then
        print_error "Rollback failed"
        exit 1
    fi

    # Validate rollback
    if ! validate_rollback; then
        print_error "Rollback validation failed"
        if [[ "$FORCE_ROLLBACK" != "true" ]]; then
            exit 1
        fi
    fi

    local end_time=$(date +%s)
    success="true"

    # Generate report
    generate_report "$start_time" "$end_time" "$success"

    print_info "=== Rollback Completed Successfully ==="
    print_info "Previous version: $current_version"
    print_info "Current version: $(get_current_version)"
    print_info "Duration: $((end_time - start_time))s"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        print_info "Please verify the system is working correctly"
        print_info "Run health checks: $SCRIPT_DIR/health-check.sh"
    fi
}

# Handle script interruption
cleanup() {
    print_warning "Rollback interrupted. System may be in inconsistent state."
    print_warning "Please run health checks and manual verification."
    exit 130
}

trap cleanup INT TERM

# Parse arguments and run main function
parse_args "$@"
main