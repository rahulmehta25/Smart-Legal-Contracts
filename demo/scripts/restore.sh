#!/bin/bash

# Database Restore Script for Arbitration Detection System
# Handles disaster recovery with various restore options and validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/restore.log}"
COMPOSE_FILE="$PROJECT_ROOT/demo/production/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"

# Default settings
BACKUP_FILE=""
RESTORE_UPLOADS=true
RESTORE_DATABASE=true
POINT_IN_TIME=""
DRY_RUN=false
FORCE_RESTORE=false
VALIDATE_RESTORE=true
CREATE_BACKUP_BEFORE_RESTORE=true

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
Usage: $0 [OPTIONS] BACKUP_FILE

Restore the Arbitration Detection System from backup.

ARGUMENTS:
    BACKUP_FILE             Path to backup file to restore

OPTIONS:
    --database-only         Restore only database (skip uploads)
    --uploads-only          Restore only uploads (skip database)
    --point-in-time TIME    Restore to specific point in time (YYYY-MM-DD HH:MM:SS)
    --no-validation         Skip restore validation
    --no-pre-backup         Skip creating backup before restore
    --force                 Force restore even with warnings
    --dry-run               Show what would be restored without executing
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0 backup_full_20240101_120000.sql.gz      # Full restore from backup
    $0 --database-only backup.sql              # Restore only database
    $0 --point-in-time "2024-01-01 12:00:00"   # Point-in-time recovery
    $0 --dry-run backup.sql                     # Preview restore actions

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --database-only)
                RESTORE_UPLOADS=false
                shift
                ;;
            --uploads-only)
                RESTORE_DATABASE=false
                shift
                ;;
            --point-in-time)
                POINT_IN_TIME="$2"
                shift 2
                ;;
            --no-validation)
                VALIDATE_RESTORE=false
                shift
                ;;
            --no-pre-backup)
                CREATE_BACKUP_BEFORE_RESTORE=false
                shift
                ;;
            --force)
                FORCE_RESTORE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                BACKUP_FILE="$1"
                shift
                ;;
        esac
    done

    if [[ -z "$BACKUP_FILE" ]]; then
        print_error "Backup file is required"
        usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check required commands
    local required_commands=("docker" "docker-compose")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done

    # Check backup file
    if [[ ! -f "$BACKUP_FILE" ]]; then
        print_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi

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

# List available backups
list_backups() {
    print_info "Available backups in $BACKUP_DIR:"
    find "$BACKUP_DIR" -name "backup_*" -type f -exec ls -lh {} \; | sort -k6,7
}

# Detect backup type and format
detect_backup_format() {
    local backup_file="$1"
    
    print_info "Detecting backup format..."
    
    # Check file extension
    case "$backup_file" in
        *.gz)
            print_info "Detected compressed backup (gzip)"
            echo "compressed"
            ;;
        *.enc)
            print_info "Detected encrypted backup"
            echo "encrypted"
            ;;
        *.sql)
            print_info "Detected SQL backup"
            echo "sql"
            ;;
        *)
            print_info "Detecting format from file content..."
            if file "$backup_file" | grep -q "gzip"; then
                echo "compressed"
            elif file "$backup_file" | grep -q "PostgreSQL"; then
                echo "custom"
            else
                echo "unknown"
            fi
            ;;
    esac
}

# Prepare backup file for restore
prepare_backup_file() {
    local backup_file="$1"
    local format="$2"
    local temp_file=""
    
    case "$format" in
        "compressed")
            print_info "Decompressing backup file..."
            temp_file="${backup_file%.gz}.tmp"
            
            if [[ "${DRY_RUN:-false}" == "true" ]]; then
                print_info "[DRY RUN] Would decompress: $backup_file"
                echo "$backup_file"
                return 0
            fi
            
            gunzip -c "$backup_file" > "$temp_file"
            echo "$temp_file"
            ;;
        "encrypted")
            print_info "Decrypting backup file..."
            if [[ -z "${ENCRYPTION_KEY:-}" ]]; then
                read -s -p "Enter encryption key: " ENCRYPTION_KEY
                echo
            fi
            
            temp_file="${backup_file%.enc}.tmp"
            
            if [[ "${DRY_RUN:-false}" == "true" ]]; then
                print_info "[DRY RUN] Would decrypt: $backup_file"
                echo "$backup_file"
                return 0
            fi
            
            openssl enc -aes-256-cbc -d -in "$backup_file" -out "$temp_file" -pass pass:"$ENCRYPTION_KEY"
            echo "$temp_file"
            ;;
        *)
            echo "$backup_file"
            ;;
    esac
}

# Create pre-restore backup
create_pre_restore_backup() {
    if [[ "$CREATE_BACKUP_BEFORE_RESTORE" != "true" ]]; then
        print_info "Skipping pre-restore backup as requested"
        return 0
    fi

    print_info "Creating pre-restore backup..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would create pre-restore backup"
        return 0
    fi

    local pre_restore_backup="$BACKUP_DIR/pre-restore-$(date +%Y%m%d_%H%M%S).sql"
    
    # Call backup script
    "$SCRIPT_DIR/backup.sh" -t full -d "$BACKUP_DIR" --no-uploads
    
    if [[ $? -eq 0 ]]; then
        print_info "Pre-restore backup created successfully"
        return 0
    else
        print_error "Pre-restore backup failed"
        if [[ "$FORCE_RESTORE" != "true" ]]; then
            return 1
        else
            print_warning "Continuing with restore despite backup failure (--force enabled)"
            return 0
        fi
    fi
}

# Stop application services
stop_services() {
    print_info "Stopping application services..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would stop application services"
        return 0
    fi

    # Stop application services but keep database running
    docker-compose -f "$COMPOSE_FILE" stop backend frontend nginx
    
    print_info "Application services stopped"
}

# Start application services
start_services() {
    print_info "Starting application services..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would start application services"
        return 0
    fi

    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    sleep 30
    
    print_info "Application services started"
}

# Restore database
restore_database() {
    local backup_file="$1"
    
    if [[ "$RESTORE_DATABASE" != "true" ]]; then
        print_info "Skipping database restore as requested"
        return 0
    fi

    print_info "Restoring database from: $backup_file"
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would restore database from: $backup_file"
        return 0
    fi

    # Check if database is accessible
    if ! docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; then
        print_error "Database is not accessible"
        return 1
    fi

    # Drop existing connections
    print_info "Terminating existing database connections..."
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d postgres -c "
        SELECT pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE pg_stat_activity.datname = '$POSTGRES_DB'
          AND pid <> pg_backend_pid();"

    # Drop and recreate database
    print_info "Recreating database..."
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d postgres -c "DROP DATABASE IF EXISTS $POSTGRES_DB;"
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE $POSTGRES_DB;"

    # Restore from backup
    print_info "Restoring data..."
    local file_extension="${backup_file##*.}"
    
    case "$file_extension" in
        "sql")
            docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$backup_file"
            ;;
        *)
            # Assume custom format
            docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --verbose --clean --no-acl --no-owner < "$backup_file"
            ;;
    esac

    if [[ $? -eq 0 ]]; then
        print_info "Database restore completed successfully"
        return 0
    else
        print_error "Database restore failed"
        return 1
    fi
}

# Restore uploads
restore_uploads() {
    if [[ "$RESTORE_UPLOADS" != "true" ]]; then
        print_info "Skipping uploads restore as requested"
        return 0
    fi

    # Find corresponding uploads backup
    local backup_basename=$(basename "$BACKUP_FILE")
    local backup_date=$(echo "$backup_basename" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1)
    
    if [[ -z "$backup_date" ]]; then
        print_warning "Cannot determine backup date for uploads restore"
        return 0
    fi

    local uploads_backup="$BACKUP_DIR/uploads_${backup_date}.tar.gz"
    
    if [[ ! -f "$uploads_backup" ]]; then
        print_warning "Uploads backup not found: $uploads_backup"
        return 0
    fi

    print_info "Restoring uploads from: $uploads_backup"
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would restore uploads from: $uploads_backup"
        return 0
    fi

    # Remove existing uploads and restore
    docker run --rm \
        -v production_backend_uploads:/data \
        -v "$BACKUP_DIR:/backup" \
        alpine:latest \
        sh -c "rm -rf /data/* && tar -xzf /backup/$(basename "$uploads_backup") -C /data"

    if [[ $? -eq 0 ]]; then
        print_info "Uploads restore completed successfully"
        return 0
    else
        print_error "Uploads restore failed"
        return 1
    fi
}

# Validate restore
validate_restore() {
    if [[ "$VALIDATE_RESTORE" != "true" ]]; then
        print_info "Skipping restore validation as requested"
        return 0
    fi

    print_info "Validating restore..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would validate restore"
        return 0
    fi

    # Check database connectivity
    if ! docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; then
        print_error "Database validation failed: not accessible"
        return 1
    fi

    # Check table count
    local table_count=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    
    if [[ $table_count -lt 5 ]]; then
        print_error "Database validation failed: insufficient tables ($table_count found)"
        return 1
    fi

    print_info "Found $table_count tables in database"

    # Check application connectivity
    sleep 30  # Wait for application to start

    # Test health endpoint
    local max_attempts=10
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health &> /dev/null; then
            print_info "Application health check passed"
            break
        fi
        
        attempt=$((attempt + 1))
        print_debug "Health check attempt $attempt/$max_attempts"
        sleep 10
    done

    if [[ $attempt -eq $max_attempts ]]; then
        print_warning "Application health check failed, but database restore appears successful"
        return 0
    fi

    print_info "Restore validation completed successfully"
    return 0
}

# Point-in-time recovery
point_in_time_recovery() {
    local target_time="$1"
    
    print_info "Performing point-in-time recovery to: $target_time"
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would perform point-in-time recovery to: $target_time"
        return 0
    fi

    # This is a simplified implementation
    # In a production environment, you would use WAL archives and pg_basebackup
    print_warning "Point-in-time recovery requires WAL archiving setup"
    print_info "For full PITR support, implement WAL archiving and use pg_basebackup"
    
    # Find the closest backup before the target time
    local target_timestamp=$(date -d "$target_time" +%s)
    local best_backup=""
    local best_timestamp=0
    
    for backup in "$BACKUP_DIR"/backup_full_*.sql*; do
        if [[ -f "$backup" ]]; then
            local backup_date=$(basename "$backup" | grep -oE '[0-9]{8}_[0-9]{6}')
            local backup_datetime="${backup_date:0:8} ${backup_date:9:2}:${backup_date:11:2}:${backup_date:13:2}"
            local backup_timestamp=$(date -d "$backup_datetime" +%s)
            
            if [[ $backup_timestamp -le $target_timestamp && $backup_timestamp -gt $best_timestamp ]]; then
                best_backup="$backup"
                best_timestamp=$backup_timestamp
            fi
        fi
    done

    if [[ -z "$best_backup" ]]; then
        print_error "No suitable backup found for point-in-time recovery"
        return 1
    fi

    print_info "Using backup: $best_backup"
    BACKUP_FILE="$best_backup"
    
    # Proceed with regular restore
    return 0
}

# Generate restore report
generate_report() {
    local start_time="$1"
    local end_time="$2"
    local success="$3"
    
    local duration=$((end_time - start_time))
    local backup_size=$(du -h "$BACKUP_FILE" 2>/dev/null | cut -f1 || echo "N/A")
    
    cat << EOF

=== RESTORE REPORT ===
Date: $(date)
Backup File: $BACKUP_FILE
Backup Size: $backup_size
Duration: ${duration}s
Database Restored: $RESTORE_DATABASE
Uploads Restored: $RESTORE_UPLOADS
Validation: $VALIDATE_RESTORE
Status: $([ "$success" = "true" ] && echo "SUCCESS" || echo "FAILED")

EOF
}

# Main restore process
main() {
    local start_time=$(date +%s)
    local success="false"
    
    print_info "=== Starting Arbitration Detection System Restore ==="
    print_info "Backup File: $BACKUP_FILE"
    print_info "Restore Database: $RESTORE_DATABASE"
    print_info "Restore Uploads: $RESTORE_UPLOADS"
    print_info "Validation: $VALIDATE_RESTORE"

    # Check prerequisites
    check_prerequisites

    # Handle point-in-time recovery
    if [[ -n "$POINT_IN_TIME" ]]; then
        if ! point_in_time_recovery "$POINT_IN_TIME"; then
            print_error "Point-in-time recovery failed"
            exit 1
        fi
    fi

    # Detect backup format
    local backup_format=$(detect_backup_format "$BACKUP_FILE")
    
    # Prepare backup file
    local prepared_backup=$(prepare_backup_file "$BACKUP_FILE" "$backup_format")
    
    # Create pre-restore backup
    if ! create_pre_restore_backup; then
        if [[ "$FORCE_RESTORE" != "true" ]]; then
            print_error "Pre-restore backup failed. Use --force to continue anyway."
            exit 1
        fi
    fi

    # Stop services
    stop_services

    # Restore database
    if ! restore_database "$prepared_backup"; then
        print_error "Database restore failed"
        start_services
        exit 1
    fi

    # Restore uploads
    restore_uploads

    # Start services
    start_services

    # Validate restore
    if ! validate_restore; then
        print_error "Restore validation failed"
        if [[ "$FORCE_RESTORE" != "true" ]]; then
            exit 1
        fi
    fi

    # Clean up temporary files
    if [[ "$prepared_backup" != "$BACKUP_FILE" ]]; then
        rm -f "$prepared_backup"
    fi

    local end_time=$(date +%s)
    success="true"

    # Generate report
    generate_report "$start_time" "$end_time" "$success"

    print_info "=== Restore Completed Successfully ==="
    print_info "Duration: $((end_time - start_time))s"
}

# Handle script interruption
cleanup() {
    print_warning "Restore interrupted. Starting services..."
    start_services
    exit 130
}

trap cleanup INT TERM

# Parse arguments and run main function
parse_args "$@"

# Show available backups if no backup file specified
if [[ -z "$BACKUP_FILE" ]]; then
    list_backups
    exit 1
fi

main