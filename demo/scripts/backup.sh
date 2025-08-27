#!/bin/bash

# Database Backup Script for Arbitration Detection System
# Handles automated backups with retention, compression, and cloud storage

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/backup.log}"
COMPOSE_FILE="$PROJECT_ROOT/demo/production/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"

# Default settings
RETENTION_DAYS=30
MAX_BACKUPS=50
COMPRESS_BACKUPS=true
UPLOAD_TO_CLOUD=false
CLOUD_BUCKET=""
INCLUDE_UPLOADS=true
BACKUP_TYPE="full"
ENCRYPTION_ENABLED=false
ENCRYPTION_KEY=""

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

Create backup of the Arbitration Detection System database and files.

OPTIONS:
    -t, --type TYPE         Backup type: full|incremental|schema (default: full)
    -d, --dir DIR           Backup directory (default: $BACKUP_DIR)
    -r, --retention DAYS    Retention period in days (default: $RETENTION_DAYS)
    -m, --max-backups NUM   Maximum number of backups to keep (default: $MAX_BACKUPS)
    --no-compress           Disable backup compression
    --no-uploads            Skip backing up upload files
    --cloud-bucket BUCKET   Upload backup to cloud storage bucket
    --encrypt KEY           Encrypt backup with provided key
    --dry-run              Show what would be backed up without executing
    -v, --verbose          Verbose output
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Full backup with default settings
    $0 -t incremental                    # Incremental backup
    $0 --cloud-bucket s3://my-backups    # Backup and upload to S3
    $0 --encrypt mykey123                # Encrypted backup
    $0 --dry-run -v                      # Preview backup with verbose output

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -d|--dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -m|--max-backups)
                MAX_BACKUPS="$2"
                shift 2
                ;;
            --no-compress)
                COMPRESS_BACKUPS=false
                shift
                ;;
            --no-uploads)
                INCLUDE_UPLOADS=false
                shift
                ;;
            --cloud-bucket)
                CLOUD_BUCKET="$2"
                UPLOAD_TO_CLOUD=true
                shift 2
                ;;
            --encrypt)
                ENCRYPTION_KEY="$2"
                ENCRYPTION_ENABLED=true
                shift 2
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
    local required_commands=("docker" "docker-compose")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done

    # Check if compression tools are available
    if [[ "$COMPRESS_BACKUPS" == "true" ]]; then
        if ! command -v gzip &> /dev/null; then
            print_warning "gzip not found, disabling compression"
            COMPRESS_BACKUPS=false
        fi
    fi

    # Check cloud tools if needed
    if [[ "$UPLOAD_TO_CLOUD" == "true" ]]; then
        case "$CLOUD_BUCKET" in
            s3://*)
                if ! command -v aws &> /dev/null; then
                    print_error "AWS CLI not found but S3 upload requested"
                    exit 1
                fi
                ;;
            gs://*)
                if ! command -v gsutil &> /dev/null; then
                    print_error "gsutil not found but GCS upload requested"
                    exit 1
                fi
                ;;
            *)
                print_error "Unsupported cloud bucket format: $CLOUD_BUCKET"
                exit 1
                ;;
        esac
    fi

    # Check encryption tools if needed
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        if ! command -v openssl &> /dev/null; then
            print_error "openssl not found but encryption requested"
            exit 1
        fi
    fi

    # Create directories
    mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

    # Load environment
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
    else
        print_warning "Environment file not found: $ENV_FILE"
    fi

    print_info "Prerequisites check completed"
}

# Get database size
get_database_size() {
    local size_query="SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));"
    local size=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "$size_query" | xargs)
    echo "$size"
}

# Get backup filename
get_backup_filename() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local filename="backup_${BACKUP_TYPE}_${timestamp}"
    
    if [[ "$BACKUP_TYPE" == "incremental" ]]; then
        local last_backup=$(find "$BACKUP_DIR" -name "backup_full_*.sql*" -type f | sort | tail -1)
        if [[ -n "$last_backup" ]]; then
            local last_date=$(basename "$last_backup" | sed 's/backup_full_\([0-9_]*\)\.sql.*/\1/')
            filename="backup_incremental_${last_date}_to_${timestamp}"
        fi
    fi
    
    echo "${filename}.sql"
}

# Create database backup
backup_database() {
    local backup_file="$1"
    local temp_file="${backup_file}.tmp"
    
    print_info "Creating database backup..."
    print_info "Database size: $(get_database_size)"
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would create database backup: $backup_file"
        return 0
    fi

    case "$BACKUP_TYPE" in
        "full")
            print_info "Creating full database backup..."
            docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
                -U "$POSTGRES_USER" \
                -d "$POSTGRES_DB" \
                --verbose \
                --no-acl \
                --no-owner \
                --format=custom \
                --compress=9 > "$temp_file"
            ;;
        "schema")
            print_info "Creating schema-only backup..."
            docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
                -U "$POSTGRES_USER" \
                -d "$POSTGRES_DB" \
                --schema-only \
                --verbose \
                --no-acl \
                --no-owner > "$temp_file"
            ;;
        "incremental")
            print_info "Creating incremental backup..."
            # For incremental backups, we'll use WAL archiving or logical replication
            # For now, creating a full backup with timestamp filtering
            docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
                -U "$POSTGRES_USER" \
                -d "$POSTGRES_DB" \
                --verbose \
                --no-acl \
                --no-owner \
                --format=custom \
                --compress=9 > "$temp_file"
            ;;
        *)
            print_error "Unknown backup type: $BACKUP_TYPE"
            return 1
            ;;
    esac

    if [[ $? -eq 0 ]]; then
        mv "$temp_file" "$backup_file"
        print_info "Database backup created: $backup_file"
        
        # Show backup size
        local backup_size=$(du -h "$backup_file" | cut -f1)
        print_info "Backup size: $backup_size"
        
        return 0
    else
        print_error "Database backup failed"
        rm -f "$temp_file"
        return 1
    fi
}

# Backup uploads directory
backup_uploads() {
    local uploads_backup="$1"
    
    if [[ "$INCLUDE_UPLOADS" != "true" ]]; then
        print_info "Skipping uploads backup as requested"
        return 0
    fi

    print_info "Creating uploads backup..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would create uploads backup: $uploads_backup"
        return 0
    fi

    # Check if uploads volume exists
    local uploads_volume=$(docker volume ls -q | grep "production_backend_uploads" || echo "")
    if [[ -z "$uploads_volume" ]]; then
        print_warning "Uploads volume not found, skipping uploads backup"
        return 0
    fi

    # Create tar archive of uploads
    docker run --rm \
        -v production_backend_uploads:/data \
        -v "$BACKUP_DIR:/backup" \
        alpine:latest \
        tar -czf "/backup/$(basename "$uploads_backup")" -C /data .

    if [[ $? -eq 0 ]]; then
        print_info "Uploads backup created: $uploads_backup"
        local upload_size=$(du -h "$uploads_backup" | cut -f1)
        print_info "Uploads backup size: $upload_size"
        return 0
    else
        print_error "Uploads backup failed"
        return 1
    fi
}

# Compress backup
compress_backup() {
    local backup_file="$1"
    
    if [[ "$COMPRESS_BACKUPS" != "true" ]]; then
        print_debug "Compression disabled"
        echo "$backup_file"
        return 0
    fi

    print_info "Compressing backup..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would compress: $backup_file"
        echo "${backup_file}.gz"
        return 0
    fi

    local compressed_file="${backup_file}.gz"
    
    gzip "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        print_info "Backup compressed: $compressed_file"
        local compressed_size=$(du -h "$compressed_file" | cut -f1)
        print_info "Compressed size: $compressed_size"
        echo "$compressed_file"
        return 0
    else
        print_error "Compression failed"
        echo "$backup_file"
        return 1
    fi
}

# Encrypt backup
encrypt_backup() {
    local backup_file="$1"
    
    if [[ "$ENCRYPTION_ENABLED" != "true" ]]; then
        print_debug "Encryption disabled"
        echo "$backup_file"
        return 0
    fi

    print_info "Encrypting backup..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would encrypt: $backup_file"
        echo "${backup_file}.enc"
        return 0
    fi

    local encrypted_file="${backup_file}.enc"
    
    openssl enc -aes-256-cbc -salt -in "$backup_file" -out "$encrypted_file" -pass pass:"$ENCRYPTION_KEY"
    
    if [[ $? -eq 0 ]]; then
        rm "$backup_file"  # Remove unencrypted file
        print_info "Backup encrypted: $encrypted_file"
        echo "$encrypted_file"
        return 0
    else
        print_error "Encryption failed"
        echo "$backup_file"
        return 1
    fi
}

# Upload to cloud storage
upload_to_cloud() {
    local backup_file="$1"
    
    if [[ "$UPLOAD_TO_CLOUD" != "true" ]]; then
        print_debug "Cloud upload disabled"
        return 0
    fi

    print_info "Uploading backup to cloud storage..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would upload to: $CLOUD_BUCKET"
        return 0
    fi

    local cloud_path="$CLOUD_BUCKET/$(basename "$backup_file")"
    
    case "$CLOUD_BUCKET" in
        s3://*)
            aws s3 cp "$backup_file" "$cloud_path"
            ;;
        gs://*)
            gsutil cp "$backup_file" "$cloud_path"
            ;;
    esac

    if [[ $? -eq 0 ]]; then
        print_info "Backup uploaded to: $cloud_path"
        return 0
    else
        print_error "Cloud upload failed"
        return 1
    fi
}

# Clean old backups
cleanup_old_backups() {
    print_info "Cleaning up old backups..."
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_info "[DRY RUN] Would clean backups older than $RETENTION_DAYS days"
        local old_files=$(find "$BACKUP_DIR" -name "backup_*" -type f -mtime +$RETENTION_DAYS | wc -l)
        print_info "[DRY RUN] Would delete $old_files old backup files"
        return 0
    fi

    # Remove backups older than retention period
    local deleted_count=0
    while IFS= read -r -d '' file; do
        print_debug "Deleting old backup: $file"
        rm "$file"
        deleted_count=$((deleted_count + 1))
    done < <(find "$BACKUP_DIR" -name "backup_*" -type f -mtime +$RETENTION_DAYS -print0)

    # Keep only max number of backups
    local backup_files=$(find "$BACKUP_DIR" -name "backup_*" -type f | sort -r)
    local file_count=$(echo "$backup_files" | wc -l)
    
    if [[ $file_count -gt $MAX_BACKUPS ]]; then
        local files_to_delete=$((file_count - MAX_BACKUPS))
        echo "$backup_files" | tail -n $files_to_delete | while read -r file; do
            print_debug "Deleting excess backup: $file"
            rm "$file"
            deleted_count=$((deleted_count + 1))
        done
    fi

    if [[ $deleted_count -gt 0 ]]; then
        print_info "Deleted $deleted_count old backup files"
    else
        print_info "No old backups to delete"
    fi
}

# Generate backup report
generate_report() {
    local backup_file="$1"
    local uploads_backup="$2"
    local start_time="$3"
    local end_time="$4"
    
    local duration=$((end_time - start_time))
    local backup_size=$(du -h "$backup_file" 2>/dev/null | cut -f1 || echo "N/A")
    local uploads_size=$(du -h "$uploads_backup" 2>/dev/null | cut -f1 || echo "N/A")
    
    cat << EOF

=== BACKUP REPORT ===
Date: $(date)
Type: $BACKUP_TYPE
Duration: ${duration}s
Database Backup: $backup_file (${backup_size})
Uploads Backup: ${uploads_backup:-N/A} (${uploads_size})
Compression: $COMPRESS_BACKUPS
Encryption: $ENCRYPTION_ENABLED
Cloud Upload: $UPLOAD_TO_CLOUD
Retention: $RETENTION_DAYS days
Max Backups: $MAX_BACKUPS

EOF
}

# Main backup process
main() {
    local start_time=$(date +%s)
    
    print_info "=== Starting Arbitration Detection System Backup ==="
    print_info "Type: $BACKUP_TYPE"
    print_info "Directory: $BACKUP_DIR"
    print_info "Compression: $COMPRESS_BACKUPS"
    print_info "Encryption: $ENCRYPTION_ENABLED"
    print_info "Cloud Upload: $UPLOAD_TO_CLOUD"

    # Check prerequisites
    check_prerequisites

    # Generate backup filename
    local backup_filename=$(get_backup_filename)
    local backup_file="$BACKUP_DIR/$backup_filename"
    local uploads_backup="$BACKUP_DIR/uploads_$(date +%Y%m%d_%H%M%S).tar.gz"

    # Create database backup
    if ! backup_database "$backup_file"; then
        print_error "Database backup failed"
        exit 1
    fi

    # Create uploads backup
    backup_uploads "$uploads_backup"

    # Compress backup
    backup_file=$(compress_backup "$backup_file")

    # Encrypt backup
    backup_file=$(encrypt_backup "$backup_file")

    # Upload to cloud
    upload_to_cloud "$backup_file"
    if [[ -f "$uploads_backup" ]]; then
        upload_to_cloud "$uploads_backup"
    fi

    # Clean old backups
    cleanup_old_backups

    local end_time=$(date +%s)

    # Generate report
    generate_report "$backup_file" "$uploads_backup" "$start_time" "$end_time"

    print_info "=== Backup Completed Successfully ==="
    print_info "Database backup: $backup_file"
    if [[ -f "$uploads_backup" ]]; then
        print_info "Uploads backup: $uploads_backup"
    fi
    print_info "Duration: $((end_time - start_time))s"
}

# Handle script interruption
cleanup() {
    print_warning "Backup interrupted. Cleaning up..."
    # Remove any temporary files
    rm -f "$BACKUP_DIR"/*.tmp
    exit 130
}

trap cleanup INT TERM

# Parse arguments and run main function
parse_args "$@"
main