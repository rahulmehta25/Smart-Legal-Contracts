#!/bin/bash

# Comprehensive Health Check Script for Arbitration Detection System
# Monitors all components and provides detailed status reporting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMPOSE_FILE="$PROJECT_ROOT/demo/production/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"
LOG_FILE="$PROJECT_ROOT/logs/health-check.log"

# Health check endpoints
BACKEND_HEALTH_URL="http://localhost:8000/health"
FRONTEND_HEALTH_URL="http://localhost:3000/api/health"
GRAFANA_URL="http://localhost:3001/api/health"
PROMETHEUS_URL="http://localhost:9090/-/ready"

# Thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=80
DISK_THRESHOLD=90
RESPONSE_TIME_THRESHOLD=5000  # milliseconds
DATABASE_CONNECTIONS_THRESHOLD=80  # percentage of max connections

# Output format
OUTPUT_FORMAT="text"  # text, json, or prometheus
NAGIOS_MODE=false
VERBOSE=false
ALERT_ON_FAILURE=false
SEND_SLACK_ALERTS=false
SLACK_WEBHOOK_URL=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Status tracking
OVERALL_STATUS="OK"
FAILED_CHECKS=()
WARNING_CHECKS=()

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

print_info() {
    [[ "$VERBOSE" == "true" ]] && echo -e "${GREEN}[INFO]${NC} $1"
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

print_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
    log "OK" "$1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Perform comprehensive health checks on the Arbitration Detection System.

OPTIONS:
    -f, --format FORMAT     Output format: text, json, prometheus (default: text)
    -n, --nagios           Nagios-compatible exit codes
    -v, --verbose          Verbose output
    -a, --alert            Send alerts on failure
    --slack-webhook URL    Slack webhook URL for alerts
    --cpu-threshold N      CPU usage threshold (default: $CPU_THRESHOLD%)
    --memory-threshold N   Memory usage threshold (default: $MEMORY_THRESHOLD%)
    --disk-threshold N     Disk usage threshold (default: $DISK_THRESHOLD%)
    -h, --help             Show this help message

EXAMPLES:
    $0                           # Basic health check
    $0 -v                        # Verbose health check
    $0 -f json                   # JSON output format
    $0 -n                        # Nagios mode with exit codes
    $0 --alert --slack-webhook URL # Send Slack alerts

EXIT CODES (Nagios mode):
    0 - OK
    1 - WARNING
    2 - CRITICAL
    3 - UNKNOWN

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -n|--nagios)
                NAGIOS_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -a|--alert)
                ALERT_ON_FAILURE=true
                shift
                ;;
            --slack-webhook)
                SLACK_WEBHOOK_URL="$2"
                SEND_SLACK_ALERTS=true
                shift 2
                ;;
            --cpu-threshold)
                CPU_THRESHOLD="$2"
                shift 2
                ;;
            --memory-threshold)
                MEMORY_THRESHOLD="$2"
                shift 2
                ;;
            --disk-threshold)
                DISK_THRESHOLD="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Load environment
load_environment() {
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
    else
        print_warning "Environment file not found: $ENV_FILE"
    fi
}

# Check if Docker is running
check_docker() {
    print_info "Checking Docker daemon..."
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        OVERALL_STATUS="CRITICAL"
        FAILED_CHECKS+=("docker")
        return 1
    fi
    
    print_ok "Docker daemon is running"
    return 0
}

# Check container status
check_containers() {
    print_info "Checking container status..."
    
    local containers=("arbitration-postgres-prod" "arbitration-redis-prod" "arbitration-backend-prod" "arbitration-frontend-prod" "arbitration-nginx-prod")
    local failed_containers=()
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container.*Up"; then
            print_ok "Container $container is running"
        else
            print_error "Container $container is not running"
            failed_containers+=("$container")
        fi
    done
    
    if [[ ${#failed_containers[@]} -gt 0 ]]; then
        OVERALL_STATUS="CRITICAL"
        FAILED_CHECKS+=("containers")
        return 1
    fi
    
    return 0
}

# Check service health endpoints
check_service_health() {
    local service_name="$1"
    local health_url="$2"
    local timeout="${3:-10}"
    
    print_info "Checking $service_name health endpoint..."
    
    local start_time=$(date +%s%3N)
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$health_url" 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    if [[ "$http_code" == "200" ]]; then
        print_ok "$service_name health check passed (${response_time}ms)"
        
        if [[ $response_time -gt $RESPONSE_TIME_THRESHOLD ]]; then
            print_warning "$service_name response time is slow (${response_time}ms > ${RESPONSE_TIME_THRESHOLD}ms)"
            WARNING_CHECKS+=("$service_name-response-time")
        fi
        
        return 0
    else
        print_error "$service_name health check failed (HTTP $http_code)"
        OVERALL_STATUS="CRITICAL"
        FAILED_CHECKS+=("$service_name")
        return 1
    fi
}

# Check database connectivity and performance
check_database() {
    print_info "Checking database..."
    
    # Basic connectivity
    if ! docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" &> /dev/null; then
        print_error "Database is not accessible"
        OVERALL_STATUS="CRITICAL"
        FAILED_CHECKS+=("database-connectivity")
        return 1
    fi
    
    print_ok "Database connectivity verified"
    
    # Connection count
    local max_connections=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SHOW max_connections;" 2>/dev/null | xargs || echo "100")
    local current_connections=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "0")
    
    local connection_percentage=$((current_connections * 100 / max_connections))
    
    if [[ $connection_percentage -gt $DATABASE_CONNECTIONS_THRESHOLD ]]; then
        print_warning "High database connection usage: ${current_connections}/${max_connections} (${connection_percentage}%)"
        WARNING_CHECKS+=("database-connections")
    else
        print_ok "Database connections: ${current_connections}/${max_connections} (${connection_percentage}%)"
    fi
    
    # Database size
    local db_size=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" 2>/dev/null | xargs || echo "Unknown")
    print_info "Database size: $db_size"
    
    return 0
}

# Check Redis connectivity and memory usage
check_redis() {
    print_info "Checking Redis..."
    
    # Basic connectivity
    if ! docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &> /dev/null; then
        print_error "Redis is not accessible"
        OVERALL_STATUS="CRITICAL"
        FAILED_CHECKS+=("redis-connectivity")
        return 1
    fi
    
    print_ok "Redis connectivity verified"
    
    # Memory usage
    local memory_info=$(docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli info memory 2>/dev/null || echo "")
    if [[ -n "$memory_info" ]]; then
        local used_memory=$(echo "$memory_info" | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
        local max_memory=$(echo "$memory_info" | grep "maxmemory_human:" | cut -d: -f2 | tr -d '\r')
        print_info "Redis memory usage: ${used_memory:-Unknown}/${max_memory:-Unknown}"
    fi
    
    return 0
}

# Check system resources
check_system_resources() {
    print_info "Checking system resources..."
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' 2>/dev/null || echo "0")
    cpu_usage=${cpu_usage%.*}  # Remove decimal places
    
    if [[ $cpu_usage -gt $CPU_THRESHOLD ]]; then
        print_warning "High CPU usage: ${cpu_usage}%"
        WARNING_CHECKS+=("cpu-usage")
    else
        print_ok "CPU usage: ${cpu_usage}%"
    fi
    
    # Memory usage
    local memory_info=$(free | grep Mem)
    local total_mem=$(echo "$memory_info" | awk '{print $2}')
    local used_mem=$(echo "$memory_info" | awk '{print $3}')
    local memory_percentage=$((used_mem * 100 / total_mem))
    
    if [[ $memory_percentage -gt $MEMORY_THRESHOLD ]]; then
        print_warning "High memory usage: ${memory_percentage}%"
        WARNING_CHECKS+=("memory-usage")
    else
        print_ok "Memory usage: ${memory_percentage}%"
    fi
    
    # Disk usage
    local disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    
    if [[ $disk_usage -gt $DISK_THRESHOLD ]]; then
        print_error "Critical disk usage: ${disk_usage}%"
        OVERALL_STATUS="CRITICAL"
        FAILED_CHECKS+=("disk-usage")
    else
        print_ok "Disk usage: ${disk_usage}%"
    fi
    
    # Load average
    local load_avg=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)
    print_info "Load average: $load_avg"
    
    return 0
}

# Check SSL certificates
check_ssl_certificates() {
    print_info "Checking SSL certificates..."
    
    local ssl_dir="$PROJECT_ROOT/demo/nginx/ssl"
    local certs=("app.crt" "api.crt" "default.crt")
    
    for cert in "${certs[@]}"; do
        local cert_file="$ssl_dir/$cert"
        
        if [[ -f "$cert_file" ]]; then
            local expiry=$(openssl x509 -enddate -noout -in "$cert_file" 2>/dev/null | cut -d= -f2)
            local expiry_timestamp=$(date -d "$expiry" +%s 2>/dev/null || echo "0")
            local current_timestamp=$(date +%s)
            local days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))
            
            if [[ $days_until_expiry -lt 7 ]]; then
                print_warning "SSL certificate $cert expires in $days_until_expiry days"
                WARNING_CHECKS+=("ssl-cert-$cert")
            elif [[ $days_until_expiry -lt 0 ]]; then
                print_error "SSL certificate $cert has expired"
                OVERALL_STATUS="CRITICAL"
                FAILED_CHECKS+=("ssl-cert-$cert")
            else
                print_ok "SSL certificate $cert is valid (expires in $days_until_expiry days)"
            fi
        else
            print_warning "SSL certificate $cert not found"
            WARNING_CHECKS+=("ssl-cert-missing-$cert")
        fi
    done
    
    return 0
}

# Check log files for errors
check_logs() {
    print_info "Checking recent log errors..."
    
    local log_dirs=("$PROJECT_ROOT/logs" "/var/log/nginx")
    local error_count=0
    
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            # Check for recent errors (last 5 minutes)
            local recent_errors=$(find "$log_dir" -name "*.log" -type f -mmin -5 -exec grep -l "ERROR\|FATAL\|CRITICAL" {} \; 2>/dev/null | wc -l)
            error_count=$((error_count + recent_errors))
        fi
    done
    
    if [[ $error_count -gt 0 ]]; then
        print_warning "Found $error_count log files with recent errors"
        WARNING_CHECKS+=("recent-log-errors")
    else
        print_ok "No recent errors found in logs"
    fi
    
    return 0
}

# Check backup status
check_backups() {
    print_info "Checking backup status..."
    
    local backup_dir="$PROJECT_ROOT/backups"
    
    if [[ -d "$backup_dir" ]]; then
        local latest_backup=$(find "$backup_dir" -name "backup_*.sql*" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [[ -n "$latest_backup" ]]; then
            local backup_age=$(stat -c %Y "$latest_backup" 2>/dev/null || echo "0")
            local current_time=$(date +%s)
            local hours_since_backup=$(( (current_time - backup_age) / 3600 ))
            
            if [[ $hours_since_backup -gt 24 ]]; then
                print_warning "Latest backup is $hours_since_backup hours old"
                WARNING_CHECKS+=("backup-age")
            else
                print_ok "Latest backup is $hours_since_backup hours old"
            fi
            
            # Check backup size
            local backup_size=$(du -h "$latest_backup" 2>/dev/null | cut -f1)
            print_info "Latest backup size: $backup_size"
        else
            print_warning "No backups found"
            WARNING_CHECKS+=("no-backups")
        fi
    else
        print_warning "Backup directory not found"
        WARNING_CHECKS+=("backup-dir-missing")
    fi
    
    return 0
}

# Send Slack alert
send_slack_alert() {
    local message="$1"
    
    if [[ "$SEND_SLACK_ALERTS" != "true" || -z "$SLACK_WEBHOOK_URL" ]]; then
        return 0
    fi
    
    local payload=$(cat <<EOF
{
    "text": "🚨 Arbitration Detection System Alert",
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Health Check Alert*\n$message"
            }
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" &> /dev/null
}

# Output results in different formats
output_results() {
    case "$OUTPUT_FORMAT" in
        "json")
            output_json
            ;;
        "prometheus")
            output_prometheus
            ;;
        *)
            output_text
            ;;
    esac
}

# Output in text format
output_text() {
    echo
    echo "=== HEALTH CHECK SUMMARY ==="
    echo "Overall Status: $OVERALL_STATUS"
    echo "Timestamp: $(date)"
    
    if [[ ${#FAILED_CHECKS[@]} -gt 0 ]]; then
        echo
        echo "FAILED CHECKS:"
        for check in "${FAILED_CHECKS[@]}"; do
            echo "  - $check"
        done
    fi
    
    if [[ ${#WARNING_CHECKS[@]} -gt 0 ]]; then
        echo
        echo "WARNING CHECKS:"
        for check in "${WARNING_CHECKS[@]}"; do
            echo "  - $check"
        done
    fi
    
    if [[ ${#FAILED_CHECKS[@]} -eq 0 && ${#WARNING_CHECKS[@]} -eq 0 ]]; then
        echo
        echo "✅ All checks passed successfully!"
    fi
    
    echo "=========================="
}

# Output in JSON format
output_json() {
    local failed_json=$(printf '%s\n' "${FAILED_CHECKS[@]}" | jq -R . | jq -s .)
    local warning_json=$(printf '%s\n' "${WARNING_CHECKS[@]}" | jq -R . | jq -s .)
    
    cat <<EOF
{
    "status": "$OVERALL_STATUS",
    "timestamp": "$(date -Iseconds)",
    "failed_checks": $failed_json,
    "warning_checks": $warning_json,
    "summary": {
        "total_failed": ${#FAILED_CHECKS[@]},
        "total_warnings": ${#WARNING_CHECKS[@]}
    }
}
EOF
}

# Output in Prometheus format
output_prometheus() {
    echo "# HELP arbitration_health_status Overall health status (1=OK, 0.5=WARNING, 0=CRITICAL)"
    echo "# TYPE arbitration_health_status gauge"
    
    case "$OVERALL_STATUS" in
        "OK") echo "arbitration_health_status 1" ;;
        "WARNING") echo "arbitration_health_status 0.5" ;;
        *) echo "arbitration_health_status 0" ;;
    esac
    
    echo
    echo "# HELP arbitration_failed_checks_total Number of failed health checks"
    echo "# TYPE arbitration_failed_checks_total counter"
    echo "arbitration_failed_checks_total ${#FAILED_CHECKS[@]}"
    
    echo
    echo "# HELP arbitration_warning_checks_total Number of warning health checks"
    echo "# TYPE arbitration_warning_checks_total counter"
    echo "arbitration_warning_checks_total ${#WARNING_CHECKS[@]}"
}

# Main health check process
main() {
    print_info "=== Starting Arbitration Detection System Health Check ==="
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Load environment
    load_environment
    
    # Run health checks
    check_docker
    check_containers
    check_service_health "backend" "$BACKEND_HEALTH_URL"
    check_service_health "frontend" "$FRONTEND_HEALTH_URL"
    check_database
    check_redis
    check_system_resources
    check_ssl_certificates
    check_logs
    check_backups
    
    # Monitoring services (optional)
    if curl -s --max-time 5 "$PROMETHEUS_URL" &> /dev/null; then
        check_service_health "prometheus" "$PROMETHEUS_URL"
    fi
    
    if curl -s --max-time 5 "$GRAFANA_URL" &> /dev/null; then
        check_service_health "grafana" "$GRAFANA_URL"
    fi
    
    # Determine overall status
    if [[ ${#FAILED_CHECKS[@]} -gt 0 ]]; then
        OVERALL_STATUS="CRITICAL"
    elif [[ ${#WARNING_CHECKS[@]} -gt 0 ]]; then
        OVERALL_STATUS="WARNING"
    else
        OVERALL_STATUS="OK"
    fi
    
    # Send alerts if needed
    if [[ "$ALERT_ON_FAILURE" == "true" && "$OVERALL_STATUS" != "OK" ]]; then
        local alert_message="Health check failed with status: $OVERALL_STATUS\nFailed: ${#FAILED_CHECKS[@]} checks\nWarnings: ${#WARNING_CHECKS[@]} checks"
        send_slack_alert "$alert_message"
    fi
    
    # Output results
    output_results
    
    # Set exit code for Nagios mode
    if [[ "$NAGIOS_MODE" == "true" ]]; then
        case "$OVERALL_STATUS" in
            "OK") exit 0 ;;
            "WARNING") exit 1 ;;
            "CRITICAL") exit 2 ;;
            *) exit 3 ;;
        esac
    fi
    
    # Exit with error code if there are failures
    if [[ ${#FAILED_CHECKS[@]} -gt 0 ]]; then
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    print_warning "Health check interrupted"
    exit 130
}

trap cleanup INT TERM

# Parse arguments and run main function
parse_args "$@"
main