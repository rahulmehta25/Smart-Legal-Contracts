#!/bin/bash

# Arbitration Detection System - System Status Checker
# This script provides comprehensive system health and status monitoring

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/deploy"
LOG_DIR="${SCRIPT_DIR}/logs"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is accessible
port_check() {
    local host="$1"
    local port="$2"
    local timeout="${3:-3}"
    
    if command_exists nc; then
        nc -z -w"${timeout}" "${host}" "${port}" 2>/dev/null
    elif command_exists telnet; then
        timeout "${timeout}" telnet "${host}" "${port}" >/dev/null 2>&1
    else
        # Fallback using /dev/tcp (bash built-in)
        timeout "${timeout}" bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null
    fi
}

# Function to make HTTP request with timeout
http_check() {
    local url="$1"
    local timeout="${2:-5}"
    
    if command_exists curl; then
        curl -sf --max-time "${timeout}" "${url}" >/dev/null 2>&1
    elif command_exists wget; then
        wget -q --timeout="${timeout}" --tries=1 --spider "${url}" >/dev/null 2>&1
    else
        return 1
    fi
}

# Function to get container status
get_container_status() {
    local container_name="$1"
    
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "running"
    elif docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "stopped"
    else
        echo "not_found"
    fi
}

# Function to get container health
get_container_health() {
    local container_name="$1"
    
    local health_status
    health_status=$(docker inspect "${container_name}" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_health_check")
    
    case "$health_status" in
        "healthy") echo "✅ Healthy" ;;
        "unhealthy") echo "❌ Unhealthy" ;;
        "starting") echo "🔄 Starting" ;;
        "no_health_check") echo "ℹ️  No health check" ;;
        *) echo "❓ Unknown" ;;
    esac
}

# Function to format uptime
format_uptime() {
    local container_name="$1"
    
    local created_at
    created_at=$(docker inspect "${container_name}" --format='{{.State.StartedAt}}' 2>/dev/null || echo "")
    
    if [[ -n "$created_at" ]]; then
        if command_exists python3; then
            python3 -c "
import datetime
import sys
try:
    created = datetime.datetime.fromisoformat('${created_at}'.replace('Z', '+00:00'))
    now = datetime.datetime.now(datetime.timezone.utc)
    uptime = now - created
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    if days > 0:
        print(f'{days}d {hours}h {minutes}m')
    elif hours > 0:
        print(f'{hours}h {minutes}m')
    else:
        print(f'{minutes}m')
except:
    print('N/A')
"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# Function to get resource usage
get_resource_usage() {
    local container_name="$1"
    
    if docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" "${container_name}" 2>/dev/null | tail -n 1; then
        return 0
    else
        echo "${container_name}	N/A	N/A	N/A	N/A"
    fi
}

# Function to check system requirements
check_system_requirements() {
    echo -e "${BLUE}=== System Requirements Check ===${NC}"
    echo ""
    
    # Check Docker
    if command_exists docker; then
        local docker_version
        docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        echo -e "  Docker: ${GREEN}✅ Installed${NC} (${docker_version})"
        
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            echo -e "  Docker Daemon: ${GREEN}✅ Running${NC}"
        else
            echo -e "  Docker Daemon: ${RED}❌ Not running${NC}"
        fi
    else
        echo -e "  Docker: ${RED}❌ Not installed${NC}"
    fi
    
    # Check Docker Compose
    if command_exists docker-compose; then
        local compose_version
        compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        echo -e "  Docker Compose: ${GREEN}✅ Installed${NC} (${compose_version})"
    elif docker compose version >/dev/null 2>&1; then
        local compose_version
        compose_version=$(docker compose version --short)
        echo -e "  Docker Compose: ${GREEN}✅ Installed${NC} (${compose_version})"
    else
        echo -e "  Docker Compose: ${RED}❌ Not installed${NC}"
    fi
    
    # System resources
    echo ""
    echo -e "${BLUE}System Resources:${NC}"
    
    if command_exists free; then
        local total_mem
        local available_mem
        total_mem=$(free -h | awk 'NR==2{print $2}')
        available_mem=$(free -h | awk 'NR==2{print $7}')
        echo -e "  RAM: ${total_mem} total, ${available_mem} available"
    elif [[ "$(uname)" == "Darwin" ]]; then
        local total_mem
        total_mem=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        echo -e "  RAM: ${total_mem}GB total"
    fi
    
    if command_exists df; then
        local disk_usage
        disk_usage=$(df -h "${SCRIPT_DIR}" | awk 'NR==2{print $4 " available"}')
        echo -e "  Disk: ${disk_usage}"
    fi
    
    # CPU info
    if command_exists nproc; then
        local cpu_cores
        cpu_cores=$(nproc)
        echo -e "  CPU Cores: ${cpu_cores}"
    elif [[ "$(uname)" == "Darwin" ]]; then
        local cpu_cores
        cpu_cores=$(sysctl -n hw.ncpu)
        echo -e "  CPU Cores: ${cpu_cores}"
    fi
}

# Function to check Docker containers
check_containers() {
    echo -e "${BLUE}=== Container Status ===${NC}"
    echo ""
    
    cd "${DEPLOY_DIR}" 2>/dev/null || {
        echo -e "${RED}Deploy directory not found: ${DEPLOY_DIR}${NC}"
        return 1
    }
    
    if [[ ! -f docker-compose.yml ]]; then
        echo -e "${RED}docker-compose.yml not found${NC}"
        return 1
    fi
    
    # Define expected services
    local services=(
        "postgres:arbitration-db:Database"
        "redis:arbitration-redis:Cache"
        "chroma:arbitration-vector-db:Vector DB"
        "elasticsearch:arbitration-elasticsearch:Search"
        "rabbitmq:arbitration-rabbitmq:Message Queue"
        "backend:arbitration-backend:Backend API"
        "frontend:arbitration-frontend:Frontend"
        "graphql:arbitration-graphql:GraphQL"
        "nginx:arbitration-proxy:Reverse Proxy"
        "prometheus:arbitration-prometheus:Monitoring"
        "grafana:arbitration-grafana:Dashboards"
        "jaeger:arbitration-jaeger:Tracing"
    )
    
    printf "%-20s %-10s %-15s %-10s %s\n" "Service" "Status" "Health" "Uptime" "Description"
    printf "%-20s %-10s %-15s %-10s %s\n" "-------" "------" "------" "------" "-----------"
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service container_name description <<< "${service_info}"
        
        local status
        status=$(get_container_status "${container_name}")
        
        local status_display
        case "$status" in
            "running") status_display="${GREEN}Running${NC}" ;;
            "stopped") status_display="${RED}Stopped${NC}" ;;
            "not_found") status_display="${YELLOW}Not Found${NC}" ;;
        esac
        
        local health="N/A"
        local uptime="N/A"
        
        if [[ "$status" == "running" ]]; then
            health=$(get_container_health "${container_name}")
            uptime=$(format_uptime "${container_name}")
        fi
        
        printf "%-20s %-20s %-25s %-10s %s\n" "${service}" "${status_display}" "${health}" "${uptime}" "${description}"
    done
}

# Function to check service connectivity
check_connectivity() {
    echo ""
    echo -e "${BLUE}=== Service Connectivity ===${NC}"
    echo ""
    
    # Define services to check
    local services=(
        "PostgreSQL:localhost:5432"
        "Redis:localhost:6379"
        "Chroma:localhost:8001"
        "Elasticsearch:localhost:9200"
        "RabbitMQ:localhost:5672"
        "RabbitMQ Management:localhost:15672"
        "Backend API:localhost:8000"
        "Frontend:localhost:3000"
        "GraphQL:localhost:4000"
        "Prometheus:localhost:9090"
        "Grafana:localhost:3001"
        "Jaeger:localhost:16686"
        "Spark Master:localhost:8080"
        "Kibana:localhost:5601"
    )
    
    printf "%-25s %-10s %s\n" "Service" "Port" "Status"
    printf "%-25s %-10s %s\n" "-------" "----" "------"
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service host port <<< "${service_info}"
        
        if port_check "${host}" "${port}"; then
            printf "%-25s %-10s ${GREEN}✅ Accessible${NC}\n" "${service}" "${port}"
        else
            printf "%-25s %-10s ${RED}❌ Not accessible${NC}\n" "${service}" "${port}"
        fi
    done
}

# Function to check HTTP endpoints
check_http_endpoints() {
    echo ""
    echo -e "${BLUE}=== HTTP Endpoints Health ===${NC}"
    echo ""
    
    # Define HTTP endpoints to check
    local endpoints=(
        "Backend Health:http://localhost:8000/health"
        "Backend API Docs:http://localhost:8000/docs"
        "GraphQL Playground:http://localhost:4000/graphql"
        "Frontend:http://localhost:3000"
        "Prometheus:http://localhost:9090/-/healthy"
        "Grafana:http://localhost:3001/api/health"
        "Elasticsearch Cluster:http://localhost:9200/_cluster/health"
        "Chroma Heartbeat:http://localhost:8001/api/v1/heartbeat"
        "Kibana Status:http://localhost:5601/api/status"
        "Jaeger Health:http://localhost:16686/"
        "Spark Master:http://localhost:8080"
    )
    
    printf "%-25s %-50s %s\n" "Service" "Endpoint" "Status"
    printf "%-25s %-50s %s\n" "-------" "--------" "------"
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r service url <<< "${endpoint_info}"
        
        if http_check "${url}"; then
            printf "%-25s %-50s ${GREEN}✅ Healthy${NC}\n" "${service}" "${url}"
        else
            printf "%-25s %-50s ${RED}❌ Unhealthy${NC}\n" "${service}" "${url}"
        fi
    done
}

# Function to check resource usage
check_resources() {
    echo ""
    echo -e "${BLUE}=== Resource Usage ===${NC}"
    echo ""
    
    if ! docker stats --no-stream >/dev/null 2>&1; then
        echo -e "${RED}Cannot retrieve Docker stats${NC}"
        return 1
    fi
    
    echo "Container resource usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $(docker ps --format "{{.Names}}" | grep -E "arbitration|chroma") 2>/dev/null | head -20
}

# Function to check logs for errors
check_logs() {
    echo ""
    echo -e "${BLUE}=== Recent Log Analysis ===${NC}"
    echo ""
    
    cd "${DEPLOY_DIR}" 2>/dev/null || return 1
    
    local containers=("arbitration-backend" "arbitration-frontend" "arbitration-postgres" "arbitration-redis")
    
    for container in "${containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            echo -e "${CYAN}${container}:${NC}"
            
            # Check for errors in last 50 lines
            local error_count
            error_count=$(docker logs "${container}" --tail 50 2>&1 | grep -i -E "(error|exception|failed|fatal)" | wc -l)
            
            if [[ $error_count -gt 0 ]]; then
                echo -e "  ${RED}⚠️  ${error_count} errors found in recent logs${NC}"
                docker logs "${container}" --tail 10 2>&1 | grep -i -E "(error|exception|failed|fatal)" | head -3 | sed 's/^/    /'
            else
                echo -e "  ${GREEN}✅ No recent errors${NC}"
            fi
            echo ""
        fi
    done
}

# Function to show performance metrics
show_performance_metrics() {
    echo ""
    echo -e "${BLUE}=== Performance Metrics ===${NC}"
    echo ""
    
    # Docker system overview
    if command_exists docker; then
        echo -e "${CYAN}Docker System:${NC}"
        docker system df 2>/dev/null | sed 's/^/  /' || echo "  Unable to get Docker system info"
        echo ""
    fi
    
    # Show disk usage for project
    echo -e "${CYAN}Project Disk Usage:${NC}"
    if command_exists du; then
        du -sh "${SCRIPT_DIR}" 2>/dev/null | sed 's/^/  Project size: /' || echo "  Unable to calculate project size"
        
        if [[ -d "${LOG_DIR}" ]]; then
            du -sh "${LOG_DIR}" 2>/dev/null | sed 's/^/  Logs size: /' || echo "  Unable to calculate logs size"
        fi
        
        if [[ -d "${SCRIPT_DIR}/backups" ]]; then
            du -sh "${SCRIPT_DIR}/backups" 2>/dev/null | sed 's/^/  Backups size: /' || echo "  No backups found"
        fi
    fi
}

# Function to show quick access URLs
show_access_urls() {
    echo ""
    echo -e "${BLUE}=== Quick Access URLs ===${NC}"
    echo ""
    
    local urls=(
        "🌐 Frontend Application:http://localhost:3000"
        "🔧 Backend API:http://localhost:8000"
        "📊 API Documentation:http://localhost:8000/docs"
        "🔍 GraphQL Playground:http://localhost:4000/graphql"
        "📈 Grafana Dashboard:http://localhost:3001"
        "🎯 Prometheus Metrics:http://localhost:9090"
        "🔍 Jaeger Tracing:http://localhost:16686"
        "📋 Kibana Logs:http://localhost:5601"
        "🐰 RabbitMQ Management:http://localhost:15672"
        "⚡ Spark Master UI:http://localhost:8080"
    )
    
    for url_info in "${urls[@]}"; do
        IFS=':' read -r description url <<< "${url_info}"
        echo -e "  ${description} ${BLUE}${url}${NC}"
    done
}

# Function to show system recommendations
show_recommendations() {
    echo ""
    echo -e "${BLUE}=== System Recommendations ===${NC}"
    echo ""
    
    local recommendations=()
    
    # Check if services are running
    local running_count=0
    local total_services=12
    
    for container in "arbitration-db" "arbitration-redis" "arbitration-backend" "arbitration-frontend"; do
        if [[ "$(get_container_status "${container}")" == "running" ]]; then
            ((running_count++))
        fi
    done
    
    if [[ $running_count -eq 0 ]]; then
        recommendations+=("🚀 Start the system with: ./start.sh")
    elif [[ $running_count -lt 4 ]]; then
        recommendations+=("⚠️  Some core services are not running. Check container status above.")
    fi
    
    # Check logs directory size
    if [[ -d "${LOG_DIR}" ]] && command_exists du; then
        local log_size_mb
        log_size_mb=$(du -sm "${LOG_DIR}" 2>/dev/null | cut -f1)
        if [[ $log_size_mb -gt 500 ]]; then
            recommendations+=("🗂️  Log directory is large (${log_size_mb}MB). Consider cleaning with: rm -rf ${LOG_DIR}/*")
        fi
    fi
    
    # Check for old containers
    local old_containers
    old_containers=$(docker ps -a --format "{{.Names}}" | grep -E "arbitration|chroma" | wc -l)
    local running_containers
    running_containers=$(docker ps --format "{{.Names}}" | grep -E "arbitration|chroma" | wc -l)
    
    if [[ $old_containers -gt $running_containers ]]; then
        recommendations+=("🧹 Clean up stopped containers with: ./reset.sh")
    fi
    
    # Check available resources
    if command_exists free; then
        local mem_usage
        mem_usage=$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')
        if [[ $mem_usage -gt 80 ]]; then
            recommendations+=("💾 Memory usage is high (${mem_usage}%). Consider stopping unused services.")
        fi
    fi
    
    # Show recommendations
    if [[ ${#recommendations[@]} -gt 0 ]]; then
        for rec in "${recommendations[@]}"; do
            echo -e "  ${rec}"
        done
    else
        echo -e "  ${GREEN}✅ System is running optimally${NC}"
    fi
}

# Main function
main() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║        Arbitration Detection System - Status Monitor         ║"
    echo "║                                                               ║"
    echo "║  Comprehensive system health and status check                 ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    check_system_requirements
    echo ""
    check_containers
    check_connectivity
    check_http_endpoints
    check_resources
    check_logs
    show_performance_metrics
    show_access_urls
    show_recommendations
    
    echo ""
    echo -e "${GREEN}Status check completed! ✅${NC}"
    echo -e "${BLUE}Run this script periodically to monitor system health${NC}"
}

# Handle script options
case "${1:-}" in
    --containers|-c)
        check_containers
        ;;
    --connectivity|-n)
        check_connectivity
        ;;
    --resources|-r)
        check_resources
        ;;
    --logs|-l)
        check_logs
        ;;
    --urls|-u)
        show_access_urls
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --containers, -c     Check container status only"
        echo "  --connectivity, -n   Check network connectivity only"
        echo "  --resources, -r      Check resource usage only"
        echo "  --logs, -l           Check recent logs only"
        echo "  --urls, -u           Show access URLs only"
        echo "  --help, -h           Show this help message"
        echo ""
        echo "Run without options for complete system status check"
        ;;
    *)
        main "$@"
        ;;
esac