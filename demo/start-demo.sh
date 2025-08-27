#!/bin/bash

# Arbitration Clause Detector - Demo Startup Script
# This script sets up and starts the complete demo environment

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEMO_DIR")"
DEMO_URL="http://localhost:3001"
API_URL="http://localhost:8001"
DOCS_URL="http://localhost:8080"
ANALYTICS_URL="http://localhost:3002"

echo -e "${BLUE}🚀 Arbitration Clause Detector - Demo Setup${NC}"
echo "=================================================="
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
port_available() {
    local port=$1
    ! lsof -i:$port >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Waiting for $name to be ready"
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo ""
            print_status "$name is ready"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo ""
    print_error "$name failed to start within expected time"
    return 1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker and try again."
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    print_error "Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Check required ports
echo ""
echo "Checking port availability..."

required_ports=(3001 8001 8002 8003 8080 3002 5433 6380 9091)
ports_in_use=()

for port in "${required_ports[@]}"; do
    if ! port_available $port; then
        ports_in_use+=($port)
    fi
done

if [ ${#ports_in_use[@]} -ne 0 ]; then
    print_warning "The following ports are in use: ${ports_in_use[*]}"
    echo "Demo services might conflict. Consider stopping other services first."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_status "Port check completed"

# Create necessary directories
echo ""
echo "Creating demo directories..."

mkdir -p "$DEMO_DIR/sample-data/documents"
mkdir -p "$DEMO_DIR/sample-data/exports"
mkdir -p "$DEMO_DIR/sample-data/embeddings"
mkdir -p "$DEMO_DIR/logs"
mkdir -p "$DEMO_DIR/monitoring/prometheus"
mkdir -p "$DEMO_DIR/monitoring/grafana/dashboards"
mkdir -p "$DEMO_DIR/monitoring/grafana/datasources"
mkdir -p "$DEMO_DIR/nginx/conf.d"
mkdir -p "$DEMO_DIR/ssl"

print_status "Directories created"

# Create Nginx configuration for demo
echo ""
echo "Setting up Nginx configuration..."

cat > "$DEMO_DIR/nginx/gateway.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream demo-frontend {
        server demo-frontend:3000;
    }
    
    upstream demo-backend {
        server demo-backend:8000;
    }
    
    upstream demo-websocket {
        server demo-websocket:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Frontend
        location / {
            proxy_pass http://demo-frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Backend API
        location /api/ {
            proxy_pass http://demo-backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket
        location /ws/ {
            proxy_pass http://demo-websocket;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            return 200 "OK";
            add_header Content-Type text/plain;
        }
    }
}
EOF

# Create Prometheus configuration
cat > "$DEMO_DIR/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'demo-backend'
    static_configs:
      - targets: ['demo-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'demo-frontend'
    static_configs:
      - targets: ['demo-frontend:3000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Create Grafana datasource configuration
cat > "$DEMO_DIR/monitoring/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://demo-prometheus:9090
    isDefault: true
    editable: true
EOF

print_status "Configuration files created"

# Pull latest images
echo ""
echo "Pulling Docker images..."
cd "$DEMO_DIR"
docker-compose pull

print_status "Docker images updated"

# Build custom images
echo ""
echo "Building demo-specific images..."
docker-compose build

print_status "Demo images built"

# Start services
echo ""
echo "Starting demo services..."
docker-compose up -d

print_status "Services started"

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."

# Wait for backend
wait_for_service "$API_URL/health" "Backend API"

# Wait for frontend
wait_for_service "$DEMO_URL" "Frontend"

# Wait for documentation
wait_for_service "$DOCS_URL" "Documentation"

# Wait for analytics (Grafana)
if wait_for_service "$ANALYTICS_URL" "Analytics Dashboard"; then
    print_info "Grafana admin credentials: admin/demo123"
fi

# Load demo data
echo ""
echo "Loading demo data..."

# Give the database a moment to initialize
sleep 10

# Check if demo data is already loaded
if docker-compose exec -T demo-postgres psql -U demo_user -d demo_arbitration_db -c "SELECT COUNT(*) FROM users WHERE demo_account = true;" | grep -q "5"; then
    print_status "Demo data already loaded"
else
    print_info "Loading demo data into database..."
    if docker-compose exec -T demo-postgres psql -U demo_user -d demo_arbitration_db < "$DEMO_DIR/sample-data/db/demo-data.sql" >/dev/null 2>&1; then
        print_status "Demo data loaded successfully"
    else
        print_warning "Demo data loading had issues, but demo should still work"
    fi
fi

# Create demo summary
echo ""
echo "=================================================="
echo -e "${GREEN}🎉 Demo Environment Ready!${NC}"
echo "=================================================="
echo ""
echo "Demo URLs:"
echo "  📱 Main Demo:          $DEMO_URL"
echo "  🔧 API Documentation:  $API_URL/docs"
echo "  📚 User Guide:         $DOCS_URL"
echo "  📊 Analytics:          $ANALYTICS_URL"
echo ""
echo "Demo Accounts:"
echo "  👨‍💼 Admin:       admin@demo.com / Demo123!"
echo "  ⚖️  Legal Expert: lawyer@demo.com / Demo123!"
echo "  💼 Business User: business@demo.com / Demo123!"
echo ""
echo "API Keys:"
echo "  🔑 Admin API:     demo-api-key-12345"
echo "  🔑 Public API:    demo-public-key-99999"
echo ""
echo "Quick Start:"
echo "  1. Open $DEMO_URL in your browser"
echo "  2. Click 'Start Interactive Demo'"
echo "  3. Upload a sample document or use pre-loaded samples"
echo "  4. Explore AI analysis results and features"
echo ""
echo "Need Help?"
echo "  📖 Demo Script:   $DEMO_DIR/demo-script.md"
echo "  🛠️  Logs:          docker-compose logs [service-name]"
echo "  🔍 Status:        docker-compose ps"
echo "  ⏹️  Stop Demo:     docker-compose down"
echo ""

# Optional: Open browser
read -p "Open demo in browser? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Demo setup complete! Visit $DEMO_URL to begin."
else
    print_info "Opening demo in browser..."
    if command_exists open; then
        open "$DEMO_URL"  # macOS
    elif command_exists xdg-open; then
        xdg-open "$DEMO_URL"  # Linux
    elif command_exists start; then
        start "$DEMO_URL"  # Windows
    else
        echo "Please open $DEMO_URL in your browser manually."
    fi
fi

# Create stop script for easy cleanup
cat > "$DEMO_DIR/stop-demo.sh" << 'EOF'
#!/bin/bash
echo "Stopping Arbitration Clause Detector Demo..."
cd "$(dirname "${BASH_SOURCE[0]}")"
docker-compose down
echo "Demo stopped. Run ./start-demo.sh to restart."
EOF

chmod +x "$DEMO_DIR/stop-demo.sh"

echo ""
print_status "Demo setup complete!"
echo "Use ./stop-demo.sh to stop all demo services when finished."