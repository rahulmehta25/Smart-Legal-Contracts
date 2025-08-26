#!/bin/bash

# Production deployment script for Arbitration RAG API
# Usage: ./deploy.sh [environment]

set -e  # Exit on any error

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Deploying Arbitration RAG API to ${ENVIRONMENT} environment"
echo "================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Environment-specific configurations
case $ENVIRONMENT in
    "production")
        COMPOSE_FILE="docker-compose.production.yml"
        ENV_FILE=".env"
        ;;
    "staging")
        COMPOSE_FILE="docker-compose.staging.yml"
        ENV_FILE=".env.staging"
        ;;
    "development")
        COMPOSE_FILE="docker-compose.yml"
        ENV_FILE=".env.development"
        ;;
    *)
        echo "❌ Unknown environment: $ENVIRONMENT"
        echo "   Supported environments: production, staging, development"
        exit 1
        ;;
esac

# Check if environment file exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ Environment file $ENV_FILE not found."
    echo "   Please copy .env.example to $ENV_FILE and configure it."
    exit 1
fi

echo "✅ Using environment file: $ENV_FILE"
echo "✅ Using compose file: $COMPOSE_FILE"

# Load environment variables
export $(grep -v '^#' $ENV_FILE | xargs)

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if required environment variables are set
REQUIRED_VARS=("DATABASE_URL" "SECRET_KEY" "REDIS_URL")
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

echo "✅ Environment variables check passed"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs uploads chroma_db
chmod 755 logs uploads chroma_db

# Build and start services
echo "🔧 Building and starting services..."
docker-compose -f $COMPOSE_FILE down --remove-orphans
docker-compose -f $COMPOSE_FILE build --no-cache
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend service is healthy"
        break
    else
        echo "⏳ Backend service not ready yet (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Backend service failed to become healthy"
    echo "📊 Service logs:"
    docker-compose -f $COMPOSE_FILE logs backend
    exit 1
fi

# Run database migrations if needed
echo "🗄️ Running database migrations..."
docker-compose -f $COMPOSE_FILE exec -T backend python -c "
from app.db.database import init_db
try:
    init_db()
    print('✅ Database initialized successfully')
except Exception as e:
    print(f'❌ Database initialization failed: {e}')
    exit(1)
"

# Display deployment summary
echo ""
echo "🎉 Deployment completed successfully!"
echo "================================================="
echo "Environment: $ENVIRONMENT"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs (if enabled)"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "📊 Service Status:"
docker-compose -f $COMPOSE_FILE ps

echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
echo "  Restart services: docker-compose -f $COMPOSE_FILE restart"
echo "  Scale backend: docker-compose -f $COMPOSE_FILE up -d --scale backend=3"
echo ""
echo "✅ Deployment complete!"