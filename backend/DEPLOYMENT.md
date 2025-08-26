# Arbitration RAG API - Production Deployment Guide

This guide covers the production deployment of the optimized FastAPI backend for cloud environments.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd backend

# 2. Configure environment
cp .env.example .env
# Edit .env with your production settings

# 3. Deploy with Docker
chmod +x deploy.sh
./deploy.sh production
```

## 📋 Prerequisites

- Docker & Docker Compose
- PostgreSQL database (or SQLite for development)
- Redis for caching and rate limiting
- SSL certificates (for HTTPS)

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Core Configuration
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-super-secret-key

# Security & Performance
ALLOWED_ORIGINS=https://your-frontend.vercel.app
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
ENABLE_DOCS=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/app.log
```

### Database Setup

#### PostgreSQL (Recommended for Production)
```bash
# Create database
createdb arbitration_db

# Connection string
DATABASE_URL=postgresql://username:password@localhost:5432/arbitration_db
```

#### SQLite (Development Only)
```bash
DATABASE_URL=sqlite:///./arbitration_rag.db
```

## 🐳 Docker Deployment

### Production Deployment
```bash
# Full production stack with PostgreSQL + Redis + Nginx
./deploy.sh production
```

### Individual Services
```bash
# Backend only
docker-compose -f docker-compose.production.yml up backend -d

# Database + Redis
docker-compose -f docker-compose.production.yml up postgres redis -d
```

## ☁️ Cloud Deployment

### AWS ECS/Fargate
1. Push image to ECR:
```bash
docker build -t arbitration-rag-api .
docker tag arbitration-rag-api:latest <ecr-uri>:latest
docker push <ecr-uri>:latest
```

2. Create ECS task definition with:
   - CPU: 512-1024 (0.5-1 vCPU)
   - Memory: 1024-2048 MB
   - Environment variables from `.env`

### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy arbitration-rag-api \
  --image gcr.io/PROJECT_ID/arbitration-rag-api \
  --platform managed \
  --region us-central1 \
  --env-vars-file .env.yaml
```

### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name arbitration-rag-api \
  --image arbitration-rag-api:latest \
  --environment-variables $(cat .env)
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arbitration-rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arbitration-rag-api
  template:
    metadata:
      labels:
        app: arbitration-rag-api
    spec:
      containers:
      - name: api
        image: arbitration-rag-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: api-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 📊 Monitoring & Health Checks

### Health Endpoints
- Basic: `GET /health`
- Detailed: `GET /health/detailed`
- Readiness: `GET /health/readiness` (K8s)
- Liveness: `GET /health/liveness` (K8s)
- Metrics: `GET /health/metrics`

### Logging
Structured JSON logs with different levels:
- Console output for container logs
- File logging with rotation
- Error-specific log files

### Monitoring Setup
```bash
# View logs
docker-compose logs -f backend

# Monitor health
curl http://localhost:8000/health/detailed

# Check metrics
curl http://localhost:8000/health/metrics
```

## 🔒 Security Features

- Non-root container user
- Environment-based secrets
- Rate limiting with Redis
- CORS configuration
- Request/response logging
- Error sanitization for production
- SQL injection protection
- Input validation

## 🚦 Performance Optimizations

### Database
- Connection pooling (PostgreSQL)
- Connection pre-ping validation
- Configurable pool sizes
- Query optimization

### Application
- Multi-stage Docker builds
- Lightweight base images
- Dependency optimization
- Async request handling
- Circuit breakers for external services

### Caching
- Redis for rate limiting
- Application-level caching
- Vector store optimization

## 🔧 Maintenance

### Database Migrations
```bash
# Run migrations
docker-compose exec backend python -c "from app.db.database import init_db; init_db()"
```

### Backup & Recovery
```bash
# Database backup
docker-compose exec postgres pg_dump -U postgres arbitration_db > backup.sql

# Restore
docker-compose exec -T postgres psql -U postgres arbitration_db < backup.sql
```

### Updates
```bash
# Zero-downtime update
docker-compose build backend
docker-compose up -d --no-deps backend
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database status
   docker-compose ps postgres
   # View logs
   docker-compose logs postgres
   ```

2. **High Memory Usage**
   ```bash
   # Check metrics
   curl http://localhost:8000/health/metrics
   # Adjust batch sizes in .env
   BATCH_SIZE=8
   ```

3. **Rate Limiting Issues**
   ```bash
   # Check Redis
   docker-compose ps redis
   # Adjust limits
   RATE_LIMIT_REQUESTS=200
   ```

### Service Status
```bash
# All services
docker-compose ps

# Specific service health
docker-compose exec backend curl localhost:8000/health

# Resource usage
docker stats
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale backend replicas
docker-compose up -d --scale backend=3

# Load balancer configuration required
```

### Vertical Scaling
Update docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## 🔐 Production Checklist

- [ ] Environment variables configured
- [ ] Database properly configured and accessible
- [ ] Redis configured for caching
- [ ] SSL certificates installed (if HTTPS)
- [ ] CORS origins properly set
- [ ] Rate limits configured appropriately
- [ ] Logging destination configured
- [ ] Health checks responding correctly
- [ ] Monitoring system connected
- [ ] Backup strategy in place
- [ ] Security scan completed
- [ ] Load testing performed

## 📞 Support

For deployment issues:
1. Check health endpoints
2. Review application logs
3. Verify environment configuration
4. Test database connectivity
5. Check resource utilization

Performance metrics and detailed health information are available at `/health/detailed`.