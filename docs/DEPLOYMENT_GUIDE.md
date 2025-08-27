# Production Deployment Guide

This comprehensive guide covers deploying the Arbitration Detection System to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Environment Configuration](#environment-configuration)
- [SSL Certificate Setup](#ssl-certificate-setup)
- [Database Setup](#database-setup)
- [Application Deployment](#application-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Post-Deployment Verification](#post-deployment-verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- 4 CPU cores
- 8 GB RAM
- 100 GB SSD storage
- Ubuntu 20.04+ or similar Linux distribution

**Recommended Requirements:**
- 8 CPU cores
- 16 GB RAM
- 250 GB SSD storage
- Load balancer for high availability

### Software Dependencies

Install the following on your deployment server:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install additional tools
sudo apt install -y git curl wget jq openssl nginx certbot
```

## Infrastructure Setup

### 1. Server Setup

```bash
# Create application directory
sudo mkdir -p /opt/arbitration-detector
sudo chown $USER:$USER /opt/arbitration-detector
cd /opt/arbitration-detector

# Clone the repository
git clone https://github.com/your-org/arbitration-detector.git .
git checkout main  # or your production branch
```

### 2. Network Configuration

Configure firewall rules:

```bash
# Allow SSH
sudo ufw allow 22

# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow monitoring ports (restrict to internal network)
sudo ufw allow from 192.168.0.0/16 to any port 9090  # Prometheus
sudo ufw allow from 192.168.0.0/16 to any port 3001  # Grafana

# Enable firewall
sudo ufw --force enable
```

### 3. DNS Configuration

Set up DNS records for your domains:

```
A    arbitration-detector.com        -> YOUR_SERVER_IP
A    www.arbitration-detector.com    -> YOUR_SERVER_IP
A    api.arbitration-detector.com    -> YOUR_SERVER_IP
```

## Environment Configuration

### 1. Create Production Environment File

```bash
# Copy the example environment file
cp .env.example .env.production

# Edit the configuration
nano .env.production
```

### 2. Required Environment Variables

Update the following critical variables in `.env.production`:

```bash
# Database passwords (generate strong passwords)
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password

# Application secrets (generate with openssl rand -base64 32)
JWT_SECRET=your_jwt_secret_key
SESSION_SECRET=your_session_secret
ENCRYPTION_KEY=your_encryption_key

# External API keys
OPENAI_API_KEY=your_openai_api_key

# Domain configuration
DOMAIN_URL=arbitration-detector.com
FRONTEND_URL=https://arbitration-detector.com
BACKEND_URL=https://api.arbitration-detector.com

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

### 3. Validate Configuration

```bash
# Run configuration validation
./demo/scripts/validate-config.sh -e production

# Check for security issues
./demo/scripts/validate-config.sh --security-only .env.production
```

## SSL Certificate Setup

### Option 1: Let's Encrypt (Recommended)

```bash
# Stop any running web servers
sudo systemctl stop nginx

# Obtain certificates
sudo certbot certonly --standalone \
  -d arbitration-detector.com \
  -d www.arbitration-detector.com \
  -d api.arbitration-detector.com

# Copy certificates to nginx directory
sudo mkdir -p /opt/arbitration-detector/demo/nginx/ssl
sudo cp /etc/letsencrypt/live/arbitration-detector.com/fullchain.pem /opt/arbitration-detector/demo/nginx/ssl/app.crt
sudo cp /etc/letsencrypt/live/arbitration-detector.com/privkey.pem /opt/arbitration-detector/demo/nginx/ssl/app.key
sudo cp /etc/letsencrypt/live/arbitration-detector.com/fullchain.pem /opt/arbitration-detector/demo/nginx/ssl/api.crt
sudo cp /etc/letsencrypt/live/arbitration-detector.com/privkey.pem /opt/arbitration-detector/demo/nginx/ssl/api.key

# Set proper permissions
sudo chown -R $USER:$USER /opt/arbitration-detector/demo/nginx/ssl
chmod 600 /opt/arbitration-detector/demo/nginx/ssl/*.key
chmod 644 /opt/arbitration-detector/demo/nginx/ssl/*.crt
```

### Option 2: Self-Signed Certificates (Development)

```bash
# Generate self-signed certificates
./demo/nginx/ssl/generate.sh arbitration-detector.com
```

### Certificate Renewal

Set up automatic renewal for Let's Encrypt:

```bash
# Add renewal cron job
(crontab -l 2>/dev/null; echo "0 2 * * 1 certbot renew --quiet && docker-compose -f /opt/arbitration-detector/demo/production/docker-compose.prod.yml restart nginx") | crontab -
```

## Database Setup

### 1. Initialize Database

```bash
# Start only database services first
docker-compose -f demo/production/docker-compose.prod.yml up -d postgres redis

# Wait for database to be ready
sleep 30

# Check database connectivity
docker-compose -f demo/production/docker-compose.prod.yml exec postgres pg_isready -U postgres
```

### 2. Run Database Migrations

```bash
# If you have migration scripts
docker-compose -f demo/production/docker-compose.prod.yml exec backend python -m alembic upgrade head

# Or run initialization script
docker-compose -f demo/production/docker-compose.prod.yml exec postgres psql -U postgres -d arbitration_prod -f /docker-entrypoint-initdb.d/init.sql
```

### 3. Create Database Backup

```bash
# Create initial backup
./demo/scripts/backup.sh --type full --retention 30
```

## Application Deployment

### 1. Build and Deploy

```bash
# Build Docker images
docker-compose -f demo/production/docker-compose.prod.yml build

# Deploy all services
./demo/scripts/deploy.sh --environment production --tag latest

# Or deploy step by step
docker-compose -f demo/production/docker-compose.prod.yml up -d
```

### 2. Verify Deployment

```bash
# Check service status
docker-compose -f demo/production/docker-compose.prod.yml ps

# Check logs
docker-compose -f demo/production/docker-compose.prod.yml logs -f backend
docker-compose -f demo/production/docker-compose.prod.yml logs -f frontend

# Run health checks
./demo/scripts/health-check.sh -v
```

## Monitoring Setup

### 1. Deploy Monitoring Stack

```bash
# Deploy monitoring services
docker-compose -f demo/monitoring/docker-compose.monitoring.yml up -d

# Wait for services to start
sleep 60

# Verify monitoring stack
curl -f http://localhost:9090/-/ready  # Prometheus
curl -f http://localhost:3001/api/health  # Grafana
```

### 2. Configure Grafana

1. Access Grafana at `https://your-domain.com/grafana/`
2. Login with admin credentials from environment file
3. Import dashboards from `demo/monitoring/grafana-dashboards/`
4. Configure alert notification channels

### 3. Set Up Alerting

```bash
# Test alert manager configuration
docker-compose -f demo/monitoring/docker-compose.monitoring.yml exec alertmanager amtool config check

# Send test alert
curl -XPOST http://localhost:9093/api/v1/alerts -H "Content-Type: application/json" -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning"
  },
  "annotations": {
    "summary": "Test alert for deployment verification"
  }
}]'
```

## Post-Deployment Verification

### 1. Functional Tests

```bash
# Test API endpoints
curl -f https://api.arbitration-detector.com/health
curl -f https://api.arbitration-detector.com/docs

# Test frontend
curl -f https://arbitration-detector.com

# Test file upload
curl -X POST https://api.arbitration-detector.com/api/v1/documents \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-document.pdf"
```

### 2. Performance Tests

```bash
# Run load tests
cd backend/tests/load_tests
locust -f locustfile.py --headless \
  --users 10 \
  --spawn-rate 2 \
  --run-time 60s \
  --host https://api.arbitration-detector.com
```

### 3. Security Verification

```bash
# SSL/TLS test
curl -I https://arbitration-detector.com

# Security headers test
curl -I https://arbitration-detector.com | grep -E "(Strict-Transport-Security|X-Content-Type-Options|X-Frame-Options)"

# Test rate limiting
for i in {1..20}; do curl -s -o /dev/null -w "%{http_code}\n" https://api.arbitration-detector.com/health; done
```

## Backup and Recovery Setup

### 1. Configure Automated Backups

```bash
# Set up daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/arbitration-detector/demo/scripts/backup.sh --type full --retention 30") | crontab -

# Test backup process
./demo/scripts/backup.sh --type full --dry-run
```

### 2. Test Recovery Process

```bash
# Test database restore
./demo/scripts/restore.sh --database-only --dry-run backup_20231201_020000.sql.gz

# Test full restore
./demo/scripts/restore.sh --dry-run backup_20231201_020000.sql.gz
```

## Maintenance Tasks

### 1. Log Rotation

```bash
# Configure logrotate
sudo tee /etc/logrotate.d/arbitration-detector << EOF
/opt/arbitration-detector/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        docker-compose -f /opt/arbitration-detector/demo/production/docker-compose.prod.yml restart backend frontend
    endscript
}
EOF
```

### 2. System Updates

```bash
# Create update script
cat > /opt/arbitration-detector/update-system.sh << 'EOF'
#!/bin/bash
set -e

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose -f /opt/arbitration-detector/demo/production/docker-compose.prod.yml pull

# Restart services with new images
./demo/scripts/deploy.sh --environment production --skip-backup

# Clean up old images
docker image prune -f
EOF

chmod +x /opt/arbitration-detector/update-system.sh
```

## Troubleshooting

### Common Issues

1. **Services won't start**
   ```bash
   # Check logs
   docker-compose -f demo/production/docker-compose.prod.yml logs
   
   # Check disk space
   df -h
   
   # Check memory usage
   free -h
   ```

2. **Database connection issues**
   ```bash
   # Test database connectivity
   docker-compose -f demo/production/docker-compose.prod.yml exec postgres pg_isready
   
   # Check database logs
   docker-compose -f demo/production/docker-compose.prod.yml logs postgres
   ```

3. **SSL certificate issues**
   ```bash
   # Check certificate validity
   openssl x509 -in demo/nginx/ssl/app.crt -text -noout
   
   # Test SSL connection
   openssl s_client -connect arbitration-detector.com:443
   ```

4. **High resource usage**
   ```bash
   # Check container resource usage
   docker stats
   
   # Check system resources
   htop
   ```

### Emergency Procedures

1. **Service Rollback**
   ```bash
   # Quick rollback to previous version
   ./demo/scripts/rollback.sh --type application --force
   ```

2. **Emergency Stop**
   ```bash
   # Stop all services immediately
   docker-compose -f demo/production/docker-compose.prod.yml stop
   ```

3. **Database Recovery**
   ```bash
   # Restore from latest backup
   ./demo/scripts/restore.sh --database-only --force backup_latest.sql.gz
   ```

## Support and Contacts

- **Technical Support**: devops@arbitration-detector.com
- **Security Issues**: security@arbitration-detector.com
- **Emergency Contact**: +1-555-0123
- **Documentation**: https://docs.arbitration-detector.com
- **Status Page**: https://status.arbitration-detector.com

## Additional Resources

- [Runbook Repository](./runbooks/)
- [Monitoring Dashboards](../demo/monitoring/grafana-dashboards/)
- [API Documentation](https://api.arbitration-detector.com/docs)
- [Security Guidelines](./SECURITY.md)
- [Disaster Recovery Plan](./runbooks/disaster-recovery.md)