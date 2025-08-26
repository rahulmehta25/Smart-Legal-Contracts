#!/bin/bash

# FastAPI Backend Deployment Script for Railway
# This script prepares the project for Railway deployment

echo "🚀 Preparing FastAPI Backend for Railway Deployment..."

# Check if we're in the backend directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    exit 1
fi

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Check git status
echo "📋 Checking git status..."
git status --porcelain

# Stage deployment files
echo "📁 Staging deployment files..."
git add railway.json Dockerfile .dockerignore requirements-production.txt .env.production DEPLOYMENT_GUIDE.md

# Commit deployment configuration
echo "💾 Committing deployment configuration..."
git commit -m "feat: add Railway deployment configuration

- Add railway.json for deployment settings
- Add production Dockerfile with multi-stage build
- Add .dockerignore for optimized builds
- Add production requirements.txt
- Add environment variable templates
- Add comprehensive deployment guide
- Configure CORS for production domains
- Update settings for environment-based configuration"

echo "✅ Deployment files prepared!"
echo ""
echo "🔧 Next steps:"
echo "1. Push to GitHub: git push origin feat/frontend-integration-support"
echo "2. Go to https://railway.app and create new project"
echo "3. Connect your GitHub repository"
echo "4. Set environment variables in Railway dashboard:"
echo "   - ENVIRONMENT=production"
echo "   - SECRET_KEY=your-64-char-secret-key"
echo "   - Add PostgreSQL database (Railway will set DATABASE_URL automatically)"
echo "5. Deploy will start automatically"
echo "6. Your API will be available at: https://[your-project-name].up.railway.app"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"

# Create a quick deployment checklist
echo "📝 Creating deployment checklist..."
cat > deployment-checklist.md << 'EOF'
# Deployment Checklist

## Pre-deployment
- [ ] All deployment files committed and pushed to GitHub
- [ ] Railway account created and connected to GitHub
- [ ] Repository connected to Railway project

## Railway Configuration
- [ ] Environment variables set:
  - [ ] ENVIRONMENT=production
  - [ ] SECRET_KEY (64+ characters)
  - [ ] LOG_LEVEL=INFO
- [ ] PostgreSQL database added to project
- [ ] Build and deploy settings configured

## Post-deployment Verification
- [ ] Health check responds: GET /health
- [ ] API docs accessible: GET /docs
- [ ] Root endpoint responds: GET /
- [ ] Database connection working
- [ ] Vector store initialized
- [ ] CORS working for frontend domain

## Frontend Integration
- [ ] Update frontend API URL to Railway deployment URL
- [ ] Test all API endpoints from frontend
- [ ] Verify WebSocket connections (if used)

## Monitoring
- [ ] Check Railway deployment logs
- [ ] Monitor error rates and response times
- [ ] Set up alerting for critical issues
EOF

echo "✅ Deployment preparation complete!"
echo "📄 Check deployment-checklist.md for step-by-step verification"