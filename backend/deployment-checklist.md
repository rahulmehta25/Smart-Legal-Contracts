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
