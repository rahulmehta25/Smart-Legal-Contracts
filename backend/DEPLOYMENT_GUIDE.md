# Railway Deployment Guide

## Deployment Steps

### 1. Railway Setup
1. Go to [railway.app](https://railway.app) and sign up/login
2. Connect your GitHub account
3. Create new project from GitHub repository

### 2. Environment Variables (Set in Railway Dashboard)
```
ENVIRONMENT=production
SECRET_KEY=your-super-secure-secret-key-here-64-chars-min
DATABASE_URL=postgresql://user:pass@host:port/db
PORT=8000
LOG_LEVEL=INFO
DEVICE=cpu
BATCH_SIZE=16
```

### 3. Database Setup
Railway will automatically provide PostgreSQL database.
The DATABASE_URL will be injected automatically.

### 4. Build Configuration
Railway will use the included `railway.json` for deployment settings:
- Uses Python 3.11
- Installs dependencies from requirements.txt
- Health check on /health endpoint
- Auto-restart on failure

### 5. Domain Configuration
Railway provides automatic HTTPS domains like:
`your-app-name.up.railway.app`

## Post-Deployment Checklist

- [ ] Health check endpoint working: `/health`
- [ ] API documentation accessible: `/docs`
- [ ] Database tables created successfully
- [ ] Vector store initialized
- [ ] CORS configured for frontend domain
- [ ] Environment variables properly set
- [ ] Log monitoring active

## Deployment Commands (Alternative CLI Method)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

## Monitoring and Logs

Access logs in Railway dashboard:
- Build logs show dependency installation
- Deploy logs show startup process
- Runtime logs show API requests and errors

## Rollback Procedure

1. Railway keeps deployment history
2. Click on previous deployment in dashboard
3. Click "Redeploy" to rollback

## Performance Optimization

- Uses CPU-only PyTorch for faster startup
- Reduced batch size for memory efficiency
- Health check timeout set to 300s for ML model loading
- Restart policy configured for high availability