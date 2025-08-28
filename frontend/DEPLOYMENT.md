# Vercel Deployment Guide

This guide provides comprehensive instructions for deploying the Arbitration Detector frontend to Vercel with proper FastAPI backend integration.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ installed
- Vercel CLI installed (`npm install -g vercel`)
- Access to your FastAPI backend deployment URL
- Vercel account with appropriate permissions

### One-Command Deployment

```bash
# Set your backend URL
export BACKEND_URL="https://your-backend-api.com"

# Run the deployment script
./scripts/deploy.sh
```

## 📋 Detailed Setup Instructions

### 1. Environment Configuration

#### Required Environment Variables

Set these variables in your Vercel dashboard or use the deployment script:

```env
# Backend API Configuration
NEXT_PUBLIC_BACKEND_URL=https://your-backend-api.com
NEXT_PUBLIC_API_URL=https://your-backend-api.com
NEXT_PUBLIC_WS_URL=wss://your-backend-api.com/ws
BACKEND_URL=https://your-backend-api.com

# Application Configuration
NEXT_PUBLIC_ENV=production
NEXT_PUBLIC_APP_NAME="Arbitration Detector"
NEXT_PUBLIC_APP_VERSION="1.0.0"

# Build Configuration
NEXT_TELEMETRY_DISABLED=1
```

#### Optional Environment Variables

```env
# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_DEBUG=false
NEXT_PUBLIC_ENABLE_ERROR_REPORTING=true

# Third-party Services
NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
NEXT_PUBLIC_GA_TRACKING_ID=GA-XXXXX-X

# Performance Tuning
NEXT_PUBLIC_API_TIMEOUT=30000
NEXT_PUBLIC_MAX_FILE_SIZE=10485760
```

### 2. Local Environment Setup

1. **Copy environment template:**
   ```bash
   cp .env.example .env.local
   ```

2. **Update variables for local development:**
   ```env
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8001
   NEXT_PUBLIC_API_URL=http://localhost:8001
   NEXT_PUBLIC_WS_URL=ws://localhost:8001/ws
   NEXT_PUBLIC_ENABLE_DEBUG=true
   ```

3. **Install dependencies:**
   ```bash
   npm install
   ```

4. **Test local build:**
   ```bash
   npm run build
   npm start
   ```

### 3. Vercel Dashboard Configuration

#### Project Settings

1. **Framework Preset:** Next.js
2. **Node.js Version:** 18.x
3. **Build Command:** `npm run build`
4. **Output Directory:** `.next`
5. **Install Command:** `npm install`

#### Environment Variables Setup

Navigate to **Settings > Environment Variables** in your Vercel project dashboard and add:

| Variable | Value | Environment |
|----------|-------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | `https://your-backend-api.com` | Production, Preview |
| `NEXT_PUBLIC_API_URL` | `https://your-backend-api.com` | Production, Preview |
| `NEXT_PUBLIC_WS_URL` | `wss://your-backend-api.com/ws` | Production, Preview |
| `BACKEND_URL` | `https://your-backend-api.com` | Production, Preview |
| `NEXT_PUBLIC_ENV` | `production` | Production |
| `NEXT_TELEMETRY_DISABLED` | `1` | All |

### 4. Deployment Methods

#### Method 1: Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy to production
vercel --prod

# Deploy to preview
vercel
```

#### Method 2: Git Integration

1. Connect your repository to Vercel
2. Set environment variables in dashboard
3. Push to main branch for production deployment
4. Push to other branches for preview deployments

#### Method 3: Deployment Script

Use the provided deployment script for automated deployment:

```bash
# Production deployment
./scripts/deploy.sh

# Preview deployment
./scripts/deploy.sh --preview

# Setup environment variables only
./scripts/deploy.sh --setup-env

# Skip build step (if already built)
./scripts/deploy.sh --skip-build
```

## 🔧 Configuration Details

### vercel.json Configuration

The project includes a comprehensive `vercel.json` configuration:

- **API Routing:** Proxies `/api/v1/*` to your FastAPI backend
- **Security Headers:** Comprehensive security header setup
- **Performance:** Optimized caching and compression
- **CORS:** Proper CORS handling for cross-origin requests

### Next.js Configuration

Key optimizations in `next.config.js`:

- **Performance:** SWC minification, compression, bundle splitting
- **Security:** Security headers, powered-by header removal
- **Images:** WebP/AVIF support, optimized caching
- **API Proxying:** Automatic backend URL resolution

### Environment Resolution

The application automatically resolves backend URLs based on environment:

1. **Production:** Uses `NEXT_PUBLIC_BACKEND_URL` or `BACKEND_URL`
2. **Development:** Falls back to `http://localhost:8001`
3. **Preview:** Uses preview-specific environment variables

## 🔍 Troubleshooting

### Common Issues

#### 1. CORS Errors

**Problem:** API requests fail with CORS errors
**Solution:**
- Verify backend CORS configuration includes your Vercel domain
- Check `allowed_origins` in your FastAPI backend
- Ensure `*.vercel.app` is in allowed origins

#### 2. Environment Variables Not Loading

**Problem:** API calls fail due to undefined environment variables
**Solution:**
- Check environment variables are set in Vercel dashboard
- Ensure variables are prefixed with `NEXT_PUBLIC_` for client-side access
- Verify `.env.local` for local development

#### 3. Build Failures

**Problem:** Deployment fails during build process
**Solution:**
```bash
# Test build locally
npm run build

# Check TypeScript errors
npm run type-check

# Fix linting issues
npm run lint
```

#### 4. API Endpoint Not Found

**Problem:** 404 errors on API routes
**Solution:**
- Verify backend URL is correct and accessible
- Check API endpoint paths in backend
- Ensure rewrites configuration in `vercel.json` is correct

### Debug Mode

Enable debug mode for detailed logging:

```env
NEXT_PUBLIC_ENABLE_DEBUG=true
```

This enables:
- API request/response logging
- Environment configuration logging
- Detailed error information

### Health Checks

Test deployment health:

```bash
# Check frontend health
curl https://your-app.vercel.app

# Check API proxy health
curl https://your-app.vercel.app/api/health

# Check WebSocket connectivity
# Use browser dev tools to test WS connection
```

## 📊 Performance Optimization

### Build Optimization

The deployment is optimized for performance:

- **Code Splitting:** Automatic chunk splitting for optimal loading
- **Image Optimization:** WebP/AVIF support with lazy loading
- **Caching:** Aggressive caching for static assets
- **Compression:** Gzip compression for all assets

### Monitoring

Set up monitoring for production:

1. **Vercel Analytics:** Built-in performance monitoring
2. **Sentry Error Tracking:** Add `NEXT_PUBLIC_SENTRY_DSN`
3. **Google Analytics:** Add `NEXT_PUBLIC_GA_TRACKING_ID`

### Performance Metrics

Monitor these key metrics:
- **First Contentful Paint (FCP)**
- **Largest Contentful Paint (LCP)**
- **Cumulative Layout Shift (CLS)**
- **First Input Delay (FID)**

## 🔐 Security Considerations

### Security Headers

Automatically configured security headers:
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy`: Restrictive CSP policy

### Environment Security

- Never commit `.env.local` to version control
- Use Vercel's secure environment variable storage
- Rotate API keys and tokens regularly
- Use HTTPS for all production deployments

## 🚀 Advanced Deployment

### Multi-Environment Setup

Configure different environments:

```bash
# Production
vercel --prod

# Staging
vercel --target staging

# Preview (branch deployments)
vercel
```

### Custom Domains

1. Add custom domain in Vercel dashboard
2. Configure DNS records
3. SSL certificates are automatically provisioned

### CI/CD Integration

For automated deployments, create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Vercel
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
```

## 📞 Support

For deployment issues:
1. Check Vercel deployment logs
2. Review browser console for errors
3. Verify backend API accessibility
4. Check environment variable configuration

## 📝 Changelog

### v1.0.0
- Initial Vercel deployment configuration
- FastAPI backend integration
- Performance optimizations
- Security hardening
- Comprehensive documentation