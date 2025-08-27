# Frontend Integration Guide

This guide helps you integrate your new frontend at https://github.com/rahulmehta25/Arbitration-Frontend with the existing backend.

## Quick Start

### 1. Backend Setup

Start the backend services:

```bash
# Navigate to backend directory
cd /Users/rahulmehta/Desktop/Test

# Start database and cache
docker-compose -f demo/docker-compose.db.yml up -d

# Start backend API
cd backend
python app/main.py

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 2. Frontend Setup

In your frontend repository:

```bash
# Copy the environment file
cp /Users/rahulmehta/Desktop/Test/frontend-integration/.env.example .env.local

# Copy the API service file
cp /Users/rahulmehta/Desktop/Test/frontend-integration/api-service.ts src/services/api.ts

# Install dependencies
npm install axios

# Start development server
npm run dev
```

## API Endpoints

### Authentication
- **POST** `/api/v1/auth/register` - User registration
- **POST** `/api/v1/auth/login` - User login
- **POST** `/api/v1/auth/logout` - User logout
- **GET** `/api/v1/auth/me` - Get current user

### Document Analysis
- **POST** `/api/v1/analyze/text` - Analyze text for arbitration clauses
- **POST** `/api/v1/analyze/file` - Upload and analyze document (PDF, DOCX, TXT)
- **POST** `/api/v1/batch/upload` - Batch upload multiple documents
- **GET** `/api/v1/jobs/{id}/status` - Check batch job status

### Document Management
- **GET** `/api/v1/documents` - List all documents
- **GET** `/api/v1/documents/{id}` - Get specific document
- **DELETE** `/api/v1/documents/{id}` - Delete document
- **GET** `/api/v1/documents/{id}/download` - Download document
- **GET** `/api/v1/documents/search?q={query}` - Search documents

### Statistics
- **GET** `/api/v1/stats/overview` - System statistics
- **GET** `/api/v1/stats/history?days=30` - Analysis history
- **GET** `/api/v1/stats/clauses` - Clause distribution

### WebSocket Events
- Connect to `ws://localhost:8000/ws`
- Events: `analysis_progress`, `analysis_complete`, `error`

## Environment Variables

### Required for Frontend
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Required for Backend
```env
DATABASE_URL=postgresql://postgres:postgres123@localhost:5432/arbitration_db
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## Using the API Service

### Example: Text Analysis

```typescript
import { analysisService } from '@/services/api';

const analyzeDocument = async (text: string) => {
  try {
    const result = await analysisService.analyzeText(text);
    console.log('Analysis results:', result);
    // Result contains:
    // - detected_clauses: Array of found arbitration clauses
    // - confidence: Overall confidence score
    // - risk_level: 'high', 'medium', or 'low'
  } catch (error) {
    console.error('Analysis failed:', error);
  }
};
```

### Example: File Upload with Progress

```typescript
import { analysisService } from '@/services/api';

const uploadFile = async (file: File) => {
  try {
    const result = await analysisService.analyzeFile(
      file,
      (progress) => {
        console.log(`Upload progress: ${progress}%`);
      }
    );
    console.log('File analysis complete:', result);
  } catch (error) {
    console.error('Upload failed:', error);
  }
};
```

### Example: WebSocket Real-time Updates

```typescript
import { wsService } from '@/services/api';

// Connect to WebSocket
wsService.connect();

// Listen for analysis updates
wsService.on('analysis_progress', (data) => {
  console.log('Progress:', data.progress);
});

wsService.on('analysis_complete', (data) => {
  console.log('Analysis complete:', data.results);
});

// Send message
wsService.send('subscribe', { room: 'analysis_123' });
```

## Docker Deployment

### Full Stack with Docker Compose

```bash
# Start everything
docker-compose -f docker-compose.fullstack.yml up -d

# With monitoring
docker-compose -f docker-compose.fullstack.yml --profile monitoring up -d

# View logs
docker-compose -f docker-compose.fullstack.yml logs -f backend frontend
```

### Production Deployment with Nginx

```bash
# Start with nginx proxy
docker-compose -f docker-compose.fullstack.yml --profile production up -d

# Access via:
# - Frontend: http://localhost
# - API: http://localhost/api
# - Docs: http://localhost/docs
```

## CORS Configuration

The backend is configured to accept requests from:
- `http://localhost:3000` - Next.js default
- `http://localhost:5173` - Vite default
- `http://localhost:5174` - Vite alternative
- `https://arbitration-frontend.vercel.app` - Vercel deployment
- `https://arbitration-frontend.netlify.app` - Netlify deployment

Add custom origins via environment variable:
```bash
CORS_ORIGINS=https://your-domain.com,https://staging.your-domain.com
```

## Testing the Integration

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Test Analysis Endpoint
```bash
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Any disputes shall be resolved through binding arbitration."}'
```

### 3. Test File Upload
```bash
curl -X POST http://localhost:8000/api/v1/analyze/file \
  -F "file=@sample.pdf"
```

## Common Issues and Solutions

### CORS Errors
- Ensure backend is running with proper CORS configuration
- Check that your frontend URL is in the allowed origins list
- Verify the API URL in your .env file

### WebSocket Connection Failed
- Check if backend is running
- Verify WebSocket URL (ws:// not http://)
- Ensure no proxy is blocking WebSocket connections

### Authentication Issues
- JWT token expires after 24 hours by default
- Store token in localStorage or secure cookie
- Refresh token on 401 responses

### File Upload Limits
- Default max file size: 10MB
- Can be changed via `MAX_FILE_SIZE` environment variable
- Nginx has separate limit in production (50MB)

## Support Files Location

All integration files are in `/Users/rahulmehta/Desktop/Test/frontend-integration/`:
- `.env.example` - Environment variables template
- `api-service.ts` - Complete API service with TypeScript
- This guide - `FRONTEND_INTEGRATION_GUIDE.md`

## Next Steps

1. **Development**: Use the API service file in your frontend
2. **Testing**: Run the demo at `/demo/setup-and-run.sh`
3. **Production**: Deploy using Docker Compose or Kubernetes
4. **Monitoring**: Enable Grafana/Prometheus profiles

For production deployment, see `/docs/deployment/` for detailed guides.