#!/bin/bash

# Arbitration Clause Detector - Automated Setup Script
# This script sets up and runs a working demo of the system

set -e

echo "🚀 Arbitration Clause Detector - Setup Starting..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 is installed${NC}"
        return 0
    fi
}

echo "Checking prerequisites..."
check_command python3
check_command node
check_command npm
check_command docker

# Create project structure if it doesn't exist
echo -e "\n${YELLOW}Creating project structure...${NC}"
mkdir -p backend/app/{api,services,models,core}
mkdir -p frontend/src/{components,pages,services}
mkdir -p demo/sample-data/documents

# Setup Python backend
echo -e "\n${YELLOW}Setting up Python backend...${NC}"
cd backend

# Create minimal requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
aiofiles==23.2.1
pydantic==2.5.0
sqlalchemy==2.0.23
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
numpy==1.24.3
scikit-learn==1.3.2
redis==5.0.1
pytest==7.4.3
httpx==0.25.2
EOF

# Create simple working main.py
cat > app/main.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import re
import json
from datetime import datetime

app = FastAPI(title="Arbitration Clause Detector API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str
    language: str = "en"

class ClauseDetection(BaseModel):
    clause_type: str
    text: str
    confidence: float
    location: Dict[str, int]

class AnalysisResponse(BaseModel):
    has_arbitration: bool
    confidence: float
    clauses: List[ClauseDetection]
    summary: str
    processing_time: float

# Arbitration detection patterns
ARBITRATION_PATTERNS = {
    'binding_arbitration': [
        r'binding arbitration',
        r'mandatory arbitration',
        r'shall be resolved through arbitration',
        r'must be arbitrated',
        r'agree to arbitrate'
    ],
    'class_action_waiver': [
        r'waive.*class action',
        r'no class action',
        r'class action waiver',
        r'individual basis only'
    ],
    'jury_trial_waiver': [
        r'waive.*jury trial',
        r'no jury trial',
        r'jury trial waiver'
    ]
}

def detect_arbitration_clauses(text: str) -> AnalysisResponse:
    """Simple pattern-based arbitration detection"""
    import time
    start_time = time.time()
    
    text_lower = text.lower()
    detected_clauses = []
    max_confidence = 0.0
    
    for clause_type, patterns in ARBITRATION_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                confidence = 0.85 if 'binding' in pattern else 0.75
                detected_clauses.append(ClauseDetection(
                    clause_type=clause_type,
                    text=text[match.start():min(match.end() + 100, len(text))],
                    confidence=confidence,
                    location={"start": match.start(), "end": match.end()}
                ))
                max_confidence = max(max_confidence, confidence)
    
    has_arbitration = len(detected_clauses) > 0
    processing_time = time.time() - start_time
    
    summary = f"Found {len(detected_clauses)} arbitration-related clause(s)" if has_arbitration else "No arbitration clauses detected"
    
    return AnalysisResponse(
        has_arbitration=has_arbitration,
        confidence=max_confidence,
        clauses=detected_clauses,
        summary=summary,
        processing_time=processing_time
    )

@app.get("/")
async def root():
    return {"message": "Arbitration Clause Detector API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text for arbitration clauses"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = detect_arbitration_clauses(request.text)
    return result

@app.post("/api/v1/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file for arbitration clauses"""
    content = await file.read()
    text = content.decode('utf-8', errors='ignore')
    
    result = detect_arbitration_clauses(text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Setup virtual environment and install dependencies
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

# Start backend in background
echo -e "${GREEN}Starting backend server...${NC}"
nohup python app/main.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

cd ..

# Setup React frontend
echo -e "\n${YELLOW}Setting up React frontend...${NC}"
cd frontend

# Create package.json
cat > package.json << 'EOF'
{
  "name": "arbitration-detector-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "13.5.6",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "axios": "1.6.2"
  },
  "devDependencies": {
    "@types/node": "20.9.0",
    "@types/react": "18.2.37",
    "typescript": "5.2.2",
    "tailwindcss": "3.3.5",
    "autoprefixer": "10.4.16",
    "postcss": "8.4.31"
  }
}
EOF

# Create Next.js pages
mkdir -p pages
cat > pages/index.tsx << 'EOF'
import React, { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeText = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/v1/analyze', {
        text: text,
        language: 'en'
      });
      setResult(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Make sure the backend is running.');
    }
    setLoading(false);
  };

  const sampleText = `Any disputes arising out of or relating to these Terms of Service shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association. You agree to waive your right to a jury trial and to participate in class action lawsuits.`;

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-blue-600">
          Arbitration Clause Detector
        </h1>
        
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-2xl font-semibold mb-4">Enter Text to Analyze</h2>
          
          <textarea
            className="w-full h-48 p-4 border rounded-lg"
            placeholder="Paste your Terms of Service or legal document here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          
          <div className="flex gap-4 mt-4">
            <button
              onClick={analyzeText}
              disabled={!text || loading}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
            >
              {loading ? 'Analyzing...' : 'Analyze Text'}
            </button>
            
            <button
              onClick={() => setText(sampleText)}
              className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700"
            >
              Load Sample Text
            </button>
          </div>
        </div>

        {result && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">Analysis Results</h2>
            
            <div className="mb-4">
              <div className="flex items-center mb-2">
                <span className="font-semibold mr-2">Status:</span>
                <span className={result.has_arbitration ? 'text-red-600 font-bold' : 'text-green-600 font-bold'}>
                  {result.has_arbitration ? 'Arbitration Clause Detected!' : 'No Arbitration Clause Found'}
                </span>
              </div>
              
              <div className="flex items-center mb-2">
                <span className="font-semibold mr-2">Confidence:</span>
                <span>{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              
              <div className="flex items-center">
                <span className="font-semibold mr-2">Processing Time:</span>
                <span>{result.processing_time.toFixed(3)}s</span>
              </div>
            </div>
            
            {result.clauses.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold mb-2">Detected Clauses:</h3>
                {result.clauses.map((clause, idx) => (
                  <div key={idx} className="bg-gray-100 p-4 rounded mb-2">
                    <div className="font-semibold text-blue-600">{clause.clause_type}</div>
                    <div className="text-sm mt-1">{clause.text}</div>
                    <div className="text-xs text-gray-600 mt-1">
                      Confidence: {(clause.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
EOF

cat > pages/_app.tsx << 'EOF'
import '../styles/globals.css'
import type { AppProps } from 'next/app'

export default function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />
}
EOF

# Create styles
mkdir -p styles
cat > styles/globals.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

# Create config files
cat > next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
}

module.exports = nextConfig
EOF

cat > tailwind.config.js << 'EOF'
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOF

cat > postcss.config.js << 'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF

cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

# Install frontend dependencies
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
npm install

# Start frontend in background
echo -e "${GREEN}Starting frontend server...${NC}"
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

cd ..

# Create sample documents
echo -e "\n${YELLOW}Creating sample documents...${NC}"
cat > demo/sample-data/documents/sample_tos.txt << 'EOF'
TERMS OF SERVICE

Section 15: Dispute Resolution

Any dispute, claim or controversy arising out of or relating to these Terms of Service or the breach, termination, enforcement, interpretation or validity thereof, including the determination of the scope or applicability of this agreement to arbitrate, shall be determined by binding arbitration in San Francisco, California before one arbitrator. The arbitration shall be administered by the American Arbitration Association (AAA) in accordance with its Commercial Arbitration Rules. Judgment on the Award may be entered in any court having jurisdiction.

You and Company agree that each may bring claims against the other only in your or its individual capacity, and not as a plaintiff or class member in any purported class or representative proceeding. Further, you agree to waive your right to a jury trial.

The arbitrator shall have exclusive authority to resolve all disputes arising out of or relating to the interpretation, applicability, enforceability, or formation of these Terms of Service.
EOF

# Wait for services to start
echo -e "\n${YELLOW}Waiting for services to start...${NC}"
sleep 5

# Test the services
echo -e "\n${YELLOW}Testing services...${NC}"

# Test backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ Backend is running at http://localhost:8000${NC}"
else
    echo -e "${RED}❌ Backend failed to start. Check backend.log${NC}"
fi

# Test frontend
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is running at http://localhost:3000${NC}"
else
    echo -e "${YELLOW}⏳ Frontend is still starting up...${NC}"
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Access the demo at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the services:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Logs are available at:"
echo "  Backend: backend/backend.log"
echo "  Frontend: frontend/frontend.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Keep script running
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait