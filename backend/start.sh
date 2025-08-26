#!/bin/bash

# Production startup script for Railway deployment
# This script handles production startup with proper error handling and logging

set -e  # Exit on any error

echo "🚀 Starting Arbitration RAG API in production mode..."

# Set production environment if not already set
export ENVIRONMENT=${ENVIRONMENT:-production}
export PYTHONPATH=/app:$PYTHONPATH

# Log environment info
echo "📊 Environment: $ENVIRONMENT"
echo "🐍 Python version: $(python --version)"
echo "📦 Working directory: $(pwd)"

# Check if database URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "⚠️  Warning: DATABASE_URL not set, falling back to SQLite"
else
    echo "✅ Database URL configured"
fi

# Check if secret key is set
if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" == "your-secret-key-change-in-production" ]; then
    echo "❌ Error: SECRET_KEY must be set in production"
    exit 1
fi

# Download required NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded')
except Exception as e:
    print(f'⚠️ NLTK download failed: {e}')
" 2>/dev/null || echo "⚠️ NLTK download skipped"

# Download spaCy model
echo "🧠 Setting up spaCy model..."
python -c "
import spacy
try:
    spacy.load('en_core_web_sm')
    print('✅ spaCy model ready')
except OSError:
    print('⚠️ spaCy model not found, some features may be limited')
" 2>/dev/null || echo "⚠️ spaCy model check skipped"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads chroma_db logs

# Set proper permissions
chmod 755 uploads chroma_db

# Start the application with production settings
echo "🎯 Starting FastAPI application..."
exec python run.py