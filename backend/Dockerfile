FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps (production-slim: no torch-audio, no qdrant, use CPU-only torch)
RUN pip install --no-cache-dir \
    fastapi>=0.68.0 \
    "uvicorn[standard]>=0.15.0" \
    "pydantic[email]>=1.8.0" \
    pydantic-settings>=2.0.0 \
    python-dotenv>=0.19.0 \
    sqlalchemy>=2.0.0 \
    loguru>=0.7.0 \
    redis>=5.0.0 \
    "passlib[bcrypt]>=1.7.4" \
    python-jose>=3.3.0 \
    spacy>=3.7.0 \
    sentence-transformers>=2.6.0 \
    transformers>=4.36.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.0.0 \
    python-multipart>=0.0.5 \
    aiofiles>=23.0.0 \
    httpx>=0.24.0 \
    psutil>=5.9.0 \
    python-jose[cryptography]>=3.3.0 \
    PyJWT>=2.0.0

# Set model cache to a persistent path inside the image
ENV HF_HOME=/app/models_cache
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/models_cache

# Download spacy model and sentence-transformers model at build time
RUN python -m spacy download en_core_web_sm
# Model downloaded lazily at runtime to avoid HF rate limits during build

COPY app/ ./app/

# Create directories
RUN mkdir -p logs uploads

ENV PORT=8080 \
    PYTHONPATH=/app \
    DATABASE_URL=sqlite:////tmp/arbitration_rag.db \
    USE_QDRANT=false

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
# Build 1774159331
