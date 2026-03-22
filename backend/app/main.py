from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import os

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Lazy imports - try each, skip if broken
routers = []

try:
    from app.api.health import router as health_router
    routers.append(("", health_router))
except Exception as e:
    logger.warning(f"Health router failed: {e}")

try:
    from app.api import documents_router, analysis_router
    routers.append(("/api/v1", documents_router))
    routers.append(("/api/v1", analysis_router))
except Exception as e:
    logger.warning(f"Core API routers failed: {e}")

try:
    from app.api.advanced_rag_api import router as advanced_rag_router
    routers.append(("/api/v1", advanced_rag_router))
except Exception as e:
    logger.warning(f"RAG router failed: {e}")

try:
    from app.api.batch_analysis import router as batch_analysis_router
    routers.append(("/api/v1", batch_analysis_router))
except Exception as e:
    logger.warning(f"Batch router failed: {e}")

try:
    from app.core.config import get_settings
    settings = get_settings()
except Exception as e:
    logger.warning(f"Settings failed: {e}")
    settings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Smart Legal Contracts API")
    try:
        from app.db.database import init_db
        init_db()
    except Exception as e:
        logger.warning(f"Database init failed: {e}. Running without DB.")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Legal Contracts API",
        description="AI-powered legal document analysis",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register whatever routers loaded successfully
    for prefix, router in routers:
        try:
            app.include_router(router, prefix=prefix)
        except Exception as e:
            logger.warning(f"Failed to register router: {e}")

    # Fallback health endpoint if health router didn't load
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "loaded_routers": len(routers),
        }

    @app.get("/")
    async def root():
        return {"message": "Smart Legal Contracts API", "version": "1.0.0", "docs": "/docs"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
