#!/usr/bin/env python3
"""
Run script for the Arbitration RAG API
"""

import uvicorn
import os
from app.core.config import settings

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )