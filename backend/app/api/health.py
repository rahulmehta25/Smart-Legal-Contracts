"""
Health check endpoints for production monitoring
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import asyncio
import time
import psutil
import os
from datetime import datetime
from typing import Dict, Any

from app.db.database import get_db
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Dict[str, Any])
async def basic_health_check():
    """
    Basic health check endpoint for load balancers
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Arbitration RAG API",
        "version": "1.0.0"
    }


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check with all system components
    """
    settings = get_settings()
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Arbitration RAG API",
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database connectivity check
    try:
        start_time = time.time()
        db.execute(text("SELECT 1"))
        db_response_time = (time.time() - start_time) * 1000
        health_status["checks"]["database"] = {
            "status": "healthy",
            "response_time_ms": round(db_response_time, 2),
            "connection_pool": "active"
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # System resources check
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        health_status["checks"]["system"] = {
            "status": "healthy",
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "cpu_count": psutil.cpu_count()
        }
        
        # Check if resources are critically low
        if memory.percent > 90 or disk.percent > 90:
            health_status["checks"]["system"]["status"] = "warning"
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["checks"]["system"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Vector store check (if available)
    try:
        vector_path = settings.vector_store_path
        if os.path.exists(vector_path):
            health_status["checks"]["vector_store"] = {
                "status": "healthy",
                "path": vector_path,
                "size_mb": round(
                    sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(vector_path)
                        for filename in filenames
                    ) / (1024**2), 2
                )
            }
        else:
            health_status["checks"]["vector_store"] = {
                "status": "warning",
                "message": "Vector store path does not exist"
            }
    except Exception as e:
        health_status["checks"]["vector_store"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_status


@router.get("/health/readiness")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Kubernetes readiness probe - checks if service is ready to handle requests
    """
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        
        # Test critical directories exist
        settings = get_settings()
        required_paths = [
            settings.upload_directory,
            settings.vector_store_path
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise Exception(f"Required path does not exist: {path}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/health/liveness")
async def liveness_check():
    """
    Kubernetes liveness probe - basic service availability
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@router.get("/health/metrics")
async def health_metrics():
    """
    Basic metrics endpoint for monitoring systems
    """
    try:
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            },
            "process": {
                "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to collect metrics",
                "message": str(e)
            }
        )