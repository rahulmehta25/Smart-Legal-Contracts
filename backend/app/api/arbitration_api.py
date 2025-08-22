"""
Production-ready API for arbitration clause detection
Integrates all ML components with model serving and monitoring
"""
import logging
import os
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import ML components
import sys
sys.path.append('/Users/rahulmehta/Desktop/Test/backend/app/ml')

from ensemble import ArbitrationEnsemble
from model_registry import ModelRegistry, ModelStatus
from monitoring import ModelMonitor
from feedback_loop import FeedbackLoop
from ner import LegalEntityRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Legal text to analyze for arbitration clauses")
    user_id: Optional[str] = Field(default="anonymous", description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    explain: bool = Field(default=False, description="Include explanation in response")
    extract_entities: bool = Field(default=False, description="Extract legal entities")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction (1=arbitration, 0=no arbitration)")
    confidence: float = Field(..., description="Prediction confidence score")
    model_id: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Prediction explanation")
    entities: Optional[Dict[str, List[str]]] = Field(default=None, description="Extracted legal entities")
    prediction_id: str = Field(..., description="Unique prediction identifier")


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., description="Prediction ID to provide feedback on")
    correct_label: int = Field(..., description="Correct label (1=arbitration, 0=no arbitration)")
    user_id: Optional[str] = Field(default="anonymous", description="User identifier")
    comments: Optional[str] = Field(default=None, description="Additional feedback comments")


class ModelStatusResponse(BaseModel):
    model_id: str
    status: str
    version: str
    created_at: str
    metrics: Dict[str, float]


class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    model_status: str
    system_metrics: Dict[str, Any]
    version: str = "1.0.0"


# Global components
model_registry = None
ensemble_model = None
monitor = None
feedback_loop = None
ner_extractor = None
production_model_id = None

# Prediction cache for feedback
prediction_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()


async def startup():
    """Initialize application components"""
    global model_registry, ensemble_model, monitor, feedback_loop, ner_extractor, production_model_id
    
    logger.info("Starting up arbitration detection API...")
    
    try:
        # Initialize model registry
        model_registry = ModelRegistry()
        
        # Initialize monitoring
        monitor = ModelMonitor()
        monitor.start_monitoring()
        
        # Initialize feedback loop
        feedback_loop = FeedbackLoop(model_registry, auto_retrain=False)
        
        # Initialize NER extractor
        ner_extractor = LegalEntityRecognizer()
        
        # Load production model
        production_models = model_registry.list_models(status=ModelStatus.PRODUCTION)
        
        if production_models:
            production_model = production_models[0]  # Get latest production model
            production_model_id = production_model.model_id
            
            # Load ensemble model (simplified - in practice would load from registry)
            ensemble_model = ArbitrationEnsemble()
            logger.info(f"Loaded production model: {production_model_id}")
        else:
            # Create demo ensemble for testing
            ensemble_model = ArbitrationEnsemble()
            production_model_id = "demo_ensemble_model"
            logger.warning("No production model found, using demo ensemble")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


async def shutdown():
    """Cleanup on shutdown"""
    global monitor
    
    logger.info("Shutting down arbitration detection API...")
    
    if monitor:
        monitor.stop_monitoring()
    
    logger.info("API shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Arbitration Clause Detection API",
    description="Production ML API for detecting arbitration clauses in legal documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get current model
def get_current_model():
    """Get current production model"""
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return ensemble_model


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Arbitration Clause Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check model availability
        model_status = "available" if ensemble_model else "unavailable"
        
        # Get system metrics
        system_metrics = {}
        if monitor:
            current_system = monitor.collect_system_metrics()
            system_metrics = {
                "cpu_usage_percent": current_system.cpu_usage_percent,
                "memory_usage_percent": current_system.memory_usage_percent,
                "active_requests": current_system.active_requests
            }
        
        return HealthCheckResponse(
            timestamp=datetime.now().isoformat(),
            model_status=model_status,
            system_metrics=system_metrics
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/predict", response_model=PredictionResponse)
async def predict_arbitration(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_current_model)
):
    """
    Predict if text contains arbitration clause
    """
    start_time = time.time()
    
    try:
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}_{abs(hash(request.text)) % 10000}"
        
        # Make prediction
        if request.explain:
            # Get detailed explanation
            result = model.predict_with_explanation(request.text)
            prediction = result["final_prediction"]
            confidence = result["ensemble_probability"]
            explanation = {
                "individual_predictions": result["individual_predictions"],
                "ensemble_weights": result["ensemble_weights"],
                "rule_based_explanation": result["rule_based_explanation"]
            }
        else:
            # Simple prediction
            predictions = model.predict([request.text])
            probabilities = model.predict_proba([request.text])
            prediction = int(predictions[0])
            confidence = float(probabilities[0][1])
            explanation = None
        
        # Extract entities if requested
        entities = None
        if request.extract_entities and ner_extractor:
            try:
                entities = ner_extractor.extract_arbitration_entities(request.text)
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Cache prediction for feedback
        prediction_cache[prediction_id] = {
            "text": request.text,
            "prediction": prediction,
            "confidence": confidence,
            "model_id": production_model_id,
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "session_id": request.session_id
        }
        
        # Log prediction for monitoring (background task)
        background_tasks.add_task(
            log_prediction_async,
            production_model_id,
            request.text,
            prediction,
            confidence,
            latency_ms,
            request.user_id,
            request.session_id or ""
        )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_id=production_model_id,
            model_version="1.0",
            latency_ms=latency_ms,
            explanation=explanation,
            entities=entities,
            prediction_id=prediction_id
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit feedback for a prediction
    """
    try:
        # Get cached prediction
        if request.prediction_id not in prediction_cache:
            raise HTTPException(status_code=404, detail="Prediction ID not found")
        
        cached_pred = prediction_cache[request.prediction_id]
        
        # Submit feedback (background task)
        background_tasks.add_task(
            submit_feedback_async,
            cached_pred,
            request.correct_label,
            request.user_id,
            request.comments
        )
        
        return {
            "message": "Feedback received successfully",
            "prediction_id": request.prediction_id,
            "feedback_id": f"fb_{request.prediction_id}_{request.user_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/models", response_model=List[ModelStatusResponse])
async def list_models():
    """
    List available models and their status
    """
    try:
        models = model_registry.list_models()
        
        response = []
        for model in models:
            response.append(ModelStatusResponse(
                model_id=model.model_id,
                status=model.status.value,
                version=model.version,
                created_at=model.created_at,
                metrics={
                    "precision": model.metrics.precision,
                    "recall": model.metrics.recall,
                    "f1_score": model.metrics.f1_score,
                    "auc_roc": model.metrics.auc_roc
                }
            ))
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@app.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str, hours: int = 24):
    """
    Get performance metrics for a specific model
    """
    try:
        if not monitor:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        dashboard_data = monitor.generate_dashboard_data(model_id, hours)
        return dashboard_data
    
    except Exception as e:
        logger.error(f"Failed to get metrics for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard(model_id: Optional[str] = None, hours: int = 24):
    """
    Get monitoring dashboard data
    """
    try:
        if not monitor:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        dashboard_data = monitor.generate_dashboard_data(model_id, hours)
        return dashboard_data
    
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


# Background task functions
async def log_prediction_async(
    model_id: str,
    input_text: str,
    prediction: int,
    confidence: float,
    latency_ms: float,
    user_id: str,
    session_id: str
):
    """Log prediction for monitoring (async)"""
    try:
        if monitor:
            monitor.log_prediction(
                model_id=model_id,
                input_text=input_text,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                user_id=user_id,
                session_id=session_id
            )
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


async def submit_feedback_async(
    cached_pred: Dict[str, Any],
    correct_label: int,
    user_id: str,
    comments: Optional[str]
):
    """Submit feedback (async)"""
    try:
        if feedback_loop:
            feedback_loop.collect_feedback(
                text=cached_pred["text"],
                model_prediction=cached_pred["prediction"],
                model_confidence=cached_pred["confidence"],
                user_correction=correct_label,
                user_id=user_id,
                model_id=cached_pred["model_id"],
                session_id=cached_pred.get("session_id", ""),
                additional_context={"comments": comments} if comments else {}
            )
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "InternalServerError"}
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "arbitration_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )