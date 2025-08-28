"""FastAPI application for RAG-based arbitration detection system."""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from ..core.arbitration_detector import ArbitrationDetectionPipeline, ArbitrationClause
from ..comparison.comparison_engine import ClauseComparisonEngine
from ..explainability.explainer import ArbitrationExplainer, VisualExplainer

# Initialize FastAPI app
app = FastAPI(
    title="RAG Arbitration Detection API",
    description="Advanced Legal-BERT based arbitration clause detection with RAG capabilities",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pipeline = ArbitrationDetectionPipeline(cache_enabled=True)
comparison_engine = ClauseComparisonEngine()

# Pydantic models
class DetectionRequest(BaseModel):
    """Request model for text-based detection."""
    text: str = Field(..., description="Text to analyze for arbitration clauses")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Detection confidence threshold")
    explain: bool = Field(True, description="Include explainability analysis")
    compare: bool = Field(True, description="Compare with database")

class DetectionResponse(BaseModel):
    """Response model for detection results."""
    detected: bool
    confidence: float
    clause_text: Optional[str] = None
    clause_type: Optional[str] = None
    location: Optional[Dict] = None
    key_provisions: Optional[List[str]] = None
    explanation: Optional[Dict] = None
    similar_clauses: Optional[List[Dict]] = None
    recommendations: Optional[List[str]] = None
    timestamp: str

class ClauseAddRequest(BaseModel):
    """Request model for adding a clause to database."""
    text: str = Field(..., description="Clause text")
    company: str = Field("Unknown", description="Company name")
    industry: str = Field("Unknown", description="Industry")
    document_type: str = Field("TOS", description="Document type")
    jurisdiction: str = Field("US", description="Jurisdiction")
    enforceability: float = Field(0.5, ge=0.0, le=1.0, description="Enforceability score")
    risk_score: float = Field(0.5, ge=0.0, le=1.0, description="Risk score")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

class ComparisonRequest(BaseModel):
    """Request model for clause comparison."""
    clause_text: str = Field(..., description="Clause text to compare")
    top_k: int = Field(10, ge=1, le=100, description="Number of similar clauses to return")

class BulkImportRequest(BaseModel):
    """Request model for bulk import."""
    clauses: List[Dict] = Field(..., description="List of clauses to import")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Arbitration Detection API",
        "version": "2.0.0",
        "documentation": "/api/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connectivity
        db_stats = comparison_engine.get_database_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "components": {
                "detection_pipeline": "operational",
                "database": db_stats.get('status', 'unknown'),
                "vector_store": db_stats.get('vector_store', {}).get('total_vectors', 0) > 0
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/detect", response_model=DetectionResponse)
async def detect_arbitration(
    file: UploadFile = File(...),
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    explain: bool = Query(True),
    compare: bool = Query(True)
):
    """
    Detect arbitration clause in uploaded document.
    
    Supports PDF, TXT, and DOCX files.
    """
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.text', '.doc', '.docx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Run detection
        result = pipeline.detect_arbitration_clause(tmp_path)
        
        response_data = {
            "detected": result is not None,
            "confidence": result.confidence if result else 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result:
            response_data.update({
                "clause_text": result.summary,
                "clause_type": result.clause_type,
                "location": result.location,
                "key_provisions": result.key_provisions
            })
            
            # Add explanation if requested
            if explain and result:
                try:
                    # Generate explanation
                    explanation = {
                        "detection_method": result.detection_method,
                        "confidence_breakdown": {
                            "overall": result.confidence,
                            "threshold": threshold
                        },
                        "key_indicators": result.key_provisions[:5] if result.key_provisions else []
                    }
                    response_data["explanation"] = explanation
                except Exception as e:
                    logger.error(f"Error generating explanation: {e}")
            
            # Add comparison if requested
            if compare and result:
                try:
                    comparison = comparison_engine.compare_clause(result.full_text)
                    response_data["similar_clauses"] = comparison.get("similar_clauses", [])
                    response_data["recommendations"] = comparison.get("analysis", {}).get("recommendations", [])
                except Exception as e:
                    logger.error(f"Error comparing clause: {e}")
        
        return DetectionResponse(**response_data)
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")

@app.post("/detect/text", response_model=DetectionResponse)
async def detect_arbitration_text(request: DetectionRequest):
    """
    Detect arbitration clause in provided text.
    """
    try:
        # Run detection on text
        result = pipeline.detect_from_text(request.text)
        
        response_data = {
            "detected": result is not None,
            "confidence": result.confidence if result else 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result:
            response_data.update({
                "clause_text": result.summary,
                "clause_type": result.clause_type,
                "location": result.location,
                "key_provisions": result.key_provisions
            })
            
            # Add comparison if requested
            if request.compare and result:
                comparison = comparison_engine.compare_clause(result.full_text)
                response_data["similar_clauses"] = comparison.get("similar_clauses", [])
                response_data["recommendations"] = comparison.get("analysis", {}).get("recommendations", [])
        
        return DetectionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error detecting arbitration in text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_clause(request: ComparisonRequest):
    """
    Compare a clause with the database.
    """
    try:
        comparison = comparison_engine.compare_clause(
            request.clause_text,
            request.top_k
        )
        return JSONResponse(content=comparison)
    except Exception as e:
        logger.error(f"Error comparing clause: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/add")
async def add_clause(request: ClauseAddRequest):
    """
    Add a new clause to the comparison database.
    """
    try:
        clause_data = request.dict()
        clause_id = comparison_engine.add_clause_to_database(clause_data)
        return {
            "success": True,
            "clause_id": clause_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding clause: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/bulk-import")
async def bulk_import(request: BulkImportRequest, background_tasks: BackgroundTasks):
    """
    Bulk import clauses to the database.
    """
    try:
        # Run import in background for large datasets
        if len(request.clauses) > 100:
            background_tasks.add_task(
                comparison_engine.bulk_import_clauses,
                request.clauses
            )
            return {
                "message": "Bulk import started in background",
                "count": len(request.clauses),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Import directly for small datasets
            result = comparison_engine.bulk_import_clauses(request.clauses)
            return result
            
    except Exception as e:
        logger.error(f"Error in bulk import: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/stats")
async def database_stats():
    """
    Get statistics about the comparison database.
    """
    try:
        stats = comparison_engine.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/search")
async def search_database(
    company: Optional[str] = None,
    industry: Optional[str] = None,
    document_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    min_risk: Optional[float] = None,
    max_risk: Optional[float] = None
):
    """
    Search clauses in the database with filters.
    """
    try:
        filters = {}
        if company:
            filters['company_name'] = company
        if industry:
            filters['industry'] = industry
        if document_type:
            filters['document_type'] = document_type
        if jurisdiction:
            filters['jurisdiction'] = jurisdiction
        if min_risk is not None:
            filters['min_risk'] = min_risk
        if max_risk is not None:
            filters['max_risk'] = max_risk
        
        results = comparison_engine.db_manager.search_clauses(filters)
        return {
            "count": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """
    Analyze multiple documents in batch.
    """
    results = []
    
    for file in files[:20]:  # Limit to 20 files
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Analyze file
            result = pipeline.detect_arbitration_clause(tmp_path)
            
            results.append({
                "filename": file.filename,
                "detected": result is not None,
                "confidence": result.confidence if result else 0.0,
                "clause_type": result.clause_type if result else None,
                "key_provisions": result.key_provisions if result else []
            })
        except Exception as e:
            logger.error(f"Error analyzing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    return {
        "total_files": len(files),
        "processed": len(results),
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/explain/{clause_id}")
async def get_explanation(clause_id: int):
    """
    Get detailed explanation for a specific clause.
    """
    try:
        # Get clause from database
        clause = comparison_engine.db_manager.get_clause(clause_id)
        if not clause:
            raise HTTPException(status_code=404, detail="Clause not found")
        
        # Generate explanation
        explanation = {
            "clause_id": clause_id,
            "clause_type": clause.get('document_type'),
            "risk_level": "High" if clause.get('risk_score', 0) > 0.7 else "Medium" if clause.get('risk_score', 0) > 0.4 else "Low",
            "enforceability": "Strong" if clause.get('enforceability_score', 0) > 0.7 else "Moderate" if clause.get('enforceability_score', 0) > 0.4 else "Weak",
            "key_provisions": clause.get('key_provisions', []),
            "jurisdiction": clause.get('jurisdiction'),
            "company": clause.get('company_name'),
            "industry": clause.get('industry')
        }
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")