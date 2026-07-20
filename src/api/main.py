from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import os

from core.arbitration_detector import ArbitrationDetectionPipeline
from comparison.comparison_engine import ClauseComparisonEngine
from explainability.explainer import ArbitrationExplainer, VisualExplainer

app = FastAPI(title="Arbitration Clause Detection API")

# Initialize components
pipeline = ArbitrationDetectionPipeline(cache_enabled=True)
comparison_engine = ClauseComparisonEngine()
explainer = ArbitrationExplainer(pipeline.bert_detector)
visual_explainer = VisualExplainer()

class DetectionResponse(BaseModel):
    detected: bool
    confidence: float
    clause_text: Optional[str]
    location: Optional[Dict]
    explanation: Optional[Dict]
    similar_clauses: Optional[List[Dict]]
    recommendations: Optional[List[str]]

@app.post("/detect", response_model=DetectionResponse)
async def detect_arbitration(file: UploadFile = File(...), 
                            explain: bool = True,
                            compare: bool = True):
    """
    Detect arbitration clause in uploaded document
    
    Args:
        file: Document file (PDF, TXT)
        explain: Include explainability analysis
        compare: Include comparison with database
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Run detection
        result = pipeline.detect_arbitration_clause(tmp_path)
        
        if result:
            response = {
                "detected": True,
                "confidence": result.confidence,
                "clause_text": result.summary,  # Send summary, not full text
                "location": result.location
            }
            
            # Add explanation if requested
            if explain:
                explanation = explainer.explain_detection(
                    result.full_text,
                    result
                )
                response["explanation"] = explanation
            
            # Add comparison if requested
            if compare:
                comparison = comparison_engine.compare_clause(result.full_text)
                response["similar_clauses"] = comparison["similar_clauses"]
                response["recommendations"] = comparison["analysis"]["recommendations"]
            
            return DetectionResponse(**response)
        else:
            return DetectionResponse(
                detected=False,
                confidence=0.0,
                clause_text=None,
                location=None
            )
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

@app.post("/compare")
async def compare_clause(clause_text: str):
    """Compare a clause with the database"""
    comparison = comparison_engine.compare_clause(clause_text)
    return JSONResponse(content=comparison)

@app.post("/add_to_database")
async def add_clause(clause_data: Dict):
    """Add a new clause to the comparison database"""
    clause_id = comparison_engine.add_clause_to_database(clause_data)
    return {"success": True, "clause_id": clause_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Arbitration Clause Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "Upload document for arbitration clause detection",
            "/compare": "Compare clause text with database",
            "/add_to_database": "Add new clause to comparison database",
            "/health": "Health check"
        }
    }

