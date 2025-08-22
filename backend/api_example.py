"""
FastAPI example for Arbitration Clause Detection API.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import logging
from pathlib import Path
import sys

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag import ArbitrationDetector, RetrievalConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Arbitration Clause Detection API",
    description="AI-powered detection of arbitration clauses in Terms of Use documents",
    version="1.0.0"
)

# Initialize detector (singleton pattern)
detector = None


def get_detector():
    """Get or initialize the arbitration detector."""
    global detector
    if detector is None:
        config = RetrievalConfig(
            chunk_size=512,
            chunk_overlap=128,
            top_k_retrieval=10,
            similarity_threshold=0.5,
            use_hybrid_search=True
        )
        detector = ArbitrationDetector(config)
        logger.info("Arbitration detector initialized")
    return detector


# Request/Response models
class DetectionRequest(BaseModel):
    """Request model for text detection."""
    text: str = Field(..., description="Terms of Use document text to analyze")
    document_id: Optional[str] = Field(None, description="Optional document identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Any disputes arising from these Terms shall be resolved through binding arbitration...",
                "document_id": "doc_123"
            }
        }


class ClauseResponse(BaseModel):
    """Response model for detected clause."""
    text: str
    confidence_score: float
    arbitration_type: str
    clause_types: List[str]
    location: Dict[str, int]
    details: Dict[str, Any]
    semantic_score: float
    keyword_score: float


class DetectionResponse(BaseModel):
    """Response model for detection results."""
    document_id: str
    has_arbitration: bool
    confidence: float
    clauses: List[ClauseResponse]
    summary: Dict[str, Any]
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_123",
                "has_arbitration": True,
                "confidence": 0.92,
                "clauses": [
                    {
                        "text": "Disputes will be resolved through binding arbitration...",
                        "confidence_score": 0.95,
                        "arbitration_type": "binding",
                        "clause_types": ["arbitration", "jury_trial_waiver"],
                        "location": {"start_char": 1000, "end_char": 1500},
                        "details": {
                            "provider": "AAA",
                            "class_action_waiver": True
                        },
                        "semantic_score": 0.88,
                        "keyword_score": 0.91
                    }
                ],
                "summary": {
                    "has_binding_arbitration": True,
                    "has_class_action_waiver": True,
                    "has_jury_trial_waiver": True,
                    "has_opt_out": False,
                    "arbitration_provider": "AAA"
                },
                "processing_time": 0.45
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    version: str


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Arbitration Clause Detection API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        detector = get_detector()
        return HealthResponse(
            status="healthy",
            model="sentence-transformers/all-MiniLM-L6-v2",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/detect", response_model=DetectionResponse)
async def detect_arbitration(request: DetectionRequest):
    """
    Detect arbitration clauses in provided text.
    
    This endpoint analyzes Terms of Use text to identify:
    - Binding arbitration clauses
    - Class action waivers
    - Jury trial waivers
    - Opt-out provisions
    - Arbitration providers and locations
    """
    try:
        detector = get_detector()
        
        # Perform detection
        result = detector.detect(
            document_text=request.text,
            document_id=request.document_id
        )
        
        # Convert to response format
        response = DetectionResponse(
            document_id=result.document_id,
            has_arbitration=result.has_arbitration,
            confidence=result.confidence,
            clauses=[
                ClauseResponse(
                    text=clause.text,
                    confidence_score=clause.confidence_score,
                    arbitration_type=clause.arbitration_type.value,
                    clause_types=[ct.value for ct in clause.clause_types],
                    location=clause.location,
                    details=clause.details,
                    semantic_score=clause.semantic_score,
                    keyword_score=clause.keyword_score
                )
                for clause in result.clauses
            ],
            summary=result.summary,
            processing_time=result.processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/file")
async def detect_arbitration_file(file: UploadFile = File(...)):
    """
    Detect arbitration clauses in uploaded document file.
    
    Accepts text files (.txt) or JSON files containing Terms of Use text.
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.txt', '.json')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .json files are supported"
            )
        
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # If JSON, extract text field
        if file.filename.endswith('.json'):
            try:
                data = json.loads(text)
                text = data.get('text', data.get('content', ''))
                if not text:
                    raise ValueError("JSON must contain 'text' or 'content' field")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON file")
        
        # Get detector
        detector = get_detector()
        
        # Perform detection
        result = detector.detect(
            document_text=text,
            document_id=file.filename
        )
        
        # Convert to response format
        response = DetectionResponse(
            document_id=result.document_id,
            has_arbitration=result.has_arbitration,
            confidence=result.confidence,
            clauses=[
                ClauseResponse(
                    text=clause.text,
                    confidence_score=clause.confidence_score,
                    arbitration_type=clause.arbitration_type.value,
                    clause_types=[ct.value for ct in clause.clause_types],
                    location=clause.location,
                    details=clause.details,
                    semantic_score=clause.semantic_score,
                    keyword_score=clause.keyword_score
                )
                for clause in result.clauses
            ],
            summary=result.summary,
            processing_time=result.processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/patterns")
async def analyze_patterns(request: DetectionRequest):
    """
    Analyze text for specific arbitration patterns and indicators.
    
    Returns detailed pattern matching results without full detection.
    """
    try:
        from app.rag import ArbitrationPatterns
        import re
        
        patterns = ArbitrationPatterns()
        text_lower = request.text.lower()
        
        # Extract details
        details = ArbitrationPatterns.extract_arbitration_details(request.text)
        
        # Match patterns
        matched_patterns = []
        pattern_scores = {}
        
        for pattern in ArbitrationPatterns.get_all_patterns():
            matched = False
            
            if pattern.pattern_type == 'regex':
                if re.search(pattern.pattern, text_lower):
                    matched = True
            else:
                if pattern.pattern in text_lower:
                    matched = True
            
            if matched:
                matched_patterns.append({
                    "pattern": pattern.pattern,
                    "weight": pattern.weight,
                    "type": pattern.arbitration_type
                })
                
                if pattern.arbitration_type not in pattern_scores:
                    pattern_scores[pattern.arbitration_type] = 0
                pattern_scores[pattern.arbitration_type] += pattern.weight
        
        # Check negative indicators
        negative_matches = []
        for negative_pattern, weight in ArbitrationPatterns.get_negative_indicators():
            if negative_pattern in text_lower:
                negative_matches.append({
                    "pattern": negative_pattern,
                    "weight": weight
                })
        
        return JSONResponse(content={
            "matched_patterns": matched_patterns,
            "negative_indicators": negative_matches,
            "pattern_scores": pattern_scores,
            "extracted_details": details,
            "total_patterns_matched": len(matched_patterns),
            "likelihood": "high" if len(matched_patterns) > 5 else "medium" if len(matched_patterns) > 2 else "low"
        })
        
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )