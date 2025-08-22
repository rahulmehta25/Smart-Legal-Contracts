from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.analysis_service import AnalysisService
from app.models.analysis import (
    ArbitrationAnalysisResponse, AnalysisRequest, QuickAnalysisRequest,
    QuickAnalysisResponse, ArbitrationClauseResponse
)

router = APIRouter(prefix="/analysis", tags=["analysis"])
analysis_service = AnalysisService()


@router.post("/analyze", response_model=ArbitrationAnalysisResponse)
async def analyze_document(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze a document for arbitration clauses
    """
    try:
        analysis = analysis_service.analyze_document(db, request)
        
        # Convert to response model
        clauses = [ArbitrationClauseResponse.from_orm(clause) for clause in analysis.clauses]
        
        response = ArbitrationAnalysisResponse(
            id=analysis.id,
            document_id=analysis.document_id,
            has_arbitration_clause=analysis.has_arbitration_clause,
            confidence_score=analysis.confidence_score,
            analysis_summary=analysis.analysis_summary,
            analyzed_at=analysis.analyzed_at,
            analysis_version=analysis.analysis_version,
            processing_time_ms=analysis.processing_time_ms,
            clauses=clauses
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")


@router.post("/quick-analyze", response_model=QuickAnalysisResponse)
async def quick_analyze_text(
    request: QuickAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Quick analysis of text without storing in database
    """
    try:
        result = analysis_service.quick_analyze_text(db, request)
        
        return QuickAnalysisResponse(
            has_arbitration_clause=result["has_arbitration_clause"],
            confidence_score=result["confidence_score"],
            clauses_found=result["clauses_found"],
            summary=result["summary"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in quick analysis: {str(e)}")


@router.get("/", response_model=List[ArbitrationAnalysisResponse])
async def get_analyses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    has_arbitration_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Get list of analyses with pagination
    """
    try:
        analyses = analysis_service.get_analyses(
            db, skip=skip, limit=limit, has_arbitration_only=has_arbitration_only
        )
        
        response_list = []
        for analysis in analyses:
            clauses = [ArbitrationClauseResponse.from_orm(clause) for clause in analysis.clauses]
            
            response = ArbitrationAnalysisResponse(
                id=analysis.id,
                document_id=analysis.document_id,
                has_arbitration_clause=analysis.has_arbitration_clause,
                confidence_score=analysis.confidence_score,
                analysis_summary=analysis.analysis_summary,
                analyzed_at=analysis.analyzed_at,
                analysis_version=analysis.analysis_version,
                processing_time_ms=analysis.processing_time_ms,
                clauses=clauses
            )
            response_list.append(response)
        
        return response_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analyses: {str(e)}")


@router.get("/{analysis_id}", response_model=ArbitrationAnalysisResponse)
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific analysis by ID
    """
    try:
        analysis = analysis_service.get_analysis(db, analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        clauses = [ArbitrationClauseResponse.from_orm(clause) for clause in analysis.clauses]
        
        return ArbitrationAnalysisResponse(
            id=analysis.id,
            document_id=analysis.document_id,
            has_arbitration_clause=analysis.has_arbitration_clause,
            confidence_score=analysis.confidence_score,
            analysis_summary=analysis.analysis_summary,
            analyzed_at=analysis.analyzed_at,
            analysis_version=analysis.analysis_version,
            processing_time_ms=analysis.processing_time_ms,
            clauses=clauses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis: {str(e)}")


@router.get("/document/{document_id}", response_model=List[ArbitrationAnalysisResponse])
async def get_document_analyses(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all analyses for a specific document
    """
    try:
        analyses = analysis_service.get_document_analyses(db, document_id)
        
        response_list = []
        for analysis in analyses:
            clauses = [ArbitrationClauseResponse.from_orm(clause) for clause in analysis.clauses]
            
            response = ArbitrationAnalysisResponse(
                id=analysis.id,
                document_id=analysis.document_id,
                has_arbitration_clause=analysis.has_arbitration_clause,
                confidence_score=analysis.confidence_score,
                analysis_summary=analysis.analysis_summary,
                analyzed_at=analysis.analyzed_at,
                analysis_version=analysis.analysis_version,
                processing_time_ms=analysis.processing_time_ms,
                clauses=clauses
            )
            response_list.append(response)
        
        return response_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document analyses: {str(e)}")


@router.get("/document/{document_id}/latest", response_model=ArbitrationAnalysisResponse)
async def get_latest_document_analysis(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the latest analysis for a specific document
    """
    try:
        analysis = analysis_service.get_latest_analysis(db, document_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="No analysis found for this document")
        
        clauses = [ArbitrationClauseResponse.from_orm(clause) for clause in analysis.clauses]
        
        return ArbitrationAnalysisResponse(
            id=analysis.id,
            document_id=analysis.document_id,
            has_arbitration_clause=analysis.has_arbitration_clause,
            confidence_score=analysis.confidence_score,
            analysis_summary=analysis.analysis_summary,
            analyzed_at=analysis.analyzed_at,
            analysis_version=analysis.analysis_version,
            processing_time_ms=analysis.processing_time_ms,
            clauses=clauses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving latest analysis: {str(e)}")


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete an analysis
    """
    try:
        success = analysis_service.delete_analysis(db, analysis_id)
        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {"message": f"Analysis {analysis_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting analysis: {str(e)}")


@router.get("/stats/overview")
async def get_analysis_statistics(
    db: Session = Depends(get_db)
):
    """
    Get analysis statistics and overview
    """
    try:
        stats = analysis_service.get_analysis_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@router.get("/search/clause-type/{clause_type}", response_model=List[ArbitrationAnalysisResponse])
async def search_by_clause_type(
    clause_type: str,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Search analyses by clause type
    """
    try:
        analyses = analysis_service.search_analyses_by_clause_type(db, clause_type, limit)
        
        response_list = []
        for analysis in analyses:
            clauses = [ArbitrationClauseResponse.from_orm(clause) for clause in analysis.clauses]
            
            response = ArbitrationAnalysisResponse(
                id=analysis.id,
                document_id=analysis.document_id,
                has_arbitration_clause=analysis.has_arbitration_clause,
                confidence_score=analysis.confidence_score,
                analysis_summary=analysis.analysis_summary,
                analyzed_at=analysis.analyzed_at,
                analysis_version=analysis.analysis_version,
                processing_time_ms=analysis.processing_time_ms,
                clauses=clauses
            )
            response_list.append(response)
        
        return response_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching by clause type: {str(e)}")


@router.get("/stats/clause-types")
async def get_clause_types_summary(
    db: Session = Depends(get_db)
):
    """
    Get summary of clause types found across all analyses
    """
    try:
        summary = analysis_service.get_clause_types_summary(db)
        return {
            "clause_types": summary,
            "total_types": len(summary),
            "total_clauses": sum(summary.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clause types summary: {str(e)}")