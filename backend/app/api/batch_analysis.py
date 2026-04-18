"""
Batch Analysis API Endpoints

Provides endpoints for:
- Batch document analysis
- Analysis history and audit trail
- Bulk operations status
"""

from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, BackgroundTasks, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.batch_analysis_service import get_batch_analysis_service
from app.services.audit_service import get_audit_service, AuditAction
from app.schemas.analysis import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AnalysisStatistics,
)
from app.schemas.audit import (
    AuditLogQuery,
    AuditLogResponse,
    AnalysisHistoryQuery,
    AnalysisHistoryResponse,
)
from app.core.exceptions import ValidationError

router = APIRouter(prefix="/analysis", tags=["batch-analysis"])


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    summary="Batch analyze multiple documents",
    description="""
    Analyze multiple documents or text snippets for arbitration clauses in a single request.

    Supports:
    - Up to 50 items per batch
    - Mix of document IDs and raw text
    - Parallel or sequential processing
    - Per-item reference IDs for tracking

    Returns individual results for each item, including partial failures.
    """,
    responses={
        200: {
            "description": "Batch analysis completed",
            "content": {
                "application/json": {
                    "example": {
                        "total": 3,
                        "successful": 2,
                        "failed": 1,
                        "results": [
                            {
                                "reference_id": "contract-001",
                                "document_id": 1,
                                "success": True,
                                "result": {
                                    "has_arbitration_clause": True,
                                    "confidence_score": 0.87
                                }
                            }
                        ],
                        "total_processing_time_ms": 1250
                    }
                }
            }
        },
        422: {"description": "Validation error"}
    }
)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Perform batch analysis on multiple documents or texts.

    - **items**: List of documents (by ID) or texts to analyze
    - **min_confidence_threshold**: Minimum confidence score (0.0-1.0)
    - **include_context**: Include surrounding context for clauses
    - **parallel**: Process items in parallel for faster results
    """
    batch_service = get_batch_analysis_service()
    audit_service = get_audit_service()

    # Get user info from request state
    user_data = getattr(http_request.state, 'current_user', None)
    user_id = user_data.get('user_id') if user_data else None
    request_id = getattr(http_request.state, 'request_id', None)

    # Perform batch analysis
    result = await batch_service.analyze_batch(
        db=db,
        request=request,
        user_id=user_id,
        request_id=request_id
    )

    # Log audit event in background
    background_tasks.add_task(
        audit_service.log_action,
        action=AuditAction.BATCH_ANALYSIS,
        user_id=user_id,
        resource_type="batch_analysis",
        details={
            "total_items": result.total,
            "successful": result.successful,
            "failed": result.failed,
            "processing_time_ms": result.total_processing_time_ms
        },
        request_id=request_id
    )

    return result


@router.get(
    "/history",
    response_model=AnalysisHistoryResponse,
    summary="Get analysis history",
    description="""
    Retrieve historical analysis records with filtering and pagination.

    Useful for:
    - Reviewing past analyses
    - Tracking analysis patterns
    - Compliance auditing
    """
)
async def get_analysis_history(
    document_id: Optional[int] = Query(None, description="Filter by document ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    has_arbitration: Optional[bool] = Query(None, description="Filter by detection result"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    start_date: Optional[datetime] = Query(None, description="From date"),
    end_date: Optional[datetime] = Query(None, description="To date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    db: Session = Depends(get_db)
):
    """
    Get paginated analysis history with optional filters.
    """
    audit_service = get_audit_service()

    query = AnalysisHistoryQuery(
        document_id=document_id,
        user_id=user_id,
        has_arbitration=has_arbitration,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size
    )

    return await audit_service.get_analysis_history(db, query)


@router.get(
    "/audit-log",
    response_model=AuditLogResponse,
    summary="Get audit log",
    description="""
    Retrieve audit log entries for analysis operations.

    Tracks all analysis-related actions including:
    - Document uploads
    - Analysis requests
    - Batch operations
    - Failures and errors
    """
)
async def get_audit_log(
    action: Optional[str] = Query(None, description="Filter by action type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    start_date: Optional[datetime] = Query(None, description="From date"),
    end_date: Optional[datetime] = Query(None, description="To date"),
    success_only: Optional[bool] = Query(None, description="Filter by success status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Get paginated audit log entries.
    """
    audit_service = get_audit_service()

    # Convert action string to enum if provided
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action)
        except ValueError:
            pass

    query = AuditLogQuery(
        action=action_enum,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        start_date=start_date,
        end_date=end_date,
        success_only=success_only,
        page=page,
        page_size=page_size
    )

    return await audit_service.get_audit_log(db, query)


@router.get(
    "/statistics",
    response_model=AnalysisStatistics,
    summary="Get analysis statistics",
    description="Get comprehensive statistics about analyses in the system."
)
async def get_analysis_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    db: Session = Depends(get_db)
):
    """
    Get analysis statistics for the specified period.
    """
    from app.services.analysis_service import AnalysisService

    service = AnalysisService()
    stats = service.get_analysis_statistics(db)

    # Get additional breakdown
    from app.models.analysis import ArbitrationAnalysis
    from sqlalchemy import func

    start_date = datetime.utcnow() - timedelta(days=days)

    # Daily counts
    daily_counts = db.query(
        func.date(ArbitrationAnalysis.analyzed_at).label('date'),
        func.count(ArbitrationAnalysis.id).label('count')
    ).filter(
        ArbitrationAnalysis.analyzed_at >= start_date
    ).group_by(
        func.date(ArbitrationAnalysis.analyzed_at)
    ).all()

    stats['analyses_by_day'] = {
        str(row.date): row.count for row in daily_counts
    }

    return AnalysisStatistics(**stats)
