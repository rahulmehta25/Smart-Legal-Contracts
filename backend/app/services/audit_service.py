"""
Audit Service

Provides comprehensive audit trail functionality:
- Action logging
- Analysis history tracking
- Compliance reporting
- Activity analytics
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, desc, func

from app.db.database import Base
from app.schemas.audit import (
    AuditLogEntry,
    AuditLogQuery,
    AuditLogResponse,
    AnalysisHistoryEntry,
    AnalysisHistoryQuery,
    AnalysisHistoryResponse,
    AuditAction,
)


class AuditLog(Base):
    """SQLAlchemy model for audit log entries"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    action = Column(String(50), index=True, nullable=False)
    user_id = Column(String(100), index=True, nullable=True)
    user_email = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(100), index=True, nullable=True)
    resource_type = Column(String(50), index=True, nullable=False)
    resource_id = Column(String(100), index=True, nullable=True)
    resource_name = Column(String(255), nullable=True)
    details = Column(JSON, nullable=True)
    changes = Column(JSON, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)


class AnalysisHistory(Base):
    """SQLAlchemy model for analysis history"""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True, nullable=False)
    document_name = Column(String(255), nullable=True)
    user_id = Column(String(100), index=True, nullable=True)
    user_email = Column(String(255), nullable=True)
    has_arbitration_clause = Column(Boolean, nullable=False)
    confidence_score = Column(Integer, nullable=False)  # Stored as int (score * 100)
    clauses_count = Column(Integer, default=0)
    analysis_summary = Column(Text, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    analysis_version = Column(String(20), default="1.0")
    processing_time_ms = Column(Integer, nullable=True)
    request_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    source = Column(String(50), default="api")


class AuditService:
    """
    Service for audit trail and analysis history
    """

    def __init__(self, db_session_factory=None):
        self.db_session_factory = db_session_factory

    async def log_action(
        self,
        action: AuditAction,
        resource_type: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        changes: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Optional[int]:
        """
        Log an auditable action

        Args:
            action: Type of action performed
            resource_type: Type of resource affected
            user_id: ID of user performing action
            user_email: Email of user
            resource_id: ID of affected resource
            resource_name: Name of affected resource
            details: Additional action details
            changes: Before/after values
            success: Whether action succeeded
            error_message: Error message if failed
            duration_ms: Action duration
            ip_address: Client IP
            user_agent: Client user agent
            request_id: Request ID for tracing
            db: Database session

        Returns:
            Audit log entry ID if successful
        """
        try:
            if db is None and self.db_session_factory:
                db = self.db_session_factory()
                should_close = True
            else:
                should_close = False

            if db is None:
                logger.warning("No database session available for audit logging")
                return None

            log_entry = AuditLog(
                timestamp=datetime.utcnow(),
                action=action.value,
                user_id=user_id,
                user_email=user_email,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id else None,
                resource_name=resource_name,
                details=details,
                changes=changes,
                success=success,
                error_message=error_message,
                duration_ms=duration_ms
            )

            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)

            if should_close:
                db.close()

            logger.debug(
                f"Audit logged: {action.value} on {resource_type}/{resource_id} "
                f"by {user_id or 'anonymous'}"
            )

            return log_entry.id

        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
            return None

    async def log_analysis(
        self,
        db: Session,
        document_id: int,
        document_name: Optional[str],
        has_arbitration_clause: bool,
        confidence_score: float,
        clauses_count: int,
        analysis_summary: Optional[str],
        processing_time_ms: Optional[int],
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        source: str = "api"
    ) -> Optional[int]:
        """Log an analysis to history"""
        try:
            history_entry = AnalysisHistory(
                document_id=document_id,
                document_name=document_name,
                user_id=user_id,
                user_email=user_email,
                has_arbitration_clause=has_arbitration_clause,
                confidence_score=int(confidence_score * 100),
                clauses_count=clauses_count,
                analysis_summary=analysis_summary,
                analyzed_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
                request_id=request_id,
                ip_address=ip_address,
                source=source
            )

            db.add(history_entry)
            db.commit()
            db.refresh(history_entry)

            return history_entry.id

        except Exception as e:
            logger.error(f"Failed to log analysis history: {e}")
            db.rollback()
            return None

    async def get_audit_log(
        self,
        db: Session,
        query: AuditLogQuery
    ) -> AuditLogResponse:
        """Get paginated audit log entries"""
        base_query = db.query(AuditLog)

        # Apply filters
        if query.action:
            base_query = base_query.filter(AuditLog.action == query.action.value)
        if query.user_id:
            base_query = base_query.filter(AuditLog.user_id == query.user_id)
        if query.resource_type:
            base_query = base_query.filter(AuditLog.resource_type == query.resource_type)
        if query.resource_id:
            base_query = base_query.filter(AuditLog.resource_id == query.resource_id)
        if query.start_date:
            base_query = base_query.filter(AuditLog.timestamp >= query.start_date)
        if query.end_date:
            base_query = base_query.filter(AuditLog.timestamp <= query.end_date)
        if query.success_only is not None:
            base_query = base_query.filter(AuditLog.success == query.success_only)

        # Get total count
        total = base_query.count()

        # Apply pagination
        offset = (query.page - 1) * query.page_size
        entries = base_query.order_by(desc(AuditLog.timestamp)).offset(offset).limit(query.page_size).all()

        # Convert to response models
        entry_models = [
            AuditLogEntry(
                id=e.id,
                timestamp=e.timestamp,
                action=AuditAction(e.action),
                user_id=e.user_id,
                user_email=e.user_email,
                ip_address=e.ip_address,
                user_agent=e.user_agent,
                request_id=e.request_id,
                resource_type=e.resource_type,
                resource_id=e.resource_id,
                resource_name=e.resource_name,
                details=e.details,
                changes=e.changes,
                success=e.success,
                error_message=e.error_message,
                duration_ms=e.duration_ms
            )
            for e in entries
        ]

        total_pages = (total + query.page_size - 1) // query.page_size

        return AuditLogResponse(
            entries=entry_models,
            total=total,
            page=query.page,
            page_size=query.page_size,
            total_pages=total_pages,
            has_next=query.page < total_pages,
            has_previous=query.page > 1
        )

    async def get_analysis_history(
        self,
        db: Session,
        query: AnalysisHistoryQuery
    ) -> AnalysisHistoryResponse:
        """Get paginated analysis history"""
        base_query = db.query(AnalysisHistory)

        # Apply filters
        if query.document_id:
            base_query = base_query.filter(AnalysisHistory.document_id == query.document_id)
        if query.user_id:
            base_query = base_query.filter(AnalysisHistory.user_id == query.user_id)
        if query.has_arbitration is not None:
            base_query = base_query.filter(AnalysisHistory.has_arbitration_clause == query.has_arbitration)
        if query.min_confidence:
            min_score = int(query.min_confidence * 100)
            base_query = base_query.filter(AnalysisHistory.confidence_score >= min_score)
        if query.start_date:
            base_query = base_query.filter(AnalysisHistory.analyzed_at >= query.start_date)
        if query.end_date:
            base_query = base_query.filter(AnalysisHistory.analyzed_at <= query.end_date)

        # Get total count
        total = base_query.count()

        # Get aggregations
        total_with_arb = base_query.filter(AnalysisHistory.has_arbitration_clause == True).count()
        avg_confidence = db.query(func.avg(AnalysisHistory.confidence_score)).filter(
            *[f for f in base_query.whereclause.clauses] if base_query.whereclause is not None else []
        ).scalar() or 0

        # Apply pagination
        offset = (query.page - 1) * query.page_size
        entries = base_query.order_by(desc(AnalysisHistory.analyzed_at)).offset(offset).limit(query.page_size).all()

        # Convert to response models
        entry_models = [
            AnalysisHistoryEntry(
                id=e.id,
                document_id=e.document_id,
                document_name=e.document_name,
                user_id=e.user_id,
                user_email=e.user_email,
                has_arbitration_clause=e.has_arbitration_clause,
                confidence_score=e.confidence_score / 100.0,
                clauses_count=e.clauses_count,
                analysis_summary=e.analysis_summary,
                analyzed_at=e.analyzed_at,
                analysis_version=e.analysis_version,
                processing_time_ms=e.processing_time_ms,
                request_id=e.request_id,
                ip_address=e.ip_address,
                source=e.source
            )
            for e in entries
        ]

        total_pages = (total + query.page_size - 1) // query.page_size

        return AnalysisHistoryResponse(
            entries=entry_models,
            total=total,
            page=query.page,
            page_size=query.page_size,
            total_pages=total_pages,
            has_next=query.page < total_pages,
            has_previous=query.page > 1,
            total_with_arbitration=total_with_arb,
            average_confidence=avg_confidence / 100.0 if avg_confidence else 0.0
        )


# Singleton
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get audit service singleton"""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
