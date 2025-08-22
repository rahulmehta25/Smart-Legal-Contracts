"""
Filtering utilities for GraphQL queries
"""

from sqlalchemy.orm import Query
from sqlalchemy import and_, or_, func
from datetime import datetime

from ..types import DocumentFilter, DetectionFilter, AnalysisFilter


def apply_document_filter(query: Query, filter: DocumentFilter) -> Query:
    """Apply document filters to SQLAlchemy query"""
    if not filter:
        return query
    
    conditions = []
    
    if filter.filename:
        conditions.append(query.model.filename.ilike(f"%{filter.filename}%"))
    
    if filter.file_type:
        conditions.append(query.model.file_type == filter.file_type)
    
    if filter.processing_status:
        conditions.append(query.model.processing_status == filter.processing_status.value)
    
    if filter.uploaded_after:
        conditions.append(query.model.upload_date >= filter.uploaded_after)
    
    if filter.uploaded_before:
        conditions.append(query.model.upload_date <= filter.uploaded_before)
    
    if filter.content_search:
        # This would require full-text search setup
        # For now, search in filename
        conditions.append(query.model.filename.ilike(f"%{filter.content_search}%"))
    
    if conditions:
        return query.filter(and_(*conditions))
    
    return query


def apply_detection_filter(query: Query, filter: DetectionFilter) -> Query:
    """Apply detection filters to SQLAlchemy query"""
    if not filter:
        return query
    
    conditions = []
    
    if filter.detection_type:
        conditions.append(query.model.detection_type == filter.detection_type.value)
    
    if filter.confidence_score:
        if filter.confidence_score.min is not None:
            conditions.append(query.model.confidence_score >= filter.confidence_score.min)
        if filter.confidence_score.max is not None:
            conditions.append(query.model.confidence_score <= filter.confidence_score.max)
    
    if filter.detection_method:
        conditions.append(query.model.detection_method == filter.detection_method.value)
    
    if filter.is_validated is not None:
        conditions.append(query.model.is_validated == filter.is_validated)
    
    if filter.is_high_confidence is not None:
        if filter.is_high_confidence:
            conditions.append(query.model.confidence_score >= 0.8)
        else:
            conditions.append(query.model.confidence_score < 0.8)
    
    if conditions:
        return query.filter(and_(*conditions))
    
    return query


def apply_analysis_filter(query: Query, filter: AnalysisFilter) -> Query:
    """Apply analysis filters to SQLAlchemy query"""
    if not filter:
        return query
    
    conditions = []
    
    if filter.has_arbitration_clause is not None:
        conditions.append(query.model.has_arbitration_clause == filter.has_arbitration_clause)
    
    if filter.confidence_score:
        if filter.confidence_score.min is not None:
            conditions.append(query.model.confidence_score >= filter.confidence_score.min)
        if filter.confidence_score.max is not None:
            conditions.append(query.model.confidence_score <= filter.confidence_score.max)
    
    if filter.analysis_version:
        conditions.append(query.model.analysis_version == filter.analysis_version)
    
    if filter.analyzed_after:
        conditions.append(query.model.analyzed_at >= filter.analyzed_after)
    
    if filter.analyzed_before:
        conditions.append(query.model.analyzed_at <= filter.analyzed_before)
    
    if conditions:
        return query.filter(and_(*conditions))
    
    return query