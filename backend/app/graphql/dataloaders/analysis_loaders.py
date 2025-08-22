"""
DataLoaders for Analysis, Detection, and Clause entities
"""

from typing import List, Dict, Any
from aiodataloader import DataLoader
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from ..types import (
    ArbitrationAnalysis as AnalysisType,
    Detection as DetectionType,
    ArbitrationClause as ClauseType
)
from ...db.models import Detection, QueryCache
from ...models.analysis import ArbitrationAnalysis, ArbitrationClause
from ...models.document import Document


class AnalysisDataLoader(DataLoader):
    """DataLoader for ArbitrationAnalysis entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[AnalysisType]:
        """Batch load analyses by IDs"""
        try:
            analyses = self.session.query(ArbitrationAnalysis).filter(
                ArbitrationAnalysis.id.in_(keys)
            ).all()
            
            analysis_map = {
                analysis.id: self._convert_to_graphql_type(analysis) 
                for analysis in analyses
            }
            
            return [analysis_map.get(key) for key in keys]
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, analysis: ArbitrationAnalysis) -> AnalysisType:
        """Convert SQLAlchemy Analysis to GraphQL AnalysisType"""
        # Calculate risk level based on confidence and clause count
        risk_level = "LOW"
        if analysis.has_arbitration_clause and analysis.confidence_score:
            if analysis.confidence_score > 0.8:
                risk_level = "HIGH"
            elif analysis.confidence_score > 0.5:
                risk_level = "MEDIUM"
        
        return AnalysisType(
            id=str(analysis.id),
            document_id=str(analysis.document_id),
            has_arbitration_clause=analysis.has_arbitration_clause,
            confidence_score=analysis.confidence_score,
            analysis_summary=analysis.analysis_summary,
            analyzed_at=analysis.analyzed_at,
            analysis_version=analysis.analysis_version,
            processing_time_ms=analysis.processing_time_ms,
            metadata=str(analysis.metadata) if analysis.metadata else None,
            created_at=analysis.analyzed_at,  # Using analyzed_at as created_at
            updated_at=None,
            # Computed fields
            clause_count=len(analysis.clauses) if analysis.clauses else 0,
            average_clause_score=self._calculate_avg_clause_score(analysis.clauses),
            risk_level=risk_level
        )
    
    def _calculate_avg_clause_score(self, clauses) -> float:
        """Calculate average clause relevance score"""
        if not clauses:
            return None
        
        scores = [c.relevance_score for c in clauses if c.relevance_score is not None]
        if not scores:
            return None
        
        return sum(scores) / len(scores)


class AnalysesByDocumentDataLoader(DataLoader):
    """DataLoader for Analyses by Document ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, document_ids: List[int]) -> List[List[AnalysisType]]:
        """Batch load analyses by document IDs"""
        try:
            analyses = self.session.query(ArbitrationAnalysis).filter(
                ArbitrationAnalysis.document_id.in_(document_ids)
            ).order_by(ArbitrationAnalysis.analyzed_at.desc()).all()
            
            # Group analyses by document_id
            analyses_by_doc = {}
            for analysis in analyses:
                if analysis.document_id not in analyses_by_doc:
                    analyses_by_doc[analysis.document_id] = []
                analyses_by_doc[analysis.document_id].append(
                    AnalysisDataLoader(self.session)._convert_to_graphql_type(analysis)
                )
            
            return [analyses_by_doc.get(doc_id, []) for doc_id in document_ids]
            
        except Exception as e:
            return [[] for _ in document_ids]


class DetectionDataLoader(DataLoader):
    """DataLoader for Detection entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[DetectionType]:
        """Batch load detections by IDs"""
        try:
            detections = self.session.query(Detection).filter(
                Detection.id.in_(keys)
            ).all()
            
            detection_map = {
                detection.id: self._convert_to_graphql_type(detection)
                for detection in detections
            }
            
            return [detection_map.get(key) for key in keys]
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, detection: Detection) -> DetectionType:
        """Convert SQLAlchemy Detection to GraphQL DetectionType"""
        # Calculate severity based on detection type and confidence
        severity = detection.confidence_score
        if detection.detection_type in ['mandatory', 'binding_arbitration']:
            severity *= 1.2  # Increase severity for mandatory clauses
        
        return DetectionType(
            id=str(detection.id),
            chunk_id=str(detection.chunk_id),
            document_id=str(detection.document_id),
            detection_type=detection.detection_type,
            confidence_score=detection.confidence_score,
            pattern_id=str(detection.pattern_id) if detection.pattern_id else None,
            matched_text=detection.matched_text,
            context_before=detection.context_before,
            context_after=detection.context_after,
            start_position=detection.start_position,
            end_position=detection.end_position,
            page_number=detection.page_number,
            detection_method=detection.detection_method,
            model_version=detection.model_version,
            is_validated=detection.is_validated,
            validation_score=detection.validation_score,
            notes=detection.notes,
            created_at=detection.created_at,
            updated_at=None,
            # Computed fields
            is_high_confidence=detection.is_high_confidence,
            context_snippet=detection.context_snippet,
            severity=min(severity, 1.0)  # Cap at 1.0
        )


class DetectionsByDocumentDataLoader(DataLoader):
    """DataLoader for Detections by Document ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, document_ids: List[int]) -> List[List[DetectionType]]:
        """Batch load detections by document IDs"""
        try:
            detections = self.session.query(Detection).filter(
                Detection.document_id.in_(document_ids)
            ).order_by(Detection.confidence_score.desc()).all()
            
            # Group detections by document_id
            detections_by_doc = {}
            for detection in detections:
                if detection.document_id not in detections_by_doc:
                    detections_by_doc[detection.document_id] = []
                detections_by_doc[detection.document_id].append(
                    DetectionDataLoader(self.session)._convert_to_graphql_type(detection)
                )
            
            return [detections_by_doc.get(doc_id, []) for doc_id in document_ids]
            
        except Exception as e:
            return [[] for _ in document_ids]


class DetectionsByChunkDataLoader(DataLoader):
    """DataLoader for Detections by Chunk ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, chunk_ids: List[int]) -> List[List[DetectionType]]:
        """Batch load detections by chunk IDs"""
        try:
            detections = self.session.query(Detection).filter(
                Detection.chunk_id.in_(chunk_ids)
            ).order_by(Detection.confidence_score.desc()).all()
            
            # Group detections by chunk_id
            detections_by_chunk = {}
            for detection in detections:
                if detection.chunk_id not in detections_by_chunk:
                    detections_by_chunk[detection.chunk_id] = []
                detections_by_chunk[detection.chunk_id].append(
                    DetectionDataLoader(self.session)._convert_to_graphql_type(detection)
                )
            
            return [detections_by_chunk.get(chunk_id, []) for chunk_id in chunk_ids]
            
        except Exception as e:
            return [[] for _ in chunk_ids]


class ClauseDataLoader(DataLoader):
    """DataLoader for ArbitrationClause entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[ClauseType]:
        """Batch load clauses by IDs"""
        try:
            clauses = self.session.query(ArbitrationClause).filter(
                ArbitrationClause.id.in_(keys)
            ).all()
            
            clause_map = {
                clause.id: self._convert_to_graphql_type(clause)
                for clause in clauses
            }
            
            return [clause_map.get(key) for key in keys]
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, clause: ArbitrationClause) -> ClauseType:
        """Convert SQLAlchemy Clause to GraphQL ClauseType"""
        # Determine risk level based on clause type and scores
        risk_level = "LOW"
        if clause.clause_type in ['mandatory', 'binding_arbitration']:
            risk_level = "HIGH"
        elif clause.clause_type in ['class_action_waiver']:
            risk_level = "MEDIUM"
        
        # Check if clause is binding
        is_binding = clause.clause_type in ['mandatory', 'binding_arbitration']
        
        return ClauseType(
            id=str(clause.id),
            analysis_id=str(clause.analysis_id),
            clause_text=clause.clause_text,
            clause_type=clause.clause_type,
            start_position=clause.start_position,
            end_position=clause.end_position,
            relevance_score=clause.relevance_score,
            severity_score=clause.severity_score,
            surrounding_context=clause.surrounding_context,
            section_title=clause.section_title,
            # Computed fields
            risk_level=risk_level,
            is_binding=is_binding
        )


class ClausesByAnalysisDataLoader(DataLoader):
    """DataLoader for Clauses by Analysis ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, analysis_ids: List[int]) -> List[List[ClauseType]]:
        """Batch load clauses by analysis IDs"""
        try:
            clauses = self.session.query(ArbitrationClause).filter(
                ArbitrationClause.analysis_id.in_(analysis_ids)
            ).order_by(ArbitrationClause.relevance_score.desc()).all()
            
            # Group clauses by analysis_id
            clauses_by_analysis = {}
            for clause in clauses:
                if clause.analysis_id not in clauses_by_analysis:
                    clauses_by_analysis[clause.analysis_id] = []
                clauses_by_analysis[clause.analysis_id].append(
                    ClauseDataLoader(self.session)._convert_to_graphql_type(clause)
                )
            
            return [clauses_by_analysis.get(analysis_id, []) for analysis_id in analysis_ids]
            
        except Exception as e:
            return [[] for _ in analysis_ids]


class DetectionStatsDataLoader(DataLoader):
    """DataLoader for Detection statistics"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Batch load detection statistics"""
        try:
            total_detections = self.session.query(Detection).count()
            high_confidence_detections = self.session.query(Detection).filter(
                Detection.confidence_score >= 0.8
            ).count()
            
            # Calculate average confidence score
            avg_confidence = self.session.query(func.avg(Detection.confidence_score)).scalar()
            
            # Get detection counts by type
            detections_by_type = self.session.query(
                Detection.detection_type,
                func.count(Detection.id).label('count'),
                func.avg(Detection.confidence_score).label('avg_confidence')
            ).group_by(Detection.detection_type).all()
            
            # Get detection counts by method
            detections_by_method = self.session.query(
                Detection.detection_method,
                func.count(Detection.id).label('count'),
                func.avg(Detection.confidence_score).label('avg_confidence')
            ).group_by(Detection.detection_method).all()
            
            stats = {
                'total_detections': total_detections,
                'high_confidence_detections': high_confidence_detections,
                'average_confidence_score': float(avg_confidence) if avg_confidence else 0.0,
                'detections_by_type': [
                    {
                        'type': row.detection_type,
                        'count': row.count,
                        'average_confidence': float(row.avg_confidence)
                    }
                    for row in detections_by_type
                ],
                'detections_by_method': [
                    {
                        'method': row.detection_method,
                        'count': row.count,
                        'average_confidence': float(row.avg_confidence)
                    }
                    for row in detections_by_method
                ]
            }
            
            return [stats for _ in keys]
            
        except Exception as e:
            return [{} for _ in keys]


def create_analysis_loaders(session: Session) -> Dict[str, DataLoader]:
    """Create all analysis-related DataLoaders"""
    return {
        'analysis': AnalysisDataLoader(session),
        'analyses_by_document': AnalysesByDocumentDataLoader(session),
        'detection': DetectionDataLoader(session),
        'detections_by_document': DetectionsByDocumentDataLoader(session),
        'detections_by_chunk': DetectionsByChunkDataLoader(session),
        'clause': ClauseDataLoader(session),
        'clauses_by_analysis': ClausesByAnalysisDataLoader(session),
        'detection_stats': DetectionStatsDataLoader(session),
    }