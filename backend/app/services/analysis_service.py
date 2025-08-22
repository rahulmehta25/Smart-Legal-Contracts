from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from loguru import logger

from app.models.analysis import (
    ArbitrationAnalysis, ArbitrationClause, 
    ArbitrationAnalysisResponse, AnalysisRequest, QuickAnalysisRequest
)
from app.models.document import Document
from app.rag.pipeline import RAGPipeline, AnalysisResult
from app.services.document_service import DocumentService


class AnalysisService:
    """
    Service for analyzing documents for arbitration clauses
    """
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.document_service = DocumentService()
    
    def analyze_document(self, 
                        db: Session, 
                        request: AnalysisRequest) -> ArbitrationAnalysis:
        """
        Analyze a document for arbitration clauses
        
        Args:
            db: Database session
            request: Analysis request
            
        Returns:
            ArbitrationAnalysis result
        """
        try:
            document_id = request.document_id
            
            # Check if document exists and is processed
            document = self.document_service.get_document(db, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            if not document.is_processed:
                raise ValueError(f"Document {document_id} has not been processed yet")
            
            # Check if analysis already exists and force_reanalysis is False
            existing_analysis = self.get_latest_analysis(db, document_id)
            if existing_analysis and not request.force_reanalysis:
                logger.info(f"Returning existing analysis for document {document_id}")
                return existing_analysis
            
            # Perform analysis using RAG pipeline
            analysis_result = self.rag_pipeline.analyze_document_for_arbitration(document_id)
            
            # Store analysis in database
            db_analysis = self._create_analysis_record(
                db, document_id, analysis_result
            )
            
            logger.info(f"Analysis completed for document {document_id}: "
                       f"{'HAS' if db_analysis.has_arbitration_clause else 'NO'} arbitration clause")
            
            return db_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document {document_id}: {e}")
            raise
    
    def quick_analyze_text(self, 
                          db: Session, 
                          request: QuickAnalysisRequest) -> Dict[str, Any]:
        """
        Perform quick analysis on raw text without storing
        
        Args:
            db: Database session
            request: Quick analysis request
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Perform quick analysis
            analysis_result = self.rag_pipeline.quick_text_analysis(request.text)
            
            # Format response
            response = {
                "has_arbitration_clause": analysis_result.has_arbitration_clause,
                "confidence_score": analysis_result.confidence_score,
                "summary": analysis_result.summary,
                "clauses_found": analysis_result.clauses,
                "processing_time_ms": analysis_result.processing_time_ms,
                "metadata": analysis_result.metadata
            }
            
            if request.include_context:
                response["analysis_details"] = {
                    "confidence_level": analysis_result.metadata.get("confidence_level"),
                    "total_chunks_analyzed": analysis_result.metadata.get("total_chunks", 0),
                    "arbitration_chunks_found": analysis_result.metadata.get("arbitration_chunks", 0)
                }
            
            logger.info(f"Quick analysis completed: "
                       f"{'HAS' if analysis_result.has_arbitration_clause else 'NO'} arbitration clause "
                       f"(confidence: {analysis_result.confidence_score:.2f})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in quick text analysis: {e}")
            raise
    
    def get_analysis(self, 
                    db: Session, 
                    analysis_id: int) -> Optional[ArbitrationAnalysis]:
        """
        Get analysis by ID
        """
        return db.query(ArbitrationAnalysis).filter(
            ArbitrationAnalysis.id == analysis_id
        ).first()
    
    def get_latest_analysis(self, 
                           db: Session, 
                           document_id: int) -> Optional[ArbitrationAnalysis]:
        """
        Get the latest analysis for a document
        """
        return db.query(ArbitrationAnalysis).filter(
            ArbitrationAnalysis.document_id == document_id
        ).order_by(desc(ArbitrationAnalysis.analyzed_at)).first()
    
    def get_document_analyses(self, 
                             db: Session, 
                             document_id: int) -> List[ArbitrationAnalysis]:
        """
        Get all analyses for a document
        """
        return db.query(ArbitrationAnalysis).filter(
            ArbitrationAnalysis.document_id == document_id
        ).order_by(desc(ArbitrationAnalysis.analyzed_at)).all()
    
    def get_analyses(self, 
                    db: Session, 
                    skip: int = 0, 
                    limit: int = 100,
                    has_arbitration_only: bool = False) -> List[ArbitrationAnalysis]:
        """
        Get list of analyses with pagination
        """
        query = db.query(ArbitrationAnalysis)
        
        if has_arbitration_only:
            query = query.filter(ArbitrationAnalysis.has_arbitration_clause == True)
        
        return query.order_by(desc(ArbitrationAnalysis.analyzed_at)).offset(skip).limit(limit).all()
    
    def delete_analysis(self, db: Session, analysis_id: int) -> bool:
        """
        Delete an analysis
        """
        try:
            analysis = self.get_analysis(db, analysis_id)
            if not analysis:
                return False
            
            db.delete(analysis)
            db.commit()
            
            logger.info(f"Deleted analysis {analysis_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting analysis {analysis_id}: {e}")
            raise
    
    def get_analysis_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics about analyses in the system
        """
        try:
            total_analyses = db.query(ArbitrationAnalysis).count()
            
            analyses_with_arbitration = db.query(ArbitrationAnalysis).filter(
                ArbitrationAnalysis.has_arbitration_clause == True
            ).count()
            
            analyses_without_arbitration = total_analyses - analyses_with_arbitration
            
            # Get confidence distribution
            high_confidence = db.query(ArbitrationAnalysis).filter(
                ArbitrationAnalysis.confidence_score >= 0.8
            ).count()
            
            medium_confidence = db.query(ArbitrationAnalysis).filter(
                ArbitrationAnalysis.confidence_score >= 0.6,
                ArbitrationAnalysis.confidence_score < 0.8
            ).count()
            
            low_confidence = db.query(ArbitrationAnalysis).filter(
                ArbitrationAnalysis.confidence_score < 0.6
            ).count()
            
            # Get average processing time
            avg_processing_time = db.query(
                db.func.avg(ArbitrationAnalysis.processing_time_ms)
            ).scalar() or 0
            
            return {
                'total_analyses': total_analyses,
                'with_arbitration_clause': analyses_with_arbitration,
                'without_arbitration_clause': analyses_without_arbitration,
                'arbitration_detection_rate': (
                    analyses_with_arbitration / total_analyses 
                    if total_analyses > 0 else 0
                ),
                'confidence_distribution': {
                    'high_confidence': high_confidence,
                    'medium_confidence': medium_confidence,
                    'low_confidence': low_confidence
                },
                'average_processing_time_ms': float(avg_processing_time)
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis statistics: {e}")
            return {'error': str(e)}
    
    def search_analyses_by_clause_type(self, 
                                      db: Session, 
                                      clause_type: str,
                                      limit: int = 50) -> List[ArbitrationAnalysis]:
        """
        Search analyses that contain specific clause types
        
        Args:
            db: Database session
            clause_type: Type of clause to search for
            limit: Maximum number of results
            
        Returns:
            List of matching analyses
        """
        try:
            # Get analyses that have clauses of the specified type
            analyses = db.query(ArbitrationAnalysis).join(ArbitrationClause).filter(
                ArbitrationClause.clause_type == clause_type
            ).order_by(desc(ArbitrationAnalysis.confidence_score)).limit(limit).all()
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error searching analyses by clause type {clause_type}: {e}")
            return []
    
    def _create_analysis_record(self, 
                               db: Session, 
                               document_id: int, 
                               analysis_result: AnalysisResult) -> ArbitrationAnalysis:
        """
        Create analysis record in database
        """
        try:
            # Create main analysis record
            db_analysis = ArbitrationAnalysis(
                document_id=document_id,
                has_arbitration_clause=analysis_result.has_arbitration_clause,
                confidence_score=analysis_result.confidence_score,
                analysis_summary=analysis_result.summary,
                analyzed_at=datetime.utcnow(),
                analysis_version="1.0",
                processing_time_ms=analysis_result.processing_time_ms,
                metadata=analysis_result.metadata
            )
            
            db.add(db_analysis)
            db.flush()  # Get the ID
            
            # Create clause records
            db_clauses = []
            for clause_data in analysis_result.clauses:
                db_clause = ArbitrationClause(
                    analysis_id=db_analysis.id,
                    clause_text=clause_data["text"],
                    clause_type=clause_data.get("type", "general_arbitration"),
                    relevance_score=clause_data.get("relevance_score", 0.0),
                    severity_score=self._calculate_severity_score(clause_data),
                    start_position=clause_data.get("start_position"),
                    end_position=clause_data.get("end_position"),
                    surrounding_context=clause_data.get("context", ""),
                    section_title=""
                )
                db_clauses.append(db_clause)
            
            if db_clauses:
                db.add_all(db_clauses)
            
            db.commit()
            db.refresh(db_analysis)
            
            return db_analysis
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating analysis record: {e}")
            raise
    
    def _calculate_severity_score(self, clause_data: Dict[str, Any]) -> float:
        """
        Calculate severity score for a clause based on its characteristics
        """
        severity = 0.5  # Base severity
        
        signals = clause_data.get("signals", {})
        
        # Increase severity for restrictive clauses
        if signals.get("binding_arbitration", False):
            severity += 0.2
        
        if signals.get("mandatory_arbitration", False):
            severity += 0.2
        
        if signals.get("class_action_waiver", False):
            severity += 0.15
        
        if signals.get("jury_waiver", False):
            severity += 0.1
        
        # Cap at 1.0
        return min(severity, 1.0)
    
    def get_clause_types_summary(self, db: Session) -> Dict[str, int]:
        """
        Get summary of clause types found across all analyses
        """
        try:
            clause_types = db.query(
                ArbitrationClause.clause_type,
                db.func.count(ArbitrationClause.id).label('count')
            ).group_by(ArbitrationClause.clause_type).all()
            
            return {clause_type: count for clause_type, count in clause_types}
            
        except Exception as e:
            logger.error(f"Error getting clause types summary: {e}")
            return {}