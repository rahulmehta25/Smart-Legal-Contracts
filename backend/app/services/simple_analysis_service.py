"""
Simplified analysis service for basic API testing
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
from loguru import logger

from app.models.simple_analysis import Analysis
from app.schemas.analysis import AnalysisRequest, QuickAnalysisRequest


class AnalysisService:
    """Simplified analysis service for testing"""
    
    def analyze_document(self, db: Session, request: AnalysisRequest) -> Analysis:
        """Analyze a document for arbitration clauses"""
        try:
            # Create mock analysis
            analysis = Analysis(
                document_id=request.document_id,
                user_id=1,  # Mock user ID
                analysis_type=request.analysis_type,
                status="completed",
                overall_score=0.85,
                risk_level="medium",
                summary="Mock analysis completed successfully",
                findings=[{
                    "clause_type": "arbitration",
                    "confidence": 0.85,
                    "text": "Mock arbitration clause found"
                }],
                processing_time_ms=1000,
                completed_at=datetime.utcnow()
            )
            
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            raise
    
    def quick_analyze_text(self, db: Session, request: QuickAnalysisRequest) -> Dict[str, Any]:
        """Quick analysis of text without storing in database"""
        try:
            # Mock quick analysis
            has_arbitration = "arbitration" in request.text.lower()
            
            return {
                "has_arbitration_clause": has_arbitration,
                "confidence_score": 0.8 if has_arbitration else 0.2,
                "clauses_found": [
                    {
                        "text": "Mock arbitration clause",
                        "confidence": 0.8
                    }
                ] if has_arbitration else [],
                "summary": f"Analysis of {len(request.text)} characters completed",
                "processing_time_ms": 500
            }
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            raise
    
    def get_analysis(self, db: Session, analysis_id: int) -> Optional[Analysis]:
        """Get analysis by ID"""
        try:
            return db.query(Analysis).filter(Analysis.id == analysis_id).first()
        except Exception as e:
            logger.error(f"Error getting analysis: {e}")
            return None
    
    def get_analyses(self, db: Session, skip: int = 0, limit: int = 100, 
                    has_arbitration_only: bool = False) -> List[Analysis]:
        """Get list of analyses"""
        try:
            query = db.query(Analysis)
            if has_arbitration_only:
                query = query.filter(Analysis.overall_score > 0.5)
            return query.offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting analyses: {e}")
            return []
    
    def get_document_analyses(self, db: Session, document_id: int) -> List[Analysis]:
        """Get all analyses for a document"""
        try:
            return db.query(Analysis).filter(Analysis.document_id == document_id).all()
        except Exception as e:
            logger.error(f"Error getting document analyses: {e}")
            return []
    
    def get_latest_analysis(self, db: Session, document_id: int) -> Optional[Analysis]:
        """Get latest analysis for a document"""
        try:
            return db.query(Analysis).filter(
                Analysis.document_id == document_id
            ).order_by(Analysis.created_at.desc()).first()
        except Exception as e:
            logger.error(f"Error getting latest analysis: {e}")
            return None
    
    def delete_analysis(self, db: Session, analysis_id: int) -> bool:
        """Delete an analysis"""
        try:
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if analysis:
                db.delete(analysis)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting analysis: {e}")
            return False
    
    def get_analysis_statistics(self, db: Session) -> Dict[str, Any]:
        """Get analysis statistics"""
        try:
            total_analyses = db.query(Analysis).count()
            return {
                "total_analyses": total_analyses,
                "completed_analyses": total_analyses,
                "average_confidence": 0.75,
                "high_risk_count": 0
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_analyses": 0, "completed_analyses": 0}
    
    def search_analyses_by_clause_type(self, db: Session, clause_type: str, limit: int = 50) -> List[Analysis]:
        """Search analyses by clause type"""
        return []
    
    def get_clause_types_summary(self, db: Session) -> Dict[str, int]:
        """Get summary of clause types"""
        return {"arbitration": 10, "mediation": 5}