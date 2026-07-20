"""
Enhanced analysis service that uses the improved arbitration detection logic.
This service includes document validation to prevent false positives.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from loguru import logger
import traceback
import time
from dataclasses import dataclass

from app.models.analysis import (
    ArbitrationAnalysis, ArbitrationClause, 
    ArbitrationAnalysisResponse, AnalysisRequest, QuickAnalysisRequest
)
from app.models.document import Document
from app.services.document_service import DocumentService

# Import our enhanced arbitration detector
try:
    from app.rag.arbitration_detector import ArbitrationDetector, DetectionResult
    from app.rag.document_validator import DocumentValidator, DocumentType
    ENHANCED_DETECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced detection not available: {e}")
    ENHANCED_DETECTION_AVAILABLE = False
    # Fallback to original pipeline
    from app.rag.pipeline import RAGPipeline, AnalysisResult


@dataclass
class EnhancedAnalysisResult:
    """Enhanced analysis result with validation info."""
    has_arbitration_clause: bool
    confidence_score: float
    summary: str
    clauses: List[Dict[str, Any]]
    processing_time_ms: int
    metadata: Dict[str, Any]
    validation_result: Optional[Dict[str, Any]] = None


class EnhancedAnalysisService:
    """
    Enhanced service for analyzing documents with improved false positive prevention.
    """
    
    def __init__(self):
        self.document_service = DocumentService()
        
        if ENHANCED_DETECTION_AVAILABLE:
            self.arbitration_detector = ArbitrationDetector()
            self.document_validator = DocumentValidator()
            logger.info("Enhanced arbitration detection initialized")
        else:
            # Fallback to original pipeline
            self.rag_pipeline = RAGPipeline()
            logger.info("Fallback to original RAG pipeline")
    
    def analyze_document(self, 
                        db: Session, 
                        request: AnalysisRequest) -> ArbitrationAnalysis:
        """
        Analyze a document for arbitration clauses with enhanced validation.
        
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
            
            # Get document text
            document_text = document.content or ""
            
            # Perform enhanced analysis if available
            if ENHANCED_DETECTION_AVAILABLE:
                analysis_result = self._enhanced_analysis(document_text, str(document_id))
            else:
                # Fallback to original analysis
                analysis_result = self.rag_pipeline.analyze_document_for_arbitration(document_id)
            
            # Store analysis in database
            db_analysis = self._create_analysis_record(
                db, document_id, analysis_result
            )
            
            logger.info(f"Enhanced analysis completed for document {document_id}: "
                       f"{'HAS' if db_analysis.has_arbitration_clause else 'NO'} arbitration clause")
            
            return db_analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis for document {document_id}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def quick_analyze_text(self, 
                          db: Session, 
                          request: QuickAnalysisRequest) -> Dict[str, Any]:
        """
        Perform quick analysis on raw text with enhanced validation.
        
        Args:
            db: Database session
            request: Quick analysis request
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Perform enhanced analysis if available
            if ENHANCED_DETECTION_AVAILABLE:
                analysis_result = self._enhanced_analysis(request.text, "quick_analysis")
            else:
                # Fallback to original quick analysis
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
            
            # Add validation info if available
            if hasattr(analysis_result, 'validation_result') and analysis_result.validation_result:
                response["document_validation"] = {
                    "is_legal_document": analysis_result.validation_result.is_legal_document,
                    "document_type": analysis_result.validation_result.document_type.value,
                    "validation_confidence": analysis_result.validation_result.confidence,
                    "warning_flags": analysis_result.validation_result.warning_flags
                }
            
            if request.include_context:
                response["analysis_details"] = {
                    "confidence_level": analysis_result.metadata.get("confidence_level"),
                    "validation_passed": analysis_result.metadata.get("validation_passed", True),
                    "document_type": analysis_result.metadata.get("document_type", "unknown")
                }
            
            logger.info(f"Enhanced quick analysis completed: "
                       f"{'HAS' if analysis_result.has_arbitration_clause else 'NO'} arbitration clause "
                       f"(confidence: {analysis_result.confidence_score:.2f})")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced quick text analysis: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _enhanced_analysis(self, text: str, document_id: str) -> EnhancedAnalysisResult:
        """
        Perform enhanced arbitration analysis with document validation.
        
        Args:
            text: Document text
            document_id: Document identifier
            
        Returns:
            Enhanced analysis result
        """
        start_time = time.time()
        
        try:
            # Use enhanced arbitration detector
            detection_result = self.arbitration_detector.detect(text, document_id)
            
            # Convert to our format
            clauses = []
            for clause in detection_result.clauses:
                clause_dict = {
                    "text": clause.text,
                    "type": clause.arbitration_type.value,
                    "relevance_score": clause.confidence_score,
                    "signals": {
                        "binding_arbitration": clause.arbitration_type.value in ["binding", "mandatory"],
                        "class_action_waiver": "class_action_waiver" in [ct.value for ct in clause.clause_types],
                        "jury_waiver": "jury_trial_waiver" in [ct.value for ct in clause.clause_types],
                        "arbitration_keywords_count": len(clause.pattern_matches)
                    },
                    "start_position": clause.location.get("start_char"),
                    "end_position": clause.location.get("end_char"),
                    "context": clause.text[:200] + "..." if len(clause.text) > 200 else clause.text
                }
                clauses.append(clause_dict)
            
            # Enhanced metadata
            metadata = {
                "analysis_method": "enhanced_arbitration_detector",
                "total_clauses": len(detection_result.clauses),
                "validation_passed": detection_result.validation_result.is_legal_document if detection_result.validation_result else True,
                "document_type": detection_result.validation_result.document_type.value if detection_result.validation_result else "unknown",
                "confidence_level": self._get_confidence_level(detection_result.confidence),
                "processing_time_ms": detection_result.processing_time * 1000
            }
            
            # Add rejection reason if document was rejected
            if detection_result.validation_result and not detection_result.validation_result.is_legal_document:
                metadata["rejection_reason"] = f"Document identified as {detection_result.validation_result.document_type.value}"
                metadata["warning_flags"] = detection_result.validation_result.warning_flags
            
            processing_time = (time.time() - start_time) * 1000
            
            return EnhancedAnalysisResult(
                has_arbitration_clause=detection_result.has_arbitration,
                confidence_score=detection_result.confidence,
                summary=self._generate_enhanced_summary(detection_result),
                clauses=clauses,
                processing_time_ms=int(processing_time),
                metadata=metadata,
                validation_result=detection_result.validation_result.to_dict() if detection_result.validation_result else None
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            # Return safe default
            return EnhancedAnalysisResult(
                has_arbitration_clause=False,
                confidence_score=0.0,
                summary="Analysis failed due to error",
                clauses=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e), "analysis_method": "enhanced_arbitration_detector"}
            )
    
    def _generate_enhanced_summary(self, detection_result: DetectionResult) -> str:
        """Generate enhanced summary with validation info."""
        if not detection_result.has_arbitration:
            if detection_result.validation_result and not detection_result.validation_result.is_legal_document:
                return f"Document rejected as {detection_result.validation_result.document_type.value.replace('_', ' ')} - no arbitration analysis performed."
            else:
                return "No arbitration clauses detected in the legal document."
        
        summary_parts = [
            f"Arbitration clauses detected with {detection_result.confidence:.1%} confidence."
        ]
        
        # Add clause type information
        clause_types = set()
        for clause in detection_result.clauses:
            clause_types.add(clause.arbitration_type.value)
            clause_types.update(ct.value for ct in clause.clause_types)
        
        if "binding" in clause_types or "mandatory" in clause_types:
            summary_parts.append("Contains binding arbitration requirements.")
        
        if "class_action_waiver" in clause_types:
            summary_parts.append("Includes class action waivers.")
        
        if "jury_trial_waiver" in clause_types:
            summary_parts.append("Contains jury trial waivers.")
        
        summary_parts.append(f"Found {len(detection_result.clauses)} relevant clause(s).")
        
        # Add validation info
        if detection_result.validation_result:
            summary_parts.append(f"Document validated as {detection_result.validation_result.document_type.value.replace('_', ' ')}.")
        
        return " ".join(summary_parts)
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string."""
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.75:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"
    
    # Delegate other methods to original service
    def get_analysis(self, db: Session, analysis_id: int) -> Optional[ArbitrationAnalysis]:
        """Get analysis by ID."""
        return db.query(ArbitrationAnalysis).filter(
            ArbitrationAnalysis.id == analysis_id
        ).first()
    
    def get_latest_analysis(self, db: Session, document_id: int) -> Optional[ArbitrationAnalysis]:
        """Get the latest analysis for a document."""
        return db.query(ArbitrationAnalysis).filter(
            ArbitrationAnalysis.document_id == document_id
        ).order_by(desc(ArbitrationAnalysis.analyzed_at)).first()
    
    def get_document_analyses(self, db: Session, document_id: int) -> List[ArbitrationAnalysis]:
        """Get all analyses for a document."""
        return db.query(ArbitrationAnalysis).filter(
            ArbitrationAnalysis.document_id == document_id
        ).order_by(desc(ArbitrationAnalysis.analyzed_at)).all()
    
    def _create_analysis_record(self, 
                               db: Session, 
                               document_id: int, 
                               analysis_result) -> ArbitrationAnalysis:
        """Create analysis record in database."""
        try:
            # Handle both enhanced and original analysis results
            if hasattr(analysis_result, 'validation_result'):
                # Enhanced result
                metadata = analysis_result.metadata.copy()
                if analysis_result.validation_result:
                    metadata["validation"] = {
                        "is_legal_document": analysis_result.validation_result.is_legal_document,
                        "document_type": analysis_result.validation_result.document_type.value,
                        "confidence": analysis_result.validation_result.confidence
                    }
            else:
                # Original result
                metadata = getattr(analysis_result, 'metadata', {})
            
            # Create main analysis record
            db_analysis = ArbitrationAnalysis(
                document_id=document_id,
                has_arbitration_clause=analysis_result.has_arbitration_clause,
                confidence_score=analysis_result.confidence_score,
                analysis_summary=analysis_result.summary,
                analyzed_at=datetime.utcnow(),
                analysis_version="2.0-enhanced",  # Mark as enhanced version
                processing_time_ms=analysis_result.processing_time_ms,
                metadata=metadata
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
            logger.error(f"Error creating enhanced analysis record: {e}")
            raise
    
    def _calculate_severity_score(self, clause_data: Dict[str, Any]) -> float:
        """Calculate severity score for a clause."""
        severity = 0.5  # Base severity
        
        signals = clause_data.get("signals", {})
        
        # Increase severity for restrictive clauses
        if signals.get("binding_arbitration", False):
            severity += 0.2
        
        if signals.get("class_action_waiver", False):
            severity += 0.15
        
        if signals.get("jury_waiver", False):
            severity += 0.1
        
        # Cap at 1.0
        return min(severity, 1.0)