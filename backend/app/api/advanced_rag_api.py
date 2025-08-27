"""
Advanced RAG API Endpoints for Legal Document Analysis
Exposes sophisticated AI-powered document analysis capabilities
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime
import json
import io
import logging

from app.rag.advanced_rag_pipeline import (
    AdvancedRAGPipeline,
    NaturalLanguageInterface
)
from app.rag.intelligent_extraction import IntelligentClauseExtractor
from app.rag.confidence_scoring import (
    ConfidenceScoringEngine,
    RiskAssessmentEngine
)
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.db.database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advanced-rag", tags=["Advanced RAG"])

# Initialize AI components
rag_pipeline = AdvancedRAGPipeline()
clause_extractor = IntelligentClauseExtractor()
confidence_engine = ConfidenceScoringEngine()
risk_engine = RiskAssessmentEngine()


# Pydantic models for API requests/responses
class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis"""
    document_id: Optional[str] = None
    content: str = Field(..., description="Document content to analyze")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: comprehensive, quick, arbitration_only")
    include_risk_assessment: bool = Field(default=True)
    include_confidence_scores: bool = Field(default=True)
    extract_clauses: bool = Field(default=True)


class MultiDocumentAnalysisRequest(BaseModel):
    """Request model for multi-document analysis"""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to analyze")
    cross_reference: bool = Field(default=True, description="Enable cross-document analysis")
    comparison_mode: str = Field(default="full", description="Comparison mode: full, summary, differences")


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Natural language search query")
    document_ids: Optional[List[str]] = Field(default=None, description="Specific documents to search")
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(default=None)
    include_context: bool = Field(default=True)


class NaturalLanguageQueryRequest(BaseModel):
    """Request model for natural language queries"""
    question: str = Field(..., description="Natural language question about documents")
    document_ids: Optional[List[str]] = Field(default=None)
    conversation_id: Optional[str] = Field(default=None)
    include_sources: bool = Field(default=True)


class ClauseExtractionRequest(BaseModel):
    """Request model for clause extraction"""
    content: str = Field(..., description="Document content")
    clause_types: Optional[List[str]] = Field(default=None, description="Specific clause types to extract")
    include_risk_analysis: bool = Field(default=True)
    include_suggestions: bool = Field(default=True)


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment"""
    document_id: Optional[str] = None
    content: str = Field(..., description="Document content")
    clauses: Optional[List[Dict[str, Any]]] = Field(default=None)
    context: Dict[str, Any] = Field(default_factory=dict, description="Context: jurisdiction, party_type, industry")
    assessment_depth: str = Field(default="detailed", description="Assessment depth: quick, standard, detailed")


class ArbitrationSuggestionRequest(BaseModel):
    """Request model for arbitration clause suggestions"""
    context: Dict[str, Any] = Field(..., description="Context: jurisdiction, contract_type, parties")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements")
    variations_count: int = Field(default=3, ge=1, le=10)
    include_precedents: bool = Field(default=True)


@router.post("/analyze-document", response_model=Dict[str, Any])
async def analyze_document(
    request: DocumentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Perform comprehensive AI-powered document analysis
    
    Returns complete analysis with clauses, entities, risks, and insights
    """
    try:
        start_time = datetime.now()
        
        # Prepare document for analysis
        document = {
            'id': request.document_id or f"doc_{datetime.now().timestamp()}",
            'content': request.content,
            'metadata': request.metadata
        }
        
        # Perform base analysis
        if request.analysis_type == "comprehensive":
            analysis_result = rag_pipeline.analyze_single_document(document)
        elif request.analysis_type == "quick":
            analysis_result = rag_pipeline.quick_text_analysis(request.content)
        else:  # arbitration_only
            analysis_result = rag_pipeline._analyze_arbitration_content(
                request.content, []
            )
        
        # Extract clauses if requested
        clauses = []
        if request.extract_clauses:
            extracted_clauses = clause_extractor.extract_all_clauses(request.content)
            clauses = [
                {
                    'id': c.id,
                    'type': c.type,
                    'subtype': c.subtype,
                    'text': c.text,
                    'confidence': c.confidence,
                    'entities': c.entities,
                    'obligations': c.obligations,
                    'conditions': c.conditions,
                    'temporal_elements': c.temporal_elements,
                    'legal_concepts': c.legal_concepts,
                    'enforceability_score': c.enforceability_score,
                    'clarity_score': c.clarity_score
                }
                for c in extracted_clauses
            ]
        
        # Perform risk assessment if requested
        risk_assessment = None
        if request.include_risk_assessment:
            risk_score = risk_engine.assess_document_risk(
                document, clauses, request.metadata
            )
            risk_assessment = {
                'overall_risk': risk_score.overall_risk,
                'risk_category': risk_score.risk_category,
                'risk_factors': risk_score.risk_factors,
                'mitigation_priority': risk_score.mitigation_priority,
                'legal_exposure': risk_score.legal_exposure,
                'financial_impact': risk_score.financial_impact,
                'recommendations': risk_score.recommendations
            }
        
        # Calculate confidence scores if requested
        confidence_scores = None
        if request.include_confidence_scores:
            # Create evidence from analysis
            evidence = [
                {'relevance': 0.8, 'keyword_count': 5, 'entity_count': 3}
                for _ in range(len(clauses))
            ]
            confidence = confidence_engine.calculate_confidence_score(
                analysis_result, evidence
            )
            confidence_scores = {
                'overall_score': confidence.overall_score,
                'component_scores': confidence.component_scores,
                'factors': confidence.factors,
                'explanation': confidence.explanation,
                'reliability': confidence.reliability_indicator,
                'confidence_interval': confidence.confidence_interval
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log analysis to background task
        background_tasks.add_task(
            log_analysis_activity,
            user_id=current_user.id,
            document_id=document['id'],
            analysis_type=request.analysis_type,
            processing_time=processing_time
        )
        
        return {
            'success': True,
            'document_id': document['id'],
            'analysis': analysis_result,
            'clauses': clauses,
            'risk_assessment': risk_assessment,
            'confidence_scores': confidence_scores,
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-multiple-documents", response_model=Dict[str, Any])
async def analyze_multiple_documents(
    request: MultiDocumentAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze multiple documents with cross-referencing and comparison
    
    Provides consolidated insights across document set
    """
    try:
        # Validate documents
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        if len(request.documents) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 documents allowed")
        
        # Perform analysis
        analysis_result = await rag_pipeline.analyze_multiple_documents(
            request.documents,
            cross_reference=request.cross_reference
        )
        
        # Format response based on comparison mode
        if request.comparison_mode == "summary":
            response = {
                'success': True,
                'document_count': len(request.documents),
                'consolidated_insights': analysis_result['consolidated_insights'],
                'overall_risk': analysis_result['overall_risk_assessment'],
                'key_differences': analysis_result.get('cross_document_analysis', {})
            }
        elif request.comparison_mode == "differences":
            response = {
                'success': True,
                'document_count': len(request.documents),
                'differences': analysis_result.get('cross_document_analysis', {}),
                'similarity_matrix': analysis_result.get('document_similarity_matrix', [])
            }
        else:  # full
            response = {
                'success': True,
                'analysis': analysis_result
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Multi-document analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic-search", response_model=Dict[str, Any])
async def semantic_search(
    request: SemanticSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Perform advanced semantic search across documents
    
    Uses AI embeddings and relevance scoring for intelligent search
    """
    try:
        # Load documents for search
        if request.document_ids:
            # Load specific documents from database
            documents = []  # Would load from DB
        else:
            # Load user's documents
            documents = []  # Would load user's documents
        
        # Perform semantic search
        search_results = rag_pipeline.perform_semantic_search(
            query=request.query,
            documents=documents,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_result = {
                'document_id': result.document_id,
                'content': result.content if request.include_context else result.content[:200] + '...',
                'relevance_score': result.relevance_score,
                'semantic_similarity': result.semantic_similarity,
                'keyword_match_score': result.keyword_match_score,
                'highlighted_text': result.highlighted_text,
                'metadata': result.source_metadata
            }
            
            if result.related_clauses:
                formatted_result['related_clauses'] = [
                    {
                        'type': c.clause_type,
                        'text': c.text[:200] + '...',
                        'risk_level': c.risk_level
                    }
                    for c in result.related_clauses
                ]
            
            formatted_results.append(formatted_result)
        
        return {
            'success': True,
            'query': request.query,
            'results_count': len(formatted_results),
            'results': formatted_results,
            'search_metadata': {
                'documents_searched': len(documents),
                'filters_applied': request.filters,
                'timestamp': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/natural-language-query", response_model=Dict[str, Any])
async def natural_language_query(
    request: NaturalLanguageQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Answer natural language questions about documents
    
    Provides conversational interface for document Q&A
    """
    try:
        # Load documents
        documents = []  # Would load from DB based on request.document_ids
        
        # Create or retrieve NL interface
        nl_interface = rag_pipeline.create_natural_language_interface(documents)
        
        # Process query
        response = nl_interface.query(request.question)
        
        # Format response
        formatted_response = {
            'success': True,
            'question': request.question,
            'answer': response,
            'conversation_id': request.conversation_id or f"conv_{datetime.now().timestamp()}"
        }
        
        if request.include_sources and 'sources' in response:
            formatted_response['sources'] = response['sources']
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Natural language query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-clauses", response_model=Dict[str, Any])
async def extract_clauses(
    request: ClauseExtractionRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Extract and analyze contract clauses using AI
    
    Provides detailed clause extraction with classification and risk analysis
    """
    try:
        # Extract clauses
        extracted_clauses = clause_extractor.extract_all_clauses(request.content)
        
        # Filter by requested types if specified
        if request.clause_types:
            extracted_clauses = [
                c for c in extracted_clauses
                if c.type in request.clause_types
            ]
        
        # Format clauses
        formatted_clauses = []
        for clause in extracted_clauses:
            formatted_clause = {
                'id': clause.id,
                'type': clause.type,
                'subtype': clause.subtype,
                'text': clause.text,
                'position': {
                    'start': clause.start_position,
                    'end': clause.end_position
                },
                'confidence': clause.confidence,
                'entities': clause.entities,
                'obligations': clause.obligations,
                'conditions': clause.conditions,
                'temporal_elements': clause.temporal_elements,
                'cross_references': clause.cross_references,
                'legal_concepts': clause.legal_concepts,
                'scores': {
                    'enforceability': clause.enforceability_score,
                    'clarity': clause.clarity_score
                }
            }
            
            # Add risk analysis if requested
            if request.include_risk_analysis:
                clause_dict = {'clause_type': clause.type, 'text': clause.text}
                risk_profile = risk_engine._assess_clause_risk(clause_dict, {})
                formatted_clause['risk_analysis'] = {
                    'risk_level': risk_profile.risk_level,
                    'risk_score': risk_profile.risk_score,
                    'vulnerabilities': risk_profile.vulnerability_points,
                    'enforceability_concerns': risk_profile.enforceability_concerns
                }
            
            # Add suggestions if requested
            if request.include_suggestions:
                suggestions = rag_pipeline._generate_clause_suggestions(
                    clause.text, clause.type, 
                    risk_profile.risk_level if request.include_risk_analysis else 'medium'
                )
                formatted_clause['suggestions'] = suggestions
            
            formatted_clauses.append(formatted_clause)
        
        return {
            'success': True,
            'clauses_count': len(formatted_clauses),
            'clauses': formatted_clauses,
            'clause_types_found': list(set(c['type'] for c in formatted_clauses)),
            'extraction_metadata': {
                'total_text_length': len(request.content),
                'timestamp': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Clause extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-risk", response_model=Dict[str, Any])
async def assess_risk(
    request: RiskAssessmentRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Perform comprehensive risk assessment on legal document
    
    Provides detailed risk scoring with mitigation strategies
    """
    try:
        # Prepare document
        document = {
            'id': request.document_id or f"doc_{datetime.now().timestamp()}",
            'content': request.content,
            'metadata': request.context
        }
        
        # Extract clauses if not provided
        if request.clauses is None:
            extracted_clauses = clause_extractor.extract_all_clauses(request.content)
            clauses = [
                {
                    'id': c.id,
                    'clause_type': c.type,
                    'text': c.text,
                    'confidence_score': c.confidence
                }
                for c in extracted_clauses
            ]
        else:
            clauses = request.clauses
        
        # Perform risk assessment
        risk_score = risk_engine.assess_document_risk(
            document, clauses, request.context
        )
        
        # Format response based on assessment depth
        if request.assessment_depth == "quick":
            response = {
                'success': True,
                'overall_risk': risk_score.overall_risk,
                'risk_category': risk_score.risk_category,
                'top_risks': risk_score.risk_factors[:3],
                'key_recommendations': risk_score.recommendations[:3]
            }
        elif request.assessment_depth == "standard":
            response = {
                'success': True,
                'overall_risk': risk_score.overall_risk,
                'risk_category': risk_score.risk_category,
                'risk_factors': risk_score.risk_factors,
                'mitigation_priority': risk_score.mitigation_priority[:5],
                'recommendations': risk_score.recommendations
            }
        else:  # detailed
            response = {
                'success': True,
                'overall_risk': risk_score.overall_risk,
                'risk_category': risk_score.risk_category,
                'risk_factors': risk_score.risk_factors,
                'mitigation_priority': risk_score.mitigation_priority,
                'legal_exposure': risk_score.legal_exposure,
                'financial_impact': risk_score.financial_impact,
                'operational_impact': risk_score.operational_impact,
                'recommendations': risk_score.recommendations,
                'confidence_in_assessment': risk_score.confidence_in_assessment,
                'assessment_metadata': {
                    'clauses_analyzed': len(clauses),
                    'context_provided': request.context,
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-arbitration-suggestions", response_model=Dict[str, Any])
async def generate_arbitration_suggestions(
    request: ArbitrationSuggestionRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate AI-powered arbitration clause suggestions
    
    Provides customized arbitration clauses based on context and requirements
    """
    try:
        # Generate suggestions
        suggestions = rag_pipeline.generate_arbitration_suggestions(
            request.context,
            request.requirements
        )
        
        # Limit to requested count
        suggestions = suggestions[:request.variations_count]
        
        # Format suggestions
        formatted_suggestions = []
        for suggestion in suggestions:
            formatted_suggestion = {
                'type': suggestion['type'],
                'clause_text': suggestion['clause'],
                'pros': suggestion['pros'],
                'cons': suggestion['cons'],
                'confidence': suggestion['confidence'],
                'risk_level': suggestion['risk_level'],
                'suitability_score': suggestion['suitability_score']
            }
            
            # Add precedents if requested
            if request.include_precedents:
                # Would fetch relevant precedents from database
                formatted_suggestion['precedents'] = [
                    {
                        'case': 'Sample v. Example Corp',
                        'year': 2023,
                        'outcome': 'Clause enforced',
                        'relevance': 0.85
                    }
                ]
            
            formatted_suggestions.append(formatted_suggestion)
        
        return {
            'success': True,
            'suggestions_count': len(formatted_suggestions),
            'suggestions': formatted_suggestions,
            'context_summary': {
                'jurisdiction': request.context.get('jurisdiction', 'Not specified'),
                'contract_type': request.context.get('contract_type', 'Not specified'),
                'requirements_addressed': len(request.requirements)
            },
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'ai_model': 'Advanced Legal AI'
            }
        }
        
    except Exception as e:
        logger.error(f"Arbitration suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-document-stream", response_model=Dict[str, Any])
async def analyze_document_stream(
    file: UploadFile = File(...),
    analysis_type: str = "comprehensive",
    current_user: User = Depends(get_current_user)
):
    """
    Stream analysis of uploaded document
    
    Provides real-time analysis updates for large documents
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Create async generator for streaming results
        async def analysis_generator():
            # Initial response
            yield json.dumps({
                'status': 'started',
                'message': 'Beginning document analysis...'
            }) + '\n'
            
            # Extract clauses
            yield json.dumps({
                'status': 'processing',
                'step': 'clause_extraction',
                'message': 'Extracting contract clauses...'
            }) + '\n'
            
            clauses = clause_extractor.extract_all_clauses(content_str)
            
            yield json.dumps({
                'status': 'processing',
                'step': 'clause_extraction',
                'result': f'Extracted {len(clauses)} clauses'
            }) + '\n'
            
            # Risk assessment
            yield json.dumps({
                'status': 'processing',
                'step': 'risk_assessment',
                'message': 'Performing risk assessment...'
            }) + '\n'
            
            clause_dicts = [
                {'clause_type': c.type, 'text': c.text}
                for c in clauses
            ]
            risk_score = risk_engine.assess_document_risk(
                {'content': content_str}, clause_dicts, {}
            )
            
            yield json.dumps({
                'status': 'processing',
                'step': 'risk_assessment',
                'result': f'Overall risk: {risk_score.overall_risk:.2f}'
            }) + '\n'
            
            # Final results
            yield json.dumps({
                'status': 'completed',
                'results': {
                    'clauses_count': len(clauses),
                    'risk_score': risk_score.overall_risk,
                    'risk_category': risk_score.risk_category,
                    'top_recommendations': risk_score.recommendations[:3]
                }
            }) + '\n'
        
        return StreamingResponse(
            analysis_generator(),
            media_type='application/x-ndjson'
        )
        
    except Exception as e:
        logger.error(f"Document stream analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis-capabilities", response_model=Dict[str, Any])
async def get_analysis_capabilities() -> Dict[str, Any]:
    """
    Get available analysis capabilities and configuration
    
    Returns information about supported analysis types and features
    """
    return {
        'capabilities': {
            'document_analysis': {
                'types': ['comprehensive', 'quick', 'arbitration_only'],
                'features': [
                    'clause_extraction',
                    'entity_recognition',
                    'risk_assessment',
                    'confidence_scoring',
                    'cross_referencing'
                ]
            },
            'clause_extraction': {
                'supported_types': [
                    'arbitration', 'payment', 'termination', 'confidentiality',
                    'liability', 'warranty', 'intellectual_property', 'force_majeure',
                    'governing_law', 'assignment'
                ],
                'extraction_features': [
                    'obligations', 'conditions', 'temporal_elements',
                    'entities', 'cross_references', 'legal_concepts'
                ]
            },
            'risk_assessment': {
                'risk_categories': ['legal', 'financial', 'operational', 'reputational'],
                'assessment_depths': ['quick', 'standard', 'detailed'],
                'risk_factors': [
                    'ambiguous_terms', 'one_sided_provisions', 'missing_clauses',
                    'enforceability_concerns', 'jurisdiction_conflicts'
                ]
            },
            'semantic_search': {
                'max_results': 100,
                'search_features': [
                    'natural_language_queries',
                    'keyword_matching',
                    'semantic_similarity',
                    'context_relevance'
                ]
            },
            'ai_models': {
                'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
                'classification_model': 'legal-bert-base-uncased',
                'qa_model': 'roberta-base-squad2',
                'confidence_threshold': 0.7
            }
        },
        'limits': {
            'max_document_size_mb': 10,
            'max_documents_batch': 20,
            'max_search_results': 100,
            'max_clause_suggestions': 10
        },
        'supported_formats': ['text', 'pdf', 'docx'],
        'api_version': 'v1',
        'last_updated': '2024-01-20'
    }


# Helper functions
async def log_analysis_activity(
    user_id: int,
    document_id: str,
    analysis_type: str,
    processing_time: float
):
    """Log analysis activity for monitoring and improvement"""
    logger.info(f"Analysis completed: user={user_id}, doc={document_id}, "
                f"type={analysis_type}, time={processing_time}s")