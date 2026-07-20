"""Integration module to connect existing backend with new RAG system."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Add RAG system to path
rag_path = Path(__file__).parent / "rag_system" / "src"
sys.path.insert(0, str(rag_path))

from rag_system.src.core.arbitration_detector import ArbitrationDetectionPipeline, ArbitrationClause
from rag_system.src.comparison.comparison_engine import ClauseComparisonEngine
from rag_system.src.database.schema import DatabaseManager, VectorStore

logger = logging.getLogger(__name__)

class RAGIntegration:
    """Integration class for RAG system with existing backend."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize RAG integration."""
        self.config = config or {}
        
        # Initialize components
        self.pipeline = ArbitrationDetectionPipeline(
            cache_enabled=self.config.get('cache_enabled', True)
        )
        
        self.comparison_engine = ClauseComparisonEngine(
            db_url=self.config.get('database_url')
        )
        
        logger.info("RAG Integration initialized")
    
    def detect_arbitration(self, file_path: str) -> Dict[str, Any]:
        """
        Detect arbitration clause in document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Detection result dictionary
        """
        try:
            result = self.pipeline.detect_arbitration_clause(file_path)
            
            if result:
                return {
                    'detected': True,
                    'confidence': result.confidence,
                    'clause_type': result.clause_type,
                    'summary': result.summary,
                    'key_provisions': result.key_provisions,
                    'location': result.location,
                    'full_text': result.full_text[:1000]  # Truncate for response
                }
            else:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'message': 'No arbitration clause detected'
                }
                
        except Exception as e:
            logger.error(f"Error detecting arbitration: {e}")
            return {
                'error': str(e),
                'detected': False
            }
    
    def detect_from_text(self, text: str) -> Dict[str, Any]:
        """
        Detect arbitration clause from text.
        
        Args:
            text: Document text
            
        Returns:
            Detection result dictionary
        """
        try:
            result = self.pipeline.detect_from_text(text)
            
            if result:
                return {
                    'detected': True,
                    'confidence': result.confidence,
                    'clause_type': result.clause_type,
                    'summary': result.summary,
                    'key_provisions': result.key_provisions,
                    'location': result.location
                }
            else:
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'message': 'No arbitration clause detected'
                }
                
        except Exception as e:
            logger.error(f"Error detecting from text: {e}")
            return {
                'error': str(e),
                'detected': False
            }
    
    def compare_clause(self, clause_text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compare clause with database.
        
        Args:
            clause_text: Clause to compare
            top_k: Number of similar clauses to return
            
        Returns:
            Comparison results
        """
        try:
            return self.comparison_engine.compare_clause(clause_text, top_k)
        except Exception as e:
            logger.error(f"Error comparing clause: {e}")
            return {
                'error': str(e),
                'similar_clauses': []
            }
    
    def add_to_database(self, clause_data: Dict) -> str:
        """
        Add clause to comparison database.
        
        Args:
            clause_data: Clause information
            
        Returns:
            Clause ID
        """
        try:
            return self.comparison_engine.add_clause_to_database(clause_data)
        except Exception as e:
            logger.error(f"Error adding to database: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            return self.comparison_engine.get_database_stats()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def batch_process(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of document paths
            
        Returns:
            List of detection results
        """
        results = []
        for path in file_paths:
            result = self.detect_arbitration(path)
            result['file'] = path
            results.append(result)
        return results

# Update existing backend endpoints to use RAG system
def update_backend_endpoints():
    """
    Update existing FastAPI endpoints to use RAG system.
    
    This function should be called from the main backend application.
    """
    from fastapi import FastAPI, UploadFile, File
    import tempfile
    
    # Initialize RAG integration
    rag = RAGIntegration()
    
    def create_rag_endpoints(app: FastAPI):
        """Add RAG endpoints to existing FastAPI app."""
        
        @app.post("/api/v2/analyze")
        async def analyze_document_rag(file: UploadFile = File(...)):
            """Analyze document using RAG system."""
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Use RAG system
                result = rag.detect_arbitration(tmp_path)
                
                # Add comparison if detected
                if result.get('detected'):
                    comparison = rag.compare_clause(
                        result.get('full_text', ''),
                        top_k=5
                    )
                    result['similar_clauses'] = comparison.get('similar_clauses', [])
                    result['recommendations'] = comparison.get('analysis', {}).get('recommendations', [])
                
                return result
                
            finally:
                os.unlink(tmp_path)
        
        @app.post("/api/v2/analyze/text")
        async def analyze_text_rag(text: str):
            """Analyze text using RAG system."""
            result = rag.detect_from_text(text)
            
            # Add comparison if detected
            if result.get('detected'):
                comparison = rag.compare_clause(text, top_k=5)
                result['similar_clauses'] = comparison.get('similar_clauses', [])
                result['recommendations'] = comparison.get('analysis', {}).get('recommendations', [])
            
            return result
        
        @app.get("/api/v2/stats")
        async def get_rag_stats():
            """Get RAG system statistics."""
            return rag.get_database_stats()
        
        @app.post("/api/v2/compare")
        async def compare_clause_rag(clause_text: str):
            """Compare clause using RAG system."""
            return rag.compare_clause(clause_text)
    
    return create_rag_endpoints

# Backward compatibility wrapper
class RAGWrapper:
    """Wrapper to provide backward compatibility with existing code."""
    
    def __init__(self):
        """Initialize wrapper."""
        self.rag = RAGIntegration()
    
    def analyze_document(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Analyze document (backward compatible)."""
        result = self.rag.detect_arbitration(file_path)
        
        # Transform to match old format if needed
        if result.get('detected'):
            return {
                'has_arbitration_clause': True,
                'confidence_score': result['confidence'],
                'clause_details': {
                    'type': result.get('clause_type'),
                    'provisions': result.get('key_provisions', []),
                    'summary': result.get('summary')
                },
                'risk_level': 'high' if result['confidence'] > 0.8 else 'medium'
            }
        else:
            return {
                'has_arbitration_clause': False,
                'confidence_score': 0.0,
                'risk_level': 'low'
            }
    
    def analyze_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze text (backward compatible)."""
        result = self.rag.detect_from_text(text)
        
        # Transform to match old format
        if result.get('detected'):
            return {
                'has_arbitration_clause': True,
                'confidence_score': result['confidence'],
                'clause_details': {
                    'type': result.get('clause_type'),
                    'provisions': result.get('key_provisions', []),
                    'summary': result.get('summary')
                }
            }
        else:
            return {
                'has_arbitration_clause': False,
                'confidence_score': 0.0
            }

# Function to migrate existing data to RAG system
def migrate_existing_data(source_db_path: str = None):
    """
    Migrate existing arbitration data to RAG system.
    
    Args:
        source_db_path: Path to existing database
    """
    rag = RAGIntegration()
    
    # This would connect to existing database and migrate data
    # Implementation depends on existing database structure
    
    logger.info("Data migration completed")

# Export main components
__all__ = [
    'RAGIntegration',
    'RAGWrapper',
    'update_backend_endpoints',
    'migrate_existing_data'
]