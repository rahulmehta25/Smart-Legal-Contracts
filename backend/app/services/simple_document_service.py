"""
Simplified document service for basic API testing
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import os
import uuid
from datetime import datetime
from loguru import logger

from app.models.simple_document import Document, Chunk
from app.schemas.document import DocumentCreate, DocumentResponse


class DocumentService:
    """Simplified document service for testing"""
    
    def create_document(self, db: Session, document_data: DocumentCreate) -> Document:
        """Create a new document"""
        try:
            # Create document instance
            document = Document(
                user_id=1,  # Mock user ID for testing
                filename=document_data.filename,
                original_filename=document_data.filename,
                file_path=f"/tmp/{document_data.filename}",
                file_type=document_data.content_type or "text/plain",
                file_size=len(document_data.content or ""),
                content_hash="mock_hash",
                processing_status="pending"
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            return document
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            db.rollback()
            raise
    
    def process_document(self, db: Session, document_id: int) -> Dict[str, Any]:
        """Process a document (simplified version)"""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError("Document not found")
            
            # Mock processing
            document.processing_status = "completed"
            document.total_chunks = 1
            db.commit()
            
            return {"chunks_created": 1, "status": "completed"}
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def get_document(self, db: Session, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        try:
            return db.query(Document).filter(Document.id == document_id).first()
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    def get_documents(self, db: Session, skip: int = 0, limit: int = 100, 
                     processed_only: bool = False) -> List[Document]:
        """Get list of documents"""
        try:
            query = db.query(Document)
            if processed_only:
                query = query.filter(Document.processing_status == "completed")
            return query.offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
    
    def search_documents(self, db: Session, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents (simplified version)"""
        try:
            # Mock search results
            return [
                {
                    "document_id": 1,
                    "filename": "mock_document.txt",
                    "similarity_score": 0.85,
                    "excerpt": f"Mock search result for query: {query}"
                }
            ]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """Delete a document"""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                db.delete(document)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_document_statistics(self, db: Session) -> Dict[str, Any]:
        """Get document statistics"""
        try:
            total_docs = db.query(Document).count()
            return {
                "total_documents": total_docs,
                "processed_documents": total_docs,
                "pending_documents": 0,
                "total_storage_bytes": 1024
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_documents": 0, "processed_documents": 0}
    
    def get_document_chunks(self, db: Session, document_id: int) -> List[Chunk]:
        """Get document chunks"""
        return []
    
    def reprocess_document(self, db: Session, document_id: int) -> Dict[str, Any]:
        """Reprocess a document"""
        return self.process_document(db, document_id)