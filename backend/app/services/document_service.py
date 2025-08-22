from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
import os
import uuid
from datetime import datetime
from loguru import logger

from app.models.document import Document, DocumentChunk, DocumentCreate, DocumentResponse
from app.rag.pipeline import RAGPipeline
from app.db.vector_store import get_vector_store


class DocumentService:
    """
    Service for managing documents and their processing
    """
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.vector_store = get_vector_store()
    
    def create_document(self, 
                       db: Session, 
                       document_data: DocumentCreate,
                       file_path: Optional[str] = None) -> Document:
        """
        Create a new document in the database
        
        Args:
            db: Database session
            document_data: Document data
            file_path: Optional file path if document was uploaded
            
        Returns:
            Created Document instance
        """
        try:
            # Create document record
            db_document = Document(
                filename=document_data.filename,
                file_path=file_path,
                content=document_data.content,
                content_type=document_data.content_type,
                file_size=len(document_data.content.encode('utf-8')),
                uploaded_at=datetime.utcnow(),
                is_processed=False
            )
            
            db.add(db_document)
            db.commit()
            db.refresh(db_document)
            
            logger.info(f"Created document {db_document.id}: {document_data.filename}")
            return db_document
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating document: {e}")
            raise
    
    def process_document(self, db: Session, document_id: int) -> Dict[str, Any]:
        """
        Process a document through the RAG pipeline
        
        Args:
            db: Database session
            document_id: ID of document to process
            
        Returns:
            Processing results
        """
        try:
            # Get document
            document = self.get_document(db, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            if document.is_processed:
                logger.info(f"Document {document_id} already processed")
                return {"message": "Document already processed", "document_id": document_id}
            
            # Process through RAG pipeline
            chunks, processing_metadata = self.rag_pipeline.process_document(
                document_id=document_id,
                text=document.content,
                chunk_size=1000
            )
            
            # Store chunks in database
            db_chunks = []
            for chunk in chunks:
                db_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    token_count=chunk.token_count,
                    embedding_id=getattr(chunk, 'embedding_id', None)
                )
                db_chunks.append(db_chunk)
            
            db.add_all(db_chunks)
            
            # Update document as processed
            document.is_processed = True
            document.processed_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Processed document {document_id}: {len(chunks)} chunks created")
            
            return {
                "message": "Document processed successfully",
                "document_id": document_id,
                "chunks_created": len(chunks),
                "processing_time_ms": processing_metadata.get("processing_time_ms", 0),
                "metadata": processing_metadata
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing document {document_id}: {e}")
            raise
    
    def get_document(self, db: Session, document_id: int) -> Optional[Document]:
        """
        Get a document by ID
        """
        return db.query(Document).filter(Document.id == document_id).first()
    
    def get_documents(self, 
                     db: Session, 
                     skip: int = 0, 
                     limit: int = 100,
                     processed_only: bool = False) -> List[Document]:
        """
        Get list of documents with pagination
        """
        query = db.query(Document)
        
        if processed_only:
            query = query.filter(Document.is_processed == True)
        
        return query.order_by(desc(Document.uploaded_at)).offset(skip).limit(limit).all()
    
    def get_document_chunks(self, 
                           db: Session, 
                           document_id: int) -> List[DocumentChunk]:
        """
        Get all chunks for a document
        """
        return db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """
        Delete a document and all its associated data
        
        Args:
            db: Database session
            document_id: ID of document to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get document
            document = self.get_document(db, document_id)
            if not document:
                return False
            
            # Delete from vector store
            try:
                self.vector_store.delete_document(document_id)
            except Exception as e:
                logger.warning(f"Error deleting from vector store: {e}")
            
            # Delete file if it exists
            if document.file_path and os.path.exists(document.file_path):
                try:
                    os.remove(document.file_path)
                except Exception as e:
                    logger.warning(f"Error deleting file {document.file_path}: {e}")
            
            # Delete from database (cascades to chunks and analyses)
            db.delete(document)
            db.commit()
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    def search_documents(self, 
                        db: Session, 
                        query: str, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using vector similarity
        
        Args:
            db: Database session
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results with document info
        """
        try:
            # Perform vector search
            vector_results = self.vector_store.similarity_search(
                query=query,
                k=limit * 2  # Get more results to filter
            )
            
            # Group results by document and get document info
            document_scores = {}
            for result in vector_results:
                doc_id = result['metadata']['document_id']
                score = 1.0 - result['distance']  # Convert distance to similarity
                
                if doc_id not in document_scores:
                    document_scores[doc_id] = {
                        'max_score': score,
                        'avg_score': score,
                        'chunk_count': 1,
                        'best_chunk': result['document'][:200] + "..."
                    }
                else:
                    document_scores[doc_id]['max_score'] = max(
                        document_scores[doc_id]['max_score'], score
                    )
                    document_scores[doc_id]['avg_score'] = (
                        document_scores[doc_id]['avg_score'] * document_scores[doc_id]['chunk_count'] + score
                    ) / (document_scores[doc_id]['chunk_count'] + 1)
                    document_scores[doc_id]['chunk_count'] += 1
                    
                    if score > document_scores[doc_id]['max_score']:
                        document_scores[doc_id]['best_chunk'] = result['document'][:200] + "..."
            
            # Get document details and create results
            search_results = []
            for doc_id, scores in document_scores.items():
                document = self.get_document(db, doc_id)
                if document:
                    result = {
                        'document_id': doc_id,
                        'filename': document.filename,
                        'uploaded_at': document.uploaded_at,
                        'is_processed': document.is_processed,
                        'relevance_score': scores['max_score'],
                        'avg_relevance': scores['avg_score'],
                        'matching_chunks': scores['chunk_count'],
                        'preview': scores['best_chunk']
                    }
                    search_results.append(result)
            
            # Sort by relevance and return top results
            search_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics about documents in the system
        """
        try:
            total_documents = db.query(Document).count()
            processed_documents = db.query(Document).filter(Document.is_processed == True).count()
            total_chunks = db.query(DocumentChunk).count()
            
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'total_documents': total_documents,
                'processed_documents': processed_documents,
                'unprocessed_documents': total_documents - processed_documents,
                'total_chunks': total_chunks,
                'vector_store_stats': vector_stats,
                'processing_rate': processed_documents / total_documents if total_documents > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {'error': str(e)}
    
    def reprocess_document(self, db: Session, document_id: int) -> Dict[str, Any]:
        """
        Reprocess an existing document
        
        Args:
            db: Database session
            document_id: ID of document to reprocess
            
        Returns:
            Reprocessing results
        """
        try:
            # Get document
            document = self.get_document(db, document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Delete existing chunks
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            
            # Delete from vector store
            try:
                self.vector_store.delete_document(document_id)
            except Exception as e:
                logger.warning(f"Error deleting from vector store during reprocessing: {e}")
            
            # Mark as unprocessed
            document.is_processed = False
            document.processed_at = None
            db.commit()
            
            # Reprocess
            return self.process_document(db, document_id)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error reprocessing document {document_id}: {e}")
            raise