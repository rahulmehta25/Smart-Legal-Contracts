"""
DataLoaders for Document and Chunk entities
"""

from typing import List, Dict, Any
from aiodataloader import DataLoader
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_
from ..types import Document as DocumentType, Chunk as ChunkType
from ...db.models import Document, Chunk, Detection
from ...db.database import get_session


class DocumentDataLoader(DataLoader):
    """DataLoader for Document entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[DocumentType]:
        """Batch load documents by IDs"""
        try:
            # Fetch documents in batch
            documents = self.session.query(Document).filter(
                Document.id.in_(keys)
            ).all()
            
            # Create a mapping for efficient lookup
            doc_map = {doc.id: self._convert_to_graphql_type(doc) for doc in documents}
            
            # Return documents in the same order as keys
            return [doc_map.get(key) for key in keys]
            
        except Exception as e:
            # Return None for all keys on error
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, doc: Document) -> DocumentType:
        """Convert SQLAlchemy Document to GraphQL DocumentType"""
        return DocumentType(
            id=str(doc.id),
            filename=doc.filename,
            file_path=doc.file_path,
            file_type=doc.file_type,
            file_size=doc.file_size,
            content_hash=doc.content_hash,
            upload_date=doc.upload_date,
            last_processed=doc.last_processed,
            processing_status=doc.processing_status,
            total_pages=doc.total_pages,
            total_chunks=doc.total_chunks or 0,
            metadata=doc.metadata,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            # Computed fields
            is_processed=doc.is_processed,
            detection_count=doc.detection_count,
            has_arbitration_clauses=len([d for d in doc.detections if d.confidence_score > 0.5]) > 0,
            average_confidence_score=self._calculate_avg_confidence(doc.detections),
            processing_time_ms=None  # Would need to be calculated from processing logs
        )
    
    def _calculate_avg_confidence(self, detections) -> float:
        """Calculate average confidence score from detections"""
        if not detections:
            return None
        
        total_score = sum(d.confidence_score for d in detections)
        return total_score / len(detections)


class DocumentsByUserDataLoader(DataLoader):
    """DataLoader for Documents by User ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, user_ids: List[int]) -> List[List[DocumentType]]:
        """Batch load documents by user IDs"""
        try:
            # This would need a user_id field in Document model
            # For now, return empty lists
            return [[] for _ in user_ids]
            
        except Exception as e:
            return [[] for _ in user_ids]


class ChunkDataLoader(DataLoader):
    """DataLoader for Chunk entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[ChunkType]:
        """Batch load chunks by IDs"""
        try:
            chunks = self.session.query(Chunk).filter(
                Chunk.id.in_(keys)
            ).all()
            
            chunk_map = {chunk.id: self._convert_to_graphql_type(chunk) for chunk in chunks}
            return [chunk_map.get(key) for key in keys]
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, chunk: Chunk) -> ChunkType:
        """Convert SQLAlchemy Chunk to GraphQL ChunkType"""
        return ChunkType(
            id=str(chunk.id),
            document_id=str(chunk.document_id),
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            content_length=chunk.content_length,
            chunk_hash=chunk.chunk_hash,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            embedding_model=chunk.embedding_model,
            similarity_threshold=chunk.similarity_threshold,
            created_at=chunk.created_at,
            updated_at=None,  # Not in the model
            # Computed fields
            has_embedding=chunk.embedding_vector is not None,
            has_detections=chunk.has_detections,
            embedding=chunk.get_embedding()  # Only if authorized
        )


class ChunksByDocumentDataLoader(DataLoader):
    """DataLoader for Chunks by Document ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, document_ids: List[int]) -> List[List[ChunkType]]:
        """Batch load chunks by document IDs"""
        try:
            chunks = self.session.query(Chunk).filter(
                Chunk.document_id.in_(document_ids)
            ).order_by(Chunk.chunk_index).all()
            
            # Group chunks by document_id
            chunks_by_doc = {}
            for chunk in chunks:
                if chunk.document_id not in chunks_by_doc:
                    chunks_by_doc[chunk.document_id] = []
                chunks_by_doc[chunk.document_id].append(
                    ChunkDataLoader(self.session)._convert_to_graphql_type(chunk)
                )
            
            return [chunks_by_doc.get(doc_id, []) for doc_id in document_ids]
            
        except Exception as e:
            return [[] for _ in document_ids]


class DocumentStatsDataLoader(DataLoader):
    """DataLoader for Document statistics"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Batch load document statistics"""
        try:
            # Calculate various document statistics
            total_docs = self.session.query(Document).count()
            processed_docs = self.session.query(Document).filter(
                Document.processing_status == 'completed'
            ).count()
            
            # Calculate documents with arbitration clauses
            docs_with_arbitration = self.session.query(Document).join(Detection).filter(
                Detection.confidence_score > 0.5
            ).distinct().count()
            
            stats = {
                'total_documents': total_docs,
                'processed_documents': processed_docs,
                'documents_with_arbitration': docs_with_arbitration,
                'processing_rate': processed_docs / total_docs if total_docs > 0 else 0,
                'average_processing_time': 0.0  # Would need processing time tracking
            }
            
            # Return same stats for all keys (stats are global)
            return [stats for _ in keys]
            
        except Exception as e:
            return [{} for _ in keys]


def create_document_loaders(session: Session) -> Dict[str, DataLoader]:
    """Create all document-related DataLoaders"""
    return {
        'document': DocumentDataLoader(session),
        'chunks_by_document': ChunksByDocumentDataLoader(session),
        'documents_by_user': DocumentsByUserDataLoader(session),
        'chunk': ChunkDataLoader(session),
        'document_stats': DocumentStatsDataLoader(session),
    }