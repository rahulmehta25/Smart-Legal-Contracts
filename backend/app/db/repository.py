"""
Repository Layer for Arbitration Detection RAG System
Optimized database operations with connection pooling and query optimization
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy import create_engine, func, and_, or_, desc, asc, event
from sqlalchemy.orm import sessionmaker, Session, selectinload, joinedload
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text
from contextlib import contextmanager
import hashlib
import json

from .models import (
    Base, Document, Chunk, Detection, Pattern, QueryCache, 
    DatabaseOptimizer
)
from .vector_store import VectorStoreManager, VectorStoreConfig

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration with optimization settings"""
    
    def __init__(
        self,
        database_url: str = "sqlite:///arbitration_rag.db",
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
        enable_wal_mode: bool = True,
        cache_size: int = 10000
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        self.enable_wal_mode = enable_wal_mode
        self.cache_size = cache_size


class ArbitrationRepository:
    """Repository class with optimized database operations"""
    
    def __init__(self, config: DatabaseConfig, vector_store_config: VectorStoreConfig):
        self.config = config
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.vector_store = VectorStoreManager(vector_store_config)
        
        # Initialize database
        self._initialize_database()
        
        logger.info("ArbitrationRepository initialized with optimized settings")
    
    def _create_engine(self):
        """Create SQLAlchemy engine with optimized pool settings"""
        if self.config.database_url.startswith('sqlite'):
            # SQLite-specific optimizations
            engine = create_engine(
                self.config.database_url,
                echo=self.config.echo,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                }
            )
            
            # Apply SQLite optimizations
            if self.config.enable_wal_mode:
                @event.listens_for(engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    # Enable WAL mode for better concurrency
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute(f"PRAGMA cache_size={self.config.cache_size}")
                    cursor.execute("PRAGMA temp_store=memory")
                    cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                    cursor.close()
        else:
            # PostgreSQL/MySQL optimizations
            engine = create_engine(
                self.config.database_url,
                echo=self.config.echo,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle
            )
        
        return engine
    
    def _initialize_database(self):
        """Initialize database tables and load vector store"""
        Base.metadata.create_all(bind=self.engine)
        self.vector_store.load()
        logger.info("Database initialized and vector store loaded")
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    # Document operations
    def create_document(
        self,
        filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """Create a new document with optimized bulk insert"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(f"{filename}{file_size}".encode()).hexdigest()
            
            with self.get_session() as session:
                # Check for existing document
                existing = session.query(Document).filter(
                    Document.content_hash == content_hash
                ).first()
                
                if existing:
                    logger.info(f"Document with hash {content_hash} already exists")
                    return existing
                
                document = Document(
                    filename=filename,
                    file_path=file_path,
                    file_type=file_type,
                    file_size=file_size,
                    content_hash=content_hash,
                    metadata=metadata or {}
                )
                
                session.add(document)
                session.flush()  # Get the ID without committing
                
                logger.info(f"Created document: {filename} (ID: {document.id})")
                return document
                
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            return None
    
    def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Get document by ID with optimized loading"""
        try:
            with self.get_session() as session:
                document = session.query(Document).options(
                    selectinload(Document.chunks),
                    selectinload(Document.detections)
                ).filter(Document.id == document_id).first()
                
                return document
                
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None
    
    def get_documents_by_status(
        self, 
        status: str, 
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """Get documents by processing status with pagination"""
        try:
            with self.get_session() as session:
                documents = session.query(Document).filter(
                    Document.processing_status == status
                ).order_by(Document.upload_date.desc()).offset(offset).limit(limit).all()
                
                return documents
                
        except Exception as e:
            logger.error(f"Error getting documents by status {status}: {e}")
            return []
    
    def update_document_status(
        self, 
        document_id: int, 
        status: str,
        total_pages: Optional[int] = None
    ) -> bool:
        """Update document processing status"""
        try:
            with self.get_session() as session:
                document = session.query(Document).filter(
                    Document.id == document_id
                ).first()
                
                if document:
                    document.processing_status = status
                    document.last_processed = datetime.utcnow()
                    if total_pages:
                        document.total_pages = total_pages
                    
                    logger.info(f"Updated document {document_id} status to {status}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            return False
    
    # Chunk operations with batch processing
    def create_chunks_batch(
        self, 
        chunks_data: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """Create multiple chunks in a single transaction for better performance"""
        try:
            with self.get_session() as session:
                chunks = []
                embeddings_data = []
                
                for chunk_data in chunks_data:
                    # Create chunk object
                    chunk = Chunk(
                        document_id=chunk_data['document_id'],
                        chunk_index=chunk_data['chunk_index'],
                        content=chunk_data['content'],
                        page_number=chunk_data.get('page_number'),
                        section_title=chunk_data.get('section_title'),
                        embedding_model=chunk_data.get('embedding_model', 'text-embedding-ada-002')
                    )
                    
                    # Set embedding if provided
                    if 'embedding' in chunk_data:
                        chunk.set_embedding(chunk_data['embedding'])
                    
                    chunks.append(chunk)
                
                # Bulk insert chunks
                session.bulk_save_objects(chunks, return_defaults=True)
                session.flush()
                
                # Prepare data for vector store
                for i, chunk in enumerate(chunks):
                    if 'embedding' in chunks_data[i]:
                        embedding_data = {
                            'id': chunk.id,
                            'document_id': chunk.document_id,
                            'chunk_index': chunk.chunk_index,
                            'content': chunk.content,
                            'page_number': chunk.page_number,
                            'section_title': chunk.section_title,
                            'content_length': chunk.content_length,
                            'chunk_hash': chunk.chunk_hash,
                            'embedding': chunks_data[i]['embedding']
                        }
                        embeddings_data.append(embedding_data)
                
                # Add embeddings to vector store
                if embeddings_data:
                    self.vector_store.add_chunk_embeddings(embeddings_data)
                
                logger.info(f"Created {len(chunks)} chunks with embeddings")
                return chunks
                
        except Exception as e:
            logger.error(f"Error creating chunks batch: {e}")
            return []
    
    def get_chunks_by_document(
        self, 
        document_id: int,
        include_embeddings: bool = False
    ) -> List[Chunk]:
        """Get all chunks for a document with optional embedding loading"""
        try:
            with self.get_session() as session:
                query = session.query(Chunk).filter(
                    Chunk.document_id == document_id
                ).order_by(Chunk.chunk_index)
                
                if include_embeddings:
                    # Only load chunks that have embeddings
                    query = query.filter(Chunk.embedding_vector.isnot(None))
                
                chunks = query.all()
                return chunks
                
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
    
    def search_similar_chunks(
        self,
        query_embedding: List[float],
        k: int = 10,
        document_id: Optional[int] = None,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector store"""
        try:
            # Search vector store
            search_results = self.vector_store.search_relevant_chunks(
                query_embedding=query_embedding,
                k=k,
                document_id=document_id,
                min_confidence=min_similarity
            )
            
            if not search_results:
                return []
            
            # Get chunk IDs
            chunk_ids = [result.chunk_id for result in search_results]
            
            # Fetch full chunk data from database
            with self.get_session() as session:
                chunks = session.query(Chunk).options(
                    joinedload(Chunk.document),
                    selectinload(Chunk.detections)
                ).filter(Chunk.id.in_(chunk_ids)).all()
                
                # Combine with similarity scores
                chunk_dict = {chunk.id: chunk for chunk in chunks}
                results = []
                
                for search_result in search_results:
                    if search_result.chunk_id in chunk_dict:
                        chunk = chunk_dict[search_result.chunk_id]
                        result_data = {
                            'chunk': chunk,
                            'similarity_score': search_result.similarity_score,
                            'metadata': search_result.metadata
                        }
                        results.append(result_data)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    # Detection operations
    def create_detection(
        self,
        chunk_id: int,
        document_id: int,
        detection_type: str,
        confidence_score: float,
        matched_text: str,
        detection_method: str,
        pattern_id: Optional[int] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        page_number: Optional[int] = None,
        model_version: Optional[str] = None
    ) -> Optional[Detection]:
        """Create a new detection with validation"""
        try:
            with self.get_session() as session:
                detection = Detection(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    detection_type=detection_type,
                    confidence_score=confidence_score,
                    matched_text=matched_text,
                    detection_method=detection_method,
                    pattern_id=pattern_id,
                    context_before=context_before,
                    context_after=context_after,
                    start_position=start_position,
                    end_position=end_position,
                    page_number=page_number,
                    model_version=model_version
                )
                
                session.add(detection)
                session.flush()
                
                logger.info(f"Created detection: {detection_type} (confidence: {confidence_score})")
                return detection
                
        except Exception as e:
            logger.error(f"Error creating detection: {e}")
            return None
    
    def create_detections_batch(
        self, 
        detections_data: List[Dict[str, Any]]
    ) -> List[Detection]:
        """Create multiple detections in batch for better performance"""
        try:
            with self.get_session() as session:
                detections = []
                
                for detection_data in detections_data:
                    detection = Detection(**detection_data)
                    detections.append(detection)
                
                # Bulk insert detections
                session.bulk_save_objects(detections, return_defaults=True)
                
                logger.info(f"Created {len(detections)} detections in batch")
                return detections
                
        except Exception as e:
            logger.error(f"Error creating detections batch: {e}")
            return []
    
    def get_detections_by_document(
        self,
        document_id: int,
        detection_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[Detection]:
        """Get detections for a document with filtering"""
        try:
            with self.get_session() as session:
                query = session.query(Detection).options(
                    joinedload(Detection.chunk),
                    joinedload(Detection.pattern)
                ).filter(Detection.document_id == document_id)
                
                if detection_type:
                    query = query.filter(Detection.detection_type == detection_type)
                
                if min_confidence > 0:
                    query = query.filter(Detection.confidence_score >= min_confidence)
                
                detections = query.order_by(
                    Detection.confidence_score.desc()
                ).limit(limit).all()
                
                return detections
                
        except Exception as e:
            logger.error(f"Error getting detections for document {document_id}: {e}")
            return []
    
    def get_high_confidence_detections(
        self,
        min_confidence: float = 0.8,
        detection_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Detection]:
        """Get high confidence detections across all documents"""
        try:
            with self.get_session() as session:
                query = session.query(Detection).options(
                    joinedload(Detection.chunk),
                    joinedload(Detection.document),
                    joinedload(Detection.pattern)
                ).filter(Detection.confidence_score >= min_confidence)
                
                if detection_type:
                    query = query.filter(Detection.detection_type == detection_type)
                
                detections = query.order_by(
                    Detection.confidence_score.desc()
                ).offset(offset).limit(limit).all()
                
                return detections
                
        except Exception as e:
            logger.error(f"Error getting high confidence detections: {e}")
            return []
    
    def update_detection_validation(
        self,
        detection_id: int,
        is_validated: bool,
        validation_score: Optional[float] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Update detection validation status"""
        try:
            with self.get_session() as session:
                detection = session.query(Detection).filter(
                    Detection.id == detection_id
                ).first()
                
                if detection:
                    detection.is_validated = is_validated
                    detection.validation_score = validation_score
                    detection.notes = notes
                    
                    logger.info(f"Updated detection {detection_id} validation")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating detection validation: {e}")
            return False
    
    # Pattern operations
    def create_pattern(
        self,
        pattern_name: str,
        pattern_text: str,
        pattern_type: str,
        category: str,
        language: str = 'en',
        effectiveness_score: float = 0.5
    ) -> Optional[Pattern]:
        """Create a new pattern"""
        try:
            with self.get_session() as session:
                pattern = Pattern(
                    pattern_name=pattern_name,
                    pattern_text=pattern_text,
                    pattern_type=pattern_type,
                    category=category,
                    language=language,
                    effectiveness_score=effectiveness_score
                )
                
                session.add(pattern)
                session.flush()
                
                logger.info(f"Created pattern: {pattern_name}")
                return pattern
                
        except Exception as e:
            logger.error(f"Error creating pattern: {e}")
            return None
    
    def get_active_patterns(
        self,
        pattern_type: Optional[str] = None,
        category: Optional[str] = None,
        min_effectiveness: float = 0.0
    ) -> List[Pattern]:
        """Get active patterns with filtering"""
        try:
            with self.get_session() as session:
                query = session.query(Pattern).filter(
                    Pattern.is_active == True,
                    Pattern.effectiveness_score >= min_effectiveness
                )
                
                if pattern_type:
                    query = query.filter(Pattern.pattern_type == pattern_type)
                
                if category:
                    query = query.filter(Pattern.category == category)
                
                patterns = query.order_by(
                    Pattern.effectiveness_score.desc(),
                    Pattern.usage_count.desc()
                ).all()
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error getting active patterns: {e}")
            return []
    
    def update_pattern_effectiveness(
        self,
        pattern_id: int,
        effectiveness_score: float
    ) -> bool:
        """Update pattern effectiveness score"""
        try:
            with self.get_session() as session:
                pattern = session.query(Pattern).filter(
                    Pattern.id == pattern_id
                ).first()
                
                if pattern:
                    pattern.effectiveness_score = effectiveness_score
                    
                    logger.info(f"Updated pattern {pattern_id} effectiveness to {effectiveness_score}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating pattern effectiveness: {e}")
            return False
    
    # Caching operations
    def get_cached_query(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        try:
            with self.get_session() as session:
                cache_entry = session.query(QueryCache).filter(
                    QueryCache.cache_key == cache_key,
                    QueryCache.expires_at > func.now()
                ).first()
                
                if cache_entry:
                    cache_entry.increment_access()
                    return cache_entry.result_data
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached query: {e}")
            return None
    
    def set_cached_query(
        self,
        cache_key: str,
        result_data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Set cached query result"""
        try:
            with self.get_session() as session:
                # Check if cache entry exists
                existing = session.query(QueryCache).filter(
                    QueryCache.cache_key == cache_key
                ).first()
                
                if existing:
                    # Update existing entry
                    existing.result_data = result_data
                    existing.ttl = ttl
                    existing.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                    existing.increment_access()
                else:
                    # Create new entry
                    query_hash = hashlib.md5(json.dumps(result_data, sort_keys=True).encode()).hexdigest()
                    cache_entry = QueryCache(
                        cache_key=cache_key,
                        query_hash=query_hash,
                        result_data=result_data,
                        result_count=len(result_data) if isinstance(result_data, list) else 1,
                        ttl=ttl
                    )
                    session.add(cache_entry)
                
                return True
                
        except Exception as e:
            logger.error(f"Error setting cached query: {e}")
            return False
    
    # Analytics and statistics
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        try:
            with self.get_session() as session:
                stats = DatabaseOptimizer.get_table_stats(session)
                
                # Additional statistics
                total_size = session.query(func.sum(Document.file_size)).scalar() or 0
                avg_chunks_per_doc = session.query(func.avg(Document.total_chunks)).scalar() or 0
                
                stats['storage'] = {
                    'total_file_size_bytes': int(total_size),
                    'total_file_size_mb': round(total_size / (1024 * 1024), 2),
                    'avg_chunks_per_document': round(float(avg_chunks_per_doc), 2)
                }
                
                # Vector store statistics
                stats['vector_store'] = self.vector_store.get_stats()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {}
    
    def get_detection_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detection analytics for a time period"""
        try:
            with self.get_session() as session:
                query = session.query(Detection)
                
                if start_date:
                    query = query.filter(Detection.created_at >= start_date)
                if end_date:
                    query = query.filter(Detection.created_at <= end_date)
                
                # Detection type distribution
                type_stats = session.query(
                    Detection.detection_type,
                    func.count(Detection.id).label('count'),
                    func.avg(Detection.confidence_score).label('avg_confidence')
                ).filter(
                    Detection.created_at >= start_date if start_date else True,
                    Detection.created_at <= end_date if end_date else True
                ).group_by(Detection.detection_type).all()
                
                # Confidence distribution
                confidence_ranges = {
                    'high_confidence': query.filter(Detection.confidence_score >= 0.8).count(),
                    'medium_confidence': query.filter(
                        and_(Detection.confidence_score >= 0.5, Detection.confidence_score < 0.8)
                    ).count(),
                    'low_confidence': query.filter(Detection.confidence_score < 0.5).count()
                }
                
                return {
                    'type_distribution': [
                        {
                            'type': stat.detection_type,
                            'count': stat.count,
                            'avg_confidence': round(float(stat.avg_confidence), 3)
                        }
                        for stat in type_stats
                    ],
                    'confidence_distribution': confidence_ranges
                }
                
        except Exception as e:
            logger.error(f"Error getting detection analytics: {e}")
            return {}
    
    # Maintenance operations
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        try:
            with self.get_session() as session:
                expired_count = DatabaseOptimizer.cleanup_expired_cache(session)
                logger.info(f"Cleaned up {expired_count} expired cache entries")
                return expired_count
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def optimize_database(self) -> bool:
        """Run database optimization operations"""
        try:
            with self.get_session() as session:
                # SQLite-specific optimizations
                if self.config.database_url.startswith('sqlite'):
                    session.execute(text("VACUUM"))
                    session.execute(text("ANALYZE"))
                    
                # Clean up expired cache
                self.cleanup_expired_cache()
                
                # Persist vector store
                self.vector_store.persist()
                
                logger.info("Database optimization completed")
                return True
                
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False
    
    def close(self):
        """Clean up resources"""
        try:
            self.engine.dispose()
            self.vector_store.persist()
            logger.info("Repository closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing repository: {e}")


# Factory function
def create_repository(
    database_url: str = "sqlite:///arbitration_rag.db",
    vector_store_type: str = "chroma",
    enable_cache: bool = True
) -> ArbitrationRepository:
    """Factory function to create repository with default settings"""
    
    db_config = DatabaseConfig(
        database_url=database_url,
        pool_size=20,
        max_overflow=30,
        enable_wal_mode=True,
        cache_size=10000
    )
    
    vector_config = VectorStoreConfig(
        store_type=vector_store_type,
        embedding_dimension=1536,
        similarity_threshold=0.7,
        persist_directory="./vector_db"
    )
    
    return ArbitrationRepository(db_config, vector_config)