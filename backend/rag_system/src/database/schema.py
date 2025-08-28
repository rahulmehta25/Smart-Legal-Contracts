"""Database schema and vector store for arbitration clauses."""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

Base = declarative_base()

class ArbitrationClauseDB(Base):
    """Database model for arbitration clauses."""
    __tablename__ = 'arbitration_clauses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(200), index=True)
    industry = Column(String(100), index=True)
    document_type = Column(String(100), index=True)  # TOS, Employment, etc.
    clause_text = Column(Text)
    clause_summary = Column(Text)
    key_provisions = Column(JSON)  # Stored as JSON array
    enforceability_score = Column(Float)
    risk_score = Column(Float)
    jurisdiction = Column(String(100), index=True)
    date_added = Column(DateTime, default=datetime.utcnow)
    date_effective = Column(DateTime)
    vector_id = Column(String(100), unique=True)  # Reference to vector store
    metadata = Column(JSON)
    
    # Add composite index for common queries
    __table_args__ = (
        Index('idx_company_industry', 'company_name', 'industry'),
        Index('idx_document_jurisdiction', 'document_type', 'jurisdiction'),
        Index('idx_risk_enforceability', 'risk_score', 'enforceability_score'),
    )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'company_name': self.company_name,
            'industry': self.industry,
            'document_type': self.document_type,
            'clause_text': self.clause_text,
            'clause_summary': self.clause_summary,
            'key_provisions': self.key_provisions,
            'enforceability_score': self.enforceability_score,
            'risk_score': self.risk_score,
            'jurisdiction': self.jurisdiction,
            'date_added': self.date_added.isoformat() if self.date_added else None,
            'date_effective': self.date_effective.isoformat() if self.date_effective else None,
            'vector_id': self.vector_id,
            'metadata': self.metadata
        }

class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, dimension: int = 768):
        """Initialize vector store."""
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
        
        self.dimension = dimension
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = self.faiss.IndexFlatIP(dimension)
        self.id_map = {}  # Map FAISS indices to database IDs
        self.current_idx = 0
        
        # Initialize storage directory
        self.storage_dir = os.path.join(
            os.path.dirname(__file__), 
            '../../data/vectors'
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Try to load existing index
        self._load_if_exists()
    
    def add_clause(self, clause_id: str, embedding: np.ndarray) -> bool:
        """
        Add clause embedding to vector store.
        
        Args:
            clause_id: Unique identifier for the clause
            embedding: Numpy array of the embedding
            
        Returns:
            Success status
        """
        try:
            # Ensure embedding is the right shape
            if embedding.shape[0] != self.dimension:
                logger.error(f"Embedding dimension mismatch: {embedding.shape[0]} != {self.dimension}")
                return False
            
            # Normalize for cosine similarity
            embedding = embedding.astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to index
            self.index.add(embedding.reshape(1, -1))
            self.id_map[self.current_idx] = clause_id
            self.current_idx += 1
            
            # Auto-save periodically
            if self.current_idx % 100 == 0:
                self.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar clauses.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of (clause_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Normalize query
            query_embedding = query_embedding.astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Ensure k doesn't exceed available vectors
            k = min(k, self.index.ntotal)
            
            # Search
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx in self.id_map:
                    # Convert distance to similarity score (0-1 range)
                    similarity = float(distance)  # Inner product is already similarity
                    results.append((self.id_map[idx], similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def remove_clause(self, clause_id: str) -> bool:
        """
        Remove a clause from the vector store.
        
        Note: FAISS doesn't support direct removal, so we rebuild the index
        """
        try:
            # Find the index to remove
            idx_to_remove = None
            for idx, cid in self.id_map.items():
                if cid == clause_id:
                    idx_to_remove = idx
                    break
            
            if idx_to_remove is None:
                logger.warning(f"Clause {clause_id} not found in vector store")
                return False
            
            # Remove from mapping
            del self.id_map[idx_to_remove]
            
            # Note: In production, you'd want to rebuild the index periodically
            # rather than on every removal
            logger.info(f"Removed clause {clause_id} from mapping. Consider rebuilding index.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing from vector store: {e}")
            return False
    
    def save(self, filepath: Optional[str] = None):
        """Save index to disk."""
        try:
            import pickle
            
            if filepath is None:
                filepath = os.path.join(self.storage_dir, 'clause_vectors')
            
            # Save FAISS index
            self.faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save ID mapping
            with open(f"{filepath}.map", 'wb') as f:
                pickle.dump({
                    'id_map': self.id_map,
                    'current_idx': self.current_idx,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Vector store saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self, filepath: Optional[str] = None):
        """Load index from disk."""
        try:
            import pickle
            
            if filepath is None:
                filepath = os.path.join(self.storage_dir, 'clause_vectors')
            
            # Load FAISS index
            self.index = self.faiss.read_index(f"{filepath}.faiss")
            
            # Load ID mapping
            with open(f"{filepath}.map", 'rb') as f:
                data = pickle.load(f)
                self.id_map = data['id_map']
                self.current_idx = data['current_idx']
                self.dimension = data['dimension']
            
            logger.info(f"Vector store loaded from {filepath}. Contains {self.index.ntotal} vectors.")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
    
    def _load_if_exists(self):
        """Load existing index if available."""
        default_path = os.path.join(self.storage_dir, 'clause_vectors')
        if os.path.exists(f"{default_path}.faiss") and os.path.exists(f"{default_path}.map"):
            self.load()
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'mapped_ids': len(self.id_map),
            'current_index': self.current_idx
        }
    
    def clear(self):
        """Clear the vector store."""
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.id_map = {}
        self.current_idx = 0
        logger.info("Vector store cleared")

class DatabaseManager:
    """Manager for database operations."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize database manager."""
        if db_url is None:
            # Use SQLite for development/testing
            db_path = os.path.join(
                os.path.dirname(__file__),
                '../../data/arbitration.db'
            )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {db_url}")
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def add_clause(self, clause_data: Dict) -> int:
        """Add a clause to the database."""
        session = self.get_session()
        try:
            clause = ArbitrationClauseDB(**clause_data)
            session.add(clause)
            session.commit()
            clause_id = clause.id
            logger.info(f"Added clause {clause_id} to database")
            return clause_id
        except Exception as e:
            logger.error(f"Error adding clause to database: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_clause(self, clause_id: int) -> Optional[Dict]:
        """Get a clause by ID."""
        session = self.get_session()
        try:
            clause = session.query(ArbitrationClauseDB).filter_by(id=clause_id).first()
            return clause.to_dict() if clause else None
        finally:
            session.close()
    
    def search_clauses(self, filters: Dict) -> List[Dict]:
        """Search clauses with filters."""
        session = self.get_session()
        try:
            query = session.query(ArbitrationClauseDB)
            
            # Apply filters
            if 'company_name' in filters:
                query = query.filter(ArbitrationClauseDB.company_name.like(f"%{filters['company_name']}%"))
            if 'industry' in filters:
                query = query.filter_by(industry=filters['industry'])
            if 'document_type' in filters:
                query = query.filter_by(document_type=filters['document_type'])
            if 'jurisdiction' in filters:
                query = query.filter_by(jurisdiction=filters['jurisdiction'])
            if 'min_risk' in filters:
                query = query.filter(ArbitrationClauseDB.risk_score >= filters['min_risk'])
            if 'max_risk' in filters:
                query = query.filter(ArbitrationClauseDB.risk_score <= filters['max_risk'])
            
            # Limit results
            query = query.limit(100)
            
            clauses = query.all()
            return [clause.to_dict() for clause in clauses]
            
        finally:
            session.close()