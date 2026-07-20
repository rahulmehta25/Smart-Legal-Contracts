from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple

Base = declarative_base()

class ArbitrationClauseDB(Base):
    """Database model for arbitration clauses"""
    __tablename__ = 'arbitration_clauses'
    
    id = Column(Integer, primary_key=True)
    company_name = Column(String(200))
    industry = Column(String(100))
    document_type = Column(String(100))  # TOS, Employment, etc.
    clause_text = Column(Text)
    clause_summary = Column(Text)
    key_provisions = Column(JSON)  # Stored as JSON array
    enforceability_score = Column(Float)
    risk_score = Column(Float)
    jurisdiction = Column(String(100))
    date_added = Column(DateTime, default=datetime.utcnow)
    date_effective = Column(DateTime)
    vector_id = Column(String(100))  # Reference to vector store
    metadata = Column(JSON)

class VectorStore:
    """FAISS-based vector store for similarity search"""
    def __init__(self, dimension: int = 768):
        try:
            import faiss
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.id_map = {}  # Map FAISS indices to database IDs
            self.current_idx = 0
        except ImportError:
            print("FAISS not available. Using simple cosine similarity.")
            self.faiss_available = False
            self.dimension = dimension
            self.embeddings = []
            self.id_map = {}
        
    def add_clause(self, clause_id: str, embedding: np.ndarray):
        """Add clause embedding to vector store"""
        if hasattr(self, 'faiss_available') and not self.faiss_available:
            # Simple storage for when FAISS is not available
            self.embeddings.append(embedding)
            self.id_map[len(self.embeddings) - 1] = clause_id
        else:
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            self.index.add(embedding.reshape(1, -1))
            self.id_map[self.current_idx] = clause_id
            self.current_idx += 1
        
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar clauses"""
        if hasattr(self, 'faiss_available') and not self.faiss_available:
            # Simple cosine similarity search
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            similarities = []
            
            for i, stored_embedding in enumerate(self.embeddings):
                stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                similarity = np.dot(query_embedding.flatten(), stored_embedding.flatten())
                similarities.append((i, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = []
            for idx, similarity in similarities[:k]:
                if idx in self.id_map:
                    results.append((self.id_map[idx], float(similarity)))
            
            return results
        else:
            # Normalize query
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx in self.id_map:
                    results.append((self.id_map[idx], float(distance)))
            
            return results
    
    def save(self, filepath: str):
        """Save index to disk"""
        if hasattr(self, 'faiss_available') and not self.faiss_available:
            import pickle
            with open(f"{filepath}.simple", 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'id_map': self.id_map
                }, f)
        else:
            import faiss
            import pickle
            
            faiss.write_index(self.index, f"{filepath}.faiss")
            with open(f"{filepath}.map", 'wb') as f:
                pickle.dump(self.id_map, f)
    
    def load(self, filepath: str):
        """Load index from disk"""
        if hasattr(self, 'faiss_available') and not self.faiss_available:
            import pickle
            try:
                with open(f"{filepath}.simple", 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.id_map = data['id_map']
            except FileNotFoundError:
                print("No existing simple vector store found. Starting fresh.")
        else:
            import faiss
            import pickle
            
            try:
                self.index = faiss.read_index(f"{filepath}.faiss")
                with open(f"{filepath}.map", 'rb') as f:
                    self.id_map = pickle.load(f)
                self.current_idx = len(self.id_map)
            except FileNotFoundError:
                print("No existing FAISS index found. Starting fresh.")
