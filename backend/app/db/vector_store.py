"""
Vector Database Configuration for Arbitration Detection RAG System
Supports multiple vector stores: Chroma, FAISS, and in-memory storage
Optimized for embedding storage and similarity search
"""

import os
import json
import pickle
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: int
    document_id: int
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    section_title: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    store_type: str = "chroma"  # chroma, faiss, memory
    embedding_dimension: int = 1536  # OpenAI ada-002 dimension
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    index_file_path: str = "vector_index"
    max_results: int = 100
    similarity_threshold: float = 0.7
    enable_metadata_filtering: bool = True
    batch_size: int = 1000
    persist_directory: str = "./vector_db"


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.embedding_dimension = config.embedding_dimension
        
    @abstractmethod
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> bool:
        """Add embeddings with metadata"""
        pass
    
    @abstractmethod
    def search_similar(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs"""
        pass
    
    @abstractmethod
    def get_embedding_count(self) -> int:
        """Get total number of stored embeddings"""
        pass
    
    @abstractmethod
    def persist(self) -> bool:
        """Persist the index to disk"""
        pass
    
    @abstractmethod
    def load(self) -> bool:
        """Load the index from disk"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB with persistence
            self.client = chromadb.PersistentClient(
                path=config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="arbitration_chunks",
                metadata={"hnsw:space": config.similarity_metric}
            )
            
            logger.info(f"ChromaDB initialized with {self.get_embedding_count()} embeddings")
            
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
    
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> bool:
        """Add embeddings to ChromaDB"""
        try:
            # Convert metadata to strings for ChromaDB compatibility
            processed_metadatas = []
            for metadata in metadatas:
                processed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        processed_metadata[key] = json.dumps(value)
                    else:
                        processed_metadata[key] = str(value)
                processed_metadatas.append(processed_metadata)
            
            # Add to collection in batches
            batch_size = self.config.batch_size
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = processed_metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                # Extract documents from metadata for ChromaDB
                documents = [meta.get('content', '') for meta in batch_metadatas]
                
                self.collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=documents,
                    ids=batch_ids
                )
            
            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to ChromaDB: {e}")
            return False
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings in ChromaDB"""
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata and self.config.enable_metadata_filtering:
                where_clause = {}
                for key, value in filter_metadata.items():
                    if isinstance(value, (dict, list)):
                        where_clause[key] = json.dumps(value)
                    else:
                        where_clause[key] = str(value)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.config.max_results),
                where=where_clause
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'][0] else 0.0
                    
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1.0 - distance if self.config.similarity_metric == "cosine" else distance
                    
                    # Filter by similarity threshold
                    if similarity_score >= self.config.similarity_threshold:
                        # Parse metadata
                        parsed_metadata = {}
                        for key, value in metadata.items():
                            try:
                                if key in ['document_metadata', 'detection_metadata']:
                                    parsed_metadata[key] = json.loads(value)
                                else:
                                    parsed_metadata[key] = value
                            except:
                                parsed_metadata[key] = value
                        
                        search_result = SearchResult(
                            chunk_id=int(chunk_id),
                            document_id=int(parsed_metadata.get('document_id', 0)),
                            content=parsed_metadata.get('content', ''),
                            similarity_score=similarity_score,
                            metadata=parsed_metadata,
                            page_number=int(parsed_metadata.get('page_number')) if parsed_metadata.get('page_number') else None,
                            section_title=parsed_metadata.get('section_title')
                        )
                        search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings from ChromaDB"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings from ChromaDB: {e}")
            return False
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings in ChromaDB"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting embedding count from ChromaDB: {e}")
            return 0
    
    def persist(self) -> bool:
        """ChromaDB automatically persists"""
        return True
    
    def load(self) -> bool:
        """ChromaDB automatically loads"""
        return True


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation for high-performance similarity search"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        try:
            import faiss
            self.faiss = faiss
            
            # Initialize FAISS index based on similarity metric
            if config.similarity_metric == "cosine":
                # Normalize embeddings for cosine similarity
                self.index = faiss.IndexFlatIP(config.embedding_dimension)
                self.normalize_embeddings = True
            elif config.similarity_metric == "euclidean":
                self.index = faiss.IndexFlatL2(config.embedding_dimension)
                self.normalize_embeddings = False
            else:
                self.index = faiss.IndexFlatIP(config.embedding_dimension)
                self.normalize_embeddings = False
            
            # Metadata storage
            self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
            self.id_to_chunk_id: Dict[int, str] = {}
            self.chunk_id_to_id: Dict[str, int] = {}
            self.next_id = 0
            
            # Try to load existing index
            self.load()
            
            logger.info(f"FAISS initialized with {self.get_embedding_count()} embeddings")
            
        except ImportError:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")
    
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> bool:
        """Add embeddings to FAISS index"""
        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize if using cosine similarity
            if self.normalize_embeddings:
                self.faiss.normalize_L2(embeddings_array)
            
            # Add to index
            start_id = self.next_id
            self.index.add(embeddings_array)
            
            # Store metadata mapping
            for i, (chunk_id, metadata) in enumerate(zip(ids, metadatas)):
                internal_id = start_id + i
                self.id_to_metadata[internal_id] = metadata
                self.id_to_chunk_id[internal_id] = chunk_id
                self.chunk_id_to_id[chunk_id] = internal_id
            
            self.next_id += len(embeddings)
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {e}")
            return False
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings in FAISS"""
        try:
            # Convert to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Normalize if using cosine similarity
            if self.normalize_embeddings:
                self.faiss.normalize_L2(query_array)
            
            # Search with larger k to allow for filtering
            search_k = min(k * 5 if filter_metadata else k, self.config.max_results)
            distances, indices = self.index.search(query_array, search_k)
            
            search_results = []
            for i, (distance, internal_id) in enumerate(zip(distances[0], indices[0])):
                if internal_id == -1:  # No more results
                    break
                
                # Get metadata
                metadata = self.id_to_metadata.get(internal_id, {})
                chunk_id = self.id_to_chunk_id.get(internal_id, str(internal_id))
                
                # Apply metadata filtering
                if filter_metadata and self.config.enable_metadata_filtering:
                    if not self._matches_filter(metadata, filter_metadata):
                        continue
                
                # Convert distance to similarity score
                if self.config.similarity_metric == "cosine":
                    similarity_score = float(distance)  # FAISS IP returns similarity directly
                else:
                    # Convert L2 distance to similarity (inverse relationship)
                    similarity_score = 1.0 / (1.0 + float(distance))
                
                # Filter by similarity threshold
                if similarity_score >= self.config.similarity_threshold:
                    search_result = SearchResult(
                        chunk_id=int(chunk_id),
                        document_id=int(metadata.get('document_id', 0)),
                        content=metadata.get('content', ''),
                        similarity_score=similarity_score,
                        metadata=metadata,
                        page_number=metadata.get('page_number'),
                        section_title=metadata.get('section_title')
                    )
                    search_results.append(search_result)
                
                # Stop if we have enough results
                if len(search_results) >= k:
                    break
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings from FAISS (rebuild index without deleted items)"""
        try:
            # FAISS doesn't support individual deletion, so we need to rebuild
            # Remove from metadata mappings
            internal_ids_to_remove = []
            for chunk_id in ids:
                if chunk_id in self.chunk_id_to_id:
                    internal_id = self.chunk_id_to_id[chunk_id]
                    internal_ids_to_remove.append(internal_id)
                    del self.chunk_id_to_id[chunk_id]
                    del self.id_to_chunk_id[internal_id]
                    del self.id_to_metadata[internal_id]
            
            # Rebuild index if there are deletions
            if internal_ids_to_remove:
                logger.warning(f"FAISS index rebuild required for {len(internal_ids_to_remove)} deletions")
                # Note: For production, consider using IndexIDMap for easier deletion
            
            logger.info(f"Marked {len(ids)} embeddings for deletion from FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings from FAISS: {e}")
            return False
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings in FAISS"""
        return self.index.ntotal
    
    def persist(self) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            index_path = Path(self.config.persist_directory)
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.faiss.write_index(self.index, str(index_path / "faiss.index"))
            
            # Save metadata
            metadata_file = index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'id_to_metadata': self.id_to_metadata,
                    'id_to_chunk_id': self.id_to_chunk_id,
                    'chunk_id_to_id': self.chunk_id_to_id,
                    'next_id': self.next_id
                }, f)
            
            logger.info("FAISS index and metadata saved to disk")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False
    
    def load(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            index_path = Path(self.config.persist_directory)
            faiss_file = index_path / "faiss.index"
            metadata_file = index_path / "metadata.pkl"
            
            if faiss_file.exists() and metadata_file.exists():
                # Load FAISS index
                self.index = self.faiss.read_index(str(faiss_file))
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_to_metadata = metadata['id_to_metadata']
                    self.id_to_chunk_id = metadata['id_to_chunk_id']
                    self.chunk_id_to_id = metadata['chunk_id_to_id']
                    self.next_id = metadata['next_id']
                
                logger.info("FAISS index and metadata loaded from disk")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False


class MemoryVectorStore(VectorStore):
    """In-memory vector store for development and testing"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        logger.info("Memory vector store initialized")
    
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> bool:
        """Add embeddings to memory store"""
        try:
            for embedding, metadata, chunk_id in zip(embeddings, metadatas, ids):
                self.embeddings.append(np.array(embedding, dtype=np.float32))
                self.metadatas.append(metadata)
                self.ids.append(chunk_id)
            
            logger.info(f"Added {len(embeddings)} embeddings to memory store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to memory store: {e}")
            return False
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings in memory"""
        try:
            if not self.embeddings:
                return []
            
            query_array = np.array(query_embedding, dtype=np.float32)
            similarities = []
            
            # Calculate similarities
            for i, embedding in enumerate(self.embeddings):
                # Apply metadata filtering
                if filter_metadata and self.config.enable_metadata_filtering:
                    if not self._matches_filter(self.metadatas[i], filter_metadata):
                        continue
                
                # Calculate similarity based on metric
                if self.config.similarity_metric == "cosine":
                    similarity = np.dot(query_array, embedding) / (
                        np.linalg.norm(query_array) * np.linalg.norm(embedding)
                    )
                elif self.config.similarity_metric == "euclidean":
                    distance = np.linalg.norm(query_array - embedding)
                    similarity = 1.0 / (1.0 + distance)
                else:  # dot_product
                    similarity = np.dot(query_array, embedding)
                
                similarities.append((similarity, i))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            similarities = similarities[:k]
            
            # Convert to SearchResult objects
            search_results = []
            for similarity, idx in similarities:
                if similarity >= self.config.similarity_threshold:
                    metadata = self.metadatas[idx]
                    search_result = SearchResult(
                        chunk_id=int(self.ids[idx]),
                        document_id=int(metadata.get('document_id', 0)),
                        content=metadata.get('content', ''),
                        similarity_score=float(similarity),
                        metadata=metadata,
                        page_number=metadata.get('page_number'),
                        section_title=metadata.get('section_title')
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching memory store: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def delete_embeddings(self, ids: List[str]) -> bool:
        """Delete embeddings from memory store"""
        try:
            indices_to_remove = []
            for chunk_id in ids:
                try:
                    idx = self.ids.index(chunk_id)
                    indices_to_remove.append(idx)
                except ValueError:
                    continue
            
            # Remove in reverse order to maintain indices
            for idx in sorted(indices_to_remove, reverse=True):
                del self.embeddings[idx]
                del self.metadatas[idx]
                del self.ids[idx]
            
            logger.info(f"Deleted {len(indices_to_remove)} embeddings from memory store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings from memory store: {e}")
            return False
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings in memory"""
        return len(self.embeddings)
    
    def persist(self) -> bool:
        """Save memory store to disk"""
        try:
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            data = {
                'embeddings': [emb.tolist() for emb in self.embeddings],
                'metadatas': self.metadatas,
                'ids': self.ids
            }
            
            with open(persist_path / "memory_store.json", 'w') as f:
                json.dump(data, f)
            
            logger.info("Memory store saved to disk")
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory store: {e}")
            return False
    
    def load(self) -> bool:
        """Load memory store from disk"""
        try:
            persist_path = Path(self.config.persist_directory)
            store_file = persist_path / "memory_store.json"
            
            if store_file.exists():
                with open(store_file, 'r') as f:
                    data = json.load(f)
                
                self.embeddings = [np.array(emb, dtype=np.float32) for emb in data['embeddings']]
                self.metadatas = data['metadatas']
                self.ids = data['ids']
                
                logger.info("Memory store loaded from disk")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading memory store: {e}")
            return False


class VectorStoreManager:
    """Manager class for vector store operations"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.store = self._create_store()
    
    def _create_store(self) -> VectorStore:
        """Create vector store based on configuration"""
        if self.config.store_type == "chroma":
            return ChromaVectorStore(self.config)
        elif self.config.store_type == "faiss":
            return FAISSVectorStore(self.config)
        elif self.config.store_type == "memory":
            return MemoryVectorStore(self.config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.store_type}")
    
    def add_chunk_embeddings(
        self, 
        chunks_data: List[Dict[str, Any]]
    ) -> bool:
        """Add embeddings for document chunks"""
        try:
            embeddings = []
            metadatas = []
            ids = []
            
            for chunk_data in chunks_data:
                if 'embedding' in chunk_data and chunk_data['embedding']:
                    embeddings.append(chunk_data['embedding'])
                    
                    # Prepare metadata
                    metadata = {
                        'content': chunk_data.get('content', ''),
                        'document_id': chunk_data.get('document_id'),
                        'chunk_index': chunk_data.get('chunk_index'),
                        'page_number': chunk_data.get('page_number'),
                        'section_title': chunk_data.get('section_title'),
                        'content_length': chunk_data.get('content_length'),
                        'chunk_hash': chunk_data.get('chunk_hash')
                    }
                    metadatas.append(metadata)
                    ids.append(str(chunk_data.get('id')))
            
            if embeddings:
                return self.store.add_embeddings(embeddings, metadatas, ids)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunk embeddings: {e}")
            return False
    
    def search_relevant_chunks(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        document_id: Optional[int] = None,
        min_confidence: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for relevant chunks"""
        try:
            # Prepare metadata filter
            filter_metadata = {}
            if document_id:
                filter_metadata['document_id'] = str(document_id)
            
            # Override similarity threshold if min_confidence provided
            if min_confidence:
                original_threshold = self.config.similarity_threshold
                self.config.similarity_threshold = min_confidence
            
            results = self.store.search_similar(
                query_embedding=query_embedding,
                k=k,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            # Restore original threshold
            if min_confidence:
                self.config.similarity_threshold = original_threshold
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching relevant chunks: {e}")
            return []
    
    def delete_document_embeddings(self, document_id: int) -> bool:
        """Delete all embeddings for a document"""
        try:
            # For now, we need to search for all chunks of this document
            # In a production system, you might want to maintain a separate index
            logger.warning(f"Deleting embeddings for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document embeddings: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'store_type': self.config.store_type,
            'total_embeddings': self.store.get_embedding_count(),
            'embedding_dimension': self.config.embedding_dimension,
            'similarity_metric': self.config.similarity_metric,
            'similarity_threshold': self.config.similarity_threshold
        }
    
    def persist(self) -> bool:
        """Persist vector store to disk"""
        return self.store.persist()
    
    def load(self) -> bool:
        """Load vector store from disk"""
        return self.store.load()


# Utility functions
def create_vector_store_config(
    store_type: str = "chroma",
    embedding_dimension: int = 1536,
    persist_directory: str = "./vector_db"
) -> VectorStoreConfig:
    """Create a vector store configuration"""
    return VectorStoreConfig(
        store_type=store_type,
        embedding_dimension=embedding_dimension,
        persist_directory=persist_directory
    )


def get_default_vector_store() -> VectorStoreManager:
    """Get default vector store manager"""
    config = create_vector_store_config()
    return VectorStoreManager(config)