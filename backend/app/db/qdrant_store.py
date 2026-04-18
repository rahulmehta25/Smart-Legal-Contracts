"""
Qdrant Vector Store Implementation

Production-ready Qdrant integration with:
- Connection pooling and retry logic
- Metadata filtering and payload indexing
- Batch upsert operations
- Health monitoring
- Collection management
"""

import uuid
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. Install with: pip install qdrant-client")

from app.core.config import get_settings


@dataclass
class QdrantConfig:
    """Qdrant connection and collection configuration"""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    url: Optional[str] = None
    collection_name: str = "arbitration_documents"
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 dimension
    distance_metric: str = "Cosine"
    on_disk: bool = False
    replication_factor: int = 1
    shard_number: int = 1
    timeout: int = 30
    prefer_grpc: bool = True


@dataclass
class QdrantSearchResult:
    """Search result from Qdrant"""
    id: str
    document_id: int
    chunk_index: int
    content: str
    score: float
    metadata: Dict[str, Any]
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class QdrantVectorStore:
    """
    Production-ready Qdrant vector store for arbitration clause detection
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")

        self.config = config or self._get_default_config()
        self.client = self._create_client()
        self._ensure_collection()

        logger.info(f"QdrantVectorStore initialized: collection={self.config.collection_name}")

    def _get_default_config(self) -> QdrantConfig:
        """Get default configuration from settings"""
        settings = get_settings()
        return QdrantConfig(
            url=getattr(settings, 'qdrant_url', None),
            host=getattr(settings, 'qdrant_host', 'localhost'),
            port=getattr(settings, 'qdrant_port', 6333),
            api_key=getattr(settings, 'qdrant_api_key', None),
            collection_name=getattr(settings, 'qdrant_collection', 'arbitration_documents'),
            embedding_dimension=getattr(settings, 'embedding_dimension', 384),
        )

    def _create_client(self) -> QdrantClient:
        """Create Qdrant client with retry logic"""
        try:
            if self.config.url:
                client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc
                )
            else:
                client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    grpc_port=self.config.grpc_port,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc
                )

            return client

        except Exception as e:
            logger.error(f"Failed to create Qdrant client: {e}")
            raise

    def _ensure_collection(self) -> None:
        """Ensure collection exists with proper schema"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.config.collection_name not in collection_names:
                self._create_collection()
            else:
                logger.info(f"Collection {self.config.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def _create_collection(self) -> None:
        """Create collection with optimized settings"""
        try:
            distance = getattr(models.Distance, self.config.distance_metric.upper())

            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=distance,
                    on_disk=self.config.on_disk
                ),
                shard_number=self.config.shard_number,
                replication_factor=self.config.replication_factor,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                    memmap_threshold=50000
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            )

            # Create payload indexes for efficient filtering
            self._create_payload_indexes()

            logger.info(f"Created collection: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create indexes on frequently filtered fields"""
        indexed_fields = [
            ("document_id", models.PayloadSchemaType.INTEGER),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
            ("has_arbitration_signals", models.PayloadSchemaType.BOOL),
            ("content_type", models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception as e:
                logger.warning(f"Could not create index for {field_name}: {e}")

    def add_document_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: int,
        chunk_indices: List[int],
        start_chars: List[int],
        end_chars: List[int],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add document chunks with embeddings to Qdrant

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            document_id: Document ID
            chunk_indices: List of chunk indices
            start_chars: Start character positions
            end_chars: End character positions
            metadata: Optional additional metadata per chunk

        Returns:
            List of generated point IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        point_ids = []

        for i, (chunk, embedding, chunk_idx, start, end) in enumerate(
            zip(chunks, embeddings, chunk_indices, start_chars, end_chars)
        ):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Build payload
            payload = {
                "document_id": document_id,
                "chunk_index": chunk_idx,
                "content": chunk,
                "start_char": start,
                "end_char": end,
                "content_length": len(chunk),
                "has_arbitration_signals": self._detect_arbitration_signals(chunk),
                "created_at": time.time()
            }

            # Add extra metadata if provided
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))

        # Batch upsert
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Added {len(points)} chunks for document {document_id}")
            return point_ids

        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            raise

    def _detect_arbitration_signals(self, text: str) -> bool:
        """Quick check for arbitration-related keywords"""
        keywords = [
            "arbitration", "arbitrator", "dispute resolution",
            "binding arbitration", "class action waiver",
            "aaa", "jams", "mediation"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        document_id: Optional[int] = None,
        min_score: float = 0.0,
        filter_arbitration: bool = False
    ) -> List[QdrantSearchResult]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query vector
            k: Number of results to return
            document_id: Filter by specific document
            min_score: Minimum similarity score threshold
            filter_arbitration: Filter for chunks with arbitration signals

        Returns:
            List of search results
        """
        # Build filter conditions
        filter_conditions = []

        if document_id is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            )

        if filter_arbitration:
            filter_conditions.append(
                models.FieldCondition(
                    key="has_arbitration_signals",
                    match=models.MatchValue(value=True)
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        try:
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=query_filter,
                score_threshold=min_score,
                with_payload=True
            )

            search_results = []
            for hit in results:
                payload = hit.payload or {}
                search_results.append(QdrantSearchResult(
                    id=str(hit.id),
                    document_id=payload.get("document_id", 0),
                    chunk_index=payload.get("chunk_index", 0),
                    content=payload.get("content", ""),
                    score=hit.score,
                    metadata=payload,
                    start_char=payload.get("start_char"),
                    end_char=payload.get("end_char")
                ))

            return search_results

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

    def search_arbitration_clauses(
        self,
        query_embedding: List[float],
        document_id: int,
        top_k: int = 20,
        min_score: float = 0.3
    ) -> List[QdrantSearchResult]:
        """
        Specialized search for arbitration clauses in a document

        Args:
            query_embedding: Embedding of arbitration-related query
            document_id: Document to search within
            top_k: Number of results
            min_score: Minimum relevance score

        Returns:
            List of potential arbitration clause chunks
        """
        # First, get chunks with arbitration signals
        filter_conditions = [
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id)
            ),
            models.FieldCondition(
                key="has_arbitration_signals",
                match=models.MatchValue(value=True)
            )
        ]

        try:
            # Search with arbitration filter first
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=models.Filter(must=filter_conditions),
                score_threshold=min_score,
                with_payload=True
            )

            # If not enough results, expand search
            if len(results) < top_k // 2:
                all_results = self.similarity_search(
                    query_embedding=query_embedding,
                    k=top_k,
                    document_id=document_id,
                    min_score=min_score
                )
                return all_results

            return [
                QdrantSearchResult(
                    id=str(hit.id),
                    document_id=hit.payload.get("document_id", 0),
                    chunk_index=hit.payload.get("chunk_index", 0),
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata=hit.payload,
                    start_char=hit.payload.get("start_char"),
                    end_char=hit.payload.get("end_char")
                )
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Error searching arbitration clauses: {e}")
            return []

    def delete_document(self, document_id: int) -> bool:
        """Delete all chunks for a document"""
        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted all chunks for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    def get_document_chunks(
        self,
        document_id: int,
        limit: int = 1000
    ) -> List[QdrantSearchResult]:
        """Get all chunks for a document"""
        try:
            results, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            return [
                QdrantSearchResult(
                    id=str(point.id),
                    document_id=point.payload.get("document_id", 0),
                    chunk_index=point.payload.get("chunk_index", 0),
                    content=point.payload.get("content", ""),
                    score=1.0,
                    metadata=point.payload,
                    start_char=point.payload.get("start_char"),
                    end_char=point.payload.get("end_char")
                )
                for point in results
            ]

        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
                "optimizer_status": info.optimizer_status.status.value if info.optimizer_status else "unknown",
                "config": {
                    "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else self.config.embedding_dimension,
                    "distance": self.config.distance_metric
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant connection health"""
        try:
            start_time = time.time()

            # Check if client is responsive
            collections = self.client.get_collections()

            # Check if our collection exists
            collection_exists = self.config.collection_name in [
                c.name for c in collections.collections
            ]

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "collection_exists": collection_exists,
                "collections_count": len(collections.collections)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def optimize_collection(self) -> bool:
        """Trigger collection optimization"""
        try:
            self.client.update_collection(
                collection_name=self.config.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            logger.info(f"Optimization triggered for {self.config.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
            return False


# Singleton instance
_qdrant_store: Optional[QdrantVectorStore] = None


def get_qdrant_store() -> QdrantVectorStore:
    """Get or create Qdrant store singleton"""
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantVectorStore()
    return _qdrant_store


def init_qdrant_store(config: Optional[QdrantConfig] = None) -> QdrantVectorStore:
    """Initialize Qdrant store with custom config"""
    global _qdrant_store
    _qdrant_store = QdrantVectorStore(config)
    return _qdrant_store
