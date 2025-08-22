"""
GraphQL Query Resolvers
"""

import strawberry
from typing import List, Optional, Union
from strawberry.types import Info
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from ..types import (
    Document, DocumentConnection, DocumentFilter,
    Chunk, ChunkConnection, ChunkFilter,
    Detection, DetectionConnection, DetectionFilter,
    Pattern, PatternConnection,
    ArbitrationAnalysis, AnalysisConnection, AnalysisFilter,
    User, UserConnection, UserFilter,
    Comment, CommentConnection,
    Annotation,
    SystemStats, DocumentStats, DetectionStats,
    Node
)
from ..dataloaders import create_document_loaders, create_analysis_loaders, create_user_loaders, create_pattern_loaders
from ...db.database import get_session
from ...db.models import Document as DocumentModel, Chunk as ChunkModel, Detection as DetectionModel, Pattern as PatternModel
from ...models.analysis import ArbitrationAnalysis as AnalysisModel
from ...models.user import User as UserModel
from ..utils.auth import get_current_user, require_auth
from ..utils.pagination import create_connection, encode_cursor, decode_cursor
from ..utils.filtering import apply_document_filter, apply_detection_filter, apply_analysis_filter


@strawberry.type
class Query:
    """Root Query type"""
    
    @strawberry.field
    async def node(self, info: Info, id: strawberry.ID) -> Optional[Node]:
        """Fetch any node by ID (Relay Global Object Identification)"""
        try:
            # Parse the global ID to get type and local ID
            node_type, local_id = self._parse_global_id(id)
            session = info.context["session"]
            loaders = info.context["loaders"]
            
            if node_type == "Document":
                return await loaders["document"].load(int(local_id))
            elif node_type == "Chunk":
                return await loaders["chunk"].load(int(local_id))
            elif node_type == "Detection":
                return await loaders["detection"].load(int(local_id))
            elif node_type == "Analysis":
                return await loaders["analysis"].load(int(local_id))
            elif node_type == "Pattern":
                return await loaders["pattern"].load(int(local_id))
            elif node_type == "User":
                return await loaders["user"].load(int(local_id))
            
            return None
            
        except Exception as e:
            return None
    
    @strawberry.field
    async def document(self, info: Info, id: strawberry.ID) -> Optional[Document]:
        """Get a single document by ID"""
        try:
            loaders = info.context["loaders"]
            return await loaders["document"].load(int(id))
        except Exception as e:
            return None
    
    @strawberry.field
    async def documents(
        self,
        info: Info,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[DocumentFilter] = None,
        order_by: str = "upload_date",
        order_direction: str = "DESC"
    ) -> DocumentConnection:
        """Get paginated list of documents"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            # Build base query
            query = session.query(DocumentModel)
            
            # Apply filters
            if filter:
                query = apply_document_filter(query, filter)
            
            # Apply user-specific filtering if not admin
            if current_user and current_user.role != "ADMIN":
                # Add user-specific filtering here
                pass
            
            # Apply ordering
            if order_by == "upload_date":
                if order_direction == "DESC":
                    query = query.order_by(desc(DocumentModel.upload_date))
                else:
                    query = query.order_by(asc(DocumentModel.upload_date))
            elif order_by == "filename":
                if order_direction == "DESC":
                    query = query.order_by(desc(DocumentModel.filename))
                else:
                    query = query.order_by(asc(DocumentModel.filename))
            
            # Apply pagination
            return await create_connection(
                query=query,
                first=first,
                after=after,
                loader=info.context["loaders"]["document"],
                cursor_field="id"
            )
            
        except Exception as e:
            # Return empty connection on error
            return DocumentConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def chunk(self, info: Info, id: strawberry.ID) -> Optional[Chunk]:
        """Get a single chunk by ID"""
        try:
            loaders = info.context["loaders"]
            return await loaders["chunk"].load(int(id))
        except Exception as e:
            return None
    
    @strawberry.field
    async def chunks(
        self,
        info: Info,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[ChunkFilter] = None
    ) -> ChunkConnection:
        """Get paginated list of chunks"""
        try:
            session = info.context["session"]
            
            query = session.query(ChunkModel)
            
            # Apply filters
            if filter:
                if filter.page_number is not None:
                    query = query.filter(ChunkModel.page_number == filter.page_number)
                if filter.has_embedding is not None:
                    if filter.has_embedding:
                        query = query.filter(ChunkModel.embedding_vector.isnot(None))
                    else:
                        query = query.filter(ChunkModel.embedding_vector.is_(None))
                if filter.content_length:
                    if filter.content_length.min is not None:
                        query = query.filter(ChunkModel.content_length >= filter.content_length.min)
                    if filter.content_length.max is not None:
                        query = query.filter(ChunkModel.content_length <= filter.content_length.max)
            
            query = query.order_by(ChunkModel.id)
            
            return await create_connection(
                query=query,
                first=first,
                after=after,
                loader=info.context["loaders"]["chunk"],
                cursor_field="id"
            )
            
        except Exception as e:
            return ChunkConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def detection(self, info: Info, id: strawberry.ID) -> Optional[Detection]:
        """Get a single detection by ID"""
        try:
            loaders = info.context["loaders"]
            return await loaders["detection"].load(int(id))
        except Exception as e:
            return None
    
    @strawberry.field
    async def detections(
        self,
        info: Info,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[DetectionFilter] = None
    ) -> DetectionConnection:
        """Get paginated list of detections"""
        try:
            session = info.context["session"]
            
            query = session.query(DetectionModel)
            
            # Apply filters
            if filter:
                query = apply_detection_filter(query, filter)
            
            query = query.order_by(desc(DetectionModel.confidence_score))
            
            return await create_connection(
                query=query,
                first=first,
                after=after,
                loader=info.context["loaders"]["detection"],
                cursor_field="id"
            )
            
        except Exception as e:
            return DetectionConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def pattern(self, info: Info, id: strawberry.ID) -> Optional[Pattern]:
        """Get a single pattern by ID"""
        try:
            loaders = info.context["loaders"]
            return await loaders["pattern"].load(int(id))
        except Exception as e:
            return None
    
    @strawberry.field
    async def patterns(
        self,
        info: Info,
        first: int = 20,
        after: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> PatternConnection:
        """Get paginated list of patterns"""
        try:
            session = info.context["session"]
            
            query = session.query(PatternModel)
            
            if is_active is not None:
                query = query.filter(PatternModel.is_active == is_active)
            
            query = query.order_by(desc(PatternModel.effectiveness_score))
            
            return await create_connection(
                query=query,
                first=first,
                after=after,
                loader=info.context["loaders"]["pattern"],
                cursor_field="id"
            )
            
        except Exception as e:
            return PatternConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def analysis(self, info: Info, id: strawberry.ID) -> Optional[ArbitrationAnalysis]:
        """Get a single analysis by ID"""
        try:
            loaders = info.context["loaders"]
            return await loaders["analysis"].load(int(id))
        except Exception as e:
            return None
    
    @strawberry.field
    async def analyses(
        self,
        info: Info,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[AnalysisFilter] = None
    ) -> AnalysisConnection:
        """Get paginated list of analyses"""
        try:
            session = info.context["session"]
            
            query = session.query(AnalysisModel)
            
            # Apply filters
            if filter:
                query = apply_analysis_filter(query, filter)
            
            query = query.order_by(desc(AnalysisModel.analyzed_at))
            
            return await create_connection(
                query=query,
                first=first,
                after=after,
                loader=info.context["loaders"]["analysis"],
                cursor_field="id"
            )
            
        except Exception as e:
            return AnalysisConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    @require_auth
    async def user(self, info: Info, id: strawberry.ID) -> Optional[User]:
        """Get a single user by ID (requires auth)"""
        try:
            current_user = await get_current_user(info)
            
            # Users can only see their own profile unless they're admin
            if current_user.role != "ADMIN" and str(current_user.id) != id:
                return None
            
            loaders = info.context["loaders"]
            return await loaders["user"].load(int(id))
        except Exception as e:
            return None
    
    @strawberry.field
    @require_auth
    async def current_user(self, info: Info) -> Optional[User]:
        """Get the current authenticated user"""
        try:
            return await get_current_user(info)
        except Exception as e:
            return None
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def users(
        self,
        info: Info,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[UserFilter] = None
    ) -> UserConnection:
        """Get paginated list of users (admin only)"""
        try:
            session = info.context["session"]
            
            query = session.query(UserModel)
            
            # Apply filters
            if filter:
                if filter.role:
                    # Role filtering would need to be implemented in User model
                    pass
                if filter.is_active is not None:
                    query = query.filter(UserModel.is_active == filter.is_active)
                if filter.is_verified is not None:
                    query = query.filter(UserModel.is_verified == filter.is_verified)
                if filter.organization:
                    query = query.filter(UserModel.organization.ilike(f"%{filter.organization}%"))
            
            query = query.order_by(UserModel.created_at.desc())
            
            return await create_connection(
                query=query,
                first=first,
                after=after,
                loader=info.context["loaders"]["user"],
                cursor_field="id"
            )
            
        except Exception as e:
            return UserConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def comments(
        self,
        info: Info,
        document_id: strawberry.ID,
        first: int = 20,
        after: Optional[str] = None
    ) -> CommentConnection:
        """Get comments for a document"""
        try:
            # For now, return empty connection as Comment model doesn't exist
            return CommentConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
        except Exception as e:
            return CommentConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def annotations(
        self,
        info: Info,
        document_id: strawberry.ID,
        first: int = 20,
        after: Optional[str] = None
    ) -> List[Annotation]:
        """Get annotations for a document"""
        try:
            # For now, return empty list as Annotation model doesn't exist
            return []
        except Exception as e:
            return []
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def system_stats(self, info: Info) -> SystemStats:
        """Get system-wide statistics (admin only)"""
        try:
            loaders = info.context["loaders"]
            
            # Load all stats using DataLoaders
            doc_stats = await loaders["document_stats"].load("global")
            detection_stats = await loaders["detection_stats"].load("global")
            pattern_stats = await loaders["pattern_stats"].load("global")
            
            return SystemStats(
                documents=doc_stats,
                detections=detection_stats,
                patterns=pattern_stats,
                uptime="Unknown",  # Would need to track server start time
                version="1.0.0"
            )
            
        except Exception as e:
            # Return empty stats on error
            return SystemStats(
                documents={},
                detections={},
                patterns={},
                uptime="Unknown",
                version="1.0.0"
            )
    
    @strawberry.field
    async def document_stats(
        self,
        info: Info,
        filter: Optional[DocumentFilter] = None,
        date_range: Optional[str] = None
    ) -> DocumentStats:
        """Get document statistics"""
        try:
            loaders = info.context["loaders"]
            return await loaders["document_stats"].load("global")
        except Exception as e:
            return DocumentStats(
                total_documents=0,
                processed_documents=0,
                documents_with_arbitration=0,
                average_processing_time=0.0,
                processing_rate=0.0
            )
    
    @strawberry.field
    async def detection_stats(
        self,
        info: Info,
        filter: Optional[DetectionFilter] = None,
        date_range: Optional[str] = None
    ) -> DetectionStats:
        """Get detection statistics"""
        try:
            loaders = info.context["loaders"]
            return await loaders["detection_stats"].load("global")
        except Exception as e:
            return DetectionStats(
                total_detections=0,
                high_confidence_detections=0,
                average_confidence_score=0.0,
                detections_by_type=[],
                detections_by_method=[]
            )
    
    @strawberry.field
    async def search_documents(
        self,
        info: Info,
        query: str,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[DocumentFilter] = None
    ) -> DocumentConnection:
        """Search documents by content"""
        try:
            session = info.context["session"]
            
            # Build search query
            search_query = session.query(DocumentModel)
            
            # Add text search (would need full-text search implementation)
            search_query = search_query.filter(
                or_(
                    DocumentModel.filename.ilike(f"%{query}%"),
                    # Would need to join with content/chunks for full-text search
                )
            )
            
            # Apply additional filters
            if filter:
                search_query = apply_document_filter(search_query, filter)
            
            search_query = search_query.order_by(desc(DocumentModel.upload_date))
            
            return await create_connection(
                query=search_query,
                first=first,
                after=after,
                loader=info.context["loaders"]["document"],
                cursor_field="id"
            )
            
        except Exception as e:
            return DocumentConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)
    
    @strawberry.field
    async def search_clauses(
        self,
        info: Info,
        query: str,
        first: int = 20,
        after: Optional[str] = None
    ) -> List[ArbitrationClause]:
        """Search arbitration clauses by content"""
        try:
            # For now, return empty list as complex search needs to be implemented
            return []
        except Exception as e:
            return []
    
    def _parse_global_id(self, global_id: str) -> tuple:
        """Parse a Relay global ID to get type and local ID"""
        try:
            import base64
            decoded = base64.b64decode(global_id).decode('utf-8')
            node_type, local_id = decoded.split(':', 1)
            return node_type, local_id
        except Exception:
            raise ValueError("Invalid global ID format")


# Field resolvers for relationships
@strawberry.field
async def document_chunks(
    self: Document, 
    info: Info,
    first: int = 20,
    after: Optional[str] = None,
    filter: Optional[ChunkFilter] = None
) -> ChunkConnection:
    """Resolve chunks for a document"""
    try:
        loaders = info.context["loaders"]
        chunks = await loaders["chunks_by_document"].load(int(self.id))
        
        # Apply filtering and pagination
        # This is a simplified version - proper implementation would handle pagination
        return ChunkConnection(
            edges=[{"node": chunk, "cursor": encode_cursor(chunk.id)} for chunk in chunks[:first]],
            page_info={
                "has_next_page": len(chunks) > first,
                "has_previous_page": False,
                "start_cursor": encode_cursor(chunks[0].id) if chunks else None,
                "end_cursor": encode_cursor(chunks[min(first-1, len(chunks)-1)].id) if chunks else None
            },
            total_count=len(chunks)
        )
    except Exception as e:
        return ChunkConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)


@strawberry.field
async def document_detections(
    self: Document,
    info: Info,
    first: int = 20,
    after: Optional[str] = None,
    filter: Optional[DetectionFilter] = None
) -> DetectionConnection:
    """Resolve detections for a document"""
    try:
        loaders = info.context["loaders"]
        detections = await loaders["detections_by_document"].load(int(self.id))
        
        # Apply filtering and pagination
        return DetectionConnection(
            edges=[{"node": detection, "cursor": encode_cursor(detection.id)} for detection in detections[:first]],
            page_info={
                "has_next_page": len(detections) > first,
                "has_previous_page": False,
                "start_cursor": encode_cursor(detections[0].id) if detections else None,
                "end_cursor": encode_cursor(detections[min(first-1, len(detections)-1)].id) if detections else None
            },
            total_count=len(detections)
        )
    except Exception as e:
        return DetectionConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)


@strawberry.field
async def document_analyses(
    self: Document,
    info: Info,
    first: int = 20,
    after: Optional[str] = None,
    version_filter: Optional[str] = None
) -> AnalysisConnection:
    """Resolve analyses for a document"""
    try:
        loaders = info.context["loaders"]
        analyses = await loaders["analyses_by_document"].load(int(self.id))
        
        # Apply version filtering if specified
        if version_filter:
            analyses = [a for a in analyses if a.analysis_version == version_filter]
        
        return AnalysisConnection(
            edges=[{"node": analysis, "cursor": encode_cursor(analysis.id)} for analysis in analyses[:first]],
            page_info={
                "has_next_page": len(analyses) > first,
                "has_previous_page": False,
                "start_cursor": encode_cursor(analyses[0].id) if analyses else None,
                "end_cursor": encode_cursor(analyses[min(first-1, len(analyses)-1)].id) if analyses else None
            },
            total_count=len(analyses)
        )
    except Exception as e:
        return AnalysisConnection(edges=[], page_info={"has_next_page": False, "has_previous_page": False}, total_count=0)


# Add resolvers to the Document type
Document.chunks = document_chunks
Document.detections = document_detections  
Document.analyses = document_analyses