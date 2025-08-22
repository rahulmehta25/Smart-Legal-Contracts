"""
GraphQL Types using Strawberry
"""

import strawberry
from typing import List, Optional, Union
from datetime import datetime
from enum import Enum

# Enums
@strawberry.enum
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"  
    COMPLETED = "completed"
    FAILED = "failed"

@strawberry.enum
class DetectionType(Enum):
    MANDATORY_ARBITRATION = "mandatory_arbitration"
    OPTIONAL_ARBITRATION = "optional_arbitration"
    CLASS_ACTION_WAIVER = "class_action_waiver"
    BINDING_ARBITRATION = "binding_arbitration"
    DISPUTE_RESOLUTION = "dispute_resolution"

@strawberry.enum
class ClauseType(Enum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"
    CLASS_ACTION_WAIVER = "class_action_waiver"
    BINDING_ARBITRATION = "binding_arbitration"
    DISPUTE_RESOLUTION = "dispute_resolution"

@strawberry.enum
class PatternType(Enum):
    REGEX = "regex"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"

@strawberry.enum
class DetectionMethod(Enum):
    RULE_BASED = "rule_based"
    ML_MODEL = "ml_model"
    HYBRID = "hybrid"

@strawberry.enum
class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

@strawberry.enum
class SubscriptionType(Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

# Interfaces
@strawberry.interface
class Node:
    id: strawberry.ID

@strawberry.interface
class Timestamped:
    created_at: datetime
    updated_at: Optional[datetime] = None

# Input Types
@strawberry.input
class IntRange:
    min: Optional[int] = None
    max: Optional[int] = None

@strawberry.input
class FloatRange:
    min: Optional[float] = None
    max: Optional[float] = None

@strawberry.input
class DocumentFilter:
    filename: Optional[str] = None
    file_type: Optional[str] = None
    processing_status: Optional[ProcessingStatus] = None
    has_arbitration_clauses: Optional[bool] = None
    uploaded_after: Optional[datetime] = None
    uploaded_before: Optional[datetime] = None
    content_search: Optional[str] = None

@strawberry.input
class ChunkFilter:
    page_number: Optional[int] = None
    has_embedding: Optional[bool] = None
    has_detections: Optional[bool] = None
    content_length: Optional[IntRange] = None

@strawberry.input
class DetectionFilter:
    detection_type: Optional[DetectionType] = None
    confidence_score: Optional[FloatRange] = None
    detection_method: Optional[DetectionMethod] = None
    is_validated: Optional[bool] = None
    is_high_confidence: Optional[bool] = None

@strawberry.input
class ClauseFilter:
    clause_type: Optional[ClauseType] = None
    relevance_score: Optional[FloatRange] = None
    severity_score: Optional[FloatRange] = None

@strawberry.input
class AnalysisFilter:
    has_arbitration_clause: Optional[bool] = None
    confidence_score: Optional[FloatRange] = None
    analysis_version: Optional[str] = None
    analyzed_after: Optional[datetime] = None
    analyzed_before: Optional[datetime] = None

@strawberry.input
class UserFilter:
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    organization: Optional[str] = None

# Mutation Input Types
@strawberry.input
class DocumentUploadInput:
    filename: str
    content: str
    file_type: str = "text/plain"
    metadata: Optional[str] = None  # JSON string

@strawberry.input
class DocumentUpdateInput:
    id: strawberry.ID
    filename: Optional[str] = None
    metadata: Optional[str] = None  # JSON string

@strawberry.input
class AnalysisOptionsInput:
    include_context: bool = True
    confidence_threshold: float = 0.5
    max_clauses: int = 50

@strawberry.input
class AnalysisRequestInput:
    document_id: strawberry.ID
    force_reanalysis: bool = False
    analysis_options: Optional[AnalysisOptionsInput] = None

@strawberry.input
class QuickAnalysisInput:
    text: str
    include_context: bool = True

@strawberry.input
class PatternCreateInput:
    pattern_name: str
    pattern_text: str
    pattern_type: PatternType
    category: str
    language: str = "en"

@strawberry.input
class PatternUpdateInput:
    id: strawberry.ID
    pattern_name: Optional[str] = None
    pattern_text: Optional[str] = None
    is_active: Optional[bool] = None

@strawberry.input
class UserCreateInput:
    email: str
    username: str
    password: str
    full_name: Optional[str] = None
    organization: Optional[str] = None

@strawberry.input
class UserUpdateInput:
    id: strawberry.ID
    full_name: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[UserRole] = None

@strawberry.input
class CommentCreateInput:
    document_id: strawberry.ID
    content: str
    parent_comment_id: Optional[strawberry.ID] = None

@strawberry.input
class CommentUpdateInput:
    id: strawberry.ID
    content: Optional[str] = None
    is_resolved: Optional[bool] = None

@strawberry.input
class AnnotationCreateInput:
    document_id: strawberry.ID
    chunk_id: Optional[strawberry.ID] = None
    start_position: int
    end_position: int
    content: str
    annotation_type: str
    metadata: Optional[str] = None  # JSON string

# Page Info for Relay-style pagination
@strawberry.type
class PageInfo:
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None

# Connection Types
@strawberry.type
class DocumentEdge:
    node: "Document"
    cursor: str

@strawberry.type
class DocumentConnection:
    edges: List[DocumentEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class ChunkEdge:
    node: "Chunk"
    cursor: str

@strawberry.type
class ChunkConnection:
    edges: List[ChunkEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class DetectionEdge:
    node: "Detection"
    cursor: str

@strawberry.type
class DetectionConnection:
    edges: List[DetectionEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class ClauseEdge:
    node: "ArbitrationClause"
    cursor: str

@strawberry.type
class ClauseConnection:
    edges: List[ClauseEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class AnalysisEdge:
    node: "ArbitrationAnalysis"
    cursor: str

@strawberry.type
class AnalysisConnection:
    edges: List[AnalysisEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class PatternEdge:
    node: "Pattern"
    cursor: str

@strawberry.type
class PatternConnection:
    edges: List[PatternEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class UserEdge:
    node: "User"
    cursor: str

@strawberry.type
class UserConnection:
    edges: List[UserEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class CommentEdge:
    node: "Comment"
    cursor: str

@strawberry.type
class CommentConnection:
    edges: List[CommentEdge]
    page_info: PageInfo
    total_count: int

# Core Types
@strawberry.type
class Document(Node, Timestamped):
    id: strawberry.ID
    filename: str
    file_path: Optional[str] = None
    file_type: str
    file_size: int
    content_hash: str
    upload_date: datetime
    last_processed: Optional[datetime] = None
    processing_status: ProcessingStatus
    total_pages: Optional[int] = None
    total_chunks: int
    metadata: Optional[str] = None  # JSON string
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Computed fields
    is_processed: bool
    detection_count: int
    has_arbitration_clauses: bool
    average_confidence_score: Optional[float] = None
    processing_time_ms: Optional[int] = None

@strawberry.type
class Chunk(Node, Timestamped):
    id: strawberry.ID
    document_id: strawberry.ID
    chunk_index: int
    content: str
    content_length: int
    chunk_hash: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    embedding_model: str
    similarity_threshold: float
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Computed fields
    has_embedding: bool
    has_detections: bool
    embedding: Optional[List[float]] = None  # Only return if requested and authorized

@strawberry.type
class Detection(Node, Timestamped):
    id: strawberry.ID
    chunk_id: strawberry.ID
    document_id: strawberry.ID
    detection_type: DetectionType
    confidence_score: float
    pattern_id: Optional[strawberry.ID] = None
    matched_text: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    page_number: Optional[int] = None
    detection_method: DetectionMethod
    model_version: Optional[str] = None
    is_validated: bool
    validation_score: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Computed fields
    is_high_confidence: bool
    context_snippet: str
    severity: Optional[float] = None

@strawberry.type
class Pattern(Node, Timestamped):
    id: strawberry.ID
    pattern_name: str
    pattern_text: str
    pattern_type: PatternType
    category: str
    language: str
    effectiveness_score: float
    usage_count: int
    last_used: Optional[datetime] = None
    is_active: bool
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Computed fields
    is_effective: bool
    average_confidence_score: Optional[float] = None

@strawberry.type
class ArbitrationAnalysis(Node, Timestamped):
    id: strawberry.ID
    document_id: strawberry.ID
    has_arbitration_clause: bool
    confidence_score: Optional[float] = None
    analysis_summary: Optional[str] = None
    analyzed_at: datetime
    analysis_version: str
    processing_time_ms: Optional[int] = None
    metadata: Optional[str] = None  # JSON string
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Computed fields
    clause_count: int
    average_clause_score: Optional[float] = None
    risk_level: str

@strawberry.type
class ArbitrationClause(Node):
    id: strawberry.ID
    analysis_id: strawberry.ID
    clause_text: str
    clause_type: Optional[ClauseType] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    relevance_score: Optional[float] = None
    severity_score: Optional[float] = None
    surrounding_context: Optional[str] = None
    section_title: Optional[str] = None
    
    # Computed fields
    risk_level: str
    is_binding: bool

@strawberry.type
class User(Node, Timestamped):
    id: strawberry.ID
    email: str
    username: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    role: UserRole
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Computed fields
    document_count: int
    analysis_count: int

@strawberry.type
class Organization(Node, Timestamped):
    id: strawberry.ID
    name: str
    domain: Optional[str] = None
    subscription_type: SubscriptionType
    max_users: int
    max_documents: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Usage statistics
    current_user_count: int
    current_document_count: int
    monthly_analysis_count: int

@strawberry.type
class Comment(Node, Timestamped):
    id: strawberry.ID
    document_id: strawberry.ID
    user_id: strawberry.ID
    content: str
    is_resolved: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

@strawberry.type
class Annotation(Node, Timestamped):
    id: strawberry.ID
    document_id: strawberry.ID
    user_id: strawberry.ID
    chunk_id: Optional[strawberry.ID] = None
    start_position: int
    end_position: int
    content: str
    annotation_type: str
    metadata: Optional[str] = None  # JSON string
    created_at: datetime
    updated_at: Optional[datetime] = None

# Mutation Response Types
@strawberry.type
class DocumentUploadResult:
    success: bool
    message: str
    document: Optional[Document] = None
    errors: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class AnalysisResult:
    success: bool
    message: str
    analysis: Optional[ArbitrationAnalysis] = None
    processing_time_ms: Optional[int] = None
    errors: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class QuickClause:
    text: str
    type: str
    confidence: float
    start_position: int
    end_position: int

@strawberry.type
class QuickAnalysisResult:
    has_arbitration_clause: bool
    confidence_score: float
    clauses_found: List[QuickClause]
    summary: str
    processing_time_ms: int

@strawberry.type
class PatternResult:
    success: bool
    message: str
    pattern: Optional[Pattern] = None
    errors: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class UserResult:
    success: bool
    message: str
    user: Optional[User] = None
    token: Optional[str] = None  # Only returned on login/register
    errors: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class CommentResult:
    success: bool
    message: str
    comment: Optional[Comment] = None
    errors: List[str] = strawberry.field(default_factory=list)

@strawberry.type
class AnnotationResult:
    success: bool
    message: str
    annotation: Optional[Annotation] = None
    errors: List[str] = strawberry.field(default_factory=list)

# Subscription Types
@strawberry.type
class DocumentProcessingUpdate:
    document_id: strawberry.ID
    status: ProcessingStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    error_message: Optional[str] = None

@strawberry.type
class AnalysisUpdate:
    analysis_id: strawberry.ID
    document_id: strawberry.ID
    status: str
    progress: Optional[float] = None
    results: Optional[ArbitrationAnalysis] = None

@strawberry.type
class CommentUpdate:
    document_id: strawberry.ID
    comment: Comment
    action: str  # CREATED, UPDATED, DELETED

@strawberry.type
class CollaborationUpdate:
    document_id: strawberry.ID
    user_id: strawberry.ID
    action: str
    data: Optional[str] = None  # JSON string

# Statistics Types
@strawberry.type
class DocumentStats:
    total_documents: int
    processed_documents: int
    documents_with_arbitration: int
    average_processing_time: float
    processing_rate: float

@strawberry.type
class DetectionTypeStat:
    type: DetectionType
    count: int
    average_confidence: float

@strawberry.type
class DetectionMethodStat:
    method: DetectionMethod
    count: int
    average_confidence: float

@strawberry.type
class DetectionStats:
    total_detections: int
    high_confidence_detections: int
    average_confidence_score: float
    detections_by_type: List[DetectionTypeStat]
    detections_by_method: List[DetectionMethodStat]

@strawberry.type
class PatternStats:
    total_patterns: int
    active_patterns: int
    average_effectiveness: float
    most_used_patterns: List[Pattern]

@strawberry.type
class SystemStats:
    documents: DocumentStats
    detections: DetectionStats
    patterns: PatternStats
    uptime: str
    version: str