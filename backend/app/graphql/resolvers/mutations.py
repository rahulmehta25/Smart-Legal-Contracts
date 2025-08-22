"""
GraphQL Mutation Resolvers
"""

import strawberry
from typing import List, Optional
from strawberry.types import Info
from sqlalchemy.orm import Session
from datetime import datetime
import json
import hashlib

from ..types import (
    DocumentUploadInput, DocumentUploadResult, DocumentUpdateInput,
    AnalysisRequestInput, AnalysisResult, AnalysisOptionsInput,
    QuickAnalysisInput, QuickAnalysisResult, QuickClause,
    PatternCreateInput, PatternUpdateInput, PatternResult,
    UserCreateInput, UserUpdateInput, UserResult,
    CommentCreateInput, CommentUpdateInput, CommentResult,
    AnnotationCreateInput, AnnotationResult
)
from ...db.database import get_session
from ...db.models import Document as DocumentModel, Chunk as ChunkModel, Pattern as PatternModel
from ...models.analysis import ArbitrationAnalysis as AnalysisModel, ArbitrationClause as ClauseModel
from ...models.user import User as UserModel
from ...services.analysis_service import AnalysisService
from ...services.document_service import DocumentService
from ...services.user_service import UserService
from ...rag.pipeline import RAGPipeline
from ..utils.auth import get_current_user, require_auth, hash_password, verify_password, create_access_token
from ..utils.validation import validate_document_upload, validate_pattern_input, validate_user_input


@strawberry.type
class Mutation:
    """Root Mutation type"""
    
    @strawberry.field
    @require_auth
    async def upload_document(self, info: Info, input: DocumentUploadInput) -> DocumentUploadResult:
        """Upload a new document for analysis"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            # Validate input
            validation_errors = validate_document_upload(input)
            if validation_errors:
                return DocumentUploadResult(
                    success=False,
                    message="Validation failed",
                    errors=validation_errors
                )
            
            # Check file size limits and content
            if len(input.content) > 10 * 1024 * 1024:  # 10MB limit
                return DocumentUploadResult(
                    success=False,
                    message="File too large",
                    errors=["File size exceeds 10MB limit"]
                )
            
            # Create document hash for deduplication
            content_hash = hashlib.sha256(input.content.encode('utf-8')).hexdigest()
            
            # Check if document already exists
            existing_doc = session.query(DocumentModel).filter(
                DocumentModel.content_hash == content_hash
            ).first()
            
            if existing_doc:
                return DocumentUploadResult(
                    success=False,
                    message="Document already exists",
                    errors=["A document with identical content already exists"]
                )
            
            # Create new document
            document = DocumentModel(
                filename=input.filename,
                file_type=input.file_type,
                file_size=len(input.content),
                content_hash=content_hash,
                upload_date=datetime.utcnow(),
                processing_status='pending',
                metadata=json.loads(input.metadata) if input.metadata else None
            )
            
            session.add(document)
            session.flush()  # Get the ID
            
            # Process document into chunks using DocumentService
            document_service = DocumentService(session)
            chunks_created = await document_service.process_document_content(
                document_id=document.id,
                content=input.content
            )
            
            document.total_chunks = chunks_created
            document.processing_status = 'completed'
            document.last_processed = datetime.utcnow()
            
            session.commit()
            
            # Convert to GraphQL type
            from ..dataloaders.document_loaders import DocumentDataLoader
            doc_loader = DocumentDataLoader(session)
            document_result = doc_loader._convert_to_graphql_type(document)
            
            return DocumentUploadResult(
                success=True,
                message=f"Document uploaded successfully. Created {chunks_created} chunks.",
                document=document_result
            )
            
        except Exception as e:
            session.rollback()
            return DocumentUploadResult(
                success=False,
                message="Upload failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def update_document(self, info: Info, input: DocumentUpdateInput) -> DocumentUploadResult:
        """Update document metadata"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            document = session.query(DocumentModel).get(int(input.id))
            if not document:
                return DocumentUploadResult(
                    success=False,
                    message="Document not found",
                    errors=["Document does not exist"]
                )
            
            # Update fields
            if input.filename:
                document.filename = input.filename
            if input.metadata:
                document.metadata = json.loads(input.metadata)
            
            document.updated_at = datetime.utcnow()
            session.commit()
            
            # Convert to GraphQL type
            from ..dataloaders.document_loaders import DocumentDataLoader
            doc_loader = DocumentDataLoader(session)
            document_result = doc_loader._convert_to_graphql_type(document)
            
            return DocumentUploadResult(
                success=True,
                message="Document updated successfully",
                document=document_result
            )
            
        except Exception as e:
            session.rollback()
            return DocumentUploadResult(
                success=False,
                message="Update failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def delete_document(self, info: Info, id: strawberry.ID) -> bool:
        """Delete a document and all associated data"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            document = session.query(DocumentModel).get(int(id))
            if not document:
                return False
            
            # Check permissions (users can only delete their own documents unless admin)
            if current_user.role != "ADMIN":
                # Would need user_id field in document to check ownership
                pass
            
            # Delete document (cascading deletes will handle chunks, detections, etc.)
            session.delete(document)
            session.commit()
            
            return True
            
        except Exception as e:
            session.rollback()
            return False
    
    @strawberry.field
    @require_auth
    async def request_analysis(self, info: Info, input: AnalysisRequestInput) -> AnalysisResult:
        """Request arbitration analysis for a document"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            document = session.query(DocumentModel).get(int(input.document_id))
            if not document:
                return AnalysisResult(
                    success=False,
                    message="Document not found",
                    errors=["Document does not exist"]
                )
            
            # Check if analysis already exists and force_reanalysis is False
            if not input.force_reanalysis:
                existing_analysis = session.query(AnalysisModel).filter(
                    AnalysisModel.document_id == document.id
                ).first()
                
                if existing_analysis:
                    from ..dataloaders.analysis_loaders import AnalysisDataLoader
                    analysis_loader = AnalysisDataLoader(session)
                    analysis_result = analysis_loader._convert_to_graphql_type(existing_analysis)
                    
                    return AnalysisResult(
                        success=True,
                        message="Analysis already exists",
                        analysis=analysis_result,
                        processing_time_ms=existing_analysis.processing_time_ms
                    )
            
            # Run analysis using AnalysisService
            analysis_service = AnalysisService(session)
            start_time = datetime.utcnow()
            
            analysis_results = await analysis_service.analyze_document(
                document_id=document.id,
                options={
                    'include_context': input.analysis_options.include_context if input.analysis_options else True,
                    'confidence_threshold': input.analysis_options.confidence_threshold if input.analysis_options else 0.5,
                    'max_clauses': input.analysis_options.max_clauses if input.analysis_options else 50
                }
            )
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Convert to GraphQL type
            from ..dataloaders.analysis_loaders import AnalysisDataLoader
            analysis_loader = AnalysisDataLoader(session)
            analysis_result = analysis_loader._convert_to_graphql_type(analysis_results)
            
            return AnalysisResult(
                success=True,
                message="Analysis completed successfully",
                analysis=analysis_result,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return AnalysisResult(
                success=False,
                message="Analysis failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    async def quick_analysis(self, info: Info, input: QuickAnalysisInput) -> QuickAnalysisResult:
        """Perform quick analysis on text without storing"""
        try:
            # Initialize RAG pipeline for quick analysis
            rag_pipeline = RAGPipeline()
            
            start_time = datetime.utcnow()
            
            # Run analysis
            results = await rag_pipeline.analyze_text(
                text=input.text,
                include_context=input.include_context
            )
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Convert results to GraphQL format
            clauses_found = [
                QuickClause(
                    text=clause['text'],
                    type=clause['type'],
                    confidence=clause['confidence'],
                    start_position=clause['start_pos'],
                    end_position=clause['end_pos']
                )
                for clause in results.get('clauses', [])
            ]
            
            return QuickAnalysisResult(
                has_arbitration_clause=results.get('has_arbitration_clause', False),
                confidence_score=results.get('confidence_score', 0.0),
                clauses_found=clauses_found,
                summary=results.get('summary', ''),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return QuickAnalysisResult(
                has_arbitration_clause=False,
                confidence_score=0.0,
                clauses_found=[],
                summary=f"Analysis failed: {str(e)}",
                processing_time_ms=0
            )
    
    @strawberry.field
    @require_auth
    async def validate_detection(self, info: Info, detection_id: strawberry.ID, is_valid: bool) -> bool:
        """Validate or invalidate a detection"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            from ...db.models import Detection
            detection = session.query(Detection).get(int(detection_id))
            if not detection:
                return False
            
            detection.is_validated = is_valid
            detection.validation_score = 1.0 if is_valid else 0.0
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            return False
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def create_pattern(self, info: Info, input: PatternCreateInput) -> PatternResult:
        """Create a new pattern (admin only)"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            # Validate input
            validation_errors = validate_pattern_input(input)
            if validation_errors:
                return PatternResult(
                    success=False,
                    message="Validation failed",
                    errors=validation_errors
                )
            
            # Check if pattern already exists
            existing_pattern = session.query(PatternModel).filter(
                PatternModel.pattern_name == input.pattern_name
            ).first()
            
            if existing_pattern:
                return PatternResult(
                    success=False,
                    message="Pattern already exists",
                    errors=["A pattern with this name already exists"]
                )
            
            # Create new pattern
            pattern = PatternModel(
                pattern_name=input.pattern_name,
                pattern_text=input.pattern_text,
                pattern_type=input.pattern_type,
                category=input.category,
                language=input.language,
                effectiveness_score=0.5,  # Default effectiveness
                usage_count=0,
                is_active=True,
                created_by=current_user.username
            )
            
            session.add(pattern)
            session.commit()
            
            # Convert to GraphQL type
            from ..dataloaders.pattern_loaders import PatternDataLoader
            pattern_loader = PatternDataLoader(session)
            pattern_result = pattern_loader._convert_to_graphql_type(pattern)
            
            return PatternResult(
                success=True,
                message="Pattern created successfully",
                pattern=pattern_result
            )
            
        except Exception as e:
            session.rollback()
            return PatternResult(
                success=False,
                message="Pattern creation failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def update_pattern(self, info: Info, input: PatternUpdateInput) -> PatternResult:
        """Update an existing pattern (admin only)"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            pattern = session.query(PatternModel).get(int(input.id))
            if not pattern:
                return PatternResult(
                    success=False,
                    message="Pattern not found",
                    errors=["Pattern does not exist"]
                )
            
            # Update fields
            if input.pattern_name:
                pattern.pattern_name = input.pattern_name
            if input.pattern_text:
                pattern.pattern_text = input.pattern_text
            if input.is_active is not None:
                pattern.is_active = input.is_active
            
            pattern.updated_at = datetime.utcnow()
            session.commit()
            
            # Convert to GraphQL type
            from ..dataloaders.pattern_loaders import PatternDataLoader
            pattern_loader = PatternDataLoader(session)
            pattern_result = pattern_loader._convert_to_graphql_type(pattern)
            
            return PatternResult(
                success=True,
                message="Pattern updated successfully",
                pattern=pattern_result
            )
            
        except Exception as e:
            session.rollback()
            return PatternResult(
                success=False,
                message="Pattern update failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def delete_pattern(self, info: Info, id: strawberry.ID) -> bool:
        """Delete a pattern (admin only)"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            pattern = session.query(PatternModel).get(int(id))
            if not pattern:
                return False
            
            # Soft delete by setting inactive instead of hard delete
            pattern.is_active = False
            session.commit()
            
            return True
            
        except Exception as e:
            session.rollback()
            return False
    
    @strawberry.field
    async def register_user(self, info: Info, input: UserCreateInput) -> UserResult:
        """Register a new user"""
        try:
            session = info.context["session"]
            
            # Validate input
            validation_errors = validate_user_input(input)
            if validation_errors:
                return UserResult(
                    success=False,
                    message="Validation failed",
                    errors=validation_errors
                )
            
            # Check if user already exists
            existing_user = session.query(UserModel).filter(
                or_(
                    UserModel.email == input.email,
                    UserModel.username == input.username
                )
            ).first()
            
            if existing_user:
                return UserResult(
                    success=False,
                    message="User already exists",
                    errors=["Email or username already taken"]
                )
            
            # Create new user
            hashed_password = hash_password(input.password)
            user = UserModel(
                email=input.email,
                username=input.username,
                hashed_password=hashed_password,
                full_name=input.full_name,
                organization=input.organization,
                is_active=True,
                is_verified=False  # Would need email verification
            )
            
            session.add(user)
            session.commit()
            
            # Create access token
            token = create_access_token(user.username)
            
            # Convert to GraphQL type
            from ..dataloaders.user_loaders import UserDataLoader
            user_loader = UserDataLoader(session)
            user_result = user_loader._convert_to_graphql_type(user)
            
            return UserResult(
                success=True,
                message="User registered successfully",
                user=user_result,
                token=token
            )
            
        except Exception as e:
            session.rollback()
            return UserResult(
                success=False,
                message="Registration failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    async def login_user(self, info: Info, username: str, password: str) -> UserResult:
        """Login user and return access token"""
        try:
            session = info.context["session"]
            
            # Find user
            user = session.query(UserModel).filter(
                or_(
                    UserModel.username == username,
                    UserModel.email == username
                )
            ).first()
            
            if not user or not verify_password(password, user.hashed_password):
                return UserResult(
                    success=False,
                    message="Invalid credentials",
                    errors=["Username/email or password is incorrect"]
                )
            
            if not user.is_active:
                return UserResult(
                    success=False,
                    message="Account disabled",
                    errors=["User account is disabled"]
                )
            
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            # Create access token
            token = create_access_token(user.username)
            
            # Convert to GraphQL type
            from ..dataloaders.user_loaders import UserDataLoader
            user_loader = UserDataLoader(session)
            user_result = user_loader._convert_to_graphql_type(user)
            
            return UserResult(
                success=True,
                message="Login successful",
                user=user_result,
                token=token
            )
            
        except Exception as e:
            return UserResult(
                success=False,
                message="Login failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def update_user(self, info: Info, input: UserUpdateInput) -> UserResult:
        """Update user profile"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            # Users can only update their own profile unless they're admin
            if current_user.role != "ADMIN" and str(current_user.id) != input.id:
                return UserResult(
                    success=False,
                    message="Permission denied",
                    errors=["You can only update your own profile"]
                )
            
            user = session.query(UserModel).get(int(input.id))
            if not user:
                return UserResult(
                    success=False,
                    message="User not found",
                    errors=["User does not exist"]
                )
            
            # Update fields
            if input.full_name:
                user.full_name = input.full_name
            if input.organization:
                user.organization = input.organization
            # Only admins can change roles
            if input.role and current_user.role == "ADMIN":
                # Would need role field in User model
                pass
            
            session.commit()
            
            # Convert to GraphQL type
            from ..dataloaders.user_loaders import UserDataLoader
            user_loader = UserDataLoader(session)
            user_result = user_loader._convert_to_graphql_type(user)
            
            return UserResult(
                success=True,
                message="User updated successfully",
                user=user_result
            )
            
        except Exception as e:
            session.rollback()
            return UserResult(
                success=False,
                message="Update failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def delete_user(self, info: Info, id: strawberry.ID) -> bool:
        """Delete a user (admin only)"""
        try:
            session = info.context["session"]
            current_user = await get_current_user(info)
            
            user = session.query(UserModel).get(int(id))
            if not user:
                return False
            
            # Prevent deleting yourself
            if user.id == current_user.id:
                return False
            
            # Soft delete by deactivating instead of hard delete
            user.is_active = False
            session.commit()
            
            return True
            
        except Exception as e:
            session.rollback()
            return False
    
    @strawberry.field
    @require_auth
    async def create_comment(self, info: Info, input: CommentCreateInput) -> CommentResult:
        """Create a comment on a document"""
        try:
            # For now, return placeholder as Comment model doesn't exist
            return CommentResult(
                success=False,
                message="Comments not implemented yet",
                errors=["Comment functionality is not yet implemented"]
            )
        except Exception as e:
            return CommentResult(
                success=False,
                message="Comment creation failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def update_comment(self, info: Info, input: CommentUpdateInput) -> CommentResult:
        """Update a comment"""
        try:
            # For now, return placeholder as Comment model doesn't exist
            return CommentResult(
                success=False,
                message="Comments not implemented yet",
                errors=["Comment functionality is not yet implemented"]
            )
        except Exception as e:
            return CommentResult(
                success=False,
                message="Comment update failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def delete_comment(self, info: Info, id: strawberry.ID) -> bool:
        """Delete a comment"""
        try:
            # For now, return False as Comment model doesn't exist
            return False
        except Exception as e:
            return False
    
    @strawberry.field
    @require_auth
    async def create_annotation(self, info: Info, input: AnnotationCreateInput) -> AnnotationResult:
        """Create an annotation on a document"""
        try:
            # For now, return placeholder as Annotation model doesn't exist
            return AnnotationResult(
                success=False,
                message="Annotations not implemented yet",
                errors=["Annotation functionality is not yet implemented"]
            )
        except Exception as e:
            return AnnotationResult(
                success=False,
                message="Annotation creation failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def update_annotation(self, info: Info, input: AnnotationCreateInput) -> AnnotationResult:
        """Update an annotation"""
        try:
            # For now, return placeholder as Annotation model doesn't exist
            return AnnotationResult(
                success=False,
                message="Annotations not implemented yet",
                errors=["Annotation functionality is not yet implemented"]
            )
        except Exception as e:
            return AnnotationResult(
                success=False,
                message="Annotation update failed",
                errors=[str(e)]
            )
    
    @strawberry.field
    @require_auth
    async def delete_annotation(self, info: Info, id: strawberry.ID) -> bool:
        """Delete an annotation"""
        try:
            # For now, return False as Annotation model doesn't exist
            return False
        except Exception as e:
            return False
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def clear_cache(self, info: Info, cache_key: Optional[str] = None) -> bool:
        """Clear cache entries (admin only)"""
        try:
            session = info.context["session"]
            
            from ...db.models import QueryCache
            if cache_key:
                # Clear specific cache key
                session.query(QueryCache).filter(
                    QueryCache.cache_key == cache_key
                ).delete()
            else:
                # Clear all cache
                session.query(QueryCache).delete()
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            return False
    
    @strawberry.field
    @require_auth(role="ADMIN")
    async def refresh_stats(self, info: Info) -> bool:
        """Refresh system statistics (admin only)"""
        try:
            # Would trigger recalculation of cached statistics
            # For now, just return True
            return True
        except Exception as e:
            return False