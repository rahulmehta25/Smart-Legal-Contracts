"""
File Storage Service

Comprehensive file storage service supporting:
- Local file storage with directory management
- S3-compatible storage (AWS S3, MinIO, etc.)
- File versioning and deduplication
- Thumbnail generation for images/PDFs
- Secure URL generation with expiration
- Metadata storage and indexing
- File validation and security scanning
"""

import hashlib
import io
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json

# Cloud storage
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from minio import Minio
from minio.error import S3Error

# Image processing for thumbnails
from PIL import Image, ImageOps
import cv2
import numpy as np
from pdf2image import convert_from_bytes, convert_from_path

# Security
import filetype
import magic

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends"""
    LOCAL = "local"
    S3 = "s3"
    MINIO = "minio"


class FileStatus(Enum):
    """File processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


@dataclass
class FileMetadata:
    """Complete file metadata"""
    file_id: str
    original_filename: str
    content_type: str
    file_size: int
    checksum: str
    upload_date: datetime
    storage_backend: StorageBackend
    storage_path: str
    version: int = 1
    status: FileStatus = FileStatus.PENDING
    thumbnails: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    expiry_date: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class StorageConfig:
    """Storage configuration"""
    backend: StorageBackend
    local_base_path: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_secure: bool = True
    enable_versioning: bool = True
    enable_thumbnails: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: Optional[List[str]] = None
    quarantine_unknown_files: bool = True


@dataclass
class UploadResult:
    """File upload result"""
    file_id: str
    file_metadata: FileMetadata
    thumbnail_urls: Dict[str, str] = field(default_factory=dict)
    secure_url: Optional[str] = None
    public_url: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StorageService:
    """Multi-backend file storage service"""
    
    def __init__(self, config: StorageConfig):
        """
        Initialize storage service
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.backend = config.backend
        
        # Initialize storage client based on backend
        self._init_storage_client()
        
        # Create local directories if using local storage
        if self.backend == StorageBackend.LOCAL and config.local_base_path:
            self._ensure_local_directories()
        
        # Metadata store (in production, use a proper database)
        self.metadata_store: Dict[str, FileMetadata] = {}
    
    def _init_storage_client(self):
        """Initialize storage client based on backend"""
        
        if self.backend == StorageBackend.S3:
            try:
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.config.s3_region,
                    aws_access_key_id=self.config.s3_access_key,
                    aws_secret_access_key=self.config.s3_secret_key,
                    endpoint_url=self.config.s3_endpoint_url
                )
                # Test connection
                self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
                logger.info("S3 storage initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize S3 storage: {e}")
                raise
        
        elif self.backend == StorageBackend.MINIO:
            try:
                self.minio_client = Minio(
                    self.config.minio_endpoint,
                    access_key=self.config.minio_access_key,
                    secret_key=self.config.minio_secret_key,
                    secure=self.config.minio_secure
                )
                # Test connection
                if not self.minio_client.bucket_exists(self.config.s3_bucket):
                    self.minio_client.make_bucket(self.config.s3_bucket)
                logger.info("MinIO storage initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO storage: {e}")
                raise
    
    def _ensure_local_directories(self):
        """Create necessary local directories"""
        base_path = Path(self.config.local_base_path)
        
        directories = [
            base_path,
            base_path / "files",
            base_path / "thumbnails",
            base_path / "temp",
            base_path / "quarantine"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def upload_file(self, 
                   file_data: bytes,
                   filename: str,
                   content_type: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None,
                   generate_thumbnails: bool = None) -> UploadResult:
        """
        Upload file to storage
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            content_type: MIME type (auto-detected if None)
            metadata: Additional metadata
            tags: File tags
            generate_thumbnails: Whether to generate thumbnails
            
        Returns:
            UploadResult with file information and URLs
        """
        
        errors = []
        warnings = []
        
        try:
            # Validate file
            validation_result = self._validate_file(file_data, filename)
            if validation_result['errors']:
                errors.extend(validation_result['errors'])
                if validation_result['quarantine']:
                    return self._quarantine_file(file_data, filename, errors)
            
            # Detect content type if not provided
            if not content_type:
                content_type = validation_result.get('content_type', 'application/octet-stream')
            
            # Generate file ID and checksum
            file_id = str(uuid.uuid4())
            checksum = hashlib.sha256(file_data).hexdigest()
            
            # Check for duplicate files
            existing_file = self._find_by_checksum(checksum)
            if existing_file and self.config.enable_versioning:
                return self._handle_duplicate(existing_file, file_data, filename, metadata)
            
            # Determine storage path
            storage_path = self._generate_storage_path(file_id, filename)
            
            # Upload to storage backend
            upload_success = False
            if self.backend == StorageBackend.LOCAL:
                upload_success = self._upload_local(file_data, storage_path)
            elif self.backend == StorageBackend.S3:
                upload_success = self._upload_s3(file_data, storage_path, content_type)
            elif self.backend == StorageBackend.MINIO:
                upload_success = self._upload_minio(file_data, storage_path, content_type)
            
            if not upload_success:
                errors.append("File upload failed")
                raise Exception("Storage upload failed")
            
            # Create file metadata
            file_metadata = FileMetadata(
                file_id=file_id,
                original_filename=filename,
                content_type=content_type,
                file_size=len(file_data),
                checksum=checksum,
                upload_date=datetime.utcnow(),
                storage_backend=self.backend,
                storage_path=storage_path,
                metadata=metadata or {},
                tags=tags or [],
                status=FileStatus.PROCESSING
            )
            
            # Generate thumbnails if enabled and applicable
            thumbnail_urls = {}
            if (generate_thumbnails is True or 
                (generate_thumbnails is None and self.config.enable_thumbnails)):
                try:
                    thumbnails = self._generate_thumbnails(file_data, file_id, content_type)
                    file_metadata.thumbnails = thumbnails
                    thumbnail_urls = self._get_thumbnail_urls(thumbnails)
                except Exception as e:
                    warnings.append(f"Thumbnail generation failed: {e}")
            
            # Store metadata
            file_metadata.status = FileStatus.COMPLETED
            self.metadata_store[file_id] = file_metadata
            
            # Generate URLs
            secure_url = self.generate_secure_url(file_id, expires_in=3600)
            public_url = self.get_public_url(file_id) if self._is_public_readable() else None
            
            return UploadResult(
                file_id=file_id,
                file_metadata=file_metadata,
                thumbnail_urls=thumbnail_urls,
                secure_url=secure_url,
                public_url=public_url,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            errors.append(str(e))
            
            # Create failed result
            return UploadResult(
                file_id="",
                file_metadata=FileMetadata(
                    file_id="",
                    original_filename=filename,
                    content_type=content_type or "application/octet-stream",
                    file_size=len(file_data),
                    checksum="",
                    upload_date=datetime.utcnow(),
                    storage_backend=self.backend,
                    storage_path="",
                    status=FileStatus.FAILED
                ),
                errors=errors,
                warnings=warnings
            )
    
    def get_file(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve file content by ID
        
        Args:
            file_id: File identifier
            
        Returns:
            File content as bytes or None if not found
        """
        
        metadata = self.metadata_store.get(file_id)
        if not metadata:
            logger.warning(f"File not found: {file_id}")
            return None
        
        try:
            # Update access statistics
            metadata.access_count += 1
            metadata.last_accessed = datetime.utcnow()
            
            # Retrieve from storage backend
            if metadata.storage_backend == StorageBackend.LOCAL:
                return self._get_local(metadata.storage_path)
            elif metadata.storage_backend == StorageBackend.S3:
                return self._get_s3(metadata.storage_path)
            elif metadata.storage_backend == StorageBackend.MINIO:
                return self._get_minio(metadata.storage_path)
            
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            return None
    
    def get_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata by ID"""
        return self.metadata_store.get(file_id)
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete file and its metadata
        
        Args:
            file_id: File identifier
            
        Returns:
            True if deleted successfully
        """
        
        metadata = self.metadata_store.get(file_id)
        if not metadata:
            return False
        
        try:
            # Delete from storage backend
            success = False
            if metadata.storage_backend == StorageBackend.LOCAL:
                success = self._delete_local(metadata.storage_path)
            elif metadata.storage_backend == StorageBackend.S3:
                success = self._delete_s3(metadata.storage_path)
            elif metadata.storage_backend == StorageBackend.MINIO:
                success = self._delete_minio(metadata.storage_path)
            
            # Delete thumbnails
            if metadata.thumbnails:
                for thumbnail_path in metadata.thumbnails.values():
                    try:
                        if metadata.storage_backend == StorageBackend.LOCAL:
                            self._delete_local(thumbnail_path)
                        elif metadata.storage_backend == StorageBackend.S3:
                            self._delete_s3(thumbnail_path)
                        elif metadata.storage_backend == StorageBackend.MINIO:
                            self._delete_minio(thumbnail_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete thumbnail: {e}")
            
            # Remove from metadata store
            if success:
                del self.metadata_store[file_id]
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def generate_secure_url(self, 
                           file_id: str, 
                           expires_in: int = 3600) -> Optional[str]:
        """
        Generate secure URL with expiration
        
        Args:
            file_id: File identifier
            expires_in: URL expiration time in seconds
            
        Returns:
            Secure URL or None if not available
        """
        
        metadata = self.metadata_store.get(file_id)
        if not metadata:
            return None
        
        try:
            if metadata.storage_backend == StorageBackend.S3:
                return self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.config.s3_bucket, 'Key': metadata.storage_path},
                    ExpiresIn=expires_in
                )
            elif metadata.storage_backend == StorageBackend.MINIO:
                return self.minio_client.presigned_get_object(
                    self.config.s3_bucket,
                    metadata.storage_path,
                    expires=timedelta(seconds=expires_in)
                )
            elif metadata.storage_backend == StorageBackend.LOCAL:
                # For local storage, return a path that can be served by the web server
                # This would need to be implemented in your API layer
                return f"/api/files/{file_id}/download"
            
        except Exception as e:
            logger.error(f"Failed to generate secure URL for {file_id}: {e}")
            return None
    
    def get_public_url(self, file_id: str) -> Optional[str]:
        """Get public URL if available"""
        
        metadata = self.metadata_store.get(file_id)
        if not metadata:
            return None
        
        if self._is_public_readable():
            if self.backend in [StorageBackend.S3, StorageBackend.MINIO]:
                endpoint = self.config.s3_endpoint_url or self.config.minio_endpoint
                return f"{endpoint}/{self.config.s3_bucket}/{metadata.storage_path}"
            elif self.backend == StorageBackend.LOCAL:
                return f"/api/files/{file_id}/public"
        
        return None
    
    def list_files(self, 
                   limit: int = 100, 
                   offset: int = 0,
                   tags: Optional[List[str]] = None,
                   content_type: Optional[str] = None) -> List[FileMetadata]:
        """
        List files with filtering and pagination
        
        Args:
            limit: Maximum number of files to return
            offset: Number of files to skip
            tags: Filter by tags
            content_type: Filter by content type
            
        Returns:
            List of file metadata
        """
        
        files = list(self.metadata_store.values())
        
        # Apply filters
        if tags:
            files = [f for f in files if any(tag in f.tags for tag in tags)]
        
        if content_type:
            files = [f for f in files if f.content_type.startswith(content_type)]
        
        # Sort by upload date (newest first)
        files.sort(key=lambda f: f.upload_date, reverse=True)
        
        # Apply pagination
        return files[offset:offset + limit]
    
    def _validate_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate uploaded file"""
        
        errors = []
        warnings = []
        quarantine = False
        
        # Check file size
        if len(file_data) > self.config.max_file_size:
            errors.append(f"File size {len(file_data)} exceeds maximum {self.config.max_file_size}")
        
        # Check file extension
        if self.config.allowed_extensions:
            ext = Path(filename).suffix.lower()
            if ext not in self.config.allowed_extensions:
                errors.append(f"File extension {ext} not allowed")
        
        # Detect file type
        try:
            kind = filetype.guess(file_data)
            if kind:
                content_type = kind.mime
                detected_ext = f".{kind.extension}"
                
                # Check if extension matches content
                actual_ext = Path(filename).suffix.lower()
                if actual_ext and actual_ext != detected_ext:
                    warnings.append(f"File extension {actual_ext} doesn't match content type {content_type}")
            else:
                content_type = "application/octet-stream"
                if self.config.quarantine_unknown_files:
                    warnings.append("Unknown file type detected")
                    quarantine = True
        except Exception as e:
            warnings.append(f"File type detection failed: {e}")
            content_type = "application/octet-stream"
        
        # Basic security scan (check for malicious patterns)
        try:
            self._security_scan(file_data, filename)
        except Exception as e:
            errors.append(f"Security scan failed: {e}")
            quarantine = True
        
        return {
            'errors': errors,
            'warnings': warnings,
            'quarantine': quarantine,
            'content_type': content_type
        }
    
    def _security_scan(self, file_data: bytes, filename: str):
        """Basic security scan for malicious content"""
        
        # Check for suspicious file patterns
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            b'onload=',
            b'onerror=',
            b'<?php',
            b'<%',
            b'exec(',
            b'system(',
            b'shell_exec('
        ]
        
        file_lower = file_data.lower()
        for pattern in suspicious_patterns:
            if pattern in file_lower:
                raise Exception(f"Suspicious content detected: {pattern.decode('utf-8', errors='ignore')}")
        
        # Check filename for suspicious patterns
        suspicious_names = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
        filename_lower = filename.lower()
        for ext in suspicious_names:
            if filename_lower.endswith(ext):
                raise Exception(f"Potentially dangerous file extension: {ext}")
    
    def _generate_thumbnails(self, 
                           file_data: bytes, 
                           file_id: str, 
                           content_type: str) -> Dict[str, str]:
        """Generate thumbnails for images and PDFs"""
        
        thumbnails = {}
        
        try:
            if content_type.startswith('image/'):
                thumbnails = self._generate_image_thumbnails(file_data, file_id)
            elif content_type == 'application/pdf':
                thumbnails = self._generate_pdf_thumbnails(file_data, file_id)
        except Exception as e:
            logger.warning(f"Thumbnail generation failed for {file_id}: {e}")
        
        return thumbnails
    
    def _generate_image_thumbnails(self, file_data: bytes, file_id: str) -> Dict[str, str]:
        """Generate thumbnails for image files"""
        
        thumbnails = {}
        sizes = {'small': (150, 150), 'medium': (300, 300), 'large': (800, 600)}
        
        try:
            with Image.open(io.BytesIO(file_data)) as img:
                # Fix orientation
                img = ImageOps.exif_transpose(img)
                
                for size_name, size in sizes.items():
                    # Create thumbnail
                    thumb = img.copy()
                    thumb.thumbnail(size, Image.Resampling.LANCZOS)
                    
                    # Save thumbnail
                    thumb_buffer = io.BytesIO()
                    thumb.save(thumb_buffer, format='JPEG', quality=85)
                    thumb_data = thumb_buffer.getvalue()
                    
                    # Upload thumbnail
                    thumb_path = f"thumbnails/{file_id}_{size_name}.jpg"
                    
                    if self.backend == StorageBackend.LOCAL:
                        self._upload_local(thumb_data, thumb_path)
                    elif self.backend == StorageBackend.S3:
                        self._upload_s3(thumb_data, thumb_path, 'image/jpeg')
                    elif self.backend == StorageBackend.MINIO:
                        self._upload_minio(thumb_data, thumb_path, 'image/jpeg')
                    
                    thumbnails[size_name] = thumb_path
                    
        except Exception as e:
            logger.error(f"Image thumbnail generation failed: {e}")
            raise
        
        return thumbnails
    
    def _generate_pdf_thumbnails(self, file_data: bytes, file_id: str) -> Dict[str, str]:
        """Generate thumbnails for PDF files"""
        
        thumbnails = {}
        
        try:
            # Convert first page to image
            images = convert_from_bytes(file_data, first_page=1, last_page=1, dpi=150)
            if images:
                first_page = images[0]
                
                sizes = {'small': (150, 200), 'medium': (300, 400), 'large': (600, 800)}
                
                for size_name, size in sizes.items():
                    # Create thumbnail
                    thumb = first_page.copy()
                    thumb.thumbnail(size, Image.Resampling.LANCZOS)
                    
                    # Save thumbnail
                    thumb_buffer = io.BytesIO()
                    thumb.save(thumb_buffer, format='JPEG', quality=85)
                    thumb_data = thumb_buffer.getvalue()
                    
                    # Upload thumbnail
                    thumb_path = f"thumbnails/{file_id}_pdf_{size_name}.jpg"
                    
                    if self.backend == StorageBackend.LOCAL:
                        self._upload_local(thumb_data, thumb_path)
                    elif self.backend == StorageBackend.S3:
                        self._upload_s3(thumb_data, thumb_path, 'image/jpeg')
                    elif self.backend == StorageBackend.MINIO:
                        self._upload_minio(thumb_data, thumb_path, 'image/jpeg')
                    
                    thumbnails[size_name] = thumb_path
                    
        except Exception as e:
            logger.error(f"PDF thumbnail generation failed: {e}")
            raise
        
        return thumbnails
    
    def _generate_storage_path(self, file_id: str, filename: str) -> str:
        """Generate storage path for file"""
        
        # Use date-based directory structure
        now = datetime.utcnow()
        date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
        
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        
        return f"files/{date_path}/{file_id}_{safe_filename}"
    
    def _find_by_checksum(self, checksum: str) -> Optional[FileMetadata]:
        """Find existing file by checksum"""
        
        for metadata in self.metadata_store.values():
            if metadata.checksum == checksum:
                return metadata
        return None
    
    def _handle_duplicate(self, 
                         existing_file: FileMetadata,
                         file_data: bytes,
                         filename: str,
                         metadata: Optional[Dict[str, Any]]) -> UploadResult:
        """Handle duplicate file upload"""
        
        # Create new version or return existing
        if self.config.enable_versioning:
            # Create new version
            new_version = existing_file.version + 1
            new_metadata = existing_file
            new_metadata.version = new_version
            new_metadata.original_filename = filename
            new_metadata.upload_date = datetime.utcnow()
            if metadata:
                new_metadata.metadata.update(metadata)
            
            return UploadResult(
                file_id=existing_file.file_id,
                file_metadata=new_metadata,
                secure_url=self.generate_secure_url(existing_file.file_id),
                warnings=["File already exists, created new version"]
            )
        else:
            # Return existing file
            return UploadResult(
                file_id=existing_file.file_id,
                file_metadata=existing_file,
                secure_url=self.generate_secure_url(existing_file.file_id),
                warnings=["File already exists, returning existing version"]
            )
    
    def _quarantine_file(self, 
                        file_data: bytes, 
                        filename: str, 
                        errors: List[str]) -> UploadResult:
        """Quarantine suspicious file"""
        
        file_id = str(uuid.uuid4())
        quarantine_path = f"quarantine/{file_id}_{filename}"
        
        # Store in quarantine
        if self.backend == StorageBackend.LOCAL:
            self._upload_local(file_data, quarantine_path)
        elif self.backend == StorageBackend.S3:
            self._upload_s3(file_data, quarantine_path, 'application/octet-stream')
        elif self.backend == StorageBackend.MINIO:
            self._upload_minio(file_data, quarantine_path, 'application/octet-stream')
        
        # Create quarantined metadata
        quarantine_metadata = FileMetadata(
            file_id=file_id,
            original_filename=filename,
            content_type='application/octet-stream',
            file_size=len(file_data),
            checksum=hashlib.sha256(file_data).hexdigest(),
            upload_date=datetime.utcnow(),
            storage_backend=self.backend,
            storage_path=quarantine_path,
            status=FileStatus.QUARANTINED
        )
        
        return UploadResult(
            file_id=file_id,
            file_metadata=quarantine_metadata,
            errors=errors
        )
    
    def _is_public_readable(self) -> bool:
        """Check if storage backend supports public URLs"""
        # This would depend on your bucket/storage configuration
        return False  # Conservative default
    
    def _get_thumbnail_urls(self, thumbnails: Dict[str, str]) -> Dict[str, str]:
        """Get URLs for thumbnails"""
        urls = {}
        for size, path in thumbnails.items():
            # This is simplified - you'd generate proper URLs based on your setup
            urls[size] = f"/api/thumbnails/{path}"
        return urls
    
    # Storage backend implementations
    def _upload_local(self, file_data: bytes, storage_path: str) -> bool:
        """Upload to local filesystem"""
        try:
            full_path = Path(self.config.local_base_path) / storage_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(file_data)
            return True
        except Exception as e:
            logger.error(f"Local upload failed: {e}")
            return False
    
    def _upload_s3(self, file_data: bytes, storage_path: str, content_type: str) -> bool:
        """Upload to S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=storage_path,
                Body=file_data,
                ContentType=content_type
            )
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def _upload_minio(self, file_data: bytes, storage_path: str, content_type: str) -> bool:
        """Upload to MinIO"""
        try:
            self.minio_client.put_object(
                self.config.s3_bucket,
                storage_path,
                io.BytesIO(file_data),
                len(file_data),
                content_type=content_type
            )
            return True
        except Exception as e:
            logger.error(f"MinIO upload failed: {e}")
            return False
    
    def _get_local(self, storage_path: str) -> Optional[bytes]:
        """Get file from local filesystem"""
        try:
            full_path = Path(self.config.local_base_path) / storage_path
            with open(full_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Local get failed: {e}")
            return None
    
    def _get_s3(self, storage_path: str) -> Optional[bytes]:
        """Get file from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=storage_path
            )
            return response['Body'].read()
        except Exception as e:
            logger.error(f"S3 get failed: {e}")
            return None
    
    def _get_minio(self, storage_path: str) -> Optional[bytes]:
        """Get file from MinIO"""
        try:
            response = self.minio_client.get_object(
                self.config.s3_bucket,
                storage_path
            )
            return response.read()
        except Exception as e:
            logger.error(f"MinIO get failed: {e}")
            return None
    
    def _delete_local(self, storage_path: str) -> bool:
        """Delete file from local filesystem"""
        try:
            full_path = Path(self.config.local_base_path) / storage_path
            full_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Local delete failed: {e}")
            return False
    
    def _delete_s3(self, storage_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(
                Bucket=self.config.s3_bucket,
                Key=storage_path
            )
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False
    
    def _delete_minio(self, storage_path: str) -> bool:
        """Delete file from MinIO"""
        try:
            self.minio_client.remove_object(
                self.config.s3_bucket,
                storage_path
            )
            return True
        except Exception as e:
            logger.error(f"MinIO delete failed: {e}")
            return False


# Convenience functions
def create_storage_service(backend: StorageBackend = StorageBackend.LOCAL,
                          **kwargs) -> StorageService:
    """Create storage service with default configuration"""
    
    config = StorageConfig(backend=backend, **kwargs)
    return StorageService(config)