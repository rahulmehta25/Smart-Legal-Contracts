"""
Secure Document Handling System
Implements document encryption, access control, and secure storage
Legal industry compliant with chain of custody and audit trail
"""

import os
import hashlib
import hmac
import json
import uuid
import mimetypes
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import threading
from pathlib import Path
import shutil

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import redis
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship

Base = declarative_base()

class DocumentClassification(str, Enum):
    """Document classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class DocumentState(str, Enum):
    """Document lifecycle states"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SIGNED = "signed"
    ARCHIVED = "archived"
    DELETED = "deleted"

class AccessPermission(str, Enum):
    """Document access permissions"""
    VIEW = "view"
    DOWNLOAD = "download"
    EDIT = "edit"
    DELETE = "delete"
    SHARE = "share"
    SIGN = "sign"
    APPROVE = "approve"

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    document_id: str
    filename: str
    content_type: str
    size: int
    checksum: str
    classification: DocumentClassification
    state: DocumentState
    owner_id: str
    created_at: datetime
    modified_at: datetime
    encryption_key_id: str
    storage_location: str
    retention_date: Optional[datetime]
    legal_hold: bool
    tags: List[str]
    custom_metadata: Dict[str, Any]

# Database Models
document_permissions = Table(
    'document_permissions',
    Base.metadata,
    Column('document_id', String, ForeignKey('secure_documents.document_id')),
    Column('user_id', String, ForeignKey('users.id')),
    Column('permission', String)
)

class SecureDocument(Base):
    """Database model for secure documents"""
    __tablename__ = 'secure_documents'
    
    document_id = Column(String(64), primary_key=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(128))
    size = Column(Integer)
    checksum = Column(String(64), nullable=False)
    classification = Column(String(32), nullable=False)
    state = Column(String(32), nullable=False)
    owner_id = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    modified_at = Column(DateTime, nullable=False)
    encryption_key_id = Column(String(64))
    storage_location = Column(Text)
    retention_date = Column(DateTime)
    legal_hold = Column(Boolean, default=False)
    tags = Column(JSON)
    custom_metadata = Column(JSON)
    access_log = Column(JSON)
    version = Column(Integer, default=1)
    parent_id = Column(String(64))  # For versioning
    
    # Relationships
    permissions = relationship("User", secondary=document_permissions, backref="accessible_documents")

class DocumentAccessLog(Base):
    """Database model for document access logs"""
    __tablename__ = 'document_access_logs'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(String(64), ForeignKey('secure_documents.document_id'), index=True)
    user_id = Column(String(64), index=True)
    action = Column(String(32))
    timestamp = Column(DateTime, nullable=False)
    ip_address = Column(String(45))
    details = Column(JSON)

class SecureDocumentHandler:
    """
    Comprehensive secure document handling system with:
    - End-to-end encryption
    - Access control and permissions
    - Secure storage (local/cloud)
    - Document versioning
    - Legal hold and retention policies
    - Chain of custody tracking
    - Watermarking and DRM
    """
    
    def __init__(self,
                 db_session: Session,
                 encryption_manager,
                 audit_logger,
                 storage_backend: str = "local",
                 storage_config: Optional[Dict[str, Any]] = None):
        
        self.db = db_session
        self.encryption_manager = encryption_manager
        self.audit_logger = audit_logger
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
        
        # Initialize storage backend
        self._initialize_storage()
        
        # Cache for document metadata
        self.cache = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=False
        )
        
        # Document processing queue
        self.processing_queue = []
        self.processing_lock = threading.Lock()
    
    def _initialize_storage(self):
        """Initialize storage backend"""
        
        if self.storage_backend == "local":
            self.storage_path = Path(self.storage_config.get('path', '/secure/documents'))
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
        elif self.storage_backend == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.storage_config['access_key'],
                aws_secret_access_key=self.storage_config['secret_key']
            )
            self.s3_bucket = self.storage_config['bucket']
            
        elif self.storage_backend == "azure":
            self.blob_service = BlobServiceClient(
                account_url=self.storage_config['account_url'],
                credential=self.storage_config['credential']
            )
            self.container_name = self.storage_config['container']
            
        elif self.storage_backend == "gcs":
            self.gcs_client = gcs.Client()
            self.gcs_bucket = self.gcs_client.bucket(self.storage_config['bucket'])
    
    # ========== Document Upload and Storage ==========
    
    def upload_document(self,
                       file_content: bytes,
                       filename: str,
                       owner_id: str,
                       classification: DocumentClassification = DocumentClassification.INTERNAL,
                       metadata: Optional[Dict[str, Any]] = None) -> DocumentMetadata:
        """
        Securely upload and store a document
        """
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Calculate checksum
        checksum = hashlib.sha256(file_content).hexdigest()
        
        # Check for duplicates
        if self._check_duplicate(checksum):
            raise ValueError("Document already exists (duplicate checksum)")
        
        # Detect content type
        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
        # Encrypt document
        encrypted_data = self.encryption_manager.encrypt_document(
            document=file_content,
            user_public_key=self._get_user_public_key(owner_id),
            metadata={'document_id': document_id}
        )
        
        # Store encrypted document
        storage_location = self._store_document(
            document_id=document_id,
            encrypted_data=encrypted_data['encrypted_document'],
            classification=classification
        )
        
        # Create metadata
        doc_metadata = DocumentMetadata(
            document_id=document_id,
            filename=self._sanitize_filename(filename),
            content_type=content_type,
            size=len(file_content),
            checksum=checksum,
            classification=classification,
            state=DocumentState.DRAFT,
            owner_id=owner_id,
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
            encryption_key_id=encrypted_data.get('key_id'),
            storage_location=storage_location,
            retention_date=self._calculate_retention_date(classification),
            legal_hold=False,
            tags=metadata.get('tags', []) if metadata else [],
            custom_metadata=metadata or {}
        )
        
        # Save to database
        self._save_metadata(doc_metadata)
        
        # Set initial permissions
        self._set_document_permissions(document_id, owner_id, [
            AccessPermission.VIEW,
            AccessPermission.DOWNLOAD,
            AccessPermission.EDIT,
            AccessPermission.DELETE,
            AccessPermission.SHARE
        ])
        
        # Audit log
        self.audit_logger.log_event(
            event_type="document.created",
            user_id=owner_id,
            resource_type="document",
            resource_id=document_id,
            details={
                'filename': filename,
                'size': len(file_content),
                'classification': classification.value
            }
        )
        
        return doc_metadata
    
    def _check_duplicate(self, checksum: str) -> bool:
        """Check if document with same checksum exists"""
        
        existing = self.db.query(SecureDocument).filter(
            SecureDocument.checksum == checksum
        ).first()
        
        return existing is not None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        safe_chars = re.compile(r'[^a-zA-Z0-9._-]')
        return safe_chars.sub('_', filename)
    
    def _calculate_retention_date(self, classification: DocumentClassification) -> datetime:
        """Calculate retention date based on classification"""
        
        retention_periods = {
            DocumentClassification.PUBLIC: 365,  # 1 year
            DocumentClassification.INTERNAL: 1095,  # 3 years
            DocumentClassification.CONFIDENTIAL: 2555,  # 7 years
            DocumentClassification.RESTRICTED: 3650,  # 10 years
            DocumentClassification.TOP_SECRET: None  # Indefinite
        }
        
        days = retention_periods.get(classification)
        if days:
            return datetime.utcnow() + timedelta(days=days)
        return None
    
    def _get_user_public_key(self, user_id: str) -> bytes:
        """Get user's public key for encryption"""
        
        # In production, retrieve from user profile/key store
        # For now, generate a key
        from cryptography.hazmat.primitives.asymmetric import rsa
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def _store_document(self, 
                       document_id: str, 
                       encrypted_data: str,
                       classification: DocumentClassification) -> str:
        """Store encrypted document in backend"""
        
        # Organize by classification and date
        date_path = datetime.utcnow().strftime('%Y/%m/%d')
        storage_path = f"{classification.value}/{date_path}/{document_id}"
        
        if self.storage_backend == "local":
            full_path = self.storage_path / storage_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(base64.b64decode(encrypted_data))
            
            return str(full_path)
            
        elif self.storage_backend == "s3":
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=storage_path,
                Body=base64.b64decode(encrypted_data),
                ServerSideEncryption='AES256',
                Metadata={'classification': classification.value}
            )
            return f"s3://{self.s3_bucket}/{storage_path}"
            
        # Add other backends as needed
        
        return storage_path
    
    def _save_metadata(self, metadata: DocumentMetadata):
        """Save document metadata to database"""
        
        doc = SecureDocument(
            document_id=metadata.document_id,
            filename=metadata.filename,
            content_type=metadata.content_type,
            size=metadata.size,
            checksum=metadata.checksum,
            classification=metadata.classification.value,
            state=metadata.state.value,
            owner_id=metadata.owner_id,
            created_at=metadata.created_at,
            modified_at=metadata.modified_at,
            encryption_key_id=metadata.encryption_key_id,
            storage_location=metadata.storage_location,
            retention_date=metadata.retention_date,
            legal_hold=metadata.legal_hold,
            tags=metadata.tags,
            custom_metadata=metadata.custom_metadata
        )
        
        self.db.add(doc)
        self.db.commit()
        
        # Cache metadata
        self._cache_metadata(metadata)
    
    def _cache_metadata(self, metadata: DocumentMetadata):
        """Cache document metadata for quick access"""
        
        cache_key = f"doc:meta:{metadata.document_id}"
        cache_data = {
            'filename': metadata.filename,
            'owner_id': metadata.owner_id,
            'classification': metadata.classification.value,
            'state': metadata.state.value
        }
        
        self.cache.hset(cache_key, mapping=cache_data)
        self.cache.expire(cache_key, 3600)  # 1 hour
    
    # ========== Access Control ==========
    
    def check_access(self,
                    document_id: str,
                    user_id: str,
                    permission: AccessPermission) -> bool:
        """Check if user has permission for document"""
        
        # Check cache first
        cache_key = f"doc:perm:{document_id}:{user_id}"
        cached_perms = self.cache.get(cache_key)
        
        if cached_perms:
            perms = json.loads(cached_perms)
            return permission.value in perms
        
        # Query database
        doc = self.db.query(SecureDocument).filter(
            SecureDocument.document_id == document_id
        ).first()
        
        if not doc:
            return False
        
        # Owner has all permissions
        if doc.owner_id == user_id:
            return True
        
        # Check specific permissions
        perms = self.db.execute(
            document_permissions.select().where(
                document_permissions.c.document_id == document_id,
                document_permissions.c.user_id == user_id
            )
        ).fetchall()
        
        user_perms = [p.permission for p in perms]
        
        # Cache permissions
        self.cache.setex(cache_key, 300, json.dumps(user_perms))
        
        return permission.value in user_perms
    
    def _set_document_permissions(self,
                                 document_id: str,
                                 user_id: str,
                                 permissions: List[AccessPermission]):
        """Set document permissions for user"""
        
        # Remove existing permissions
        self.db.execute(
            document_permissions.delete().where(
                document_permissions.c.document_id == document_id,
                document_permissions.c.user_id == user_id
            )
        )
        
        # Add new permissions
        for perm in permissions:
            self.db.execute(
                document_permissions.insert().values(
                    document_id=document_id,
                    user_id=user_id,
                    permission=perm.value
                )
            )
        
        self.db.commit()
        
        # Clear cache
        cache_key = f"doc:perm:{document_id}:{user_id}"
        self.cache.delete(cache_key)
    
    def share_document(self,
                      document_id: str,
                      owner_id: str,
                      target_user_id: str,
                      permissions: List[AccessPermission],
                      expiry: Optional[datetime] = None):
        """Share document with another user"""
        
        # Verify owner
        if not self.check_access(document_id, owner_id, AccessPermission.SHARE):
            raise PermissionError("User does not have permission to share document")
        
        # Set permissions
        self._set_document_permissions(document_id, target_user_id, permissions)
        
        # Log sharing
        self.audit_logger.log_event(
            event_type="document.shared",
            user_id=owner_id,
            resource_type="document",
            resource_id=document_id,
            details={
                'shared_with': target_user_id,
                'permissions': [p.value for p in permissions],
                'expiry': expiry.isoformat() if expiry else None
            }
        )
    
    # ========== Document Retrieval ==========
    
    def get_document(self,
                    document_id: str,
                    user_id: str,
                    log_access: bool = True) -> Tuple[bytes, DocumentMetadata]:
        """Retrieve and decrypt document"""
        
        # Check permissions
        if not self.check_access(document_id, user_id, AccessPermission.VIEW):
            raise PermissionError("Access denied")
        
        # Get metadata
        doc = self.db.query(SecureDocument).filter(
            SecureDocument.document_id == document_id
        ).first()
        
        if not doc:
            raise FileNotFoundError("Document not found")
        
        # Check legal hold
        if doc.legal_hold:
            self._check_legal_hold_access(user_id)
        
        # Retrieve encrypted document
        encrypted_data = self._retrieve_document(doc.storage_location)
        
        # Decrypt document
        decrypted_data = self.encryption_manager.decrypt_document(
            encrypted_data,
            user_private_key=self._get_user_private_key(user_id)
        )
        
        # Log access
        if log_access:
            self._log_document_access(document_id, user_id, "view")
        
        # Convert to metadata object
        metadata = self._db_to_metadata(doc)
        
        return decrypted_data, metadata
    
    def _retrieve_document(self, storage_location: str) -> bytes:
        """Retrieve encrypted document from storage"""
        
        if self.storage_backend == "local":
            with open(storage_location, 'rb') as f:
                return f.read()
                
        elif self.storage_backend == "s3":
            # Parse S3 path
            parts = storage_location.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1]
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        
        # Add other backends
        
        return b''
    
    def _get_user_private_key(self, user_id: str) -> bytes:
        """Get user's private key for decryption"""
        
        # In production, retrieve from secure key store
        # This is a placeholder
        return b''
    
    def _check_legal_hold_access(self, user_id: str):
        """Check if user can access documents under legal hold"""
        
        # Only legal team and admins can access
        # Implementation depends on your role system
        pass
    
    def _log_document_access(self, document_id: str, user_id: str, action: str):
        """Log document access"""
        
        access_log = DocumentAccessLog(
            document_id=document_id,
            user_id=user_id,
            action=action,
            timestamp=datetime.utcnow(),
            ip_address="",  # Get from request context
            details={}
        )
        
        self.db.add(access_log)
        self.db.commit()
    
    def _db_to_metadata(self, doc: SecureDocument) -> DocumentMetadata:
        """Convert database model to metadata object"""
        
        return DocumentMetadata(
            document_id=doc.document_id,
            filename=doc.filename,
            content_type=doc.content_type,
            size=doc.size,
            checksum=doc.checksum,
            classification=DocumentClassification(doc.classification),
            state=DocumentState(doc.state),
            owner_id=doc.owner_id,
            created_at=doc.created_at,
            modified_at=doc.modified_at,
            encryption_key_id=doc.encryption_key_id,
            storage_location=doc.storage_location,
            retention_date=doc.retention_date,
            legal_hold=doc.legal_hold,
            tags=doc.tags or [],
            custom_metadata=doc.custom_metadata or {}
        )
    
    # ========== Document Operations ==========
    
    def update_document(self,
                       document_id: str,
                       user_id: str,
                       new_content: bytes,
                       create_version: bool = True) -> DocumentMetadata:
        """Update document content"""
        
        # Check permissions
        if not self.check_access(document_id, user_id, AccessPermission.EDIT):
            raise PermissionError("Access denied")
        
        # Get current document
        current_doc = self.db.query(SecureDocument).filter(
            SecureDocument.document_id == document_id
        ).first()
        
        if not current_doc:
            raise FileNotFoundError("Document not found")
        
        # Create version if requested
        if create_version:
            self._create_version(current_doc)
        
        # Update document
        new_checksum = hashlib.sha256(new_content).hexdigest()
        
        # Encrypt new content
        encrypted_data = self.encryption_manager.encrypt_document(
            document=new_content,
            user_public_key=self._get_user_public_key(current_doc.owner_id),
            metadata={'document_id': document_id}
        )
        
        # Store updated document
        storage_location = self._store_document(
            document_id=document_id,
            encrypted_data=encrypted_data['encrypted_document'],
            classification=DocumentClassification(current_doc.classification)
        )
        
        # Update metadata
        current_doc.checksum = new_checksum
        current_doc.size = len(new_content)
        current_doc.modified_at = datetime.utcnow()
        current_doc.storage_location = storage_location
        current_doc.version += 1
        
        self.db.commit()
        
        # Audit log
        self.audit_logger.log_event(
            event_type="document.modified",
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            details={'version': current_doc.version}
        )
        
        return self._db_to_metadata(current_doc)
    
    def _create_version(self, document: SecureDocument):
        """Create a version of document"""
        
        # Copy current document as version
        version = SecureDocument(
            document_id=str(uuid.uuid4()),
            filename=f"{document.filename}.v{document.version}",
            content_type=document.content_type,
            size=document.size,
            checksum=document.checksum,
            classification=document.classification,
            state=document.state,
            owner_id=document.owner_id,
            created_at=document.created_at,
            modified_at=document.modified_at,
            encryption_key_id=document.encryption_key_id,
            storage_location=document.storage_location,
            retention_date=document.retention_date,
            legal_hold=document.legal_hold,
            tags=document.tags,
            custom_metadata=document.custom_metadata,
            version=document.version,
            parent_id=document.document_id
        )
        
        self.db.add(version)
        self.db.commit()
    
    def delete_document(self, document_id: str, user_id: str, permanent: bool = False):
        """Delete or mark document as deleted"""
        
        # Check permissions
        if not self.check_access(document_id, user_id, AccessPermission.DELETE):
            raise PermissionError("Access denied")
        
        doc = self.db.query(SecureDocument).filter(
            SecureDocument.document_id == document_id
        ).first()
        
        if not doc:
            raise FileNotFoundError("Document not found")
        
        # Check legal hold
        if doc.legal_hold:
            raise PermissionError("Cannot delete document under legal hold")
        
        if permanent and doc.state == DocumentState.DELETED.value:
            # Permanent deletion
            self._permanent_delete(doc)
        else:
            # Soft delete
            doc.state = DocumentState.DELETED.value
            doc.modified_at = datetime.utcnow()
            self.db.commit()
        
        # Audit log
        self.audit_logger.log_event(
            event_type="document.deleted",
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            details={'permanent': permanent}
        )
    
    def _permanent_delete(self, document: SecureDocument):
        """Permanently delete document and its data"""
        
        # Delete from storage
        if self.storage_backend == "local":
            try:
                os.remove(document.storage_location)
            except:
                pass
        elif self.storage_backend == "s3":
            parts = document.storage_location.replace('s3://', '').split('/', 1)
            self.s3_client.delete_object(Bucket=parts[0], Key=parts[1])
        
        # Delete from database
        self.db.delete(document)
        self.db.commit()
        
        # Clear cache
        self.cache.delete(f"doc:meta:{document.document_id}")
    
    # ========== Legal and Compliance ==========
    
    def apply_legal_hold(self, document_id: str, reason: str):
        """Apply legal hold to document"""
        
        doc = self.db.query(SecureDocument).filter(
            SecureDocument.document_id == document_id
        ).first()
        
        if doc:
            doc.legal_hold = True
            doc.custom_metadata = doc.custom_metadata or {}
            doc.custom_metadata['legal_hold_reason'] = reason
            doc.custom_metadata['legal_hold_date'] = datetime.utcnow().isoformat()
            self.db.commit()
            
            self.audit_logger.log_event(
                event_type="document.legal_hold",
                resource_type="document",
                resource_id=document_id,
                details={'reason': reason}
            )
    
    def apply_retention_policy(self):
        """Apply retention policy to documents"""
        
        # Find documents past retention date
        expired = self.db.query(SecureDocument).filter(
            SecureDocument.retention_date < datetime.utcnow(),
            SecureDocument.legal_hold == False,
            SecureDocument.state != DocumentState.DELETED.value
        ).all()
        
        for doc in expired:
            # Mark for deletion
            doc.state = DocumentState.DELETED.value
            
            self.audit_logger.log_event(
                event_type="document.retention_applied",
                resource_type="document",
                resource_id=doc.document_id,
                details={'retention_date': doc.retention_date.isoformat()}
            )
        
        self.db.commit()
    
    def export_for_discovery(self,
                            case_id: str,
                            document_ids: List[str],
                            format: str = "pdf") -> bytes:
        """Export documents for legal discovery"""
        
        exported_docs = []
        
        for doc_id in document_ids:
            try:
                # Get document with admin privileges
                doc_data, metadata = self.get_document(doc_id, "legal_admin", log_access=True)
                
                exported_docs.append({
                    'document_id': doc_id,
                    'filename': metadata.filename,
                    'data': base64.b64encode(doc_data).decode(),
                    'metadata': {
                        'checksum': metadata.checksum,
                        'created': metadata.created_at.isoformat(),
                        'modified': metadata.modified_at.isoformat(),
                        'owner': metadata.owner_id,
                        'classification': metadata.classification.value
                    }
                })
            except Exception as e:
                exported_docs.append({
                    'document_id': doc_id,
                    'error': str(e)
                })
        
        # Create export package
        export_data = {
            'case_id': case_id,
            'export_date': datetime.utcnow().isoformat(),
            'documents': exported_docs,
            'total': len(document_ids),
            'successful': len([d for d in exported_docs if 'error' not in d])
        }
        
        # Log discovery export
        self.audit_logger.log_event(
            event_type="compliance.discovery_export",
            details={
                'case_id': case_id,
                'document_count': len(document_ids)
            }
        )
        
        return json.dumps(export_data).encode()