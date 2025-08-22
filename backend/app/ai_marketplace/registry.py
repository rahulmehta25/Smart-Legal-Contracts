"""
Model Registry and Versioning System

Manages model registration, versioning, metadata, and lifecycle.
"""

import os
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import semver
import pickle
import joblib
import torch
import tensorflow as tf
from pathlib import Path
import yaml
import git
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Float, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import redis
import logging

Base = declarative_base()
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    DRAFT = "draft"
    TESTING = "testing"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Types of models supported"""
    CLASSIFICATION = "classification"
    NER = "named_entity_recognition"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    ENSEMBLE = "ensemble"
    FINE_TUNED = "fine_tuned"
    CUSTOM = "custom"


class LegalDomain(Enum):
    """Legal domains for specialized models"""
    CONTRACT_LAW = "contract_law"
    CORPORATE_LAW = "corporate_law"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    EMPLOYMENT_LAW = "employment_law"
    INTERNATIONAL_ARBITRATION = "international_arbitration"
    COMMERCIAL_DISPUTES = "commercial_disputes"
    CONSTRUCTION_LAW = "construction_law"
    INSURANCE_LAW = "insurance_law"
    MARITIME_LAW = "maritime_law"
    ENERGY_LAW = "energy_law"
    GENERAL = "general"


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    name: str
    version: str
    type: ModelType
    domain: LegalDomain
    description: str
    author: str
    organization: Optional[str]
    license: str
    framework: str
    requirements: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_data: Dict[str, Any]
    tags: List[str]
    dependencies: List[str]
    hardware_requirements: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    checksum: str


class ModelDB(Base):
    """Database model for registry"""
    __tablename__ = 'ai_models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    type = Column(String(50), nullable=False)
    domain = Column(String(100), nullable=False)
    status = Column(String(50), default=ModelStatus.DRAFT.value)
    description = Column(Text)
    author = Column(String(255), nullable=False)
    organization = Column(String(255))
    license = Column(String(100), nullable=False)
    framework = Column(String(50))
    requirements = Column(JSON)
    input_schema = Column(JSON)
    output_schema = Column(JSON)
    performance_metrics = Column(JSON)
    training_data = Column(JSON)
    tags = Column(JSON)
    dependencies = Column(JSON)
    hardware_requirements = Column(JSON)
    storage_path = Column(String(500))
    checksum = Column(String(128))
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions = relationship("ModelVersion", back_populates="model")
    reviews = relationship("ModelReview", back_populates="model")


class ModelVersion(Base):
    """Model version tracking"""
    __tablename__ = 'model_versions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey('ai_models.id'))
    version = Column(String(50), nullable=False)
    changelog = Column(Text)
    storage_path = Column(String(500))
    checksum = Column(String(128))
    is_stable = Column(Boolean, default=False)
    is_latest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model = relationship("ModelDB", back_populates="versions")


class ModelReview(Base):
    """Model reviews and ratings"""
    __tablename__ = 'model_reviews'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey('ai_models.id'))
    user_id = Column(String(255), nullable=False)
    rating = Column(Float, nullable=False)
    review = Column(Text)
    use_case = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model = relationship("ModelDB", back_populates="reviews")


class StorageBackend:
    """Abstract storage backend"""
    
    def upload(self, model_path: str, destination: str) -> str:
        raise NotImplementedError
    
    def download(self, source: str, destination: str) -> str:
        raise NotImplementedError
    
    def delete(self, path: str) -> bool:
        raise NotImplementedError
    
    def exists(self, path: str) -> bool:
        raise NotImplementedError


class S3Storage(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region)
    
    def upload(self, model_path: str, destination: str) -> str:
        try:
            self.s3_client.upload_file(model_path, self.bucket_name, destination)
            return f"s3://{self.bucket_name}/{destination}"
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def download(self, source: str, destination: str) -> str:
        try:
            key = source.replace(f"s3://{self.bucket_name}/", "")
            self.s3_client.download_file(self.bucket_name, key, destination)
            return destination
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    def delete(self, path: str) -> bool:
        try:
            key = path.replace(f"s3://{self.bucket_name}/", "")
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False
    
    def exists(self, path: str) -> bool:
        try:
            key = path.replace(f"s3://{self.bucket_name}/", "")
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except:
            return False


class LocalStorage(StorageBackend):
    """Local filesystem storage"""
    
    def __init__(self, base_path: str = "/var/lib/ai_marketplace/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload(self, model_path: str, destination: str) -> str:
        dest_path = self.base_path / destination
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest_path)
        return str(dest_path)
    
    def download(self, source: str, destination: str) -> str:
        shutil.copy2(source, destination)
        return destination
    
    def delete(self, path: str) -> bool:
        try:
            Path(path).unlink()
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    def exists(self, path: str) -> bool:
        return Path(path).exists()


class ModelRegistry:
    """
    Central registry for AI models in the marketplace
    """
    
    def __init__(self, 
                 db_url: str = "postgresql://localhost/ai_marketplace",
                 storage_backend: Optional[StorageBackend] = None,
                 cache_enabled: bool = True):
        
        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize storage
        self.storage = storage_backend or LocalStorage()
        
        # Initialize cache
        self.cache = redis.Redis(host='localhost', port=6379, db=0) if cache_enabled else None
        
        # Initialize git for version control
        self.repo_path = Path("/var/lib/ai_marketplace/repos")
        self.repo_path.mkdir(parents=True, exist_ok=True)
    
    def register_model(self, 
                      model_path: str,
                      metadata: ModelMetadata,
                      validation_results: Optional[Dict] = None) -> str:
        """
        Register a new model in the marketplace
        
        Args:
            model_path: Path to model file/directory
            metadata: Model metadata
            validation_results: Optional validation results
        
        Returns:
            Model ID
        """
        try:
            # Validate model format
            if not self._validate_model_format(model_path):
                raise ValueError("Invalid model format")
            
            # Calculate checksum
            checksum = self._calculate_checksum(model_path)
            metadata.checksum = checksum
            
            # Check for duplicates
            existing = self.session.query(ModelDB).filter_by(
                name=metadata.name,
                version=metadata.version
            ).first()
            
            if existing:
                raise ValueError(f"Model {metadata.name} v{metadata.version} already exists")
            
            # Create database entry
            model_db = ModelDB(
                name=metadata.name,
                version=metadata.version,
                type=metadata.type.value,
                domain=metadata.domain.value,
                description=metadata.description,
                author=metadata.author,
                organization=metadata.organization,
                license=metadata.license,
                framework=metadata.framework,
                requirements=metadata.requirements,
                input_schema=metadata.input_schema,
                output_schema=metadata.output_schema,
                performance_metrics=metadata.performance_metrics,
                training_data=metadata.training_data,
                tags=metadata.tags,
                dependencies=metadata.dependencies,
                hardware_requirements=metadata.hardware_requirements,
                checksum=checksum,
                status=ModelStatus.DRAFT.value
            )
            
            # Upload model to storage
            storage_key = f"{metadata.name}/{metadata.version}/model"
            storage_path = self.storage.upload(model_path, storage_key)
            model_db.storage_path = storage_path
            
            # Save metadata
            metadata_path = Path(model_path).parent / "metadata.yaml"
            with open(metadata_path, 'w') as f:
                yaml.dump(asdict(metadata), f)
            
            metadata_key = f"{metadata.name}/{metadata.version}/metadata.yaml"
            self.storage.upload(str(metadata_path), metadata_key)
            
            # Create version entry
            version = ModelVersion(
                model_id=model_db.id,
                version=metadata.version,
                storage_path=storage_path,
                checksum=checksum,
                is_latest=True
            )
            
            # Update cache
            if self.cache:
                cache_key = f"model:{metadata.name}:{metadata.version}"
                self.cache.setex(cache_key, 3600, json.dumps(asdict(metadata)))
            
            # Commit to database
            self.session.add(model_db)
            self.session.add(version)
            self.session.commit()
            
            # Initialize git repository for model
            self._init_model_repo(metadata.name, model_path)
            
            logger.info(f"Model {metadata.name} v{metadata.version} registered successfully")
            return str(model_db.id)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Model registration failed: {e}")
            raise
    
    def update_model(self, 
                    model_id: str,
                    new_version: str,
                    model_path: str,
                    changelog: str) -> bool:
        """
        Update existing model with new version
        
        Args:
            model_id: Model ID
            new_version: New version string
            model_path: Path to updated model
            changelog: Version changelog
        
        Returns:
            Success status
        """
        try:
            # Get existing model
            model = self.session.query(ModelDB).filter_by(id=model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Validate version increment
            if not self._validate_version_increment(model.version, new_version):
                raise ValueError(f"Invalid version increment: {model.version} -> {new_version}")
            
            # Calculate new checksum
            checksum = self._calculate_checksum(model_path)
            
            # Upload new version
            storage_key = f"{model.name}/{new_version}/model"
            storage_path = self.storage.upload(model_path, storage_key)
            
            # Update latest flags
            self.session.query(ModelVersion).filter_by(
                model_id=model_id,
                is_latest=True
            ).update({'is_latest': False})
            
            # Create new version entry
            new_version_entry = ModelVersion(
                model_id=model_id,
                version=new_version,
                changelog=changelog,
                storage_path=storage_path,
                checksum=checksum,
                is_latest=True
            )
            
            # Update model record
            model.version = new_version
            model.storage_path = storage_path
            model.checksum = checksum
            model.updated_at = datetime.utcnow()
            
            # Invalidate cache
            if self.cache:
                cache_key = f"model:{model.name}:*"
                for key in self.cache.scan_iter(match=cache_key):
                    self.cache.delete(key)
            
            # Commit changes
            self.session.add(new_version_entry)
            self.session.commit()
            
            # Update git repository
            self._commit_model_changes(model.name, new_version, changelog)
            
            logger.info(f"Model {model.name} updated to v{new_version}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Model update failed: {e}")
            raise
    
    def get_model(self, 
                 name: str,
                 version: Optional[str] = None) -> Tuple[ModelDB, str]:
        """
        Retrieve model from registry
        
        Args:
            name: Model name
            version: Optional version (latest if not specified)
        
        Returns:
            Model metadata and download path
        """
        try:
            # Check cache first
            if self.cache and version:
                cache_key = f"model:{name}:{version}"
                cached = self.cache.get(cache_key)
                if cached:
                    return json.loads(cached), None
            
            # Query database
            query = self.session.query(ModelDB).filter_by(name=name)
            
            if version:
                model = query.filter_by(version=version).first()
            else:
                model = query.filter_by(
                    status=ModelStatus.PUBLISHED.value
                ).order_by(ModelDB.created_at.desc()).first()
            
            if not model:
                raise ValueError(f"Model {name} not found")
            
            # Download model if needed
            local_path = Path(f"/tmp/models/{name}/{model.version}")
            local_path.mkdir(parents=True, exist_ok=True)
            
            model_file = local_path / "model"
            if not model_file.exists():
                self.storage.download(model.storage_path, str(model_file))
            
            # Update download count
            model.downloads += 1
            self.session.commit()
            
            return model, str(model_file)
            
        except Exception as e:
            logger.error(f"Model retrieval failed: {e}")
            raise
    
    def search_models(self,
                     query: Optional[str] = None,
                     type: Optional[ModelType] = None,
                     domain: Optional[LegalDomain] = None,
                     tags: Optional[List[str]] = None,
                     min_rating: Optional[float] = None,
                     limit: int = 20) -> List[ModelDB]:
        """
        Search models in registry
        
        Args:
            query: Search query
            type: Model type filter
            domain: Legal domain filter
            tags: Tag filters
            min_rating: Minimum rating filter
            limit: Result limit
        
        Returns:
            List of matching models
        """
        try:
            # Build query
            q = self.session.query(ModelDB).filter_by(
                status=ModelStatus.PUBLISHED.value,
                is_public=True
            )
            
            if query:
                q = q.filter(
                    (ModelDB.name.ilike(f"%{query}%")) |
                    (ModelDB.description.ilike(f"%{query}%"))
                )
            
            if type:
                q = q.filter_by(type=type.value)
            
            if domain:
                q = q.filter_by(domain=domain.value)
            
            if min_rating:
                q = q.filter(ModelDB.rating >= min_rating)
            
            if tags:
                for tag in tags:
                    q = q.filter(ModelDB.tags.contains([tag]))
            
            # Order by relevance and rating
            models = q.order_by(
                ModelDB.rating.desc(),
                ModelDB.downloads.desc()
            ).limit(limit).all()
            
            return models
            
        except Exception as e:
            logger.error(f"Model search failed: {e}")
            raise
    
    def rollback_version(self, model_id: str, target_version: str) -> bool:
        """
        Rollback model to previous version
        
        Args:
            model_id: Model ID
            target_version: Target version to rollback to
        
        Returns:
            Success status
        """
        try:
            # Get model and target version
            model = self.session.query(ModelDB).filter_by(id=model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            version = self.session.query(ModelVersion).filter_by(
                model_id=model_id,
                version=target_version
            ).first()
            
            if not version:
                raise ValueError(f"Version {target_version} not found")
            
            # Update model to target version
            model.version = target_version
            model.storage_path = version.storage_path
            model.checksum = version.checksum
            model.updated_at = datetime.utcnow()
            
            # Update version flags
            self.session.query(ModelVersion).filter_by(
                model_id=model_id,
                is_latest=True
            ).update({'is_latest': False})
            
            version.is_latest = True
            
            # Clear cache
            if self.cache:
                cache_key = f"model:{model.name}:*"
                for key in self.cache.scan_iter(match=cache_key):
                    self.cache.delete(key)
            
            self.session.commit()
            
            logger.info(f"Model {model.name} rolled back to v{target_version}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Rollback failed: {e}")
            raise
    
    def deprecate_model(self, model_id: str, reason: str) -> bool:
        """
        Mark model as deprecated
        
        Args:
            model_id: Model ID
            reason: Deprecation reason
        
        Returns:
            Success status
        """
        try:
            model = self.session.query(ModelDB).filter_by(id=model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            model.status = ModelStatus.DEPRECATED.value
            model.updated_at = datetime.utcnow()
            
            # Add deprecation notice to description
            model.description = f"[DEPRECATED: {reason}] {model.description}"
            
            self.session.commit()
            
            logger.info(f"Model {model.name} deprecated: {reason}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Deprecation failed: {e}")
            raise
    
    def _validate_model_format(self, model_path: str) -> bool:
        """Validate model file format"""
        path = Path(model_path)
        
        # Check for supported formats
        valid_extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.onnx', '.pb']
        
        if path.is_file():
            return path.suffix in valid_extensions
        elif path.is_dir():
            # Check for common model directory structures
            return any([
                (path / "saved_model.pb").exists(),  # TensorFlow SavedModel
                (path / "model.pkl").exists(),  # Scikit-learn
                (path / "pytorch_model.bin").exists(),  # PyTorch/Transformers
                (path / "model.onnx").exists()  # ONNX
            ])
        
        return False
    
    def _calculate_checksum(self, model_path: str) -> str:
        """Calculate SHA256 checksum of model"""
        sha256_hash = hashlib.sha256()
        
        path = Path(model_path)
        if path.is_file():
            with open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        else:
            # For directories, hash all files
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _validate_version_increment(self, current: str, new: str) -> bool:
        """Validate semantic version increment"""
        try:
            current_ver = semver.VersionInfo.parse(current)
            new_ver = semver.VersionInfo.parse(new)
            return new_ver > current_ver
        except:
            return False
    
    def _init_model_repo(self, model_name: str, model_path: str):
        """Initialize git repository for model version control"""
        repo_path = self.repo_path / model_name
        repo_path.mkdir(exist_ok=True)
        
        if not (repo_path / ".git").exists():
            repo = git.Repo.init(repo_path)
            
            # Copy model files
            if Path(model_path).is_file():
                shutil.copy2(model_path, repo_path / "model")
            else:
                shutil.copytree(model_path, repo_path / "model", dirs_exist_ok=True)
            
            # Initial commit
            repo.index.add(["*"])
            repo.index.commit(f"Initial commit for {model_name}")
    
    def _commit_model_changes(self, model_name: str, version: str, changelog: str):
        """Commit model changes to git"""
        repo_path = self.repo_path / model_name
        repo = git.Repo(repo_path)
        
        # Create version tag
        repo.create_tag(f"v{version}", message=changelog)
        
        # Commit changes
        repo.index.add(["*"])
        repo.index.commit(f"Version {version}: {changelog}")