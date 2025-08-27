"""
Base SQLAlchemy model classes for the arbitration detection system.
Provides common functionality and database configuration.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy import DateTime, String, Boolean, event, Column
from sqlalchemy import String
try:
    from sqlalchemy.dialects.postgresql import UUID
except ImportError:
    # Fallback for SQLite
    from sqlalchemy import String as UUID
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

# Type variable for model classes
ModelType = TypeVar("ModelType", bound="BaseModel")

Base = declarative_base()


class TimestampMixin:
    """Mixin class to add created_at and updated_at timestamps."""
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class BaseModel(Base, TimestampMixin):
    """
    Abstract base model class providing common functionality.
    
    Features:
    - UUID primary key
    - Timestamps (created_at, updated_at)
    - Common query methods
    - JSON serialization
    - Audit trail support
    """
    
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower() + 's'
    
    def to_dict(self, exclude: Optional[List[str]] = None, include_relationships: bool = False) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            exclude: List of column names to exclude
            include_relationships: Whether to include relationship data
            
        Returns:
            Dictionary representation of the model
        """
        exclude = exclude or []
        result = {}
        
        # Include column attributes
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                result[column.name] = value
        
        # Include relationships if requested
        if include_relationships:
            for relationship in self.__mapper__.relationships:
                if relationship.key not in exclude:
                    value = getattr(self, relationship.key)
                    if value is not None:
                        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                            # Handle collections
                            result[relationship.key] = [
                                item.to_dict(exclude=exclude) if hasattr(item, 'to_dict') else str(item)
                                for item in value
                            ]
                        else:
                            # Handle single relationships
                            result[relationship.key] = (
                                value.to_dict(exclude=exclude) if hasattr(value, 'to_dict') else str(value)
                            )
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[List[str]] = None) -> None:
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary with new values
            exclude: List of keys to exclude from update
        """
        exclude = exclude or ['id', 'created_at']
        
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def get_by_id(cls: Type[ModelType], db: Session, id: uuid.UUID) -> Optional[ModelType]:
        """Get model instance by ID."""
        return db.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def get_by_ids(cls: Type[ModelType], db: Session, ids: List[uuid.UUID]) -> List[ModelType]:
        """Get multiple model instances by IDs."""
        return db.query(cls).filter(cls.id.in_(ids)).all()
    
    @classmethod
    def create(cls: Type[ModelType], db: Session, **kwargs) -> ModelType:
        """Create and save new model instance."""
        instance = cls(**kwargs)
        db.add(instance)
        db.flush()  # Flush to get the ID without committing
        return instance
    
    @classmethod
    def get_or_create(cls: Type[ModelType], db: Session, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[ModelType, bool]:
        """Get existing instance or create new one."""
        instance = db.query(cls).filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            params = kwargs.copy()
            if defaults:
                params.update(defaults)
            instance = cls.create(db, **params)
            return instance, True
    
    @classmethod
    def bulk_create(cls: Type[ModelType], db: Session, data_list: List[Dict[str, Any]]) -> List[ModelType]:
        """Create multiple instances efficiently."""
        instances = [cls(**data) for data in data_list]
        db.add_all(instances)
        db.flush()
        return instances
    
    def save(self, db: Session) -> None:
        """Save instance to database."""
        db.add(self)
        db.flush()
    
    def delete(self, db: Session) -> None:
        """Delete instance from database."""
        db.delete(self)
        db.flush()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class AuditMixin:
    """Mixin to add audit trail fields."""
    
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    
    def set_audit_fields(self, user_id: uuid.UUID, is_creation: bool = False) -> None:
        """Set audit fields for create/update operations."""
        if is_creation:
            self.created_by = user_id
        self.updated_by = user_id


class SoftDeleteMixin:
    """Mixin to add soft delete functionality."""
    
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    
    def soft_delete(self) -> None:
        """Mark instance as deleted without removing from database."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self) -> None:
        """Restore soft-deleted instance."""
        self.is_deleted = False
        self.deleted_at = None
    
    @classmethod
    def get_active(cls: Type[ModelType], db: Session) -> List[ModelType]:
        """Get all non-deleted instances."""
        return db.query(cls).filter(cls.is_deleted == False).all()


# Event listeners for automatic timestamp updates
@event.listens_for(BaseModel, 'before_update', propagate=True)
def update_timestamp(mapper, connection, target):
    """Automatically update the updated_at timestamp on model updates."""
    target.updated_at = datetime.now(timezone.utc)


def configure_mappers():
    """Configure all SQLAlchemy mappers. Call after all models are defined."""
    try:
        Base.registry.configure()
    except Exception as e:
        # Log the error if logging is available
        print(f"Error configuring mappers: {e}")
        raise