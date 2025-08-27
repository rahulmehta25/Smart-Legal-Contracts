"""
Simple base model for SQLite testing
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy import DateTime, String, Boolean, event, Column, Integer
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

# Type variable for model classes
ModelType = TypeVar("ModelType", bound="BaseModel")

Base = declarative_base()


class TimestampMixin:
    """Mixin class to add created_at and updated_at timestamps."""
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class BaseModel(Base, TimestampMixin):
    """
    Simple base model class for SQLite testing.
    """
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower() + 's'
    
    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        exclude = exclude or []
        result = {}
        
        # Include column attributes
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[List[str]] = None) -> None:
        """Update model instance from dictionary."""
        exclude = exclude or ['id', 'created_at']
        
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def get_by_id(cls: Type[ModelType], db: Session, id: int) -> Optional[ModelType]:
        """Get model instance by ID."""
        return db.query(cls).filter(cls.id == id).first()
    
    @classmethod
    def create(cls: Type[ModelType], db: Session, **kwargs) -> ModelType:
        """Create and save new model instance."""
        instance = cls(**kwargs)
        db.add(instance)
        db.flush()
        return instance
    
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
    
    created_by = Column(Integer, nullable=True)
    updated_by = Column(Integer, nullable=True)


class SoftDeleteMixin:
    """Mixin to add soft delete functionality."""
    
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)