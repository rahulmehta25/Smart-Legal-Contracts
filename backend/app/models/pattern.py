"""
Pattern model for storing arbitration detection patterns.
Supports multiple pattern types and pattern management.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, Column, Integer, String, Text, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Session, relationship

from .base import BaseModel, AuditMixin


class Pattern(BaseModel, AuditMixin):
    """
    Pattern model for storing arbitration clause detection patterns.
    
    Features:
    - Multiple pattern types (regex, keyword, semantic, ml_model)
    - Effectiveness and accuracy tracking
    - Usage statistics
    - Versioning and deprecation support
    - Multi-tenant organization support
    """
    
    __tablename__ = 'patterns'
    
    # Basic pattern information
    pattern_name = Column(String(200), nullable=False)
    pattern_text = Column(Text, nullable=False)
    pattern_type = Column(String(100), nullable=False, index=True)  # regex, keyword, semantic, ml_model
    
    # Classification
    category = Column(String(100), nullable=False, index=True)  # mandatory_arbitration, opt_out_clause, class_action_waiver, etc.
    subcategory = Column(String(100), nullable=True, index=True)
    language = Column(String(10), default='en', nullable=False, index=True)
    jurisdiction = Column(String(100), nullable=True, index=True)  # US, EU, UK, etc.
    
    # Performance metrics
    effectiveness_score = Column(Float, default=0.5, nullable=False, index=True)  # 0.0 to 1.0
    accuracy_score = Column(Float, nullable=True, index=True)  # 0.0 to 1.0
    usage_count = Column(Integer, default=0, nullable=False, index=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Status and versioning
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_deprecated = Column(Boolean, default=False, nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    parent_pattern_id = Column(UUID(as_uuid=True), ForeignKey('patterns.id'), nullable=True)
    
    # Ownership
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey('organizations.id'), nullable=True, index=True)
    
    # Additional metadata
    tags = Column(ARRAY(Text), nullable=True)
    description = Column(Text, nullable=True)
    examples = Column(ARRAY(Text), nullable=True)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_patterns")
    organization = relationship("Organization", back_populates="patterns")
    parent_pattern = relationship("Pattern", remote_side="Pattern.id")
    versions = relationship("Pattern", remote_side="Pattern.parent_pattern_id")
    
    @property
    def is_system_pattern(self) -> bool:
        """Check if this is a system-wide pattern."""
        return self.organization_id is None
    
    @property
    def is_custom_pattern(self) -> bool:
        """Check if this is a custom organization pattern."""
        return self.organization_id is not None
    
    @property
    def is_high_performing(self) -> bool:
        """Check if pattern has high performance scores."""
        return (self.effectiveness_score >= 0.8 and 
                (self.accuracy_score is None or self.accuracy_score >= 0.8))
    
    @property
    def needs_review(self) -> bool:
        """Check if pattern needs performance review."""
        return (self.usage_count > 100 and 
                (self.effectiveness_score < 0.5 or 
                 (self.accuracy_score is not None and self.accuracy_score < 0.5)))
    
    def increment_usage(self) -> None:
        """Increment usage count and update last_used timestamp."""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
    
    def update_performance_metrics(self, effectiveness: Optional[float] = None, accuracy: Optional[float] = None) -> None:
        """Update performance metrics."""
        if effectiveness is not None:
            self.effectiveness_score = max(0.0, min(1.0, effectiveness))
        if accuracy is not None:
            self.accuracy_score = max(0.0, min(1.0, accuracy))
    
    def deprecate(self, replacement_pattern_id: Optional[str] = None) -> None:
        """Deprecate this pattern."""
        self.is_deprecated = True
        self.is_active = False
        if replacement_pattern_id:
            # Could add a replacement_pattern_id field if needed
            pass
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the pattern."""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags = self.tags + [tag]
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the pattern."""
        if self.tags and tag in self.tags:
            self.tags = [t for t in self.tags if t != tag]
    
    def has_tag(self, tag: str) -> bool:
        """Check if pattern has a specific tag."""
        return self.tags is not None and tag in self.tags
    
    def create_new_version(self, db: Session, **kwargs) -> 'Pattern':
        """
        Create a new version of this pattern.
        
        Args:
            db: Database session
            **kwargs: Fields to update in the new version
            
        Returns:
            New pattern version
        """
        # Get the latest version number
        if self.parent_pattern_id is None:
            # This is the root pattern
            latest_version = max([v.version for v in self.versions], default=self.version)
            parent_id = self.id
        else:
            # This is already a version, use the parent
            latest_version = self.parent_pattern.get_latest_version()
            parent_id = self.parent_pattern_id
        
        # Create new version
        new_pattern = Pattern(
            pattern_name=kwargs.get('pattern_name', self.pattern_name),
            pattern_text=kwargs.get('pattern_text', self.pattern_text),
            pattern_type=kwargs.get('pattern_type', self.pattern_type),
            category=kwargs.get('category', self.category),
            subcategory=kwargs.get('subcategory', self.subcategory),
            language=kwargs.get('language', self.language),
            jurisdiction=kwargs.get('jurisdiction', self.jurisdiction),
            description=kwargs.get('description', self.description),
            tags=kwargs.get('tags', self.tags.copy() if self.tags else None),
            examples=kwargs.get('examples', self.examples.copy() if self.examples else None),
            version=latest_version + 1,
            parent_pattern_id=parent_id,
            created_by=kwargs.get('created_by', self.created_by),
            organization_id=kwargs.get('organization_id', self.organization_id)
        )
        
        db.add(new_pattern)
        db.flush()
        return new_pattern
    
    def get_latest_version(self) -> int:
        """Get the latest version number for this pattern family."""
        if self.parent_pattern_id is None:
            # This is the root pattern, check all its versions
            max_version = max([v.version for v in self.versions], default=self.version)
            return max(self.version, max_version)
        else:
            # This is a version, ask the parent
            return self.parent_pattern.get_latest_version()
    
    @classmethod
    def get_active_patterns(cls, db: Session, pattern_type: Optional[str] = None, 
                           category: Optional[str] = None, organization_id: Optional[str] = None) -> List['Pattern']:
        """Get active patterns with optional filters."""
        query = db.query(cls).filter(
            cls.is_active == True,
            cls.is_deprecated == False
        )
        
        if pattern_type:
            query = query.filter(cls.pattern_type == pattern_type)
        
        if category:
            query = query.filter(cls.category == category)
        
        if organization_id:
            # Include both system patterns and organization-specific patterns
            query = query.filter(
                (cls.organization_id == organization_id) | 
                (cls.organization_id.is_(None))
            )
        
        return query.order_by(cls.effectiveness_score.desc()).all()
    
    @classmethod
    def get_by_category(cls, db: Session, category: str, active_only: bool = True) -> List['Pattern']:
        """Get patterns by category."""
        query = db.query(cls).filter(cls.category == category)
        
        if active_only:
            query = query.filter(cls.is_active == True, cls.is_deprecated == False)
        
        return query.order_by(cls.effectiveness_score.desc()).all()
    
    @classmethod
    def get_high_performance(cls, db: Session, min_effectiveness: float = 0.8, 
                           min_accuracy: Optional[float] = None, limit: int = 100) -> List['Pattern']:
        """Get high-performance patterns."""
        query = db.query(cls).filter(
            cls.is_active == True,
            cls.is_deprecated == False,
            cls.effectiveness_score >= min_effectiveness
        )
        
        if min_accuracy is not None:
            query = query.filter(cls.accuracy_score >= min_accuracy)
        
        return query.order_by(cls.effectiveness_score.desc()).limit(limit).all()
    
    @classmethod
    def get_low_performance(cls, db: Session, max_effectiveness: float = 0.5, 
                          min_usage: int = 10, limit: int = 100) -> List['Pattern']:
        """Get low-performance patterns that may need review."""
        return db.query(cls).filter(
            cls.is_active == True,
            cls.effectiveness_score <= max_effectiveness,
            cls.usage_count >= min_usage
        ).order_by(cls.effectiveness_score.asc()).limit(limit).all()
    
    @classmethod
    def search_patterns(cls, db: Session, query: str, organization_id: Optional[str] = None, 
                       limit: int = 50) -> List['Pattern']:
        """
        Search patterns by name, text, or description.
        
        Args:
            db: Database session
            query: Search query
            organization_id: Optional organization filter
            limit: Maximum results to return
            
        Returns:
            List of matching patterns
        """
        search = f"%{query.lower()}%"
        db_query = db.query(cls).filter(
            cls.is_active == True
        ).filter(
            cls.pattern_name.ilike(search) |
            cls.pattern_text.ilike(search) |
            cls.description.ilike(search)
        )
        
        if organization_id:
            db_query = db_query.filter(
                (cls.organization_id == organization_id) | 
                (cls.organization_id.is_(None))
            )
        
        return db_query.order_by(cls.effectiveness_score.desc()).limit(limit).all()
    
    @classmethod
    def get_unused_patterns(cls, db: Session, days_unused: int = 90) -> List['Pattern']:
        """Get patterns that haven't been used recently."""
        from datetime import timedelta
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_unused)
        return db.query(cls).filter(
            cls.is_active == True,
            (cls.last_used.is_(None)) | (cls.last_used < cutoff_date)
        ).all()
    
    @classmethod
    def get_system_patterns(cls, db: Session) -> List['Pattern']:
        """Get all system-wide patterns."""
        return db.query(cls).filter(
            cls.organization_id.is_(None),
            cls.is_active == True
        ).order_by(cls.category, cls.effectiveness_score.desc()).all()
    
    def __repr__(self) -> str:
        return f"<Pattern(id={self.id}, name={self.pattern_name}, type={self.pattern_type}, category={self.category})>"