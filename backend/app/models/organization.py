"""
Organization model for multi-tenant support.
Manages subscriptions, limits, and team features.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, relationship

from .base import BaseModel, AuditMixin


class Organization(BaseModel, AuditMixin):
    """
    Organization model for multi-tenant architecture.
    
    Features:
    - Subscription management
    - Resource limits and quotas
    - Team settings and preferences
    - Usage tracking
    - Custom branding support
    """
    
    __tablename__ = 'organizations'
    
    # Basic information
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    domain = Column(String(255), nullable=True, index=True)
    logo_url = Column(Text, nullable=True)
    
    # Settings and preferences
    settings = Column(JSONB, default={}, nullable=False)
    
    # Subscription information
    subscription_tier = Column(
        String(50), 
        default='basic',
        nullable=False,
        index=True
    )  # basic, pro, enterprise
    subscription_status = Column(
        String(50),
        default='active',
        nullable=False,
        index=True
    )  # active, suspended, cancelled
    subscription_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Resource limits
    max_documents = Column(Integer, default=100, nullable=False)
    max_users = Column(Integer, default=5, nullable=False)
    max_storage_gb = Column(Integer, default=10, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Relationships
    users = relationship("User", back_populates="organization")
    documents = relationship("Document", back_populates="organization")
    analyses = relationship("Analysis", back_populates="organization")
    patterns = relationship("Pattern", back_populates="organization")
    
    @property
    def is_premium(self) -> bool:
        """Check if organization has premium features."""
        return self.subscription_tier in ['pro', 'enterprise']
    
    @property
    def is_enterprise(self) -> bool:
        """Check if organization has enterprise features."""
        return self.subscription_tier == 'enterprise'
    
    @property
    def is_subscription_active(self) -> bool:
        """Check if subscription is currently active."""
        if self.subscription_status != 'active':
            return False
        
        if self.subscription_expires_at is None:
            return True  # No expiration date means active
        
        return datetime.now(timezone.utc) < self.subscription_expires_at
    
    def get_usage_stats(self, db: Session) -> dict:
        """
        Get current usage statistics for the organization.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with usage statistics
        """
        from .document import Document
        from .user import User
        from .analysis import Analysis
        
        # Count active users
        user_count = db.query(User).filter(
            User.organization_id == self.id,
            User.is_active == True
        ).count()
        
        # Count documents and calculate storage usage
        document_stats = db.query(
            func.count(Document.id),
            func.coalesce(func.sum(Document.file_size), 0)
        ).filter(Document.organization_id == self.id).first()
        
        document_count = document_stats[0] if document_stats else 0
        storage_used_bytes = document_stats[1] if document_stats else 0
        storage_used_gb = round(storage_used_bytes / (1024 * 1024 * 1024), 2)
        
        # Count analyses this month
        from sqlalchemy import func
        current_month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        analyses_this_month = db.query(Analysis).filter(
            Analysis.organization_id == self.id,
            Analysis.created_at >= current_month_start
        ).count()
        
        return {
            'users': {
                'current': user_count,
                'limit': self.max_users,
                'percentage': round((user_count / self.max_users) * 100, 1) if self.max_users > 0 else 0
            },
            'documents': {
                'current': document_count,
                'limit': self.max_documents,
                'percentage': round((document_count / self.max_documents) * 100, 1) if self.max_documents > 0 else 0
            },
            'storage': {
                'used_gb': storage_used_gb,
                'limit_gb': self.max_storage_gb,
                'percentage': round((storage_used_gb / self.max_storage_gb) * 100, 1) if self.max_storage_gb > 0 else 0
            },
            'analyses_this_month': analyses_this_month
        }
    
    def can_add_user(self, db: Session) -> bool:
        """Check if organization can add another user."""
        if not self.is_subscription_active:
            return False
        
        current_users = db.query(User).filter(
            User.organization_id == self.id,
            User.is_active == True
        ).count()
        
        return current_users < self.max_users
    
    def can_add_document(self, db: Session) -> bool:
        """Check if organization can add another document."""
        if not self.is_subscription_active:
            return False
        
        current_documents = db.query(Document).filter(
            Document.organization_id == self.id
        ).count()
        
        return current_documents < self.max_documents
    
    def can_use_storage(self, db: Session, additional_bytes: int = 0) -> bool:
        """Check if organization can use additional storage."""
        if not self.is_subscription_active:
            return False
        
        current_storage = db.query(func.coalesce(func.sum(Document.file_size), 0)).filter(
            Document.organization_id == self.id
        ).scalar()
        
        total_storage_gb = (current_storage + additional_bytes) / (1024 * 1024 * 1024)
        return total_storage_gb <= self.max_storage_gb
    
    def get_setting(self, key: str, default=None):
        """Get organization setting value."""
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value) -> None:
        """Set organization setting value."""
        if self.settings is None:
            self.settings = {}
        self.settings[key] = value
    
    def update_subscription(self, tier: str, expires_at: Optional[datetime] = None) -> None:
        """
        Update subscription tier and expiration.
        
        Args:
            tier: New subscription tier (basic, pro, enterprise)
            expires_at: Optional expiration date
        """
        self.subscription_tier = tier
        self.subscription_expires_at = expires_at
        self.subscription_status = 'active'
        
        # Update limits based on tier
        if tier == 'basic':
            self.max_documents = 100
            self.max_users = 5
            self.max_storage_gb = 10
        elif tier == 'pro':
            self.max_documents = 1000
            self.max_users = 25
            self.max_storage_gb = 100
        elif tier == 'enterprise':
            self.max_documents = 10000
            self.max_users = 100
            self.max_storage_gb = 1000
    
    def suspend_subscription(self, reason: str = None) -> None:
        """Suspend the organization's subscription."""
        self.subscription_status = 'suspended'
        if reason:
            self.set_setting('suspension_reason', reason)
    
    def reactivate_subscription(self) -> None:
        """Reactivate a suspended subscription."""
        self.subscription_status = 'active'
        if 'suspension_reason' in (self.settings or {}):
            del self.settings['suspension_reason']
    
    def cancel_subscription(self) -> None:
        """Cancel the organization's subscription."""
        self.subscription_status = 'cancelled'
        # Keep access until expiration date
    
    @classmethod
    def get_by_slug(cls, db: Session, slug: str) -> Optional['Organization']:
        """Get organization by slug."""
        return db.query(cls).filter(cls.slug == slug.lower()).first()
    
    @classmethod
    def get_by_domain(cls, db: Session, domain: str) -> Optional['Organization']:
        """Get organization by domain."""
        return db.query(cls).filter(cls.domain == domain.lower()).first()
    
    @classmethod
    def search_organizations(cls, db: Session, query: str, limit: int = 50) -> List['Organization']:
        """
        Search organizations by name or domain.
        
        Args:
            db: Database session
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching organizations
        """
        search = f"%{query.lower()}%"
        return db.query(cls).filter(
            cls.is_active == True
        ).filter(
            cls.name.ilike(search) | 
            cls.domain.ilike(search)
        ).limit(limit).all()
    
    @classmethod
    def get_expired_subscriptions(cls, db: Session) -> List['Organization']:
        """Get organizations with expired subscriptions."""
        return db.query(cls).filter(
            cls.subscription_expires_at < datetime.now(timezone.utc),
            cls.subscription_status == 'active'
        ).all()
    
    @classmethod
    def get_trial_organizations(cls, db: Session, days_left: int = 7) -> List['Organization']:
        """Get organizations with trials expiring soon."""
        from datetime import timedelta
        
        cutoff_date = datetime.now(timezone.utc) + timedelta(days=days_left)
        return db.query(cls).filter(
            cls.subscription_tier == 'basic',
            cls.subscription_expires_at <= cutoff_date,
            cls.subscription_expires_at > datetime.now(timezone.utc),
            cls.subscription_status == 'active'
        ).all()
    
    def __repr__(self) -> str:
        return f"<Organization(id={self.id}, name={self.name}, tier={self.subscription_tier})>"