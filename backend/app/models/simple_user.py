"""
Simple user model for SQLite testing
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional

import bcrypt
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .simple_base import BaseModel, AuditMixin


class User(BaseModel, AuditMixin):
    """Simple user model for testing"""
    
    __tablename__ = 'users'
    
    # Basic information
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(200), nullable=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    
    # Role and status
    role = Column(String(50), default='user', nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Login tracking
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    
    # Relationships - commented out to avoid circular imports
    # documents = relationship("Document", foreign_keys="Document.user_id", back_populates="user")
    # analyses = relationship("Analysis", foreign_keys="Analysis.user_id", back_populates="user")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.salt:
            self.salt = secrets.token_hex(16)
    
    def set_password(self, password: str) -> None:
        """Set user password with bcrypt hashing."""
        password_bytes = password.encode('utf-8')
        salt_bytes = self.salt.encode('utf-8')
        
        # Create bcrypt hash
        hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
        self.password_hash = hashed.decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """Check if provided password matches stored hash."""
        try:
            password_bytes = password.encode('utf-8')
            stored_hash = self.password_hash.encode('utf-8')
            return bcrypt.checkpw(password_bytes, stored_hash)
        except Exception:
            return False
    
    def update_login_info(self) -> None:
        """Update login information."""
        self.last_login = datetime.utcnow()
        self.login_count += 1
    
    @property
    def display_name(self) -> str:
        """Get display name for the user."""
        return self.full_name or self.username
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert to dictionary, optionally including sensitive data."""
        result = super().to_dict(exclude=['password_hash', 'salt'])
        
        if include_sensitive:
            result.update({
                'password_hash': self.password_hash,
                'salt': self.salt
            })
        
        return result