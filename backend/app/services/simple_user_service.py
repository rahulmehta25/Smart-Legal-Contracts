"""
Simplified user service for basic API testing
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import uuid
from datetime import datetime, timedelta
from loguru import logger

from app.models.simple_user import User
from app.schemas.user import UserCreate, UserLogin, Token


class UserService:
    """Simplified user service for testing"""
    
    def create_user(self, db: Session, user_data: UserCreate) -> User:
        """Create a new user"""
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.email == user_data.email) | (User.username == user_data.username)
            ).first()
            
            if existing_user:
                raise ValueError("User with this email or username already exists")
            
            # Create user
            user = User(
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name,
                is_active=True
            )
            
            # Set password (simplified - in production use proper hashing)
            user.set_password(user_data.password)
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            return user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            db.rollback()
            raise
    
    def login_user(self, db: Session, login_data: UserLogin) -> Optional[Token]:
        """Login user and return token"""
        try:
            # Find user by username or email
            user = db.query(User).filter(
                (User.username == login_data.username) | (User.email == login_data.username)
            ).first()
            
            if not user or not user.check_password(login_data.password):
                return None
            
            if not user.is_active:
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            # Return mock token
            return Token(
                access_token="mock_access_token_" + str(user.id),
                token_type="bearer",
                expires_in=3600
            )
        except Exception as e:
            logger.error(f"Error logging in user: {e}")
            return None
    
    def get_current_user(self, db: Session, token: str) -> Optional[User]:
        """Get current user from token"""
        try:
            # In a real implementation, you'd validate the JWT token
            # For now, extract user ID from mock token
            if token.startswith("mock_access_token_"):
                user_id = token.replace("mock_access_token_", "")
                return db.query(User).filter(User.id == user_id).first()
            return None
        except Exception as e:
            logger.error(f"Error getting current user: {e}")
            return None
    
    def get_user(self, db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_users(self, db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[User]:
        """Get list of users"""
        try:
            query = db.query(User)
            if active_only:
                query = query.filter(User.is_active == True)
            return query.offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
    
    def update_user(self, db: Session, user_id: int, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user information"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            for key, value in update_data.items():
                if hasattr(user, key) and key != 'password':
                    setattr(user, key, value)
            
            db.commit()
            db.refresh(user)
            return user
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return None
    
    def change_password(self, db: Session, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.check_password(current_password):
                return False
            
            user.set_password(new_password)
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return False
    
    def deactivate_user(self, db: Session, user_id: int) -> bool:
        """Deactivate a user account"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.is_active = False
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            return False
    
    def get_user_statistics(self, db: Session) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "inactive_users": total_users - active_users
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_users": 0, "active_users": 0}