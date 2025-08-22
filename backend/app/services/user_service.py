from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from loguru import logger
import os

from app.models.user import User, UserCreate, UserResponse, UserLogin, Token, TokenData


class UserService:
    """
    Service for managing users and authentication
    """
    
    def __init__(self):
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT settings
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        Hash a password
        """
        return self.pwd_context.hash(password)
    
    def create_user(self, db: Session, user_data: UserCreate) -> User:
        """
        Create a new user
        
        Args:
            db: Database session
            user_data: User creation data
            
        Returns:
            Created User instance
        """
        try:
            # Check if user already exists
            if self.get_user_by_email(db, user_data.email):
                raise ValueError("Email already registered")
            
            if self.get_user_by_username(db, user_data.username):
                raise ValueError("Username already taken")
            
            # Hash password
            hashed_password = self.get_password_hash(user_data.password)
            
            # Create user
            db_user = User(
                email=user_data.email,
                username=user_data.username,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                organization=user_data.organization,
                created_at=datetime.utcnow(),
                is_active=True,
                is_verified=False
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"Created user: {user_data.username} ({user_data.email})")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    def authenticate_user(self, db: Session, login_data: UserLogin) -> Optional[User]:
        """
        Authenticate a user with username/password
        
        Args:
            db: Database session
            login_data: Login credentials
            
        Returns:
            User if authenticated, None otherwise
        """
        try:
            # Get user by username
            user = self.get_user_by_username(db, login_data.username)
            if not user:
                return None
            
            # Verify password
            if not self.verify_password(login_data.password, user.hashed_password):
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            logger.info(f"User authenticated: {user.username}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                return None
            
            return TokenData(username=username)
            
        except JWTError:
            return None
    
    def get_current_user(self, db: Session, token: str) -> Optional[User]:
        """
        Get current user from token
        
        Args:
            db: Database session
            token: JWT token
            
        Returns:
            Current user if token is valid
        """
        try:
            token_data = self.verify_token(token)
            if token_data is None:
                return None
            
            user = self.get_user_by_username(db, token_data.username)
            if user is None or not user.is_active:
                return None
            
            return user
            
        except Exception as e:
            logger.error(f"Error getting current user: {e}")
            return None
    
    def get_user(self, db: Session, user_id: int) -> Optional[User]:
        """
        Get user by ID
        """
        return db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """
        Get user by email
        """
        return db.query(User).filter(User.email == email).first()
    
    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """
        Get user by username
        """
        return db.query(User).filter(User.username == username).first()
    
    def get_users(self, 
                  db: Session, 
                  skip: int = 0, 
                  limit: int = 100,
                  active_only: bool = True) -> List[User]:
        """
        Get list of users with pagination
        """
        query = db.query(User)
        
        if active_only:
            query = query.filter(User.is_active == True)
        
        return query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()
    
    def update_user(self, 
                   db: Session, 
                   user_id: int, 
                   update_data: dict) -> Optional[User]:
        """
        Update user information
        
        Args:
            db: Database session
            user_id: ID of user to update
            update_data: Dictionary of fields to update
            
        Returns:
            Updated user or None if not found
        """
        try:
            user = self.get_user(db, user_id)
            if not user:
                return None
            
            # Update allowed fields
            allowed_fields = ['full_name', 'organization', 'is_active', 'is_verified']
            
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(user, field):
                    setattr(user, field, value)
            
            db.commit()
            db.refresh(user)
            
            logger.info(f"Updated user {user_id}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            raise
    
    def change_password(self, 
                       db: Session, 
                       user_id: int, 
                       current_password: str, 
                       new_password: str) -> bool:
        """
        Change user password
        
        Args:
            db: Database session
            user_id: ID of user
            current_password: Current password for verification
            new_password: New password
            
        Returns:
            True if password changed successfully
        """
        try:
            user = self.get_user(db, user_id)
            if not user:
                return False
            
            # Verify current password
            if not self.verify_password(current_password, user.hashed_password):
                return False
            
            # Set new password
            user.hashed_password = self.get_password_hash(new_password)
            db.commit()
            
            logger.info(f"Password changed for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error changing password for user {user_id}: {e}")
            return False
    
    def deactivate_user(self, db: Session, user_id: int) -> bool:
        """
        Deactivate a user account
        """
        try:
            user = self.get_user(db, user_id)
            if not user:
                return False
            
            user.is_active = False
            db.commit()
            
            logger.info(f"Deactivated user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deactivating user {user_id}: {e}")
            return False
    
    def get_user_statistics(self, db: Session) -> dict:
        """
        Get user statistics
        """
        try:
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            verified_users = db.query(User).filter(User.is_verified == True).count()
            
            # Recent registrations (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_registrations = db.query(User).filter(
                User.created_at >= thirty_days_ago
            ).count()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': total_users - active_users,
                'verified_users': verified_users,
                'unverified_users': total_users - verified_users,
                'recent_registrations': recent_registrations
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {'error': str(e)}
    
    def login_user(self, db: Session, login_data: UserLogin) -> Optional[Token]:
        """
        Login user and return access token
        
        Args:
            db: Database session
            login_data: Login credentials
            
        Returns:
            Token if login successful, None otherwise
        """
        user = self.authenticate_user(db, login_data)
        if not user:
            return None
        
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"sub": user.username}, 
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60  # Convert to seconds
        )