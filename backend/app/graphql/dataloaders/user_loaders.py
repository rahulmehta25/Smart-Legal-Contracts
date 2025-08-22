"""
DataLoaders for User and collaboration entities
"""

from typing import List, Dict, Any
from aiodataloader import DataLoader
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from ..types import User as UserType, Comment as CommentType
from ...models.user import User


class UserDataLoader(DataLoader):
    """DataLoader for User entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[UserType]:
        """Batch load users by IDs"""
        try:
            users = self.session.query(User).filter(
                User.id.in_(keys)
            ).all()
            
            user_map = {
                user.id: self._convert_to_graphql_type(user)
                for user in users
            }
            
            return [user_map.get(key) for key in keys]
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, user: User) -> UserType:
        """Convert SQLAlchemy User to GraphQL UserType"""
        return UserType(
            id=str(user.id),
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            organization=user.organization,
            role="USER",  # Default role, would need to be in User model
            is_active=user.is_active,
            is_verified=user.is_verified,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=None,  # Not in current model
            # Computed fields - would need proper queries
            document_count=0,  # Would need to count user's documents
            analysis_count=0   # Would need to count user's analyses
        )


class UserByEmailDataLoader(DataLoader):
    """DataLoader for Users by email"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, emails: List[str]) -> List[UserType]:
        """Batch load users by emails"""
        try:
            users = self.session.query(User).filter(
                User.email.in_(emails)
            ).all()
            
            user_map = {
                user.email: UserDataLoader(self.session)._convert_to_graphql_type(user)
                for user in users
            }
            
            return [user_map.get(email) for email in emails]
            
        except Exception as e:
            return [None] * len(emails)


class UserByUsernameDataLoader(DataLoader):
    """DataLoader for Users by username"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, usernames: List[str]) -> List[UserType]:
        """Batch load users by usernames"""
        try:
            users = self.session.query(User).filter(
                User.username.in_(usernames)
            ).all()
            
            user_map = {
                user.username: UserDataLoader(self.session)._convert_to_graphql_type(user)
                for user in users
            }
            
            return [user_map.get(username) for username in usernames]
            
        except Exception as e:
            return [None] * len(usernames)


class CommentDataLoader(DataLoader):
    """DataLoader for Comment entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[CommentType]:
        """Batch load comments by IDs"""
        try:
            # For now, return empty as Comment model doesn't exist yet
            # In a real implementation, this would query the Comment table
            return [None] * len(keys)
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, comment) -> CommentType:
        """Convert SQLAlchemy Comment to GraphQL CommentType"""
        # Placeholder implementation
        return CommentType(
            id=str(comment.id),
            document_id=str(comment.document_id),
            user_id=str(comment.user_id),
            content=comment.content,
            is_resolved=comment.is_resolved,
            created_at=comment.created_at,
            updated_at=comment.updated_at
        )


class CommentsByDocumentDataLoader(DataLoader):
    """DataLoader for Comments by Document ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, document_ids: List[int]) -> List[List[CommentType]]:
        """Batch load comments by document IDs"""
        try:
            # For now, return empty lists as Comment model doesn't exist yet
            return [[] for _ in document_ids]
            
        except Exception as e:
            return [[] for _ in document_ids]


class CommentsByUserDataLoader(DataLoader):
    """DataLoader for Comments by User ID"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, user_ids: List[int]) -> List[List[CommentType]]:
        """Batch load comments by user IDs"""
        try:
            # For now, return empty lists as Comment model doesn't exist yet
            return [[] for _ in user_ids]
            
        except Exception as e:
            return [[] for _ in user_ids]


class UserStatsDataLoader(DataLoader):
    """DataLoader for User statistics"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Batch load user statistics"""
        try:
            total_users = self.session.query(User).count()
            active_users = self.session.query(User).filter(
                User.is_active == True
            ).count()
            verified_users = self.session.query(User).filter(
                User.is_verified == True
            ).count()
            
            # Group users by organization
            users_by_org = self.session.query(
                User.organization,
                func.count(User.id).label('count')
            ).group_by(User.organization).all()
            
            stats = {
                'total_users': total_users,
                'active_users': active_users,
                'verified_users': verified_users,
                'verification_rate': verified_users / total_users if total_users > 0 else 0,
                'users_by_organization': [
                    {
                        'organization': row.organization or 'No Organization',
                        'count': row.count
                    }
                    for row in users_by_org
                ]
            }
            
            return [stats for _ in keys]
            
        except Exception as e:
            return [{} for _ in keys]


def create_user_loaders(session: Session) -> Dict[str, DataLoader]:
    """Create all user-related DataLoaders"""
    return {
        'user': UserDataLoader(session),
        'user_by_email': UserByEmailDataLoader(session),
        'user_by_username': UserByUsernameDataLoader(session),
        'comment': CommentDataLoader(session),
        'comments_by_document': CommentsByDocumentDataLoader(session),
        'comments_by_user': CommentsByUserDataLoader(session),
        'user_stats': UserStatsDataLoader(session),
    }