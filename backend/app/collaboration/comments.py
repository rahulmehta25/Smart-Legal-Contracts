"""
Threaded comments system for collaborative discussions.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import json

import redis.asyncio as redis
from pydantic import BaseModel, Field


class CommentType(Enum):
    TEXT = "text"
    VOICE = "voice"
    VIDEO = "video"
    LINK = "link"
    MENTION = "mention"
    EMOJI_REACTION = "emoji_reaction"


class CommentStatus(Enum):
    ACTIVE = "active"
    EDITED = "edited"
    DELETED = "deleted"
    HIDDEN = "hidden"
    RESOLVED = "resolved"


@dataclass
class CommentAttachment:
    """File attachment for comments."""
    id: str
    filename: str
    file_type: str
    file_size: int
    url: str
    thumbnail_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentAttachment':
        return cls(**data)


@dataclass
class CommentMention:
    """User mention in a comment."""
    user_id: str
    username: str
    display_name: Optional[str] = None
    start_index: int = 0
    end_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentMention':
        return cls(**data)


@dataclass
class CommentReaction:
    """Emoji reaction to a comment."""
    emoji: str
    user_id: str
    username: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'emoji': self.emoji,
            'user_id': self.user_id,
            'username': self.username,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentReaction':
        return cls(
            emoji=data['emoji'],
            user_id=data['user_id'],
            username=data['username'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class CommentEditHistory:
    """Edit history for a comment."""
    timestamp: datetime
    previous_content: str
    user_id: str
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'previous_content': self.previous_content,
            'user_id': self.user_id,
            'reason': self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentEditHistory':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            previous_content=data['previous_content'],
            user_id=data['user_id'],
            reason=data.get('reason')
        )


@dataclass
class Comment:
    """A threaded comment with rich features."""
    id: str
    document_id: str
    room_id: str
    user_id: str
    username: str
    user_avatar: Optional[str]
    content: str
    comment_type: CommentType
    status: CommentStatus
    created_at: datetime
    updated_at: datetime
    parent_id: Optional[str] = None  # For threading
    thread_id: Optional[str] = None  # Root thread ID
    reply_to_id: Optional[str] = None  # Direct reply to specific comment
    depth: int = 0  # Nesting depth
    position: Optional[Dict[str, Any]] = None  # Position in document
    attachments: List[CommentAttachment] = field(default_factory=list)
    mentions: List[CommentMention] = field(default_factory=list)
    reactions: List[CommentReaction] = field(default_factory=list)
    edit_history: List[CommentEditHistory] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_pinned: bool = False
    is_private: bool = False
    upvotes: int = 0
    downvotes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'document_id': self.document_id,
            'room_id': self.room_id,
            'user_id': self.user_id,
            'username': self.username,
            'user_avatar': self.user_avatar,
            'content': self.content,
            'comment_type': self.comment_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'parent_id': self.parent_id,
            'thread_id': self.thread_id,
            'reply_to_id': self.reply_to_id,
            'depth': self.depth,
            'position': self.position,
            'attachments': [att.to_dict() for att in self.attachments],
            'mentions': [mention.to_dict() for mention in self.mentions],
            'reactions': [reaction.to_dict() for reaction in self.reactions],
            'edit_history': [edit.to_dict() for edit in self.edit_history],
            'tags': self.tags,
            'metadata': self.metadata,
            'is_pinned': self.is_pinned,
            'is_private': self.is_private,
            'upvotes': self.upvotes,
            'downvotes': self.downvotes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Comment':
        return cls(
            id=data['id'],
            document_id=data['document_id'],
            room_id=data['room_id'],
            user_id=data['user_id'],
            username=data['username'],
            user_avatar=data.get('user_avatar'),
            content=data['content'],
            comment_type=CommentType(data['comment_type']),
            status=CommentStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            parent_id=data.get('parent_id'),
            thread_id=data.get('thread_id'),
            reply_to_id=data.get('reply_to_id'),
            depth=data.get('depth', 0),
            position=data.get('position'),
            attachments=[CommentAttachment.from_dict(att) for att in data.get('attachments', [])],
            mentions=[CommentMention.from_dict(mention) for mention in data.get('mentions', [])],
            reactions=[CommentReaction.from_dict(reaction) for reaction in data.get('reactions', [])],
            edit_history=[CommentEditHistory.from_dict(edit) for edit in data.get('edit_history', [])],
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            is_pinned=data.get('is_pinned', False),
            is_private=data.get('is_private', False),
            upvotes=data.get('upvotes', 0),
            downvotes=data.get('downvotes', 0)
        )


@dataclass
class CommentThread:
    """A complete comment thread."""
    id: str
    document_id: str
    room_id: str
    root_comment: Comment
    replies: List[Comment]
    participant_count: int
    last_activity: datetime
    is_resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'document_id': self.document_id,
            'room_id': self.room_id,
            'root_comment': self.root_comment.to_dict(),
            'replies': [reply.to_dict() for reply in self.replies],
            'participant_count': self.participant_count,
            'last_activity': self.last_activity.isoformat(),
            'is_resolved': self.is_resolved
        }


@dataclass
class CommentEvent:
    """Event for comment changes."""
    event_id: str
    event_type: str  # created, updated, deleted, reaction_added, etc.
    comment: Comment
    timestamp: datetime
    user_id: str
    changes: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'comment': self.comment.to_dict(),
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'changes': self.changes
        }


class CommentManager:
    """Manages threaded comments and discussions."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.comment_threads: Dict[str, CommentThread] = {}  # thread_id -> thread
        self.document_comments: Dict[str, List[Comment]] = {}  # document_id -> comments
        self.room_comments: Dict[str, List[Comment]] = {}     # room_id -> comments
        self.user_comments: Dict[str, List[Comment]] = {}     # user_id -> comments
        self.notification_callbacks: List[callable] = []
        self.cleanup_task = None
        
    async def start(self):
        """Start the comment manager."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logging.info("Comment manager started")
    
    async def stop(self):
        """Stop the comment manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        await self.redis.close()
        logging.info("Comment manager stopped")
    
    def add_notification_callback(self, callback: callable):
        """Add a callback for comment notifications."""
        self.notification_callbacks.append(callback)
    
    async def create_comment(self, document_id: str, room_id: str, user_id: str, username: str,
                           content: str, comment_type: CommentType = CommentType.TEXT,
                           parent_id: Optional[str] = None, reply_to_id: Optional[str] = None,
                           position: Optional[Dict[str, Any]] = None,
                           attachments: Optional[List[CommentAttachment]] = None,
                           mentions: Optional[List[CommentMention]] = None,
                           tags: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           is_private: bool = False,
                           user_avatar: Optional[str] = None) -> Comment:
        """Create a new comment."""
        comment_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Determine threading information
        thread_id = parent_id or comment_id  # Root comment uses its own ID as thread ID
        depth = 0
        
        if parent_id:
            parent_comment = await self.get_comment(parent_id)
            if parent_comment:
                thread_id = parent_comment.thread_id or parent_comment.id
                depth = parent_comment.depth + 1
        
        comment = Comment(
            id=comment_id,
            document_id=document_id,
            room_id=room_id,
            user_id=user_id,
            username=username,
            user_avatar=user_avatar,
            content=content,
            comment_type=comment_type,
            status=CommentStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            parent_id=parent_id,
            thread_id=thread_id,
            reply_to_id=reply_to_id,
            depth=depth,
            position=position,
            attachments=attachments or [],
            mentions=mentions or [],
            tags=tags or [],
            metadata=metadata or {},
            is_private=is_private
        )
        
        # Store comment
        await self._store_comment(comment)
        
        # Update indexes
        await self._index_comment(comment)
        
        # Update thread information
        await self._update_thread(comment)
        
        # Send notifications for mentions
        if mentions:
            await self._send_mention_notifications(comment)
        
        # Create event
        event = CommentEvent(
            event_id=str(uuid.uuid4()),
            event_type="created",
            comment=comment,
            timestamp=now,
            user_id=user_id
        )
        
        await self._store_event(event)
        
        logging.info(f"Created comment {comment_id} by {username} in document {document_id}")
        return comment
    
    async def update_comment(self, comment_id: str, user_id: str,
                           content: Optional[str] = None,
                           attachments: Optional[List[CommentAttachment]] = None,
                           tags: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           reason: Optional[str] = None) -> Optional[Comment]:
        """Update an existing comment."""
        comment = await self.get_comment(comment_id)
        if not comment:
            return None
        
        # Check permissions
        if not await self._can_edit_comment(comment, user_id):
            return None
        
        changes = {}
        
        # Store edit history
        if content is not None and content != comment.content:
            edit_history_entry = CommentEditHistory(
                timestamp=datetime.utcnow(),
                previous_content=comment.content,
                user_id=user_id,
                reason=reason
            )
            comment.edit_history.append(edit_history_entry)
            changes['content'] = {'old': comment.content, 'new': content}
            comment.content = content
            comment.status = CommentStatus.EDITED
        
        if attachments is not None:
            changes['attachments'] = {'old': len(comment.attachments), 'new': len(attachments)}
            comment.attachments = attachments
        
        if tags is not None:
            changes['tags'] = {'old': comment.tags, 'new': tags}
            comment.tags = tags
        
        if metadata is not None:
            changes['metadata'] = {'old': comment.metadata, 'new': metadata}
            comment.metadata.update(metadata)
        
        if not changes:
            return comment
        
        comment.updated_at = datetime.utcnow()
        
        # Store updated comment
        await self._store_comment(comment)
        
        # Create event
        event = CommentEvent(
            event_id=str(uuid.uuid4()),
            event_type="updated",
            comment=comment,
            timestamp=comment.updated_at,
            user_id=user_id,
            changes=changes
        )
        
        await self._store_event(event)
        
        logging.info(f"Updated comment {comment_id} by {user_id}")
        return comment
    
    async def delete_comment(self, comment_id: str, user_id: str, soft_delete: bool = True) -> bool:
        """Delete a comment (soft or hard delete)."""
        comment = await self.get_comment(comment_id)
        if not comment:
            return False
        
        # Check permissions
        if not await self._can_delete_comment(comment, user_id):
            return False
        
        if soft_delete:
            # Soft delete - mark as deleted but keep data
            comment.status = CommentStatus.DELETED
            comment.content = "[This comment has been deleted]"
            comment.updated_at = datetime.utcnow()
            await self._store_comment(comment)
        else:
            # Hard delete - remove completely
            await self._remove_comment_from_indexes(comment)
            await self.redis.delete(f"comment:{comment_id}")
        
        # Create event
        event = CommentEvent(
            event_id=str(uuid.uuid4()),
            event_type="deleted",
            comment=comment,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        
        await self._store_event(event)
        
        logging.info(f"Deleted comment {comment_id} by {user_id}")
        return True
    
    async def add_reaction(self, comment_id: str, user_id: str, username: str, emoji: str) -> Optional[Comment]:
        """Add an emoji reaction to a comment."""
        comment = await self.get_comment(comment_id)
        if not comment:
            return None
        
        # Check if user already reacted with this emoji
        existing_reaction = next(
            (r for r in comment.reactions if r.user_id == user_id and r.emoji == emoji),
            None
        )
        
        if existing_reaction:
            return comment  # Already reacted
        
        # Add reaction
        reaction = CommentReaction(
            emoji=emoji,
            user_id=user_id,
            username=username,
            timestamp=datetime.utcnow()
        )
        
        comment.reactions.append(reaction)
        comment.updated_at = datetime.utcnow()
        
        await self._store_comment(comment)
        
        # Create event
        event = CommentEvent(
            event_id=str(uuid.uuid4()),
            event_type="reaction_added",
            comment=comment,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            changes={'emoji': emoji}
        )
        
        await self._store_event(event)
        
        return comment
    
    async def remove_reaction(self, comment_id: str, user_id: str, emoji: str) -> Optional[Comment]:
        """Remove an emoji reaction from a comment."""
        comment = await self.get_comment(comment_id)
        if not comment:
            return None
        
        # Find and remove reaction
        original_count = len(comment.reactions)
        comment.reactions = [
            r for r in comment.reactions 
            if not (r.user_id == user_id and r.emoji == emoji)
        ]
        
        if len(comment.reactions) == original_count:
            return comment  # No reaction found
        
        comment.updated_at = datetime.utcnow()
        await self._store_comment(comment)
        
        # Create event
        event = CommentEvent(
            event_id=str(uuid.uuid4()),
            event_type="reaction_removed",
            comment=comment,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            changes={'emoji': emoji}
        )
        
        await self._store_event(event)
        
        return comment
    
    async def pin_comment(self, comment_id: str, user_id: str) -> Optional[Comment]:
        """Pin a comment."""
        comment = await self.get_comment(comment_id)
        if not comment:
            return None
        
        comment.is_pinned = True
        comment.updated_at = datetime.utcnow()
        
        await self._store_comment(comment)
        
        # Create event
        event = CommentEvent(
            event_id=str(uuid.uuid4()),
            event_type="pinned",
            comment=comment,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        
        await self._store_event(event)
        
        return comment
    
    async def resolve_thread(self, thread_id: str, user_id: str) -> bool:
        """Mark a comment thread as resolved."""
        thread = await self.get_thread(thread_id)
        if not thread:
            return False
        
        thread.is_resolved = True
        
        # Update root comment
        root_comment = thread.root_comment
        root_comment.status = CommentStatus.RESOLVED
        root_comment.updated_at = datetime.utcnow()
        
        await self._store_comment(root_comment)
        
        # Store updated thread info
        await self._store_thread_info(thread)
        
        logging.info(f"Resolved thread {thread_id} by {user_id}")
        return True
    
    async def get_comment(self, comment_id: str) -> Optional[Comment]:
        """Get a specific comment by ID."""
        data = await self.redis.get(f"comment:{comment_id}")
        if data:
            try:
                comment_data = json.loads(data)
                return Comment.from_dict(comment_data)
            except json.JSONDecodeError:
                logging.error(f"Invalid comment data for ID {comment_id}")
        
        return None
    
    async def get_document_comments(self, document_id: str, include_private: bool = False,
                                  include_deleted: bool = False) -> List[Comment]:
        """Get all comments for a document."""
        comment_ids = await self.redis.smembers(f"document_comments:{document_id}")
        comments = []
        
        for comment_id in comment_ids:
            comment = await self.get_comment(comment_id)
            if comment:
                if not include_private and comment.is_private:
                    continue
                if not include_deleted and comment.status == CommentStatus.DELETED:
                    continue
                comments.append(comment)
        
        # Sort by creation time
        comments.sort(key=lambda c: c.created_at)
        return comments
    
    async def get_thread(self, thread_id: str) -> Optional[CommentThread]:
        """Get a complete comment thread."""
        # Get thread info from cache first
        if thread_id in self.comment_threads:
            return self.comment_threads[thread_id]
        
        # Load from Redis
        thread_data = await self.redis.get(f"thread:{thread_id}")
        if thread_data:
            try:
                data = json.loads(thread_data)
                thread = CommentThread(
                    id=data['id'],
                    document_id=data['document_id'],
                    room_id=data['room_id'],
                    root_comment=Comment.from_dict(data['root_comment']),
                    replies=[Comment.from_dict(reply) for reply in data['replies']],
                    participant_count=data['participant_count'],
                    last_activity=datetime.fromisoformat(data['last_activity']),
                    is_resolved=data.get('is_resolved', False)
                )
                
                self.comment_threads[thread_id] = thread
                return thread
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Invalid thread data for ID {thread_id}: {e}")
        
        return None
    
    async def get_document_threads(self, document_id: str) -> List[CommentThread]:
        """Get all comment threads for a document."""
        thread_ids = await self.redis.smembers(f"document_threads:{document_id}")
        threads = []
        
        for thread_id in thread_ids:
            thread = await self.get_thread(thread_id)
            if thread:
                threads.append(thread)
        
        # Sort by last activity
        threads.sort(key=lambda t: t.last_activity, reverse=True)
        return threads
    
    async def search_comments(self, document_id: str, query: str,
                            user_id: Optional[str] = None) -> List[Comment]:
        """Search comments by content."""
        comments = await self.get_document_comments(document_id)
        results = []
        
        query_lower = query.lower()
        for comment in comments:
            if (query_lower in comment.content.lower() or
                any(query_lower in tag.lower() for tag in comment.tags)):
                if user_id is None or not comment.is_private or comment.user_id == user_id:
                    results.append(comment)
        
        return results
    
    async def _store_comment(self, comment: Comment):
        """Store comment in Redis."""
        try:
            data = json.dumps(comment.to_dict())
            await self.redis.set(f"comment:{comment.id}", data)
        except Exception as e:
            logging.error(f"Error storing comment: {e}")
    
    async def _index_comment(self, comment: Comment):
        """Add comment to various indexes."""
        try:
            # Document index
            await self.redis.sadd(f"document_comments:{comment.document_id}", comment.id)
            
            # Room index
            await self.redis.sadd(f"room_comments:{comment.room_id}", comment.id)
            
            # User index
            await self.redis.sadd(f"user_comments:{comment.user_id}", comment.id)
            
            # Thread index
            if comment.thread_id:
                await self.redis.sadd(f"thread_comments:{comment.thread_id}", comment.id)
            
        except Exception as e:
            logging.error(f"Error indexing comment: {e}")
    
    async def _update_thread(self, comment: Comment):
        """Update thread information when a comment is added."""
        thread_id = comment.thread_id
        if not thread_id:
            return
        
        # Get existing thread or create new one
        thread = await self.get_thread(thread_id)
        
        if not thread:
            # Create new thread
            if comment.parent_id is None:
                # This is the root comment
                thread = CommentThread(
                    id=thread_id,
                    document_id=comment.document_id,
                    room_id=comment.room_id,
                    root_comment=comment,
                    replies=[],
                    participant_count=1,
                    last_activity=comment.created_at
                )
            else:
                # This shouldn't happen, but handle gracefully
                logging.warning(f"Thread {thread_id} not found for reply comment {comment.id}")
                return
        else:
            # Update existing thread
            if comment.parent_id is None:
                thread.root_comment = comment
            else:
                # Add to replies if not already there
                if comment.id not in [r.id for r in thread.replies]:
                    thread.replies.append(comment)
            
            # Update participant count
            participants = set([thread.root_comment.user_id])
            participants.update(reply.user_id for reply in thread.replies)
            thread.participant_count = len(participants)
            
            thread.last_activity = comment.updated_at
        
        # Store updated thread
        await self._store_thread_info(thread)
        
        # Index thread
        await self.redis.sadd(f"document_threads:{comment.document_id}", thread_id)
        
        # Cache locally
        self.comment_threads[thread_id] = thread
    
    async def _store_thread_info(self, thread: CommentThread):
        """Store thread information in Redis."""
        try:
            data = json.dumps(thread.to_dict())
            await self.redis.set(f"thread:{thread.id}", data)
        except Exception as e:
            logging.error(f"Error storing thread: {e}")
    
    async def _send_mention_notifications(self, comment: Comment):
        """Send notifications for user mentions."""
        for mention in comment.mentions:
            for callback in self.notification_callbacks:
                try:
                    await callback('mention', {
                        'comment': comment.to_dict(),
                        'mentioned_user': mention.to_dict()
                    })
                except Exception as e:
                    logging.error(f"Error sending mention notification: {e}")
    
    async def _can_edit_comment(self, comment: Comment, user_id: str) -> bool:
        """Check if user can edit comment."""
        return comment.user_id == user_id
    
    async def _can_delete_comment(self, comment: Comment, user_id: str) -> bool:
        """Check if user can delete comment."""
        return comment.user_id == user_id
    
    async def _remove_comment_from_indexes(self, comment: Comment):
        """Remove comment from all indexes."""
        try:
            await self.redis.srem(f"document_comments:{comment.document_id}", comment.id)
            await self.redis.srem(f"room_comments:{comment.room_id}", comment.id)
            await self.redis.srem(f"user_comments:{comment.user_id}", comment.id)
            if comment.thread_id:
                await self.redis.srem(f"thread_comments:{comment.thread_id}", comment.id)
        except Exception as e:
            logging.error(f"Error removing comment from indexes: {e}")
    
    async def _store_event(self, event: CommentEvent):
        """Store comment event for audit trail."""
        try:
            data = json.dumps(event.to_dict())
            await self.redis.lpush(f"comment_events:{event.comment.document_id}", data)
            await self.redis.ltrim(f"comment_events:{event.comment.document_id}", 0, 999)
        except Exception as e:
            logging.error(f"Error storing comment event: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of comment caches."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_caches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in comment cleanup: {e}")
    
    async def _cleanup_caches(self):
        """Clean up local comment caches."""
        # Clean up old cached threads
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        old_threads = [
            thread_id for thread_id, thread in self.comment_threads.items()
            if thread.last_activity < cutoff_time
        ]
        
        for thread_id in old_threads:
            del self.comment_threads[thread_id]
    
    def get_comment_stats(self) -> Dict[str, Any]:
        """Get comment statistics."""
        total_threads = len(self.comment_threads)
        resolved_threads = sum(1 for t in self.comment_threads.values() if t.is_resolved)
        
        return {
            'total_threads': total_threads,
            'resolved_threads': resolved_threads,
            'active_threads': total_threads - resolved_threads,
            'cached_threads': len(self.comment_threads)
        }


# Global instance
comment_manager = CommentManager()