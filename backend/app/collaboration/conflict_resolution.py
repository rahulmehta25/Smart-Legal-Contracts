"""
CRDT-based conflict resolution for collaborative editing.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import json
import hashlib

import redis.asyncio as redis
from pydantic import BaseModel, Field


class OperationType(Enum):
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"
    FORMAT = "format"
    REPLACE = "replace"


class CRDTType(Enum):
    G_COUNTER = "g_counter"
    PN_COUNTER = "pn_counter"
    G_SET = "g_set"
    TWO_P_SET = "2p_set"
    LWW_REGISTER = "lww_register"
    OR_SET = "or_set"
    SEQUENCE = "sequence"
    TEXT = "text"


@dataclass
class VectorClock:
    """Vector clock for tracking causality."""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str):
        """Increment the clock for a node."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        """Update with another vector clock."""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this vector clock happens before another."""
        less_than_or_equal = True
        strictly_less = False
        
        for node_id in set(self.clocks.keys()) | set(other.clocks.keys()):
            self_clock = self.clocks.get(node_id, 0)
            other_clock = other.clocks.get(node_id, 0)
            
            if self_clock > other_clock:
                less_than_or_equal = False
                break
            elif self_clock < other_clock:
                strictly_less = True
        
        return less_than_or_equal and strictly_less
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if this vector clock is concurrent with another."""
        return not (self.happens_before(other) or other.happens_before(self))
    
    def to_dict(self) -> Dict[str, Any]:
        return {'clocks': self.clocks}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorClock':
        return cls(clocks=data.get('clocks', {}))


@dataclass
class Operation:
    """A single operation in the CRDT."""
    id: str
    operation_type: OperationType
    position: int
    content: Any
    length: int
    user_id: str
    timestamp: datetime
    vector_clock: VectorClock
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'operation_type': self.operation_type.value,
            'position': self.position,
            'content': self.content,
            'length': self.length,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'vector_clock': self.vector_clock.to_dict(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        return cls(
            id=data['id'],
            operation_type=OperationType(data['operation_type']),
            position=data['position'],
            content=data['content'],
            length=data['length'],
            user_id=data['user_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            vector_clock=VectorClock.from_dict(data['vector_clock']),
            metadata=data.get('metadata', {})
        )


@dataclass
class CRDTState:
    """State of a CRDT document."""
    document_id: str
    crdt_type: CRDTType
    content: str
    operations: List[Operation]
    vector_clock: VectorClock
    checksum: str
    version: int
    last_modified: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'crdt_type': self.crdt_type.value,
            'content': self.content,
            'operations': [op.to_dict() for op in self.operations],
            'vector_clock': self.vector_clock.to_dict(),
            'checksum': self.checksum,
            'version': self.version,
            'last_modified': self.last_modified.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTState':
        return cls(
            document_id=data['document_id'],
            crdt_type=CRDTType(data['crdt_type']),
            content=data['content'],
            operations=[Operation.from_dict(op) for op in data['operations']],
            vector_clock=VectorClock.from_dict(data['vector_clock']),
            checksum=data['checksum'],
            version=data['version'],
            last_modified=datetime.fromisoformat(data['last_modified'])
        )


class TextCRDT:
    """Text-based CRDT implementation using RGA (Replicated Growable Array)."""
    
    def __init__(self, document_id: str, node_id: str):
        self.document_id = document_id
        self.node_id = node_id
        self.content = ""
        self.operations: List[Operation] = []
        self.vector_clock = VectorClock()
        self.version = 0
        
    def insert(self, position: int, text: str, user_id: str) -> Operation:
        """Insert text at position."""
        operation_id = f"{self.node_id}_{self.version}_{uuid.uuid4().hex[:8]}"
        self.vector_clock.increment(self.node_id)
        
        operation = Operation(
            id=operation_id,
            operation_type=OperationType.INSERT,
            position=position,
            content=text,
            length=len(text),
            user_id=user_id,
            timestamp=datetime.utcnow(),
            vector_clock=VectorClock(clocks=self.vector_clock.clocks.copy())
        )
        
        # Apply operation locally
        self._apply_operation(operation)
        
        return operation
    
    def delete(self, position: int, length: int, user_id: str) -> Operation:
        """Delete text at position."""
        operation_id = f"{self.node_id}_{self.version}_{uuid.uuid4().hex[:8]}"
        self.vector_clock.increment(self.node_id)
        
        # Get the content being deleted for the operation
        deleted_content = self.content[position:position + length]
        
        operation = Operation(
            id=operation_id,
            operation_type=OperationType.DELETE,
            position=position,
            content=deleted_content,
            length=length,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            vector_clock=VectorClock(clocks=self.vector_clock.clocks.copy())
        )
        
        # Apply operation locally
        self._apply_operation(operation)
        
        return operation
    
    def apply_remote_operation(self, operation: Operation) -> bool:
        """Apply an operation from a remote node."""
        # Check if we've already applied this operation
        if any(op.id == operation.id for op in self.operations):
            return False
        
        # Update vector clock
        self.vector_clock.update(operation.vector_clock)
        
        # Transform the operation if necessary
        transformed_op = self._transform_operation(operation)
        
        # Apply the transformed operation
        self._apply_operation(transformed_op)
        
        return True
    
    def _apply_operation(self, operation: Operation):
        """Apply an operation to the local state."""
        if operation.operation_type == OperationType.INSERT:
            self.content = (self.content[:operation.position] + 
                          operation.content + 
                          self.content[operation.position:])
        
        elif operation.operation_type == OperationType.DELETE:
            end_pos = operation.position + operation.length
            self.content = (self.content[:operation.position] + 
                          self.content[end_pos:])
        
        # Add to operation history
        self.operations.append(operation)
        self.operations.sort(key=lambda op: (op.timestamp, op.id))
        
        self.version += 1
    
    def _transform_operation(self, operation: Operation) -> Operation:
        """Transform an operation based on concurrent operations."""
        transformed_position = operation.position
        
        # Transform against all operations that happened concurrently
        for local_op in self.operations:
            if (local_op.vector_clock.concurrent_with(operation.vector_clock) and
                local_op.timestamp <= operation.timestamp):
                
                if local_op.position <= operation.position:
                    if local_op.operation_type == OperationType.INSERT:
                        transformed_position += local_op.length
                    elif local_op.operation_type == OperationType.DELETE:
                        transformed_position -= local_op.length
        
        # Create transformed operation
        transformed_op = Operation(
            id=operation.id,
            operation_type=operation.operation_type,
            position=max(0, transformed_position),
            content=operation.content,
            length=operation.length,
            user_id=operation.user_id,
            timestamp=operation.timestamp,
            vector_clock=operation.vector_clock,
            metadata=operation.metadata
        )
        
        return transformed_op
    
    def get_state(self) -> CRDTState:
        """Get current CRDT state."""
        checksum = hashlib.md5(self.content.encode()).hexdigest()
        
        return CRDTState(
            document_id=self.document_id,
            crdt_type=CRDTType.TEXT,
            content=self.content,
            operations=self.operations[-100:],  # Keep last 100 operations
            vector_clock=self.vector_clock,
            checksum=checksum,
            version=self.version,
            last_modified=datetime.utcnow()
        )
    
    def merge_state(self, other_state: CRDTState):
        """Merge with another CRDT state."""
        # Apply all remote operations
        for operation in other_state.operations:
            self.apply_remote_operation(operation)
        
        # Update vector clock
        self.vector_clock.update(other_state.vector_clock)


class CRDTManager:
    """Manages CRDT documents and conflict resolution."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.documents: Dict[str, TextCRDT] = {}  # document_id -> CRDT
        self.node_id = str(uuid.uuid4())
        self.operation_queue: Dict[str, List[Operation]] = {}  # document_id -> operations
        self.sync_intervals: Dict[str, float] = {}  # document_id -> sync interval
        self.cleanup_task = None
        self.sync_task = None
        
    async def start(self):
        """Start the CRDT manager."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.sync_task = asyncio.create_task(self._sync_loop())
        logging.info(f"CRDT manager started with node ID: {self.node_id}")
    
    async def stop(self):
        """Stop the CRDT manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.sync_task:
            self.sync_task.cancel()
        
        try:
            if self.cleanup_task:
                await self.cleanup_task
            if self.sync_task:
                await self.sync_task
        except asyncio.CancelledError:
            pass
        
        await self.redis.close()
        logging.info("CRDT manager stopped")
    
    async def create_document(self, document_id: str, initial_content: str = "") -> TextCRDT:
        """Create a new CRDT document."""
        if document_id in self.documents:
            return self.documents[document_id]
        
        crdt = TextCRDT(document_id, self.node_id)
        crdt.content = initial_content
        
        self.documents[document_id] = crdt
        self.operation_queue[document_id] = []
        
        # Store initial state in Redis
        await self._store_document_state(crdt)
        
        logging.info(f"Created CRDT document: {document_id}")
        return crdt
    
    async def get_document(self, document_id: str) -> Optional[TextCRDT]:
        """Get a CRDT document."""
        if document_id in self.documents:
            return self.documents[document_id]
        
        # Try to load from Redis
        state_data = await self.redis.get(f"crdt_state:{document_id}")
        if state_data:
            try:
                state = CRDTState.from_dict(json.loads(state_data))
                crdt = TextCRDT(document_id, self.node_id)
                crdt.merge_state(state)
                
                self.documents[document_id] = crdt
                self.operation_queue[document_id] = []
                
                return crdt
            except Exception as e:
                logging.error(f"Error loading CRDT document: {e}")
        
        return None
    
    async def apply_operation(self, document_id: str, operation: Operation) -> bool:
        """Apply an operation to a document."""
        crdt = await self.get_document(document_id)
        if not crdt:
            return False
        
        # Add to operation queue for broadcasting
        if document_id not in self.operation_queue:
            self.operation_queue[document_id] = []
        
        # Apply operation
        if operation.user_id == crdt.node_id:
            # Local operation - already applied, just queue for broadcast
            self.operation_queue[document_id].append(operation)
        else:
            # Remote operation - apply and queue for broadcast
            success = crdt.apply_remote_operation(operation)
            if success:
                self.operation_queue[document_id].append(operation)
            else:
                return False
        
        # Store updated state
        await self._store_document_state(crdt)
        
        return True
    
    async def insert_text(self, document_id: str, position: int, text: str, user_id: str) -> Optional[Operation]:
        """Insert text into a document."""
        crdt = await self.get_document(document_id)
        if not crdt:
            return None
        
        operation = crdt.insert(position, text, user_id)
        
        # Queue for broadcasting
        if document_id not in self.operation_queue:
            self.operation_queue[document_id] = []
        self.operation_queue[document_id].append(operation)
        
        # Store updated state
        await self._store_document_state(crdt)
        
        return operation
    
    async def delete_text(self, document_id: str, position: int, length: int, user_id: str) -> Optional[Operation]:
        """Delete text from a document."""
        crdt = await self.get_document(document_id)
        if not crdt:
            return None
        
        operation = crdt.delete(position, length, user_id)
        
        # Queue for broadcasting
        if document_id not in self.operation_queue:
            self.operation_queue[document_id] = []
        self.operation_queue[document_id].append(operation)
        
        # Store updated state
        await self._store_document_state(crdt)
        
        return operation
    
    async def get_pending_operations(self, document_id: str) -> List[Operation]:
        """Get pending operations for a document."""
        return self.operation_queue.get(document_id, [])
    
    async def clear_pending_operations(self, document_id: str):
        """Clear pending operations for a document."""
        if document_id in self.operation_queue:
            self.operation_queue[document_id].clear()
    
    async def sync_document(self, document_id: str, remote_state: CRDTState) -> bool:
        """Sync document with remote state."""
        crdt = await self.get_document(document_id)
        if not crdt:
            crdt = await self.create_document(document_id)
        
        # Merge remote state
        crdt.merge_state(remote_state)
        
        # Store updated state
        await self._store_document_state(crdt)
        
        logging.debug(f"Synced document {document_id}")
        return True
    
    async def get_document_state(self, document_id: str) -> Optional[CRDTState]:
        """Get current document state."""
        crdt = await self.get_document(document_id)
        if not crdt:
            return None
        
        return crdt.get_state()
    
    async def resolve_conflicts(self, document_id: str, operations: List[Operation]) -> List[Operation]:
        """Resolve conflicts between operations."""
        crdt = await self.get_document(document_id)
        if not crdt:
            return []
        
        resolved_operations = []
        
        # Sort operations by timestamp and apply them
        sorted_operations = sorted(operations, key=lambda op: (op.timestamp, op.id))
        
        for operation in sorted_operations:
            # Transform operation against already applied operations
            transformed_op = crdt._transform_operation(operation)
            
            # Apply if not already applied
            if not any(op.id == operation.id for op in crdt.operations):
                crdt._apply_operation(transformed_op)
                resolved_operations.append(transformed_op)
        
        # Store updated state
        await self._store_document_state(crdt)
        
        return resolved_operations
    
    async def _store_document_state(self, crdt: TextCRDT):
        """Store document state in Redis."""
        try:
            state = crdt.get_state()
            data = json.dumps(state.to_dict())
            await self.redis.set(f"crdt_state:{crdt.document_id}", data)
            await self.redis.expire(f"crdt_state:{crdt.document_id}", 86400)  # 24 hours
        except Exception as e:
            logging.error(f"Error storing CRDT state: {e}")
    
    async def _store_operation(self, operation: Operation):
        """Store operation in Redis for persistence."""
        try:
            data = json.dumps(operation.to_dict())
            await self.redis.lpush(f"crdt_operations:{operation.document_id}", data)
            await self.redis.ltrim(f"crdt_operations:{operation.document_id}", 0, 999)  # Keep last 1000
        except Exception as e:
            logging.error(f"Error storing CRDT operation: {e}")
    
    async def _sync_loop(self):
        """Periodic synchronization with Redis."""
        while True:
            try:
                await asyncio.sleep(5)  # Sync every 5 seconds
                await self._sync_all_documents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in CRDT sync loop: {e}")
    
    async def _sync_all_documents(self):
        """Sync all documents with Redis."""
        for document_id, crdt in self.documents.items():
            try:
                # Store current state
                await self._store_document_state(crdt)
                
                # Store pending operations
                for operation in self.operation_queue.get(document_id, []):
                    await self._store_operation(operation)
                
            except Exception as e:
                logging.error(f"Error syncing document {document_id}: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of CRDT data."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_operations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in CRDT cleanup: {e}")
    
    async def _cleanup_old_operations(self):
        """Clean up old operations."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for document_id, crdt in list(self.documents.items()):
            # Remove old operations
            crdt.operations = [
                op for op in crdt.operations 
                if op.timestamp > cutoff_time
            ]
            
            # Clear old pending operations
            if document_id in self.operation_queue:
                self.operation_queue[document_id] = [
                    op for op in self.operation_queue[document_id]
                    if op.timestamp > cutoff_time
                ]
    
    def get_crdt_stats(self) -> Dict[str, Any]:
        """Get CRDT statistics."""
        total_documents = len(self.documents)
        total_operations = sum(len(crdt.operations) for crdt in self.documents.values())
        pending_operations = sum(len(ops) for ops in self.operation_queue.values())
        
        return {
            'node_id': self.node_id,
            'total_documents': total_documents,
            'total_operations': total_operations,
            'pending_operations': pending_operations,
            'documents': list(self.documents.keys())
        }


# Global instance
crdt_manager = CRDTManager()