"""
Event Bus Implementation

Central event bus for publishing and subscribing to events across services.
Supports event filtering, routing, and guaranteed delivery.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Event:
    """Represents an event in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    source: str = ""
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    status: EventStatus = EventStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "data": self.data,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        event = cls()
        event.id = data.get("id", event.id)
        event.type = data.get("type", "")
        event.source = data.get("source", "")
        event.target = data.get("target")
        event.data = data.get("data", {})
        event.metadata = data.get("metadata", {})
        event.priority = EventPriority(data.get("priority", EventPriority.NORMAL.value))
        event.timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        event.correlation_id = data.get("correlation_id")
        event.causation_id = data.get("causation_id")
        event.retry_count = data.get("retry_count", 0)
        event.max_retries = data.get("max_retries", 3)
        event.status = EventStatus(data.get("status", EventStatus.PENDING.value))
        return event


@dataclass
class EventHandler:
    """Event handler configuration"""
    handler_id: str
    event_type: str
    handler_func: Callable
    service_name: str
    filters: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    timeout: float = 30.0
    retry_strategy: str = "exponential_backoff"


class EventBus:
    """
    Central event bus for inter-service communication
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_store: Dict[str, Event] = {}
        self.dead_letter_queue: List[Event] = []
        self.processing_events: Set[str] = set()
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
        # Configuration
        self.max_concurrent_events = self.config.get('max_concurrent_events', 100)
        self.event_retention_hours = self.config.get('event_retention_hours', 24)
        self.dead_letter_max_size = self.config.get('dead_letter_max_size', 1000)
        
        # Statistics
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "handlers_registered": 0,
            "dead_letter_count": 0
        }
    
    async def start(self):
        """Start the event bus"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting Event Bus")
        
        # Start background tasks
        processor_task = asyncio.create_task(self._event_processor_loop())
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._tasks.update([processor_task, cleanup_task])
        
        # Register event handlers for new features
        await self._register_feature_handlers()
        
        logger.info("Event Bus started successfully")
    
    async def stop(self):
        """Stop the event bus"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping Event Bus")
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("Event Bus stopped")
    
    def subscribe(
        self, 
        event_type: str, 
        handler: Callable,
        service_name: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Subscribe to events of a specific type"""
        handler_id = str(uuid.uuid4())
        
        event_handler = EventHandler(
            handler_id=handler_id,
            event_type=event_type,
            handler_func=handler,
            service_name=service_name,
            filters=filters or {},
            **kwargs
        )
        
        self.handlers[event_type].append(event_handler)
        self.stats["handlers_registered"] += 1
        
        logger.info(f"Registered handler {handler_id} for event type {event_type}")
        return handler_id
    
    def unsubscribe(self, handler_id: str) -> bool:
        """Unsubscribe event handler"""
        for event_type, handlers in self.handlers.items():
            for i, handler in enumerate(handlers):
                if handler.handler_id == handler_id:
                    del handlers[i]
                    logger.info(f"Unregistered handler {handler_id}")
                    return True
        
        logger.warning(f"Handler {handler_id} not found")
        return False
    
    async def publish(self, event: Event) -> str:
        """Publish an event to the bus"""
        # Store event
        self.event_store[event.id] = event
        self.stats["events_published"] += 1
        
        logger.debug(f"Published event {event.id} of type {event.type}")
        
        # Trigger immediate processing for high priority events
        if event.priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
            asyncio.create_task(self._process_event(event))
        
        return event.id
    
    async def publish_dict(self, event_data: Dict[str, Any]) -> str:
        """Publish event from dictionary data"""
        event = Event.from_dict(event_data)
        return await self.publish(event)
    
    async def _event_processor_loop(self):
        """Background loop for processing events"""
        while self._running:
            try:
                # Process pending events
                pending_events = [
                    event for event in self.event_store.values()
                    if event.status == EventStatus.PENDING 
                    and event.id not in self.processing_events
                ]
                
                # Sort by priority and timestamp
                pending_events.sort(
                    key=lambda x: (x.priority.value, x.timestamp), 
                    reverse=True
                )
                
                # Process events with concurrency limit
                semaphore = asyncio.Semaphore(self.max_concurrent_events)
                tasks = []
                
                for event in pending_events[:self.max_concurrent_events]:
                    if len(self.processing_events) >= self.max_concurrent_events:
                        break
                    
                    task = asyncio.create_task(
                        self._process_event_with_semaphore(event, semaphore)
                    )
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                logger.error(f"Error in event processor loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_with_semaphore(self, event: Event, semaphore: asyncio.Semaphore):
        """Process event with semaphore for concurrency control"""
        async with semaphore:
            await self._process_event(event)
    
    async def _process_event(self, event: Event):
        """Process a single event"""
        if event.id in self.processing_events:
            return
        
        self.processing_events.add(event.id)
        event.status = EventStatus.PROCESSING
        
        try:
            # Find handlers for this event type
            handlers = self.handlers.get(event.type, [])
            
            # Apply filters
            filtered_handlers = []
            for handler in handlers:
                if self._match_filters(event, handler.filters):
                    filtered_handlers.append(handler)
            
            if not filtered_handlers:
                logger.debug(f"No handlers found for event {event.id} of type {event.type}")
                event.status = EventStatus.COMPLETED
                self.processing_events.discard(event.id)
                return
            
            # Execute handlers
            handler_tasks = []
            for handler in filtered_handlers:
                task = asyncio.create_task(
                    self._execute_handler(event, handler)
                )
                handler_tasks.append(task)
            
            # Wait for all handlers to complete
            results = await asyncio.gather(*handler_tasks, return_exceptions=True)
            
            # Check results
            failed_handlers = [
                result for result in results 
                if isinstance(result, Exception)
            ]
            
            if failed_handlers:
                logger.error(f"Some handlers failed for event {event.id}: {failed_handlers}")
                await self._handle_event_failure(event)
            else:
                event.status = EventStatus.COMPLETED
                self.stats["events_processed"] += 1
                logger.debug(f"Successfully processed event {event.id}")
        
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")
            await self._handle_event_failure(event)
        
        finally:
            self.processing_events.discard(event.id)
    
    async def _execute_handler(self, event: Event, handler: EventHandler):
        """Execute a single event handler"""
        try:
            # Set timeout
            await asyncio.wait_for(
                handler.handler_func(event),
                timeout=handler.timeout
            )
            
            logger.debug(f"Handler {handler.handler_id} completed for event {event.id}")
            
        except asyncio.TimeoutError:
            logger.error(f"Handler {handler.handler_id} timed out for event {event.id}")
            raise
        except Exception as e:
            logger.error(f"Handler {handler.handler_id} failed for event {event.id}: {e}")
            raise
    
    def _match_filters(self, event: Event, filters: Dict[str, Any]) -> bool:
        """Check if event matches handler filters"""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key == "source":
                if event.source != expected_value:
                    return False
            elif key == "target":
                if event.target != expected_value:
                    return False
            elif key == "correlation_id":
                if event.correlation_id != expected_value:
                    return False
            elif key in event.data:
                if event.data[key] != expected_value:
                    return False
            elif key in event.metadata:
                if event.metadata[key] != expected_value:
                    return False
            else:
                return False
        
        return True
    
    async def _handle_event_failure(self, event: Event):
        """Handle event processing failure"""
        event.retry_count += 1
        
        if event.retry_count <= event.max_retries:
            # Retry with exponential backoff
            delay = min(2 ** event.retry_count, 300)  # Max 5 minutes
            event.status = EventStatus.PENDING
            
            # Schedule retry
            async def retry_later():
                await asyncio.sleep(delay)
                if event.id in self.event_store:
                    await self._process_event(event)
            
            asyncio.create_task(retry_later())
            logger.info(f"Retrying event {event.id} in {delay} seconds (attempt {event.retry_count})")
        else:
            # Move to dead letter queue
            event.status = EventStatus.DEAD_LETTER
            self.dead_letter_queue.append(event)
            self.stats["events_failed"] += 1
            self.stats["dead_letter_count"] += 1
            
            # Limit dead letter queue size
            if len(self.dead_letter_queue) > self.dead_letter_max_size:
                self.dead_letter_queue.pop(0)
            
            logger.error(f"Event {event.id} moved to dead letter queue after {event.retry_count} retries")
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up old events"""
        while self._running:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=self.event_retention_hours)
                
                # Remove old completed events
                events_to_remove = []
                for event_id, event in self.event_store.items():
                    if (event.status == EventStatus.COMPLETED and 
                        event.timestamp < cutoff_time):
                        events_to_remove.append(event_id)
                
                for event_id in events_to_remove:
                    del self.event_store[event_id]
                
                if events_to_remove:
                    logger.info(f"Cleaned up {len(events_to_remove)} old events")
                
                # Clean up old dead letter events
                self.dead_letter_queue = [
                    event for event in self.dead_letter_queue
                    if event.timestamp >= cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _register_feature_handlers(self):
        """Register event handlers for new features"""
        # Voice interface events
        self.subscribe(
            "voice.command.received", 
            self._handle_voice_command,
            "voice-interface"
        )
        
        # Document comparison events
        self.subscribe(
            "document.comparison.requested",
            self._handle_document_comparison,
            "document-comparison"
        )
        
        # White-label events
        self.subscribe(
            "tenant.created",
            self._handle_tenant_creation,
            "whitelabel-service"
        )
        
        # Compliance automation events
        self.subscribe(
            "compliance.check.triggered",
            self._handle_compliance_check,
            "compliance-automation"
        )
        
        # Visualization events
        self.subscribe(
            "visualization.generate.requested",
            self._handle_visualization_request,
            "visualization-engine"
        )
    
    async def _handle_voice_command(self, event: Event):
        """Handle voice command events"""
        logger.info(f"Processing voice command: {event.data.get('command')}")
        # Integration with voice interface service
    
    async def _handle_document_comparison(self, event: Event):
        """Handle document comparison events"""
        logger.info(f"Processing document comparison: {event.data.get('document_ids')}")
        # Integration with document comparison service
    
    async def _handle_tenant_creation(self, event: Event):
        """Handle tenant creation events"""
        logger.info(f"Processing tenant creation: {event.data.get('tenant_id')}")
        # Integration with white-label service
    
    async def _handle_compliance_check(self, event: Event):
        """Handle compliance check events"""
        logger.info(f"Processing compliance check: {event.data.get('document_id')}")
        # Integration with compliance automation service
    
    async def _handle_visualization_request(self, event: Event):
        """Handle visualization generation events"""
        logger.info(f"Processing visualization request: {event.data.get('visualization_type')}")
        # Integration with visualization engine
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self.stats,
            "event_store_size": len(self.event_store),
            "processing_events": len(self.processing_events),
            "dead_letter_size": len(self.dead_letter_queue),
            "registered_handlers": sum(len(handlers) for handlers in self.handlers.values()),
            "event_types": list(self.handlers.keys())
        }
    
    def get_event_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific event"""
        event = self.event_store.get(event_id)
        if event:
            return {
                "id": event.id,
                "type": event.type,
                "status": event.status.value,
                "retry_count": event.retry_count,
                "timestamp": event.timestamp.isoformat(),
                "processing": event_id in self.processing_events
            }
        return None
    
    async def replay_dead_letter_events(self, max_count: Optional[int] = None) -> int:
        """Replay events from dead letter queue"""
        events_to_replay = self.dead_letter_queue[:max_count] if max_count else self.dead_letter_queue[:]
        replayed_count = 0
        
        for event in events_to_replay:
            # Reset event status and retry count
            event.status = EventStatus.PENDING
            event.retry_count = 0
            
            # Move back to main event store
            self.event_store[event.id] = event
            self.dead_letter_queue.remove(event)
            
            replayed_count += 1
        
        logger.info(f"Replayed {replayed_count} events from dead letter queue")
        return replayed_count