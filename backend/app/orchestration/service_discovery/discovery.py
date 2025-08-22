"""
Service Discovery Implementation

Manages service registration, discovery, and health monitoring for all microservices.
Integrates with service mesh and provides failover capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import time

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"


class DiscoveryStrategy(Enum):
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    IN_MEMORY = "in_memory"


@dataclass
class ServiceInstance:
    """Represents a service instance in the discovery system"""
    service_id: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    version: str = "1.0.0"
    environment: str = "production"
    datacenter: str = "dc1"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.STARTING
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    
    def __post_init__(self):
        if not self.health_check:
            self.health_check = {
                "type": "http",
                "path": "/health",
                "interval": "30s",
                "timeout": "5s",
                "deregister_after": "300s"
            }
    
    @property
    def address(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def health_score(self) -> float:
        """Calculate health score based on success/failure ratio"""
        total_checks = self.success_count + self.failure_count
        if total_checks == 0:
            return 1.0
        
        success_rate = self.success_count / total_checks
        recency_factor = 1.0
        
        # Factor in recency of last heartbeat
        if self.last_heartbeat:
            minutes_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds() / 60
            recency_factor = max(0.1, 1.0 - (minutes_since_heartbeat / 10))
        
        return success_rate * recency_factor
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "version": self.version,
            "environment": self.environment,
            "datacenter": self.datacenter,
            "tags": self.tags,
            "metadata": self.metadata,
            "health_check": self.health_check,
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "health_score": self.health_score,
            "address": self.address
        }


@dataclass
class ServiceQuery:
    """Query parameters for service discovery"""
    service_name: Optional[str] = None
    environment: Optional[str] = None
    datacenter: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: Optional[ServiceStatus] = None
    min_health_score: float = 0.0
    version: Optional[str] = None


class ServiceDiscovery:
    """
    Service Discovery system for managing microservice registration and discovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategy = DiscoveryStrategy(self.config.get('strategy', 'in_memory'))
        self.services: Dict[str, ServiceInstance] = {}
        self.service_watchers: Dict[str, List[Callable]] = {}
        self.discovery_cache: Dict[str, Any] = {}
        
        # Configuration
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)
        self.cleanup_interval = self.config.get('cleanup_interval', 60)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
        # Integration with external discovery systems
        self._external_clients: Dict[str, Any] = {}
        
    async def start(self):
        """Start the service discovery system"""
        if self._running:
            return
            
        self._running = True
        logger.info(f"Starting Service Discovery with strategy: {self.strategy.value}")
        
        # Initialize external clients if needed
        await self._initialize_external_clients()
        
        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        sync_task = asyncio.create_task(self._sync_loop())
        
        self._tasks.update([heartbeat_task, cleanup_task, sync_task])
        
        # Register core services
        await self._register_core_services()
        
        logger.info("Service Discovery started successfully")
    
    async def stop(self):
        """Stop the service discovery system"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping Service Discovery")
        
        # Deregister all services
        for service in list(self.services.values()):
            await self.deregister_service(service.service_id)
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("Service Discovery stopped")
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance"""
        try:
            # Validate service instance
            if not service.service_name or not service.host or not service.port:
                raise ValueError("Service name, host, and port are required")
            
            # Generate service_id if not provided
            if not service.service_id:
                service.service_id = self._generate_service_id(service)
            
            # Store service instance
            self.services[service.service_id] = service
            service.status = ServiceStatus.HEALTHY
            service.last_heartbeat = datetime.now()
            
            # Register with external discovery system if configured
            await self._register_with_external_system(service)
            
            # Notify watchers
            await self._notify_service_watchers(service.service_name, "registered", service)
            
            # Clear relevant cache entries
            self._invalidate_cache(service.service_name)
            
            logger.info(f"Registered service: {service.service_name} ({service.service_id}) at {service.address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_name}: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        try:
            service = self.services.get(service_id)
            if not service:
                logger.warning(f"Service {service_id} not found for deregistration")
                return False
            
            service.status = ServiceStatus.STOPPED
            
            # Deregister from external discovery system
            await self._deregister_from_external_system(service)
            
            # Notify watchers
            await self._notify_service_watchers(service.service_name, "deregistered", service)
            
            # Remove from local registry
            del self.services[service_id]
            
            # Clear cache
            self._invalidate_cache(service.service_name)
            
            logger.info(f"Deregistered service: {service.service_name} ({service_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
    
    async def discover_services(self, query: ServiceQuery) -> List[ServiceInstance]:
        """Discover services matching the query criteria"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Query local registry
            matching_services = []
            
            for service in self.services.values():
                if self._matches_query(service, query):
                    matching_services.append(service)
            
            # Query external discovery system if configured
            external_services = await self._discover_from_external_system(query)
            matching_services.extend(external_services)
            
            # Sort by health score (descending)
            matching_services.sort(key=lambda x: x.health_score, reverse=True)
            
            # Cache the results
            self._set_cache(cache_key, matching_services)
            
            return matching_services
            
        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
            return []
    
    async def get_service(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a specific service"""
        query = ServiceQuery(
            service_name=service_name,
            status=ServiceStatus.HEALTHY,
            min_health_score=0.5
        )
        return await self.discover_services(query)
    
    async def get_healthy_service(self, service_name: str) -> Optional[ServiceInstance]:
        """Get a single healthy instance of a service (highest health score)"""
        services = await self.get_service(service_name)
        return services[0] if services else None
    
    def watch_service(self, service_name: str, callback: Callable):
        """Watch for changes in service instances"""
        if service_name not in self.service_watchers:
            self.service_watchers[service_name] = []
        
        self.service_watchers[service_name].append(callback)
        logger.info(f"Added watcher for service: {service_name}")
    
    def unwatch_service(self, service_name: str, callback: Callable):
        """Remove service watcher"""
        if service_name in self.service_watchers:
            try:
                self.service_watchers[service_name].remove(callback)
                logger.info(f"Removed watcher for service: {service_name}")
            except ValueError:
                logger.warning(f"Watcher not found for service: {service_name}")
    
    async def heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat"""
        service = self.services.get(service_id)
        if service:
            service.last_heartbeat = datetime.now()
            service.success_count += 1
            
            # Update status based on health checks
            if service.status == ServiceStatus.DEGRADED:
                # Promote back to healthy if enough successful heartbeats
                if service.success_count > service.failure_count * 2:
                    service.status = ServiceStatus.HEALTHY
                    await self._notify_service_watchers(service.service_name, "healthy", service)
            
            return True
        
        logger.warning(f"Heartbeat received for unknown service: {service_id}")
        return False
    
    async def mark_unhealthy(self, service_id: str, reason: str = ""):
        """Mark a service as unhealthy"""
        service = self.services.get(service_id)
        if service:
            service.failure_count += 1
            old_status = service.status
            
            # Determine new status based on failure count
            if service.failure_count >= 5:
                service.status = ServiceStatus.UNHEALTHY
            elif service.failure_count >= 2:
                service.status = ServiceStatus.DEGRADED
            
            if old_status != service.status:
                await self._notify_service_watchers(service.service_name, "status_changed", service)
            
            logger.warning(f"Marked service {service_id} as {service.status.value}: {reason}")
    
    def _generate_service_id(self, service: ServiceInstance) -> str:
        """Generate unique service ID"""
        content = f"{service.service_name}:{service.host}:{service.port}:{service.version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _matches_query(self, service: ServiceInstance, query: ServiceQuery) -> bool:
        """Check if service matches query criteria"""
        if query.service_name and service.service_name != query.service_name:
            return False
        
        if query.environment and service.environment != query.environment:
            return False
        
        if query.datacenter and service.datacenter != query.datacenter:
            return False
        
        if query.status and service.status != query.status:
            return False
        
        if query.version and service.version != query.version:
            return False
        
        if query.min_health_score and service.health_score < query.min_health_score:
            return False
        
        if query.tags:
            if not all(tag in service.tags for tag in query.tags):
                return False
        
        return True
    
    async def _notify_service_watchers(self, service_name: str, event: str, service: ServiceInstance):
        """Notify all watchers of service changes"""
        watchers = self.service_watchers.get(service_name, [])
        
        for watcher in watchers:
            try:
                await watcher(event, service)
            except Exception as e:
                logger.error(f"Service watcher failed: {e}")
    
    async def _heartbeat_loop(self):
        """Background task for monitoring service heartbeats"""
        while self._running:
            try:
                current_time = datetime.now()
                stale_threshold = timedelta(seconds=self.heartbeat_interval * 3)
                
                for service in list(self.services.values()):
                    if (service.last_heartbeat and 
                        current_time - service.last_heartbeat > stale_threshold):
                        
                        await self.mark_unhealthy(service.service_id, "stale heartbeat")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Background task for cleaning up stale services"""
        while self._running:
            try:
                current_time = datetime.now()
                cleanup_threshold = timedelta(seconds=self.cleanup_interval * 10)
                
                services_to_remove = []
                
                for service_id, service in self.services.items():
                    # Remove services that have been unhealthy for too long
                    if (service.status == ServiceStatus.UNHEALTHY and 
                        service.last_heartbeat and
                        current_time - service.last_heartbeat > cleanup_threshold):
                        
                        services_to_remove.append(service_id)
                
                for service_id in services_to_remove:
                    await self.deregister_service(service_id)
                    logger.info(f"Cleaned up stale service: {service_id}")
                
                # Clean up discovery cache
                self._cleanup_cache()
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def _sync_loop(self):
        """Background task for syncing with external discovery systems"""
        while self._running:
            try:
                await self._sync_with_external_systems()
                await asyncio.sleep(60)  # Sync every minute
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(30)
    
    async def _initialize_external_clients(self):
        """Initialize external discovery system clients"""
        if self.strategy == DiscoveryStrategy.CONSUL:
            # Initialize Consul client
            pass
        elif self.strategy == DiscoveryStrategy.ETCD:
            # Initialize etcd client
            pass
        elif self.strategy == DiscoveryStrategy.KUBERNETES:
            # Initialize Kubernetes client
            pass
    
    async def _register_with_external_system(self, service: ServiceInstance):
        """Register service with external discovery system"""
        # Implementation depends on the discovery strategy
        pass
    
    async def _deregister_from_external_system(self, service: ServiceInstance):
        """Deregister service from external discovery system"""
        # Implementation depends on the discovery strategy
        pass
    
    async def _discover_from_external_system(self, query: ServiceQuery) -> List[ServiceInstance]:
        """Discover services from external discovery system"""
        # Implementation depends on the discovery strategy
        return []
    
    async def _sync_with_external_systems(self):
        """Sync with external discovery systems"""
        # Implementation depends on the discovery strategy
        pass
    
    def _get_cache_key(self, query: ServiceQuery) -> str:
        """Generate cache key for service query"""
        key_components = [
            query.service_name or "all",
            query.environment or "any",
            query.datacenter or "any",
            ",".join(sorted(query.tags)),
            query.status.value if query.status else "any",
            str(query.min_health_score),
            query.version or "any"
        ]
        return hashlib.sha256(":".join(key_components).encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[List[ServiceInstance]]:
        """Get results from discovery cache"""
        cache_entry = self.discovery_cache.get(key)
        if cache_entry:
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["data"]
            else:
                # Cache expired
                del self.discovery_cache[key]
        return None
    
    def _set_cache(self, key: str, data: List[ServiceInstance]):
        """Set results in discovery cache"""
        self.discovery_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def _invalidate_cache(self, service_name: str):
        """Invalidate cache entries related to a service"""
        keys_to_remove = []
        for key, entry in self.discovery_cache.items():
            if service_name in key or key.startswith("all:"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.discovery_cache[key]
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.discovery_cache.items()
            if current_time - entry["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.discovery_cache[key]
    
    async def _register_core_services(self):
        """Register core system services for discovery"""
        core_services = [
            ("document-service", 8001),
            ("analysis-service", 8002),
            ("ml-service", 8003),
            ("legal-service", 8004),
            ("blockchain-service", 8005),
            ("user-service", 8006),
            ("payment-service", 8007),
            ("notification-service", 8008)
        ]
        
        for service_name, port in core_services:
            service = ServiceInstance(
                service_id=f"{service_name}-default",
                service_name=service_name,
                host="localhost",
                port=port,
                tags=["core", "api"]
            )
            await self.register_service(service)
    
    def get_discovery_status(self) -> Dict[str, Any]:
        """Get comprehensive discovery status"""
        service_counts = {}
        status_counts = {}
        
        for service in self.services.values():
            # Count by service name
            service_counts[service.service_name] = service_counts.get(service.service_name, 0) + 1
            
            # Count by status
            status_counts[service.status.value] = status_counts.get(service.status.value, 0) + 1
        
        return {
            "total_services": len(self.services),
            "service_counts": service_counts,
            "status_counts": status_counts,
            "cache_size": len(self.discovery_cache),
            "watchers": {name: len(watchers) for name, watchers in self.service_watchers.items()},
            "strategy": self.strategy.value
        }