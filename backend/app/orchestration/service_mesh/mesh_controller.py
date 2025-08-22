"""
Service Mesh Controller

Manages service-to-service communication, traffic routing,
and service discovery for all microservices.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class TrafficPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    STICKY_SESSION = "sticky_session"


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint in the mesh"""
    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    health_check_path: str = "/health"
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def health_score(self) -> float:
        """Calculate health score based on various metrics"""
        if self.request_count == 0:
            return 1.0
            
        error_rate = self.error_count / self.request_count
        response_time_score = max(0, 1 - (self.average_response_time / 1000))  # 1s baseline
        
        return max(0, 1 - error_rate) * response_time_score


@dataclass
class ServiceConfig:
    """Service configuration in the mesh"""
    name: str
    version: str
    namespace: str = "default"
    traffic_policy: TrafficPolicy = TrafficPolicy.ROUND_ROBIN
    circuit_breaker_enabled: bool = True
    rate_limit_enabled: bool = True
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout_policy: Dict[str, Any] = field(default_factory=dict)
    security_policy: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.retry_policy:
            self.retry_policy = {
                "max_retries": 3,
                "retry_timeout": 1.0,
                "exponential_backoff": True
            }
        
        if not self.timeout_policy:
            self.timeout_policy = {
                "connection_timeout": 5.0,
                "request_timeout": 30.0,
                "idle_timeout": 60.0
            }


class ServiceMesh:
    """
    Central service mesh controller that manages all inter-service communication
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.services: Dict[str, ServiceConfig] = {}
        self.endpoints: Dict[str, List[ServiceEndpoint]] = {}
        self.route_table: Dict[str, str] = {}
        self.traffic_policies: Dict[str, TrafficPolicy] = {}
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
        # Integration points for new features
        self.voice_service_endpoints = []
        self.document_comparison_endpoints = []
        self.whitelabel_endpoints = []
        self.compliance_endpoints = []
        self.visualization_endpoints = []
        
    async def start(self):
        """Start the service mesh"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting Service Mesh Controller")
        
        # Start background tasks
        health_check_task = asyncio.create_task(self._health_check_loop())
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        self._tasks.update([health_check_task, metrics_task])
        
        # Register core services
        await self._register_core_services()
        
        # Register new feature services
        await self._register_feature_services()
        
        logger.info("Service Mesh Controller started successfully")
    
    async def stop(self):
        """Stop the service mesh"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping Service Mesh Controller")
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("Service Mesh Controller stopped")
    
    async def register_service(self, service_config: ServiceConfig) -> bool:
        """Register a new service in the mesh"""
        try:
            service_key = f"{service_config.namespace}.{service_config.name}"
            self.services[service_key] = service_config
            
            if service_key not in self.endpoints:
                self.endpoints[service_key] = []
            
            logger.info(f"Registered service: {service_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to register service {service_config.name}: {e}")
            return False
    
    async def register_endpoint(self, endpoint: ServiceEndpoint) -> bool:
        """Register a service endpoint"""
        try:
            service_key = f"default.{endpoint.service_name}"
            
            if service_key not in self.endpoints:
                self.endpoints[service_key] = []
            
            # Check if endpoint already exists
            existing = next(
                (ep for ep in self.endpoints[service_key] 
                 if ep.instance_id == endpoint.instance_id), 
                None
            )
            
            if existing:
                # Update existing endpoint
                existing.__dict__.update(endpoint.__dict__)
                logger.info(f"Updated endpoint: {endpoint.instance_id}")
            else:
                # Add new endpoint
                self.endpoints[service_key].append(endpoint)
                logger.info(f"Added endpoint: {endpoint.instance_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to register endpoint {endpoint.instance_id}: {e}")
            return False
    
    async def discover_service(self, service_name: str, namespace: str = "default") -> List[ServiceEndpoint]:
        """Discover healthy endpoints for a service"""
        service_key = f"{namespace}.{service_name}"
        endpoints = self.endpoints.get(service_key, [])
        
        # Filter healthy endpoints
        healthy_endpoints = [
            ep for ep in endpoints 
            if ep.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
        ]
        
        # Sort by health score
        healthy_endpoints.sort(key=lambda x: x.health_score, reverse=True)
        
        return healthy_endpoints
    
    async def route_request(
        self, 
        service_name: str, 
        request_data: Dict[str, Any],
        routing_key: Optional[str] = None
    ) -> Optional[ServiceEndpoint]:
        """Route request to appropriate service endpoint"""
        endpoints = await self.discover_service(service_name)
        
        if not endpoints:
            logger.warning(f"No healthy endpoints found for service: {service_name}")
            return None
        
        # Apply traffic policy
        service_config = self.services.get(f"default.{service_name}")
        if not service_config:
            return endpoints[0]  # Default to first endpoint
        
        return await self._apply_traffic_policy(
            endpoints, 
            service_config.traffic_policy,
            routing_key
        )
    
    async def _apply_traffic_policy(
        self, 
        endpoints: List[ServiceEndpoint], 
        policy: TrafficPolicy,
        routing_key: Optional[str] = None
    ) -> ServiceEndpoint:
        """Apply traffic routing policy"""
        if policy == TrafficPolicy.ROUND_ROBIN:
            return self._round_robin_selection(endpoints)
        elif policy == TrafficPolicy.LEAST_CONNECTIONS:
            return self._least_connections_selection(endpoints)
        elif policy == TrafficPolicy.WEIGHTED:
            return self._weighted_selection(endpoints)
        elif policy == TrafficPolicy.STICKY_SESSION and routing_key:
            return self._sticky_session_selection(endpoints, routing_key)
        else:
            return endpoints[0]
    
    def _round_robin_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Simple round-robin selection"""
        if not hasattr(self, '_rr_counter'):
            self._rr_counter = 0
        
        endpoint = endpoints[self._rr_counter % len(endpoints)]
        self._rr_counter += 1
        return endpoint
    
    def _least_connections_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with least connections (using request count as proxy)"""
        return min(endpoints, key=lambda x: x.request_count)
    
    def _weighted_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted selection based on health score"""
        import random
        
        weights = [ep.health_score for ep in endpoints]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return endpoints[0]
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return endpoints[i]
        
        return endpoints[-1]
    
    def _sticky_session_selection(self, endpoints: List[ServiceEndpoint], routing_key: str) -> ServiceEndpoint:
        """Consistent hash-based selection for sticky sessions"""
        hash_value = int(hashlib.md5(routing_key.encode()).hexdigest(), 16)
        index = hash_value % len(endpoints)
        return endpoints[index]
    
    async def _health_check_loop(self):
        """Background loop for health checking all endpoints"""
        import aiohttp
        
        while self._running:
            try:
                tasks = []
                
                for service_key, endpoints in self.endpoints.items():
                    for endpoint in endpoints:
                        task = self._check_endpoint_health(endpoint)
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_endpoint_health(self, endpoint: ServiceEndpoint):
        """Check health of a single endpoint"""
        import aiohttp
        
        try:
            health_url = f"{endpoint.url}{endpoint.health_check_path}"
            
            async with aiohttp.ClientSession() as session:
                start_time = asyncio.get_event_loop().time()
                async with session.get(health_url, timeout=5) as response:
                    end_time = asyncio.get_event_loop().time()
                    response_time = (end_time - start_time) * 1000
                    
                    if response.status == 200:
                        endpoint.status = ServiceStatus.HEALTHY
                    elif 200 <= response.status < 300:
                        endpoint.status = ServiceStatus.DEGRADED
                    else:
                        endpoint.status = ServiceStatus.UNHEALTHY
                    
                    # Update metrics
                    endpoint.last_health_check = datetime.now()
                    endpoint.average_response_time = (
                        endpoint.average_response_time * 0.9 + response_time * 0.1
                    )
                    
        except Exception as e:
            endpoint.status = ServiceStatus.UNHEALTHY
            endpoint.last_health_check = datetime.now()
            logger.debug(f"Health check failed for {endpoint.instance_id}: {e}")
    
    async def _metrics_collection_loop(self):
        """Background loop for collecting and aggregating metrics"""
        while self._running:
            try:
                await self._collect_service_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_service_metrics(self):
        """Collect metrics from all services"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "total_endpoints": 0,
            "healthy_endpoints": 0
        }
        
        for service_key, endpoints in self.endpoints.items():
            service_metrics = {
                "endpoint_count": len(endpoints),
                "healthy_count": len([ep for ep in endpoints if ep.status == ServiceStatus.HEALTHY]),
                "degraded_count": len([ep for ep in endpoints if ep.status == ServiceStatus.DEGRADED]),
                "unhealthy_count": len([ep for ep in endpoints if ep.status == ServiceStatus.UNHEALTHY]),
                "total_requests": sum(ep.request_count for ep in endpoints),
                "total_errors": sum(ep.error_count for ep in endpoints),
                "average_response_time": sum(ep.average_response_time for ep in endpoints) / len(endpoints) if endpoints else 0
            }
            
            metrics["services"][service_key] = service_metrics
            metrics["total_endpoints"] += len(endpoints)
            metrics["healthy_endpoints"] += service_metrics["healthy_count"]
        
        # Store metrics (could be sent to monitoring system)
        logger.debug(f"Collected service mesh metrics: {json.dumps(metrics, indent=2)}")
    
    async def _register_core_services(self):
        """Register core system services"""
        core_services = [
            ServiceConfig(name="document-service", version="1.0.0"),
            ServiceConfig(name="analysis-service", version="1.0.0"),
            ServiceConfig(name="ml-service", version="1.0.0"),
            ServiceConfig(name="legal-service", version="1.0.0"),
            ServiceConfig(name="blockchain-service", version="1.0.0"),
            ServiceConfig(name="user-service", version="1.0.0"),
            ServiceConfig(name="payment-service", version="1.0.0"),
            ServiceConfig(name="notification-service", version="1.0.0")
        ]
        
        for service in core_services:
            await self.register_service(service)
    
    async def _register_feature_services(self):
        """Register new feature services"""
        feature_services = [
            ServiceConfig(name="voice-interface", version="1.0.0", 
                         traffic_policy=TrafficPolicy.LEAST_CONNECTIONS),
            ServiceConfig(name="document-comparison", version="1.0.0",
                         traffic_policy=TrafficPolicy.WEIGHTED),
            ServiceConfig(name="whitelabel-service", version="1.0.0",
                         traffic_policy=TrafficPolicy.STICKY_SESSION),
            ServiceConfig(name="compliance-automation", version="1.0.0"),
            ServiceConfig(name="visualization-engine", version="1.0.0",
                         traffic_policy=TrafficPolicy.WEIGHTED)
        ]
        
        for service in feature_services:
            await self.register_service(service)
    
    def get_service_topology(self) -> Dict[str, Any]:
        """Get current service topology"""
        topology = {
            "services": {},
            "dependencies": {},
            "health_summary": {
                "total_services": len(self.services),
                "total_endpoints": sum(len(endpoints) for endpoints in self.endpoints.values()),
                "healthy_endpoints": 0,
                "degraded_endpoints": 0,
                "unhealthy_endpoints": 0
            }
        }
        
        for service_key, config in self.services.items():
            endpoints = self.endpoints.get(service_key, [])
            
            topology["services"][service_key] = {
                "config": {
                    "name": config.name,
                    "version": config.version,
                    "namespace": config.namespace,
                    "traffic_policy": config.traffic_policy.value
                },
                "endpoints": [
                    {
                        "instance_id": ep.instance_id,
                        "url": ep.url,
                        "status": ep.status.value,
                        "health_score": ep.health_score,
                        "request_count": ep.request_count,
                        "error_count": ep.error_count
                    }
                    for ep in endpoints
                ]
            }
            
            # Update health summary
            for ep in endpoints:
                if ep.status == ServiceStatus.HEALTHY:
                    topology["health_summary"]["healthy_endpoints"] += 1
                elif ep.status == ServiceStatus.DEGRADED:
                    topology["health_summary"]["degraded_endpoints"] += 1
                else:
                    topology["health_summary"]["unhealthy_endpoints"] += 1
        
        return topology
    
    async def update_traffic_policy(self, service_name: str, policy: TrafficPolicy) -> bool:
        """Update traffic policy for a service"""
        try:
            service_key = f"default.{service_name}"
            if service_key in self.services:
                self.services[service_key].traffic_policy = policy
                logger.info(f"Updated traffic policy for {service_name} to {policy.value}")
                return True
            else:
                logger.warning(f"Service not found: {service_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to update traffic policy for {service_name}: {e}")
            return False