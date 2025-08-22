"""
Main Orchestration Controller

Central controller that integrates all orchestration components including
service mesh, API gateway, event bus, saga patterns, monitoring, and
deployment management for the complete microservices ecosystem.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import signal
import sys

# Import orchestration components
from .service_mesh.mesh_controller import ServiceMesh, ServiceEndpoint, ServiceConfig
from .message_queue.event_bus import EventBus, Event, EventPriority
from .api_gateway.gateway import APIGateway, RouteRule, RouteMethod, LoadBalanceStrategy
from .service_discovery.discovery import ServiceDiscovery, ServiceInstance
from .saga_patterns.orchestrator import SagaOrchestrator
from .monitoring.system_monitor import SystemMonitor, HealthCheck

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration system"""
    environment: str = "production"
    enable_service_mesh: bool = True
    enable_api_gateway: bool = True
    enable_event_bus: bool = True
    enable_saga_orchestrator: bool = True
    enable_service_discovery: bool = True
    enable_monitoring: bool = True
    
    # Component-specific configurations
    service_mesh_config: Dict[str, Any] = None
    api_gateway_config: Dict[str, Any] = None
    event_bus_config: Dict[str, Any] = None
    saga_config: Dict[str, Any] = None
    discovery_config: Dict[str, Any] = None
    monitoring_config: Dict[str, Any] = None


class MainOrchestrator:
    """
    Main orchestration controller that manages the entire microservices ecosystem
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        
        # Core orchestration components
        self.service_mesh: Optional[ServiceMesh] = None
        self.api_gateway: Optional[APIGateway] = None
        self.event_bus: Optional[EventBus] = None
        self.saga_orchestrator: Optional[SagaOrchestrator] = None
        self.service_discovery: Optional[ServiceDiscovery] = None
        self.system_monitor: Optional[SystemMonitor] = None
        
        # State management
        self._running = False
        self._startup_complete = False
        self._shutdown_gracefully = False
        
        # Component initialization order (important for dependencies)
        self._initialization_order = [
            'service_discovery',
            'service_mesh', 
            'event_bus',
            'saga_orchestrator',
            'api_gateway',
            'system_monitor'
        ]
    
    async def start(self):
        """Start the complete orchestration system"""
        if self._running:
            logger.warning("Orchestration system is already running")
            return
        
        logger.info("Starting Main Orchestration System")
        self._running = True
        
        try:
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Wire components together
            await self._wire_components()
            
            # Register signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start all components
            await self._start_components()
            
            # Register services and routes
            await self._setup_system_integration()
            
            # Perform initial system health check
            await self._initial_health_check()
            
            self._startup_complete = True
            logger.info("Main Orchestration System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start orchestration system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the orchestration system gracefully"""
        if not self._running:
            return
        
        self._shutdown_gracefully = True
        logger.info("Stopping Main Orchestration System")
        
        try:
            # Stop components in reverse order
            await self._stop_components()
            
            self._running = False
            logger.info("Main Orchestration System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during orchestration system shutdown: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all orchestration components"""
        logger.info("Initializing orchestration components")
        
        # Service Discovery
        if self.config.enable_service_discovery:
            self.service_discovery = ServiceDiscovery(
                config=self.config.discovery_config or {}
            )
            logger.info("Initialized Service Discovery")
        
        # Service Mesh
        if self.config.enable_service_mesh:
            self.service_mesh = ServiceMesh(
                config=self.config.service_mesh_config or {}
            )
            logger.info("Initialized Service Mesh")
        
        # Event Bus
        if self.config.enable_event_bus:
            self.event_bus = EventBus(
                config=self.config.event_bus_config or {}
            )
            logger.info("Initialized Event Bus")
        
        # Saga Orchestrator
        if self.config.enable_saga_orchestrator:
            self.saga_orchestrator = SagaOrchestrator(
                config=self.config.saga_config or {}
            )
            logger.info("Initialized Saga Orchestrator")
        
        # API Gateway
        if self.config.enable_api_gateway:
            self.api_gateway = APIGateway(
                config=self.config.api_gateway_config or {}
            )
            logger.info("Initialized API Gateway")
        
        # System Monitor
        if self.config.enable_monitoring:
            self.system_monitor = SystemMonitor(
                config=self.config.monitoring_config or {}
            )
            logger.info("Initialized System Monitor")
    
    async def _wire_components(self):
        """Wire components together with cross-references"""
        logger.info("Wiring orchestration components")
        
        # Inject dependencies
        if self.saga_orchestrator:
            self.saga_orchestrator.event_bus = self.event_bus
            self.saga_orchestrator.service_discovery = self.service_discovery
        
        if self.system_monitor:
            self.system_monitor.service_mesh = self.service_mesh
            self.system_monitor.api_gateway = self.api_gateway
            self.system_monitor.event_bus = self.event_bus
            self.system_monitor.saga_orchestrator = self.saga_orchestrator
            self.system_monitor.service_discovery = self.service_discovery
        
        logger.info("Component wiring completed")
    
    async def _start_components(self):
        """Start all components in the correct order"""
        logger.info("Starting orchestration components")
        
        for component_name in self._initialization_order:
            component = getattr(self, component_name, None)
            if component:
                await component.start()
                logger.info(f"Started {component_name}")
        
        logger.info("All orchestration components started")
    
    async def _stop_components(self):
        """Stop all components in reverse order"""
        logger.info("Stopping orchestration components")
        
        # Stop in reverse order to handle dependencies
        for component_name in reversed(self._initialization_order):
            component = getattr(self, component_name, None)
            if component:
                try:
                    await component.stop()
                    logger.info(f"Stopped {component_name}")
                except Exception as e:
                    logger.error(f"Error stopping {component_name}: {e}")
        
        logger.info("All orchestration components stopped")
    
    async def _setup_system_integration(self):
        """Setup integration between all new and existing system components"""
        logger.info("Setting up system integration")
        
        # Register all services with service discovery and mesh
        await self._register_all_services()
        
        # Setup API Gateway routes for all services
        await self._setup_api_routes()
        
        # Configure event-driven workflows for new features
        await self._setup_event_workflows()
        
        # Setup monitoring for all components
        await self._setup_comprehensive_monitoring()
        
        logger.info("System integration setup completed")
    
    async def _register_all_services(self):
        """Register all services with discovery and mesh"""
        # Core services
        core_services = [
            ("document-service", 8001, ["core", "api", "documents"]),
            ("analysis-service", 8002, ["core", "api", "ml"]),
            ("ml-service", 8003, ["core", "ml", "ai"]),
            ("legal-service", 8004, ["core", "legal", "compliance"]),
            ("blockchain-service", 8005, ["core", "blockchain", "audit"]),
            ("user-service", 8006, ["core", "api", "auth"]),
            ("payment-service", 8007, ["core", "api", "billing"]),
            ("notification-service", 8008, ["core", "api", "messaging"]),
        ]
        
        # New feature services
        feature_services = [
            ("voice-interface", 8009, ["feature", "voice", "accessibility"]),
            ("document-comparison", 8010, ["feature", "documents", "analysis"]),
            ("whitelabel-service", 8011, ["feature", "tenancy", "customization"]),
            ("compliance-automation", 8012, ["feature", "compliance", "automation"]),
            ("visualization-engine", 8013, ["feature", "analytics", "reporting"]),
        ]
        
        all_services = core_services + feature_services
        
        for service_name, port, tags in all_services:
            # Register with service discovery
            if self.service_discovery:
                service_instance = ServiceInstance(
                    service_id=f"{service_name}-primary",
                    service_name=service_name,
                    host="localhost",  # In production, this would be dynamic
                    port=port,
                    tags=tags,
                    metadata={
                        "version": "1.0.0",
                        "environment": self.config.environment,
                        "features": tags
                    }
                )
                await self.service_discovery.register_service(service_instance)
            
            # Register with service mesh
            if self.service_mesh:
                service_config = ServiceConfig(
                    name=service_name,
                    version="1.0.0",
                    namespace="default"
                )
                await self.service_mesh.register_service(service_config)
                
                # Register endpoint
                endpoint = ServiceEndpoint(
                    service_name=service_name,
                    instance_id=f"{service_name}-primary",
                    host="localhost",
                    port=port,
                    metadata={"tags": tags}
                )
                await self.service_mesh.register_endpoint(endpoint)
        
        logger.info(f"Registered {len(all_services)} services")
    
    async def _setup_api_routes(self):
        """Setup API Gateway routes for all services"""
        if not self.api_gateway:
            return
        
        # Core API routes
        core_routes = [
            # Document routes
            RouteRule(
                path="/api/v1/documents",
                methods=[RouteMethod.GET, RouteMethod.POST, RouteMethod.PUT, RouteMethod.DELETE],
                service_name="document-service",
                rate_limit={"tokens": 100, "refill_rate": 1.0}
            ),
            
            # Analysis routes
            RouteRule(
                path="/api/v1/analysis",
                methods=[RouteMethod.POST, RouteMethod.GET],
                service_name="analysis-service",
                rate_limit={"tokens": 50, "refill_rate": 0.5}
            ),
            
            # User routes
            RouteRule(
                path="/api/v1/users",
                methods=[RouteMethod.GET, RouteMethod.POST, RouteMethod.PUT, RouteMethod.DELETE],
                service_name="user-service",
                rate_limit={"tokens": 200, "refill_rate": 2.0}
            ),
            
            # Payment routes
            RouteRule(
                path="/api/v1/payments",
                methods=[RouteMethod.POST, RouteMethod.GET],
                service_name="payment-service",
                rate_limit={"tokens": 20, "refill_rate": 0.2}
            ),
        ]
        
        # Feature routes (already registered in gateway.py, but ensuring consistency)
        feature_routes = [
            # Voice interface
            RouteRule(
                path="/api/v1/voice",
                methods=[RouteMethod.POST, RouteMethod.GET],
                service_name="voice-interface",
                rate_limit={"tokens": 10, "refill_rate": 0.1},
                timeout=45.0
            ),
            
            # Document comparison
            RouteRule(
                path="/api/v1/documents/compare",
                methods=[RouteMethod.POST],
                service_name="document-comparison",
                rate_limit={"tokens": 5, "refill_rate": 0.05},
                timeout=60.0
            ),
            
            # White-label
            RouteRule(
                path="/api/v1/tenants",
                methods=[RouteMethod.GET, RouteMethod.POST, RouteMethod.PUT, RouteMethod.DELETE],
                service_name="whitelabel-service",
                load_balance_strategy=LoadBalanceStrategy.WEIGHTED
            ),
            
            # Compliance automation
            RouteRule(
                path="/api/v1/compliance",
                methods=[RouteMethod.POST, RouteMethod.GET],
                service_name="compliance-automation"
            ),
            
            # Visualization
            RouteRule(
                path="/api/v1/visualizations",
                methods=[RouteMethod.POST, RouteMethod.GET],
                service_name="visualization-engine",
                rate_limit={"tokens": 20, "refill_rate": 0.2}
            ),
        ]
        
        all_routes = core_routes + feature_routes
        
        for route in all_routes:
            self.api_gateway.add_route(route)
        
        logger.info(f"Setup {len(all_routes)} API routes")
    
    async def _setup_event_workflows(self):
        """Setup event-driven workflows for integrated system operations"""
        if not self.event_bus:
            return
        
        # Document processing workflow events
        self.event_bus.subscribe(
            "document.uploaded",
            self._handle_document_uploaded,
            "orchestrator"
        )
        
        # User lifecycle events
        self.event_bus.subscribe(
            "user.created",
            self._handle_user_created,
            "orchestrator"
        )
        
        # Compliance workflow events
        self.event_bus.subscribe(
            "compliance.check.required",
            self._handle_compliance_check_required,
            "orchestrator"
        )
        
        # Voice command processing events
        self.event_bus.subscribe(
            "voice.command.received",
            self._handle_voice_command_received,
            "orchestrator"
        )
        
        # Document comparison events
        self.event_bus.subscribe(
            "document.comparison.requested",
            self._handle_document_comparison_requested,
            "orchestrator"
        )
        
        logger.info("Setup event-driven workflows")
    
    async def _setup_comprehensive_monitoring(self):
        """Setup monitoring for all system components"""
        if not self.system_monitor:
            return
        
        # Add health checks for all services
        services_to_monitor = [
            ("document-service", "http://localhost:8001/health"),
            ("analysis-service", "http://localhost:8002/health"),
            ("ml-service", "http://localhost:8003/health"),
            ("legal-service", "http://localhost:8004/health"),
            ("blockchain-service", "http://localhost:8005/health"),
            ("user-service", "http://localhost:8006/health"),
            ("payment-service", "http://localhost:8007/health"),
            ("notification-service", "http://localhost:8008/health"),
            ("voice-interface", "http://localhost:8009/health"),
            ("document-comparison", "http://localhost:8010/health"),
            ("whitelabel-service", "http://localhost:8011/health"),
            ("compliance-automation", "http://localhost:8012/health"),
            ("visualization-engine", "http://localhost:8013/health"),
        ]
        
        for service_name, health_url in services_to_monitor:
            health_check = HealthCheck(
                name=service_name,
                endpoint=health_url,
                interval=30,
                timeout=5
            )
            self.system_monitor.register_health_check(health_check)
        
        logger.info("Setup comprehensive monitoring")
    
    async def _initial_health_check(self):
        """Perform initial health check of all system components"""
        logger.info("Performing initial system health check")
        
        health_issues = []
        
        # Check each component
        if self.service_discovery:
            discovery_status = self.service_discovery.get_discovery_status()
            if discovery_status.get("total_services", 0) == 0:
                health_issues.append("No services registered with service discovery")
        
        if self.service_mesh:
            topology = self.service_mesh.get_service_topology()
            if topology.get("health_summary", {}).get("healthy_endpoints", 0) == 0:
                health_issues.append("No healthy endpoints in service mesh")
        
        if self.api_gateway:
            gateway_status = self.api_gateway.get_gateway_status()
            if gateway_status.get("registered_routes", 0) == 0:
                health_issues.append("No routes registered with API gateway")
        
        if health_issues:
            logger.warning(f"Initial health check found issues: {health_issues}")
            # Publish system health event
            if self.event_bus:
                await self.event_bus.publish(Event(
                    type="system.health.issues_detected",
                    source="main-orchestrator",
                    data={"issues": health_issues},
                    priority=EventPriority.HIGH
                ))
        else:
            logger.info("Initial health check passed - all systems operational")
    
    # Event handlers for integrated workflows
    async def _handle_document_uploaded(self, event: Event):
        """Handle document uploaded event - trigger analysis saga"""
        if self.saga_orchestrator:
            await self.saga_orchestrator.start_saga(
                "document_analysis",
                {
                    "document_id": event.data.get("document_id"),
                    "document_name": event.data.get("document_name"),
                    "user_id": event.data.get("user_id")
                }
            )
        logger.info(f"Triggered document analysis saga for: {event.data.get('document_name')}")
    
    async def _handle_user_created(self, event: Event):
        """Handle user created event - trigger onboarding saga"""
        if self.saga_orchestrator:
            await self.saga_orchestrator.start_saga(
                "user_onboarding",
                {
                    "user_id": event.data.get("user_id"),
                    "email": event.data.get("email"),
                    "organization": event.data.get("organization")
                }
            )
        logger.info(f"Triggered user onboarding saga for: {event.data.get('email')}")
    
    async def _handle_compliance_check_required(self, event: Event):
        """Handle compliance check required event"""
        if self.saga_orchestrator:
            await self.saga_orchestrator.start_saga(
                "compliance_check",
                {
                    "document_id": event.data.get("document_id"),
                    "jurisdiction": event.data.get("jurisdiction"),
                    "requested_by": event.data.get("requested_by")
                }
            )
        logger.info(f"Triggered compliance check saga for document: {event.data.get('document_id')}")
    
    async def _handle_voice_command_received(self, event: Event):
        """Handle voice command received event"""
        command = event.data.get("command", "").lower()
        
        if "analyze document" in command:
            # Trigger document analysis
            await self.event_bus.publish(Event(
                type="document.analysis.requested",
                source="voice-interface",
                data=event.data
            ))
        elif "check compliance" in command:
            # Trigger compliance check
            await self.event_bus.publish(Event(
                type="compliance.check.required",
                source="voice-interface", 
                data=event.data
            ))
        
        logger.info(f"Processed voice command: {command}")
    
    async def _handle_document_comparison_requested(self, event: Event):
        """Handle document comparison request event"""
        document_ids = event.data.get("document_ids", [])
        
        # Trigger comparison analysis
        if self.event_bus:
            await self.event_bus.publish(Event(
                type="analysis.comparison.started",
                source="document-comparison",
                data={
                    "document_ids": document_ids,
                    "comparison_id": event.data.get("comparison_id"),
                    "requested_by": event.data.get("requested_by")
                }
            ))
        
        logger.info(f"Processing document comparison for {len(document_ids)} documents")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration system status"""
        status = {
            "orchestrator": {
                "running": self._running,
                "startup_complete": self._startup_complete,
                "environment": self.config.environment
            },
            "components": {}
        }
        
        # Get status from each component
        if self.service_discovery:
            status["components"]["service_discovery"] = self.service_discovery.get_discovery_status()
        
        if self.service_mesh:
            status["components"]["service_mesh"] = self.service_mesh.get_service_topology()
        
        if self.api_gateway:
            status["components"]["api_gateway"] = self.api_gateway.get_gateway_status()
        
        if self.event_bus:
            status["components"]["event_bus"] = self.event_bus.get_stats()
        
        if self.saga_orchestrator:
            status["components"]["saga_orchestrator"] = self.saga_orchestrator.get_orchestrator_stats()
        
        if self.system_monitor:
            status["components"]["system_monitor"] = self.system_monitor.get_system_status()
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "summary": {
                "total_components": 0,
                "healthy_components": 0,
                "issues": []
            }
        }
        
        component_checks = [
            ("service_discovery", self.service_discovery),
            ("service_mesh", self.service_mesh), 
            ("api_gateway", self.api_gateway),
            ("event_bus", self.event_bus),
            ("saga_orchestrator", self.saga_orchestrator),
            ("system_monitor", self.system_monitor)
        ]
        
        for component_name, component in component_checks:
            if component:
                health_status["components"][component_name] = "healthy"
                health_status["summary"]["healthy_components"] += 1
            else:
                health_status["components"][component_name] = "disabled"
            
            health_status["summary"]["total_components"] += 1
        
        # Determine overall health
        if health_status["summary"]["healthy_components"] < health_status["summary"]["total_components"]:
            if health_status["summary"]["healthy_components"] == 0:
                health_status["status"] = "unhealthy"
            else:
                health_status["status"] = "degraded"
        
        return health_status


# Convenience function to start the orchestration system
async def start_orchestration_system(config: Optional[OrchestrationConfig] = None) -> MainOrchestrator:
    """Start the complete orchestration system with default or provided configuration"""
    if config is None:
        config = OrchestrationConfig()
    
    orchestrator = MainOrchestrator(config)
    await orchestrator.start()
    return orchestrator


# Example usage and testing
async def main():
    """Example main function for testing the orchestration system"""
    # Create configuration
    config = OrchestrationConfig(
        environment="development",
        enable_service_mesh=True,
        enable_api_gateway=True,
        enable_event_bus=True,
        enable_saga_orchestrator=True,
        enable_service_discovery=True,
        enable_monitoring=True
    )
    
    # Start orchestration system
    orchestrator = await start_orchestration_system(config)
    
    # Keep running
    try:
        while orchestrator._running:
            await asyncio.sleep(60)
            
            # Periodic status logging
            status = orchestrator.get_orchestration_status()
            logger.info(f"System Status: {json.dumps(status, indent=2)}")
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())