"""
Orchestration Layer for Arbitration Detection Platform

This module provides comprehensive integration and orchestration capabilities
for all system components including:

- Service mesh management
- Message queue architecture  
- API gateway routing
- Service discovery
- Event-driven architecture
- Saga patterns for distributed transactions
- System monitoring
- Deployment automation
"""

from .service_mesh import ServiceMesh, ServiceRegistry
from .message_queue import EventBus, MessageQueue
from .api_gateway import APIGateway, RouteManager
from .service_discovery import ServiceDiscovery, HealthChecker
from .event_bus import EventOrchestrator, EventStore
from .saga_patterns import SagaOrchestrator, TransactionManager
from .monitoring import SystemMonitor, MetricsCollector
from .deployment import DeploymentManager, FeatureFlags

__all__ = [
    'ServiceMesh',
    'ServiceRegistry', 
    'EventBus',
    'MessageQueue',
    'APIGateway',
    'RouteManager',
    'ServiceDiscovery',
    'HealthChecker',
    'EventOrchestrator',
    'EventStore',
    'SagaOrchestrator',
    'TransactionManager',
    'SystemMonitor',
    'MetricsCollector',
    'DeploymentManager',
    'FeatureFlags'
]

# Version and metadata
__version__ = '1.0.0'
__author__ = 'System Architecture Team'
__description__ = 'Comprehensive orchestration layer for microservices architecture'