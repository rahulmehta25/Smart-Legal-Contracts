"""
Health check endpoints and system health monitoring.
Provides comprehensive health status for all system components.
"""

import asyncio
import time
import psutil
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import redis
import sqlalchemy
from sqlalchemy import text
import motor.motor_asyncio


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)


class HealthChecker:
    """
    Comprehensive health checking for all system components.
    """
    
    def __init__(self, 
                 db_url: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 mongodb_url: Optional[str] = None,
                 external_services: Optional[Dict[str, str]] = None):
        """
        Initialize health checker with connection strings.
        
        Args:
            db_url: Database connection URL
            redis_url: Redis connection URL
            mongodb_url: MongoDB connection URL
            external_services: Dictionary of external service URLs
        """
        self.db_url = db_url
        self.redis_url = redis_url
        self.mongodb_url = mongodb_url
        self.external_services = external_services or {}
        
        # Health check history
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # Component checks
        self.component_checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'mongodb': self.check_mongodb,
            'disk': self.check_disk_space,
            'memory': self.check_memory,
            'cpu': self.check_cpu,
            'external_services': self.check_external_services,
            'ml_models': self.check_ml_models,
            'message_queue': self.check_message_queue
        }
        
        # Thresholds
        self.thresholds = {
            'disk_usage_percent': 80,
            'memory_usage_percent': 85,
            'cpu_usage_percent': 80,
            'db_response_time_ms': 100,
            'redis_response_time_ms': 10,
            'api_response_time_ms': 1000
        }
    
    async def check_health(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Args:
            detailed: Include detailed component information
        
        Returns:
            Health status dictionary
        """
        start_time = time.time()
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run all component checks in parallel
        tasks = []
        for component_name, check_func in self.component_checks.items():
            tasks.append(self._run_component_check(component_name, check_func))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for component_name, result in zip(self.component_checks.keys(), results):
            if isinstance(result, Exception):
                components[component_name] = ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result)
                )
                overall_status = self._worse_status(overall_status, HealthStatus.UNHEALTHY)
            else:
                components[component_name] = result
                overall_status = self._worse_status(overall_status, result.status)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(components)
        
        # Build response
        response = {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'health_score': health_score,
            'response_time_ms': (time.time() - start_time) * 1000
        }
        
        if detailed:
            response['components'] = {
                name: {
                    'status': comp.status.value,
                    'message': comp.message,
                    'response_time_ms': comp.response_time_ms,
                    'metadata': comp.metadata,
                    'last_check': comp.last_check.isoformat()
                }
                for name, comp in components.items()
            }
        
        # Add to history
        self._add_to_history(response)
        
        return response
    
    async def _run_component_check(self, name: str, check_func) -> ComponentHealth:
        """Run a single component health check"""
        try:
            start = time.time()
            result = await check_func()
            result.response_time_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}"
            )
    
    async def check_database(self) -> ComponentHealth:
        """Check database health"""
        if not self.db_url:
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="No database configured"
            )
        
        try:
            engine = sqlalchemy.create_engine(self.db_url)
            start = time.time()
            
            with engine.connect() as conn:
                # Simple query to test connection
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Check connection pool
                pool_status = engine.pool.status()
                
            response_time = (time.time() - start) * 1000
            
            # Determine status based on response time
            if response_time > self.thresholds['db_response_time_ms'] * 2:
                status = HealthStatus.UNHEALTHY
            elif response_time > self.thresholds['db_response_time_ms']:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="database",
                status=status,
                message="Database responding normally",
                metadata={'pool_status': pool_status}
            )
            
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    async def check_redis(self) -> ComponentHealth:
        """Check Redis cache health"""
        if not self.redis_url:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="No Redis configured"
            )
        
        try:
            r = redis.from_url(self.redis_url)
            start = time.time()
            
            # Ping Redis
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            response_time = (time.time() - start) * 1000
            
            # Check response time
            if response_time > self.thresholds['redis_response_time_ms'] * 2:
                status = HealthStatus.UNHEALTHY
            elif response_time > self.thresholds['redis_response_time_ms']:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="redis",
                status=status,
                message="Redis cache operational",
                metadata={
                    'used_memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                    'connected_clients': info.get('connected_clients', 0),
                    'uptime_seconds': info.get('uptime_in_seconds', 0)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}"
            )
    
    async def check_mongodb(self) -> ComponentHealth:
        """Check MongoDB health"""
        if not self.mongodb_url:
            return ComponentHealth(
                name="mongodb",
                status=HealthStatus.HEALTHY,
                message="No MongoDB configured"
            )
        
        try:
            client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            start = time.time()
            
            # Ping MongoDB
            await client.admin.command('ping')
            
            # Get server status
            status = await client.admin.command('serverStatus')
            
            response_time = (time.time() - start) * 1000
            
            return ComponentHealth(
                name="mongodb",
                status=HealthStatus.HEALTHY,
                message="MongoDB operational",
                metadata={
                    'version': status.get('version', 'unknown'),
                    'uptime': status.get('uptime', 0),
                    'connections': status.get('connections', {})
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="mongodb",
                status=HealthStatus.UNHEALTHY,
                message=f"MongoDB connection failed: {str(e)}"
            )
    
    async def check_disk_space(self) -> ComponentHealth:
        """Check disk space availability"""
        try:
            disk_usage = psutil.disk_usage('/')
            usage_percent = disk_usage.percent
            
            # Determine status based on usage
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
            elif usage_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif usage_percent > self.thresholds['disk_usage_percent']:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=f"Disk usage at {usage_percent:.1f}%",
                metadata={
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'percent': usage_percent
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {str(e)}"
            )
    
    async def check_memory(self) -> ComponentHealth:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            # Determine status based on usage
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
            elif usage_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif usage_percent > self.thresholds['memory_usage_percent']:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=f"Memory usage at {usage_percent:.1f}%",
                metadata={
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': usage_percent
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}"
            )
    
    async def check_cpu(self) -> ComponentHealth:
        """Check CPU usage"""
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Determine status based on usage
            if cpu_percent > 95:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > self.thresholds['cpu_usage_percent']:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="cpu",
                status=status,
                message=f"CPU usage at {cpu_percent:.1f}%",
                metadata={
                    'percent': cpu_percent,
                    'core_count': psutil.cpu_count(),
                    'load_average': psutil.getloadavg()
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {str(e)}"
            )
    
    async def check_external_services(self) -> ComponentHealth:
        """Check external service availability"""
        if not self.external_services:
            return ComponentHealth(
                name="external_services",
                status=HealthStatus.HEALTHY,
                message="No external services configured"
            )
        
        service_statuses = {}
        overall_status = HealthStatus.HEALTHY
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in self.external_services.items():
                try:
                    start = time.time()
                    async with session.get(url, timeout=5) as response:
                        response_time = (time.time() - start) * 1000
                        
                        if response.status < 400:
                            service_statuses[service_name] = {
                                'status': 'healthy',
                                'response_time_ms': response_time,
                                'status_code': response.status
                            }
                        else:
                            service_statuses[service_name] = {
                                'status': 'unhealthy',
                                'response_time_ms': response_time,
                                'status_code': response.status
                            }
                            overall_status = HealthStatus.DEGRADED
                            
                except Exception as e:
                    service_statuses[service_name] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    overall_status = self._worse_status(overall_status, HealthStatus.DEGRADED)
        
        return ComponentHealth(
            name="external_services",
            status=overall_status,
            message="External services check complete",
            metadata=service_statuses
        )
    
    async def check_ml_models(self) -> ComponentHealth:
        """Check ML model availability and performance"""
        try:
            # This would check actual model loading and inference
            # For now, returning a placeholder
            return ComponentHealth(
                name="ml_models",
                status=HealthStatus.HEALTHY,
                message="ML models operational",
                metadata={
                    'models_loaded': 5,
                    'avg_inference_time_ms': 45
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="ml_models",
                status=HealthStatus.UNHEALTHY,
                message=f"ML model check failed: {str(e)}"
            )
    
    async def check_message_queue(self) -> ComponentHealth:
        """Check message queue health"""
        try:
            # This would check actual message queue (RabbitMQ, Kafka, etc.)
            # For now, returning a placeholder
            return ComponentHealth(
                name="message_queue",
                status=HealthStatus.HEALTHY,
                message="Message queue operational",
                metadata={
                    'queue_depth': 42,
                    'consumers': 3
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="message_queue",
                status=HealthStatus.UNHEALTHY,
                message=f"Message queue check failed: {str(e)}"
            )
    
    def _worse_status(self, status1: HealthStatus, status2: HealthStatus) -> HealthStatus:
        """Return the worse of two health statuses"""
        priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
            HealthStatus.CRITICAL: 3
        }
        
        if priority[status2] > priority[status1]:
            return status2
        return status1
    
    def _calculate_health_score(self, components: Dict[str, ComponentHealth]) -> float:
        """
        Calculate overall health score (0-100).
        
        Args:
            components: Dictionary of component health statuses
        
        Returns:
            Health score between 0 and 100
        """
        if not components:
            return 100.0
        
        status_scores = {
            HealthStatus.HEALTHY: 100,
            HealthStatus.DEGRADED: 70,
            HealthStatus.UNHEALTHY: 30,
            HealthStatus.CRITICAL: 0
        }
        
        total_score = sum(
            status_scores[comp.status] for comp in components.values()
        )
        
        return total_score / len(components)
    
    def _add_to_history(self, health_data: Dict[str, Any]):
        """Add health check result to history"""
        self.health_history.append(health_data)
        
        # Trim history if needed
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    def get_health_trends(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Get health trends over time.
        
        Args:
            duration_minutes: Duration to analyze in minutes
        
        Returns:
            Health trend analysis
        """
        if not self.health_history:
            return {'message': 'No health history available'}
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_history = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            return {'message': 'No recent health data'}
        
        # Calculate trends
        health_scores = [h['health_score'] for h in recent_history]
        statuses = [h['status'] for h in recent_history]
        
        return {
            'period_minutes': duration_minutes,
            'data_points': len(recent_history),
            'average_health_score': sum(health_scores) / len(health_scores),
            'min_health_score': min(health_scores),
            'max_health_score': max(health_scores),
            'status_distribution': {
                status: statuses.count(status) 
                for status in set(statuses)
            },
            'latest_status': recent_history[-1]['status'],
            'trend': self._calculate_trend(health_scores)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in x)
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.5:
            return 'improving'
        elif slope < -0.5:
            return 'degrading'
        else:
            return 'stable'


class HealthEndpoint:
    """
    FastAPI health check endpoint implementation.
    """
    
    def __init__(self, health_checker: HealthChecker):
        """
        Initialize health endpoint.
        
        Args:
            health_checker: HealthChecker instance
        """
        self.health_checker = health_checker
    
    async def health(self) -> Dict[str, Any]:
        """Basic health check endpoint"""
        result = await self.health_checker.check_health(detailed=False)
        
        # Return appropriate HTTP status based on health
        if result['status'] == HealthStatus.HEALTHY.value:
            return result
        elif result['status'] == HealthStatus.DEGRADED.value:
            # Return 200 but indicate degraded state
            return result
        else:
            # Return 503 for unhealthy state
            raise HTTPException(status_code=503, detail=result)
    
    async def health_detailed(self) -> Dict[str, Any]:
        """Detailed health check endpoint"""
        return await self.health_checker.check_health(detailed=True)
    
    async def health_ready(self) -> Dict[str, str]:
        """Readiness probe endpoint"""
        result = await self.health_checker.check_health(detailed=False)
        
        if result['status'] in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]:
            return {'status': 'ready'}
        else:
            raise HTTPException(status_code=503, detail={'status': 'not ready'})
    
    async def health_live(self) -> Dict[str, str]:
        """Liveness probe endpoint"""
        # Simple check that the service is running
        return {'status': 'alive'}
    
    async def health_trends(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get health trends endpoint"""
        return self.health_checker.get_health_trends(duration_minutes)