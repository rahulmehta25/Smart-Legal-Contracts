"""
Blockchain monitoring system for audit trail infrastructure.

Provides comprehensive monitoring of blockchain networks, transactions,
smart contracts, and system health with alerting capabilities.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import websockets
from enum import Enum
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class NetworkStatus(Enum):
    """Network status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class BlockchainMetrics:
    """Blockchain network metrics."""
    network_id: str
    block_height: int
    block_time: float
    transaction_count: int
    pending_transactions: int
    gas_price: float
    network_hashrate: float
    peer_count: int
    sync_status: bool
    timestamp: int

@dataclass
class TransactionMetrics:
    """Transaction monitoring metrics."""
    tx_hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: float
    status: str
    block_number: int
    confirmation_time: float
    timestamp: int

@dataclass
class ContractMetrics:
    """Smart contract metrics."""
    contract_address: str
    function_calls: int
    gas_consumption: int
    events_emitted: int
    error_count: int
    avg_execution_time: float
    last_activity: int

@dataclass
class Alert:
    """System alert structure."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: int
    source: str
    metadata: Dict[str, Any]
    is_resolved: bool = False
    resolved_at: Optional[int] = None

@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_time: float
    error_rate: float
    throughput: float
    timestamp: int

class BlockchainMonitor:
    """
    Comprehensive blockchain monitoring system.
    
    Monitors multiple blockchain networks, tracks performance metrics,
    manages alerts, and provides real-time health monitoring.
    """
    
    def __init__(self,
                 networks: Dict[str, Dict[str, str]],
                 monitoring_interval: int = 30,
                 alert_thresholds: Optional[Dict[str, Any]] = None,
                 enable_websocket: bool = True,
                 enable_email_alerts: bool = False):
        """
        Initialize blockchain monitoring system.
        
        Args:
            networks: Network configurations (name -> config)
            monitoring_interval: Monitoring interval in seconds
            alert_thresholds: Custom alert thresholds
            enable_websocket: Enable WebSocket monitoring
            enable_email_alerts: Enable email alert notifications
        """
        self.networks = networks
        self.monitoring_interval = monitoring_interval
        self.enable_websocket = enable_websocket
        self.enable_email_alerts = enable_email_alerts
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Monitoring data storage
        self.blockchain_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.transaction_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.contract_metrics: Dict[str, ContractMetrics] = {}
        self.performance_metrics: deque = deque(maxlen=1000)
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # WebSocket connections
        self.websocket_connections: Dict[str, Any] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Default alert thresholds
        self.alert_thresholds = {
            'block_time_threshold': 30.0,  # seconds
            'gas_price_threshold': 100.0,  # gwei
            'pending_tx_threshold': 1000,
            'peer_count_threshold': 5,
            'confirmation_time_threshold': 300.0,  # seconds
            'error_rate_threshold': 0.05,  # 5%
            'response_time_threshold': 5.0,  # seconds
            'memory_usage_threshold': 0.8,  # 80%
            'cpu_usage_threshold': 0.8,  # 80%
            'disk_usage_threshold': 0.9,  # 90%
            **alert_thresholds or {}
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start_monitoring(self) -> bool:
        """
        Start blockchain monitoring system.
        
        Returns:
            bool: True if monitoring started successfully
        """
        try:
            self.session = aiohttp.ClientSession()
            self.is_monitoring = True
            
            # Start monitoring tasks for each network
            for network_name in self.networks.keys():
                task = asyncio.create_task(self._monitor_network(network_name))
                self.monitoring_tasks.append(task)
            
            # Start WebSocket monitoring if enabled
            if self.enable_websocket:
                for network_name in self.networks.keys():
                    if 'websocket_url' in self.networks[network_name]:
                        task = asyncio.create_task(self._monitor_websocket(network_name))
                        self.monitoring_tasks.append(task)
            
            # Start performance monitoring
            task = asyncio.create_task(self._monitor_performance())
            self.monitoring_tasks.append(task)
            
            # Start alert processing
            task = asyncio.create_task(self._process_alerts())
            self.monitoring_tasks.append(task)
            
            logger.info("Blockchain monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    async def stop_monitoring(self):
        """Stop blockchain monitoring system."""
        try:
            self.is_monitoring = False
            
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            # Close WebSocket connections
            for ws in self.websocket_connections.values():
                if ws and not ws.closed:
                    await ws.close()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            logger.info("Blockchain monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def _monitor_network(self, network_name: str):
        """Monitor specific blockchain network."""
        while self.is_monitoring:
            try:
                network_config = self.networks[network_name]
                rpc_url = network_config['rpc_url']
                
                # Get blockchain metrics
                metrics = await self._get_blockchain_metrics(network_name, rpc_url)
                if metrics:
                    self.blockchain_metrics[network_name].append(metrics)
                    await self._check_blockchain_alerts(network_name, metrics)
                
                # Get recent transactions
                await self._monitor_transactions(network_name, rpc_url)
                
                # Monitor smart contracts if configured
                if 'contracts' in network_config:
                    await self._monitor_contracts(network_name, network_config['contracts'])
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring network {network_name}: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _get_blockchain_metrics(self, network_name: str, rpc_url: str) -> Optional[BlockchainMetrics]:
        """Get blockchain network metrics via RPC."""
        try:
            # Ethereum/compatible RPC calls
            rpc_calls = [
                {'method': 'eth_blockNumber', 'params': []},
                {'method': 'eth_gasPrice', 'params': []},
                {'method': 'net_peerCount', 'params': []},
                {'method': 'eth_syncing', 'params': []},
                {'method': 'eth_getBlockByNumber', 'params': ['latest', False]}
            ]
            
            results = []
            for call in rpc_calls:
                payload = {
                    'jsonrpc': '2.0',
                    'method': call['method'],
                    'params': call['params'],
                    'id': 1
                }
                
                async with self.session.post(rpc_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        results.append(data.get('result'))
                    else:
                        results.append(None)
            
            if not any(results):
                return None
            
            # Parse results
            block_number = int(results[0], 16) if results[0] else 0
            gas_price = int(results[1], 16) / 10**9 if results[1] else 0  # Convert to Gwei
            peer_count = int(results[2], 16) if results[2] else 0
            sync_status = results[3] is False  # False means synced
            latest_block = results[4] if results[4] else {}
            
            # Calculate block time
            block_time = 0
            if latest_block and 'timestamp' in latest_block:
                block_timestamp = int(latest_block['timestamp'], 16)
                current_time = int(datetime.now().timestamp())
                block_time = current_time - block_timestamp
            
            # Get transaction count from latest block
            tx_count = len(latest_block.get('transactions', [])) if latest_block else 0
            
            return BlockchainMetrics(
                network_id=network_name,
                block_height=block_number,
                block_time=block_time,
                transaction_count=tx_count,
                pending_transactions=0,  # Would need additional RPC call
                gas_price=gas_price,
                network_hashrate=0,  # Would need additional data
                peer_count=peer_count,
                sync_status=sync_status,
                timestamp=int(datetime.now().timestamp())
            )
            
        except Exception as e:
            logger.error(f"Failed to get blockchain metrics for {network_name}: {e}")
            return None
    
    async def _monitor_transactions(self, network_name: str, rpc_url: str):
        """Monitor recent transactions for performance metrics."""
        try:
            # Get latest block
            payload = {
                'jsonrpc': '2.0',
                'method': 'eth_getBlockByNumber',
                'params': ['latest', True],
                'id': 1
            }
            
            async with self.session.post(rpc_url, json=payload) as response:
                if response.status != 200:
                    return
                
                data = await response.json()
                block = data.get('result')
                if not block or 'transactions' not in block:
                    return
                
                # Process recent transactions
                for tx in block['transactions'][-10:]:  # Last 10 transactions
                    tx_metrics = TransactionMetrics(
                        tx_hash=tx.get('hash', ''),
                        from_address=tx.get('from', ''),
                        to_address=tx.get('to', '') or '',
                        value=int(tx.get('value', '0'), 16) / 10**18,  # Convert to ETH
                        gas_used=int(tx.get('gas', '0'), 16),
                        gas_price=int(tx.get('gasPrice', '0'), 16) / 10**9,  # Convert to Gwei
                        status='pending',  # Would need receipt for actual status
                        block_number=int(block['number'], 16),
                        confirmation_time=0,  # Would calculate from pending time
                        timestamp=int(datetime.now().timestamp())
                    )
                    
                    self.transaction_metrics[network_name].append(tx_metrics)
                    
        except Exception as e:
            logger.error(f"Failed to monitor transactions for {network_name}: {e}")
    
    async def _monitor_contracts(self, network_name: str, contracts: List[Dict[str, str]]):
        """Monitor smart contract activity."""
        try:
            for contract in contracts:
                contract_address = contract['address']
                
                # Get contract events (simplified)
                # In production, would use event filters and logs
                
                if contract_address not in self.contract_metrics:
                    self.contract_metrics[contract_address] = ContractMetrics(
                        contract_address=contract_address,
                        function_calls=0,
                        gas_consumption=0,
                        events_emitted=0,
                        error_count=0,
                        avg_execution_time=0.0,
                        last_activity=int(datetime.now().timestamp())
                    )
                
        except Exception as e:
            logger.error(f"Failed to monitor contracts for {network_name}: {e}")
    
    async def _monitor_websocket(self, network_name: str):
        """Monitor blockchain via WebSocket for real-time updates."""
        network_config = self.networks[network_name]
        websocket_url = network_config.get('websocket_url')
        
        if not websocket_url:
            return
        
        while self.is_monitoring:
            try:
                async with websockets.connect(websocket_url) as ws:
                    self.websocket_connections[network_name] = ws
                    
                    # Subscribe to new blocks
                    subscribe_msg = {
                        'jsonrpc': '2.0',
                        'method': 'eth_subscribe',
                        'params': ['newHeads'],
                        'id': 1
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    # Listen for updates
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(network_name, data)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"WebSocket error for {network_name}: {e}")
                await asyncio.sleep(10)  # Wait before reconnecting
    
    async def _handle_websocket_message(self, network_name: str, data: Dict[str, Any]):
        """Handle WebSocket messages from blockchain networks."""
        try:
            if 'params' in data and 'result' in data['params']:
                block_header = data['params']['result']
                
                # Emit event for new block
                await self._emit_event('new_block', {
                    'network': network_name,
                    'block_number': int(block_header.get('number', '0'), 16),
                    'timestamp': int(block_header.get('timestamp', '0'), 16)
                })
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _monitor_performance(self):
        """Monitor system performance metrics."""
        while self.is_monitoring:
            try:
                # Get system metrics (simplified)
                import psutil
                
                cpu_usage = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network_io = psutil.net_io_counters()
                
                metrics = PerformanceMetrics(
                    cpu_usage=cpu_usage / 100.0,
                    memory_usage=memory.percent / 100.0,
                    disk_usage=disk.percent / 100.0,
                    network_io={
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv
                    },
                    response_time=0.0,  # Would measure API response times
                    error_rate=0.0,  # Would calculate from error logs
                    throughput=0.0,  # Would calculate from request logs
                    timestamp=int(datetime.now().timestamp())
                )
                
                self.performance_metrics.append(metrics)
                await self._check_performance_alerts(metrics)
                
            except ImportError:
                logger.warning("psutil not available for performance monitoring")
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _check_blockchain_alerts(self, network_name: str, metrics: BlockchainMetrics):
        """Check blockchain metrics against alert thresholds."""
        alerts_to_create = []
        
        # Check block time
        if metrics.block_time > self.alert_thresholds['block_time_threshold']:
            alerts_to_create.append(Alert(
                alert_id=f"{network_name}_slow_blocks",
                severity=AlertSeverity.WARNING,
                title=f"Slow block production on {network_name}",
                description=f"Block time: {metrics.block_time}s (threshold: {self.alert_thresholds['block_time_threshold']}s)",
                timestamp=int(datetime.now().timestamp()),
                source=f"blockchain_monitor_{network_name}",
                metadata={'network': network_name, 'block_time': metrics.block_time}
            ))
        
        # Check gas price
        if metrics.gas_price > self.alert_thresholds['gas_price_threshold']:
            alerts_to_create.append(Alert(
                alert_id=f"{network_name}_high_gas",
                severity=AlertSeverity.WARNING,
                title=f"High gas prices on {network_name}",
                description=f"Gas price: {metrics.gas_price} Gwei (threshold: {self.alert_thresholds['gas_price_threshold']} Gwei)",
                timestamp=int(datetime.now().timestamp()),
                source=f"blockchain_monitor_{network_name}",
                metadata={'network': network_name, 'gas_price': metrics.gas_price}
            ))
        
        # Check peer count
        if metrics.peer_count < self.alert_thresholds['peer_count_threshold']:
            alerts_to_create.append(Alert(
                alert_id=f"{network_name}_low_peers",
                severity=AlertSeverity.ERROR,
                title=f"Low peer count on {network_name}",
                description=f"Peer count: {metrics.peer_count} (threshold: {self.alert_thresholds['peer_count_threshold']})",
                timestamp=int(datetime.now().timestamp()),
                source=f"blockchain_monitor_{network_name}",
                metadata={'network': network_name, 'peer_count': metrics.peer_count}
            ))
        
        # Check sync status
        if not metrics.sync_status:
            alerts_to_create.append(Alert(
                alert_id=f"{network_name}_not_synced",
                severity=AlertSeverity.CRITICAL,
                title=f"Node not synced on {network_name}",
                description=f"Node is not fully synced with the network",
                timestamp=int(datetime.now().timestamp()),
                source=f"blockchain_monitor_{network_name}",
                metadata={'network': network_name, 'sync_status': metrics.sync_status}
            ))
        
        # Create alerts
        for alert in alerts_to_create:
            await self._create_alert(alert)
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check performance metrics against alert thresholds."""
        alerts_to_create = []
        
        # Check CPU usage
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage_threshold']:
            alerts_to_create.append(Alert(
                alert_id="high_cpu_usage",
                severity=AlertSeverity.WARNING,
                title="High CPU usage",
                description=f"CPU usage: {metrics.cpu_usage*100:.1f}% (threshold: {self.alert_thresholds['cpu_usage_threshold']*100:.1f}%)",
                timestamp=int(datetime.now().timestamp()),
                source="performance_monitor",
                metadata={'cpu_usage': metrics.cpu_usage}
            ))
        
        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds['memory_usage_threshold']:
            alerts_to_create.append(Alert(
                alert_id="high_memory_usage",
                severity=AlertSeverity.WARNING,
                title="High memory usage",
                description=f"Memory usage: {metrics.memory_usage*100:.1f}% (threshold: {self.alert_thresholds['memory_usage_threshold']*100:.1f}%)",
                timestamp=int(datetime.now().timestamp()),
                source="performance_monitor",
                metadata={'memory_usage': metrics.memory_usage}
            ))
        
        # Check disk usage
        if metrics.disk_usage > self.alert_thresholds['disk_usage_threshold']:
            alerts_to_create.append(Alert(
                alert_id="high_disk_usage",
                severity=AlertSeverity.ERROR,
                title="High disk usage",
                description=f"Disk usage: {metrics.disk_usage*100:.1f}% (threshold: {self.alert_thresholds['disk_usage_threshold']*100:.1f}%)",
                timestamp=int(datetime.now().timestamp()),
                source="performance_monitor",
                metadata={'disk_usage': metrics.disk_usage}
            ))
        
        # Create alerts
        for alert in alerts_to_create:
            await self._create_alert(alert)
    
    async def _create_alert(self, alert: Alert):
        """Create and process a new alert."""
        # Check if alert already exists and is active
        if alert.alert_id in self.active_alerts:
            return
        
        self.alerts.append(alert)
        self.active_alerts[alert.alert_id] = alert
        
        # Emit alert event
        await self._emit_event('alert_created', asdict(alert))
        
        logger.warning(f"Alert created: {alert.title} - {alert.description}")
        
        # Send email if enabled
        if self.enable_email_alerts:
            await self._send_email_alert(alert)
    
    async def _process_alerts(self):
        """Process and auto-resolve alerts."""
        while self.is_monitoring:
            try:
                current_time = int(datetime.now().timestamp())
                alerts_to_resolve = []
                
                # Check for alerts that should be auto-resolved
                for alert_id, alert in self.active_alerts.items():
                    # Auto-resolve alerts older than 1 hour if conditions are met
                    if current_time - alert.timestamp > 3600:
                        if await self._should_auto_resolve_alert(alert):
                            alerts_to_resolve.append(alert_id)
                
                # Resolve alerts
                for alert_id in alerts_to_resolve:
                    await self.resolve_alert(alert_id, "Auto-resolved")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(60)
    
    async def _should_auto_resolve_alert(self, alert: Alert) -> bool:
        """Check if alert conditions are no longer present."""
        try:
            # Get recent metrics for the alert's source
            if alert.source.startswith('blockchain_monitor_'):
                network_name = alert.source.replace('blockchain_monitor_', '')
                if network_name in self.blockchain_metrics:
                    recent_metrics = list(self.blockchain_metrics[network_name])[-5:]  # Last 5 readings
                    
                    # Check if conditions have improved
                    if alert.alert_id.endswith('_slow_blocks'):
                        return all(m.block_time <= self.alert_thresholds['block_time_threshold'] 
                                 for m in recent_metrics)
                    elif alert.alert_id.endswith('_high_gas'):
                        return all(m.gas_price <= self.alert_thresholds['gas_price_threshold'] 
                                 for m in recent_metrics)
                    elif alert.alert_id.endswith('_low_peers'):
                        return all(m.peer_count >= self.alert_thresholds['peer_count_threshold'] 
                                 for m in recent_metrics)
                    elif alert.alert_id.endswith('_not_synced'):
                        return all(m.sync_status for m in recent_metrics)
            
            elif alert.source == 'performance_monitor':
                recent_metrics = list(self.performance_metrics)[-5:]  # Last 5 readings
                
                if alert.alert_id == 'high_cpu_usage':
                    return all(m.cpu_usage <= self.alert_thresholds['cpu_usage_threshold'] 
                             for m in recent_metrics)
                elif alert.alert_id == 'high_memory_usage':
                    return all(m.memory_usage <= self.alert_thresholds['memory_usage_threshold'] 
                             for m in recent_metrics)
                elif alert.alert_id == 'high_disk_usage':
                    return all(m.disk_usage <= self.alert_thresholds['disk_usage_threshold'] 
                             for m in recent_metrics)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking auto-resolve conditions: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Manually resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.is_resolved = True
            alert.resolved_at = int(datetime.now().timestamp())
            
            del self.active_alerts[alert_id]
            
            await self._emit_event('alert_resolved', {
                'alert_id': alert_id,
                'resolution_note': resolution_note,
                'resolved_at': alert.resolved_at
            })
            
            logger.info(f"Alert resolved: {alert_id} - {resolution_note}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for specific event type."""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification."""
        try:
            # Email sending would be implemented here
            # For now, just log the alert
            logger.info(f"Email alert would be sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_network_status(self, network_name: str) -> NetworkStatus:
        """Get overall status of a blockchain network."""
        if network_name not in self.blockchain_metrics:
            return NetworkStatus.OFFLINE
        
        recent_metrics = list(self.blockchain_metrics[network_name])[-5:]
        if not recent_metrics:
            return NetworkStatus.OFFLINE
        
        # Check for critical issues
        if any(not m.sync_status for m in recent_metrics):
            return NetworkStatus.UNHEALTHY
        
        if any(m.peer_count < self.alert_thresholds['peer_count_threshold'] for m in recent_metrics):
            return NetworkStatus.DEGRADED
        
        # Check for performance issues
        avg_block_time = statistics.mean(m.block_time for m in recent_metrics)
        if avg_block_time > self.alert_thresholds['block_time_threshold']:
            return NetworkStatus.DEGRADED
        
        return NetworkStatus.HEALTHY
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        summary = {
            'networks': {},
            'alerts': {
                'active': len(self.active_alerts),
                'total': len(self.alerts),
                'by_severity': defaultdict(int)
            },
            'performance': {},
            'uptime': int(datetime.now().timestamp()) if self.is_monitoring else 0
        }
        
        # Network summaries
        for network_name in self.networks.keys():
            if network_name in self.blockchain_metrics:
                recent_metrics = list(self.blockchain_metrics[network_name])[-1:]
                if recent_metrics:
                    latest = recent_metrics[0]
                    summary['networks'][network_name] = {
                        'status': self.get_network_status(network_name).value,
                        'block_height': latest.block_height,
                        'peer_count': latest.peer_count,
                        'gas_price': latest.gas_price,
                        'last_update': latest.timestamp
                    }
        
        # Alert summary
        for alert in self.alerts:
            summary['alerts']['by_severity'][alert.severity.value] += 1
        
        # Performance summary
        if self.performance_metrics:
            latest_perf = self.performance_metrics[-1]
            summary['performance'] = {
                'cpu_usage': latest_perf.cpu_usage,
                'memory_usage': latest_perf.memory_usage,
                'disk_usage': latest_perf.disk_usage,
                'last_update': latest_perf.timestamp
            }
        
        return summary
    
    async def cleanup(self):
        """Cleanup monitoring resources."""
        await self.stop_monitoring()
        
        if self.executor:
            self.executor.shutdown(wait=True)