"""
Federated Learning Orchestration System

This module implements comprehensive orchestration for federated learning including
client selection strategies, resource allocation, fault tolerance, and incentive
mechanisms.
"""

import asyncio
import logging
import time
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
from collections import defaultdict, deque
import uuid
import math

logger = logging.getLogger(__name__)


class ClientStatus(Enum):
    """Client status enumeration"""
    AVAILABLE = "available"
    TRAINING = "training"
    UPLOADING = "uploading"
    OFFLINE = "offline"
    FAULT = "fault"
    EXCLUDED = "excluded"


class SelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_AWARE = "resource_aware"
    FAIRNESS_AWARE = "fairness_aware"
    CLUSTERED = "clustered"
    AUCTION_BASED = "auction_based"
    ADAPTIVE = "adaptive"


@dataclass
class ClientProfile:
    """Comprehensive client profile"""
    client_id: str
    status: ClientStatus = ClientStatus.AVAILABLE
    
    # Hardware capabilities
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    storage_gb: float = 1.0
    network_bandwidth_mbps: float = 10.0
    
    # Performance metrics
    avg_training_time: float = 0.0
    avg_communication_time: float = 0.0
    success_rate: float = 1.0
    reliability_score: float = 1.0
    
    # Data characteristics
    data_size: int = 0
    data_quality_score: float = 1.0
    label_distribution: Optional[Dict[int, int]] = None
    
    # Availability
    online_hours: List[Tuple[int, int]] = field(default_factory=list)  # (start_hour, end_hour)
    timezone_offset: int = 0
    battery_level: float = 100.0
    is_charging: bool = True
    
    # Participation history
    rounds_participated: int = 0
    total_contribution: float = 0.0
    last_participation: Optional[float] = None
    consecutive_failures: int = 0
    
    # Incentives
    reputation_score: float = 1.0
    reward_balance: float = 0.0
    cost_per_round: float = 1.0
    
    # Privacy
    privacy_sensitivity: float = 1.0
    privacy_budget: float = 10.0


@dataclass
class TrainingRoundPlan:
    """Training round execution plan"""
    round_id: int
    selected_clients: List[str]
    resource_allocation: Dict[str, Dict] = field(default_factory=dict)
    expected_duration: float = 0.0
    timeout: float = 0.0
    min_successful_clients: int = 0
    backup_clients: List[str] = field(default_factory=list)


@dataclass
class ResourceAllocation:
    """Resource allocation for a client"""
    client_id: str
    allocated_cpu: float = 1.0
    allocated_memory_gb: float = 1.0
    allocated_bandwidth_mbps: float = 10.0
    training_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    model_compression_ratio: float = 1.0


@dataclass
class IncentiveScheme:
    """Incentive mechanism configuration"""
    base_reward: float = 1.0
    performance_bonus: float = 0.5
    data_quality_bonus: float = 0.3
    participation_bonus: float = 0.2
    early_completion_bonus: float = 0.1
    reputation_multiplier: float = 1.0


class FederatedOrchestrator:
    """
    Comprehensive orchestration system for federated learning
    
    Features:
    - Intelligent client selection strategies
    - Dynamic resource allocation
    - Fault tolerance and recovery
    - Incentive mechanisms
    - Load balancing
    - Performance optimization
    - Fairness guarantees
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Configuration
        self.selection_strategy = SelectionStrategy(
            self.config.get("selection_strategy", "adaptive")
        )
        self.min_clients_per_round = self.config.get("min_clients_per_round", 5)
        self.max_clients_per_round = self.config.get("max_clients_per_round", 20)
        self.round_timeout = self.config.get("round_timeout", 300)  # seconds
        self.fault_tolerance_enabled = self.config.get("fault_tolerance", True)
        self.incentives_enabled = self.config.get("incentives", True)
        
        # State
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.active_rounds: Dict[int, TrainingRoundPlan] = {}
        self.round_history: List[TrainingRoundPlan] = []
        self.client_clusters: Dict[str, List[str]] = {}
        
        # Resource management
        self.total_server_resources = {
            "cpu_cores": self.config.get("server_cpu_cores", 8),
            "memory_gb": self.config.get("server_memory_gb", 32),
            "bandwidth_gbps": self.config.get("server_bandwidth_gbps", 1)
        }
        self.allocated_resources = {
            "cpu_cores": 0,
            "memory_gb": 0,
            "bandwidth_gbps": 0
        }
        
        # Fairness tracking
        self.client_participation_history = defaultdict(list)
        self.fairness_weights = defaultdict(float)
        
        # Incentive system
        self.incentive_scheme = IncentiveScheme(
            base_reward=self.config.get("base_reward", 1.0),
            performance_bonus=self.config.get("performance_bonus", 0.5)
        )
        
        # Performance tracking
        self.performance_metrics = {
            "avg_round_time": 0.0,
            "client_satisfaction": 0.0,
            "resource_utilization": 0.0,
            "fault_recovery_rate": 0.0
        }
        
        # Background tasks
        self.orchestration_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info(f"Federated orchestrator initialized with {self.selection_strategy.value} strategy")
    
    async def start_orchestration(self):
        """Start orchestration services"""
        self.is_running = True
        
        # Start background tasks
        self.orchestration_tasks = [
            asyncio.create_task(self._client_monitor()),
            asyncio.create_task(self._resource_optimizer()),
            asyncio.create_task(self._fault_detector()),
            asyncio.create_task(self._performance_analyzer()),
            asyncio.create_task(self._fairness_monitor())
        ]
        
        logger.info("Federated orchestration started")
    
    async def stop_orchestration(self):
        """Stop orchestration services"""
        self.is_running = False
        
        for task in self.orchestration_tasks:
            task.cancel()
        
        await asyncio.gather(*self.orchestration_tasks, return_exceptions=True)
        
        logger.info("Federated orchestration stopped")
    
    def register_client(
        self, 
        client_id: str, 
        capabilities: Dict[str, Any],
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register a new client with the orchestrator"""
        try:
            profile = ClientProfile(
                client_id=client_id,
                cpu_cores=capabilities.get("cpu_cores", 1),
                memory_gb=capabilities.get("memory_gb", 1.0),
                gpu_available=capabilities.get("gpu_available", False),
                gpu_memory_gb=capabilities.get("gpu_memory_gb", 0.0),
                storage_gb=capabilities.get("storage_gb", 1.0),
                network_bandwidth_mbps=capabilities.get("network_bandwidth_mbps", 10.0),
                data_size=capabilities.get("data_size", 0),
                data_quality_score=capabilities.get("data_quality_score", 1.0),
                battery_level=capabilities.get("battery_level", 100.0),
                is_charging=capabilities.get("is_charging", True)
            )
            
            # Set preferences
            if preferences:
                profile.online_hours = preferences.get("online_hours", [(0, 24)])
                profile.timezone_offset = preferences.get("timezone_offset", 0)
                profile.privacy_sensitivity = preferences.get("privacy_sensitivity", 1.0)
                profile.cost_per_round = preferences.get("cost_per_round", 1.0)
            
            self.client_profiles[client_id] = profile
            
            # Initialize fairness weight
            self.fairness_weights[client_id] = 1.0
            
            # Assign to cluster if using clustered strategy
            if self.selection_strategy == SelectionStrategy.CLUSTERED:
                self._assign_client_to_cluster(client_id, profile)
            
            logger.info(f"Client {client_id} registered with orchestrator")
            
            return {
                "status": "registered",
                "client_id": client_id,
                "assigned_cluster": self._get_client_cluster(client_id),
                "fairness_weight": self.fairness_weights[client_id]
            }
            
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def unregister_client(self, client_id: str) -> Dict[str, Any]:
        """Unregister a client"""
        try:
            if client_id in self.client_profiles:
                self.client_profiles[client_id].status = ClientStatus.OFFLINE
                logger.info(f"Client {client_id} unregistered")
                return {"status": "unregistered"}
            else:
                return {"status": "not_found"}
                
        except Exception as e:
            logger.error(f"Client unregistration failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def update_client_status(self, client_id: str, status: ClientStatus, metadata: Optional[Dict] = None):
        """Update client status"""
        if client_id in self.client_profiles:
            self.client_profiles[client_id].status = status
            
            if metadata:
                profile = self.client_profiles[client_id]
                profile.battery_level = metadata.get("battery_level", profile.battery_level)
                profile.is_charging = metadata.get("is_charging", profile.is_charging)
                
    async def plan_training_round(self, round_id: int, requirements: Optional[Dict] = None) -> TrainingRoundPlan:
        """Plan a training round with optimal client selection and resource allocation"""
        try:
            requirements = requirements or {}
            
            # Determine number of clients needed
            min_clients = requirements.get("min_clients", self.min_clients_per_round)
            max_clients = requirements.get("max_clients", self.max_clients_per_round)
            target_clients = min(max_clients, max(min_clients, len(self._get_available_clients()) // 2))
            
            # Select clients based on strategy
            selected_clients = await self._select_clients(target_clients, requirements)
            
            if len(selected_clients) < min_clients:
                raise ValueError(f"Insufficient clients available: {len(selected_clients)} < {min_clients}")
            
            # Select backup clients
            all_available = set(self._get_available_clients())
            backup_pool = all_available - set(selected_clients)
            backup_clients = list(backup_pool)[:min(3, len(backup_pool))]
            
            # Allocate resources
            resource_allocation = await self._allocate_resources(selected_clients, requirements)
            
            # Estimate duration
            expected_duration = self._estimate_round_duration(selected_clients, resource_allocation)
            timeout = expected_duration * 2  # 2x safety margin
            
            # Create training plan
            plan = TrainingRoundPlan(
                round_id=round_id,
                selected_clients=selected_clients,
                resource_allocation=resource_allocation,
                expected_duration=expected_duration,
                timeout=timeout,
                min_successful_clients=max(1, len(selected_clients) // 2),
                backup_clients=backup_clients
            )
            
            self.active_rounds[round_id] = plan
            
            # Update client statuses
            for client_id in selected_clients:
                self.client_profiles[client_id].status = ClientStatus.TRAINING
                self.client_profiles[client_id].last_participation = time.time()
            
            logger.info(f"Training round {round_id} planned with {len(selected_clients)} clients")
            
            return plan
            
        except Exception as e:
            logger.error(f"Training round planning failed: {e}")
            raise
    
    def _get_available_clients(self) -> List[str]:
        """Get list of available clients"""
        available = []
        current_time = time.time()
        current_hour = datetime.fromtimestamp(current_time).hour
        
        for client_id, profile in self.client_profiles.items():
            if profile.status != ClientStatus.AVAILABLE:
                continue
            
            # Check availability hours
            if profile.online_hours:
                is_online = any(
                    start <= current_hour < end 
                    for start, end in profile.online_hours
                )
                if not is_online:
                    continue
            
            # Check battery level for mobile devices
            if profile.battery_level < 20 and not profile.is_charging:
                continue
            
            # Check consecutive failures
            if profile.consecutive_failures > 3:
                continue
            
            available.append(client_id)
        
        return available
    
    async def _select_clients(
        self, 
        target_count: int, 
        requirements: Dict
    ) -> List[str]:
        """Select clients based on configured strategy"""
        available_clients = self._get_available_clients()
        
        if not available_clients:
            return []
        
        if self.selection_strategy == SelectionStrategy.RANDOM:
            return self._random_selection(available_clients, target_count)
        elif self.selection_strategy == SelectionStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_clients, target_count)
        elif self.selection_strategy == SelectionStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(available_clients, target_count)
        elif self.selection_strategy == SelectionStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(available_clients, target_count, requirements)
        elif self.selection_strategy == SelectionStrategy.FAIRNESS_AWARE:
            return self._fairness_aware_selection(available_clients, target_count)
        elif self.selection_strategy == SelectionStrategy.CLUSTERED:
            return self._clustered_selection(available_clients, target_count)
        elif self.selection_strategy == SelectionStrategy.AUCTION_BASED:
            return await self._auction_based_selection(available_clients, target_count)
        elif self.selection_strategy == SelectionStrategy.ADAPTIVE:
            return await self._adaptive_selection(available_clients, target_count, requirements)
        else:
            return self._random_selection(available_clients, target_count)
    
    def _random_selection(self, available_clients: List[str], target_count: int) -> List[str]:
        """Random client selection"""
        return random.sample(available_clients, min(target_count, len(available_clients)))
    
    def _round_robin_selection(self, available_clients: List[str], target_count: int) -> List[str]:
        """Round-robin client selection for fairness"""
        # Sort by last participation time (oldest first)
        clients_by_participation = sorted(
            available_clients,
            key=lambda cid: self.client_profiles[cid].last_participation or 0
        )
        
        return clients_by_participation[:target_count]
    
    def _performance_based_selection(self, available_clients: List[str], target_count: int) -> List[str]:
        """Select clients based on performance metrics"""
        # Calculate performance scores
        client_scores = []
        
        for client_id in available_clients:
            profile = self.client_profiles[client_id]
            
            # Performance score components
            reliability_score = profile.reliability_score
            success_rate = profile.success_rate
            
            # Speed score (inverse of training time)
            speed_score = 1.0 / (profile.avg_training_time + 1.0)
            
            # Data quality score
            data_score = profile.data_quality_score
            
            # Combined score
            total_score = (
                0.3 * reliability_score +
                0.3 * success_rate +
                0.2 * speed_score +
                0.2 * data_score
            )
            
            client_scores.append((client_id, total_score))
        
        # Select top performers
        client_scores.sort(key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in client_scores[:target_count]]
    
    def _resource_aware_selection(
        self, 
        available_clients: List[str], 
        target_count: int,
        requirements: Dict
    ) -> List[str]:
        """Select clients based on resource requirements"""
        min_memory = requirements.get("min_memory_gb", 1.0)
        min_cpu_cores = requirements.get("min_cpu_cores", 1)
        prefer_gpu = requirements.get("prefer_gpu", False)
        
        # Filter and score clients
        suitable_clients = []
        
        for client_id in available_clients:
            profile = self.client_profiles[client_id]
            
            # Check minimum requirements
            if profile.memory_gb < min_memory or profile.cpu_cores < min_cpu_cores:
                continue
            
            # Calculate resource score
            cpu_score = min(1.0, profile.cpu_cores / 4.0)  # Normalize to 4 cores
            memory_score = min(1.0, profile.memory_gb / 8.0)  # Normalize to 8GB
            gpu_score = 1.0 if profile.gpu_available and prefer_gpu else 0.5
            network_score = min(1.0, profile.network_bandwidth_mbps / 100.0)
            
            resource_score = (
                0.3 * cpu_score +
                0.3 * memory_score +
                0.2 * gpu_score +
                0.2 * network_score
            )
            
            suitable_clients.append((client_id, resource_score))
        
        # Select clients with best resources
        suitable_clients.sort(key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in suitable_clients[:target_count]]
    
    def _fairness_aware_selection(self, available_clients: List[str], target_count: int) -> List[str]:
        """Select clients to ensure fairness across participants"""
        # Use fairness weights to prioritize underrepresented clients
        weighted_clients = [
            (client_id, self.fairness_weights[client_id])
            for client_id in available_clients
        ]
        
        # Sort by fairness weight (highest first)
        weighted_clients.sort(key=lambda x: x[1], reverse=True)
        
        return [client_id for client_id, _ in weighted_clients[:target_count]]
    
    def _clustered_selection(self, available_clients: List[str], target_count: int) -> List[str]:
        """Select clients from different clusters for diversity"""
        if not self.client_clusters:
            return self._random_selection(available_clients, target_count)
        
        selected = []
        clients_per_cluster = max(1, target_count // len(self.client_clusters))
        
        for cluster_id, cluster_clients in self.client_clusters.items():
            available_in_cluster = [c for c in cluster_clients if c in available_clients]
            
            if available_in_cluster:
                cluster_selection = random.sample(
                    available_in_cluster,
                    min(clients_per_cluster, len(available_in_cluster))
                )
                selected.extend(cluster_selection)
        
        # Fill remaining slots randomly
        remaining = target_count - len(selected)
        if remaining > 0:
            remaining_clients = [c for c in available_clients if c not in selected]
            additional = random.sample(
                remaining_clients,
                min(remaining, len(remaining_clients))
            )
            selected.extend(additional)
        
        return selected[:target_count]
    
    async def _auction_based_selection(self, available_clients: List[str], target_count: int) -> List[str]:
        """Auction-based client selection with cost consideration"""
        if not self.incentives_enabled:
            return self._random_selection(available_clients, target_count)
        
        # Simulate auction bids
        bids = []
        
        for client_id in available_clients:
            profile = self.client_profiles[client_id]
            
            # Calculate bid based on cost and quality
            base_cost = profile.cost_per_round
            quality_multiplier = profile.data_quality_score * profile.reliability_score
            
            # Add some randomness to simulate real auction
            bid = base_cost * quality_multiplier * (0.8 + 0.4 * random.random())
            
            bids.append((client_id, bid))
        
        # Select clients with best value (lowest cost for quality)
        bids.sort(key=lambda x: x[1])
        return [client_id for client_id, _ in bids[:target_count]]
    
    async def _adaptive_selection(
        self, 
        available_clients: List[str], 
        target_count: int,
        requirements: Dict
    ) -> List[str]:
        """Adaptive selection combining multiple strategies"""
        # Analyze current system state
        round_count = len(self.round_history)
        
        if round_count < 5:
            # Early rounds: focus on diversity
            return self._fairness_aware_selection(available_clients, target_count)
        elif round_count < 20:
            # Middle rounds: balance performance and fairness
            performance_clients = self._performance_based_selection(available_clients, target_count // 2)
            fairness_clients = self._fairness_aware_selection(
                [c for c in available_clients if c not in performance_clients],
                target_count - len(performance_clients)
            )
            return performance_clients + fairness_clients
        else:
            # Later rounds: optimize for performance
            return self._performance_based_selection(available_clients, target_count)
    
    async def _allocate_resources(
        self, 
        selected_clients: List[str],
        requirements: Dict
    ) -> Dict[str, ResourceAllocation]:
        """Allocate resources to selected clients"""
        allocations = {}
        
        # Get resource requirements
        target_epochs = requirements.get("epochs", 1)
        base_batch_size = requirements.get("batch_size", 32)
        base_learning_rate = requirements.get("learning_rate", 0.01)
        
        for client_id in selected_clients:
            profile = self.client_profiles[client_id]
            
            # Adjust parameters based on client capabilities
            # More capable clients can handle larger batches and more epochs
            capability_factor = min(2.0, profile.cpu_cores / 2.0 * profile.memory_gb / 4.0)
            
            epochs = max(1, int(target_epochs * capability_factor))
            batch_size = max(16, int(base_batch_size * capability_factor))
            
            # Adjust learning rate based on data size
            data_factor = min(2.0, profile.data_size / 1000.0)
            learning_rate = base_learning_rate * math.sqrt(data_factor)
            
            # Model compression for low-resource clients
            compression_ratio = 1.0
            if profile.memory_gb < 2.0 or profile.network_bandwidth_mbps < 50:
                compression_ratio = 0.5
            
            allocation = ResourceAllocation(
                client_id=client_id,
                allocated_cpu=min(profile.cpu_cores, 4),
                allocated_memory_gb=min(profile.memory_gb * 0.8, 4.0),
                allocated_bandwidth_mbps=min(profile.network_bandwidth_mbps, 100.0),
                training_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_compression_ratio=compression_ratio
            )
            
            allocations[client_id] = allocation
        
        return allocations
    
    def _estimate_round_duration(
        self, 
        selected_clients: List[str],
        resource_allocation: Dict[str, ResourceAllocation]
    ) -> float:
        """Estimate training round duration"""
        max_duration = 0.0
        
        for client_id in selected_clients:
            profile = self.client_profiles[client_id]
            allocation = resource_allocation[client_id]
            
            # Base training time estimate
            if profile.avg_training_time > 0:
                training_time = profile.avg_training_time * allocation.training_epochs
            else:
                # Default estimate based on capabilities
                training_time = 60.0 * allocation.training_epochs / max(1, profile.cpu_cores)
            
            # Communication time
            comm_time = profile.avg_communication_time or 30.0
            
            total_time = training_time + comm_time
            max_duration = max(max_duration, total_time)
        
        return max_duration
    
    def _assign_client_to_cluster(self, client_id: str, profile: ClientProfile):
        """Assign client to a cluster based on characteristics"""
        # Simple clustering based on resource tiers
        if profile.gpu_available and profile.memory_gb >= 4.0:
            cluster = "high_resource"
        elif profile.memory_gb >= 2.0 and profile.cpu_cores >= 2:
            cluster = "medium_resource"
        else:
            cluster = "low_resource"
        
        if cluster not in self.client_clusters:
            self.client_clusters[cluster] = []
        
        self.client_clusters[cluster].append(client_id)
    
    def _get_client_cluster(self, client_id: str) -> Optional[str]:
        """Get cluster assignment for client"""
        for cluster_id, clients in self.client_clusters.items():
            if client_id in clients:
                return cluster_id
        return None
    
    async def handle_client_failure(self, client_id: str, round_id: int, error_info: Dict):
        """Handle client failure during training"""
        if round_id not in self.active_rounds:
            return
        
        plan = self.active_rounds[round_id]
        profile = self.client_profiles.get(client_id)
        
        if profile:
            profile.consecutive_failures += 1
            profile.success_rate = max(0.0, profile.success_rate - 0.1)
            profile.status = ClientStatus.FAULT
            
            logger.warning(f"Client {client_id} failed in round {round_id}: {error_info}")
        
        # Check if we need backup clients
        failed_clients = sum(
            1 for cid in plan.selected_clients
            if self.client_profiles.get(cid, {}).status == ClientStatus.FAULT
        )
        
        successful_clients = len(plan.selected_clients) - failed_clients
        
        if successful_clients < plan.min_successful_clients and plan.backup_clients:
            # Activate backup client
            backup_client = plan.backup_clients.pop(0)
            plan.selected_clients.append(backup_client)
            
            # Allocate resources to backup client
            backup_allocation = await self._allocate_resources([backup_client], {})
            plan.resource_allocation.update(backup_allocation)
            
            # Update status
            self.client_profiles[backup_client].status = ClientStatus.TRAINING
            
            logger.info(f"Activated backup client {backup_client} for round {round_id}")
    
    async def complete_training_round(self, round_id: int, results: Dict[str, Any]):
        """Complete a training round and update metrics"""
        if round_id not in self.active_rounds:
            return
        
        plan = self.active_rounds[round_id]
        
        # Update client profiles
        for client_id in plan.selected_clients:
            if client_id not in results:
                continue
            
            profile = self.client_profiles[client_id]
            client_result = results[client_id]
            
            if client_result.get("success", False):
                profile.consecutive_failures = 0
                profile.success_rate = min(1.0, profile.success_rate + 0.05)
                profile.rounds_participated += 1
                profile.total_contribution += client_result.get("contribution", 1.0)
                
                # Update timing metrics
                if "training_time" in client_result:
                    profile.avg_training_time = (
                        0.8 * profile.avg_training_time + 
                        0.2 * client_result["training_time"]
                    )
                
                if "communication_time" in client_result:
                    profile.avg_communication_time = (
                        0.8 * profile.avg_communication_time +
                        0.2 * client_result["communication_time"]
                    )
                
                # Calculate and distribute incentives
                if self.incentives_enabled:
                    reward = self._calculate_client_reward(client_id, client_result)
                    profile.reward_balance += reward
                
            else:
                profile.consecutive_failures += 1
                profile.success_rate = max(0.0, profile.success_rate - 0.1)
            
            profile.status = ClientStatus.AVAILABLE
        
        # Update fairness weights
        self._update_fairness_weights(plan.selected_clients)
        
        # Move to history
        self.round_history.append(plan)
        del self.active_rounds[round_id]
        
        # Update performance metrics
        self._update_performance_metrics(plan, results)
        
        logger.info(f"Training round {round_id} completed with {len(results)} participants")
    
    def _calculate_client_reward(self, client_id: str, result: Dict[str, Any]) -> float:
        """Calculate reward for client based on contribution"""
        profile = self.client_profiles[client_id]
        
        base_reward = self.incentive_scheme.base_reward
        
        # Performance bonus
        performance_bonus = 0.0
        if "loss_improvement" in result:
            performance_bonus = self.incentive_scheme.performance_bonus * result["loss_improvement"]
        
        # Data quality bonus
        data_bonus = self.incentive_scheme.data_quality_bonus * profile.data_quality_score
        
        # Participation bonus for consistent participation
        participation_bonus = 0.0
        if profile.rounds_participated > 0:
            participation_rate = profile.rounds_participated / max(1, len(self.round_history))
            participation_bonus = self.incentive_scheme.participation_bonus * participation_rate
        
        # Early completion bonus
        early_bonus = 0.0
        if "completed_early" in result and result["completed_early"]:
            early_bonus = self.incentive_scheme.early_completion_bonus
        
        # Reputation multiplier
        reputation_multiplier = min(2.0, profile.reputation_score * self.incentive_scheme.reputation_multiplier)
        
        total_reward = (base_reward + performance_bonus + data_bonus + participation_bonus + early_bonus) * reputation_multiplier
        
        return max(0.0, total_reward)
    
    def _update_fairness_weights(self, participated_clients: List[str]):
        """Update fairness weights to ensure equitable participation"""
        # Decrease weights for participants
        for client_id in participated_clients:
            self.fairness_weights[client_id] *= 0.8
        
        # Increase weights for non-participants
        all_clients = set(self.client_profiles.keys())
        non_participants = all_clients - set(participated_clients)
        
        for client_id in non_participants:
            if self.client_profiles[client_id].status == ClientStatus.AVAILABLE:
                self.fairness_weights[client_id] *= 1.1
        
        # Normalize weights
        total_weight = sum(self.fairness_weights.values())
        if total_weight > 0:
            for client_id in self.fairness_weights:
                self.fairness_weights[client_id] /= total_weight
                self.fairness_weights[client_id] *= len(self.client_profiles)
    
    def _update_performance_metrics(self, plan: TrainingRoundPlan, results: Dict[str, Any]):
        """Update orchestrator performance metrics"""
        # Average round time
        if plan.expected_duration > 0:
            actual_duration = max(r.get("total_time", 0) for r in results.values())
            self.performance_metrics["avg_round_time"] = (
                0.8 * self.performance_metrics["avg_round_time"] +
                0.2 * actual_duration
            )
        
        # Client satisfaction (based on success rate)
        success_rate = sum(1 for r in results.values() if r.get("success", False)) / len(results)
        self.performance_metrics["client_satisfaction"] = (
            0.8 * self.performance_metrics["client_satisfaction"] +
            0.2 * success_rate
        )
    
    # Background monitoring tasks
    async def _client_monitor(self):
        """Monitor client status and availability"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for client_id, profile in self.client_profiles.items():
                    # Check for stale clients
                    if (profile.last_participation and 
                        current_time - profile.last_participation > 3600):  # 1 hour
                        if profile.status == ClientStatus.TRAINING:
                            profile.status = ClientStatus.OFFLINE
                            logger.warning(f"Client {client_id} appears to be stale")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Client monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _resource_optimizer(self):
        """Optimize resource allocation and usage"""
        while self.is_running:
            try:
                # Analyze resource utilization
                total_clients = len(self.client_profiles)
                active_clients = sum(
                    1 for p in self.client_profiles.values() 
                    if p.status in [ClientStatus.TRAINING, ClientStatus.UPLOADING]
                )
                
                utilization = active_clients / total_clients if total_clients > 0 else 0
                self.performance_metrics["resource_utilization"] = utilization
                
                # Suggest optimizations
                if utilization < 0.3:
                    logger.info("Low resource utilization - consider increasing clients per round")
                elif utilization > 0.8:
                    logger.info("High resource utilization - consider load balancing")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Resource optimization error: {e}")
                await asyncio.sleep(300)
    
    async def _fault_detector(self):
        """Detect and handle system faults"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for stuck rounds
                for round_id, plan in list(self.active_rounds.items()):
                    if current_time - plan.expected_duration > plan.timeout:
                        logger.warning(f"Round {round_id} appears to be stuck")
                        # Could implement automatic recovery here
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Fault detection error: {e}")
                await asyncio.sleep(120)
    
    async def _performance_analyzer(self):
        """Analyze and optimize system performance"""
        while self.is_running:
            try:
                # Analyze recent performance
                if len(self.round_history) >= 5:
                    recent_rounds = self.round_history[-5:]
                    
                    # Calculate average success rate
                    # This would require additional data collection
                
                await asyncio.sleep(600)  # Analyze every 10 minutes
                
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                await asyncio.sleep(600)
    
    async def _fairness_monitor(self):
        """Monitor and maintain fairness across clients"""
        while self.is_running:
            try:
                # Calculate participation statistics
                total_rounds = len(self.round_history)
                
                if total_rounds >= 10:
                    participation_stats = {}
                    
                    for client_id, profile in self.client_profiles.items():
                        participation_rate = profile.rounds_participated / total_rounds
                        participation_stats[client_id] = participation_rate
                    
                    # Identify under-participating clients
                    avg_participation = np.mean(list(participation_stats.values()))
                    under_participating = [
                        cid for cid, rate in participation_stats.items()
                        if rate < avg_participation * 0.5
                    ]
                    
                    if under_participating:
                        logger.info(f"Under-participating clients: {under_participating}")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Fairness monitoring error: {e}")
                await asyncio.sleep(1800)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "is_running": self.is_running,
            "selection_strategy": self.selection_strategy.value,
            "total_clients": len(self.client_profiles),
            "available_clients": len(self._get_available_clients()),
            "active_rounds": len(self.active_rounds),
            "completed_rounds": len(self.round_history),
            "performance_metrics": self.performance_metrics,
            "client_clusters": {k: len(v) for k, v in self.client_clusters.items()},
            "fairness_summary": {
                "min_weight": min(self.fairness_weights.values()) if self.fairness_weights else 0,
                "max_weight": max(self.fairness_weights.values()) if self.fairness_weights else 0,
                "avg_weight": np.mean(list(self.fairness_weights.values())) if self.fairness_weights else 0
            }
        }
    
    def get_client_profiles(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Get client profile information"""
        if client_id:
            if client_id in self.client_profiles:
                profile = self.client_profiles[client_id]
                return {
                    "client_id": client_id,
                    "status": profile.status.value,
                    "rounds_participated": profile.rounds_participated,
                    "success_rate": profile.success_rate,
                    "reputation_score": profile.reputation_score,
                    "reward_balance": profile.reward_balance,
                    "fairness_weight": self.fairness_weights.get(client_id, 0),
                    "cluster": self._get_client_cluster(client_id)
                }
            else:
                return {"error": "Client not found"}
        else:
            return {
                client_id: {
                    "status": profile.status.value,
                    "rounds_participated": profile.rounds_participated,
                    "success_rate": profile.success_rate,
                    "reputation_score": profile.reputation_score
                }
                for client_id, profile in self.client_profiles.items()
            }