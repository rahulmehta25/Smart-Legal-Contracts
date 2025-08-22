"""
Federated Learning Central Aggregation Server

This module implements the central server for federated learning that coordinates
training across multiple edge clients, aggregates model updates, and manages
the global model state.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_federated as tff
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from .aggregation import ModelAggregator, AggregationStrategy
from .privacy import PrivacyEngine
from .communication import SecureCommunication
from .monitoring import FederatedMonitor

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """Information about a federated client"""
    client_id: str
    last_seen: datetime
    model_version: int
    data_size: int
    compute_capability: float
    network_quality: float
    privacy_budget: float
    is_active: bool = True
    reputation_score: float = 1.0
    contribution_history: List[float] = field(default_factory=list)


@dataclass
class TrainingRound:
    """Information about a training round"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participating_clients: List[str] = field(default_factory=list)
    model_updates_received: int = 0
    convergence_metric: Optional[float] = None
    privacy_cost: float = 0.0
    communication_cost: float = 0.0


class FederatedServer:
    """
    Central server for federated learning coordination
    
    Features:
    - Client management and selection
    - Model aggregation and versioning
    - Privacy-preserving mechanisms
    - Adaptive learning strategies
    - Real-time monitoring
    """
    
    def __init__(
        self,
        model_fn,
        aggregation_strategy: str = "fedavg",
        privacy_engine: Optional[PrivacyEngine] = None,
        secure_comm: Optional[SecureCommunication] = None,
        monitor: Optional[FederatedMonitor] = None,
        config: Optional[Dict] = None
    ):
        self.model_fn = model_fn
        self.config = config or {}
        
        # Initialize components
        self.aggregator = ModelAggregator(aggregation_strategy)
        self.privacy_engine = privacy_engine or PrivacyEngine()
        self.secure_comm = secure_comm or SecureCommunication()
        self.monitor = monitor or FederatedMonitor()
        
        # Server state
        self.current_round = 0
        self.global_model = None
        self.clients: Dict[str, ClientInfo] = {}
        self.training_history: List[TrainingRound] = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Configuration
        self.min_clients = self.config.get("min_clients", 3)
        self.max_clients = self.config.get("max_clients", 100)
        self.clients_per_round = self.config.get("clients_per_round", 10)
        self.convergence_threshold = self.config.get("convergence_threshold", 0.001)
        self.max_rounds = self.config.get("max_rounds", 100)
        self.client_timeout = self.config.get("client_timeout", 300)  # seconds
        
        # Privacy settings
        self.privacy_budget_per_round = self.config.get("privacy_budget_per_round", 1.0)
        self.total_privacy_budget = self.config.get("total_privacy_budget", 10.0)
        self.current_privacy_spent = 0.0
        
        # Adaptive settings
        self.adaptive_client_selection = self.config.get("adaptive_client_selection", True)
        self.dynamic_learning_rate = self.config.get("dynamic_learning_rate", True)
        self.early_stopping = self.config.get("early_stopping", True)
        
        logger.info(f"Federated server initialized with {aggregation_strategy} aggregation")
    
    def initialize_model(self, sample_data: Optional[Any] = None) -> None:
        """Initialize the global model"""
        try:
            if sample_data is not None:
                # Initialize with sample data
                self.global_model = self.model_fn(sample_data)
            else:
                # Initialize with default parameters
                self.global_model = self.model_fn()
            
            logger.info("Global model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize global model: {e}")
            raise
    
    async def register_client(
        self, 
        client_id: str, 
        capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register a new client"""
        try:
            client_info = ClientInfo(
                client_id=client_id,
                last_seen=datetime.now(),
                model_version=0,
                data_size=capabilities.get("data_size", 0),
                compute_capability=capabilities.get("compute_capability", 1.0),
                network_quality=capabilities.get("network_quality", 1.0),
                privacy_budget=capabilities.get("privacy_budget", 5.0)
            )
            
            self.clients[client_id] = client_info
            
            # Secure communication setup
            comm_keys = await self.secure_comm.setup_client_keys(client_id)
            
            logger.info(f"Client {client_id} registered successfully")
            
            return {
                "status": "registered",
                "server_round": self.current_round,
                "communication_keys": comm_keys,
                "privacy_budget_allocated": client_info.privacy_budget
            }
            
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def unregister_client(self, client_id: str) -> Dict[str, Any]:
        """Unregister a client"""
        try:
            if client_id in self.clients:
                del self.clients[client_id]
                await self.secure_comm.cleanup_client_keys(client_id)
                logger.info(f"Client {client_id} unregistered")
                return {"status": "unregistered"}
            else:
                return {"status": "not_found"}
                
        except Exception as e:
            logger.error(f"Failed to unregister client {client_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def select_clients(self, round_num: int) -> List[str]:
        """Select clients for training round"""
        active_clients = [
            cid for cid, client in self.clients.items()
            if client.is_active and 
            (datetime.now() - client.last_seen).seconds < self.client_timeout and
            client.privacy_budget > 0
        ]
        
        if len(active_clients) < self.min_clients:
            logger.warning(f"Insufficient active clients: {len(active_clients)}")
            return []
        
        num_selected = min(self.clients_per_round, len(active_clients))
        
        if self.adaptive_client_selection:
            # Select clients based on multiple criteria
            return self._adaptive_client_selection(active_clients, num_selected)
        else:
            # Random selection
            return np.random.choice(active_clients, num_selected, replace=False).tolist()
    
    def _adaptive_client_selection(self, active_clients: List[str], num_selected: int) -> List[str]:
        """Advanced client selection based on multiple criteria"""
        # Calculate selection scores
        client_scores = {}
        
        for client_id in active_clients:
            client = self.clients[client_id]
            
            # Factors for selection
            data_factor = np.log1p(client.data_size)  # More data is better
            compute_factor = client.compute_capability  # Better compute is better
            network_factor = client.network_quality  # Better network is better
            reputation_factor = client.reputation_score  # Higher reputation is better
            privacy_factor = client.privacy_budget / 10.0  # More budget is better
            
            # Recent contribution factor
            if client.contribution_history:
                recent_contrib = np.mean(client.contribution_history[-5:])
            else:
                recent_contrib = 0.5  # Neutral for new clients
            
            # Combined score
            score = (
                0.3 * data_factor +
                0.2 * compute_factor +
                0.2 * network_factor +
                0.15 * reputation_factor +
                0.1 * privacy_factor +
                0.05 * recent_contrib
            )
            
            client_scores[client_id] = score
        
        # Select top clients with some randomness
        sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 70% deterministic, 30% random for diversity
        deterministic_count = int(num_selected * 0.7)
        random_count = num_selected - deterministic_count
        
        selected = [client_id for client_id, _ in sorted_clients[:deterministic_count]]
        
        if random_count > 0:
            remaining = [client_id for client_id, _ in sorted_clients[deterministic_count:]]
            if len(remaining) >= random_count:
                random_selected = np.random.choice(remaining, random_count, replace=False)
                selected.extend(random_selected)
            else:
                selected.extend(remaining)
        
        return selected[:num_selected]
    
    async def start_training_round(self) -> Dict[str, Any]:
        """Start a new training round"""
        if not self.global_model:
            raise ValueError("Global model not initialized")
        
        self.current_round += 1
        round_info = TrainingRound(
            round_id=self.current_round,
            start_time=datetime.now()
        )
        
        # Select clients for this round
        selected_clients = self.select_clients(self.current_round)
        
        if not selected_clients:
            logger.warning("No clients selected for training round")
            return {"status": "no_clients", "round": self.current_round}
        
        round_info.participating_clients = selected_clients
        
        # Prepare model for distribution
        model_weights = self._get_model_weights()
        
        # Apply privacy mechanisms if needed
        if self.privacy_engine.differential_privacy_enabled:
            model_weights = await self.privacy_engine.add_model_noise(
                model_weights, self.privacy_budget_per_round
            )
        
        # Send training instructions to selected clients
        training_tasks = []
        for client_id in selected_clients:
            task = self._send_training_instruction(client_id, model_weights, round_info)
            training_tasks.append(task)
        
        # Wait for all clients to acknowledge
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Update client statuses
        successful_clients = []
        for client_id, result in zip(selected_clients, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to send training instruction to {client_id}: {result}")
                self.clients[client_id].is_active = False
            else:
                successful_clients.append(client_id)
        
        round_info.participating_clients = successful_clients
        self.training_history.append(round_info)
        
        logger.info(f"Training round {self.current_round} started with {len(successful_clients)} clients")
        
        return {
            "status": "started",
            "round": self.current_round,
            "participating_clients": successful_clients,
            "expected_updates": len(successful_clients)
        }
    
    async def _send_training_instruction(
        self, 
        client_id: str, 
        model_weights: Dict, 
        round_info: TrainingRound
    ) -> None:
        """Send training instruction to a specific client"""
        client = self.clients[client_id]
        
        instruction = {
            "round_id": round_info.round_id,
            "model_weights": model_weights,
            "learning_rate": self._get_adaptive_learning_rate(),
            "epochs": self.config.get("local_epochs", 1),
            "batch_size": self.config.get("batch_size", 32),
            "privacy_budget": min(client.privacy_budget, self.privacy_budget_per_round),
            "timestamp": round_info.start_time.isoformat()
        }
        
        # Encrypt instruction
        encrypted_instruction = await self.secure_comm.encrypt_message(
            client_id, instruction
        )
        
        # Send via secure communication
        await self.secure_comm.send_to_client(client_id, encrypted_instruction)
        
        # Update client info
        client.last_seen = datetime.now()
        client.model_version = self.current_round
    
    def _get_adaptive_learning_rate(self) -> float:
        """Calculate adaptive learning rate based on training progress"""
        if not self.dynamic_learning_rate:
            return self.config.get("learning_rate", 0.01)
        
        base_lr = self.config.get("learning_rate", 0.01)
        
        # Decay based on round number
        decay_factor = 0.99 ** (self.current_round - 1)
        
        # Adjust based on convergence
        if len(self.training_history) >= 2:
            recent_rounds = self.training_history[-2:]
            if all(r.convergence_metric for r in recent_rounds):
                metrics = [r.convergence_metric for r in recent_rounds]
                if metrics[-1] > metrics[-2]:  # Convergence worsened
                    decay_factor *= 0.5  # Reduce learning rate more
        
        return base_lr * decay_factor
    
    async def receive_model_update(
        self, 
        client_id: str, 
        encrypted_update: bytes
    ) -> Dict[str, Any]:
        """Receive and process model update from client"""
        try:
            # Decrypt update
            update_data = await self.secure_comm.decrypt_message(client_id, encrypted_update)
            
            # Validate update
            if not self._validate_model_update(client_id, update_data):
                return {"status": "invalid", "message": "Update validation failed"}
            
            # Store update temporarily
            update_id = str(uuid.uuid4())
            await self._store_model_update(update_id, client_id, update_data)
            
            # Update client info
            client = self.clients[client_id]
            client.last_seen = datetime.now()
            client.privacy_budget -= update_data.get("privacy_cost", 0)
            
            # Check if round is complete
            current_round = self.training_history[-1]
            current_round.model_updates_received += 1
            
            logger.info(f"Received update from client {client_id} for round {current_round.round_id}")
            
            # Trigger aggregation if enough updates received
            if current_round.model_updates_received >= len(current_round.participating_clients):
                await self._trigger_aggregation(current_round.round_id)
            
            return {
                "status": "received",
                "update_id": update_id,
                "round": current_round.round_id
            }
            
        except Exception as e:
            logger.error(f"Failed to process update from client {client_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _validate_model_update(self, client_id: str, update_data: Dict) -> bool:
        """Validate model update from client"""
        try:
            # Check required fields
            required_fields = ["round_id", "model_weights", "data_size", "loss"]
            if not all(field in update_data for field in required_fields):
                return False
            
            # Check round consistency
            if update_data["round_id"] != self.current_round:
                logger.warning(f"Round mismatch from client {client_id}")
                return False
            
            # Check client authorization
            if client_id not in self.clients or not self.clients[client_id].is_active:
                return False
            
            # Validate model weights structure
            model_weights = update_data["model_weights"]
            if not isinstance(model_weights, dict):
                return False
            
            # Check for NaN or Inf values
            for layer_name, weights in model_weights.items():
                if isinstance(weights, (list, np.ndarray)):
                    weights = np.array(weights)
                    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                        logger.warning(f"Invalid weights from client {client_id}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for client {client_id}: {e}")
            return False
    
    async def _store_model_update(self, update_id: str, client_id: str, update_data: Dict) -> None:
        """Store model update temporarily for aggregation"""
        # In production, this would use a database or distributed storage
        # For now, we'll use in-memory storage
        if not hasattr(self, "_pending_updates"):
            self._pending_updates = {}
        
        self._pending_updates[update_id] = {
            "client_id": client_id,
            "update_data": update_data,
            "received_at": datetime.now()
        }
    
    async def _trigger_aggregation(self, round_id: int) -> None:
        """Trigger model aggregation for completed round"""
        logger.info(f"Starting aggregation for round {round_id}")
        
        try:
            # Collect all updates for this round
            round_updates = []
            for update_id, update_info in self._pending_updates.items():
                if update_info["update_data"]["round_id"] == round_id:
                    round_updates.append(update_info)
            
            if not round_updates:
                logger.warning(f"No updates found for round {round_id}")
                return
            
            # Perform aggregation
            aggregated_weights = await self.aggregator.aggregate_models(
                [u["update_data"] for u in round_updates],
                [self.clients[u["client_id"]] for u in round_updates]
            )
            
            # Apply privacy mechanisms
            if self.privacy_engine.secure_aggregation_enabled:
                aggregated_weights = await self.privacy_engine.secure_aggregate(
                    aggregated_weights, len(round_updates)
                )
            
            # Update global model
            self._update_global_model(aggregated_weights)
            
            # Calculate convergence metrics
            convergence_metric = self._calculate_convergence_metric(round_updates)
            
            # Update training history
            current_round = self.training_history[-1]
            current_round.end_time = datetime.now()
            current_round.convergence_metric = convergence_metric
            
            # Update privacy spending
            privacy_cost = sum(u["update_data"].get("privacy_cost", 0) for u in round_updates)
            current_round.privacy_cost = privacy_cost
            self.current_privacy_spent += privacy_cost
            
            # Update client reputation scores
            self._update_client_reputations(round_updates)
            
            # Clean up pending updates
            for update_id in list(self._pending_updates.keys()):
                if self._pending_updates[update_id]["update_data"]["round_id"] == round_id:
                    del self._pending_updates[update_id]
            
            # Log monitoring metrics
            await self.monitor.log_training_round(current_round, aggregated_weights)
            
            logger.info(f"Aggregation completed for round {round_id}, convergence: {convergence_metric}")
            
            # Check for early stopping
            if self._should_stop_early():
                logger.info("Early stopping triggered")
                await self.stop_training()
            
        except Exception as e:
            logger.error(f"Aggregation failed for round {round_id}: {e}")
            raise
    
    def _update_global_model(self, aggregated_weights: Dict) -> None:
        """Update the global model with aggregated weights"""
        try:
            if hasattr(self.global_model, 'set_weights'):
                # TensorFlow/Keras model
                weights_list = [aggregated_weights[name] for name in sorted(aggregated_weights.keys())]
                self.global_model.set_weights(weights_list)
            elif hasattr(self.global_model, 'state_dict'):
                # PyTorch model
                state_dict = {}
                for name, weight in aggregated_weights.items():
                    state_dict[name] = torch.tensor(weight)
                self.global_model.load_state_dict(state_dict)
            else:
                # Custom model format
                for name, weight in aggregated_weights.items():
                    setattr(self.global_model, name, weight)
                    
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
            raise
    
    def _calculate_convergence_metric(self, round_updates: List[Dict]) -> float:
        """Calculate convergence metric based on client updates"""
        try:
            # Use average loss as convergence metric
            losses = [u["update_data"]["loss"] for u in round_updates]
            return np.mean(losses)
            
        except Exception as e:
            logger.warning(f"Failed to calculate convergence metric: {e}")
            return float('inf')
    
    def _update_client_reputations(self, round_updates: List[Dict]) -> None:
        """Update client reputation scores based on contribution quality"""
        try:
            # Calculate contribution quality for each client
            losses = [u["update_data"]["loss"] for u in round_updates]
            mean_loss = np.mean(losses)
            
            for update_info in round_updates:
                client_id = update_info["client_id"]
                client_loss = update_info["update_data"]["loss"]
                
                # Better performance (lower loss) increases reputation
                contribution_score = max(0, 2 - (client_loss / mean_loss))
                
                client = self.clients[client_id]
                client.contribution_history.append(contribution_score)
                
                # Keep only recent history
                if len(client.contribution_history) > 10:
                    client.contribution_history = client.contribution_history[-10:]
                
                # Update reputation as moving average
                client.reputation_score = 0.9 * client.reputation_score + 0.1 * contribution_score
                
        except Exception as e:
            logger.warning(f"Failed to update client reputations: {e}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        if not self.early_stopping or len(self.training_history) < 3:
            return False
        
        # Check convergence
        recent_metrics = [r.convergence_metric for r in self.training_history[-3:] 
                         if r.convergence_metric is not None]
        
        if len(recent_metrics) >= 3:
            # Check if improvement has stagnated
            improvements = [recent_metrics[i-1] - recent_metrics[i] for i in range(1, len(recent_metrics))]
            avg_improvement = np.mean(improvements)
            
            if avg_improvement < self.convergence_threshold:
                return True
        
        # Check privacy budget
        if self.current_privacy_spent >= self.total_privacy_budget:
            logger.info("Privacy budget exhausted")
            return True
        
        # Check maximum rounds
        if self.current_round >= self.max_rounds:
            logger.info("Maximum rounds reached")
            return True
        
        return False
    
    def _get_model_weights(self) -> Dict:
        """Extract weights from global model"""
        try:
            if hasattr(self.global_model, 'get_weights'):
                # TensorFlow/Keras model
                weights = self.global_model.get_weights()
                return {f"layer_{i}": w.tolist() for i, w in enumerate(weights)}
            elif hasattr(self.global_model, 'state_dict'):
                # PyTorch model
                state_dict = self.global_model.state_dict()
                return {name: tensor.cpu().numpy().tolist() for name, tensor in state_dict.items()}
            else:
                # Custom model format
                return {name: getattr(self.global_model, name) for name in dir(self.global_model) 
                       if not name.startswith('_')}
                       
        except Exception as e:
            logger.error(f"Failed to extract model weights: {e}")
            raise
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get current server status"""
        active_clients = len([c for c in self.clients.values() if c.is_active])
        
        status = {
            "is_running": self.is_running,
            "current_round": self.current_round,
            "total_clients": len(self.clients),
            "active_clients": active_clients,
            "privacy_budget_used": self.current_privacy_spent,
            "privacy_budget_remaining": self.total_privacy_budget - self.current_privacy_spent,
            "aggregation_strategy": self.aggregator.strategy,
            "last_round_time": None
        }
        
        if self.training_history:
            last_round = self.training_history[-1]
            status["last_round_time"] = last_round.start_time.isoformat()
            status["last_convergence_metric"] = last_round.convergence_metric
        
        return status
    
    async def start_training(self) -> None:
        """Start the federated training process"""
        self.is_running = True
        logger.info("Federated training started")
        
        try:
            while self.is_running and not self._should_stop_early():
                # Start new training round
                round_result = await self.start_training_round()
                
                if round_result["status"] == "no_clients":
                    logger.warning("No clients available, waiting...")
                    await asyncio.sleep(30)  # Wait 30 seconds before retry
                    continue
                
                # Wait for round completion
                await self._wait_for_round_completion()
                
                # Brief pause between rounds
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("Federated training stopped")
    
    async def _wait_for_round_completion(self) -> None:
        """Wait for current round to complete"""
        if not self.training_history:
            return
        
        current_round = self.training_history[-1]
        timeout = datetime.now() + timedelta(seconds=self.client_timeout)
        
        while datetime.now() < timeout and not current_round.end_time:
            await asyncio.sleep(5)  # Check every 5 seconds
        
        if not current_round.end_time:
            logger.warning(f"Round {current_round.round_id} timed out")
            current_round.end_time = datetime.now()
    
    async def stop_training(self) -> None:
        """Stop the federated training process"""
        self.is_running = False
        logger.info("Federated training stop requested")
    
    def get_global_model(self):
        """Get the current global model"""
        return self.global_model
    
    def export_model(self, export_path: str, format: str = "tensorflow") -> None:
        """Export the global model"""
        try:
            if format == "tensorflow":
                tf.saved_model.save(self.global_model, export_path)
            elif format == "pytorch":
                torch.save(self.global_model.state_dict(), export_path)
            elif format == "onnx":
                # Export to ONNX format for edge deployment
                import onnx
                # Implementation depends on model type
                pass
            
            logger.info(f"Model exported to {export_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise