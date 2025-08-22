"""
Model Aggregation Strategies for Federated Learning

This module implements various model aggregation algorithms for combining
model updates from distributed clients in federated learning.
"""

import logging
import numpy as np
import torch
import tensorflow as tf
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import math
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Available aggregation strategies"""
    FEDAVG = "fedavg"           # Federated Averaging
    FEDPROX = "fedprox"         # Federated Proximal
    FEDOPT = "fedopt"           # Federated Optimization
    SCAFFOLD = "scaffold"       # SCAFFOLD
    FEDNOVA = "fednova"         # FedNova
    FEDADAGRAD = "fedadagrad"   # FedAdaGrad
    FEDYOGI = "fedyogi"         # FedYogi
    FEDADAM = "fedadam"         # FedAdam
    CLUSTERED = "clustered"     # Clustered FL
    PERSONALIZED = "personalized"  # Personalized FL
    ASYNC_FEDAVG = "async_fedavg"  # Asynchronous FedAvg
    BYZANTINE_ROBUST = "byzantine_robust"  # Byzantine-robust aggregation


@dataclass
class ClientUpdate:
    """Client model update information"""
    client_id: str
    weights: Dict[str, Any]
    data_size: int
    loss: float
    learning_rate: float
    local_epochs: int
    gradient_norm: Optional[float] = None
    timestamp: Optional[float] = None
    privacy_cost: float = 0.0
    staleness: int = 0  # For asynchronous FL


@dataclass
class AggregationResult:
    """Result of model aggregation"""
    aggregated_weights: Dict[str, Any]
    participating_clients: List[str]
    total_data_size: int
    weighted_loss: float
    convergence_metric: float
    aggregation_time: float
    quality_score: float


class ModelAggregator:
    """
    Advanced model aggregation engine for federated learning
    
    Features:
    - Multiple aggregation strategies (FedAvg, FedProx, FedOpt, etc.)
    - Byzantine-robust aggregation
    - Personalized and clustered federated learning
    - Asynchronous aggregation support
    - Adaptive client weighting
    - Quality assessment and filtering
    """
    
    def __init__(
        self,
        strategy: str = "fedavg",
        config: Optional[Dict] = None
    ):
        self.strategy = AggregationStrategy(strategy)
        self.config = config or {}
        
        # Strategy-specific parameters
        self.mu = self.config.get("mu", 0.01)  # FedProx regularization
        self.server_learning_rate = self.config.get("server_learning_rate", 1.0)
        self.momentum = self.config.get("momentum", 0.9)
        self.adaptive_lr = self.config.get("adaptive_lr", True)
        
        # Quality control
        self.quality_threshold = self.config.get("quality_threshold", 0.1)
        self.byzantine_tolerance = self.config.get("byzantine_tolerance", 0.2)
        self.gradient_clipping = self.config.get("gradient_clipping", True)
        self.max_gradient_norm = self.config.get("max_gradient_norm", 10.0)
        
        # Asynchronous settings
        self.staleness_threshold = self.config.get("staleness_threshold", 5)
        self.async_buffer_size = self.config.get("async_buffer_size", 100)
        
        # State for adaptive algorithms
        self.server_optimizer_state = {}
        self.client_clusters = defaultdict(list)
        self.aggregation_history = []
        
        logger.info(f"Model aggregator initialized with {strategy} strategy")
    
    async def aggregate_models(
        self,
        client_updates: List[Dict],
        client_info: List[Any],
        global_model_weights: Optional[Dict] = None
    ) -> AggregationResult:
        """Main aggregation function"""
        import time
        start_time = time.time()
        
        try:
            # Convert to standardized format
            updates = self._prepare_updates(client_updates, client_info)
            
            # Quality filtering
            filtered_updates = self._filter_updates(updates)
            
            if not filtered_updates:
                raise ValueError("No valid updates after filtering")
            
            # Select aggregation method
            if self.strategy == AggregationStrategy.FEDAVG:
                result_weights = await self._fedavg_aggregate(filtered_updates)
            elif self.strategy == AggregationStrategy.FEDPROX:
                result_weights = await self._fedprox_aggregate(filtered_updates, global_model_weights)
            elif self.strategy == AggregationStrategy.FEDOPT:
                result_weights = await self._fedopt_aggregate(filtered_updates, global_model_weights)
            elif self.strategy == AggregationStrategy.SCAFFOLD:
                result_weights = await self._scaffold_aggregate(filtered_updates)
            elif self.strategy == AggregationStrategy.FEDNOVA:
                result_weights = await self._fednova_aggregate(filtered_updates)
            elif self.strategy == AggregationStrategy.FEDADAGRAD:
                result_weights = await self._fedadagrad_aggregate(filtered_updates, global_model_weights)
            elif self.strategy == AggregationStrategy.FEDYOGI:
                result_weights = await self._fedyogi_aggregate(filtered_updates, global_model_weights)
            elif self.strategy == AggregationStrategy.FEDADAM:
                result_weights = await self._fedadam_aggregate(filtered_updates, global_model_weights)
            elif self.strategy == AggregationStrategy.CLUSTERED:
                result_weights = await self._clustered_aggregate(filtered_updates)
            elif self.strategy == AggregationStrategy.PERSONALIZED:
                result_weights = await self._personalized_aggregate(filtered_updates)
            elif self.strategy == AggregationStrategy.ASYNC_FEDAVG:
                result_weights = await self._async_fedavg_aggregate(filtered_updates)
            elif self.strategy == AggregationStrategy.BYZANTINE_ROBUST:
                result_weights = await self._byzantine_robust_aggregate(filtered_updates)
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.strategy}")
            
            # Calculate metrics
            aggregation_time = time.time() - start_time
            
            # Create result
            result = AggregationResult(
                aggregated_weights=result_weights,
                participating_clients=[u.client_id for u in filtered_updates],
                total_data_size=sum(u.data_size for u in filtered_updates),
                weighted_loss=self._calculate_weighted_loss(filtered_updates),
                convergence_metric=self._calculate_convergence_metric(filtered_updates),
                aggregation_time=aggregation_time,
                quality_score=self._calculate_quality_score(filtered_updates)
            )
            
            # Update aggregation history
            self.aggregation_history.append(result)
            
            logger.info(f"Aggregation completed: {len(filtered_updates)} clients, "
                       f"quality={result.quality_score:.3f}, time={aggregation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise
    
    def _prepare_updates(self, client_updates: List[Dict], client_info: List[Any]) -> List[ClientUpdate]:
        """Convert client updates to standardized format"""
        updates = []
        
        for update_data, client in zip(client_updates, client_info):
            try:
                # Extract gradient norm if available
                gradient_norm = None
                if "gradient_norm" in update_data:
                    gradient_norm = update_data["gradient_norm"]
                elif "model_weights" in update_data:
                    gradient_norm = self._calculate_gradient_norm(update_data["model_weights"])
                
                # Create update object
                client_update = ClientUpdate(
                    client_id=client.client_id if hasattr(client, 'client_id') else str(client),
                    weights=update_data["model_weights"],
                    data_size=update_data.get("data_size", client.data_size if hasattr(client, 'data_size') else 1),
                    loss=update_data.get("loss", float('inf')),
                    learning_rate=update_data.get("learning_rate", 0.01),
                    local_epochs=update_data.get("local_epochs", 1),
                    gradient_norm=gradient_norm,
                    timestamp=update_data.get("timestamp"),
                    privacy_cost=update_data.get("privacy_cost", 0.0)
                )
                
                updates.append(client_update)
                
            except Exception as e:
                logger.warning(f"Failed to prepare update: {e}")
                continue
        
        return updates
    
    def _calculate_gradient_norm(self, weights: Dict) -> float:
        """Calculate gradient norm from model weights"""
        try:
            total_norm = 0.0
            
            for layer_name, weight in weights.items():
                if isinstance(weight, (list, np.ndarray)):
                    weight_array = np.array(weight)
                    total_norm += np.sum(weight_array ** 2)
                elif isinstance(weight, dict) and "values" in weight:
                    # Handle quantized weights
                    weight_array = np.array(weight["values"])
                    total_norm += np.sum(weight_array ** 2)
            
            return np.sqrt(total_norm)
            
        except Exception as e:
            logger.warning(f"Gradient norm calculation failed: {e}")
            return 0.0
    
    def _filter_updates(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Filter updates based on quality metrics"""
        filtered = []
        
        for update in updates:
            # Check for valid loss
            if math.isnan(update.loss) or math.isinf(update.loss):
                logger.warning(f"Invalid loss from client {update.client_id}")
                continue
            
            # Check gradient norm
            if (update.gradient_norm is not None and 
                self.gradient_clipping and 
                update.gradient_norm > self.max_gradient_norm):
                logger.warning(f"Large gradient norm from client {update.client_id}: {update.gradient_norm}")
                # Clip gradients instead of rejecting
                update = self._clip_gradients(update)
            
            # Check data size
            if update.data_size <= 0:
                logger.warning(f"Invalid data size from client {update.client_id}")
                continue
            
            filtered.append(update)
        
        # Byzantine detection
        if self.strategy == AggregationStrategy.BYZANTINE_ROBUST:
            filtered = self._detect_byzantine_clients(filtered)
        
        return filtered
    
    def _clip_gradients(self, update: ClientUpdate) -> ClientUpdate:
        """Clip gradients to prevent instability"""
        try:
            clipped_weights = {}
            scale_factor = self.max_gradient_norm / update.gradient_norm
            
            for layer_name, weight in update.weights.items():
                if isinstance(weight, (list, np.ndarray)):
                    weight_array = np.array(weight)
                    clipped_weights[layer_name] = (weight_array * scale_factor).tolist()
                else:
                    clipped_weights[layer_name] = weight
            
            update.weights = clipped_weights
            update.gradient_norm = self.max_gradient_norm
            
            return update
            
        except Exception as e:
            logger.warning(f"Gradient clipping failed: {e}")
            return update
    
    def _detect_byzantine_clients(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """Detect and filter Byzantine (malicious) clients"""
        if len(updates) <= 2:
            return updates
        
        try:
            # Calculate pairwise distances between updates
            distances = []
            for i, update_i in enumerate(updates):
                for j, update_j in enumerate(updates[i+1:], i+1):
                    distance = self._calculate_model_distance(update_i.weights, update_j.weights)
                    distances.append((i, j, distance))
            
            # Find outliers using median distance
            distances.sort(key=lambda x: x[2])
            median_distance = distances[len(distances)//2][2]
            
            # Mark clients with consistently high distances as Byzantine
            client_distances = defaultdict(list)
            for i, j, distance in distances:
                client_distances[i].append(distance)
                client_distances[j].append(distance)
            
            byzantine_clients = set()
            for client_idx, client_distances_list in client_distances.items():
                avg_distance = np.mean(client_distances_list)
                if avg_distance > median_distance * (1 + self.byzantine_tolerance):
                    byzantine_clients.add(client_idx)
            
            # Filter out Byzantine clients
            filtered_updates = []
            for i, update in enumerate(updates):
                if i not in byzantine_clients:
                    filtered_updates.append(update)
                else:
                    logger.warning(f"Detected Byzantine client: {update.client_id}")
            
            return filtered_updates
            
        except Exception as e:
            logger.warning(f"Byzantine detection failed: {e}")
            return updates
    
    def _calculate_model_distance(self, weights1: Dict, weights2: Dict) -> float:
        """Calculate distance between two model weight dictionaries"""
        try:
            total_distance = 0.0
            
            for layer_name in weights1.keys():
                if layer_name in weights2:
                    w1 = np.array(weights1[layer_name])
                    w2 = np.array(weights2[layer_name])
                    
                    if w1.shape == w2.shape:
                        distance = np.linalg.norm(w1 - w2)
                        total_distance += distance
            
            return total_distance
            
        except Exception as e:
            logger.warning(f"Model distance calculation failed: {e}")
            return 0.0
    
    async def _fedavg_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """Federated Averaging (FedAvg) aggregation"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Calculate weights based on data size
        total_data_size = sum(update.data_size for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get layer names from first update
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            layer_weights = []
            client_weights = []
            
            for update in updates:
                if layer_name in update.weights:
                    weight = update.weights[layer_name]
                    if isinstance(weight, (list, np.ndarray)):
                        layer_weights.append(np.array(weight))
                        client_weights.append(update.data_size / total_data_size)
            
            if layer_weights:
                # Weighted average
                weighted_sum = np.zeros_like(layer_weights[0])
                for weight, client_weight in zip(layer_weights, client_weights):
                    weighted_sum += weight * client_weight
                
                aggregated_weights[layer_name] = weighted_sum.tolist()
        
        return aggregated_weights
    
    async def _fedprox_aggregate(self, updates: List[ClientUpdate], global_weights: Optional[Dict]) -> Dict[str, Any]:
        """FedProx aggregation with proximal regularization"""
        # FedProx uses same aggregation as FedAvg but with proximal term during local training
        # The aggregation step is identical to FedAvg
        return await self._fedavg_aggregate(updates)
    
    async def _fedopt_aggregate(self, updates: List[ClientUpdate], global_weights: Optional[Dict]) -> Dict[str, Any]:
        """Federated Optimization with server-side adaptive optimization"""
        if not global_weights:
            return await self._fedavg_aggregate(updates)
        
        # Calculate client drift (pseudo-gradient)
        client_drift = await self._fedavg_aggregate(updates)
        
        # Convert to gradient-like update
        gradient = {}
        for layer_name in client_drift.keys():
            if layer_name in global_weights:
                global_weight = np.array(global_weights[layer_name])
                client_weight = np.array(client_drift[layer_name])
                gradient[layer_name] = (client_weight - global_weight).tolist()
        
        # Apply server optimizer (using Adam-like update)
        return await self._apply_server_optimizer(global_weights, gradient, "adam")
    
    async def _scaffold_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """SCAFFOLD aggregation with control variates"""
        # Simplified SCAFFOLD - in practice, would need control variates from clients
        # For now, use variance-reduced averaging
        
        if len(updates) < 2:
            return await self._fedavg_aggregate(updates)
        
        # Calculate standard FedAvg
        fedavg_result = await self._fedavg_aggregate(updates)
        
        # Apply variance reduction (simplified)
        # In full SCAFFOLD, this would use control variates
        variance_factor = 1.0 - (1.0 / len(updates))
        
        adjusted_weights = {}
        for layer_name, weight in fedavg_result.items():
            weight_array = np.array(weight)
            adjusted_weights[layer_name] = (weight_array * variance_factor).tolist()
        
        return adjusted_weights
    
    async def _fednova_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """FedNova aggregation with normalized averaging"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Calculate effective step sizes
        total_tau = 0
        for update in updates:
            tau = update.local_epochs * update.data_size
            total_tau += tau
        
        # Normalized aggregation
        aggregated_weights = {}
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            weighted_sum = None
            
            for update in updates:
                if layer_name in update.weights:
                    weight = np.array(update.weights[layer_name])
                    tau = update.local_epochs * update.data_size
                    
                    if weighted_sum is None:
                        weighted_sum = np.zeros_like(weight)
                    
                    weighted_sum += weight * (tau / total_tau)
            
            if weighted_sum is not None:
                aggregated_weights[layer_name] = weighted_sum.tolist()
        
        return aggregated_weights
    
    async def _fedadagrad_aggregate(self, updates: List[ClientUpdate], global_weights: Optional[Dict]) -> Dict[str, Any]:
        """FedAdaGrad aggregation with adaptive learning rates"""
        if not global_weights:
            return await self._fedavg_aggregate(updates)
        
        # Calculate pseudo-gradient
        client_average = await self._fedavg_aggregate(updates)
        gradient = {}
        for layer_name in client_average.keys():
            if layer_name in global_weights:
                global_weight = np.array(global_weights[layer_name])
                client_weight = np.array(client_average[layer_name])
                gradient[layer_name] = (client_weight - global_weight).tolist()
        
        return await self._apply_server_optimizer(global_weights, gradient, "adagrad")
    
    async def _fedyogi_aggregate(self, updates: List[ClientUpdate], global_weights: Optional[Dict]) -> Dict[str, Any]:
        """FedYogi aggregation with Yogi optimizer"""
        if not global_weights:
            return await self._fedavg_aggregate(updates)
        
        client_average = await self._fedavg_aggregate(updates)
        gradient = {}
        for layer_name in client_average.keys():
            if layer_name in global_weights:
                global_weight = np.array(global_weights[layer_name])
                client_weight = np.array(client_average[layer_name])
                gradient[layer_name] = (client_weight - global_weight).tolist()
        
        return await self._apply_server_optimizer(global_weights, gradient, "yogi")
    
    async def _fedadam_aggregate(self, updates: List[ClientUpdate], global_weights: Optional[Dict]) -> Dict[str, Any]:
        """FedAdam aggregation with Adam optimizer"""
        if not global_weights:
            return await self._fedavg_aggregate(updates)
        
        client_average = await self._fedavg_aggregate(updates)
        gradient = {}
        for layer_name in client_average.keys():
            if layer_name in global_weights:
                global_weight = np.array(global_weights[layer_name])
                client_weight = np.array(client_average[layer_name])
                gradient[layer_name] = (client_weight - global_weight).tolist()
        
        return await self._apply_server_optimizer(global_weights, gradient, "adam")
    
    async def _clustered_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """Clustered federated learning aggregation"""
        if len(updates) <= 2:
            return await self._fedavg_aggregate(updates)
        
        # Simple clustering based on model similarity
        clusters = self._cluster_clients(updates)
        
        if len(clusters) == 1:
            return await self._fedavg_aggregate(updates)
        
        # Aggregate within clusters and then combine
        cluster_aggregates = []
        cluster_weights = []
        
        for cluster_updates in clusters:
            cluster_agg = await self._fedavg_aggregate(cluster_updates)
            cluster_size = sum(u.data_size for u in cluster_updates)
            cluster_aggregates.append(cluster_agg)
            cluster_weights.append(cluster_size)
        
        # Combine cluster aggregates
        total_weight = sum(cluster_weights)
        final_aggregate = {}
        
        layer_names = list(cluster_aggregates[0].keys())
        for layer_name in layer_names:
            weighted_sum = None
            
            for cluster_agg, weight in zip(cluster_aggregates, cluster_weights):
                if layer_name in cluster_agg:
                    layer_weight = np.array(cluster_agg[layer_name])
                    
                    if weighted_sum is None:
                        weighted_sum = np.zeros_like(layer_weight)
                    
                    weighted_sum += layer_weight * (weight / total_weight)
            
            if weighted_sum is not None:
                final_aggregate[layer_name] = weighted_sum.tolist()
        
        return final_aggregate
    
    def _cluster_clients(self, updates: List[ClientUpdate], num_clusters: int = 2) -> List[List[ClientUpdate]]:
        """Simple clustering of clients based on model similarity"""
        try:
            if len(updates) <= num_clusters:
                return [[update] for update in updates]
            
            # Calculate pairwise distances
            distances = np.zeros((len(updates), len(updates)))
            
            for i in range(len(updates)):
                for j in range(i+1, len(updates)):
                    dist = self._calculate_model_distance(updates[i].weights, updates[j].weights)
                    distances[i][j] = dist
                    distances[j][i] = dist
            
            # Simple k-means clustering
            clusters = [[] for _ in range(num_clusters)]
            cluster_centers = list(range(0, len(updates), len(updates) // num_clusters))[:num_clusters]
            
            for i, update in enumerate(updates):
                # Find closest cluster center
                closest_cluster = 0
                min_distance = float('inf')
                
                for cluster_idx, center_idx in enumerate(cluster_centers):
                    distance = distances[i][center_idx]
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster = cluster_idx
                
                clusters[closest_cluster].append(update)
            
            # Remove empty clusters
            clusters = [cluster for cluster in clusters if cluster]
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return [updates]  # Return single cluster
    
    async def _personalized_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """Personalized federated learning aggregation"""
        # For personalized FL, we typically maintain both global and local models
        # Here we return a global model that serves as a starting point for personalization
        
        # Use similarity-weighted averaging
        similarity_weights = self._calculate_similarity_weights(updates)
        
        aggregated_weights = {}
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            weighted_sum = None
            total_weight = 0
            
            for i, update in enumerate(updates):
                if layer_name in update.weights:
                    weight = np.array(update.weights[layer_name])
                    sim_weight = similarity_weights[i]
                    
                    if weighted_sum is None:
                        weighted_sum = np.zeros_like(weight)
                    
                    weighted_sum += weight * sim_weight
                    total_weight += sim_weight
            
            if weighted_sum is not None and total_weight > 0:
                aggregated_weights[layer_name] = (weighted_sum / total_weight).tolist()
        
        return aggregated_weights
    
    def _calculate_similarity_weights(self, updates: List[ClientUpdate]) -> List[float]:
        """Calculate similarity-based weights for personalized aggregation"""
        if len(updates) <= 1:
            return [1.0] * len(updates)
        
        # Calculate average model
        avg_model = {}
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            layer_weights = []
            for update in updates:
                if layer_name in update.weights:
                    layer_weights.append(np.array(update.weights[layer_name]))
            
            if layer_weights:
                avg_model[layer_name] = np.mean(layer_weights, axis=0)
        
        # Calculate similarity to average
        similarities = []
        for update in updates:
            similarity = 0
            for layer_name in layer_names:
                if layer_name in update.weights and layer_name in avg_model:
                    weight = np.array(update.weights[layer_name])
                    avg_weight = avg_model[layer_name]
                    
                    # Cosine similarity
                    dot_product = np.sum(weight * avg_weight)
                    norm_product = np.linalg.norm(weight) * np.linalg.norm(avg_weight)
                    
                    if norm_product > 0:
                        similarity += dot_product / norm_product
            
            similarities.append(max(similarity / len(layer_names), 0.1))  # Minimum weight
        
        return similarities
    
    async def _async_fedavg_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """Asynchronous FedAvg with staleness handling"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Apply staleness penalty
        weighted_updates = []
        for update in updates:
            staleness_factor = 1.0 / (1.0 + update.staleness)
            effective_weight = update.data_size * staleness_factor
            
            weighted_update = ClientUpdate(
                client_id=update.client_id,
                weights=update.weights,
                data_size=int(effective_weight),
                loss=update.loss,
                learning_rate=update.learning_rate,
                local_epochs=update.local_epochs,
                staleness=update.staleness
            )
            weighted_updates.append(weighted_update)
        
        return await self._fedavg_aggregate(weighted_updates)
    
    async def _byzantine_robust_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, Any]:
        """Byzantine-robust aggregation using coordinate-wise median"""
        if len(updates) <= 2:
            return await self._fedavg_aggregate(updates)
        
        # Use coordinate-wise median for robustness
        robust_weights = {}
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            layer_weights = []
            
            for update in updates:
                if layer_name in update.weights:
                    weight = np.array(update.weights[layer_name])
                    layer_weights.append(weight)
            
            if layer_weights:
                # Stack weights and take coordinate-wise median
                stacked_weights = np.stack(layer_weights, axis=0)
                median_weights = np.median(stacked_weights, axis=0)
                robust_weights[layer_name] = median_weights.tolist()
        
        return robust_weights
    
    async def _apply_server_optimizer(
        self,
        global_weights: Dict,
        gradients: Dict,
        optimizer_type: str
    ) -> Dict[str, Any]:
        """Apply server-side optimization"""
        if optimizer_type not in self.server_optimizer_state:
            self.server_optimizer_state[optimizer_type] = {}
        
        state = self.server_optimizer_state[optimizer_type]
        updated_weights = {}
        
        for layer_name in gradients.keys():
            if layer_name in global_weights:
                weight = np.array(global_weights[layer_name])
                grad = np.array(gradients[layer_name])
                
                if layer_name not in state:
                    state[layer_name] = {}
                
                layer_state = state[layer_name]
                
                if optimizer_type == "adam":
                    updated_weight = self._adam_update(weight, grad, layer_state)
                elif optimizer_type == "adagrad":
                    updated_weight = self._adagrad_update(weight, grad, layer_state)
                elif optimizer_type == "yogi":
                    updated_weight = self._yogi_update(weight, grad, layer_state)
                else:
                    updated_weight = weight + self.server_learning_rate * grad
                
                updated_weights[layer_name] = updated_weight.tolist()
        
        return updated_weights
    
    def _adam_update(self, weight: np.ndarray, grad: np.ndarray, state: Dict) -> np.ndarray:
        """Adam optimizer update"""
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        if 'm' not in state:
            state['m'] = np.zeros_like(weight)
            state['v'] = np.zeros_like(weight)
            state['t'] = 0
        
        state['t'] += 1
        state['m'] = beta1 * state['m'] + (1 - beta1) * grad
        state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
        
        m_hat = state['m'] / (1 - beta1 ** state['t'])
        v_hat = state['v'] / (1 - beta2 ** state['t'])
        
        return weight + self.server_learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def _adagrad_update(self, weight: np.ndarray, grad: np.ndarray, state: Dict) -> np.ndarray:
        """AdaGrad optimizer update"""
        epsilon = 1e-8
        
        if 'sum_squared_grad' not in state:
            state['sum_squared_grad'] = np.zeros_like(weight)
        
        state['sum_squared_grad'] += grad ** 2
        
        return weight + self.server_learning_rate * grad / (np.sqrt(state['sum_squared_grad']) + epsilon)
    
    def _yogi_update(self, weight: np.ndarray, grad: np.ndarray, state: Dict) -> np.ndarray:
        """Yogi optimizer update"""
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        if 'm' not in state:
            state['m'] = np.zeros_like(weight)
            state['v'] = np.zeros_like(weight)
            state['t'] = 0
        
        state['t'] += 1
        state['m'] = beta1 * state['m'] + (1 - beta1) * grad
        
        # Yogi update for second moment
        v_t = state['v'] - (1 - beta2) * (grad ** 2) * np.sign(state['v'] - grad ** 2)
        state['v'] = v_t
        
        m_hat = state['m'] / (1 - beta1 ** state['t'])
        v_hat = state['v'] / (1 - beta2 ** state['t'])
        
        return weight + self.server_learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def _calculate_weighted_loss(self, updates: List[ClientUpdate]) -> float:
        """Calculate weighted average loss"""
        if not updates:
            return float('inf')
        
        total_weighted_loss = 0
        total_weight = 0
        
        for update in updates:
            weight = update.data_size
            total_weighted_loss += update.loss * weight
            total_weight += weight
        
        return total_weighted_loss / total_weight if total_weight > 0 else float('inf')
    
    def _calculate_convergence_metric(self, updates: List[ClientUpdate]) -> float:
        """Calculate convergence metric based on client similarity"""
        if len(updates) <= 1:
            return 0.0
        
        try:
            # Calculate pairwise similarities
            similarities = []
            
            for i in range(len(updates)):
                for j in range(i+1, len(updates)):
                    similarity = self._calculate_model_similarity(
                        updates[i].weights, updates[j].weights
                    )
                    similarities.append(similarity)
            
            # Higher similarity means better convergence
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Convergence metric calculation failed: {e}")
            return 0.0
    
    def _calculate_model_similarity(self, weights1: Dict, weights2: Dict) -> float:
        """Calculate cosine similarity between two models"""
        try:
            total_similarity = 0
            layer_count = 0
            
            for layer_name in weights1.keys():
                if layer_name in weights2:
                    w1 = np.array(weights1[layer_name]).flatten()
                    w2 = np.array(weights2[layer_name]).flatten()
                    
                    if len(w1) == len(w2) and len(w1) > 0:
                        dot_product = np.dot(w1, w2)
                        norm_product = np.linalg.norm(w1) * np.linalg.norm(w2)
                        
                        if norm_product > 0:
                            similarity = dot_product / norm_product
                            total_similarity += similarity
                            layer_count += 1
            
            return total_similarity / layer_count if layer_count > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Model similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_quality_score(self, updates: List[ClientUpdate]) -> float:
        """Calculate overall quality score for the aggregation"""
        if not updates:
            return 0.0
        
        # Factors: loss improvement, gradient norms, data diversity
        loss_scores = []
        gradient_scores = []
        
        for update in updates:
            # Loss score (lower is better, so invert)
            loss_score = 1.0 / (1.0 + update.loss) if update.loss > 0 else 1.0
            loss_scores.append(loss_score)
            
            # Gradient norm score (moderate norms are better)
            if update.gradient_norm is not None:
                # Penalize both very small and very large norms
                norm_score = np.exp(-((update.gradient_norm - 1.0) ** 2) / 2.0)
                gradient_scores.append(norm_score)
        
        quality_components = []
        
        if loss_scores:
            quality_components.append(np.mean(loss_scores))
        
        if gradient_scores:
            quality_components.append(np.mean(gradient_scores))
        
        # Data diversity score
        data_sizes = [update.data_size for update in updates]
        if len(set(data_sizes)) > 1:  # Diverse data sizes
            diversity_score = 1.0 - (np.std(data_sizes) / np.mean(data_sizes))
            quality_components.append(max(diversity_score, 0.0))
        
        return np.mean(quality_components) if quality_components else 0.5
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        if not self.aggregation_history:
            return {"total_rounds": 0}
        
        recent_results = self.aggregation_history[-10:]  # Last 10 rounds
        
        return {
            "total_rounds": len(self.aggregation_history),
            "strategy": self.strategy.value,
            "recent_quality_score": np.mean([r.quality_score for r in recent_results]),
            "recent_convergence": np.mean([r.convergence_metric for r in recent_results]),
            "recent_aggregation_time": np.mean([r.aggregation_time for r in recent_results]),
            "total_clients_seen": len(set(
                client_id 
                for result in self.aggregation_history 
                for client_id in result.participating_clients
            ))
        }