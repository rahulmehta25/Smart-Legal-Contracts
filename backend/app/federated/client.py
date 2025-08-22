"""
Federated Learning Edge Client

This module implements the edge client for federated learning that performs
local training on device data and communicates securely with the central server.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
import tensorflow_lite as tflite
import threading
from pathlib import Path
import psutil
import platform

from .privacy import PrivacyEngine
from .communication import SecureCommunication

logger = logging.getLogger(__name__)


@dataclass
class DeviceCapabilities:
    """Device capability information"""
    cpu_count: int
    memory_total: float  # GB
    memory_available: float  # GB
    storage_available: float  # GB
    gpu_available: bool
    gpu_memory: Optional[float] = None  # GB
    network_bandwidth: Optional[float] = None  # Mbps
    battery_level: Optional[float] = None  # Percentage
    is_charging: Optional[bool] = None


@dataclass
class TrainingMetrics:
    """Local training metrics"""
    round_id: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    initial_loss: float
    final_loss: float
    accuracy: Optional[float] = None
    training_time: float = 0.0
    data_size: int = 0
    privacy_cost: float = 0.0
    communication_cost: float = 0.0


class FederatedClient:
    """
    Edge client for federated learning
    
    Features:
    - Local model training with privacy preservation
    - Adaptive optimization based on device capabilities
    - Secure communication with server
    - Edge deployment optimization (TF Lite, ONNX)
    - Resource monitoring and management
    - Offline training capability
    """
    
    def __init__(
        self,
        client_id: str,
        model_fn: Callable,
        data_loader: DataLoader,
        server_endpoint: str,
        privacy_engine: Optional[PrivacyEngine] = None,
        secure_comm: Optional[SecureCommunication] = None,
        config: Optional[Dict] = None
    ):
        self.client_id = client_id
        self.model_fn = model_fn
        self.data_loader = data_loader
        self.server_endpoint = server_endpoint
        self.config = config or {}
        
        # Initialize components
        self.privacy_engine = privacy_engine or PrivacyEngine()
        self.secure_comm = secure_comm or SecureCommunication()
        
        # Client state
        self.local_model = None
        self.current_round = 0
        self.is_training = False
        self.is_connected = False
        self.registration_status = "unregistered"
        self.training_history: List[TrainingMetrics] = []
        
        # Device capabilities
        self.capabilities = self._detect_device_capabilities()
        
        # Configuration
        self.max_local_epochs = self.config.get("max_local_epochs", 5)
        self.min_data_size = self.config.get("min_data_size", 10)
        self.privacy_budget = self.config.get("privacy_budget", 10.0)
        self.participation_probability = self.config.get("participation_probability", 1.0)
        
        # Resource management
        self.max_cpu_usage = self.config.get("max_cpu_usage", 80)  # Percentage
        self.max_memory_usage = self.config.get("max_memory_usage", 70)  # Percentage
        self.min_battery_level = self.config.get("min_battery_level", 20)  # Percentage
        
        # Edge optimization
        self.use_quantization = self.config.get("use_quantization", True)
        self.use_pruning = self.config.get("use_pruning", False)
        self.target_format = self.config.get("target_format", "tensorflow_lite")
        
        # Offline capability
        self.offline_training_enabled = self.config.get("offline_training", True)
        self.offline_updates_queue = []
        
        logger.info(f"Federated client {client_id} initialized")
    
    def _detect_device_capabilities(self) -> DeviceCapabilities:
        """Detect device capabilities"""
        try:
            # CPU and memory
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU detection
            gpu_available = torch.cuda.is_available()
            gpu_memory = None
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # Battery (if available)
            battery = psutil.sensors_battery()
            battery_level = battery.percent if battery else None
            is_charging = battery.power_plugged if battery else None
            
            capabilities = DeviceCapabilities(
                cpu_count=cpu_count,
                memory_total=memory.total / (1024**3),  # GB
                memory_available=memory.available / (1024**3),  # GB
                storage_available=disk.free / (1024**3),  # GB
                gpu_available=gpu_available,
                gpu_memory=gpu_memory,
                battery_level=battery_level,
                is_charging=is_charging
            )
            
            logger.info(f"Device capabilities: CPU={cpu_count}, RAM={capabilities.memory_total:.1f}GB, "
                       f"GPU={gpu_available}, Battery={battery_level}%")
            
            return capabilities
            
        except Exception as e:
            logger.warning(f"Failed to detect capabilities: {e}")
            return DeviceCapabilities(
                cpu_count=1, 
                memory_total=1.0, 
                memory_available=0.5, 
                storage_available=1.0, 
                gpu_available=False
            )
    
    async def connect_to_server(self) -> bool:
        """Connect and register with the federated server"""
        try:
            # Initialize secure communication
            await self.secure_comm.initialize_client(self.client_id, self.server_endpoint)
            
            # Prepare registration data
            registration_data = {
                "client_id": self.client_id,
                "capabilities": {
                    "cpu_count": self.capabilities.cpu_count,
                    "memory_total": self.capabilities.memory_total,
                    "gpu_available": self.capabilities.gpu_available,
                    "data_size": len(self.data_loader.dataset) if self.data_loader else 0,
                    "compute_capability": self._calculate_compute_capability(),
                    "network_quality": await self._measure_network_quality(),
                    "privacy_budget": self.privacy_budget
                },
                "platform": platform.platform(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Send registration request
            response = await self.secure_comm.send_to_server(
                "register", registration_data
            )
            
            if response.get("status") == "registered":
                self.is_connected = True
                self.registration_status = "registered"
                logger.info(f"Client {self.client_id} registered successfully")
                return True
            else:
                logger.error(f"Registration failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _calculate_compute_capability(self) -> float:
        """Calculate relative compute capability score"""
        score = 0.0
        
        # CPU contribution (0-4 points)
        score += min(self.capabilities.cpu_count / 2, 4)
        
        # Memory contribution (0-3 points)
        score += min(self.capabilities.memory_total / 4, 3)
        
        # GPU contribution (0-3 points)
        if self.capabilities.gpu_available:
            if self.capabilities.gpu_memory:
                score += min(self.capabilities.gpu_memory / 4, 3)
            else:
                score += 1  # Basic GPU
        
        return min(score / 10, 1.0)  # Normalize to 0-1
    
    async def _measure_network_quality(self) -> float:
        """Measure network quality (latency and bandwidth estimate)"""
        try:
            import aiohttp
            import time
            
            # Measure latency with small request
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_endpoint}/health") as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000  # ms
                        
                        # Simple quality score based on latency
                        if latency < 50:
                            return 1.0  # Excellent
                        elif latency < 150:
                            return 0.8  # Good
                        elif latency < 300:
                            return 0.6  # Fair
                        else:
                            return 0.4  # Poor
            
            return 0.5  # Default
            
        except Exception:
            return 0.5  # Default quality
    
    async def listen_for_training_instructions(self) -> None:
        """Listen for training instructions from server"""
        while self.is_connected:
            try:
                message = await self.secure_comm.receive_from_server()
                
                if message and message.get("type") == "training_instruction":
                    await self._handle_training_instruction(message)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error listening for instructions: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _handle_training_instruction(self, instruction: Dict) -> None:
        """Handle training instruction from server"""
        try:
            # Check if we should participate
            if not self._should_participate(instruction):
                logger.info("Declined to participate in this round")
                return
            
            # Extract instruction data
            round_id = instruction["round_id"]
            model_weights = instruction["model_weights"]
            learning_rate = instruction.get("learning_rate", 0.01)
            epochs = instruction.get("epochs", 1)
            batch_size = instruction.get("batch_size", 32)
            privacy_budget = instruction.get("privacy_budget", 1.0)
            
            logger.info(f"Starting training for round {round_id}")
            
            # Perform local training
            training_result = await self._perform_local_training(
                round_id, model_weights, learning_rate, epochs, batch_size, privacy_budget
            )
            
            if training_result:
                # Send update to server
                await self._send_model_update(training_result)
            
        except Exception as e:
            logger.error(f"Failed to handle training instruction: {e}")
    
    def _should_participate(self, instruction: Dict) -> bool:
        """Decide whether to participate in this training round"""
        # Check participation probability
        if np.random.random() > self.participation_probability:
            return False
        
        # Check resource constraints
        if not self._check_resource_availability():
            logger.info("Insufficient resources for participation")
            return False
        
        # Check data size
        if len(self.data_loader.dataset) < self.min_data_size:
            logger.info("Insufficient data for participation")
            return False
        
        # Check privacy budget
        required_budget = instruction.get("privacy_budget", 1.0)
        if self.privacy_budget < required_budget:
            logger.info("Insufficient privacy budget")
            return False
        
        return True
    
    def _check_resource_availability(self) -> bool:
        """Check if device resources are available for training"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.max_cpu_usage:
                return False
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.max_memory_usage:
                return False
            
            # Battery level (if applicable)
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                if battery.percent < self.min_battery_level:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True  # Assume available if check fails
    
    async def _perform_local_training(
        self,
        round_id: int,
        server_weights: Dict,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        privacy_budget: float
    ) -> Optional[TrainingMetrics]:
        """Perform local model training"""
        start_time = time.time()
        
        try:
            # Initialize/update local model
            if self.local_model is None:
                self.local_model = self.model_fn()
            
            # Load server weights
            self._load_model_weights(server_weights)
            
            # Setup training
            if hasattr(self.local_model, 'compile'):
                # TensorFlow model
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                self.local_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
                initial_loss = self._evaluate_model()
                
                # Apply privacy mechanisms
                if self.privacy_engine.differential_privacy_enabled:
                    # Add DP-SGD
                    from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
                    optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
                        l2_norm_clip=1.0,
                        noise_multiplier=0.1,
                        num_microbatches=batch_size,
                        learning_rate=learning_rate
                    )
                    self.local_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
                
                # Train model
                history = self.local_model.fit(
                    self.data_loader,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                final_loss = history.history['loss'][-1]
                accuracy = history.history.get('accuracy', [None])[-1]
                
            else:
                # PyTorch model
                optimizer = optim.SGD(self.local_model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                initial_loss = self._evaluate_model()
                
                self.local_model.train()
                total_loss = 0
                num_batches = 0
                
                for epoch in range(epochs):
                    for batch_data, batch_labels in self.data_loader:
                        optimizer.zero_grad()
                        outputs = self.local_model(batch_data)
                        loss = criterion(outputs, batch_labels)
                        
                        # Apply differential privacy
                        if self.privacy_engine.differential_privacy_enabled:
                            loss = await self.privacy_engine.apply_dp_noise(loss, privacy_budget)
                        
                        loss.backward()
                        
                        # Gradient clipping for privacy
                        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                
                final_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                accuracy = self._calculate_accuracy()
            
            training_time = time.time() - start_time
            
            # Calculate privacy cost
            privacy_cost = self.privacy_engine.calculate_privacy_cost(
                epochs, len(self.data_loader.dataset), privacy_budget
            )
            
            # Create training metrics
            metrics = TrainingMetrics(
                round_id=round_id,
                local_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                initial_loss=initial_loss,
                final_loss=final_loss,
                accuracy=accuracy,
                training_time=training_time,
                data_size=len(self.data_loader.dataset),
                privacy_cost=privacy_cost
            )
            
            # Update privacy budget
            self.privacy_budget -= privacy_cost
            
            # Store metrics
            self.training_history.append(metrics)
            
            logger.info(f"Local training completed: loss {initial_loss:.4f} -> {final_loss:.4f}, "
                       f"time {training_time:.1f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            return None
    
    def _load_model_weights(self, weights: Dict) -> None:
        """Load weights into local model"""
        try:
            if hasattr(self.local_model, 'set_weights'):
                # TensorFlow model
                weights_list = [np.array(weights[name]) for name in sorted(weights.keys())]
                self.local_model.set_weights(weights_list)
            elif hasattr(self.local_model, 'state_dict'):
                # PyTorch model
                state_dict = {}
                for name, weight in weights.items():
                    state_dict[name] = torch.tensor(weight)
                self.local_model.load_state_dict(state_dict)
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def _evaluate_model(self) -> float:
        """Evaluate current model performance"""
        try:
            if hasattr(self.local_model, 'evaluate'):
                # TensorFlow model
                results = self.local_model.evaluate(self.data_loader, verbose=0)
                return results if isinstance(results, float) else results[0]
            else:
                # PyTorch model
                self.local_model.eval()
                total_loss = 0
                num_batches = 0
                
                with torch.no_grad():
                    for batch_data, batch_labels in self.data_loader:
                        outputs = self.local_model(batch_data)
                        loss = nn.functional.cross_entropy(outputs, batch_labels)
                        total_loss += loss.item()
                        num_batches += 1
                
                return total_loss / num_batches if num_batches > 0 else float('inf')
                
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return float('inf')
    
    def _calculate_accuracy(self) -> Optional[float]:
        """Calculate model accuracy on local data"""
        try:
            if not hasattr(self.local_model, 'state_dict'):  # Skip for TF models
                return None
            
            self.local_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in self.data_loader:
                    outputs = self.local_model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            return correct / total if total > 0 else None
            
        except Exception as e:
            logger.warning(f"Accuracy calculation failed: {e}")
            return None
    
    async def _send_model_update(self, metrics: TrainingMetrics) -> None:
        """Send model update to server"""
        try:
            # Extract model weights
            model_weights = self._extract_model_weights()
            
            # Apply edge optimizations
            if self.use_quantization:
                model_weights = self._quantize_weights(model_weights)
            
            if self.use_pruning:
                model_weights = self._prune_weights(model_weights)
            
            # Prepare update data
            update_data = {
                "client_id": self.client_id,
                "round_id": metrics.round_id,
                "model_weights": model_weights,
                "metrics": {
                    "initial_loss": metrics.initial_loss,
                    "final_loss": metrics.final_loss,
                    "accuracy": metrics.accuracy,
                    "data_size": metrics.data_size,
                    "training_time": metrics.training_time,
                    "privacy_cost": metrics.privacy_cost
                },
                "data_size": metrics.data_size,
                "loss": metrics.final_loss,
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate checksum
            update_data["checksum"] = self._calculate_checksum(update_data)
            
            # Send encrypted update
            if self.is_connected:
                response = await self.secure_comm.send_to_server("model_update", update_data)
                
                if response.get("status") == "received":
                    logger.info(f"Model update sent successfully for round {metrics.round_id}")
                else:
                    logger.warning(f"Model update failed: {response}")
            else:
                # Store for offline transmission
                if self.offline_training_enabled:
                    self.offline_updates_queue.append(update_data)
                    logger.info("Update queued for offline transmission")
            
        except Exception as e:
            logger.error(f"Failed to send model update: {e}")
    
    def _extract_model_weights(self) -> Dict:
        """Extract weights from local model"""
        try:
            if hasattr(self.local_model, 'get_weights'):
                # TensorFlow model
                weights = self.local_model.get_weights()
                return {f"layer_{i}": w.tolist() for i, w in enumerate(weights)}
            elif hasattr(self.local_model, 'state_dict'):
                # PyTorch model
                state_dict = self.local_model.state_dict()
                return {name: tensor.cpu().numpy().tolist() for name, tensor in state_dict.items()}
            else:
                # Custom model
                return {}
                
        except Exception as e:
            logger.error(f"Failed to extract model weights: {e}")
            return {}
    
    def _quantize_weights(self, weights: Dict) -> Dict:
        """Apply quantization to reduce model size"""
        try:
            quantized_weights = {}
            
            for name, weight in weights.items():
                weight_array = np.array(weight)
                
                # Simple 8-bit quantization
                min_val = np.min(weight_array)
                max_val = np.max(weight_array)
                
                if max_val > min_val:
                    scale = (max_val - min_val) / 255
                    quantized = np.round((weight_array - min_val) / scale).astype(np.uint8)
                    
                    quantized_weights[name] = {
                        "values": quantized.tolist(),
                        "scale": scale,
                        "zero_point": min_val,
                        "quantized": True
                    }
                else:
                    quantized_weights[name] = weight
            
            return quantized_weights
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return weights
    
    def _prune_weights(self, weights: Dict, sparsity: float = 0.1) -> Dict:
        """Apply pruning to reduce model size"""
        try:
            pruned_weights = {}
            
            for name, weight in weights.items():
                if isinstance(weight, dict) and weight.get("quantized"):
                    # Skip already processed weights
                    pruned_weights[name] = weight
                    continue
                
                weight_array = np.array(weight)
                
                # Magnitude-based pruning
                threshold = np.percentile(np.abs(weight_array), sparsity * 100)
                mask = np.abs(weight_array) > threshold
                
                pruned_array = weight_array * mask
                pruned_weights[name] = pruned_array.tolist()
            
            return pruned_weights
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return weights
    
    def _calculate_checksum(self, data: Dict) -> str:
        """Calculate checksum for data integrity"""
        try:
            # Create reproducible string representation
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except Exception:
            return ""
    
    def optimize_for_edge(self, target_format: str = None) -> None:
        """Optimize model for edge deployment"""
        if not self.local_model:
            return
        
        target = target_format or self.target_format
        
        try:
            if target == "tensorflow_lite":
                self._convert_to_tflite()
            elif target == "onnx":
                self._convert_to_onnx()
            elif target == "quantized":
                self._apply_post_training_quantization()
            
        except Exception as e:
            logger.error(f"Edge optimization failed: {e}")
    
    def _convert_to_tflite(self) -> None:
        """Convert model to TensorFlow Lite format"""
        try:
            if hasattr(self.local_model, 'save'):
                # TensorFlow model
                converter = tf.lite.TFLiteConverter.from_keras_model(self.local_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                # Save optimized model
                tflite_path = f"models/{self.client_id}_model.tflite"
                Path(tflite_path).parent.mkdir(exist_ok=True)
                
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger.info(f"Model converted to TensorFlow Lite: {tflite_path}")
                
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
    
    def _convert_to_onnx(self) -> None:
        """Convert model to ONNX format"""
        try:
            if hasattr(self.local_model, 'state_dict'):
                # PyTorch model
                import torch.onnx
                
                # Create dummy input
                dummy_input = torch.randn(1, *self._get_input_shape())
                
                onnx_path = f"models/{self.client_id}_model.onnx"
                Path(onnx_path).parent.mkdir(exist_ok=True)
                
                torch.onnx.export(
                    self.local_model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
                
                logger.info(f"Model converted to ONNX: {onnx_path}")
                
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
    
    def _get_input_shape(self) -> Tuple:
        """Get model input shape"""
        try:
            # Try to infer from data loader
            for batch_data, _ in self.data_loader:
                return batch_data.shape[1:]  # Remove batch dimension
            
            return (1,)  # Default shape
            
        except Exception:
            return (1,)
    
    def _apply_post_training_quantization(self) -> None:
        """Apply post-training quantization"""
        try:
            if hasattr(self.local_model, 'save'):
                # TensorFlow quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(self.local_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                quantized_model = converter.convert()
                
                quantized_path = f"models/{self.client_id}_quantized.tflite"
                Path(quantized_path).parent.mkdir(exist_ok=True)
                
                with open(quantized_path, 'wb') as f:
                    f.write(quantized_model)
                
                logger.info(f"Model quantized: {quantized_path}")
                
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization"""
        try:
            for batch_data, _ in self.data_loader:
                yield [batch_data[:1].numpy()]  # Use first sample
                break  # Only need one batch
        except Exception:
            pass
    
    async def sync_offline_updates(self) -> None:
        """Synchronize offline updates when connection is restored"""
        if not self.is_connected or not self.offline_updates_queue:
            return
        
        logger.info(f"Syncing {len(self.offline_updates_queue)} offline updates")
        
        successful_syncs = 0
        
        for update_data in self.offline_updates_queue[:]:
            try:
                response = await self.secure_comm.send_to_server("model_update", update_data)
                
                if response.get("status") == "received":
                    self.offline_updates_queue.remove(update_data)
                    successful_syncs += 1
                else:
                    logger.warning(f"Offline sync failed for update: {response}")
                
            except Exception as e:
                logger.error(f"Failed to sync offline update: {e}")
                break  # Stop on first failure to maintain order
        
        logger.info(f"Synced {successful_syncs} offline updates")
    
    async def disconnect_from_server(self) -> None:
        """Disconnect from the federated server"""
        try:
            if self.is_connected:
                await self.secure_comm.send_to_server("unregister", {"client_id": self.client_id})
                await self.secure_comm.cleanup()
                self.is_connected = False
                self.registration_status = "unregistered"
                logger.info(f"Client {self.client_id} disconnected")
                
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get current client status"""
        return {
            "client_id": self.client_id,
            "is_connected": self.is_connected,
            "is_training": self.is_training,
            "registration_status": self.registration_status,
            "current_round": self.current_round,
            "privacy_budget_remaining": self.privacy_budget,
            "training_history_count": len(self.training_history),
            "offline_updates_pending": len(self.offline_updates_queue),
            "device_capabilities": self.capabilities.__dict__,
            "last_training": self.training_history[-1].__dict__ if self.training_history else None
        }
    
    def export_training_history(self, export_path: str) -> None:
        """Export training history for analysis"""
        try:
            history_data = [metrics.__dict__ for metrics in self.training_history]
            
            with open(export_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            logger.info(f"Training history exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export training history: {e}")
    
    async def run_continuous_training(self) -> None:
        """Run continuous federated training loop"""
        logger.info(f"Starting continuous training for client {self.client_id}")
        
        try:
            # Connect to server
            if not await self.connect_to_server():
                logger.error("Failed to connect to server")
                return
            
            # Sync any offline updates
            await self.sync_offline_updates()
            
            # Listen for training instructions
            await self.listen_for_training_instructions()
            
        except Exception as e:
            logger.error(f"Continuous training error: {e}")
        finally:
            await self.disconnect_from_server()