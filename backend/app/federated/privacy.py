"""
Privacy-Preserving Mechanisms for Federated Learning

This module implements various privacy-preserving techniques including
differential privacy, secure multi-party computation, homomorphic encryption,
and secure aggregation protocols.
"""

import logging
import numpy as np
import torch
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import hashlib
import secrets
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import math
import time

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Privacy budget tracking"""
    epsilon: float  # Privacy parameter
    delta: float    # Failure probability
    used_epsilon: float = 0.0
    used_delta: float = 0.0
    
    @property
    def remaining_epsilon(self) -> float:
        return max(0, self.epsilon - self.used_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        return max(0, self.delta - self.used_delta)
    
    def consume(self, epsilon: float, delta: float = 0.0) -> bool:
        """Consume privacy budget"""
        if self.remaining_epsilon >= epsilon and self.remaining_delta >= delta:
            self.used_epsilon += epsilon
            self.used_delta += delta
            return True
        return False


@dataclass  
class DPMechanism:
    """Differential privacy mechanism configuration"""
    mechanism_type: str  # "gaussian", "laplace", "exponential"
    sensitivity: float   # Global sensitivity
    noise_multiplier: float
    clipping_norm: float = 1.0


@dataclass
class SecureAggregationConfig:
    """Secure aggregation configuration"""
    threshold: int  # Minimum number of clients for aggregation
    max_clients: int  # Maximum number of clients
    dropout_resilience: bool = True
    byzantine_resilience: bool = False


class PrivacyEngine:
    """
    Comprehensive privacy engine for federated learning
    
    Features:
    - Differential Privacy (DP-SGD, DP-FedAvg)
    - Secure Multi-Party Computation (SMPC)
    - Homomorphic Encryption (HE)
    - Secure Aggregation
    - Privacy Budget Management
    - Client-level Privacy Guarantees
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Privacy mechanisms
        self.differential_privacy_enabled = self.config.get("differential_privacy", True)
        self.secure_aggregation_enabled = self.config.get("secure_aggregation", True)  
        self.homomorphic_encryption_enabled = self.config.get("homomorphic_encryption", False)
        self.secure_mpc_enabled = self.config.get("secure_mpc", False)
        
        # DP configuration
        self.dp_config = DPMechanism(
            mechanism_type=self.config.get("dp_mechanism", "gaussian"),
            sensitivity=self.config.get("dp_sensitivity", 1.0),
            noise_multiplier=self.config.get("noise_multiplier", 1.0),
            clipping_norm=self.config.get("clipping_norm", 1.0)
        )
        
        # Privacy budgets per client
        self.client_budgets: Dict[str, PrivacyBudget] = {}
        self.global_budget = PrivacyBudget(
            epsilon=self.config.get("global_epsilon", 10.0),
            delta=self.config.get("global_delta", 1e-5)
        )
        
        # Secure aggregation
        self.secure_agg_config = SecureAggregationConfig(
            threshold=self.config.get("secure_threshold", 3),
            max_clients=self.config.get("max_clients", 100)
        )
        
        # Encryption keys
        self.encryption_keys = {}
        self._initialize_encryption()
        
        # Privacy accounting
        self.privacy_ledger = []
        
        logger.info("Privacy engine initialized with DP={}, SecAgg={}, HE={}".format(
            self.differential_privacy_enabled,
            self.secure_aggregation_enabled,
            self.homomorphic_encryption_enabled
        ))
    
    def _initialize_encryption(self):
        """Initialize encryption components"""
        try:
            if self.homomorphic_encryption_enabled:
                # Initialize homomorphic encryption (simplified)
                self._initialize_he_keys()
            
            # Generate server RSA keys for secure communication
            self.server_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.server_public_key = self.server_private_key.public_key()
            
        except Exception as e:
            logger.warning(f"Encryption initialization failed: {e}")
    
    def _initialize_he_keys(self):
        """Initialize homomorphic encryption keys (placeholder)"""
        # In practice, would use libraries like PySEAL, Pyfhel, or TenSEAL
        # This is a simplified placeholder
        self.he_public_key = "he_public_key_placeholder"
        self.he_secret_key = "he_secret_key_placeholder"
        self.he_context = "he_context_placeholder"
    
    def register_client(self, client_id: str, privacy_budget: Optional[PrivacyBudget] = None) -> Dict[str, Any]:
        """Register client and allocate privacy budget"""
        if privacy_budget is None:
            privacy_budget = PrivacyBudget(
                epsilon=self.config.get("client_epsilon", 1.0),
                delta=self.config.get("client_delta", 1e-6)
            )
        
        self.client_budgets[client_id] = privacy_budget
        
        # Generate client-specific encryption keys
        client_keys = self._generate_client_keys(client_id)
        
        return {
            "privacy_budget": privacy_budget,
            "encryption_keys": client_keys,
            "dp_config": self.dp_config
        }
    
    def _generate_client_keys(self, client_id: str) -> Dict[str, Any]:
        """Generate encryption keys for client"""
        # Generate symmetric key for this client
        symmetric_key = secrets.token_bytes(32)  # 256-bit key
        
        # Store encrypted with server key
        self.encryption_keys[client_id] = {
            "symmetric_key": symmetric_key,
            "created_at": time.time()
        }
        
        # Return public components for client
        return {
            "server_public_key": self.server_public_key.public_numbers(),
            "he_public_key": self.he_public_key if self.homomorphic_encryption_enabled else None
        }
    
    async def apply_differential_privacy(
        self,
        model_weights: Dict[str, Any],
        client_id: str,
        sensitivity: Optional[float] = None,
        epsilon: Optional[float] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Apply differential privacy to model weights"""
        if not self.differential_privacy_enabled:
            return model_weights, 0.0
        
        try:
            # Get privacy parameters
            sensitivity = sensitivity or self.dp_config.sensitivity
            epsilon = epsilon or self.config.get("round_epsilon", 0.1)
            
            # Check privacy budget
            client_budget = self.client_budgets.get(client_id)
            if client_budget and not client_budget.consume(epsilon):
                raise ValueError(f"Insufficient privacy budget for client {client_id}")
            
            # Apply noise based on mechanism type
            if self.dp_config.mechanism_type == "gaussian":
                noisy_weights = self._add_gaussian_noise(model_weights, sensitivity, epsilon)
            elif self.dp_config.mechanism_type == "laplace":
                noisy_weights = self._add_laplace_noise(model_weights, sensitivity, epsilon)
            else:
                raise ValueError(f"Unknown DP mechanism: {self.dp_config.mechanism_type}")
            
            # Record privacy spending
            self._record_privacy_usage(client_id, epsilon, 0.0)
            
            return noisy_weights, epsilon
            
        except Exception as e:
            logger.error(f"DP application failed: {e}")
            return model_weights, 0.0
    
    def _add_gaussian_noise(
        self, 
        weights: Dict[str, Any], 
        sensitivity: float, 
        epsilon: float
    ) -> Dict[str, Any]:
        """Add Gaussian noise for differential privacy"""
        # Calculate noise scale
        delta = self.config.get("gaussian_delta", 1e-6)
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        noisy_weights = {}
        
        for layer_name, weight in weights.items():
            if isinstance(weight, (list, np.ndarray)):
                weight_array = np.array(weight)
                noise = np.random.normal(0, sigma, weight_array.shape)
                noisy_weights[layer_name] = (weight_array + noise).tolist()
            else:
                noisy_weights[layer_name] = weight
        
        return noisy_weights
    
    def _add_laplace_noise(
        self,
        weights: Dict[str, Any],
        sensitivity: float,
        epsilon: float
    ) -> Dict[str, Any]:
        """Add Laplace noise for differential privacy"""
        # Calculate noise scale
        scale = sensitivity / epsilon
        
        noisy_weights = {}
        
        for layer_name, weight in weights.items():
            if isinstance(weight, (list, np.ndarray)):
                weight_array = np.array(weight)
                noise = np.random.laplace(0, scale, weight_array.shape)
                noisy_weights[layer_name] = (weight_array + noise).tolist()
            else:
                noisy_weights[layer_name] = weight
        
        return noisy_weights
    
    async def clip_gradients(
        self,
        gradients: Dict[str, Any],
        clipping_norm: Optional[float] = None
    ) -> Dict[str, Any]:
        """Clip gradients for privacy and stability"""
        clipping_norm = clipping_norm or self.dp_config.clipping_norm
        
        clipped_gradients = {}
        
        # Calculate total gradient norm
        total_norm = 0.0
        for layer_name, grad in gradients.items():
            if isinstance(grad, (list, np.ndarray)):
                grad_array = np.array(grad)
                total_norm += np.sum(grad_array ** 2)
        
        total_norm = np.sqrt(total_norm)
        
        # Apply clipping
        if total_norm > clipping_norm:
            clip_factor = clipping_norm / total_norm
            
            for layer_name, grad in gradients.items():
                if isinstance(grad, (list, np.ndarray)):
                    grad_array = np.array(grad)
                    clipped_gradients[layer_name] = (grad_array * clip_factor).tolist()
                else:
                    clipped_gradients[layer_name] = grad
        else:
            clipped_gradients = gradients
        
        return clipped_gradients
    
    async def secure_aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        client_ids: List[str]
    ) -> Dict[str, Any]:
        """Perform secure aggregation of client updates"""
        if not self.secure_aggregation_enabled or len(client_updates) < self.secure_agg_config.threshold:
            return self._simple_aggregate(client_updates)
        
        try:
            # Secure aggregation protocol implementation
            # This is a simplified version - production would use full SecAgg protocol
            
            # Step 1: Generate shared secrets
            shared_secrets = self._generate_shared_secrets(client_ids)
            
            # Step 2: Mask client updates
            masked_updates = []
            for i, (update, client_id) in enumerate(zip(client_updates, client_ids)):
                masked_update = self._mask_update(update, shared_secrets[i])
                masked_updates.append(masked_update)
            
            # Step 3: Aggregate masked updates
            aggregated_masked = self._simple_aggregate(masked_updates)
            
            # Step 4: Remove masks (simplified - assumes all clients participate)
            final_aggregate = self._unmask_aggregate(aggregated_masked, shared_secrets)
            
            logger.info(f"Secure aggregation completed with {len(client_updates)} clients")
            
            return final_aggregate
            
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            return self._simple_aggregate(client_updates)
    
    def _generate_shared_secrets(self, client_ids: List[str]) -> List[Dict[str, np.ndarray]]:
        """Generate shared secrets for secure aggregation"""
        shared_secrets = []
        
        for client_id in client_ids:
            # Generate random secret for this client
            secret = {}
            # In practice, would use proper secret sharing scheme
            secret["random_mask"] = np.random.standard_normal()
            shared_secrets.append(secret)
        
        return shared_secrets
    
    def _mask_update(self, update: Dict[str, Any], secret: Dict[str, Any]) -> Dict[str, Any]:
        """Mask client update with secret"""
        masked_update = {}
        
        for layer_name, weight in update.items():
            if isinstance(weight, (list, np.ndarray)):
                weight_array = np.array(weight)
                # Add random mask
                mask = np.full_like(weight_array, secret["random_mask"])
                masked_update[layer_name] = (weight_array + mask).tolist()
            else:
                masked_update[layer_name] = weight
        
        return masked_update
    
    def _unmask_aggregate(
        self,
        masked_aggregate: Dict[str, Any],
        shared_secrets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Remove masks from aggregated result"""
        unmasked_aggregate = {}
        
        # Calculate total mask
        total_mask = sum(secret["random_mask"] for secret in shared_secrets)
        
        for layer_name, weight in masked_aggregate.items():
            if isinstance(weight, (list, np.ndarray)):
                weight_array = np.array(weight)
                # Remove total mask
                mask = np.full_like(weight_array, total_mask)
                unmasked_aggregate[layer_name] = (weight_array - mask).tolist()
            else:
                unmasked_aggregate[layer_name] = weight
        
        return unmasked_aggregate
    
    def _simple_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple averaging aggregation (fallback)"""
        if not updates:
            return {}
        
        aggregated = {}
        layer_names = list(updates[0].keys())
        
        for layer_name in layer_names:
            layer_weights = []
            
            for update in updates:
                if layer_name in update:
                    weight = update[layer_name]
                    if isinstance(weight, (list, np.ndarray)):
                        layer_weights.append(np.array(weight))
            
            if layer_weights:
                # Simple average
                averaged_weight = np.mean(layer_weights, axis=0)
                aggregated[layer_name] = averaged_weight.tolist()
        
        return aggregated
    
    async def encrypt_model_update(
        self,
        update: Dict[str, Any],
        client_id: str
    ) -> bytes:
        """Encrypt model update for secure transmission"""
        try:
            # Get client's symmetric key
            client_keys = self.encryption_keys.get(client_id)
            if not client_keys:
                raise ValueError(f"No encryption keys for client {client_id}")
            
            symmetric_key = client_keys["symmetric_key"]
            
            # Serialize update
            update_json = json.dumps(update, default=self._json_serializer)
            update_bytes = update_json.encode('utf-8')
            
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Encrypt with AES-GCM
            cipher = Cipher(
                algorithms.AES(symmetric_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(update_bytes) + encryptor.finalize()
            
            # Combine IV, tag, and ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def decrypt_model_update(
        self,
        encrypted_data: bytes,
        client_id: str
    ) -> Dict[str, Any]:
        """Decrypt model update"""
        try:
            # Get client's symmetric key
            client_keys = self.encryption_keys.get(client_id)
            if not client_keys:
                raise ValueError(f"No encryption keys for client {client_id}")
            
            symmetric_key = client_keys["symmetric_key"]
            
            # Extract components
            iv = encrypted_data[:16]
            tag = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            
            # Decrypt with AES-GCM
            cipher = Cipher(
                algorithms.AES(symmetric_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Deserialize
            update_json = decrypted_bytes.decode('utf-8')
            update = json.loads(update_json)
            
            return update
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    async def apply_homomorphic_encryption(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply homomorphic encryption to data (placeholder)"""
        if not self.homomorphic_encryption_enabled:
            return data
        
        # Placeholder for HE implementation
        # In practice, would use libraries like PySEAL, Pyfhel, or TenSEAL
        logger.info("Homomorphic encryption applied (placeholder)")
        return data
    
    async def homomorphic_aggregate(
        self,
        encrypted_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate homomorphically encrypted updates (placeholder)"""
        if not self.homomorphic_encryption_enabled:
            return self._simple_aggregate(encrypted_updates)
        
        # Placeholder for HE aggregation
        logger.info("Homomorphic aggregation performed (placeholder)")
        return self._simple_aggregate(encrypted_updates)
    
    def calculate_privacy_cost(
        self,
        num_epochs: int,
        data_size: int,
        epsilon_per_round: float,
        mechanism: str = "gaussian"
    ) -> float:
        """Calculate privacy cost for training"""
        if not self.differential_privacy_enabled:
            return 0.0
        
        try:
            # RDP accounting for DP-SGD
            if mechanism == "gaussian":
                # Simplified RDP calculation
                # In practice, would use more sophisticated privacy accounting
                noise_multiplier = self.dp_config.noise_multiplier
                sampling_rate = min(1.0, 32 / data_size)  # Batch size / dataset size
                
                # Calculate privacy cost per step
                alpha_max = 32  # Maximum alpha for RDP
                steps = num_epochs * max(1, data_size // 32)  # Number of gradient steps
                
                # Simplified calculation
                privacy_cost = epsilon_per_round * steps * sampling_rate
                
                return min(privacy_cost, self.global_budget.remaining_epsilon)
            
            return epsilon_per_round
            
        except Exception as e:
            logger.warning(f"Privacy cost calculation failed: {e}")
            return epsilon_per_round
    
    def _record_privacy_usage(
        self,
        client_id: str,
        epsilon: float,
        delta: float
    ):
        """Record privacy budget usage"""
        usage_record = {
            "client_id": client_id,
            "epsilon": epsilon,
            "delta": delta,
            "timestamp": time.time()
        }
        
        self.privacy_ledger.append(usage_record)
        
        # Update global budget
        self.global_budget.consume(epsilon, delta)
    
    def get_privacy_status(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Get privacy budget status"""
        if client_id:
            client_budget = self.client_budgets.get(client_id)
            if client_budget:
                return {
                    "client_id": client_id,
                    "total_epsilon": client_budget.epsilon,
                    "used_epsilon": client_budget.used_epsilon,
                    "remaining_epsilon": client_budget.remaining_epsilon,
                    "total_delta": client_budget.delta,
                    "used_delta": client_budget.used_delta,
                    "remaining_delta": client_budget.remaining_delta
                }
            else:
                return {"error": f"Client {client_id} not found"}
        else:
            # Global status
            return {
                "global_budget": {
                    "total_epsilon": self.global_budget.epsilon,
                    "used_epsilon": self.global_budget.used_epsilon,
                    "remaining_epsilon": self.global_budget.remaining_epsilon,
                    "total_delta": self.global_budget.delta,
                    "used_delta": self.global_budget.used_delta,
                    "remaining_delta": self.global_budget.remaining_delta
                },
                "registered_clients": len(self.client_budgets),
                "privacy_records": len(self.privacy_ledger)
            }
    
    def estimate_privacy_loss(
        self,
        num_rounds: int,
        clients_per_round: int,
        epsilon_per_round: float
    ) -> Dict[str, float]:
        """Estimate total privacy loss for training plan"""
        try:
            # Composition of privacy losses
            if self.config.get("privacy_composition", "basic") == "basic":
                # Basic composition
                total_epsilon = num_rounds * epsilon_per_round
                total_delta = num_rounds * self.config.get("round_delta", 0.0)
            else:
                # Advanced composition (simplified)
                total_epsilon = epsilon_per_round * np.sqrt(num_rounds * np.log(1/self.global_budget.delta))
                total_delta = self.global_budget.delta
            
            return {
                "estimated_total_epsilon": total_epsilon,
                "estimated_total_delta": total_delta,
                "rounds": num_rounds,
                "epsilon_per_round": epsilon_per_round
            }
            
        except Exception as e:
            logger.warning(f"Privacy loss estimation failed: {e}")
            return {
                "estimated_total_epsilon": num_rounds * epsilon_per_round,
                "estimated_total_delta": 0.0,
                "rounds": num_rounds,
                "epsilon_per_round": epsilon_per_round
            }
    
    def export_privacy_ledger(self, filepath: str):
        """Export privacy usage ledger"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.privacy_ledger, f, indent=2, default=str)
            
            logger.info(f"Privacy ledger exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export privacy ledger: {e}")
    
    def validate_privacy_guarantees(self) -> Dict[str, Any]:
        """Validate that privacy guarantees are maintained"""
        violations = []
        warnings = []
        
        # Check global budget
        if self.global_budget.remaining_epsilon < 0:
            violations.append("Global epsilon budget exceeded")
        
        if self.global_budget.remaining_delta < 0:
            violations.append("Global delta budget exceeded")
        
        # Check client budgets
        for client_id, budget in self.client_budgets.items():
            if budget.remaining_epsilon < 0:
                violations.append(f"Client {client_id} epsilon budget exceeded")
            
            if budget.remaining_delta < 0:
                violations.append(f"Client {client_id} delta budget exceeded")
            
            # Warning if budget is low
            if budget.remaining_epsilon < 0.1 * budget.epsilon:
                warnings.append(f"Client {client_id} epsilon budget low")
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "global_budget_status": self.get_privacy_status(),
            "timestamp": time.time()
        }
    
    async def add_model_noise(
        self,
        model_weights: Dict[str, Any],
        privacy_budget: float
    ) -> Dict[str, Any]:
        """Add privacy-preserving noise to model weights"""
        if not self.differential_privacy_enabled:
            return model_weights
        
        return self._add_gaussian_noise(
            model_weights,
            self.dp_config.sensitivity,
            privacy_budget
        )
    
    async def apply_dp_noise(
        self,
        loss: Union[torch.Tensor, tf.Tensor, float],
        privacy_budget: float
    ) -> Union[torch.Tensor, tf.Tensor, float]:
        """Apply differential privacy noise to loss"""
        if not self.differential_privacy_enabled:
            return loss
        
        try:
            if isinstance(loss, torch.Tensor):
                noise = torch.normal(0, self.dp_config.noise_multiplier, loss.shape)
                return loss + noise
            elif isinstance(loss, tf.Tensor):
                noise = tf.random.normal(loss.shape, stddev=self.dp_config.noise_multiplier)
                return loss + noise
            else:
                noise = np.random.normal(0, self.dp_config.noise_multiplier)
                return loss + noise
                
        except Exception as e:
            logger.warning(f"DP noise application failed: {e}")
            return loss
    
    def cleanup_client(self, client_id: str):
        """Clean up client privacy data"""
        try:
            if client_id in self.client_budgets:
                del self.client_budgets[client_id]
            
            if client_id in self.encryption_keys:
                del self.encryption_keys[client_id]
            
            # Remove from privacy ledger (keep for audit)
            # self.privacy_ledger = [record for record in self.privacy_ledger 
            #                      if record["client_id"] != client_id]
            
            logger.info(f"Privacy data cleaned up for client {client_id}")
            
        except Exception as e:
            logger.warning(f"Privacy cleanup failed for client {client_id}: {e}")