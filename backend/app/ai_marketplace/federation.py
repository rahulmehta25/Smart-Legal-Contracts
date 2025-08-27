"""
Federated Model Training System

Enables collaborative model training across distributed data sources while preserving privacy.
"""

import os
import json
import pickle
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import syft as sy
import tenseal as ts
import pycryptodome
from differential_privacy import GaussianMechanism, LaplaceMechanism
import websocket
import grpc
from concurrent import futures
import logging
from pathlib import Path
import redis
import uuid
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()
logger = logging.getLogger(__name__)


class FederationStrategy(Enum):
    """Federated learning strategies"""
    FEDERATED_AVERAGING = "federated_averaging"
    FEDERATED_SGD = "federated_sgd"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SPLIT_LEARNING = "split_learning"
    VERTICAL_FL = "vertical_fl"
    HIERARCHICAL_FL = "hierarchical_fl"


class ClientStatus(Enum):
    """Client status in federation"""
    IDLE = "idle"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


class AggregationMethod(Enum):
    """Model aggregation methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    BYZANTINE_ROBUST = "byzantine_robust"


@dataclass
class FederationConfig:
    """Federation training configuration"""
    federation_id: str
    model_id: str
    strategy: FederationStrategy
    min_clients: int = 2
    max_clients: int = 100
    rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    
    # Privacy settings
    differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # Security settings
    secure_aggregation: bool = True
    homomorphic_encryption: bool = False
    client_authentication: bool = True
    model_encryption: bool = True
    
    # Performance settings
    compression_enabled: bool = True
    compression_ratio: float = 0.1
    adaptive_aggregation: bool = True
    stragglers_handling: bool = True
    timeout_seconds: int = 300


@dataclass
class ClientConfig:
    """Client configuration for federated training"""
    client_id: str
    data_size: int
    compute_capability: str
    network_bandwidth: float
    privacy_requirements: Dict[str, Any]
    trusted: bool = True


@dataclass
class TrainingRound:
    """Training round information"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime]
    participating_clients: List[str]
    global_model_hash: str
    metrics: Dict[str, float]
    aggregation_stats: Dict[str, Any]


# Database Models

class FederationDB(Base):
    """Federation training session"""
    __tablename__ = 'federations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    federation_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    strategy = Column(String(50), nullable=False)
    status = Column(String(50), default='initializing')
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    clients = relationship("FederatedClientDB", back_populates="federation")
    rounds = relationship("TrainingRoundDB", back_populates="federation")


class FederatedClientDB(Base):
    """Federated learning client"""
    __tablename__ = 'federated_clients'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    federation_id = Column(UUID(as_uuid=True), ForeignKey('federations.id'))
    client_id = Column(String(255), unique=True, nullable=False)
    status = Column(String(50), default=ClientStatus.IDLE.value)
    data_size = Column(Integer)
    compute_capability = Column(String(50))
    network_bandwidth = Column(Float)
    contribution_score = Column(Float, default=0.0)
    rounds_participated = Column(Integer, default=0)
    last_active = Column(DateTime, default=datetime.utcnow)
    trusted = Column(Boolean, default=True)
    
    federation = relationship("FederationDB", back_populates="clients")


class TrainingRoundDB(Base):
    """Training round records"""
    __tablename__ = 'training_rounds'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    federation_id = Column(UUID(as_uuid=True), ForeignKey('federations.id'))
    round_number = Column(Integer, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    participating_clients = Column(JSON)
    global_model_hash = Column(String(128))
    metrics = Column(JSON)
    aggregation_stats = Column(JSON)
    
    federation = relationship("FederationDB", back_populates="rounds")


class SecureAggregator:
    """Secure aggregation for federated learning"""
    
    def __init__(self):
        self.backend = default_backend()
        self.keys = {}
        self.shares = {}
    
    def generate_keys(self, num_clients: int) -> Dict[str, bytes]:
        """Generate keys for secure aggregation"""
        keys = {}
        for i in range(num_clients):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=self.backend
            )
            public_key = private_key.public_key()
            
            keys[f"client_{i}"] = {
                'private': private_key,
                'public': public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            }
        
        return keys
    
    def create_shares(self, model_weights: np.ndarray, num_shares: int) -> List[np.ndarray]:
        """Create secret shares of model weights"""
        shares = []
        
        # Generate random shares
        for i in range(num_shares - 1):
            share = np.random.randn(*model_weights.shape)
            shares.append(share)
        
        # Last share ensures sum equals original
        last_share = model_weights - sum(shares)
        shares.append(last_share)
        
        return shares
    
    def aggregate_shares(self, shares: List[np.ndarray]) -> np.ndarray:
        """Aggregate secret shares"""
        return np.sum(shares, axis=0)
    
    def encrypt_model(self, model_weights: np.ndarray, public_key: bytes) -> bytes:
        """Encrypt model weights"""
        # Serialize weights
        serialized = pickle.dumps(model_weights)
        
        # Load public key
        key = serialization.load_pem_public_key(public_key, backend=self.backend)
        
        # Encrypt in chunks (RSA has size limitations)
        chunk_size = 190  # Safe size for 2048-bit RSA
        encrypted_chunks = []
        
        for i in range(0, len(serialized), chunk_size):
            chunk = serialized[i:i+chunk_size]
            encrypted = key.encrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encrypted_chunks.append(encrypted)
        
        return b''.join(encrypted_chunks)
    
    def decrypt_model(self, encrypted_weights: bytes, private_key) -> np.ndarray:
        """Decrypt model weights"""
        # Decrypt chunks
        chunk_size = 256  # Encrypted chunk size for 2048-bit RSA
        decrypted_chunks = []
        
        for i in range(0, len(encrypted_weights), chunk_size):
            chunk = encrypted_weights[i:i+chunk_size]
            decrypted = private_key.decrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            decrypted_chunks.append(decrypted)
        
        # Deserialize
        serialized = b''.join(decrypted_chunks)
        return pickle.loads(serialized)


class HomomorphicAggregator:
    """Homomorphic encryption for secure aggregation"""
    
    def __init__(self):
        # Initialize CKKS context for homomorphic encryption
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
    
    def encrypt_weights(self, weights: np.ndarray) -> ts.CKKSVector:
        """Encrypt weights using homomorphic encryption"""
        # Flatten weights
        flat_weights = weights.flatten()
        
        # Encrypt
        encrypted = ts.ckks_vector(self.context, flat_weights)
        
        return encrypted
    
    def aggregate_encrypted(self, encrypted_weights: List[ts.CKKSVector]) -> ts.CKKSVector:
        """Aggregate encrypted weights"""
        result = encrypted_weights[0]
        
        for weights in encrypted_weights[1:]:
            result += weights
        
        # Average
        result *= 1.0 / len(encrypted_weights)
        
        return result
    
    def decrypt_weights(self, encrypted: ts.CKKSVector, shape: Tuple) -> np.ndarray:
        """Decrypt aggregated weights"""
        decrypted = encrypted.decrypt()
        return np.array(decrypted).reshape(shape)


class DifferentialPrivacy:
    """Differential privacy for federated learning"""
    
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.gaussian = GaussianMechanism(epsilon, delta, sensitivity)
        self.laplace = LaplaceMechanism(epsilon, sensitivity)
    
    def add_noise(self, gradients: np.ndarray, mechanism: str = 'gaussian') -> np.ndarray:
        """Add differential privacy noise to gradients"""
        if mechanism == 'gaussian':
            noise = self.gaussian.add_noise(gradients.shape)
        else:
            noise = self.laplace.add_noise(gradients.shape)
        
        return gradients + noise
    
    def clip_gradients(self, gradients: np.ndarray, max_norm: float) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)
        
        return gradients


class FederatedTrainingOrchestrator:
    """
    Orchestrates federated model training across distributed clients
    """
    
    def __init__(self,
                 db_url: str = "postgresql://localhost/ai_marketplace",
                 cache_enabled: bool = True):
        
        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize cache
        self.cache = redis.Redis(host='localhost', port=6379, db=4) if cache_enabled else None
        
        # Initialize aggregators
        self.secure_aggregator = SecureAggregator()
        self.homomorphic_aggregator = HomomorphicAggregator()
        
        # Active federations
        self.active_federations = {}
        
        # Client connections
        self.client_connections = {}
    
    async def create_federation(self, config: FederationConfig) -> str:
        """
        Create new federated training session
        
        Args:
            config: Federation configuration
        
        Returns:
            Federation ID
        """
        try:
            # Create database entry
            federation_db = FederationDB(
                federation_id=config.federation_id,
                model_id=config.model_id,
                strategy=config.strategy.value,
                status='initializing',
                config=asdict(config)
            )
            
            self.session.add(federation_db)
            self.session.commit()
            
            # Initialize federation state
            self.active_federations[config.federation_id] = {
                'config': config,
                'clients': {},
                'current_round': 0,
                'global_model': None,
                'metrics_history': []
            }
            
            # Generate encryption keys if needed
            if config.secure_aggregation:
                keys = self.secure_aggregator.generate_keys(config.max_clients)
                self.active_federations[config.federation_id]['keys'] = keys
            
            # Initialize differential privacy if needed
            if config.differential_privacy:
                dp = DifferentialPrivacy(
                    config.epsilon,
                    config.delta,
                    config.max_grad_norm
                )
                self.active_federations[config.federation_id]['dp'] = dp
            
            logger.info(f"Federation created: {config.federation_id}")
            return str(federation_db.id)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Federation creation failed: {e}")
            raise
    
    async def register_client(self, 
                            federation_id: str,
                            client_config: ClientConfig) -> bool:
        """
        Register client for federated training
        
        Args:
            federation_id: Federation ID
            client_config: Client configuration
        
        Returns:
            Registration success
        """
        try:
            # Get federation
            federation = self.session.query(FederationDB).filter_by(
                federation_id=federation_id
            ).first()
            
            if not federation:
                raise ValueError(f"Federation not found: {federation_id}")
            
            # Create client entry
            client_db = FederatedClientDB(
                federation_id=federation.id,
                client_id=client_config.client_id,
                data_size=client_config.data_size,
                compute_capability=client_config.compute_capability,
                network_bandwidth=client_config.network_bandwidth,
                trusted=client_config.trusted
            )
            
            self.session.add(client_db)
            self.session.commit()
            
            # Add to active federation
            if federation_id in self.active_federations:
                self.active_federations[federation_id]['clients'][client_config.client_id] = {
                    'config': client_config,
                    'status': ClientStatus.IDLE,
                    'current_model': None,
                    'metrics': {}
                }
            
            logger.info(f"Client {client_config.client_id} registered for federation {federation_id}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Client registration failed: {e}")
            raise
    
    async def start_training(self, federation_id: str) -> bool:
        """
        Start federated training
        
        Args:
            federation_id: Federation ID
        
        Returns:
            Success status
        """
        try:
            federation = self.active_federations.get(federation_id)
            if not federation:
                raise ValueError(f"Federation not found: {federation_id}")
            
            config = federation['config']
            
            # Check minimum clients
            if len(federation['clients']) < config.min_clients:
                raise ValueError(f"Insufficient clients: {len(federation['clients'])} < {config.min_clients}")
            
            # Initialize global model
            global_model = await self._initialize_global_model(config.model_id)
            federation['global_model'] = global_model
            
            # Start training rounds
            for round_num in range(config.rounds):
                logger.info(f"Starting round {round_num + 1}/{config.rounds}")
                
                # Select clients for this round
                selected_clients = await self._select_clients(federation_id, round_num)
                
                # Create training round
                round_db = TrainingRoundDB(
                    federation_id=federation_id,
                    round_number=round_num,
                    participating_clients=selected_clients,
                    start_time=datetime.utcnow()
                )
                self.session.add(round_db)
                
                # Distribute model to clients
                await self._distribute_model(federation_id, selected_clients, global_model)
                
                # Wait for client updates
                client_updates = await self._collect_client_updates(
                    federation_id,
                    selected_clients,
                    config.timeout_seconds
                )
                
                # Aggregate updates
                if config.strategy == FederationStrategy.SECURE_AGGREGATION:
                    global_model = await self._secure_aggregate(client_updates, config)
                elif config.strategy == FederationStrategy.HOMOMORPHIC_ENCRYPTION:
                    global_model = await self._homomorphic_aggregate(client_updates, config)
                elif config.strategy == FederationStrategy.DIFFERENTIAL_PRIVACY:
                    global_model = await self._dp_aggregate(client_updates, config)
                else:
                    global_model = await self._federated_average(client_updates, config)
                
                # Update global model
                federation['global_model'] = global_model
                federation['current_round'] = round_num + 1
                
                # Evaluate global model
                metrics = await self._evaluate_global_model(global_model, config)
                federation['metrics_history'].append(metrics)
                
                # Update round record
                round_db.end_time = datetime.utcnow()
                round_db.global_model_hash = self._hash_model(global_model)
                round_db.metrics = metrics
                
                self.session.commit()
                
                # Check convergence
                if await self._check_convergence(federation['metrics_history']):
                    logger.info(f"Model converged at round {round_num + 1}")
                    break
            
            # Finalize training
            await self._finalize_training(federation_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    async def get_global_model(self, federation_id: str) -> Any:
        """
        Get current global model
        
        Args:
            federation_id: Federation ID
        
        Returns:
            Global model
        """
        federation = self.active_federations.get(federation_id)
        if not federation:
            raise ValueError(f"Federation not found: {federation_id}")
        
        return federation['global_model']
    
    async def contribute_update(self,
                               federation_id: str,
                               client_id: str,
                               model_update: Any,
                               metrics: Dict[str, float]) -> bool:
        """
        Contribute model update from client
        
        Args:
            federation_id: Federation ID
            client_id: Client ID
            model_update: Model weights or gradients
            metrics: Training metrics
        
        Returns:
            Success status
        """
        try:
            federation = self.active_federations.get(federation_id)
            if not federation:
                raise ValueError(f"Federation not found: {federation_id}")
            
            client = federation['clients'].get(client_id)
            if not client:
                raise ValueError(f"Client not registered: {client_id}")
            
            # Validate update
            if not await self._validate_update(model_update, federation['config']):
                raise ValueError("Invalid model update")
            
            # Store update
            client['current_model'] = model_update
            client['metrics'] = metrics
            client['status'] = ClientStatus.AGGREGATING
            
            # Update client record
            client_db = self.session.query(FederatedClientDB).filter_by(
                client_id=client_id
            ).first()
            
            if client_db:
                client_db.last_active = datetime.utcnow()
                client_db.rounds_participated += 1
                self.session.commit()
            
            logger.info(f"Update received from client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Update contribution failed: {e}")
            raise
    
    async def _initialize_global_model(self, model_id: str) -> Any:
        """Initialize global model"""
        # Load model from registry
        # This is a placeholder - integrate with model registry
        return np.random.randn(100, 10)  # Example weights
    
    async def _select_clients(self, federation_id: str, round_num: int) -> List[str]:
        """Select clients for training round"""
        federation = self.active_federations[federation_id]
        config = federation['config']
        
        # Get available clients
        available_clients = [
            client_id for client_id, client in federation['clients'].items()
            if client['status'] != ClientStatus.FAILED
        ]
        
        # Random selection (can be improved with contribution-based selection)
        num_clients = min(len(available_clients), config.max_clients)
        selected = np.random.choice(available_clients, size=num_clients, replace=False)
        
        return selected.tolist()
    
    async def _distribute_model(self, 
                               federation_id: str,
                               clients: List[str],
                               model: Any):
        """Distribute model to selected clients"""
        federation = self.active_federations[federation_id]
        config = federation['config']
        
        for client_id in clients:
            client = federation['clients'][client_id]
            
            # Encrypt model if needed
            if config.model_encryption:
                if config.secure_aggregation:
                    # Use client's public key
                    client_key = federation['keys'][client_id]['public']
                    encrypted_model = self.secure_aggregator.encrypt_model(model, client_key)
                    client['current_model'] = encrypted_model
                else:
                    client['current_model'] = model
            else:
                client['current_model'] = model
            
            client['status'] = ClientStatus.TRAINING
            
            # Send notification to client (WebSocket/gRPC)
            await self._notify_client(client_id, 'model_ready')
    
    async def _collect_client_updates(self,
                                     federation_id: str,
                                     clients: List[str],
                                     timeout: int) -> Dict[str, Any]:
        """Collect updates from clients"""
        federation = self.active_federations[federation_id]
        updates = {}
        
        # Wait for updates with timeout
        start_time = datetime.utcnow()
        
        while len(updates) < len(clients):
            if (datetime.utcnow() - start_time).seconds > timeout:
                logger.warning(f"Timeout waiting for clients: {set(clients) - set(updates.keys())}")
                break
            
            for client_id in clients:
                if client_id in updates:
                    continue
                
                client = federation['clients'][client_id]
                if client['status'] == ClientStatus.AGGREGATING and client['current_model'] is not None:
                    updates[client_id] = {
                        'model': client['current_model'],
                        'metrics': client['metrics'],
                        'data_size': client['config'].data_size
                    }
            
            await asyncio.sleep(1)
        
        return updates
    
    async def _federated_average(self, 
                                updates: Dict[str, Any],
                                config: FederationConfig) -> np.ndarray:
        """Perform federated averaging"""
        total_samples = sum(u['data_size'] for u in updates.values())
        
        # Weighted average
        aggregated = None
        for client_id, update in updates.items():
            weight = update['data_size'] / total_samples
            
            if aggregated is None:
                aggregated = update['model'] * weight
            else:
                aggregated += update['model'] * weight
        
        return aggregated
    
    async def _secure_aggregate(self,
                              updates: Dict[str, Any],
                              config: FederationConfig) -> np.ndarray:
        """Perform secure aggregation"""
        # Collect encrypted shares
        shares = []
        
        for client_id, update in updates.items():
            # Decrypt share using server's private key
            # This is simplified - real implementation would use secure multi-party computation
            shares.append(update['model'])
        
        # Aggregate shares
        aggregated = self.secure_aggregator.aggregate_shares(shares)
        
        return aggregated
    
    async def _homomorphic_aggregate(self,
                                    updates: Dict[str, Any],
                                    config: FederationConfig) -> np.ndarray:
        """Perform homomorphic aggregation"""
        encrypted_updates = []
        
        for client_id, update in updates.items():
            # Updates should already be encrypted
            encrypted_updates.append(update['model'])
        
        # Aggregate encrypted
        aggregated_encrypted = self.homomorphic_aggregator.aggregate_encrypted(encrypted_updates)
        
        # Decrypt result
        shape = updates[list(updates.keys())[0]]['model'].shape
        aggregated = self.homomorphic_aggregator.decrypt_weights(aggregated_encrypted, shape)
        
        return aggregated
    
    async def _dp_aggregate(self,
                          updates: Dict[str, Any],
                          config: FederationConfig) -> np.ndarray:
        """Perform differential privacy aggregation"""
        federation = self.active_federations.get(config.federation_id)
        dp = federation['dp']
        
        # First perform standard aggregation
        aggregated = await self._federated_average(updates, config)
        
        # Clip gradients
        aggregated = dp.clip_gradients(aggregated, config.max_grad_norm)
        
        # Add noise
        aggregated = dp.add_noise(aggregated)
        
        return aggregated
    
    async def _evaluate_global_model(self, model: Any, config: FederationConfig) -> Dict[str, float]:
        """Evaluate global model"""
        # Placeholder evaluation
        return {
            'accuracy': np.random.uniform(0.7, 0.95),
            'loss': np.random.uniform(0.1, 0.5)
        }
    
    async def _check_convergence(self, metrics_history: List[Dict[str, float]]) -> bool:
        """Check if model has converged"""
        if len(metrics_history) < 3:
            return False
        
        # Check if loss is not improving
        recent_losses = [m.get('loss', float('inf')) for m in metrics_history[-3:]]
        improvement = recent_losses[-2] - recent_losses[-1]
        
        return improvement < 0.001
    
    async def _finalize_training(self, federation_id: str):
        """Finalize federated training"""
        federation = self.active_federations[federation_id]
        
        # Update database
        federation_db = self.session.query(FederationDB).filter_by(
            federation_id=federation_id
        ).first()
        
        if federation_db:
            federation_db.status = 'completed'
            federation_db.completed_at = datetime.utcnow()
            self.session.commit()
        
        # Calculate client contributions
        await self._calculate_contributions(federation_id)
        
        # Clean up
        if federation_id in self.active_federations:
            del self.active_federations[federation_id]
    
    async def _validate_update(self, update: Any, config: FederationConfig) -> bool:
        """Validate client update"""
        # Check update format and size
        # Implement validation logic
        return True
    
    async def _notify_client(self, client_id: str, message: str):
        """Notify client via WebSocket/gRPC"""
        # Implementation depends on communication protocol
        pass
    
    def _hash_model(self, model: Any) -> str:
        """Calculate hash of model weights"""
        serialized = pickle.dumps(model)
        return hashlib.sha256(serialized).hexdigest()
    
    async def _calculate_contributions(self, federation_id: str):
        """Calculate client contributions for incentivization"""
        federation = self.active_federations.get(federation_id)
        if not federation:
            return
        
        for client_id, client in federation['clients'].items():
            client_db = self.session.query(FederatedClientDB).filter_by(
                client_id=client_id
            ).first()
            
            if client_db:
                # Calculate contribution score based on:
                # - Number of rounds participated
                # - Data size contributed
                # - Model improvement
                score = client_db.rounds_participated * 0.5
                score += (client['config'].data_size / 1000) * 0.3
                score += len(client.get('metrics', {})) * 0.2
                
                client_db.contribution_score = score
        
        self.session.commit()


class GaussianMechanism:
    """Gaussian mechanism for differential privacy"""
    
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._calculate_sigma()
    
    def _calculate_sigma(self) -> float:
        """Calculate noise standard deviation"""
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise(self, shape: Tuple) -> np.ndarray:
        """Generate Gaussian noise"""
        return np.random.normal(0, self.sigma, shape)


class LaplaceMechanism:
    """Laplace mechanism for differential privacy"""
    
    def __init__(self, epsilon: float, sensitivity: float):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
    
    def add_noise(self, shape: Tuple) -> np.ndarray:
        """Generate Laplace noise"""
        return np.random.laplace(0, self.scale, shape)