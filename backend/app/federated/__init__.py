"""
Federated Learning System for Privacy-Preserving ML

This module implements a comprehensive federated learning framework for training
machine learning models across distributed edge devices while preserving privacy.

Features:
- Federated Averaging (FedAvg) and advanced algorithms
- Differential privacy and secure aggregation
- Edge deployment with TensorFlow Lite and ONNX
- Real-time monitoring and orchestration
- Support for heterogeneous networks

Supported Frameworks:
- TensorFlow Federated (TFF)
- PySyft for secure computation
- Flower framework for federated learning
"""

from .server import FederatedServer
from .client import FederatedClient
from .aggregation import ModelAggregator
from .privacy import PrivacyEngine
from .communication import SecureCommunication
from .monitoring import FederatedMonitor
from .orchestration import FederatedOrchestrator

__version__ = "1.0.0"
__author__ = "Federated Learning Team"

__all__ = [
    "FederatedServer",
    "FederatedClient", 
    "ModelAggregator",
    "PrivacyEngine",
    "SecureCommunication",
    "FederatedMonitor",
    "FederatedOrchestrator"
]