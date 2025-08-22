"""
Blockchain integration module for audit trail system.

This module provides blockchain-based audit trail functionality including:
- Ethereum smart contract integration
- Hyperledger Fabric private blockchain
- IPFS distributed document storage
- Immutable audit records
- Tamper detection and verification
"""

from .ethereum import EthereumAuditTrail
from .hyperledger import HyperledgerAuditTrail
from .ipfs import IPFSStorage
from .monitoring import BlockchainMonitor

__all__ = [
    'EthereumAuditTrail',
    'HyperledgerAuditTrail', 
    'IPFSStorage',
    'BlockchainMonitor'
]