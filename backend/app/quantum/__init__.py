"""
Quantum-Resistant Cryptography Module

Provides post-quantum cryptographic algorithms and security implementations
resistant to both classical and quantum computer attacks.
"""

from .post_quantum import (
    PostQuantumCrypto,
    KyberKEM,
    DilithiumSignature,
    FalconSignature,
    SPHINCSPlusSignature,
    NTRUEncryption,
    ClassicMcEliece
)

from .key_exchange import (
    QuantumSafeKeyExchange,
    HybridKeyExchange,
    ThresholdKeyGeneration
)

from .signatures import (
    QuantumResistantSigner,
    AggregateSignatures,
    BlindSignatures,
    RingSignatures
)

from .migration import (
    CryptoMigrationManager,
    AlgorithmAgility,
    BackwardCompatibility
)

from .hybrid import (
    HybridCryptoSystem,
    ClassicalQuantumBridge,
    FallbackMechanisms
)

__all__ = [
    'PostQuantumCrypto',
    'KyberKEM',
    'DilithiumSignature',
    'FalconSignature',
    'SPHINCSPlusSignature',
    'NTRUEncryption',
    'ClassicMcEliece',
    'QuantumSafeKeyExchange',
    'HybridKeyExchange',
    'ThresholdKeyGeneration',
    'QuantumResistantSigner',
    'AggregateSignatures',
    'BlindSignatures',
    'RingSignatures',
    'CryptoMigrationManager',
    'AlgorithmAgility',
    'BackwardCompatibility',
    'HybridCryptoSystem',
    'ClassicalQuantumBridge',
    'FallbackMechanisms'
]

__version__ = '1.0.0'