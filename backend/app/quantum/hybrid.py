"""
Hybrid Classical-Quantum Cryptography

Combines classical and post-quantum algorithms for defense in depth:
- Hybrid encryption schemes
- Fallback mechanisms
- Performance optimization
- Gradual transition support
"""

import os
import time
import hashlib
import hmac
import secrets
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
import concurrent.futures
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from .post_quantum import (
    KyberKEM,
    DilithiumSignature,
    FalconSignature,
    NTRUEncryption,
    SecurityLevel,
    QuantumRandomNumberGenerator
)


class HybridMode(Enum):
    """Hybrid cryptography modes"""
    CONCATENATE = "concatenate"      # Simply combine both
    XOR = "xor"                      # XOR the outputs
    NESTED = "nested"                # Encrypt with one, then the other
    PARALLEL = "parallel"            # Run both in parallel
    SEQUENTIAL = "sequential"        # Chain the algorithms
    ADAPTIVE = "adaptive"            # Choose based on threat level


class ThreatLevel(Enum):
    """Current quantum threat level"""
    NONE = "none"                   # No quantum threat
    EMERGING = "emerging"            # Early quantum computers
    MODERATE = "moderate"            # Moderate quantum capability
    HIGH = "high"                    # Advanced quantum computers
    CRITICAL = "critical"            # Full-scale quantum attack


@dataclass
class HybridConfig:
    """Configuration for hybrid cryptography"""
    mode: HybridMode
    classical_algorithm: str
    quantum_algorithm: str
    threat_level: ThreatLevel
    performance_priority: float  # 0.0 (security) to 1.0 (performance)
    fallback_enabled: bool
    cache_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class HybridResult:
    """Result of hybrid cryptographic operation"""
    success: bool
    data: bytes
    algorithm_used: str
    performance_ms: float
    security_score: float
    metadata: Dict[str, Any]


class HybridCryptoSystem:
    """
    Main hybrid cryptography system combining classical and quantum-safe algorithms
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.qrng = QuantumRandomNumberGenerator()
        
        # Initialize algorithms
        self._initialize_algorithms()
        
        # Performance cache
        self.performance_cache = {}
        
        # Security monitoring
        self.security_events = []
        
        # Fallback state
        self.fallback_active = False
        
    def _initialize_algorithms(self):
        """Initialize both classical and quantum algorithms"""
        # Quantum algorithms
        if self.config.quantum_algorithm == "Kyber":
            self.quantum_kem = KyberKEM(SecurityLevel.LEVEL3)
        
        if "Dilithium" in self.config.quantum_algorithm:
            self.quantum_signer = DilithiumSignature(SecurityLevel.LEVEL3)
        elif "FALCON" in self.config.quantum_algorithm:
            self.quantum_signer = FalconSignature(SecurityLevel.LEVEL5)
        
        # Classical algorithms (simulated)
        self.classical_cipher = AESGCM(self.qrng.get_random_bytes(32))
        
    def hybrid_encrypt(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> HybridResult:
        """
        Encrypt using hybrid scheme
        public_keys: {'classical': bytes, 'quantum': bytes}
        """
        start_time = time.time()
        
        try:
            if self.config.mode == HybridMode.CONCATENATE:
                result = self._encrypt_concatenate(plaintext, public_keys)
            elif self.config.mode == HybridMode.XOR:
                result = self._encrypt_xor(plaintext, public_keys)
            elif self.config.mode == HybridMode.NESTED:
                result = self._encrypt_nested(plaintext, public_keys)
            elif self.config.mode == HybridMode.PARALLEL:
                result = self._encrypt_parallel(plaintext, public_keys)
            elif self.config.mode == HybridMode.SEQUENTIAL:
                result = self._encrypt_sequential(plaintext, public_keys)
            elif self.config.mode == HybridMode.ADAPTIVE:
                result = self._encrypt_adaptive(plaintext, public_keys)
            else:
                raise ValueError(f"Unsupported mode: {self.config.mode}")
            
            performance_ms = (time.time() - start_time) * 1000
            
            return HybridResult(
                success=True,
                data=result,
                algorithm_used=f"{self.config.classical_algorithm}+{self.config.quantum_algorithm}",
                performance_ms=performance_ms,
                security_score=self._calculate_security_score(),
                metadata={'mode': self.config.mode.value}
            )
            
        except Exception as e:
            if self.config.fallback_enabled:
                return self._fallback_encrypt(plaintext, public_keys)
            raise
    
    def _encrypt_concatenate(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Concatenate encryption - both ciphertexts combined"""
        # Generate shared secrets
        quantum_ct, quantum_secret = self.quantum_kem.encapsulate(public_keys['quantum'])
        classical_secret = self._classical_key_exchange(public_keys['classical'])
        
        # Derive encryption keys
        quantum_key = self._derive_key(quantum_secret, b"quantum")
        classical_key = self._derive_key(classical_secret, b"classical")
        
        # Encrypt with both
        quantum_encrypted = self._symmetric_encrypt(plaintext, quantum_key)
        classical_encrypted = self._symmetric_encrypt(plaintext, classical_key)
        
        # Combine ciphertexts
        combined = {
            'quantum_ct': base64.b64encode(quantum_ct).decode(),
            'quantum_encrypted': base64.b64encode(quantum_encrypted).decode(),
            'classical_encrypted': base64.b64encode(classical_encrypted).decode(),
            'mode': 'concatenate'
        }
        
        return json.dumps(combined).encode()
    
    def _encrypt_xor(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """XOR encryption - combine keys via XOR"""
        # Generate shared secrets
        quantum_ct, quantum_secret = self.quantum_kem.encapsulate(public_keys['quantum'])
        classical_secret = self._classical_key_exchange(public_keys['classical'])
        
        # XOR the secrets
        min_len = min(len(quantum_secret), len(classical_secret))
        combined_secret = bytes(q ^ c for q, c in zip(quantum_secret[:min_len], classical_secret[:min_len]))
        
        # Derive final key
        final_key = self._derive_key(combined_secret, b"hybrid_xor")
        
        # Encrypt once with combined key
        encrypted = self._symmetric_encrypt(plaintext, final_key)
        
        result = {
            'quantum_ct': base64.b64encode(quantum_ct).decode(),
            'encrypted': base64.b64encode(encrypted).decode(),
            'mode': 'xor'
        }
        
        return json.dumps(result).encode()
    
    def _encrypt_nested(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Nested encryption - encrypt with one, then the other"""
        # First layer: quantum encryption
        quantum_ct, quantum_secret = self.quantum_kem.encapsulate(public_keys['quantum'])
        quantum_key = self._derive_key(quantum_secret, b"quantum_layer")
        
        first_layer = self._symmetric_encrypt(plaintext, quantum_key)
        
        # Second layer: classical encryption
        classical_secret = self._classical_key_exchange(public_keys['classical'])
        classical_key = self._derive_key(classical_secret, b"classical_layer")
        
        second_layer = self._symmetric_encrypt(first_layer, classical_key)
        
        result = {
            'quantum_ct': base64.b64encode(quantum_ct).decode(),
            'nested_ciphertext': base64.b64encode(second_layer).decode(),
            'mode': 'nested',
            'layers': ['quantum', 'classical']
        }
        
        return json.dumps(result).encode()
    
    def _encrypt_parallel(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Parallel encryption - run both simultaneously"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both encryption tasks
            quantum_future = executor.submit(
                self._quantum_encrypt_task, plaintext, public_keys['quantum']
            )
            classical_future = executor.submit(
                self._classical_encrypt_task, plaintext, public_keys['classical']
            )
            
            # Wait for both to complete
            quantum_result = quantum_future.result()
            classical_result = classical_future.result()
        
        # Combine results
        combined = {
            'quantum': quantum_result,
            'classical': classical_result,
            'mode': 'parallel'
        }
        
        return json.dumps(combined).encode()
    
    def _encrypt_sequential(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Sequential encryption - chain the algorithms"""
        # Create encryption chain
        chain = []
        
        # Step 1: Quantum KEM
        quantum_ct, quantum_secret = self.quantum_kem.encapsulate(public_keys['quantum'])
        chain.append(('quantum_kem', base64.b64encode(quantum_ct).decode()))
        
        # Step 2: Classical key exchange
        classical_secret = self._classical_key_exchange(public_keys['classical'])
        
        # Step 3: Combine secrets sequentially
        h = hashlib.sha3_512()
        h.update(quantum_secret)
        h.update(classical_secret)
        combined_secret = h.digest()
        
        # Step 4: Multi-layer encryption
        current = plaintext
        for i, (secret_part, label) in enumerate([(quantum_secret, 'q'), (classical_secret, 'c')]):
            key = self._derive_key(secret_part, f"layer_{label}".encode())
            current = self._symmetric_encrypt(current, key)
            chain.append((f'encrypt_{label}', len(current)))
        
        result = {
            'ciphertext': base64.b64encode(current).decode(),
            'chain': chain,
            'mode': 'sequential'
        }
        
        return json.dumps(result).encode()
    
    def _encrypt_adaptive(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Adaptive encryption - adjust based on threat level"""
        # Choose strategy based on threat level
        if self.config.threat_level == ThreatLevel.CRITICAL:
            # Maximum security - use nested encryption
            return self._encrypt_nested(plaintext, public_keys)
        elif self.config.threat_level == ThreatLevel.HIGH:
            # High security - use XOR combination
            return self._encrypt_xor(plaintext, public_keys)
        elif self.config.threat_level == ThreatLevel.MODERATE:
            # Balanced - use concatenation
            return self._encrypt_concatenate(plaintext, public_keys)
        else:
            # Low threat - optimize for performance
            if self.config.performance_priority > 0.5:
                # Use faster classical with quantum backup
                return self._encrypt_with_quantum_backup(plaintext, public_keys)
            else:
                # Use quantum with classical backup
                return self._encrypt_with_classical_backup(plaintext, public_keys)
    
    def _encrypt_with_quantum_backup(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Primarily classical with quantum backup"""
        # Classical encryption
        classical_secret = self._classical_key_exchange(public_keys['classical'])
        classical_key = self._derive_key(classical_secret, b"primary")
        encrypted = self._symmetric_encrypt(plaintext, classical_key)
        
        # Quantum backup (lighter)
        quantum_ct, quantum_secret = self.quantum_kem.encapsulate(public_keys['quantum'])
        
        # Create integrity tag with quantum
        h = hashlib.sha3_256()
        h.update(quantum_secret)
        h.update(encrypted)
        quantum_tag = h.digest()
        
        result = {
            'encrypted': base64.b64encode(encrypted).decode(),
            'quantum_ct': base64.b64encode(quantum_ct).decode(),
            'quantum_tag': base64.b64encode(quantum_tag).decode(),
            'primary': 'classical',
            'mode': 'adaptive'
        }
        
        return json.dumps(result).encode()
    
    def _encrypt_with_classical_backup(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> bytes:
        """Primarily quantum with classical backup"""
        # Quantum encryption
        quantum_ct, quantum_secret = self.quantum_kem.encapsulate(public_keys['quantum'])
        quantum_key = self._derive_key(quantum_secret, b"primary")
        encrypted = self._symmetric_encrypt(plaintext, quantum_key)
        
        # Classical backup (lighter)
        classical_secret = self._classical_key_exchange(public_keys['classical'])
        
        # Create integrity tag with classical
        h = hashlib.sha256()
        h.update(classical_secret)
        h.update(encrypted)
        classical_tag = h.digest()
        
        result = {
            'encrypted': base64.b64encode(encrypted).decode(),
            'quantum_ct': base64.b64encode(quantum_ct).decode(),
            'classical_tag': base64.b64encode(classical_tag).decode(),
            'primary': 'quantum',
            'mode': 'adaptive'
        }
        
        return json.dumps(result).encode()
    
    def _quantum_encrypt_task(self, plaintext: bytes, public_key: bytes) -> Dict[str, Any]:
        """Task for parallel quantum encryption"""
        ct, secret = self.quantum_kem.encapsulate(public_key)
        key = self._derive_key(secret, b"quantum_parallel")
        encrypted = self._symmetric_encrypt(plaintext, key)
        
        return {
            'ciphertext': base64.b64encode(ct).decode(),
            'encrypted': base64.b64encode(encrypted).decode()
        }
    
    def _classical_encrypt_task(self, plaintext: bytes, public_key: bytes) -> Dict[str, Any]:
        """Task for parallel classical encryption"""
        secret = self._classical_key_exchange(public_key)
        key = self._derive_key(secret, b"classical_parallel")
        encrypted = self._symmetric_encrypt(plaintext, key)
        
        return {
            'encrypted': base64.b64encode(encrypted).decode()
        }
    
    def _classical_key_exchange(self, public_key: bytes) -> bytes:
        """Simulated classical key exchange (e.g., ECDH)"""
        h = hashlib.sha256()
        h.update(public_key)
        h.update(self.qrng.get_random_bytes(32))
        return h.digest()
    
    def _derive_key(self, secret: bytes, context: bytes) -> bytes:
        """Derive encryption key from shared secret"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=context,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(secret)
    
    def _symmetric_encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Symmetric encryption using AES-GCM or ChaCha20-Poly1305"""
        nonce = self.qrng.get_random_bytes(12)
        
        if self.config.performance_priority > 0.7:
            # Use ChaCha20-Poly1305 for better performance
            cipher = ChaCha20Poly1305(key)
        else:
            # Use AES-GCM for compatibility
            cipher = AESGCM(key)
        
        ciphertext = cipher.encrypt(nonce, plaintext, None)
        return nonce + ciphertext
    
    def _calculate_security_score(self) -> float:
        """Calculate security score of current configuration"""
        score = 0.0
        
        # Quantum resistance
        if "Kyber" in self.config.quantum_algorithm:
            score += 30
        if "Dilithium" in self.config.quantum_algorithm:
            score += 20
        
        # Hybrid mode bonus
        if self.config.mode in [HybridMode.NESTED, HybridMode.XOR]:
            score += 20
        elif self.config.mode == HybridMode.CONCATENATE:
            score += 15
        
        # Threat level adjustment
        if self.config.threat_level == ThreatLevel.CRITICAL:
            score += 20
        elif self.config.threat_level == ThreatLevel.HIGH:
            score += 10
        
        # Normalize to 0-100
        return min(score, 100.0)
    
    def _fallback_encrypt(self, plaintext: bytes, public_keys: Dict[str, bytes]) -> HybridResult:
        """Fallback encryption when primary fails"""
        self.fallback_active = True
        
        try:
            # Try simpler encryption
            if 'classical' in public_keys:
                secret = self._classical_key_exchange(public_keys['classical'])
                key = self._derive_key(secret, b"fallback")
                encrypted = self._symmetric_encrypt(plaintext, key)
                
                return HybridResult(
                    success=True,
                    data=encrypted,
                    algorithm_used="classical_fallback",
                    performance_ms=0,
                    security_score=50.0,
                    metadata={'fallback': True}
                )
        except:
            pass
        
        return HybridResult(
            success=False,
            data=b'',
            algorithm_used="none",
            performance_ms=0,
            security_score=0,
            metadata={'error': 'All encryption methods failed'}
        )


class ClassicalQuantumBridge:
    """
    Bridge between classical and quantum cryptographic systems
    """
    
    def __init__(self):
        self.protocol_mappings = self._initialize_mappings()
        self.conversion_cache = {}
        
    def _initialize_mappings(self):
        """Initialize protocol mappings between classical and quantum"""
        return {
            'RSA-2048': 'Kyber512',
            'RSA-3072': 'Kyber768',
            'RSA-4096': 'Kyber1024',
            'ECDSA-P256': 'Dilithium2',
            'ECDSA-P384': 'Dilithium3',
            'ECDSA-P521': 'Dilithium5',
            'ECDH-P256': 'Kyber512',
            'ECDH-P384': 'Kyber768',
            'EdDSA': 'FALCON-512'
        }
    
    def convert_key_format(self, 
                          key_data: bytes,
                          from_format: str,
                          to_format: str) -> bytes:
        """Convert key between classical and quantum formats"""
        cache_key = f"{from_format}_{to_format}_{hashlib.sha256(key_data).hexdigest()[:8]}"
        
        if cache_key in self.conversion_cache:
            return self.conversion_cache[cache_key]
        
        # Perform conversion (simplified)
        h = hashlib.sha3_512()
        h.update(key_data)
        h.update(from_format.encode())
        h.update(to_format.encode())
        
        converted = h.digest()
        
        # Adjust size based on target format
        if 'Kyber' in to_format:
            if '512' in to_format:
                converted = converted[:800]  # Kyber512 public key size
            elif '768' in to_format:
                converted = converted[:1184]  # Kyber768 public key size
            else:
                converted = converted[:1568]  # Kyber1024 public key size
        
        self.conversion_cache[cache_key] = converted
        return converted
    
    def create_transition_wrapper(self,
                                 classical_data: bytes,
                                 quantum_data: bytes) -> bytes:
        """Create wrapper for transitional period supporting both"""
        wrapper = {
            'version': 1,
            'classical': base64.b64encode(classical_data).decode(),
            'quantum': base64.b64encode(quantum_data).decode(),
            'timestamp': time.time(),
            'transition_phase': True
        }
        
        return json.dumps(wrapper).encode()
    
    def negotiate_protocol(self,
                          client_capabilities: List[str],
                          server_capabilities: List[str]) -> str:
        """Negotiate best protocol between client and server"""
        # Find common capabilities
        common = set(client_capabilities) & set(server_capabilities)
        
        # Prefer quantum-safe
        for quantum_algo in ['Kyber1024', 'Kyber768', 'Dilithium3', 'FALCON-512']:
            if quantum_algo in common:
                return quantum_algo
        
        # Fall back to best classical
        for classical_algo in ['ECDSA-P384', 'ECDSA-P256', 'RSA-3072']:
            if classical_algo in common:
                return classical_algo
        
        # Default fallback
        return list(common)[0] if common else 'RSA-2048'


class FallbackMechanisms:
    """
    Fallback mechanisms for resilience
    """
    
    def __init__(self):
        self.fallback_chain = []
        self.failure_history = []
        self.retry_config = {
            'max_retries': 3,
            'backoff_factor': 2.0,
            'max_backoff': 30.0
        }
    
    def register_fallback(self, 
                         primary: str,
                         fallback: str,
                         condition: Optional[str] = None):
        """Register a fallback option"""
        self.fallback_chain.append({
            'primary': primary,
            'fallback': fallback,
            'condition': condition,
            'failure_count': 0
        })
    
    def execute_with_fallback(self,
                             operation,
                             *args,
                             **kwargs) -> Any:
        """Execute operation with automatic fallback"""
        last_error = None
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.failure_history.append({
                    'operation': operation.__name__,
                    'error': str(e),
                    'timestamp': time.time(),
                    'attempt': attempt + 1
                })
                
                # Exponential backoff
                if attempt < self.retry_config['max_retries'] - 1:
                    backoff = min(
                        self.retry_config['backoff_factor'] ** attempt,
                        self.retry_config['max_backoff']
                    )
                    time.sleep(backoff)
        
        # All retries failed
        raise last_error
    
    def get_fallback_algorithm(self, failed_algorithm: str) -> Optional[str]:
        """Get fallback algorithm for a failed one"""
        for item in self.fallback_chain:
            if item['primary'] == failed_algorithm:
                item['failure_count'] += 1
                return item['fallback']
        return None
    
    def reset_failure_counts(self):
        """Reset failure counts after successful period"""
        for item in self.fallback_chain:
            item['failure_count'] = 0
        self.failure_history = []


class PerformanceOptimizer:
    """
    Optimize performance of hybrid cryptography
    """
    
    def __init__(self):
        self.performance_metrics = {}
        self.optimization_hints = {}
    
    def measure_algorithm_performance(self, algorithm: str, operation: str) -> float:
        """Measure performance of specific algorithm operation"""
        start = time.perf_counter()
        
        # Simulate operation (would be actual crypto operation)
        if 'Kyber' in algorithm:
            time.sleep(0.001)  # Simulate fast KEM
        elif 'Dilithium' in algorithm:
            time.sleep(0.002)  # Simulate signature
        elif 'McEliece' in algorithm:
            time.sleep(0.005)  # Simulate large key operations
        else:
            time.sleep(0.0015)  # Default
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Store metrics
        key = f"{algorithm}_{operation}"
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []
        self.performance_metrics[key].append(elapsed_ms)
        
        return elapsed_ms
    
    def get_optimization_hints(self, use_case: str) -> Dict[str, Any]:
        """Get optimization hints for specific use case"""
        hints = {
            'web_api': {
                'recommended_algorithm': 'Kyber512',
                'cache_size': 1000,
                'parallel_operations': True,
                'precompute_keys': True
            },
            'file_encryption': {
                'recommended_algorithm': 'Kyber768',
                'chunk_size': 65536,
                'parallel_chunks': 4,
                'compression': 'before_encryption'
            },
            'real_time': {
                'recommended_algorithm': 'FALCON-512',
                'minimize_latency': True,
                'use_hardware_acceleration': True,
                'avoid_algorithms': ['McEliece', 'SPHINCS+']
            }
        }
        
        return hints.get(use_case, hints['web_api'])
    
    def auto_tune(self, workload_profile: Dict[str, int]) -> Dict[str, Any]:
        """Auto-tune configuration based on workload"""
        total_ops = sum(workload_profile.values())
        
        config = {
            'algorithm': 'Kyber768',  # Default
            'mode': HybridMode.CONCATENATE,
            'cache_size': 100,
            'parallel': False
        }
        
        # Tune based on workload
        if total_ops > 10000:
            config['parallel'] = True
            config['cache_size'] = 1000
        
        if workload_profile.get('signatures', 0) > workload_profile.get('encryption', 0):
            config['algorithm'] = 'FALCON-512'  # Optimized for signatures
        
        if workload_profile.get('large_files', 0) > 100:
            config['mode'] = HybridMode.PARALLEL
        
        return config


if __name__ == "__main__":
    print("Hybrid Classical-Quantum Cryptography Demo")
    print("=" * 50)
    
    # Configure hybrid system
    config = HybridConfig(
        mode=HybridMode.ADAPTIVE,
        classical_algorithm="ECDH-P256",
        quantum_algorithm="Kyber768",
        threat_level=ThreatLevel.MODERATE,
        performance_priority=0.5,
        fallback_enabled=True,
        cache_size=100
    )
    
    # Initialize hybrid crypto system
    hybrid = HybridCryptoSystem(config)
    
    # Generate keys
    qkem = KyberKEM(SecurityLevel.LEVEL3)
    quantum_pub, quantum_priv = qkem.generate_keypair()
    classical_pub = os.urandom(32)  # Simulated classical public key
    
    public_keys = {
        'quantum': quantum_pub,
        'classical': classical_pub
    }
    
    # Test encryption
    message = b"This is a secret message protected by hybrid cryptography!"
    
    print(f"\nEncrypting with {config.mode.value} mode...")
    result = hybrid.hybrid_encrypt(message, public_keys)
    
    print(f"Success: {result.success}")
    print(f"Algorithm: {result.algorithm_used}")
    print(f"Performance: {result.performance_ms:.2f} ms")
    print(f"Security Score: {result.security_score}/100")
    print(f"Ciphertext Size: {len(result.data)} bytes")
    
    # Test bridge
    print("\n\nClassical-Quantum Bridge")
    print("-" * 30)
    
    bridge = ClassicalQuantumBridge()
    
    # Convert RSA key to Kyber format
    rsa_key = os.urandom(256)  # Simulated RSA key
    kyber_key = bridge.convert_key_format(rsa_key, "RSA-2048", "Kyber512")
    print(f"Converted RSA-2048 ({len(rsa_key)} bytes) to Kyber512 ({len(kyber_key)} bytes)")
    
    # Protocol negotiation
    client_caps = ["RSA-2048", "Kyber768", "ECDSA-P256"]
    server_caps = ["Kyber768", "Dilithium3", "RSA-3072"]
    
    negotiated = bridge.negotiate_protocol(client_caps, server_caps)
    print(f"Negotiated Protocol: {negotiated}")
    
    # Performance optimization
    print("\n\nPerformance Optimization")
    print("-" * 30)
    
    optimizer = PerformanceOptimizer()
    
    # Measure performance
    for algo in ["Kyber512", "Dilithium3", "McEliece"]:
        perf = optimizer.measure_algorithm_performance(algo, "encrypt")
        print(f"{algo}: {perf:.2f} ms")
    
    # Get optimization hints
    hints = optimizer.get_optimization_hints("web_api")
    print(f"\nOptimization hints for web API:")
    for key, value in hints.items():
        print(f"  {key}: {value}")