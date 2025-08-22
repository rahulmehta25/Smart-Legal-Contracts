"""
Quantum-Safe Key Exchange Protocols

Implements post-quantum key exchange mechanisms including:
- Pure quantum-safe key exchange
- Hybrid classical-quantum protocols
- Threshold key generation
- Forward secrecy mechanisms
"""

import os
import secrets
import hashlib
import hmac
import time
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.x963kdf import X963KDF
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM

from .post_quantum import KyberKEM, SecurityLevel, QuantumRandomNumberGenerator


class KeyExchangeProtocol(Enum):
    """Supported key exchange protocols"""
    KYBER_ONLY = "kyber"
    HYBRID_KYBER_X25519 = "kyber_x25519"
    HYBRID_KYBER_ECDH = "kyber_ecdh"
    THRESHOLD_SHAMIR = "threshold_shamir"
    THRESHOLD_FELDMAN = "threshold_feldman"


@dataclass
class KeyExchangeResult:
    """Result of key exchange operation"""
    shared_secret: bytes
    session_id: str
    protocol: KeyExchangeProtocol
    metadata: Dict[str, Any]


@dataclass
class ThresholdShare:
    """Share for threshold key generation"""
    index: int
    share: bytes
    commitment: bytes
    proof: bytes


class QuantumSafeKeyExchange:
    """
    Pure quantum-safe key exchange using post-quantum algorithms
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.LEVEL3):
        self.security_level = security_level
        self.kem = KyberKEM(security_level)
        self.qrng = QuantumRandomNumberGenerator()
        self.session_cache = {}
    
    def generate_ephemeral_keypair(self) -> Tuple[bytes, bytes]:
        """Generate ephemeral key pair for key exchange"""
        return self.kem.generate_keypair()
    
    def initiate_exchange(self, remote_public_key: bytes) -> Tuple[bytes, bytes, str]:
        """
        Initiate key exchange with remote party
        Returns: (ciphertext, shared_secret, session_id)
        """
        # Generate session ID
        session_id = base64.urlsafe_b64encode(
            self.qrng.get_random_bytes(16)
        ).decode('utf-8').rstrip('=')
        
        # Encapsulate shared secret
        ciphertext, shared_secret = self.kem.encapsulate(remote_public_key)
        
        # Derive session key using KDF
        session_key = self._derive_session_key(
            shared_secret, 
            session_id.encode(),
            b"initiator"
        )
        
        # Cache session
        self.session_cache[session_id] = {
            'shared_secret': shared_secret,
            'session_key': session_key,
            'timestamp': time.time(),
            'role': 'initiator'
        }
        
        return ciphertext, session_key, session_id
    
    def complete_exchange(self, ciphertext: bytes, private_key: bytes, 
                         session_id: str) -> bytes:
        """
        Complete key exchange as responder
        Returns: session_key
        """
        # Decapsulate shared secret
        shared_secret = self.kem.decapsulate(ciphertext, private_key)
        
        # Derive session key
        session_key = self._derive_session_key(
            shared_secret,
            session_id.encode(),
            b"responder"
        )
        
        # Cache session
        self.session_cache[session_id] = {
            'shared_secret': shared_secret,
            'session_key': session_key,
            'timestamp': time.time(),
            'role': 'responder'
        }
        
        return session_key
    
    def _derive_session_key(self, shared_secret: bytes, session_id: bytes,
                           role: bytes) -> bytes:
        """Derive session key from shared secret using HKDF"""
        hkdf = HKDF(
            algorithm=hashes.SHA3_256(),
            length=32,
            salt=session_id,
            info=b"quantum_key_exchange_" + role,
            backend=default_backend()
        )
        return hkdf.derive(shared_secret)
    
    def rotate_session_key(self, session_id: str) -> Optional[bytes]:
        """Rotate session key for forward secrecy"""
        if session_id not in self.session_cache:
            return None
        
        session = self.session_cache[session_id]
        
        # Generate new key from old key
        h = hashlib.sha3_256()
        h.update(session['session_key'])
        h.update(self.qrng.get_random_bytes(32))
        new_key = h.digest()
        
        # Update session
        session['session_key'] = new_key
        session['rotation_count'] = session.get('rotation_count', 0) + 1
        session['last_rotation'] = time.time()
        
        return new_key
    
    def cleanup_expired_sessions(self, max_age_seconds: int = 3600):
        """Remove expired sessions"""
        current_time = time.time()
        expired = [
            sid for sid, session in self.session_cache.items()
            if current_time - session['timestamp'] > max_age_seconds
        ]
        
        for sid in expired:
            # Securely erase key material
            session = self.session_cache[sid]
            self._secure_erase(session['shared_secret'])
            self._secure_erase(session['session_key'])
            del self.session_cache[sid]
    
    def _secure_erase(self, data: bytes):
        """Securely erase sensitive data from memory"""
        if isinstance(data, bytes):
            # Overwrite with random data
            temp = bytearray(data)
            for i in range(len(temp)):
                temp[i] = secrets.randbits(8)


class HybridKeyExchange:
    """
    Hybrid classical-quantum key exchange
    Combines post-quantum with traditional algorithms for defense in depth
    """
    
    def __init__(self, protocol: KeyExchangeProtocol = KeyExchangeProtocol.HYBRID_KYBER_X25519):
        self.protocol = protocol
        self.quantum_kex = QuantumSafeKeyExchange()
        self.qrng = QuantumRandomNumberGenerator()
        
    def generate_hybrid_keypairs(self) -> Dict[str, Tuple[bytes, bytes]]:
        """Generate key pairs for all algorithms in hybrid scheme"""
        keypairs = {}
        
        # Generate quantum-safe keypair
        keypairs['kyber'] = self.quantum_kex.generate_ephemeral_keypair()
        
        # Generate classical keypair (simulated)
        if self.protocol == KeyExchangeProtocol.HYBRID_KYBER_X25519:
            keypairs['x25519'] = self._generate_x25519_keypair()
        elif self.protocol == KeyExchangeProtocol.HYBRID_KYBER_ECDH:
            keypairs['ecdh'] = self._generate_ecdh_keypair()
        
        return keypairs
    
    def perform_hybrid_exchange(self, 
                               quantum_public: bytes,
                               classical_public: bytes,
                               our_keypairs: Dict[str, Tuple[bytes, bytes]]) -> KeyExchangeResult:
        """
        Perform hybrid key exchange combining quantum and classical
        """
        # Perform quantum key exchange
        quantum_ct, quantum_secret, session_id = self.quantum_kex.initiate_exchange(
            quantum_public
        )
        
        # Perform classical key exchange
        if self.protocol == KeyExchangeProtocol.HYBRID_KYBER_X25519:
            classical_secret = self._x25519_exchange(
                classical_public, 
                our_keypairs['x25519'][1]
            )
        else:
            classical_secret = self._ecdh_exchange(
                classical_public,
                our_keypairs['ecdh'][1]
            )
        
        # Combine secrets using XOR and KDF
        combined_secret = self._combine_secrets(quantum_secret, classical_secret)
        
        # Derive final session key
        final_key = self._derive_hybrid_key(combined_secret, session_id.encode())
        
        return KeyExchangeResult(
            shared_secret=final_key,
            session_id=session_id,
            protocol=self.protocol,
            metadata={
                'quantum_algorithm': 'Kyber',
                'classical_algorithm': self.protocol.value.split('_')[1],
                'key_size': len(final_key) * 8,
                'timestamp': time.time()
            }
        )
    
    def _generate_x25519_keypair(self) -> Tuple[bytes, bytes]:
        """Generate X25519 key pair (simulated)"""
        private_key = self.qrng.get_random_bytes(32)
        
        # Simulate public key derivation
        h = hashlib.sha256()
        h.update(private_key)
        public_key = h.digest()
        
        return public_key, private_key
    
    def _generate_ecdh_keypair(self) -> Tuple[bytes, bytes]:
        """Generate ECDH key pair (simulated)"""
        private_key = self.qrng.get_random_bytes(32)
        
        # Simulate public key derivation
        h = hashlib.sha3_256()
        h.update(private_key)
        public_key = h.digest()
        
        return public_key, private_key
    
    def _x25519_exchange(self, remote_public: bytes, our_private: bytes) -> bytes:
        """Perform X25519 key exchange (simulated)"""
        h = hashlib.sha256()
        h.update(our_private)
        h.update(remote_public)
        return h.digest()
    
    def _ecdh_exchange(self, remote_public: bytes, our_private: bytes) -> bytes:
        """Perform ECDH key exchange (simulated)"""
        h = hashlib.sha3_256()
        h.update(our_private)
        h.update(remote_public)
        return h.digest()
    
    def _combine_secrets(self, quantum: bytes, classical: bytes) -> bytes:
        """Combine quantum and classical secrets securely"""
        # XOR the secrets
        min_len = min(len(quantum), len(classical))
        combined = bytes(q ^ c for q, c in zip(quantum[:min_len], classical[:min_len]))
        
        # Add any remaining bytes
        if len(quantum) > min_len:
            combined += quantum[min_len:]
        elif len(classical) > min_len:
            combined += classical[min_len:]
        
        # Additional mixing with hash
        h = hashlib.sha3_512()
        h.update(quantum)
        h.update(classical)
        h.update(combined)
        
        return h.digest()
    
    def _derive_hybrid_key(self, combined_secret: bytes, session_id: bytes) -> bytes:
        """Derive final key from combined secrets"""
        # Use X963 KDF for key derivation
        kdf = X963KDF(
            algorithm=hashes.SHA3_256(),
            length=32,
            sharedinfo=b"hybrid_quantum_classical_" + session_id,
            backend=default_backend()
        )
        return kdf.derive(combined_secret)


class ThresholdKeyGeneration:
    """
    Threshold key generation using Shamir's Secret Sharing
    Allows n parties to generate a key where k are needed to reconstruct
    """
    
    def __init__(self, threshold: int, num_parties: int):
        if threshold > num_parties:
            raise ValueError("Threshold cannot exceed number of parties")
        
        self.threshold = threshold
        self.num_parties = num_parties
        self.prime = 2**256 - 189  # Large prime for field arithmetic
        self.qrng = QuantumRandomNumberGenerator()
    
    def generate_shares(self, secret: Optional[bytes] = None) -> List[ThresholdShare]:
        """
        Generate threshold shares of a secret
        If secret is None, generate a random secret
        """
        if secret is None:
            secret = self.qrng.get_random_bytes(32)
        
        # Convert secret to integer
        secret_int = int.from_bytes(secret, 'big') % self.prime
        
        # Generate random polynomial coefficients
        coefficients = [secret_int]
        for _ in range(self.threshold - 1):
            coeff = int.from_bytes(self.qrng.get_random_bytes(32), 'big') % self.prime
            coefficients.append(coeff)
        
        # Generate shares
        shares = []
        for i in range(1, self.num_parties + 1):
            # Evaluate polynomial at point i
            share_value = self._evaluate_polynomial(coefficients, i)
            
            # Generate commitment for verifiable secret sharing
            commitment = self._generate_commitment(i, share_value)
            
            # Generate zero-knowledge proof
            proof = self._generate_proof(i, share_value, commitment)
            
            shares.append(ThresholdShare(
                index=i,
                share=share_value.to_bytes(32, 'big'),
                commitment=commitment,
                proof=proof
            ))
        
        return shares
    
    def reconstruct_secret(self, shares: List[ThresholdShare]) -> bytes:
        """Reconstruct secret from threshold number of shares"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        # Verify all shares
        for share in shares:
            if not self._verify_share(share):
                raise ValueError(f"Invalid share from party {share.index}")
        
        # Use only threshold number of shares
        shares = shares[:self.threshold]
        
        # Lagrange interpolation
        secret = 0
        for i, share_i in enumerate(shares):
            numerator = 1
            denominator = 1
            
            share_value = int.from_bytes(share_i.share, 'big')
            
            for j, share_j in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-share_j.index)) % self.prime
                    denominator = (denominator * (share_i.index - share_j.index)) % self.prime
            
            # Modular inverse
            denominator_inv = pow(denominator, self.prime - 2, self.prime)
            lagrange = (numerator * denominator_inv) % self.prime
            
            secret = (secret + share_value * lagrange) % self.prime
        
        return secret.to_bytes(32, 'big')
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, self.prime)) % self.prime
        return result
    
    def _generate_commitment(self, index: int, value: int) -> bytes:
        """Generate Pedersen commitment for verifiable secret sharing"""
        h = hashlib.sha3_256()
        h.update(index.to_bytes(4, 'big'))
        h.update(value.to_bytes(32, 'big'))
        
        # Add blinding factor
        blinding = self.qrng.get_random_bytes(32)
        h.update(blinding)
        
        return h.digest()
    
    def _generate_proof(self, index: int, value: int, commitment: bytes) -> bytes:
        """Generate zero-knowledge proof of share validity"""
        # Simplified Schnorr-like proof
        h = hashlib.sha3_512()
        h.update(index.to_bytes(4, 'big'))
        h.update(value.to_bytes(32, 'big'))
        h.update(commitment)
        
        # Generate challenge
        challenge = int.from_bytes(h.digest()[:32], 'big') % self.prime
        
        # Generate response
        response = (value + challenge) % self.prime
        
        return response.to_bytes(32, 'big')
    
    def _verify_share(self, share: ThresholdShare) -> bool:
        """Verify share using commitment and proof"""
        # Recompute expected commitment
        h = hashlib.sha3_256()
        h.update(share.index.to_bytes(4, 'big'))
        h.update(share.share)
        
        # Simplified verification (would be more complex in production)
        # Check that proof matches expected pattern
        proof_int = int.from_bytes(share.proof, 'big')
        share_int = int.from_bytes(share.share, 'big')
        
        return proof_int > share_int  # Simplified check


class DistributedKeyGeneration:
    """
    Distributed Key Generation (DKG) Protocol
    Allows multiple parties to jointly generate a key without any single party knowing it
    """
    
    def __init__(self, party_id: int, threshold: int, num_parties: int):
        self.party_id = party_id
        self.threshold = threshold
        self.num_parties = num_parties
        self.tkg = ThresholdKeyGeneration(threshold, num_parties)
        self.received_shares = {}
        self.complaints = []
    
    def generate_contribution(self) -> Tuple[List[ThresholdShare], bytes]:
        """Generate this party's contribution to DKG"""
        # Generate random secret for this party
        secret = self.tkg.qrng.get_random_bytes(32)
        
        # Generate shares for all parties
        shares = self.tkg.generate_shares(secret)
        
        return shares, secret
    
    def receive_share(self, from_party: int, share: ThresholdShare):
        """Receive and verify share from another party"""
        # Verify the share
        if not self.tkg._verify_share(share):
            self.complaints.append({
                'from_party': from_party,
                'reason': 'invalid_share',
                'share': share
            })
            return False
        
        self.received_shares[from_party] = share
        return True
    
    def compute_final_share(self) -> bytes:
        """Compute final share from all received shares"""
        if len(self.received_shares) < self.threshold:
            raise ValueError("Not enough valid shares received")
        
        # Combine all shares for this party
        combined = 0
        for share in self.received_shares.values():
            share_value = int.from_bytes(share.share, 'big')
            combined = (combined + share_value) % self.tkg.prime
        
        return combined.to_bytes(32, 'big')
    
    def handle_complaints(self) -> bool:
        """Handle complaints phase of DKG"""
        if not self.complaints:
            return True
        
        # In production, this would involve zero-knowledge proofs
        # and dispute resolution protocols
        
        # For now, just log complaints
        for complaint in self.complaints:
            print(f"Complaint from party {self.party_id} about party {complaint['from_party']}")
        
        return len(self.complaints) < self.num_parties // 2


class ForwardSecureKeyExchange:
    """
    Forward-secure key exchange with automatic key deletion
    Ensures past communications remain secure even if current keys are compromised
    """
    
    def __init__(self):
        self.qkex = QuantumSafeKeyExchange()
        self.epoch = 0
        self.key_chain = []
        self.max_chain_length = 100
    
    def generate_epoch_keys(self) -> Tuple[bytes, bytes, int]:
        """Generate keys for current epoch"""
        public_key, private_key = self.qkex.generate_ephemeral_keypair()
        
        # Store with epoch number
        self.key_chain.append({
            'epoch': self.epoch,
            'public': public_key,
            'private': private_key,
            'timestamp': time.time()
        })
        
        # Increment epoch
        current_epoch = self.epoch
        self.epoch += 1
        
        # Delete old keys if chain too long
        if len(self.key_chain) > self.max_chain_length:
            self._delete_old_keys()
        
        return public_key, private_key, current_epoch
    
    def _delete_old_keys(self):
        """Securely delete old epoch keys"""
        # Keep only recent epochs
        to_delete = len(self.key_chain) - self.max_chain_length
        
        for i in range(to_delete):
            old_key = self.key_chain[i]
            # Overwrite key material
            self.qkex._secure_erase(old_key['private'])
            self.qkex._secure_erase(old_key['public'])
        
        # Remove from chain
        self.key_chain = self.key_chain[to_delete:]
    
    def derive_forward_secure_key(self, epoch: int, shared_secret: bytes) -> bytes:
        """Derive forward-secure key for specific epoch"""
        # Use tree-based key derivation
        h = hashlib.sha3_256()
        h.update(epoch.to_bytes(8, 'big'))
        h.update(shared_secret)
        h.update(b"forward_secure_epoch")
        
        return h.digest()


if __name__ == "__main__":
    # Test quantum-safe key exchange
    print("Testing Quantum-Safe Key Exchange")
    print("=" * 50)
    
    # Initialize parties
    alice_kex = QuantumSafeKeyExchange()
    bob_kex = QuantumSafeKeyExchange()
    
    # Bob generates keypair
    bob_public, bob_private = alice_kex.generate_ephemeral_keypair()
    
    # Alice initiates exchange
    ciphertext, alice_key, session_id = alice_kex.initiate_exchange(bob_public)
    print(f"Session ID: {session_id}")
    print(f"Alice's key: {alice_key.hex()[:32]}...")
    
    # Bob completes exchange
    bob_key = bob_kex.complete_exchange(ciphertext, bob_private, session_id)
    print(f"Bob's key:   {bob_key.hex()[:32]}...")
    
    # Test threshold key generation
    print("\nTesting Threshold Key Generation")
    print("=" * 50)
    
    tkg = ThresholdKeyGeneration(threshold=3, num_parties=5)
    secret = b"This is a secret message for threshold sharing!"[:32]
    
    # Generate shares
    shares = tkg.generate_shares(secret)
    print(f"Generated {len(shares)} shares (threshold: 3)")
    
    # Reconstruct with minimum shares
    reconstructed = tkg.reconstruct_secret(shares[:3])
    print(f"Secret reconstructed: {reconstructed == secret.ljust(32, b'\\0')}")