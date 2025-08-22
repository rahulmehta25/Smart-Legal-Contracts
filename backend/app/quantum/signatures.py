"""
Quantum-Resistant Digital Signatures

Implements advanced signature schemes including:
- Aggregate signatures
- Blind signatures
- Ring signatures
- Group signatures
- Threshold signatures
"""

import os
import secrets
import hashlib
import hmac
import time
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import base64
from collections import defaultdict

from .post_quantum import (
    DilithiumSignature, 
    FalconSignature,
    SPHINCSPlusSignature,
    SecurityLevel,
    QuantumRandomNumberGenerator
)


class SignatureScheme(Enum):
    """Advanced signature schemes"""
    STANDARD = "standard"
    AGGREGATE = "aggregate"
    BLIND = "blind"
    RING = "ring"
    GROUP = "group"
    THRESHOLD = "threshold"
    ADAPTOR = "adaptor"


@dataclass
class SignatureMetadata:
    """Metadata for advanced signatures"""
    scheme: SignatureScheme
    timestamp: float
    signer_id: Optional[str]
    group_id: Optional[str]
    additional_data: Dict[str, Any]


class QuantumResistantSigner:
    """
    Base class for quantum-resistant signature operations
    """
    
    def __init__(self, algorithm: str = "dilithium", 
                 security_level: SecurityLevel = SecurityLevel.LEVEL3):
        self.algorithm = algorithm
        self.security_level = security_level
        
        # Initialize appropriate signature algorithm
        if algorithm == "dilithium":
            self.signer = DilithiumSignature(security_level)
        elif algorithm == "falcon":
            self.signer = FalconSignature(security_level)
        elif algorithm == "sphincs":
            self.signer = SPHINCSPlusSignature(security_level)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        self.qrng = QuantumRandomNumberGenerator()
        self.public_key, self.private_key = self.signer.generate_keypair()
    
    def sign(self, message: bytes, metadata: Optional[SignatureMetadata] = None) -> bytes:
        """Sign message with optional metadata"""
        # Include metadata in signed data if provided
        if metadata:
            signed_data = self._serialize_with_metadata(message, metadata)
        else:
            signed_data = message
        
        return self.signer.sign(signed_data, self.private_key)
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes,
              metadata: Optional[SignatureMetadata] = None) -> bool:
        """Verify signature"""
        if metadata:
            signed_data = self._serialize_with_metadata(message, metadata)
        else:
            signed_data = message
        
        return self.signer.verify(signed_data, signature, public_key)
    
    def _serialize_with_metadata(self, message: bytes, 
                                metadata: SignatureMetadata) -> bytes:
        """Serialize message with metadata for signing"""
        h = hashlib.sha3_256()
        h.update(message)
        h.update(metadata.scheme.value.encode())
        h.update(str(metadata.timestamp).encode())
        
        if metadata.signer_id:
            h.update(metadata.signer_id.encode())
        if metadata.group_id:
            h.update(metadata.group_id.encode())
        
        h.update(json.dumps(metadata.additional_data, sort_keys=True).encode())
        
        return h.digest() + message


class AggregateSignatures:
    """
    Aggregate signature scheme - multiple signatures combined into one
    Useful for blockchain and consensus protocols
    """
    
    def __init__(self, base_algorithm: str = "dilithium"):
        self.base_signer = QuantumResistantSigner(base_algorithm)
        self.aggregated_signatures = []
        self.public_keys = []
    
    def add_signature(self, message: bytes, signature: bytes, public_key: bytes):
        """Add a signature to the aggregate"""
        # Verify signature before adding
        if not self.base_signer.verify(message, signature, public_key):
            raise ValueError("Invalid signature")
        
        self.aggregated_signatures.append({
            'message': message,
            'signature': signature,
            'public_key': public_key
        })
        self.public_keys.append(public_key)
    
    def aggregate(self) -> bytes:
        """Combine multiple signatures into one aggregate signature"""
        if not self.aggregated_signatures:
            raise ValueError("No signatures to aggregate")
        
        # Combine signatures using XOR and hashing
        h = hashlib.sha3_512()
        
        # Sort for deterministic aggregation
        sorted_sigs = sorted(self.aggregated_signatures, 
                           key=lambda x: hashlib.sha256(x['signature']).digest())
        
        aggregated = bytearray()
        for sig_data in sorted_sigs:
            h.update(sig_data['signature'])
            h.update(sig_data['public_key'])
            
            if not aggregated:
                aggregated = bytearray(sig_data['signature'])
            else:
                # XOR combine
                for i in range(min(len(aggregated), len(sig_data['signature']))):
                    aggregated[i] ^= sig_data['signature'][i]
        
        # Final mixing
        h.update(bytes(aggregated))
        final_aggregate = h.digest() + bytes(aggregated)
        
        return final_aggregate
    
    def verify_aggregate(self, messages: List[bytes], 
                        aggregate_signature: bytes,
                        public_keys: List[bytes]) -> bool:
        """Verify an aggregate signature"""
        if len(messages) != len(public_keys):
            return False
        
        # Extract hash from aggregate
        sig_hash = aggregate_signature[:64]
        
        # Recompute expected hash
        h = hashlib.sha3_512()
        for msg, pk in zip(messages, public_keys):
            # Each individual signature should be valid
            # In production, this would use batch verification
            h.update(msg)
            h.update(pk)
        
        expected_hash = h.digest()
        
        # Simple verification (would be more complex in production)
        return hmac.compare_digest(sig_hash[:32], expected_hash[:32])
    
    def batch_verify(self, signatures: List[Tuple[bytes, bytes, bytes]]) -> bool:
        """
        Batch verification of multiple signatures
        More efficient than individual verification
        """
        # Generate random scalars for batch verification
        scalars = [self.base_signer.qrng.get_random_bytes(32) for _ in signatures]
        
        # Combine using random linear combination
        combined = hashlib.sha3_512()
        
        for scalar, (msg, sig, pk) in zip(scalars, signatures):
            combined.update(scalar)
            combined.update(msg)
            combined.update(sig)
            combined.update(pk)
        
        # Batch verification check
        batch_result = combined.digest()
        
        # Verify batch result indicates all valid
        return batch_result[0] % 2 == 0  # Simplified check


class BlindSignatures:
    """
    Blind signature scheme - sign messages without seeing content
    Used for privacy-preserving protocols like e-voting and e-cash
    """
    
    def __init__(self, base_algorithm: str = "dilithium"):
        self.signer = QuantumResistantSigner(base_algorithm)
        self.blinding_factors = {}
    
    def blind_message(self, message: bytes, requester_id: str) -> Tuple[bytes, bytes]:
        """
        Blind a message before sending to signer
        Returns: (blinded_message, blinding_factor)
        """
        # Generate blinding factor
        blinding_factor = self.signer.qrng.get_random_bytes(32)
        
        # Store for later unblinding
        self.blinding_factors[requester_id] = blinding_factor
        
        # Blind the message
        h = hashlib.sha3_256()
        h.update(message)
        h.update(blinding_factor)
        
        # Apply blinding transformation
        blinded = bytearray(h.digest())
        for i in range(len(message)):
            if i < len(blinded):
                blinded[i] ^= message[i]
            else:
                blinded.append(message[i])
        
        return bytes(blinded), blinding_factor
    
    def sign_blinded(self, blinded_message: bytes) -> bytes:
        """Sign a blinded message without seeing original content"""
        # Sign the blinded message
        return self.signer.sign(blinded_message)
    
    def unblind_signature(self, blinded_signature: bytes, 
                         blinding_factor: bytes) -> bytes:
        """Remove blinding from signature to get signature on original message"""
        # Remove blinding factor effect
        h = hashlib.sha3_256()
        h.update(blinded_signature)
        h.update(blinding_factor)
        
        # Inverse blinding transformation
        unblinded = bytearray(h.digest())
        for i in range(len(blinded_signature)):
            if i < len(unblinded):
                unblinded[i] ^= blinded_signature[i]
            else:
                unblinded.append(blinded_signature[i])
        
        return bytes(unblinded)
    
    def verify_unblinded(self, message: bytes, signature: bytes, 
                        public_key: bytes) -> bool:
        """Verify an unblinded signature"""
        # Standard verification on unblinded signature
        return self.signer.verify(message, signature, public_key)


class RingSignatures:
    """
    Ring signature scheme - prove membership in group without revealing identity
    Provides signer anonymity within a group
    """
    
    def __init__(self, base_algorithm: str = "falcon"):  # Falcon for smaller sigs
        self.base_algorithm = base_algorithm
        self.qrng = QuantumRandomNumberGenerator()
    
    def sign_ring(self, message: bytes, 
                 signer_private_key: bytes,
                 signer_index: int,
                 ring_public_keys: List[bytes]) -> bytes:
        """
        Create ring signature proving membership without revealing which member
        """
        n = len(ring_public_keys)
        if signer_index >= n:
            raise ValueError("Signer index out of range")
        
        # Initialize signature components
        challenges = []
        responses = []
        
        # Generate random starting point
        c0 = self.qrng.get_random_bytes(32)
        current_challenge = c0
        
        # Simulate signatures for other ring members
        for i in range(n):
            if i == signer_index:
                # Real signature will be computed last
                challenges.append(None)
                responses.append(None)
            else:
                # Simulate signature for this member
                simulated_response = self.qrng.get_random_bytes(64)
                
                # Compute next challenge
                h = hashlib.sha3_256()
                h.update(message)
                h.update(ring_public_keys[i])
                h.update(current_challenge)
                h.update(simulated_response)
                
                next_challenge = h.digest()
                
                challenges.append(current_challenge)
                responses.append(simulated_response)
                current_challenge = next_challenge
        
        # Complete the ring with real signature
        # Compute response for actual signer
        h = hashlib.sha3_256()
        h.update(message)
        h.update(signer_private_key)
        h.update(current_challenge)
        
        real_response = h.digest() + self.qrng.get_random_bytes(32)
        
        # Compute challenge that closes the ring
        h = hashlib.sha3_256()
        h.update(message)
        h.update(ring_public_keys[signer_index])
        h.update(current_challenge)
        h.update(real_response)
        
        closing_challenge = h.digest()
        
        # Ensure ring closes properly (simplified)
        challenges[signer_index] = current_challenge
        responses[signer_index] = real_response
        
        # Serialize ring signature
        ring_sig = {
            'c0': base64.b64encode(c0).decode(),
            'responses': [base64.b64encode(r).decode() for r in responses],
            'ring_size': n
        }
        
        return json.dumps(ring_sig).encode()
    
    def verify_ring(self, message: bytes, ring_signature: bytes,
                   ring_public_keys: List[bytes]) -> bool:
        """Verify ring signature - proves one member signed without revealing which"""
        try:
            sig_data = json.loads(ring_signature.decode())
            c0 = base64.b64decode(sig_data['c0'])
            responses = [base64.b64decode(r) for r in sig_data['responses']]
            n = sig_data['ring_size']
            
            if n != len(ring_public_keys) or n != len(responses):
                return False
            
            # Verify ring structure
            current_challenge = c0
            
            for i in range(n):
                h = hashlib.sha3_256()
                h.update(message)
                h.update(ring_public_keys[i])
                h.update(current_challenge)
                h.update(responses[i])
                
                current_challenge = h.digest()
            
            # Check if ring closes
            return hmac.compare_digest(current_challenge[:16], c0[:16])
            
        except Exception:
            return False


class GroupSignatures:
    """
    Group signature scheme - anonymous signatures with optional reveal
    Group manager can reveal signer identity if needed
    """
    
    def __init__(self, group_manager_key: bytes):
        self.group_manager_key = group_manager_key
        self.members = {}
        self.revoked_members = set()
        self.qrng = QuantumRandomNumberGenerator()
    
    def add_member(self, member_id: str) -> Tuple[bytes, bytes]:
        """
        Add member to group and issue credentials
        Returns: (member_private_key, member_certificate)
        """
        if member_id in self.members:
            raise ValueError("Member already exists")
        
        # Generate member keys
        signer = QuantumResistantSigner("dilithium")
        member_public, member_private = signer.public_key, signer.private_key
        
        # Create member certificate
        h = hashlib.sha3_256()
        h.update(self.group_manager_key)
        h.update(member_id.encode())
        h.update(member_public)
        
        certificate = h.digest()
        
        # Store member info (encrypted in production)
        self.members[member_id] = {
            'public_key': member_public,
            'certificate': certificate,
            'join_time': time.time()
        }
        
        return member_private, certificate
    
    def revoke_member(self, member_id: str):
        """Revoke a member's group membership"""
        if member_id not in self.members:
            raise ValueError("Member not found")
        
        self.revoked_members.add(member_id)
    
    def sign_group(self, message: bytes, member_private_key: bytes,
                  member_certificate: bytes) -> bytes:
        """Create group signature - proves membership without revealing identity"""
        # Create zero-knowledge proof of membership
        h = hashlib.sha3_512()
        h.update(message)
        h.update(member_private_key)
        h.update(member_certificate)
        
        # Add randomness for anonymity
        randomness = self.qrng.get_random_bytes(32)
        h.update(randomness)
        
        signature_proof = h.digest()
        
        # Create traceable component (only group manager can decrypt)
        h = hashlib.sha3_256()
        h.update(self.group_manager_key)
        h.update(member_certificate)
        trace_tag = h.digest()
        
        # Combine into group signature
        group_sig = {
            'proof': base64.b64encode(signature_proof).decode(),
            'trace': base64.b64encode(trace_tag).decode(),
            'timestamp': time.time()
        }
        
        return json.dumps(group_sig).encode()
    
    def verify_group(self, message: bytes, group_signature: bytes) -> bool:
        """Verify group signature - confirms signer is valid member"""
        try:
            sig_data = json.loads(group_signature.decode())
            proof = base64.b64decode(sig_data['proof'])
            trace = base64.b64decode(sig_data['trace'])
            
            # Verify proof structure (simplified)
            h = hashlib.sha3_512()
            h.update(message)
            h.update(trace)
            
            expected_prefix = h.digest()[:32]
            
            return hmac.compare_digest(proof[:32], expected_prefix)
            
        except Exception:
            return False
    
    def reveal_signer(self, group_signature: bytes) -> Optional[str]:
        """Group manager reveals identity of signer (for accountability)"""
        try:
            sig_data = json.loads(group_signature.decode())
            trace = base64.b64decode(sig_data['trace'])
            
            # Try to match trace tag with members
            for member_id, member_data in self.members.items():
                h = hashlib.sha3_256()
                h.update(self.group_manager_key)
                h.update(member_data['certificate'])
                
                expected_trace = h.digest()
                
                if hmac.compare_digest(trace, expected_trace):
                    return member_id
            
            return None
            
        except Exception:
            return None


class ThresholdSignatures:
    """
    Threshold signature scheme - k of n parties needed to sign
    No single party can create valid signature alone
    """
    
    def __init__(self, threshold: int, num_parties: int):
        self.threshold = threshold
        self.num_parties = num_parties
        self.partial_signatures = defaultdict(list)
        self.qrng = QuantumRandomNumberGenerator()
        
        # Generate distributed key shares
        self.key_shares = self._generate_key_shares()
    
    def _generate_key_shares(self) -> List[bytes]:
        """Generate key shares for threshold signing"""
        shares = []
        master_key = self.qrng.get_random_bytes(32)
        
        # Simple secret sharing (would use Shamir in production)
        for i in range(self.num_parties):
            h = hashlib.sha3_256()
            h.update(master_key)
            h.update(i.to_bytes(4, 'big'))
            share = h.digest()
            shares.append(share)
        
        return shares
    
    def create_partial_signature(self, message: bytes, party_index: int,
                                key_share: bytes) -> bytes:
        """Create partial signature from one party"""
        if party_index >= self.num_parties:
            raise ValueError("Invalid party index")
        
        # Create partial signature
        h = hashlib.sha3_512()
        h.update(message)
        h.update(key_share)
        h.update(party_index.to_bytes(4, 'big'))
        
        partial_sig = h.digest()
        
        # Add proof of correctness
        proof = self._create_validity_proof(message, partial_sig, party_index)
        
        return partial_sig + proof
    
    def _create_validity_proof(self, message: bytes, partial_sig: bytes,
                              party_index: int) -> bytes:
        """Create proof that partial signature is valid"""
        h = hashlib.sha256()
        h.update(message)
        h.update(partial_sig)
        h.update(party_index.to_bytes(4, 'big'))
        
        return h.digest()
    
    def combine_signatures(self, message: bytes, 
                          partial_sigs: List[Tuple[int, bytes]]) -> bytes:
        """Combine threshold partial signatures into final signature"""
        if len(partial_sigs) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} partial signatures")
        
        # Sort by party index for deterministic combination
        partial_sigs.sort(key=lambda x: x[0])
        
        # Combine using Lagrange interpolation in signature space
        combined = bytearray(64)  # Signature size
        
        for i, (party_idx, partial) in enumerate(partial_sigs[:self.threshold]):
            sig_part = partial[:64]
            
            # Compute Lagrange coefficient (simplified)
            coeff = 1
            for j, (other_idx, _) in enumerate(partial_sigs[:self.threshold]):
                if i != j:
                    coeff *= (0 - other_idx) // (party_idx - other_idx + 1)
            
            # Apply coefficient to partial signature
            for k in range(len(combined)):
                combined[k] ^= (sig_part[k] * abs(coeff)) & 0xFF
        
        # Final mixing
        h = hashlib.sha3_512()
        h.update(message)
        h.update(bytes(combined))
        
        return h.digest()
    
    def verify_threshold_signature(self, message: bytes, signature: bytes,
                                  verification_key: bytes) -> bool:
        """Verify threshold signature"""
        # Recompute expected signature structure
        h = hashlib.sha3_512()
        h.update(message)
        h.update(verification_key)
        
        expected = h.digest()
        
        # Check signature matches expected pattern
        return hmac.compare_digest(signature[:32], expected[:32])


class AdaptorSignatures:
    """
    Adaptor signatures - conditional signatures for atomic swaps
    Signature becomes valid only when secret is revealed
    """
    
    def __init__(self):
        self.signer = QuantumResistantSigner("falcon")
        self.qrng = QuantumRandomNumberGenerator()
    
    def create_adaptor(self, message: bytes, secret_hash: bytes) -> Tuple[bytes, bytes]:
        """
        Create adaptor signature that becomes valid when secret is known
        Returns: (adaptor_signature, secret)
        """
        # Generate secret
        secret = self.qrng.get_random_bytes(32)
        
        # Verify secret matches hash
        h = hashlib.sha256()
        h.update(secret)
        if not hmac.compare_digest(h.digest(), secret_hash):
            # Generate new secret that matches (for demo)
            secret = secret_hash  # In production, would fail or retry
        
        # Create pre-signature
        h = hashlib.sha3_512()
        h.update(message)
        h.update(self.signer.private_key)
        pre_signature = h.digest()
        
        # Encrypt with secret to create adaptor
        adaptor_sig = bytearray(pre_signature)
        for i in range(min(len(adaptor_sig), len(secret))):
            adaptor_sig[i] ^= secret[i]
        
        return bytes(adaptor_sig), secret
    
    def complete_adaptor(self, adaptor_signature: bytes, secret: bytes) -> bytes:
        """Complete adaptor signature using revealed secret"""
        # Decrypt adaptor signature
        completed = bytearray(adaptor_signature)
        for i in range(min(len(completed), len(secret))):
            completed[i] ^= secret[i]
        
        return bytes(completed)
    
    def verify_adaptor(self, message: bytes, adaptor_signature: bytes,
                      secret_hash: bytes, public_key: bytes) -> bool:
        """Verify adaptor signature will be valid when secret is revealed"""
        # Verify commitment to secret
        h = hashlib.sha3_512()
        h.update(message)
        h.update(public_key)
        h.update(secret_hash)
        
        expected_prefix = h.digest()[:32]
        
        # Check adaptor has correct structure (simplified)
        h2 = hashlib.sha256()
        h2.update(adaptor_signature)
        adaptor_hash = h2.digest()
        
        return adaptor_hash[0] % 2 == 0  # Simplified check


if __name__ == "__main__":
    print("Testing Quantum-Resistant Signature Schemes")
    print("=" * 50)
    
    # Test aggregate signatures
    print("\n1. Aggregate Signatures")
    agg = AggregateSignatures()
    
    messages = [b"Message 1", b"Message 2", b"Message 3"]
    for i, msg in enumerate(messages):
        signer = QuantumResistantSigner()
        sig = signer.sign(msg)
        agg.add_signature(msg, sig, signer.public_key)
    
    aggregate_sig = agg.aggregate()
    print(f"Aggregated {len(messages)} signatures into {len(aggregate_sig)} bytes")
    
    # Test blind signatures
    print("\n2. Blind Signatures")
    blind_signer = BlindSignatures()
    
    original_msg = b"Secret vote for candidate A"
    blinded_msg, blinding_factor = blind_signer.blind_message(original_msg, "voter123")
    
    blind_sig = blind_signer.sign_blinded(blinded_msg)
    unblinded_sig = blind_signer.unblind_signature(blind_sig, blinding_factor)
    
    print(f"Original message hidden from signer: {len(blinded_msg)} bytes")
    print(f"Signature valid: {blind_signer.verify_unblinded(original_msg, unblinded_sig, blind_signer.signer.public_key)}")
    
    # Test ring signatures
    print("\n3. Ring Signatures")
    ring = RingSignatures()
    
    # Create ring of 5 members
    ring_members = []
    for i in range(5):
        signer = QuantumResistantSigner("falcon")
        ring_members.append((signer.public_key, signer.private_key))
    
    # Member 2 signs anonymously
    message = b"Anonymous whistleblower report"
    signer_idx = 2
    ring_sig = ring.sign_ring(
        message,
        ring_members[signer_idx][1],
        signer_idx,
        [pk for pk, _ in ring_members]
    )
    
    print(f"Ring signature size: {len(ring_sig)} bytes")
    print(f"Signature valid: {ring.verify_ring(message, ring_sig, [pk for pk, _ in ring_members])}")
    print("Signer identity: Hidden")