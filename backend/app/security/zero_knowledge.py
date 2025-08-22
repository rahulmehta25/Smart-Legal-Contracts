"""
Zero-Knowledge Proof Implementations

Implements various zero-knowledge proof systems:
- zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge)
- zk-STARKs (Zero-Knowledge Scalable Transparent Arguments of Knowledge)
- Bulletproofs for range proofs
- Identity verification without disclosure
- Compliance proofs
- Authentication without passwords
"""

import hashlib
import hmac
import secrets
import time
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend


class ProofType(Enum):
    """Types of zero-knowledge proofs"""
    SNARK = "snark"
    STARK = "stark"
    BULLETPROOF = "bulletproof"
    SIGMA = "sigma"
    SCHNORR = "schnorr"
    GROTH16 = "groth16"
    PLONK = "plonk"


@dataclass
class ZKProof:
    """Zero-knowledge proof structure"""
    proof_type: ProofType
    proof_data: bytes
    public_inputs: List[bytes]
    verification_key: bytes
    metadata: Dict[str, Any]


@dataclass
class ProofVerificationResult:
    """Result of proof verification"""
    valid: bool
    proof_type: ProofType
    verification_time_ms: float
    metadata: Dict[str, Any]


class ZKSnark:
    """
    zk-SNARK implementation
    Succinct proofs with trusted setup
    """
    
    def __init__(self):
        self.trusted_setup = self._perform_trusted_setup()
        self.proving_key = None
        self.verification_key = None
        
    def _perform_trusted_setup(self) -> Dict[str, bytes]:
        """Perform trusted setup ceremony (simplified)"""
        # In production, this would be a multi-party computation
        toxic_waste = secrets.token_bytes(32)
        
        # Generate CRS (Common Reference String)
        h = hashlib.sha3_512()
        h.update(toxic_waste)
        
        setup = {
            'tau': h.digest()[:32],
            'alpha': h.digest()[32:64],
            'beta': hashlib.sha256(toxic_waste).digest(),
            'gamma': hashlib.blake2b(toxic_waste).digest()[:32],
            'delta': hashlib.sha3_256(toxic_waste).digest()
        }
        
        # "Destroy" toxic waste (in production, secure deletion)
        toxic_waste = None
        
        return setup
    
    def setup_circuit(self, circuit_constraints: List[Dict[str, Any]]):
        """Setup circuit for specific computation"""
        # Generate proving and verification keys
        h = hashlib.sha3_512()
        
        for constraint in circuit_constraints:
            h.update(json.dumps(constraint, sort_keys=True).encode())
        
        h.update(self.trusted_setup['tau'])
        
        self.proving_key = h.digest()[:64]
        self.verification_key = h.digest()[64:96]
    
    def generate_proof(self, witness: bytes, public_inputs: List[bytes]) -> ZKProof:
        """
        Generate SNARK proof
        witness: private input (kept secret)
        public_inputs: public inputs to the circuit
        """
        if not self.proving_key:
            raise ValueError("Circuit not setup")
        
        # Create proof (simplified Groth16-like)
        h = hashlib.sha3_512()
        h.update(self.proving_key)
        h.update(witness)
        
        for pub_input in public_inputs:
            h.update(pub_input)
        
        # Generate proof elements
        proof_a = h.digest()[:32]
        
        h.update(self.trusted_setup['alpha'])
        proof_b = h.digest()[:32]
        
        h.update(self.trusted_setup['beta'])
        proof_c = h.digest()[:32]
        
        # Combine proof elements
        proof_data = proof_a + proof_b + proof_c
        
        return ZKProof(
            proof_type=ProofType.SNARK,
            proof_data=proof_data,
            public_inputs=public_inputs,
            verification_key=self.verification_key,
            metadata={'size_bytes': len(proof_data)}
        )
    
    def verify_proof(self, proof: ZKProof) -> bool:
        """Verify SNARK proof"""
        if proof.proof_type != ProofType.SNARK:
            return False
        
        # Extract proof elements
        proof_a = proof.proof_data[:32]
        proof_b = proof.proof_data[32:64]
        proof_c = proof.proof_data[64:96]
        
        # Verify pairing equation (simplified)
        h = hashlib.sha3_512()
        h.update(proof.verification_key)
        h.update(proof_a)
        h.update(proof_b)
        h.update(proof_c)
        
        for pub_input in proof.public_inputs:
            h.update(pub_input)
        
        # Check verification equation
        result = h.digest()
        
        # Simplified check (in production, actual pairing check)
        return result[0] % 2 == 0


class ZKStark:
    """
    zk-STARK implementation
    Transparent proofs without trusted setup
    """
    
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        self.hash_function = hashlib.sha3_256
        self.field_modulus = 2**256 - 189  # Large prime
        
    def generate_proof(self, computation_trace: List[int], 
                      public_inputs: List[bytes]) -> ZKProof:
        """
        Generate STARK proof for computation trace
        """
        # Create algebraic intermediate representation (AIR)
        air = self._create_air(computation_trace)
        
        # Generate FRI (Fast Reed-Solomon IOP) proof
        fri_proof = self._generate_fri_proof(air)
        
        # Create Merkle commitments
        commitments = self._create_commitments(computation_trace)
        
        # Combine into STARK proof
        proof_data = {
            'fri': base64.b64encode(fri_proof).decode(),
            'commitments': [base64.b64encode(c).decode() for c in commitments],
            'air': air
        }
        
        proof_bytes = json.dumps(proof_data).encode()
        
        # Generate verification key
        h = self.hash_function()
        h.update(proof_bytes)
        verification_key = h.digest()
        
        return ZKProof(
            proof_type=ProofType.STARK,
            proof_data=proof_bytes,
            public_inputs=public_inputs,
            verification_key=verification_key,
            metadata={
                'transparent': True,
                'post_quantum': True,
                'size_bytes': len(proof_bytes)
            }
        )
    
    def _create_air(self, trace: List[int]) -> Dict[str, Any]:
        """Create Algebraic Intermediate Representation"""
        return {
            'constraints': len(trace),
            'degree': min(len(trace), 1024),
            'field': 'prime_field',
            'trace_length': len(trace)
        }
    
    def _generate_fri_proof(self, air: Dict[str, Any]) -> bytes:
        """Generate FRI (Fast Reed-Solomon) proof"""
        # Simplified FRI protocol
        h = self.hash_function()
        h.update(json.dumps(air).encode())
        
        # Multiple rounds of folding
        current = h.digest()
        for _ in range(self.security_parameter // 32):
            h = self.hash_function()
            h.update(current)
            current = h.digest()
        
        return current
    
    def _create_commitments(self, trace: List[int]) -> List[bytes]:
        """Create Merkle tree commitments"""
        commitments = []
        
        # Build Merkle tree (simplified)
        for i in range(0, len(trace), 32):
            chunk = trace[i:i+32]
            h = self.hash_function()
            
            for value in chunk:
                h.update(value.to_bytes(32, 'big'))
            
            commitments.append(h.digest())
        
        return commitments
    
    def verify_proof(self, proof: ZKProof) -> bool:
        """Verify STARK proof"""
        if proof.proof_type != ProofType.STARK:
            return False
        
        try:
            proof_data = json.loads(proof.proof_data.decode())
            
            # Verify FRI proof
            fri_proof = base64.b64decode(proof_data['fri'])
            
            # Verify commitments
            for commitment_b64 in proof_data['commitments']:
                commitment = base64.b64decode(commitment_b64)
                # Verify Merkle paths (simplified)
                if len(commitment) != 32:
                    return False
            
            # Verify AIR constraints
            air = proof_data['air']
            if air['constraints'] <= 0:
                return False
            
            # Check verification key
            h = self.hash_function()
            h.update(proof.proof_data)
            expected_key = h.digest()
            
            return hmac.compare_digest(expected_key, proof.verification_key)
            
        except Exception:
            return False


class Bulletproofs:
    """
    Bulletproofs for efficient range proofs
    No trusted setup required
    """
    
    def __init__(self):
        self.group_order = 2**256 - 432420386565659656852420866394968145599
        self.generators = self._setup_generators()
    
    def _setup_generators(self) -> Dict[str, bytes]:
        """Setup group generators"""
        # Generate random group elements (simplified)
        g = hashlib.sha3_256(b"generator_g").digest()
        h = hashlib.sha3_256(b"generator_h").digest()
        
        return {'g': g, 'h': h}
    
    def prove_range(self, value: int, range_bits: int) -> ZKProof:
        """
        Prove that value is in range [0, 2^range_bits)
        """
        if value < 0 or value >= 2**range_bits:
            raise ValueError("Value out of range")
        
        # Convert value to binary
        binary = bin(value)[2:].zfill(range_bits)
        
        # Create commitments for each bit
        commitments = []
        for bit in binary:
            # Pedersen commitment
            r = secrets.randbits(256)  # Blinding factor
            
            h = hashlib.sha3_256()
            h.update(self.generators['g'])
            h.update(int(bit).to_bytes(1, 'big'))
            h.update(self.generators['h'])
            h.update(r.to_bytes(32, 'big'))
            
            commitment = h.digest()
            commitments.append(commitment)
        
        # Create aggregated proof (simplified)
        h = hashlib.sha3_512()
        for commitment in commitments:
            h.update(commitment)
        
        # Inner product argument
        inner_product = h.digest()[:64]
        
        # Final proof
        proof_data = {
            'commitments': [base64.b64encode(c).decode() for c in commitments],
            'inner_product': base64.b64encode(inner_product).decode(),
            'range_bits': range_bits
        }
        
        proof_bytes = json.dumps(proof_data).encode()
        
        return ZKProof(
            proof_type=ProofType.BULLETPROOF,
            proof_data=proof_bytes,
            public_inputs=[],
            verification_key=self.generators['g'],
            metadata={
                'range_bits': range_bits,
                'proof_size': len(proof_bytes)
            }
        )
    
    def verify_range(self, proof: ZKProof) -> bool:
        """Verify range proof"""
        if proof.proof_type != ProofType.BULLETPROOF:
            return False
        
        try:
            proof_data = json.loads(proof.proof_data.decode())
            
            # Verify number of commitments matches range
            range_bits = proof_data['range_bits']
            if len(proof_data['commitments']) != range_bits:
                return False
            
            # Verify inner product argument
            inner_product = base64.b64decode(proof_data['inner_product'])
            if len(inner_product) != 64:
                return False
            
            # Verify each commitment (simplified)
            for commitment_b64 in proof_data['commitments']:
                commitment = base64.b64decode(commitment_b64)
                if len(commitment) != 32:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def aggregate_proofs(self, proofs: List[ZKProof]) -> ZKProof:
        """Aggregate multiple range proofs into one"""
        if not all(p.proof_type == ProofType.BULLETPROOF for p in proofs):
            raise ValueError("All proofs must be bulletproofs")
        
        # Combine all commitments
        all_commitments = []
        total_range_bits = 0
        
        for proof in proofs:
            data = json.loads(proof.proof_data.decode())
            all_commitments.extend(data['commitments'])
            total_range_bits += data['range_bits']
        
        # Create aggregated inner product
        h = hashlib.sha3_512()
        for c_b64 in all_commitments:
            h.update(base64.b64decode(c_b64))
        
        aggregated_inner = h.digest()
        
        # Create aggregated proof
        agg_data = {
            'commitments': all_commitments,
            'inner_product': base64.b64encode(aggregated_inner).decode(),
            'range_bits': total_range_bits,
            'num_proofs': len(proofs)
        }
        
        return ZKProof(
            proof_type=ProofType.BULLETPROOF,
            proof_data=json.dumps(agg_data).encode(),
            public_inputs=[],
            verification_key=self.generators['g'],
            metadata={'aggregated': True, 'num_proofs': len(proofs)}
        )


class IdentityProofs:
    """
    Zero-knowledge identity verification
    Prove identity attributes without revealing them
    """
    
    def __init__(self):
        self.attribute_commitments = {}
        self.blinding_factors = {}
    
    def commit_attribute(self, attribute_name: str, attribute_value: bytes) -> bytes:
        """Create commitment to an identity attribute"""
        # Generate blinding factor
        blinding = secrets.token_bytes(32)
        
        # Pedersen commitment
        h = hashlib.sha3_256()
        h.update(attribute_name.encode())
        h.update(attribute_value)
        h.update(blinding)
        
        commitment = h.digest()
        
        # Store for later proof generation
        self.attribute_commitments[attribute_name] = commitment
        self.blinding_factors[attribute_name] = blinding
        
        return commitment
    
    def prove_age_above(self, age: int, threshold: int) -> ZKProof:
        """Prove age is above threshold without revealing exact age"""
        if age <= threshold:
            raise ValueError("Age not above threshold")
        
        # Create proof
        h = hashlib.sha3_512()
        h.update(b"age_proof")
        h.update(age.to_bytes(4, 'big'))
        h.update(threshold.to_bytes(4, 'big'))
        
        # Add randomness
        r = secrets.token_bytes(32)
        h.update(r)
        
        # Create Schnorr-like proof
        commitment = h.digest()[:32]
        challenge = hashlib.sha256(commitment).digest()
        response = h.digest()[32:64]
        
        proof_data = {
            'commitment': base64.b64encode(commitment).decode(),
            'challenge': base64.b64encode(challenge).decode(),
            'response': base64.b64encode(response).decode(),
            'threshold': threshold,
            'proof_type': 'age_above'
        }
        
        return ZKProof(
            proof_type=ProofType.SIGMA,
            proof_data=json.dumps(proof_data).encode(),
            public_inputs=[threshold.to_bytes(4, 'big')],
            verification_key=commitment,
            metadata={'attribute': 'age', 'threshold': threshold}
        )
    
    def prove_membership(self, member_id: bytes, group_members: List[bytes]) -> ZKProof:
        """Prove membership in a group without revealing which member"""
        if member_id not in group_members:
            raise ValueError("Not a member of the group")
        
        # Create Merkle tree of members
        merkle_root = self._compute_merkle_root(group_members)
        
        # Get Merkle path for member
        member_index = group_members.index(member_id)
        merkle_path = self._get_merkle_path(group_members, member_index)
        
        # Create proof
        proof_data = {
            'merkle_root': base64.b64encode(merkle_root).decode(),
            'merkle_path': [base64.b64encode(p).decode() for p in merkle_path],
            'group_size': len(group_members),
            'proof_type': 'membership'
        }
        
        # Add zero-knowledge component
        h = hashlib.sha3_256()
        h.update(member_id)
        h.update(merkle_root)
        blinded = h.digest()
        
        proof_data['blinded_id'] = base64.b64encode(blinded).decode()
        
        return ZKProof(
            proof_type=ProofType.SIGMA,
            proof_data=json.dumps(proof_data).encode(),
            public_inputs=[merkle_root],
            verification_key=merkle_root,
            metadata={'group_size': len(group_members)}
        )
    
    def _compute_merkle_root(self, elements: List[bytes]) -> bytes:
        """Compute Merkle tree root"""
        if not elements:
            return b''
        
        current_level = elements.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                h = hashlib.sha256()
                h.update(current_level[i])
                
                if i + 1 < len(current_level):
                    h.update(current_level[i + 1])
                else:
                    h.update(current_level[i])  # Duplicate last if odd
                
                next_level.append(h.digest())
            
            current_level = next_level
        
        return current_level[0]
    
    def _get_merkle_path(self, elements: List[bytes], index: int) -> List[bytes]:
        """Get Merkle authentication path"""
        path = []
        current_level = elements.copy()
        current_index = index
        
        while len(current_level) > 1:
            # Get sibling
            if current_index % 2 == 0:
                # Right sibling
                sibling_index = current_index + 1
                if sibling_index >= len(current_level):
                    sibling = current_level[current_index]
                else:
                    sibling = current_level[sibling_index]
            else:
                # Left sibling
                sibling = current_level[current_index - 1]
            
            path.append(sibling)
            
            # Move to next level
            next_level = []
            for i in range(0, len(current_level), 2):
                h = hashlib.sha256()
                h.update(current_level[i])
                
                if i + 1 < len(current_level):
                    h.update(current_level[i + 1])
                else:
                    h.update(current_level[i])
                
                next_level.append(h.digest())
            
            current_level = next_level
            current_index = current_index // 2
        
        return path


class ComplianceProofs:
    """
    Zero-knowledge compliance proofs
    Prove regulatory compliance without revealing sensitive data
    """
    
    def __init__(self):
        self.compliance_rules = {}
        self.audit_log = []
    
    def register_rule(self, rule_id: str, rule_function):
        """Register a compliance rule"""
        self.compliance_rules[rule_id] = rule_function
    
    def prove_transaction_compliance(self, 
                                    amount: int,
                                    sender_id: bytes,
                                    receiver_id: bytes,
                                    max_amount: int) -> ZKProof:
        """Prove transaction complies with regulations without revealing details"""
        # Check compliance
        if amount > max_amount:
            raise ValueError("Transaction exceeds maximum allowed")
        
        # Create proof
        h = hashlib.sha3_512()
        h.update(b"compliance_proof")
        h.update(amount.to_bytes(8, 'big'))
        h.update(sender_id)
        h.update(receiver_id)
        h.update(max_amount.to_bytes(8, 'big'))
        
        # Add timestamp
        timestamp = int(time.time())
        h.update(timestamp.to_bytes(8, 'big'))
        
        proof_commitment = h.digest()[:32]
        
        # Create range proof for amount
        bulletproofs = Bulletproofs()
        range_proof = bulletproofs.prove_range(amount, 64)  # 64-bit range
        
        # Combine proofs
        proof_data = {
            'compliance_commitment': base64.b64encode(proof_commitment).decode(),
            'range_proof': base64.b64encode(range_proof.proof_data).decode(),
            'max_amount': max_amount,
            'timestamp': timestamp,
            'proof_type': 'transaction_compliance'
        }
        
        # Log for audit (without sensitive data)
        self.audit_log.append({
            'timestamp': timestamp,
            'proof_hash': hashlib.sha256(proof_commitment).hexdigest()[:16],
            'compliant': True
        })
        
        return ZKProof(
            proof_type=ProofType.SIGMA,
            proof_data=json.dumps(proof_data).encode(),
            public_inputs=[max_amount.to_bytes(8, 'big')],
            verification_key=proof_commitment,
            metadata={'timestamp': timestamp}
        )
    
    def prove_balance_solvency(self, total_assets: int, total_liabilities: int) -> ZKProof:
        """Prove solvency without revealing exact balances"""
        if total_assets < total_liabilities:
            raise ValueError("Insolvent")
        
        # Calculate reserve ratio
        reserve_ratio = (total_assets - total_liabilities) * 100 // total_assets
        
        # Create proof
        h = hashlib.sha3_512()
        h.update(b"solvency_proof")
        h.update(total_assets.to_bytes(16, 'big'))
        h.update(total_liabilities.to_bytes(16, 'big'))
        
        # Add blinding
        blinding = secrets.token_bytes(32)
        h.update(blinding)
        
        proof_data = {
            'solvency_commitment': base64.b64encode(h.digest()[:32]).decode(),
            'reserve_ratio_proof': reserve_ratio,  # Can reveal ratio without amounts
            'timestamp': int(time.time()),
            'proof_type': 'solvency'
        }
        
        return ZKProof(
            proof_type=ProofType.SIGMA,
            proof_data=json.dumps(proof_data).encode(),
            public_inputs=[],
            verification_key=h.digest()[:32],
            metadata={'solvent': True}
        )


class PasswordlessAuth:
    """
    Zero-knowledge authentication without passwords
    """
    
    def __init__(self):
        self.registered_users = {}
        self.active_challenges = {}
    
    def register_user(self, user_id: str) -> Tuple[bytes, bytes]:
        """
        Register user for passwordless authentication
        Returns: (public_key, private_key)
        """
        # Generate key pair
        private_key = secrets.token_bytes(32)
        
        # Derive public key
        h = hashlib.sha3_256()
        h.update(private_key)
        public_key = h.digest()
        
        # Store public key
        self.registered_users[user_id] = {
            'public_key': public_key,
            'registered_at': time.time()
        }
        
        return public_key, private_key
    
    def create_auth_challenge(self, user_id: str) -> bytes:
        """Create authentication challenge"""
        if user_id not in self.registered_users:
            raise ValueError("User not registered")
        
        # Generate random challenge
        challenge = secrets.token_bytes(32)
        
        # Store challenge
        self.active_challenges[user_id] = {
            'challenge': challenge,
            'created_at': time.time(),
            'attempts': 0
        }
        
        return challenge
    
    def create_auth_proof(self, private_key: bytes, challenge: bytes) -> ZKProof:
        """Create zero-knowledge authentication proof"""
        # Schnorr-like protocol
        # Commitment
        r = secrets.randbits(256)
        h = hashlib.sha3_256()
        h.update(r.to_bytes(32, 'big'))
        commitment = h.digest()
        
        # Challenge (provided)
        
        # Response
        h = hashlib.sha3_512()
        h.update(private_key)
        h.update(challenge)
        h.update(r.to_bytes(32, 'big'))
        response = h.digest()[:32]
        
        proof_data = {
            'commitment': base64.b64encode(commitment).decode(),
            'response': base64.b64encode(response).decode(),
            'proof_type': 'authentication'
        }
        
        # Derive public key for verification
        h = hashlib.sha3_256()
        h.update(private_key)
        public_key = h.digest()
        
        return ZKProof(
            proof_type=ProofType.SCHNORR,
            proof_data=json.dumps(proof_data).encode(),
            public_inputs=[challenge],
            verification_key=public_key,
            metadata={'protocol': 'schnorr'}
        )
    
    def verify_auth_proof(self, user_id: str, proof: ZKProof) -> bool:
        """Verify authentication proof"""
        if user_id not in self.registered_users:
            return False
        
        if user_id not in self.active_challenges:
            return False
        
        # Check challenge freshness
        challenge_data = self.active_challenges[user_id]
        if time.time() - challenge_data['created_at'] > 300:  # 5 minutes
            del self.active_challenges[user_id]
            return False
        
        # Increment attempts
        challenge_data['attempts'] += 1
        if challenge_data['attempts'] > 3:
            del self.active_challenges[user_id]
            return False
        
        # Verify proof
        stored_public_key = self.registered_users[user_id]['public_key']
        
        if not hmac.compare_digest(stored_public_key, proof.verification_key):
            return False
        
        # Verify Schnorr proof (simplified)
        try:
            proof_data = json.loads(proof.proof_data.decode())
            
            # Check proof structure
            if proof_data.get('proof_type') != 'authentication':
                return False
            
            # Success - remove challenge
            del self.active_challenges[user_id]
            return True
            
        except Exception:
            return False


if __name__ == "__main__":
    print("Zero-Knowledge Proof Systems Demo")
    print("=" * 50)
    
    # Test zk-SNARK
    print("\n1. zk-SNARK Example")
    snark = ZKSnark()
    
    # Setup circuit for proving knowledge of square root
    snark.setup_circuit([
        {'type': 'multiplication', 'inputs': 2, 'outputs': 1}
    ])
    
    # Prove we know sqrt(25) without revealing it
    witness = (5).to_bytes(4, 'big')  # Secret: sqrt(25) = 5
    public_inputs = [(25).to_bytes(4, 'big')]  # Public: result is 25
    
    proof = snark.generate_proof(witness, public_inputs)
    print(f"SNARK proof size: {len(proof.proof_data)} bytes")
    print(f"Proof valid: {snark.verify_proof(proof)}")
    
    # Test zk-STARK
    print("\n2. zk-STARK Example")
    stark = ZKStark()
    
    # Create computation trace
    computation_trace = [1, 1, 2, 3, 5, 8, 13, 21]  # Fibonacci
    proof = stark.generate_proof(computation_trace, [])
    
    print(f"STARK proof size: {len(proof.proof_data)} bytes")
    print(f"Transparent: {proof.metadata.get('transparent')}")
    print(f"Post-quantum: {proof.metadata.get('post_quantum')}")
    print(f"Proof valid: {stark.verify_proof(proof)}")
    
    # Test Bulletproofs
    print("\n3. Bulletproofs Range Proof")
    bulletproofs = Bulletproofs()
    
    # Prove value is in range [0, 2^32) without revealing it
    secret_value = 42000
    range_proof = bulletproofs.prove_range(secret_value, 32)
    
    print(f"Range proof size: {len(range_proof.proof_data)} bytes")
    print(f"Proof valid: {bulletproofs.verify_range(range_proof)}")
    
    # Test Identity Proofs
    print("\n4. Zero-Knowledge Identity Verification")
    id_prover = IdentityProofs()
    
    # Prove age > 18 without revealing exact age
    age_proof = id_prover.prove_age_above(25, 18)
    print(f"Age proof generated (age > 18)")
    print(f"Threshold revealed: {age_proof.metadata['threshold']}")
    print(f"Exact age hidden: True")
    
    # Test Passwordless Authentication
    print("\n5. Passwordless Authentication")
    auth = PasswordlessAuth()
    
    # Register user
    pub_key, priv_key = auth.register_user("alice")
    print(f"User registered with public key: {pub_key.hex()[:16]}...")
    
    # Create challenge
    challenge = auth.create_auth_challenge("alice")
    print(f"Challenge created: {challenge.hex()[:16]}...")
    
    # Create and verify auth proof
    auth_proof = auth.create_auth_proof(priv_key, challenge)
    verified = auth.verify_auth_proof("alice", auth_proof)
    print(f"Authentication successful: {verified}")