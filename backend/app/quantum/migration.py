"""
Cryptographic Migration and Algorithm Agility

Tools for migrating from classical to post-quantum cryptography:
- Algorithm agility framework
- Backward compatibility layers
- Key migration utilities
- Risk assessment tools
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import base64
import logging

from .post_quantum import (
    KyberKEM, 
    DilithiumSignature,
    SecurityLevel,
    QuantumRandomNumberGenerator
)


class MigrationPhase(Enum):
    """Phases of crypto migration"""
    ASSESSMENT = "assessment"
    PREPARATION = "preparation"
    DUAL_MODE = "dual_mode"
    TRANSITION = "transition"
    POST_QUANTUM = "post_quantum"
    DEPRECATED = "deprecated"


class AlgorithmType(Enum):
    """Types of cryptographic algorithms"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_ENCRYPTION = "asymmetric_encryption"
    SIGNATURE = "signature"
    KEY_EXCHANGE = "key_exchange"
    HASH = "hash"
    MAC = "mac"


class RiskLevel(Enum):
    """Risk levels for quantum threats"""
    CRITICAL = "critical"  # Immediate migration needed
    HIGH = "high"          # Migration within 1 year
    MEDIUM = "medium"      # Migration within 3 years
    LOW = "low"            # Migration within 5 years
    MINIMAL = "minimal"    # Monitor only


@dataclass
class AlgorithmProfile:
    """Profile of a cryptographic algorithm"""
    name: str
    type: AlgorithmType
    is_quantum_safe: bool
    key_size: int
    security_bits: int
    performance_factor: float  # Relative to baseline
    risk_level: RiskLevel
    migration_deadline: Optional[datetime] = None
    replacement_algorithm: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationPlan:
    """Migration plan for an organization"""
    organization_id: str
    current_phase: MigrationPhase
    target_completion: datetime
    algorithms_to_migrate: List[AlgorithmProfile]
    completed_migrations: List[str]
    risk_assessment: Dict[str, RiskLevel]
    timeline: List[Dict[str, Any]]


class CryptoMigrationManager:
    """
    Manages migration from classical to post-quantum cryptography
    """
    
    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize algorithm inventory
        self.algorithm_inventory = self._initialize_inventory()
        
        # Current migration plan
        self.migration_plan = None
        
        # Migration history
        self.migration_history = []
        
        # Risk assessment cache
        self.risk_cache = {}
        
    def _initialize_inventory(self) -> Dict[str, AlgorithmProfile]:
        """Initialize inventory of cryptographic algorithms"""
        inventory = {
            # Classical algorithms (need migration)
            "RSA-2048": AlgorithmProfile(
                name="RSA-2048",
                type=AlgorithmType.ASYMMETRIC_ENCRYPTION,
                is_quantum_safe=False,
                key_size=2048,
                security_bits=112,
                performance_factor=1.0,
                risk_level=RiskLevel.HIGH,
                replacement_algorithm="Kyber768"
            ),
            "RSA-3072": AlgorithmProfile(
                name="RSA-3072",
                type=AlgorithmType.ASYMMETRIC_ENCRYPTION,
                is_quantum_safe=False,
                key_size=3072,
                security_bits=128,
                performance_factor=0.7,
                risk_level=RiskLevel.HIGH,
                replacement_algorithm="Kyber768"
            ),
            "ECDSA-P256": AlgorithmProfile(
                name="ECDSA-P256",
                type=AlgorithmType.SIGNATURE,
                is_quantum_safe=False,
                key_size=256,
                security_bits=128,
                performance_factor=3.0,
                risk_level=RiskLevel.HIGH,
                replacement_algorithm="Dilithium3"
            ),
            "ECDH-P256": AlgorithmProfile(
                name="ECDH-P256",
                type=AlgorithmType.KEY_EXCHANGE,
                is_quantum_safe=False,
                key_size=256,
                security_bits=128,
                performance_factor=3.0,
                risk_level=RiskLevel.HIGH,
                replacement_algorithm="Kyber768"
            ),
            
            # Post-quantum algorithms (migration targets)
            "Kyber768": AlgorithmProfile(
                name="Kyber768",
                type=AlgorithmType.KEY_EXCHANGE,
                is_quantum_safe=True,
                key_size=1184,
                security_bits=192,
                performance_factor=2.5,
                risk_level=RiskLevel.MINIMAL
            ),
            "Dilithium3": AlgorithmProfile(
                name="Dilithium3",
                type=AlgorithmType.SIGNATURE,
                is_quantum_safe=True,
                key_size=1952,
                security_bits=192,
                performance_factor=2.0,
                risk_level=RiskLevel.MINIMAL
            ),
            "FALCON-512": AlgorithmProfile(
                name="FALCON-512",
                type=AlgorithmType.SIGNATURE,
                is_quantum_safe=True,
                key_size=897,
                security_bits=128,
                performance_factor=4.0,
                risk_level=RiskLevel.MINIMAL
            ),
            
            # Symmetric algorithms (already quantum-safe with sufficient key size)
            "AES-256": AlgorithmProfile(
                name="AES-256",
                type=AlgorithmType.SYMMETRIC,
                is_quantum_safe=True,
                key_size=256,
                security_bits=256,
                performance_factor=5.0,
                risk_level=RiskLevel.MINIMAL
            ),
            "ChaCha20-Poly1305": AlgorithmProfile(
                name="ChaCha20-Poly1305",
                type=AlgorithmType.SYMMETRIC,
                is_quantum_safe=True,
                key_size=256,
                security_bits=256,
                performance_factor=4.5,
                risk_level=RiskLevel.MINIMAL
            ),
            
            # Hash functions (quantum impact is limited)
            "SHA-256": AlgorithmProfile(
                name="SHA-256",
                type=AlgorithmType.HASH,
                is_quantum_safe=True,
                key_size=0,
                security_bits=128,  # Against quantum attacks
                performance_factor=6.0,
                risk_level=RiskLevel.LOW
            ),
            "SHA3-256": AlgorithmProfile(
                name="SHA3-256",
                type=AlgorithmType.HASH,
                is_quantum_safe=True,
                key_size=0,
                security_bits=128,
                performance_factor=4.0,
                risk_level=RiskLevel.MINIMAL
            )
        }
        
        return inventory
    
    def assess_quantum_risk(self, algorithm_usage: Dict[str, int]) -> Dict[str, RiskLevel]:
        """
        Assess quantum risk for current cryptographic usage
        algorithm_usage: {algorithm_name: usage_count}
        """
        risk_assessment = {}
        
        for algo_name, usage_count in algorithm_usage.items():
            if algo_name not in self.algorithm_inventory:
                # Unknown algorithm - assume high risk
                risk_assessment[algo_name] = RiskLevel.HIGH
                continue
            
            profile = self.algorithm_inventory[algo_name]
            
            # Adjust risk based on usage
            base_risk = profile.risk_level
            
            if usage_count > 10000:
                # Heavy usage increases urgency
                if base_risk == RiskLevel.MEDIUM:
                    risk_assessment[algo_name] = RiskLevel.HIGH
                elif base_risk == RiskLevel.LOW:
                    risk_assessment[algo_name] = RiskLevel.MEDIUM
                else:
                    risk_assessment[algo_name] = base_risk
            else:
                risk_assessment[algo_name] = base_risk
        
        self.risk_cache = risk_assessment
        return risk_assessment
    
    def create_migration_plan(self, 
                            algorithm_usage: Dict[str, int],
                            target_completion_months: int = 24) -> MigrationPlan:
        """Create a migration plan based on current usage and risk"""
        # Assess risk
        risk_assessment = self.assess_quantum_risk(algorithm_usage)
        
        # Identify algorithms needing migration
        to_migrate = []
        for algo_name, risk in risk_assessment.items():
            if algo_name in self.algorithm_inventory:
                profile = self.algorithm_inventory[algo_name]
                if not profile.is_quantum_safe and risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                    # Set migration deadline based on risk
                    if risk == RiskLevel.CRITICAL:
                        profile.migration_deadline = datetime.now() + timedelta(days=90)
                    else:
                        profile.migration_deadline = datetime.now() + timedelta(days=365)
                    
                    to_migrate.append(profile)
        
        # Sort by risk and deadline
        to_migrate.sort(key=lambda x: (x.risk_level.value, x.migration_deadline))
        
        # Create timeline
        timeline = self._create_migration_timeline(to_migrate, target_completion_months)
        
        # Create plan
        plan = MigrationPlan(
            organization_id=self.organization_id,
            current_phase=MigrationPhase.ASSESSMENT,
            target_completion=datetime.now() + timedelta(days=target_completion_months * 30),
            algorithms_to_migrate=to_migrate,
            completed_migrations=[],
            risk_assessment=risk_assessment,
            timeline=timeline
        )
        
        self.migration_plan = plan
        return plan
    
    def _create_migration_timeline(self, algorithms: List[AlgorithmProfile],
                                  total_months: int) -> List[Dict[str, Any]]:
        """Create detailed migration timeline"""
        timeline = []
        
        # Phase 1: Assessment (10% of time)
        assessment_end = datetime.now() + timedelta(days=total_months * 3)
        timeline.append({
            'phase': MigrationPhase.ASSESSMENT,
            'start': datetime.now(),
            'end': assessment_end,
            'tasks': [
                'Complete cryptographic inventory',
                'Identify all algorithm usage',
                'Assess quantum risk levels',
                'Review compliance requirements'
            ]
        })
        
        # Phase 2: Preparation (20% of time)
        prep_end = assessment_end + timedelta(days=total_months * 6)
        timeline.append({
            'phase': MigrationPhase.PREPARATION,
            'start': assessment_end,
            'end': prep_end,
            'tasks': [
                'Select post-quantum algorithms',
                'Update cryptographic libraries',
                'Develop migration tools',
                'Train development team'
            ]
        })
        
        # Phase 3: Dual Mode (40% of time)
        dual_end = prep_end + timedelta(days=total_months * 12)
        timeline.append({
            'phase': MigrationPhase.DUAL_MODE,
            'start': prep_end,
            'end': dual_end,
            'tasks': [
                'Implement hybrid crypto systems',
                'Deploy backward compatibility',
                'Test interoperability',
                'Monitor performance'
            ],
            'algorithms': algorithms[:len(algorithms)//2]  # First half
        })
        
        # Phase 4: Transition (20% of time)
        transition_end = dual_end + timedelta(days=total_months * 6)
        timeline.append({
            'phase': MigrationPhase.TRANSITION,
            'start': dual_end,
            'end': transition_end,
            'tasks': [
                'Migrate remaining systems',
                'Phase out classical algorithms',
                'Update all certificates',
                'Validate security'
            ],
            'algorithms': algorithms[len(algorithms)//2:]  # Second half
        })
        
        # Phase 5: Post-Quantum (10% of time)
        pq_end = transition_end + timedelta(days=total_months * 3)
        timeline.append({
            'phase': MigrationPhase.POST_QUANTUM,
            'start': transition_end,
            'end': pq_end,
            'tasks': [
                'Complete migration',
                'Decommission legacy systems',
                'Final security audit',
                'Document new procedures'
            ]
        })
        
        return timeline
    
    def execute_migration_step(self, algorithm_name: str) -> bool:
        """Execute migration for a specific algorithm"""
        if algorithm_name not in self.algorithm_inventory:
            self.logger.error(f"Unknown algorithm: {algorithm_name}")
            return False
        
        profile = self.algorithm_inventory[algorithm_name]
        
        if profile.is_quantum_safe:
            self.logger.info(f"{algorithm_name} is already quantum-safe")
            return True
        
        if not profile.replacement_algorithm:
            self.logger.error(f"No replacement algorithm defined for {algorithm_name}")
            return False
        
        try:
            # Simulate migration steps
            self.logger.info(f"Starting migration: {algorithm_name} -> {profile.replacement_algorithm}")
            
            # Step 1: Deploy new algorithm
            self._deploy_algorithm(profile.replacement_algorithm)
            
            # Step 2: Migrate keys/certificates
            self._migrate_keys(algorithm_name, profile.replacement_algorithm)
            
            # Step 3: Update configurations
            self._update_configurations(algorithm_name, profile.replacement_algorithm)
            
            # Step 4: Verify migration
            if self._verify_migration(algorithm_name, profile.replacement_algorithm):
                # Mark as completed
                if self.migration_plan:
                    self.migration_plan.completed_migrations.append(algorithm_name)
                
                # Add to history
                self.migration_history.append({
                    'algorithm': algorithm_name,
                    'replacement': profile.replacement_algorithm,
                    'timestamp': datetime.now(),
                    'success': True
                })
                
                self.logger.info(f"Successfully migrated {algorithm_name}")
                return True
            
        except Exception as e:
            self.logger.error(f"Migration failed for {algorithm_name}: {e}")
            
        return False
    
    def _deploy_algorithm(self, algorithm_name: str):
        """Deploy new post-quantum algorithm"""
        self.logger.info(f"Deploying {algorithm_name}")
        # Simulation - actual implementation would deploy libraries/configs
        time.sleep(0.1)
    
    def _migrate_keys(self, old_algo: str, new_algo: str):
        """Migrate keys from old to new algorithm"""
        self.logger.info(f"Migrating keys from {old_algo} to {new_algo}")
        # Simulation - actual implementation would re-key systems
        time.sleep(0.1)
    
    def _update_configurations(self, old_algo: str, new_algo: str):
        """Update system configurations"""
        self.logger.info(f"Updating configurations: {old_algo} -> {new_algo}")
        # Simulation - actual implementation would update config files
        time.sleep(0.1)
    
    def _verify_migration(self, old_algo: str, new_algo: str) -> bool:
        """Verify migration was successful"""
        self.logger.info(f"Verifying migration: {old_algo} -> {new_algo}")
        # Simulation - actual implementation would run tests
        return True
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        if not self.migration_plan:
            return {'status': 'No migration plan created'}
        
        plan = self.migration_plan
        total_algorithms = len(plan.algorithms_to_migrate)
        completed = len(plan.completed_migrations)
        
        return {
            'organization': self.organization_id,
            'current_phase': plan.current_phase.value,
            'progress_percentage': (completed / total_algorithms * 100) if total_algorithms > 0 else 0,
            'algorithms_migrated': completed,
            'algorithms_remaining': total_algorithms - completed,
            'target_completion': plan.target_completion.isoformat(),
            'high_risk_algorithms': [
                algo.name for algo in plan.algorithms_to_migrate 
                if algo.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
            ]
        }


class AlgorithmAgility:
    """
    Algorithm agility framework - ability to quickly switch algorithms
    """
    
    def __init__(self):
        self.supported_algorithms = {}
        self.algorithm_versions = {}
        self.compatibility_matrix = {}
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize supported algorithms with version info"""
        self.supported_algorithms = {
            'encryption': {
                'classical': ['RSA', 'ECDH', 'DH'],
                'post_quantum': ['Kyber', 'NTRU', 'McEliece'],
                'hybrid': ['Kyber+ECDH', 'NTRU+RSA']
            },
            'signature': {
                'classical': ['RSA', 'ECDSA', 'EdDSA'],
                'post_quantum': ['Dilithium', 'FALCON', 'SPHINCS+'],
                'hybrid': ['Dilithium+ECDSA', 'FALCON+EdDSA']
            },
            'symmetric': {
                'current': ['AES-128', 'AES-256', 'ChaCha20'],
                'recommended': ['AES-256', 'ChaCha20-Poly1305']
            }
        }
        
        # Version tracking
        self.algorithm_versions = {
            'Kyber': ['Kyber512', 'Kyber768', 'Kyber1024'],
            'Dilithium': ['Dilithium2', 'Dilithium3', 'Dilithium5'],
            'FALCON': ['FALCON-512', 'FALCON-1024'],
            'SPHINCS+': ['SPHINCS+-128f', 'SPHINCS+-192f', 'SPHINCS+-256f']
        }
    
    def negotiate_algorithm(self, 
                           client_algorithms: List[str],
                           server_algorithms: List[str],
                           prefer_quantum_safe: bool = True) -> Optional[str]:
        """
        Negotiate best algorithm between client and server
        """
        # Find common algorithms
        common = set(client_algorithms) & set(server_algorithms)
        
        if not common:
            return None
        
        # Rank algorithms by preference
        ranked = []
        for algo in common:
            score = self._score_algorithm(algo, prefer_quantum_safe)
            ranked.append((algo, score))
        
        # Sort by score (higher is better)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked[0][0] if ranked else None
    
    def _score_algorithm(self, algorithm: str, prefer_quantum_safe: bool) -> int:
        """Score algorithm based on security and performance"""
        score = 0
        
        # Check if quantum-safe
        is_pq = any(algorithm.startswith(pq) for cat in self.supported_algorithms.values() 
                   for pq in cat.get('post_quantum', []))
        
        if is_pq:
            score += 100 if prefer_quantum_safe else 50
        
        # Check if hybrid (best of both worlds)
        is_hybrid = any(algorithm in cat.get('hybrid', []) 
                       for cat in self.supported_algorithms.values())
        
        if is_hybrid:
            score += 75
        
        # Performance scoring (simplified)
        if 'Kyber' in algorithm:
            score += 20  # Fast KEM
        elif 'FALCON' in algorithm:
            score += 15  # Compact signatures
        elif 'Dilithium' in algorithm:
            score += 10  # Standard choice
        
        return score
    
    def create_algorithm_bundle(self, 
                              use_case: str,
                              security_level: SecurityLevel) -> Dict[str, str]:
        """
        Create a bundle of algorithms for specific use case
        """
        bundles = {
            'web_server': {
                'key_exchange': 'Kyber768' if security_level == SecurityLevel.LEVEL3 else 'Kyber512',
                'signature': 'Dilithium3' if security_level == SecurityLevel.LEVEL3 else 'Dilithium2',
                'symmetric': 'AES-256-GCM',
                'hash': 'SHA3-256'
            },
            'iot_device': {
                'key_exchange': 'Kyber512',  # Smaller keys for IoT
                'signature': 'FALCON-512',     # Compact signatures
                'symmetric': 'ChaCha20-Poly1305',  # Efficient on low-power
                'hash': 'SHA-256'
            },
            'long_term_storage': {
                'key_exchange': 'Classic-McEliece',  # Maximum security
                'signature': 'SPHINCS+-256f',        # Stateless, very secure
                'symmetric': 'AES-256-GCM',
                'hash': 'SHA3-512'
            },
            'high_performance': {
                'key_exchange': 'Kyber512',
                'signature': 'FALCON-512',
                'symmetric': 'ChaCha20-Poly1305',
                'hash': 'BLAKE2b'
            }
        }
        
        return bundles.get(use_case, bundles['web_server'])


class BackwardCompatibility:
    """
    Maintains backward compatibility during migration
    """
    
    def __init__(self):
        self.compatibility_layers = {}
        self.protocol_translators = {}
        self.legacy_support = True
    
    def create_compatibility_layer(self, 
                                  old_algorithm: str,
                                  new_algorithm: str) -> Dict[str, Any]:
        """Create compatibility layer between old and new algorithms"""
        layer = {
            'old': old_algorithm,
            'new': new_algorithm,
            'translator': self._get_translator(old_algorithm, new_algorithm),
            'performance_impact': self._estimate_performance_impact(old_algorithm, new_algorithm),
            'security_implications': self._assess_security_implications(old_algorithm, new_algorithm)
        }
        
        layer_id = f"{old_algorithm}_to_{new_algorithm}"
        self.compatibility_layers[layer_id] = layer
        
        return layer
    
    def _get_translator(self, old_algo: str, new_algo: str):
        """Get or create protocol translator"""
        translator_key = f"{old_algo}_{new_algo}"
        
        if translator_key not in self.protocol_translators:
            # Create new translator
            self.protocol_translators[translator_key] = {
                'type': 'hybrid' if 'RSA' in old_algo and 'Kyber' in new_algo else 'wrapper',
                'overhead_bytes': 1024 if 'McEliece' in new_algo else 256,
                'cpu_overhead_percent': 15
            }
        
        return self.protocol_translators[translator_key]
    
    def _estimate_performance_impact(self, old_algo: str, new_algo: str) -> float:
        """Estimate performance impact of compatibility layer"""
        # Simplified estimation
        base_impact = 1.0
        
        if 'RSA' in old_algo and 'Kyber' in new_algo:
            base_impact *= 1.5  # Hybrid mode overhead
        
        if 'McEliece' in new_algo:
            base_impact *= 2.0  # Large key overhead
        
        return base_impact
    
    def _assess_security_implications(self, old_algo: str, new_algo: str) -> Dict[str, Any]:
        """Assess security implications of compatibility"""
        return {
            'maintains_forward_secrecy': 'Kyber' in new_algo or 'X25519' in new_algo,
            'quantum_resistance': any(pq in new_algo for pq in ['Kyber', 'Dilithium', 'FALCON']),
            'weakest_link': old_algo if 'RSA' in old_algo or 'ECDSA' in old_algo else new_algo,
            'recommended_duration': '1 year' if 'RSA' in old_algo else '2 years'
        }
    
    def wrap_legacy_protocol(self, 
                            legacy_data: bytes,
                            target_algorithm: str) -> bytes:
        """Wrap legacy protocol data for new algorithm"""
        # Create wrapper header
        header = {
            'version': 1,
            'legacy_format': True,
            'target_algorithm': target_algorithm,
            'timestamp': time.time()
        }
        
        header_bytes = json.dumps(header).encode()
        header_len = len(header_bytes).to_bytes(4, 'big')
        
        # Wrap data
        wrapped = header_len + header_bytes + legacy_data
        
        return wrapped
    
    def unwrap_legacy_protocol(self, wrapped_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Unwrap legacy protocol data"""
        # Extract header
        header_len = int.from_bytes(wrapped_data[:4], 'big')
        header_bytes = wrapped_data[4:4+header_len]
        header = json.loads(header_bytes.decode())
        
        # Extract original data
        legacy_data = wrapped_data[4+header_len:]
        
        return legacy_data, header


if __name__ == "__main__":
    print("Cryptographic Migration Manager Demo")
    print("=" * 50)
    
    # Initialize migration manager
    manager = CryptoMigrationManager("example_org")
    
    # Simulate current algorithm usage
    current_usage = {
        'RSA-2048': 5000,
        'ECDSA-P256': 3000,
        'AES-256': 10000,
        'SHA-256': 15000
    }
    
    # Create migration plan
    plan = manager.create_migration_plan(current_usage, target_completion_months=24)
    
    print(f"\nMigration Plan for {plan.organization_id}")
    print(f"Target Completion: {plan.target_completion}")
    print(f"Algorithms to Migrate: {len(plan.algorithms_to_migrate)}")
    
    for algo in plan.algorithms_to_migrate:
        print(f"  - {algo.name} -> {algo.replacement_algorithm} (Risk: {algo.risk_level.value})")
    
    # Test algorithm agility
    print("\n\nAlgorithm Agility Testing")
    print("-" * 30)
    
    agility = AlgorithmAgility()
    
    client_algos = ['Kyber768', 'Dilithium3', 'RSA-2048']
    server_algos = ['Kyber768', 'FALCON-512', 'ECDSA-P256']
    
    negotiated = agility.negotiate_algorithm(client_algos, server_algos, prefer_quantum_safe=True)
    print(f"Negotiated Algorithm: {negotiated}")
    
    # Get algorithm bundle for web server
    bundle = agility.create_algorithm_bundle('web_server', SecurityLevel.LEVEL3)
    print(f"\nWeb Server Bundle:")
    for purpose, algo in bundle.items():
        print(f"  {purpose}: {algo}")
    
    # Test backward compatibility
    print("\n\nBackward Compatibility Layer")
    print("-" * 30)
    
    compat = BackwardCompatibility()
    layer = compat.create_compatibility_layer('RSA-2048', 'Kyber768')
    
    print(f"Compatibility: {layer['old']} -> {layer['new']}")
    print(f"Performance Impact: {layer['performance_impact']}x")
    print(f"Security: {layer['security_implications']}")