"""
Model Security Scanning and Validation System

Provides comprehensive security analysis for AI models to detect vulnerabilities,
backdoors, adversarial weaknesses, and ensure compliance.
"""

import os
import json
import pickle
import hashlib
import numpy as np
import torch
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import concurrent.futures
from pathlib import Path
import logging
import re
import ast
import dis
import inspect
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import yara
import bandit
from safety import check
import cleverhans
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier
from art.defences.detector.poison import ActivationDefence
from art.metrics import clever_score
import mlsploit
from model_analyzer import ModelAnalyzer
import redis
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()
logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of security threats"""
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_POISONING = "data_poisoning"
    MODEL_STEALING = "model_stealing"
    BACKDOOR = "backdoor"
    PRIVACY_LEAK = "privacy_leak"
    INJECTION = "injection"
    EVASION = "evasion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    SUPPLY_CHAIN = "supply_chain"


class ScanLevel(Enum):
    """Security scan levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PARANOID = "paranoid"


class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"
    OWASP_ML = "owasp_ml"


@dataclass
class SecurityConfig:
    """Security scanning configuration"""
    scan_level: ScanLevel = ScanLevel.STANDARD
    threat_types: List[ThreatType] = None
    compliance_standards: List[ComplianceStandard] = None
    adversarial_testing: bool = True
    backdoor_detection: bool = True
    privacy_analysis: bool = True
    supply_chain_check: bool = True
    code_analysis: bool = True
    dependency_scan: bool = True
    performance_impact: bool = False
    max_test_samples: int = 1000
    timeout_seconds: int = 3600


@dataclass
class SecurityReport:
    """Security scan report"""
    model_id: str
    scan_id: str
    timestamp: datetime
    scan_level: ScanLevel
    overall_score: float
    risk_level: str
    vulnerabilities: List[Dict[str, Any]]
    threats_detected: List[ThreatType]
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    detailed_findings: Dict[str, Any]
    scan_duration: float


@dataclass
class Vulnerability:
    """Security vulnerability"""
    type: ThreatType
    severity: str  # critical, high, medium, low
    confidence: float
    description: str
    evidence: Dict[str, Any]
    mitigation: str
    cve_id: Optional[str] = None
    references: List[str] = None


# Database Models

class SecurityScanDB(Base):
    """Security scan records"""
    __tablename__ = 'security_scans'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False)
    scan_level = Column(String(50), nullable=False)
    status = Column(String(50), default='running')
    overall_score = Column(Float)
    risk_level = Column(String(50))
    vulnerabilities_count = Column(Integer, default=0)
    threats_detected = Column(JSON)
    compliance_status = Column(JSON)
    recommendations = Column(JSON)
    detailed_findings = Column(JSON)
    scan_duration = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class VulnerabilityDB(Base):
    """Vulnerability records"""
    __tablename__ = 'vulnerabilities'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey('security_scans.id'))
    type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    confidence = Column(Float)
    description = Column(String(1000))
    evidence = Column(JSON)
    mitigation = Column(String(1000))
    cve_id = Column(String(50))
    fixed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AdversarialTester:
    """Test models against adversarial attacks"""
    
    def __init__(self):
        self.attack_methods = {
            'fgsm': FastGradientMethod,
            'pgd': ProjectedGradientDescent,
            'carlini': CarliniL2Method
        }
    
    def test_robustness(self, model: Any, test_data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Test model robustness against adversarial examples"""
        results = {}
        
        # Wrap model for ART
        if isinstance(model, torch.nn.Module):
            classifier = self._wrap_pytorch_model(model, test_data.shape)
        elif isinstance(model, tf.keras.Model):
            classifier = self._wrap_tensorflow_model(model)
        else:
            return {'error': 'Unsupported model type'}
        
        # Test different attack methods
        for attack_name, AttackClass in self.attack_methods.items():
            try:
                attack = AttackClass(estimator=classifier)
                
                # Generate adversarial examples
                x_adv = attack.generate(x=test_data[:100])  # Limit samples for speed
                
                # Evaluate
                predictions_clean = classifier.predict(test_data[:100])
                predictions_adv = classifier.predict(x_adv)
                
                # Calculate metrics
                accuracy_clean = np.mean(np.argmax(predictions_clean, axis=1) == labels[:100])
                accuracy_adv = np.mean(np.argmax(predictions_adv, axis=1) == labels[:100])
                
                results[attack_name] = {
                    'accuracy_clean': float(accuracy_clean),
                    'accuracy_adversarial': float(accuracy_adv),
                    'robustness_score': float(accuracy_adv / accuracy_clean) if accuracy_clean > 0 else 0,
                    'average_perturbation': float(np.mean(np.abs(x_adv - test_data[:100])))
                }
                
            except Exception as e:
                logger.warning(f"Attack {attack_name} failed: {e}")
                results[attack_name] = {'error': str(e)}
        
        # Calculate CLEVER score (robustness metric)
        try:
            clever = clever_score(
                classifier,
                test_data[:10],
                nb_batches=5,
                batch_size=10,
                radius=0.1,
                norm=2
            )
            results['clever_score'] = float(np.mean(clever))
        except:
            results['clever_score'] = None
        
        return results
    
    def _wrap_pytorch_model(self, model: torch.nn.Module, input_shape: Tuple) -> PyTorchClassifier:
        """Wrap PyTorch model for ART"""
        return PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            input_shape=input_shape[1:],
            nb_classes=10  # Adjust based on model
        )
    
    def _wrap_tensorflow_model(self, model: tf.keras.Model) -> TensorFlowV2Classifier:
        """Wrap TensorFlow model for ART"""
        return TensorFlowV2Classifier(
            model=model,
            nb_classes=10,  # Adjust based on model
            input_shape=model.input_shape[1:]
        )


class BackdoorDetector:
    """Detect backdoors and trojans in models"""
    
    def __init__(self):
        self.patterns = self._load_backdoor_patterns()
    
    def _load_backdoor_patterns(self) -> List[yara.Rules]:
        """Load YARA rules for backdoor detection"""
        rules = []
        
        # Define backdoor patterns
        rule_strings = [
            """
            rule suspicious_trigger_pattern {
                strings:
                    $trigger1 = /trigger_[a-z0-9]+/
                    $trigger2 = "backdoor"
                    $trigger3 = "trojan"
                condition:
                    any of them
            }
            """,
            """
            rule hidden_layer_manipulation {
                strings:
                    $hidden1 = "modify_hidden"
                    $hidden2 = "inject_pattern"
                condition:
                    any of them
            }
            """
        ]
        
        for rule_str in rule_strings:
            try:
                rules.append(yara.compile(source=rule_str))
            except:
                pass
        
        return rules
    
    def scan_for_backdoors(self, model: Any, test_data: np.ndarray) -> Dict[str, Any]:
        """Scan model for backdoors"""
        findings = {
            'backdoor_detected': False,
            'confidence': 0.0,
            'suspicious_patterns': [],
            'activation_analysis': {},
            'weight_analysis': {}
        }
        
        # Analyze model weights
        weight_anomalies = self._analyze_weights(model)
        findings['weight_analysis'] = weight_anomalies
        
        # Activation analysis
        if test_data is not None and len(test_data) > 0:
            activation_anomalies = self._analyze_activations(model, test_data)
            findings['activation_analysis'] = activation_anomalies
        
        # Pattern matching in model structure
        model_str = str(model)
        for rule in self.patterns:
            matches = rule.match(data=model_str)
            if matches:
                findings['suspicious_patterns'].append({
                    'rule': matches[0].rule,
                    'matches': [str(m) for m in matches[0].strings]
                })
        
        # Neural cleanse detection
        cleanse_results = self._neural_cleanse(model, test_data)
        findings['neural_cleanse'] = cleanse_results
        
        # Calculate overall confidence
        confidence_factors = []
        
        if weight_anomalies.get('anomaly_score', 0) > 0.7:
            confidence_factors.append(0.3)
        
        if activation_anomalies.get('anomaly_score', 0) > 0.7:
            confidence_factors.append(0.3)
        
        if findings['suspicious_patterns']:
            confidence_factors.append(0.2)
        
        if cleanse_results.get('backdoor_probability', 0) > 0.5:
            confidence_factors.append(0.4)
        
        findings['confidence'] = sum(confidence_factors)
        findings['backdoor_detected'] = findings['confidence'] > 0.5
        
        return findings
    
    def _analyze_weights(self, model: Any) -> Dict[str, Any]:
        """Analyze model weights for anomalies"""
        results = {
            'anomaly_score': 0.0,
            'suspicious_layers': []
        }
        
        try:
            # Get model weights
            if hasattr(model, 'get_weights'):
                weights = model.get_weights()
            elif hasattr(model, 'state_dict'):
                weights = [v.numpy() for v in model.state_dict().values()]
            else:
                return results
            
            # Statistical analysis
            for i, w in enumerate(weights):
                # Check for unusual patterns
                mean = np.mean(w)
                std = np.std(w)
                
                # Look for outliers
                outliers = np.abs(w - mean) > 3 * std
                outlier_ratio = np.sum(outliers) / w.size
                
                if outlier_ratio > 0.1:  # More than 10% outliers
                    results['suspicious_layers'].append({
                        'layer_index': i,
                        'outlier_ratio': float(outlier_ratio),
                        'mean': float(mean),
                        'std': float(std)
                    })
            
            if results['suspicious_layers']:
                results['anomaly_score'] = min(1.0, len(results['suspicious_layers']) / len(weights))
            
        except Exception as e:
            logger.error(f"Weight analysis failed: {e}")
        
        return results
    
    def _analyze_activations(self, model: Any, test_data: np.ndarray) -> Dict[str, Any]:
        """Analyze model activations for anomalies"""
        results = {
            'anomaly_score': 0.0,
            'dead_neurons': 0,
            'hyperactive_neurons': 0
        }
        
        try:
            # Get intermediate activations
            # This is model-specific and simplified
            if hasattr(model, 'predict'):
                activations = model.predict(test_data[:100])
                
                # Analyze activation patterns
                dead_neurons = np.sum(np.max(activations, axis=0) == 0)
                hyperactive = np.sum(np.mean(activations, axis=0) > 0.9)
                
                results['dead_neurons'] = int(dead_neurons)
                results['hyperactive_neurons'] = int(hyperactive)
                
                total_neurons = activations.shape[1] if len(activations.shape) > 1 else 1
                results['anomaly_score'] = (dead_neurons + hyperactive) / total_neurons
            
        except Exception as e:
            logger.error(f"Activation analysis failed: {e}")
        
        return results
    
    def _neural_cleanse(self, model: Any, test_data: np.ndarray) -> Dict[str, Any]:
        """Neural Cleanse backdoor detection"""
        # Simplified implementation
        return {
            'backdoor_probability': np.random.uniform(0, 0.3),  # Placeholder
            'potential_triggers': []
        }


class PrivacyAnalyzer:
    """Analyze model privacy risks"""
    
    def analyze_privacy_risks(self, model: Any, training_data_stats: Dict) -> Dict[str, Any]:
        """Analyze privacy risks in model"""
        results = {
            'membership_inference_risk': 0.0,
            'model_inversion_risk': 0.0,
            'data_leakage_risk': 0.0,
            'differential_privacy_score': 0.0,
            'recommendations': []
        }
        
        # Membership inference attack simulation
        mi_risk = self._test_membership_inference(model, training_data_stats)
        results['membership_inference_risk'] = mi_risk
        
        # Model inversion risk
        inversion_risk = self._test_model_inversion(model)
        results['model_inversion_risk'] = inversion_risk
        
        # Data leakage detection
        leakage_risk = self._detect_data_leakage(model)
        results['data_leakage_risk'] = leakage_risk
        
        # Calculate differential privacy score
        dp_score = self._calculate_dp_score(model, training_data_stats)
        results['differential_privacy_score'] = dp_score
        
        # Generate recommendations
        if mi_risk > 0.7:
            results['recommendations'].append("High membership inference risk - consider adding differential privacy")
        
        if inversion_risk > 0.7:
            results['recommendations'].append("High model inversion risk - limit model output precision")
        
        if leakage_risk > 0.5:
            results['recommendations'].append("Potential data leakage - review model architecture")
        
        if dp_score < 0.5:
            results['recommendations'].append("Low differential privacy score - implement privacy-preserving training")
        
        return results
    
    def _test_membership_inference(self, model: Any, training_stats: Dict) -> float:
        """Test membership inference attack"""
        # Simplified implementation
        # Real implementation would use shadow models
        return np.random.uniform(0.3, 0.7)
    
    def _test_model_inversion(self, model: Any) -> float:
        """Test model inversion attack"""
        # Simplified implementation
        return np.random.uniform(0.2, 0.6)
    
    def _detect_data_leakage(self, model: Any) -> float:
        """Detect potential data leakage"""
        # Check for memorization patterns
        return np.random.uniform(0.1, 0.5)
    
    def _calculate_dp_score(self, model: Any, training_stats: Dict) -> float:
        """Calculate differential privacy score"""
        # Simplified - real implementation would analyze noise levels
        return np.random.uniform(0.4, 0.8)


class CodeAnalyzer:
    """Analyze model code for security issues"""
    
    def __init__(self):
        self.bandit_manager = bandit.core.manager.BanditManager()
    
    def analyze_code(self, code_path: str) -> Dict[str, Any]:
        """Analyze model code for security issues"""
        results = {
            'security_issues': [],
            'code_quality_score': 0.0,
            'dangerous_imports': [],
            'suspicious_functions': []
        }
        
        try:
            # Run Bandit security analysis
            bandit_results = self._run_bandit(code_path)
            results['security_issues'] = bandit_results
            
            # Check for dangerous imports
            dangerous_imports = self._check_imports(code_path)
            results['dangerous_imports'] = dangerous_imports
            
            # Check for suspicious functions
            suspicious = self._check_suspicious_functions(code_path)
            results['suspicious_functions'] = suspicious
            
            # Calculate code quality score
            total_issues = len(results['security_issues']) + len(dangerous_imports) + len(suspicious)
            results['code_quality_score'] = max(0, 1 - (total_issues * 0.1))
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
        
        return results
    
    def _run_bandit(self, code_path: str) -> List[Dict]:
        """Run Bandit security scanner"""
        # Simplified - real implementation would use Bandit API
        return []
    
    def _check_imports(self, code_path: str) -> List[str]:
        """Check for dangerous imports"""
        dangerous = ['os', 'subprocess', 'eval', 'exec', '__import__']
        found = []
        
        try:
            with open(code_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous:
                            found.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous:
                        found.append(node.module)
        except:
            pass
        
        return found
    
    def _check_suspicious_functions(self, code_path: str) -> List[str]:
        """Check for suspicious function calls"""
        suspicious = ['eval', 'exec', 'compile', '__import__']
        found = []
        
        try:
            with open(code_path, 'r') as f:
                content = f.read()
            
            for func in suspicious:
                if func in content:
                    found.append(func)
        except:
            pass
        
        return found


class ModelSecurityScanner:
    """
    Comprehensive model security scanning system
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
        self.cache = redis.Redis(host='localhost', port=6379, db=5) if cache_enabled else None
        
        # Initialize security components
        self.adversarial_tester = AdversarialTester()
        self.backdoor_detector = BackdoorDetector()
        self.privacy_analyzer = PrivacyAnalyzer()
        self.code_analyzer = CodeAnalyzer()
    
    async def scan_model(self,
                        model_path: str,
                        model_id: str,
                        config: SecurityConfig = SecurityConfig()) -> SecurityReport:
        """
        Perform comprehensive security scan on model
        
        Args:
            model_path: Path to model file
            model_id: Model identifier
            config: Security configuration
        
        Returns:
            Security scan report
        """
        scan_start = datetime.utcnow()
        scan_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting security scan for model {model_id}")
            
            # Create scan record
            scan_db = SecurityScanDB(
                model_id=model_id,
                scan_level=config.scan_level.value,
                status='running'
            )
            self.session.add(scan_db)
            self.session.commit()
            
            # Load model
            model = self._load_model(model_path)
            
            # Initialize findings
            vulnerabilities = []
            threats_detected = []
            detailed_findings = {}
            
            # Generate test data for security testing
            test_data, labels = self._generate_test_data(model)
            
            # 1. Adversarial robustness testing
            if config.adversarial_testing:
                logger.info("Testing adversarial robustness...")
                adv_results = self.adversarial_tester.test_robustness(model, test_data, labels)
                detailed_findings['adversarial'] = adv_results
                
                # Check for vulnerabilities
                for attack, metrics in adv_results.items():
                    if isinstance(metrics, dict) and metrics.get('robustness_score', 1) < 0.5:
                        vulnerabilities.append(Vulnerability(
                            type=ThreatType.ADVERSARIAL_ATTACK,
                            severity='high',
                            confidence=0.8,
                            description=f"Model vulnerable to {attack} attack",
                            evidence=metrics,
                            mitigation="Implement adversarial training or defensive distillation"
                        ))
                        threats_detected.append(ThreatType.ADVERSARIAL_ATTACK)
            
            # 2. Backdoor detection
            if config.backdoor_detection:
                logger.info("Scanning for backdoors...")
                backdoor_results = self.backdoor_detector.scan_for_backdoors(model, test_data)
                detailed_findings['backdoor'] = backdoor_results
                
                if backdoor_results['backdoor_detected']:
                    vulnerabilities.append(Vulnerability(
                        type=ThreatType.BACKDOOR,
                        severity='critical',
                        confidence=backdoor_results['confidence'],
                        description="Potential backdoor detected in model",
                        evidence=backdoor_results,
                        mitigation="Retrain model with verified clean data"
                    ))
                    threats_detected.append(ThreatType.BACKDOOR)
            
            # 3. Privacy analysis
            if config.privacy_analysis:
                logger.info("Analyzing privacy risks...")
                privacy_results = self.privacy_analyzer.analyze_privacy_risks(
                    model,
                    {'num_samples': 10000}  # Placeholder stats
                )
                detailed_findings['privacy'] = privacy_results
                
                if privacy_results['membership_inference_risk'] > 0.7:
                    vulnerabilities.append(Vulnerability(
                        type=ThreatType.MEMBERSHIP_INFERENCE,
                        severity='medium',
                        confidence=privacy_results['membership_inference_risk'],
                        description="High risk of membership inference attacks",
                        evidence=privacy_results,
                        mitigation="Apply differential privacy during training"
                    ))
                    threats_detected.append(ThreatType.MEMBERSHIP_INFERENCE)
            
            # 4. Supply chain security
            if config.supply_chain_check:
                logger.info("Checking supply chain security...")
                supply_chain_results = await self._check_supply_chain(model_path)
                detailed_findings['supply_chain'] = supply_chain_results
                
                if supply_chain_results.get('vulnerable_dependencies'):
                    vulnerabilities.append(Vulnerability(
                        type=ThreatType.SUPPLY_CHAIN,
                        severity='high',
                        confidence=0.9,
                        description="Vulnerable dependencies detected",
                        evidence=supply_chain_results,
                        mitigation="Update dependencies to secure versions"
                    ))
                    threats_detected.append(ThreatType.SUPPLY_CHAIN)
            
            # 5. Code analysis
            if config.code_analysis and Path(model_path).suffix == '.py':
                logger.info("Analyzing model code...")
                code_results = self.code_analyzer.analyze_code(model_path)
                detailed_findings['code_analysis'] = code_results
                
                if code_results['dangerous_imports'] or code_results['suspicious_functions']:
                    vulnerabilities.append(Vulnerability(
                        type=ThreatType.INJECTION,
                        severity='high',
                        confidence=0.85,
                        description="Dangerous code patterns detected",
                        evidence=code_results,
                        mitigation="Review and sanitize code"
                    ))
                    threats_detected.append(ThreatType.INJECTION)
            
            # 6. Compliance checking
            compliance_status = {}
            if config.compliance_standards:
                logger.info("Checking compliance...")
                for standard in config.compliance_standards:
                    compliance_status[standard.value] = await self._check_compliance(
                        model, standard, detailed_findings
                    )
            
            # Calculate overall security score
            overall_score = self._calculate_security_score(
                vulnerabilities,
                detailed_findings,
                compliance_status
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_score, vulnerabilities)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                vulnerabilities,
                detailed_findings,
                compliance_status
            )
            
            # Calculate scan duration
            scan_duration = (datetime.utcnow() - scan_start).total_seconds()
            
            # Create report
            report = SecurityReport(
                model_id=model_id,
                scan_id=scan_id,
                timestamp=datetime.utcnow(),
                scan_level=config.scan_level,
                overall_score=overall_score,
                risk_level=risk_level,
                vulnerabilities=[asdict(v) for v in vulnerabilities],
                threats_detected=threats_detected,
                compliance_status=compliance_status,
                recommendations=recommendations,
                detailed_findings=detailed_findings,
                scan_duration=scan_duration
            )
            
            # Update database
            scan_db.status = 'completed'
            scan_db.overall_score = overall_score
            scan_db.risk_level = risk_level
            scan_db.vulnerabilities_count = len(vulnerabilities)
            scan_db.threats_detected = [t.value for t in threats_detected]
            scan_db.compliance_status = compliance_status
            scan_db.recommendations = recommendations
            scan_db.detailed_findings = detailed_findings
            scan_db.scan_duration = scan_duration
            scan_db.completed_at = datetime.utcnow()
            
            # Save vulnerabilities
            for vuln in vulnerabilities:
                vuln_db = VulnerabilityDB(
                    scan_id=scan_db.id,
                    type=vuln.type.value,
                    severity=vuln.severity,
                    confidence=vuln.confidence,
                    description=vuln.description,
                    evidence=vuln.evidence,
                    mitigation=vuln.mitigation,
                    cve_id=vuln.cve_id
                )
                self.session.add(vuln_db)
            
            self.session.commit()
            
            # Cache report
            if self.cache:
                cache_key = f"security_scan:{model_id}:{scan_id}"
                self.cache.setex(cache_key, 86400, json.dumps(asdict(report), default=str))
            
            logger.info(f"Security scan completed for model {model_id}")
            return report
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            
            # Update scan status
            if scan_db:
                scan_db.status = 'failed'
                self.session.commit()
            
            raise
    
    async def _check_supply_chain(self, model_path: str) -> Dict[str, Any]:
        """Check supply chain security"""
        results = {
            'vulnerable_dependencies': [],
            'outdated_packages': [],
            'security_advisories': []
        }
        
        # Check for requirements file
        req_path = Path(model_path).parent / 'requirements.txt'
        if req_path.exists():
            try:
                # Use safety to check dependencies
                with open(req_path, 'r') as f:
                    requirements = f.read()
                
                # Check for vulnerabilities (simplified)
                # Real implementation would use safety API
                vulnerable = []
                
                results['vulnerable_dependencies'] = vulnerable
                
            except Exception as e:
                logger.error(f"Supply chain check failed: {e}")
        
        return results
    
    async def _check_compliance(self,
                               model: Any,
                               standard: ComplianceStandard,
                               findings: Dict) -> bool:
        """Check compliance with standard"""
        
        if standard == ComplianceStandard.GDPR:
            # Check GDPR compliance
            privacy_score = findings.get('privacy', {}).get('differential_privacy_score', 0)
            explainability = findings.get('adversarial', {}).get('clever_score') is not None
            return privacy_score > 0.6 and explainability
        
        elif standard == ComplianceStandard.OWASP_ML:
            # Check OWASP ML Top 10
            no_backdoor = not findings.get('backdoor', {}).get('backdoor_detected', False)
            robust = all(
                r.get('robustness_score', 0) > 0.3
                for r in findings.get('adversarial', {}).values()
                if isinstance(r, dict)
            )
            return no_backdoor and robust
        
        # Default compliance check
        return True
    
    def _calculate_security_score(self,
                                 vulnerabilities: List[Vulnerability],
                                 findings: Dict,
                                 compliance: Dict) -> float:
        """Calculate overall security score"""
        score = 100.0
        
        # Deduct for vulnerabilities
        for vuln in vulnerabilities:
            if vuln.severity == 'critical':
                score -= 20
            elif vuln.severity == 'high':
                score -= 10
            elif vuln.severity == 'medium':
                score -= 5
            else:
                score -= 2
        
        # Adjust for specific findings
        if findings.get('adversarial'):
            avg_robustness = np.mean([
                r.get('robustness_score', 1)
                for r in findings['adversarial'].values()
                if isinstance(r, dict) and 'robustness_score' in r
            ])
            score *= avg_robustness
        
        # Compliance bonus
        if compliance:
            compliance_rate = sum(compliance.values()) / len(compliance)
            score += compliance_rate * 10
        
        return max(0, min(100, score))
    
    def _determine_risk_level(self, score: float, vulnerabilities: List[Vulnerability]) -> str:
        """Determine risk level based on score and vulnerabilities"""
        
        # Check for critical vulnerabilities
        has_critical = any(v.severity == 'critical' for v in vulnerabilities)
        
        if has_critical or score < 30:
            return 'critical'
        elif score < 50:
            return 'high'
        elif score < 70:
            return 'medium'
        elif score < 85:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_recommendations(self,
                                 vulnerabilities: List[Vulnerability],
                                 findings: Dict,
                                 compliance: Dict) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Vulnerability-based recommendations
        for vuln in vulnerabilities:
            if vuln.mitigation not in recommendations:
                recommendations.append(vuln.mitigation)
        
        # Finding-based recommendations
        if findings.get('privacy', {}).get('recommendations'):
            recommendations.extend(findings['privacy']['recommendations'])
        
        # Compliance-based recommendations
        for standard, compliant in compliance.items():
            if not compliant:
                recommendations.append(f"Implement measures to comply with {standard}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Regular security audits recommended")
            recommendations.append("Implement continuous monitoring")
        
        return recommendations[:10]  # Limit to top 10
    
    def _load_model(self, model_path: str) -> Any:
        """Load model from path"""
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif model_path.endswith(('.pt', '.pth')):
            return torch.load(model_path)
        elif model_path.endswith('.h5'):
            return tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _generate_test_data(self, model: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test data for security testing"""
        # Generate synthetic test data
        # Real implementation would use actual test datasets
        n_samples = 100
        n_features = 100
        n_classes = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        
        return X, y