"""
Deep learning model for comprehensive risk assessment of contract clauses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from transformers import (
    AutoModel, AutoTokenizer, 
    RobertaForSequenceClassification, RobertaTokenizer
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
import joblib
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for a clause."""
    clause_text: str
    overall_risk_score: float  # 0-1 scale
    risk_category: str  # low, medium, high, critical
    risk_factors: Dict[str, float]
    industry_benchmark: float
    temporal_trend: str  # increasing, stable, decreasing
    mitigation_recommendations: List[str]
    confidence: float
    explanation: str


@dataclass
class RiskMitigation:
    """Risk mitigation strategy."""
    risk_type: str
    severity: str
    mitigation_steps: List[str]
    alternative_approaches: List[str]
    residual_risk: float
    implementation_timeline: str
    cost_estimate: str


class DeepRiskNet(nn.Module):
    """Deep neural network for risk assessment."""
    
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = None):
        super(DeepRiskNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Multi-task output heads
        self.shared_layers = nn.Sequential(*layers)
        
        # Risk score prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Risk category classification head
        self.category_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 risk categories
        )
        
        # Risk factors prediction head (multiple risk dimensions)
        self.factors_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 risk factor dimensions
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        shared_features = self.shared_layers(x)
        
        risk_score = self.risk_head(shared_features)
        risk_category = self.category_head(shared_features)
        risk_factors = self.factors_head(shared_features)
        
        return risk_score, risk_category, risk_factors


class TemporalRiskAnalyzer:
    """Analyzes how clause risks change over time."""
    
    def __init__(self):
        self.historical_data = defaultdict(list)
        self.trend_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.scaler = StandardScaler()
    
    def add_historical_point(
        self,
        clause_type: str,
        timestamp: datetime,
        risk_score: float,
        context: Optional[Dict] = None
    ):
        """Add historical risk data point."""
        self.historical_data[clause_type].append({
            'timestamp': timestamp,
            'risk_score': risk_score,
            'context': context or {}
        })
    
    def analyze_trend(self, clause_type: str, lookback_days: int = 365) -> str:
        """Analyze risk trend for a clause type."""
        if clause_type not in self.historical_data:
            return "insufficient_data"
        
        data = self.historical_data[clause_type]
        if len(data) < 10:
            return "insufficient_data"
        
        # Filter to lookback period
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_data = [d for d in data if d['timestamp'] > cutoff]
        
        if len(recent_data) < 5:
            return "insufficient_data"
        
        # Calculate trend
        scores = [d['risk_score'] for d in recent_data]
        timestamps = [(d['timestamp'] - cutoff).days for d in recent_data]
        
        # Simple linear regression coefficient
        x_mean = np.mean(timestamps)
        y_mean = np.mean(scores)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, scores))
        denominator = sum((x - x_mean) ** 2 for x in timestamps)
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Determine trend based on slope
        if slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"
    
    def predict_future_risk(
        self,
        clause_type: str,
        days_ahead: int = 30
    ) -> Optional[float]:
        """Predict future risk score."""
        if clause_type not in self.historical_data:
            return None
        
        data = self.historical_data[clause_type]
        if len(data) < 20:
            return None
        
        # Prepare training data
        X = []
        y = []
        
        for i, point in enumerate(data):
            features = [
                i,  # Time index
                point['timestamp'].month,  # Seasonality
                point['timestamp'].weekday(),  # Day of week
                len(point['context']),  # Context complexity
            ]
            X.append(features)
            y.append(point['risk_score'])
        
        # Train model
        X_scaled = self.scaler.fit_transform(X)
        self.trend_model.fit(X_scaled, y)
        
        # Predict future
        future_date = datetime.now() + timedelta(days=days_ahead)
        future_features = [
            len(data) + days_ahead,
            future_date.month,
            future_date.weekday(),
            np.mean([len(d['context']) for d in data])
        ]
        
        future_scaled = self.scaler.transform([future_features])
        prediction = self.trend_model.predict(future_scaled)[0]
        
        return float(np.clip(prediction, 0, 1))


class RiskAssessmentEngine:
    """Comprehensive risk assessment engine using deep learning."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize risk assessment engine."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transformer model for embeddings
        self._initialize_transformer()
        
        # Initialize deep risk network
        self.risk_model = DeepRiskNet().to(self.device)
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Industry benchmarks
        self.industry_benchmarks = self._load_industry_benchmarks()
        
        # Temporal analyzer
        self.temporal_analyzer = TemporalRiskAnalyzer()
        
        # Anomaly detector for unusual clauses
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Risk mitigation database
        self.mitigation_strategies = self._load_mitigation_strategies()
    
    def _initialize_transformer(self):
        """Initialize transformer model for text embeddings."""
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.transformer = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=4  # Risk categories
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            # Fallback to smaller model
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.tokenizer = None
            self.transformer = None
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load industry-specific risk benchmarks."""
        return {
            "technology": {
                "arbitration": 0.7,
                "liability_limitation": 0.6,
                "ip_assignment": 0.8,
                "confidentiality": 0.5,
                "termination": 0.4
            },
            "healthcare": {
                "arbitration": 0.8,
                "liability_limitation": 0.9,
                "privacy": 0.9,
                "compliance": 0.8,
                "indemnification": 0.7
            },
            "finance": {
                "arbitration": 0.6,
                "regulatory": 0.9,
                "liability_limitation": 0.7,
                "audit_rights": 0.5,
                "data_security": 0.8
            },
            "retail": {
                "arbitration": 0.5,
                "warranty": 0.6,
                "return_policy": 0.4,
                "liability_limitation": 0.5,
                "payment_terms": 0.3
            }
        }
    
    def _load_mitigation_strategies(self) -> Dict[str, List[str]]:
        """Load risk mitigation strategies database."""
        return {
            "high_arbitration": [
                "Negotiate for opt-out provision within 30 days",
                "Request carve-outs for small claims and injunctive relief",
                "Propose mediation as first step before arbitration",
                "Ensure arbitration costs are shared or shifted to company",
                "Include class action waiver exceptions for certain claims"
            ],
            "unfair_liability": [
                "Request mutual limitation of liability",
                "Negotiate for exceptions: gross negligence, willful misconduct",
                "Propose liability cap based on fees paid, not fixed amount",
                "Include consequential damages for breach of confidentiality",
                "Request insurance requirements from the other party"
            ],
            "one_sided_termination": [
                "Negotiate for mutual termination rights",
                "Request notice period and cure provisions",
                "Include termination for convenience with compensation",
                "Add data portability and transition assistance",
                "Propose graduated penalties instead of immediate termination"
            ],
            "excessive_indemnification": [
                "Limit indemnification to third-party claims only",
                "Exclude indemnification for other party's negligence",
                "Cap indemnification at contract value or insurance limits",
                "Request mutual indemnification provisions",
                "Include notice and cooperation requirements"
            ]
        }
    
    async def assess_risk(
        self,
        clause: str,
        contract_context: Optional[str] = None,
        industry: str = "general",
        historical_context: Optional[Dict] = None
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment on a clause.
        
        Args:
            clause: Clause text to assess
            contract_context: Additional contract context
            industry: Industry vertical for benchmarking
            historical_context: Historical data for temporal analysis
            
        Returns:
            Comprehensive risk assessment
        """
        # Generate embeddings
        embeddings = self._generate_embeddings(clause, contract_context)
        
        # Deep learning risk assessment
        with torch.no_grad():
            risk_score, category_logits, risk_factors = self.risk_model(embeddings)
        
        # Convert outputs
        overall_risk = float(risk_score.cpu().numpy()[0])
        category_probs = F.softmax(category_logits, dim=1).cpu().numpy()[0]
        factors = risk_factors.cpu().numpy()[0]
        
        # Determine risk category
        risk_category = self._determine_risk_category(overall_risk, category_probs)
        
        # Extract detailed risk factors
        risk_factor_dict = self._extract_risk_factors(factors, clause)
        
        # Get industry benchmark
        benchmark = self._get_industry_benchmark(clause, industry)
        
        # Temporal trend analysis
        clause_type = self._identify_clause_type(clause)
        temporal_trend = self.temporal_analyzer.analyze_trend(clause_type)
        
        # Generate mitigation recommendations
        mitigations = self._generate_mitigations(
            risk_category, risk_factor_dict, clause_type
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(embeddings, overall_risk)
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_risk, risk_factor_dict, benchmark, temporal_trend
        )
        
        # Record for temporal analysis
        if historical_context:
            self.temporal_analyzer.add_historical_point(
                clause_type,
                datetime.now(),
                overall_risk,
                historical_context
            )
        
        return RiskAssessment(
            clause_text=clause,
            overall_risk_score=overall_risk,
            risk_category=risk_category,
            risk_factors=risk_factor_dict,
            industry_benchmark=benchmark,
            temporal_trend=temporal_trend,
            mitigation_recommendations=mitigations,
            confidence=confidence,
            explanation=explanation
        )
    
    def _generate_embeddings(
        self,
        clause: str,
        context: Optional[str] = None
    ) -> torch.Tensor:
        """Generate embeddings for clause text."""
        if self.tokenizer and self.transformer:
            # Use RoBERTa
            text = clause if not context else f"{context} [SEP] {clause}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.transformer.roberta(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            # Use sentence transformer
            text = clause if not context else f"{context} {clause}"
            embeddings_np = self.sentence_transformer.encode([text])
            embeddings = torch.tensor(embeddings_np).to(self.device)
        
        return embeddings
    
    def _determine_risk_category(
        self,
        risk_score: float,
        category_probs: np.ndarray
    ) -> str:
        """Determine risk category from score and probabilities."""
        categories = ["low", "medium", "high", "critical"]
        
        # Use both risk score and category probabilities
        if risk_score > 0.8 or category_probs[3] > 0.5:
            return "critical"
        elif risk_score > 0.6 or category_probs[2] > 0.5:
            return "high"
        elif risk_score > 0.3 or category_probs[1] > 0.5:
            return "medium"
        else:
            return "low"
    
    def _extract_risk_factors(
        self,
        factor_scores: np.ndarray,
        clause: str
    ) -> Dict[str, float]:
        """Extract detailed risk factors from model output."""
        factor_names = [
            "enforceability_risk",
            "fairness_risk",
            "ambiguity_risk",
            "compliance_risk",
            "financial_risk",
            "operational_risk",
            "reputational_risk",
            "legal_precedent_risk",
            "negotiability_risk",
            "hidden_terms_risk"
        ]
        
        risk_factors = {}
        for name, score in zip(factor_names, factor_scores):
            risk_factors[name] = float(score)
        
        # Add clause-specific risks
        clause_lower = clause.lower()
        
        if "perpetual" in clause_lower or "forever" in clause_lower:
            risk_factors["temporal_risk"] = 0.8
        
        if "worldwide" in clause_lower or "global" in clause_lower:
            risk_factors["geographic_risk"] = 0.7
        
        if "sole discretion" in clause_lower or "absolute discretion" in clause_lower:
            risk_factors["discretionary_risk"] = 0.9
        
        return risk_factors
    
    def _get_industry_benchmark(self, clause: str, industry: str) -> float:
        """Get industry-specific risk benchmark."""
        clause_type = self._identify_clause_type(clause)
        
        if industry in self.industry_benchmarks:
            benchmarks = self.industry_benchmarks[industry]
            if clause_type in benchmarks:
                return benchmarks[clause_type]
        
        # Default benchmark
        return 0.5
    
    def _identify_clause_type(self, clause: str) -> str:
        """Identify the type of clause."""
        clause_lower = clause.lower()
        
        patterns = {
            "arbitration": ["arbitration", "arbitrator", "binding", "dispute resolution"],
            "liability_limitation": ["limitation of liability", "damages", "consequential"],
            "termination": ["termination", "terminate", "cancel", "end"],
            "confidentiality": ["confidential", "non-disclosure", "proprietary"],
            "indemnification": ["indemnify", "indemnification", "hold harmless"],
            "ip_assignment": ["intellectual property", "copyright", "patent", "assignment"],
            "warranty": ["warranty", "guarantee", "representation"],
            "payment_terms": ["payment", "invoice", "billing", "fees"]
        }
        
        for clause_type, keywords in patterns.items():
            if any(keyword in clause_lower for keyword in keywords):
                return clause_type
        
        return "general"
    
    def _generate_mitigations(
        self,
        risk_category: str,
        risk_factors: Dict[str, float],
        clause_type: str
    ) -> List[str]:
        """Generate risk mitigation recommendations."""
        mitigations = []
        
        # High-level strategy based on risk category
        if risk_category in ["high", "critical"]:
            mitigations.append("Prioritize renegotiation of this clause")
            mitigations.append("Consider seeking legal counsel review")
        
        # Specific mitigations based on risk factors
        high_risk_factors = [
            (factor, score) for factor, score in risk_factors.items()
            if score > 0.7
        ]
        
        for factor, score in sorted(high_risk_factors, key=lambda x: x[1], reverse=True)[:3]:
            if factor == "enforceability_risk":
                mitigations.append("Challenge enforceability based on jurisdiction")
            elif factor == "fairness_risk":
                mitigations.append("Propose more balanced alternative language")
            elif factor == "ambiguity_risk":
                mitigations.append("Request clarification and specific definitions")
            elif factor == "compliance_risk":
                mitigations.append("Ensure regulatory compliance review")
            elif factor == "financial_risk":
                mitigations.append("Negotiate financial caps or insurance requirements")
        
        # Add clause-specific mitigations
        if clause_type in self.mitigation_strategies:
            if risk_category in ["high", "critical"]:
                key = f"high_{clause_type}"
                if key in self.mitigation_strategies:
                    mitigations.extend(self.mitigation_strategies[key][:2])
        
        return mitigations[:5]  # Return top 5 mitigations
    
    def _calculate_confidence(
        self,
        embeddings: torch.Tensor,
        risk_score: float
    ) -> float:
        """Calculate confidence in risk assessment."""
        # Use anomaly detection to assess confidence
        try:
            embeddings_np = embeddings.cpu().numpy().reshape(1, -1)
            anomaly_score = self.anomaly_detector.decision_function(embeddings_np)[0]
            
            # Convert anomaly score to confidence (inverted and normalized)
            confidence = 1.0 / (1.0 + np.exp(-anomaly_score))
            
            # Adjust based on risk score extremity
            if risk_score < 0.1 or risk_score > 0.9:
                confidence *= 0.9  # Slightly lower confidence for extreme scores
            
            return float(confidence)
        except:
            # Default confidence if anomaly detection not trained
            return 0.75
    
    def _generate_explanation(
        self,
        risk_score: float,
        risk_factors: Dict[str, float],
        benchmark: float,
        trend: str
    ) -> str:
        """Generate human-readable risk explanation."""
        explanation_parts = []
        
        # Overall risk assessment
        if risk_score > 0.7:
            explanation_parts.append(
                "This clause presents significant legal and business risks."
            )
        elif risk_score > 0.4:
            explanation_parts.append(
                "This clause contains moderate risks requiring careful consideration."
            )
        else:
            explanation_parts.append(
                "This clause presents relatively low risk."
            )
        
        # Comparison to benchmark
        if risk_score > benchmark + 0.2:
            explanation_parts.append(
                f"Risk level is notably higher than industry standard ({benchmark:.2f})."
            )
        elif risk_score < benchmark - 0.2:
            explanation_parts.append(
                f"Risk level is lower than typical industry standard ({benchmark:.2f})."
            )
        
        # Top risk factors
        top_risks = sorted(
            risk_factors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        if top_risks:
            risk_description = ", ".join([
                f"{risk.replace('_', ' ').title()}: {score:.2f}"
                for risk, score in top_risks
            ])
            explanation_parts.append(f"Primary concerns: {risk_description}")
        
        # Temporal trend
        if trend == "increasing":
            explanation_parts.append(
                "Historical data shows increasing risk trend for similar clauses."
            )
        elif trend == "decreasing":
            explanation_parts.append(
                "Risk trend is decreasing based on recent precedents."
            )
        
        return " ".join(explanation_parts)
    
    def generate_risk_report(
        self,
        contract_text: str,
        industry: str = "general"
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report for entire contract."""
        clauses = self._extract_clauses(contract_text)
        assessments = []
        
        for clause in clauses:
            assessment = asyncio.run(self.assess_risk(clause, contract_text[:500], industry))
            assessments.append(assessment)
        
        # Aggregate risk metrics
        overall_risk = np.mean([a.overall_risk_score for a in assessments])
        critical_clauses = [a for a in assessments if a.risk_category == "critical"]
        high_risk_clauses = [a for a in assessments if a.risk_category == "high"]
        
        # Risk distribution
        risk_distribution = {
            "critical": len(critical_clauses),
            "high": len(high_risk_clauses),
            "medium": len([a for a in assessments if a.risk_category == "medium"]),
            "low": len([a for a in assessments if a.risk_category == "low"])
        }
        
        # Top risk factors across contract
        all_factors = defaultdict(list)
        for assessment in assessments:
            for factor, score in assessment.risk_factors.items():
                all_factors[factor].append(score)
        
        avg_risk_factors = {
            factor: np.mean(scores)
            for factor, scores in all_factors.items()
        }
        
        return {
            "overall_risk_score": float(overall_risk),
            "risk_distribution": risk_distribution,
            "critical_clauses": [
                {
                    "clause": a.clause_text[:200] + "...",
                    "risk_score": a.overall_risk_score,
                    "primary_concern": max(a.risk_factors.items(), key=lambda x: x[1])[0]
                }
                for a in critical_clauses
            ],
            "top_risk_factors": dict(sorted(
                avg_risk_factors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "recommendations": self._generate_overall_recommendations(
                overall_risk, risk_distribution, critical_clauses
            )
        }
    
    def _extract_clauses(self, contract_text: str) -> List[str]:
        """Extract individual clauses from contract."""
        # Simple extraction by paragraph
        paragraphs = contract_text.split('\n\n')
        
        # Filter to meaningful clauses
        clauses = []
        for para in paragraphs:
            if len(para.split()) > 20:  # Minimum word count
                clauses.append(para)
        
        return clauses
    
    def _generate_overall_recommendations(
        self,
        overall_risk: float,
        risk_distribution: Dict[str, int],
        critical_clauses: List[RiskAssessment]
    ) -> List[str]:
        """Generate overall contract recommendations."""
        recommendations = []
        
        if overall_risk > 0.7:
            recommendations.append(
                "This contract presents significant risks. Legal review strongly recommended."
            )
        
        if risk_distribution["critical"] > 2:
            recommendations.append(
                f"Address {risk_distribution['critical']} critical risk clauses before signing."
            )
        
        if risk_distribution["high"] > 5:
            recommendations.append(
                "Multiple high-risk provisions require negotiation or mitigation."
            )
        
        if critical_clauses:
            recommendations.append(
                f"Priority: Renegotiate arbitration and liability clauses."
            )
        
        if overall_risk < 0.3:
            recommendations.append(
                "Contract terms are generally favorable with manageable risks."
            )
        
        return recommendations
    
    def train(
        self,
        training_data: List[Tuple[str, float, str, Dict[str, float]]],
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Train the deep risk assessment model.
        
        Args:
            training_data: List of (clause, risk_score, category, risk_factors) tuples
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare dataset
        dataset = RiskDataset(training_data, self)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.risk_model.parameters(), lr=1e-4)
        
        # Loss functions
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                embeddings, risk_scores, categories, risk_factors = batch
                
                embeddings = embeddings.to(self.device)
                risk_scores = risk_scores.to(self.device)
                categories = categories.to(self.device)
                risk_factors = risk_factors.to(self.device)
                
                # Forward pass
                pred_scores, pred_categories, pred_factors = self.risk_model(embeddings)
                
                # Calculate losses
                score_loss = mse_loss(pred_scores.squeeze(), risk_scores)
                category_loss = ce_loss(pred_categories, categories)
                factor_loss = mse_loss(pred_factors, risk_factors)
                
                # Combined loss
                loss = score_loss + category_loss + factor_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save(self.risk_model.state_dict(), f"{path}/risk_model.pth")
        joblib.dump(self.anomaly_detector, f"{path}/anomaly_detector.pkl")
    
    def load_model(self, path: str):
        """Load trained model."""
        try:
            self.risk_model.load_state_dict(torch.load(f"{path}/risk_model.pth"))
            self.anomaly_detector = joblib.load(f"{path}/anomaly_detector.pkl")
        except:
            logger.warning("Could not load pre-trained risk model")


class RiskDataset(Dataset):
    """Dataset for training risk assessment model."""
    
    def __init__(self, data: List[Tuple], engine: RiskAssessmentEngine):
        self.data = data
        self.engine = engine
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        clause, risk_score, category, risk_factors = self.data[idx]
        
        # Generate embeddings
        embeddings = self.engine._generate_embeddings(clause)
        
        # Convert category to index
        category_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        category_idx = category_map.get(category, 1)
        
        # Convert risk factors to tensor
        factor_values = list(risk_factors.values())[:10]  # Ensure 10 dimensions
        while len(factor_values) < 10:
            factor_values.append(0.0)
        
        return (
            embeddings.squeeze(),
            torch.tensor(risk_score, dtype=torch.float32),
            torch.tensor(category_idx, dtype=torch.long),
            torch.tensor(factor_values, dtype=torch.float32)
        )