"""
Advanced fraud detection system with machine learning capabilities.

This module provides comprehensive fraud detection including:
- Real-time transaction analysis
- Machine learning-based risk scoring
- Behavioral pattern analysis
- Adaptive rule engine
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)


class AdvancedFraudDetector:
    """Advanced fraud detection using machine learning"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.isolation_forest = None
        self.scaler = None
        self.feature_names = [
            'amount', 'hour_of_day', 'day_of_week', 'user_age_days',
            'transaction_count_1h', 'transaction_count_24h', 'avg_amount_30d',
            'distinct_merchants_30d', 'failed_attempts_24h', 'location_risk'
        ]
        
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize new one"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load pre-trained model
                model_data = joblib.load(self.model_path)
                self.isolation_forest = model_data.get('model')
                self.scaler = model_data.get('scaler')
                logger.info("Loaded pre-trained fraud detection model")
            else:
                # Initialize new model
                self.isolation_forest = IsolationForest(
                    contamination=0.1,  # Expected fraud rate
                    random_state=42,
                    n_estimators=100
                )
                self.scaler = StandardScaler()
                logger.info("Initialized new fraud detection model")
                
        except Exception as e:
            logger.error(f"Failed to load/initialize model: {e}")
            # Fallback to basic model
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
    
    async def analyze_transaction(
        self,
        transaction_data: Dict[str, Any],
        user_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze transaction for fraud using ML model"""
        try:
            # Extract features
            features = self._extract_features(transaction_data, user_history)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
            
            # Convert anomaly score to risk score (0-100)
            risk_score = max(0, min(100, int((1 - anomaly_score) * 50)))
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = "critical"
                recommended_action = "block"
            elif risk_score >= 60:
                risk_level = "high"
                recommended_action = "review"
            elif risk_score >= 40:
                risk_level = "medium"
                recommended_action = "flag"
            elif risk_score >= 20:
                risk_level = "low"
                recommended_action = "monitor"
            else:
                risk_level = "minimal"
                recommended_action = "approve"
            
            # Additional rule-based analysis
            rule_analysis = await self._apply_business_rules(transaction_data, user_history)
            
            # Combine ML and rule-based scores
            combined_score = max(risk_score, rule_analysis["risk_score"])
            
            if rule_analysis["recommended_action"] == "block":
                recommended_action = "block"
            elif rule_analysis["recommended_action"] == "review" and recommended_action != "block":
                recommended_action = "review"
            
            return {
                "risk_score": combined_score,
                "risk_level": risk_level,
                "recommended_action": recommended_action,
                "ml_analysis": {
                    "anomaly_score": float(anomaly_score),
                    "is_anomaly": bool(is_anomaly),
                    "features_used": dict(zip(self.feature_names, features))
                },
                "rule_analysis": rule_analysis,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "model_version": "1.0"
            }
            
        except Exception as e:
            logger.error(f"ML fraud analysis failed: {e}")
            # Fallback to rule-based analysis only
            rule_analysis = await self._apply_business_rules(transaction_data, user_history)
            return {
                "risk_score": rule_analysis["risk_score"],
                "risk_level": "unknown",
                "recommended_action": rule_analysis["recommended_action"],
                "error": str(e),
                "fallback_mode": True
            }
    
    def _extract_features(
        self,
        transaction_data: Dict[str, Any],
        user_history: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract features for ML model"""
        try:
            # Transaction amount
            amount = float(transaction_data.get("amount", 0))
            
            # Time-based features
            timestamp = datetime.fromisoformat(
                transaction_data.get("timestamp", datetime.utcnow().isoformat())
            )
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # User age (days since first transaction)
            if user_history:
                first_tx = min(user_history, key=lambda x: x.get("timestamp", ""))
                first_tx_date = datetime.fromisoformat(first_tx.get("timestamp", timestamp.isoformat()))
                user_age_days = (timestamp - first_tx_date).days
            else:
                user_age_days = 0
            
            # Transaction velocity features
            one_hour_ago = timestamp - timedelta(hours=1)
            twenty_four_hours_ago = timestamp - timedelta(hours=24)
            thirty_days_ago = timestamp - timedelta(days=30)
            
            tx_count_1h = len([
                tx for tx in user_history
                if datetime.fromisoformat(tx.get("timestamp", "")) >= one_hour_ago
            ])
            
            tx_count_24h = len([
                tx for tx in user_history
                if datetime.fromisoformat(tx.get("timestamp", "")) >= twenty_four_hours_ago
            ])
            
            # Historical patterns
            recent_transactions = [
                tx for tx in user_history
                if datetime.fromisoformat(tx.get("timestamp", "")) >= thirty_days_ago
            ]
            
            amounts = [float(tx.get("amount", 0)) for tx in recent_transactions]
            avg_amount_30d = sum(amounts) / len(amounts) if amounts else 0
            
            # Merchant diversity
            merchants = set(tx.get("merchant", "unknown") for tx in recent_transactions)
            distinct_merchants_30d = len(merchants)
            
            # Failed attempts
            failed_attempts_24h = len([
                tx for tx in user_history
                if (tx.get("status") == "failed" and
                    datetime.fromisoformat(tx.get("timestamp", "")) >= twenty_four_hours_ago)
            ])
            
            # Location risk (simplified)
            current_country = transaction_data.get("country", "US")
            recent_countries = set(tx.get("country", "US") for tx in recent_transactions[-10:])
            location_risk = 1.0 if current_country not in recent_countries else 0.0
            
            return [
                amount, hour_of_day, day_of_week, user_age_days,
                tx_count_1h, tx_count_24h, avg_amount_30d,
                distinct_merchants_30d, failed_attempts_24h, location_risk
            ]
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return [0.0] * len(self.feature_names)
    
    async def _apply_business_rules(
        self,
        transaction_data: Dict[str, Any],
        user_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply business rules for fraud detection"""
        try:
            risk_score = 0
            triggered_rules = []
            
            amount = float(transaction_data.get("amount", 0))
            
            # Rule 1: Large transaction amount
            if amount > 5000:
                risk_score += 40
                triggered_rules.append("large_amount_5k")
            elif amount > 1000:
                risk_score += 25
                triggered_rules.append("large_amount_1k")
            
            # Rule 2: High velocity
            recent_count = len([
                tx for tx in user_history
                if (datetime.utcnow() - datetime.fromisoformat(tx.get("timestamp", ""))) < timedelta(hours=1)
            ])
            
            if recent_count >= 10:
                risk_score += 50
                triggered_rules.append("high_velocity_extreme")
            elif recent_count >= 5:
                risk_score += 30
                triggered_rules.append("high_velocity")
            
            # Rule 3: Round number amounts (potential testing)
            if amount in [100, 200, 500, 1000, 2000, 5000]:
                risk_score += 15
                triggered_rules.append("round_amount")
            
            # Rule 4: Late night transactions
            hour = datetime.fromisoformat(
                transaction_data.get("timestamp", datetime.utcnow().isoformat())
            ).hour
            
            if hour >= 23 or hour <= 5:
                risk_score += 10
                triggered_rules.append("late_night")
            
            # Rule 5: New user with large transaction
            if len(user_history) <= 1 and amount > 500:
                risk_score += 35
                triggered_rules.append("new_user_large_amount")
            
            # Rule 6: Multiple failed attempts
            failed_count = len([
                tx for tx in user_history
                if tx.get("status") == "failed"
            ])
            
            if failed_count >= 5:
                risk_score += 45
                triggered_rules.append("multiple_failures")
            elif failed_count >= 3:
                risk_score += 25
                triggered_rules.append("some_failures")
            
            # Determine action
            if risk_score >= 70:
                recommended_action = "block"
            elif risk_score >= 50:
                recommended_action = "review"
            elif risk_score >= 30:
                recommended_action = "flag"
            else:
                recommended_action = "approve"
            
            return {
                "risk_score": min(risk_score, 100),
                "triggered_rules": triggered_rules,
                "recommended_action": recommended_action
            }
            
        except Exception as e:
            logger.error(f"Business rules analysis failed: {e}")
            return {
                "risk_score": 0,
                "triggered_rules": [],
                "recommended_action": "approve"
            }
    
    async def train_model(self, training_data: List[Dict[str, Any]]):
        """Train the fraud detection model with new data"""
        try:
            if len(training_data) < 100:
                logger.warning("Insufficient training data for model update")
                return False
            
            # Extract features from training data
            features_list = []
            for data_point in training_data:
                transaction_data = data_point["transaction"]
                user_history = data_point.get("user_history", [])
                features = self._extract_features(transaction_data, user_history)
                features_list.append(features)
            
            # Convert to numpy array
            X = np.array(features_list)
            
            # Fit scaler and model
            X_scaled = self.scaler.fit_transform(X)
            self.isolation_forest.fit(X_scaled)
            
            # Save model
            if self.model_path:
                model_data = {
                    'model': self.isolation_forest,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'training_date': datetime.utcnow().isoformat(),
                    'training_size': len(training_data)
                }
                
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                joblib.dump(model_data, self.model_path)
                
                logger.info(f"Trained and saved fraud detection model with {len(training_data)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for fraud detection"""
        try:
            # For isolation forest, we can't get direct feature importance
            # Instead, we can analyze feature distributions in anomalies vs normal
            
            # This is a simplified approach - in production, use more sophisticated methods
            importance = {}
            for i, feature_name in enumerate(self.feature_names):
                # Assign importance based on typical fraud patterns
                if feature_name in ['amount', 'transaction_count_1h', 'failed_attempts_24h']:
                    importance[feature_name] = 0.8
                elif feature_name in ['hour_of_day', 'location_risk', 'user_age_days']:
                    importance[feature_name] = 0.6
                else:
                    importance[feature_name] = 0.4
            
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}


class BehavioralAnalyzer:
    """Analyze user behavioral patterns for fraud detection"""
    
    def __init__(self):
        self.behavioral_models = {}
    
    async def build_user_profile(
        self,
        user_id: int,
        transaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build behavioral profile for a user"""
        try:
            if not transaction_history:
                return {"error": "No transaction history available"}
            
            # Analyze spending patterns
            amounts = [float(tx.get("amount", 0)) for tx in transaction_history]
            
            # Time patterns
            hours = []
            days_of_week = []
            for tx in transaction_history:
                timestamp = datetime.fromisoformat(tx.get("timestamp", ""))
                hours.append(timestamp.hour)
                days_of_week.append(timestamp.weekday())
            
            # Location patterns
            countries = [tx.get("country", "unknown") for tx in transaction_history]
            ip_addresses = [tx.get("ip_address", "") for tx in transaction_history]
            
            # Merchant patterns
            merchants = [tx.get("merchant", "unknown") for tx in transaction_history]
            
            profile = {
                "user_id": user_id,
                "total_transactions": len(transaction_history),
                "spending_patterns": {
                    "avg_amount": np.mean(amounts) if amounts else 0,
                    "median_amount": np.median(amounts) if amounts else 0,
                    "std_amount": np.std(amounts) if amounts else 0,
                    "min_amount": min(amounts) if amounts else 0,
                    "max_amount": max(amounts) if amounts else 0
                },
                "time_patterns": {
                    "preferred_hours": self._find_common_values(hours),
                    "preferred_days": self._find_common_values(days_of_week),
                    "avg_hour": np.mean(hours) if hours else 12,
                    "hour_variance": np.var(hours) if hours else 0
                },
                "location_patterns": {
                    "common_countries": self._find_common_values(countries),
                    "country_diversity": len(set(countries)),
                    "common_ips": self._find_common_values(ip_addresses)
                },
                "merchant_patterns": {
                    "common_merchants": self._find_common_values(merchants),
                    "merchant_diversity": len(set(merchants))
                },
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store profile
            self.behavioral_models[user_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to build user profile: {e}")
            return {"error": str(e)}
    
    def _find_common_values(self, values: List[Any], top_n: int = 3) -> List[Dict[str, Any]]:
        """Find most common values in a list"""
        try:
            from collections import Counter
            
            if not values:
                return []
            
            counter = Counter(values)
            common = counter.most_common(top_n)
            
            return [
                {"value": value, "count": count, "frequency": count / len(values)}
                for value, count in common
            ]
            
        except Exception as e:
            logger.error(f"Failed to find common values: {e}")
            return []
    
    async def detect_behavioral_anomalies(
        self,
        user_id: int,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect behavioral anomalies for a transaction"""
        try:
            profile = self.behavioral_models.get(user_id)
            if not profile:
                return {"anomaly_score": 0, "reason": "No behavioral profile available"}
            
            anomalies = []
            anomaly_score = 0
            
            # Check amount anomaly
            amount = float(transaction_data.get("amount", 0))
            avg_amount = profile["spending_patterns"]["avg_amount"]
            std_amount = profile["spending_patterns"]["std_amount"]
            
            if std_amount > 0:
                z_score = abs(amount - avg_amount) / std_amount
                if z_score > 3:  # 3 sigma rule
                    anomalies.append({
                        "type": "amount_anomaly",
                        "severity": "high" if z_score > 5 else "medium",
                        "details": f"Amount ${amount} is {z_score:.2f} standard deviations from normal"
                    })
                    anomaly_score += 40 if z_score > 5 else 25
            
            # Check time anomaly
            timestamp = datetime.fromisoformat(
                transaction_data.get("timestamp", datetime.utcnow().isoformat())
            )
            
            preferred_hours = [h["value"] for h in profile["time_patterns"]["preferred_hours"]]
            if preferred_hours and timestamp.hour not in preferred_hours:
                anomalies.append({
                    "type": "time_anomaly",
                    "severity": "low",
                    "details": f"Transaction at hour {timestamp.hour} is unusual for this user"
                })
                anomaly_score += 10
            
            # Check location anomaly
            country = transaction_data.get("country", "unknown")
            common_countries = [c["value"] for c in profile["location_patterns"]["common_countries"]]
            
            if common_countries and country not in common_countries:
                anomalies.append({
                    "type": "location_anomaly",
                    "severity": "medium",
                    "details": f"Transaction from {country} is unusual for this user"
                })
                anomaly_score += 20
            
            # Check merchant anomaly
            merchant = transaction_data.get("merchant", "unknown")
            common_merchants = [m["value"] for m in profile["merchant_patterns"]["common_merchants"]]
            
            if common_merchants and merchant not in common_merchants:
                anomalies.append({
                    "type": "merchant_anomaly",
                    "severity": "low",
                    "details": f"New merchant {merchant} for this user"
                })
                anomaly_score += 5
            
            return {
                "anomaly_score": min(anomaly_score, 100),
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies,
                "profile_age_days": self._calculate_profile_age(profile),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Behavioral anomaly detection failed: {e}")
            return {"anomaly_score": 0, "error": str(e)}
    
    def _calculate_profile_age(self, profile: Dict[str, Any]) -> int:
        """Calculate age of behavioral profile in days"""
        try:
            created_at = datetime.fromisoformat(profile["created_at"])
            return (datetime.utcnow() - created_at).days
        except:
            return 0


class AdaptiveRuleEngine:
    """Adaptive rule engine that learns from fraud patterns"""
    
    def __init__(self):
        self.rules = []
        self.rule_performance = {}
    
    async def add_rule(
        self,
        rule_name: str,
        condition: Dict[str, Any],
        action: str,
        weight: int = 10
    ):
        """Add a new fraud detection rule"""
        try:
            rule = {
                "name": rule_name,
                "condition": condition,
                "action": action,
                "weight": weight,
                "created_at": datetime.utcnow().isoformat(),
                "enabled": True
            }
            
            self.rules.append(rule)
            self.rule_performance[rule_name] = {
                "triggers": 0,
                "true_positives": 0,
                "false_positives": 0,
                "precision": 0.0
            }
            
            logger.info(f"Added fraud detection rule: {rule_name}")
            
        except Exception as e:
            logger.error(f"Failed to add rule: {e}")
    
    async def evaluate_rules(
        self,
        transaction_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate all active rules against a transaction"""
        try:
            triggered_rules = []
            
            for rule in self.rules:
                if not rule.get("enabled", True):
                    continue
                
                if self._evaluate_condition(rule["condition"], transaction_data):
                    triggered_rules.append({
                        "name": rule["name"],
                        "action": rule["action"],
                        "weight": rule["weight"],
                        "condition": rule["condition"]
                    })
                    
                    # Update performance tracking
                    self.rule_performance[rule["name"]]["triggers"] += 1
            
            return triggered_rules
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return []
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        transaction_data: Dict[str, Any]
    ) -> bool:
        """Evaluate a rule condition"""
        try:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if not all([field, operator, value]):
                return False
            
            transaction_value = transaction_data.get(field)
            
            if operator == "gt":
                return float(transaction_value or 0) > float(value)
            elif operator == "lt":
                return float(transaction_value or 0) < float(value)
            elif operator == "eq":
                return transaction_value == value
            elif operator == "in":
                return transaction_value in value
            elif operator == "contains":
                return value in str(transaction_value or "")
            else:
                return False
                
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def update_rule_performance(
        self,
        rule_name: str,
        was_fraud: bool
    ):
        """Update rule performance metrics"""
        try:
            if rule_name not in self.rule_performance:
                return
            
            perf = self.rule_performance[rule_name]
            
            if was_fraud:
                perf["true_positives"] += 1
            else:
                perf["false_positives"] += 1
            
            # Calculate precision
            total_positives = perf["true_positives"] + perf["false_positives"]
            if total_positives > 0:
                perf["precision"] = perf["true_positives"] / total_positives
            
            # Disable rules with consistently poor performance
            if total_positives >= 10 and perf["precision"] < 0.1:
                for rule in self.rules:
                    if rule["name"] == rule_name:
                        rule["enabled"] = False
                        logger.warning(f"Disabled poor-performing rule: {rule_name}")
                        break
            
        except Exception as e:
            logger.error(f"Failed to update rule performance: {e}")
    
    async def get_rule_analytics(self) -> Dict[str, Any]:
        """Get analytics on rule performance"""
        try:
            active_rules = len([r for r in self.rules if r.get("enabled", True)])
            total_rules = len(self.rules)
            
            top_performers = sorted(
                self.rule_performance.items(),
                key=lambda x: x[1]["precision"],
                reverse=True
            )[:5]
            
            return {
                "total_rules": total_rules,
                "active_rules": active_rules,
                "disabled_rules": total_rules - active_rules,
                "top_performing_rules": [
                    {
                        "name": name,
                        "precision": metrics["precision"],
                        "triggers": metrics["triggers"]
                    }
                    for name, metrics in top_performers
                ],
                "rule_performance": self.rule_performance
            }
            
        except Exception as e:
            logger.error(f"Failed to get rule analytics: {e}")
            return {"error": str(e)}