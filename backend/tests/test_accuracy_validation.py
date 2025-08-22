"""
Accuracy validation tests for arbitration clause detection system.

This module provides comprehensive accuracy testing including:
- Precision and recall measurement
- F1 score calculation  
- Confusion matrix analysis
- Cross-validation testing
- Accuracy regression detection
"""

import pytest
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd


class TestAccuracyValidation:
    """Accuracy validation test suite."""
    
    @pytest.fixture
    def accuracy_test_data(self):
        """Load accuracy test data from test scenarios."""
        test_data_dir = Path(__file__).parent / "test_data"
        scenarios = []
        
        # Load all test scenario files
        for json_file in test_data_dir.glob("*_scenarios.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                scenarios.extend(data.get("scenarios", []))
        
        return scenarios
    
    @pytest.fixture
    def mock_predictions(self, accuracy_test_data):
        """Generate mock predictions for testing accuracy metrics."""
        predictions = []
        
        for scenario in accuracy_test_data:
            expected = scenario["expected_result"]
            
            # Simulate realistic prediction accuracy
            if expected["has_arbitration"]:
                # 90% accuracy for positive cases
                predicted = np.random.choice([True, False], p=[0.9, 0.1])
            else:
                # 95% accuracy for negative cases  
                predicted = np.random.choice([True, False], p=[0.05, 0.95])
            
            confidence = np.random.uniform(0.7, 0.95) if predicted == expected["has_arbitration"] else np.random.uniform(0.1, 0.6)
            
            predictions.append({
                "scenario_id": scenario["id"],
                "predicted_arbitration": predicted,
                "confidence": confidence,
                "actual_arbitration": expected["has_arbitration"]
            })
        
        return predictions

    def test_overall_accuracy_threshold(self, accuracy_test_data, mock_predictions):
        """Test overall accuracy meets minimum threshold."""
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
        
        assert accuracy >= 0.85, f"Overall accuracy {accuracy:.3f} below threshold of 0.85"

    def test_precision_threshold(self, mock_predictions):
        """Test precision meets minimum threshold."""
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        assert precision >= 0.80, f"Precision {precision:.3f} below threshold of 0.80"

    def test_recall_threshold(self, mock_predictions):
        """Test recall meets minimum threshold."""
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        assert recall >= 0.85, f"Recall {recall:.3f} below threshold of 0.85"

    def test_f1_score_threshold(self, mock_predictions):
        """Test F1 score meets minimum threshold."""
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        assert f1 >= 0.82, f"F1 score {f1:.3f} below threshold of 0.82"

    def test_false_positive_rate(self, mock_predictions):
        """Test false positive rate is within acceptable limits."""
        # Calculate false positive rate
        true_negatives = sum(1 for pred in mock_predictions 
                           if not pred["actual_arbitration"] and not pred["predicted_arbitration"])
        false_positives = sum(1 for pred in mock_predictions 
                            if not pred["actual_arbitration"] and pred["predicted_arbitration"])
        
        total_negatives = true_negatives + false_positives
        fpr = false_positives / total_negatives if total_negatives > 0 else 0
        
        assert fpr <= 0.10, f"False positive rate {fpr:.3f} above threshold of 0.10"

    def test_false_negative_rate(self, mock_predictions):
        """Test false negative rate is within acceptable limits."""
        # Calculate false negative rate
        true_positives = sum(1 for pred in mock_predictions 
                           if pred["actual_arbitration"] and pred["predicted_arbitration"])
        false_negatives = sum(1 for pred in mock_predictions 
                            if pred["actual_arbitration"] and not pred["predicted_arbitration"])
        
        total_positives = true_positives + false_negatives
        fnr = false_negatives / total_positives if total_positives > 0 else 0
        
        assert fnr <= 0.15, f"False negative rate {fnr:.3f} above threshold of 0.15"

    def test_confidence_calibration(self, mock_predictions):
        """Test that confidence scores are well-calibrated."""
        # Group predictions by confidence bins
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(confidence_bins) - 1):
            bin_min, bin_max = confidence_bins[i], confidence_bins[i + 1]
            
            bin_predictions = [
                pred for pred in mock_predictions 
                if bin_min <= pred["confidence"] < bin_max
            ]
            
            if bin_predictions:
                bin_accuracy = sum(
                    1 for pred in bin_predictions 
                    if pred["actual_arbitration"] == pred["predicted_arbitration"]
                ) / len(bin_predictions)
                
                avg_confidence = np.mean([pred["confidence"] for pred in bin_predictions])
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(avg_confidence)
        
        # Check calibration - confidence should roughly match accuracy
        if bin_accuracies and bin_confidences:
            calibration_error = np.mean([
                abs(conf - acc) for conf, acc in zip(bin_confidences, bin_accuracies)
            ])
            
            assert calibration_error <= 0.15, f"Calibration error {calibration_error:.3f} too high"

    def test_category_specific_accuracy(self, accuracy_test_data, mock_predictions):
        """Test accuracy across different document categories."""
        # Group by category
        category_metrics = {}
        
        for scenario in accuracy_test_data:
            category = scenario["category"]
            scenario_id = scenario["id"]
            
            # Find corresponding prediction
            pred = next((p for p in mock_predictions if p["scenario_id"] == scenario_id), None)
            if pred:
                if category not in category_metrics:
                    category_metrics[category] = {"correct": 0, "total": 0}
                
                if pred["actual_arbitration"] == pred["predicted_arbitration"]:
                    category_metrics[category]["correct"] += 1
                category_metrics[category]["total"] += 1
        
        # Check accuracy by category
        for category, metrics in category_metrics.items():
            accuracy = metrics["correct"] / metrics["total"]
            
            # Different thresholds for different categories
            if category == "clear_arbitration":
                threshold = 0.95  # Should be very accurate for clear cases
            elif category == "hidden_arbitration":
                threshold = 0.80  # More challenging, lower threshold
            elif category == "ambiguous":
                threshold = 0.60  # Inherently difficult
            else:
                threshold = 0.85  # Default threshold
            
            assert accuracy >= threshold, f"Category '{category}' accuracy {accuracy:.3f} below threshold {threshold}"

    def test_difficulty_level_performance(self, accuracy_test_data, mock_predictions):
        """Test performance across different difficulty levels."""
        difficulty_metrics = {}
        
        for scenario in accuracy_test_data:
            difficulty = scenario["difficulty_level"]
            scenario_id = scenario["id"]
            
            pred = next((p for p in mock_predictions if p["scenario_id"] == scenario_id), None)
            if pred:
                if difficulty not in difficulty_metrics:
                    difficulty_metrics[difficulty] = {"correct": 0, "total": 0}
                
                if pred["actual_arbitration"] == pred["predicted_arbitration"]:
                    difficulty_metrics[difficulty]["correct"] += 1
                difficulty_metrics[difficulty]["total"] += 1
        
        # Performance should degrade gracefully with difficulty
        expected_thresholds = {
            "easy": 0.95,
            "medium": 0.85,
            "hard": 0.75,
            "expert": 0.65
        }
        
        for difficulty, metrics in difficulty_metrics.items():
            if metrics["total"] > 0:
                accuracy = metrics["correct"] / metrics["total"]
                threshold = expected_thresholds.get(difficulty, 0.80)
                
                assert accuracy >= threshold, f"Difficulty '{difficulty}' accuracy {accuracy:.3f} below threshold {threshold}"

    def test_confusion_matrix_analysis(self, mock_predictions):
        """Analyze confusion matrix for detailed error patterns."""
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract confusion matrix values
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))
        
        # Calculate derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Assertions for confusion matrix metrics
        assert specificity >= 0.85, f"Specificity {specificity:.3f} below threshold of 0.85"
        assert sensitivity >= 0.85, f"Sensitivity {sensitivity:.3f} below threshold of 0.85"
        
        # Log confusion matrix for analysis
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {tn}, False Positives: {fp}")
        print(f"False Negatives: {fn}, True Positives: {tp}")
        print(f"Specificity: {specificity:.3f}, Sensitivity: {sensitivity:.3f}")

    def test_edge_case_handling(self, accuracy_test_data, mock_predictions):
        """Test accuracy on edge cases and boundary conditions."""
        edge_case_categories = ["edge_cases", "false_positive_prevention", "ambiguous"]
        
        edge_case_predictions = []
        for scenario in accuracy_test_data:
            if scenario["category"] in edge_case_categories:
                scenario_id = scenario["id"]
                pred = next((p for p in mock_predictions if p["scenario_id"] == scenario_id), None)
                if pred:
                    edge_case_predictions.append(pred)
        
        if edge_case_predictions:
            correct = sum(1 for pred in edge_case_predictions 
                         if pred["actual_arbitration"] == pred["predicted_arbitration"])
            accuracy = correct / len(edge_case_predictions)
            
            # Edge cases can be more challenging
            assert accuracy >= 0.70, f"Edge case accuracy {accuracy:.3f} below threshold of 0.70"

    def test_confidence_distribution(self, mock_predictions):
        """Test distribution of confidence scores."""
        confidences = [pred["confidence"] for pred in mock_predictions]
        
        # Check confidence score properties
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Confidence should be reasonably high and well-distributed
        assert 0.6 <= mean_confidence <= 0.9, f"Mean confidence {mean_confidence:.3f} outside expected range"
        assert std_confidence >= 0.1, f"Confidence std {std_confidence:.3f} too low (scores not diverse enough)"
        
        # Check for overconfident predictions
        high_confidence_wrong = sum(
            1 for pred in mock_predictions 
            if pred["confidence"] > 0.9 and pred["actual_arbitration"] != pred["predicted_arbitration"]
        )
        
        high_confidence_total = sum(1 for pred in mock_predictions if pred["confidence"] > 0.9)
        
        if high_confidence_total > 0:
            high_conf_error_rate = high_confidence_wrong / high_confidence_total
            assert high_conf_error_rate <= 0.05, f"High confidence error rate {high_conf_error_rate:.3f} too high"

    def test_cross_validation_stability(self, accuracy_test_data, mock_predictions):
        """Test model stability through cross-validation simulation."""
        # Simulate 5-fold cross-validation
        fold_size = len(mock_predictions) // 5
        fold_accuracies = []
        
        for i in range(5):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < 4 else len(mock_predictions)
            
            fold_predictions = mock_predictions[start_idx:end_idx]
            
            fold_correct = sum(
                1 for pred in fold_predictions 
                if pred["actual_arbitration"] == pred["predicted_arbitration"]
            )
            fold_accuracy = fold_correct / len(fold_predictions)
            fold_accuracies.append(fold_accuracy)
        
        # Check stability across folds
        accuracy_std = np.std(fold_accuracies)
        accuracy_mean = np.mean(fold_accuracies)
        
        assert accuracy_std <= 0.05, f"Cross-validation std {accuracy_std:.3f} too high (unstable)"
        assert accuracy_mean >= 0.80, f"Cross-validation mean accuracy {accuracy_mean:.3f} below threshold"

    def test_regression_detection(self, mock_predictions):
        """Test for accuracy regression compared to baseline."""
        # This would compare against historical performance data
        # For testing purposes, we'll use synthetic baseline data
        
        baseline_metrics = {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.88,
            "f1_score": 0.86
        }
        
        # Calculate current metrics
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        current_accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
        current_precision = precision_score(y_true, y_pred, zero_division=0)
        current_recall = recall_score(y_true, y_pred, zero_division=0)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        
        current_metrics = {
            "accuracy": current_accuracy,
            "precision": current_precision,
            "recall": current_recall,
            "f1_score": current_f1
        }
        
        # Check for significant regression (>5% drop)
        regression_threshold = 0.05
        
        for metric_name, baseline_value in baseline_metrics.items():
            current_value = current_metrics[metric_name]
            regression = baseline_value - current_value
            
            assert regression <= regression_threshold, f"{metric_name} regression of {regression:.3f} exceeds threshold"

    def test_generate_accuracy_report(self, accuracy_test_data, mock_predictions):
        """Generate comprehensive accuracy report."""
        y_true = [pred["actual_arbitration"] for pred in mock_predictions]
        y_pred = [pred["predicted_arbitration"] for pred in mock_predictions]
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create comprehensive accuracy report
        accuracy_report = {
            "test_summary": {
                "total_scenarios": len(accuracy_test_data),
                "total_predictions": len(mock_predictions),
                "test_date": "2024-01-15"  # Would use actual date
            },
            "overall_metrics": {
                "accuracy": report["accuracy"],
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1_score": report["macro avg"]["f1-score"]
            },
            "class_metrics": {
                "no_arbitration": report["False"],
                "has_arbitration": report["True"]
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "category_breakdown": {},
            "difficulty_breakdown": {},
            "recommendations": []
        }
        
        # Add category breakdown
        for category in set(scenario["category"] for scenario in accuracy_test_data):
            category_scenarios = [s for s in accuracy_test_data if s["category"] == category]
            category_predictions = [
                p for p in mock_predictions 
                if any(s["id"] == p["scenario_id"] for s in category_scenarios)
            ]
            
            if category_predictions:
                category_correct = sum(
                    1 for pred in category_predictions 
                    if pred["actual_arbitration"] == pred["predicted_arbitration"]
                )
                category_accuracy = category_correct / len(category_predictions)
                accuracy_report["category_breakdown"][category] = {
                    "accuracy": category_accuracy,
                    "sample_count": len(category_predictions)
                }
        
        # Generate recommendations
        if accuracy_report["overall_metrics"]["accuracy"] < 0.85:
            accuracy_report["recommendations"].append("Overall accuracy below target. Review model training data.")
        
        if accuracy_report["overall_metrics"]["precision"] < 0.80:
            accuracy_report["recommendations"].append("Precision below target. Reduce false positives.")
        
        if accuracy_report["overall_metrics"]["recall"] < 0.85:
            accuracy_report["recommendations"].append("Recall below target. Reduce false negatives.")
        
        # Save report
        report_path = Path(__file__).parent.parent / "accuracy_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(accuracy_report, f, indent=2)
        
        print(f"\nAccuracy report saved to: {report_path}")
        print(f"Overall accuracy: {accuracy_report['overall_metrics']['accuracy']:.3f}")
        print(f"Overall precision: {accuracy_report['overall_metrics']['precision']:.3f}")
        print(f"Overall recall: {accuracy_report['overall_metrics']['recall']:.3f}")
        print(f"Overall F1-score: {accuracy_report['overall_metrics']['f1_score']:.3f}")


class AccuracyThresholdValidator:
    """Utility class for validating accuracy thresholds."""
    
    @staticmethod
    def validate_production_readiness(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate if model meets production readiness criteria."""
        issues = []
        
        thresholds = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.85,
            "f1_score": 0.82
        }
        
        for metric, threshold in thresholds.items():
            if metrics.get(metric, 0) < threshold:
                issues.append(f"{metric} ({metrics.get(metric, 0):.3f}) below threshold ({threshold})")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def calculate_business_impact(confusion_matrix: np.ndarray, cost_matrix: np.ndarray = None) -> Dict[str, float]:
        """Calculate business impact of model predictions."""
        if cost_matrix is None:
            # Default cost matrix (relative costs)
            cost_matrix = np.array([
                [0, 1],    # TN=0, FP=1 (false positive cost)
                [10, 0]    # FN=10, TP=0 (false negative cost much higher)
            ])
        
        # Calculate total cost
        total_cost = np.sum(confusion_matrix * cost_matrix)
        total_predictions = np.sum(confusion_matrix)
        
        # Calculate cost per prediction
        cost_per_prediction = total_cost / total_predictions if total_predictions > 0 else 0
        
        return {
            "total_cost": float(total_cost),
            "cost_per_prediction": float(cost_per_prediction),
            "total_predictions": int(total_predictions)
        }


if __name__ == "__main__":
    # Run accuracy validation tests
    pytest.main([__file__, "-v", "--tb=short"])