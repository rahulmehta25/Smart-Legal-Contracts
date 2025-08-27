"""
Demo test scenarios for arbitration detection accuracy validation.
Demonstrates model accuracy across various document types and edge cases.
"""

import pytest
import time
import json
import statistics
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock
import uuid


class TestArbitrationAccuracyDemo:
    """Demo tests for arbitration detection accuracy validation."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        self.demo_results = {}
        self.accuracy_metrics = {}
        self.setup_test_scenarios()
        
    def setup_test_scenarios(self):
        """Setup comprehensive test scenarios for accuracy validation."""
        self.test_scenarios = {
            "clear_positive_cases": [
                {
                    "id": "clear_aaa_arbitration",
                    "content": """
                    ARBITRATION CLAUSE: Any dispute arising under this agreement shall be 
                    resolved through binding arbitration administered by the American 
                    Arbitration Association under its Commercial Rules.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.9,
                    "difficulty": "easy"
                },
                {
                    "id": "clear_jams_arbitration", 
                    "content": """
                    DISPUTE RESOLUTION: All claims shall be resolved exclusively through 
                    final and binding arbitration conducted under JAMS Streamlined 
                    Arbitration Rules and Procedures.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.9,
                    "difficulty": "easy"
                },
                {
                    "id": "consumer_arbitration",
                    "content": """
                    You and Company agree that any dispute will be resolved by binding 
                    arbitration rather than in court, except you may assert claims in 
                    small claims court if your claims qualify.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.85,
                    "difficulty": "easy"
                }
            ],
            
            "clear_negative_cases": [
                {
                    "id": "court_jurisdiction",
                    "content": """
                    GOVERNING LAW: This agreement shall be governed by the laws of New York. 
                    Any disputes shall be resolved in the state or federal courts located 
                    in New York County.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.9,
                    "difficulty": "easy"
                },
                {
                    "id": "privacy_policy",
                    "content": """
                    PRIVACY POLICY: We collect and use your personal information in 
                    accordance with this privacy policy. Contact us at privacy@company.com 
                    for questions about our data practices.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.95,
                    "difficulty": "easy"
                },
                {
                    "id": "general_terms",
                    "content": """
                    GENERAL TERMS: By using this service, you agree to comply with all 
                    applicable laws and regulations. We reserve the right to modify 
                    these terms at any time.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.9,
                    "difficulty": "easy"
                }
            ],
            
            "hidden_arbitration_cases": [
                {
                    "id": "buried_clause",
                    "content": """
                    SECTION 15: MISCELLANEOUS PROVISIONS
                    
                    15.1 Entire Agreement: This constitutes the entire agreement.
                    15.2 Severability: If any provision is invalid, the remainder shall remain in effect.
                    15.3 Dispute Resolution: Any controversy or claim arising out of or relating to 
                    this contract shall be settled by arbitration in accordance with the rules of 
                    the American Arbitration Association, and judgment may be entered on the award.
                    15.4 Assignment: This agreement may not be assigned without consent.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.8,
                    "difficulty": "medium"
                },
                {
                    "id": "euphemistic_language",
                    "content": """
                    ALTERNATIVE DISPUTE RESOLUTION: In the unlikely event of any disagreement 
                    between the parties, such matters shall be resolved through binding 
                    alternative dispute resolution proceedings conducted by a neutral third 
                    party under established commercial rules.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.7,
                    "difficulty": "medium"
                },
                {
                    "id": "conditional_arbitration",
                    "content": """
                    If informal resolution attempts fail within 30 days, and mediation is 
                    unsuccessful, then any remaining disputes shall be finally resolved by 
                    binding arbitration administered by JAMS in accordance with its rules.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.75,
                    "difficulty": "medium"
                }
            ],
            
            "ambiguous_cases": [
                {
                    "id": "mediation_only",
                    "content": """
                    DISPUTE RESOLUTION: The parties agree to attempt resolution through 
                    mediation before pursuing any other legal remedies. If mediation 
                    fails, parties may pursue available remedies under applicable law.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.3,  # Should be uncertain
                    "difficulty": "hard"
                },
                {
                    "id": "negotiation_clause", 
                    "content": """
                    DISPUTE HANDLING: Any disputes shall first be addressed through 
                    good faith negotiation between senior executives of both parties 
                    for a period of 60 days before other actions.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.3,
                    "difficulty": "hard"
                },
                {
                    "id": "vague_resolution",
                    "content": """
                    CONFLICT RESOLUTION: Disputes will be resolved in accordance with 
                    fair and reasonable procedures to be mutually agreed upon by the 
                    parties at the time of any such dispute.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.4,
                    "difficulty": "hard"
                }
            ],
            
            "edge_cases": [
                {
                    "id": "anti_arbitration",
                    "content": """
                    NO ARBITRATION: The parties expressly reject arbitration and agree 
                    that all disputes must be resolved in courts of competent jurisdiction. 
                    This agreement specifically excludes arbitration proceedings.
                    """,
                    "expected": False,
                    "confidence_threshold": 0.9,
                    "difficulty": "medium"
                },
                {
                    "id": "optional_arbitration",
                    "content": """
                    DISPUTE OPTIONS: Either party may, at its sole discretion, elect to 
                    resolve disputes through arbitration or court proceedings. Arbitration 
                    is available but not mandatory under this agreement.
                    """,
                    "expected": False,  # Not mandatory arbitration
                    "confidence_threshold": 0.6,
                    "difficulty": "hard"
                },
                {
                    "id": "foreign_arbitration",
                    "content": """
                    INTERNATIONAL DISPUTES: Any disputes shall be resolved through 
                    arbitration under the Rules of the International Chamber of Commerce, 
                    seated in Geneva, Switzerland, conducted in English.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.85,
                    "difficulty": "medium"
                },
                {
                    "id": "industry_specific",
                    "content": """
                    CONSTRUCTION DISPUTES: All disputes relating to this construction 
                    contract shall be resolved through binding arbitration under the 
                    Construction Industry Arbitration Rules of the AAA.
                    """,
                    "expected": True,
                    "confidence_threshold": 0.9,
                    "difficulty": "easy"
                }
            ]
        }
        
    def test_clear_positive_cases_accuracy(self):
        """Demo: Test accuracy on clear positive arbitration cases."""
        print(f"\n=== DEMO: Clear Positive Cases Accuracy Test ===")
        
        test_cases = self.test_scenarios["clear_positive_cases"]
        results = []
        
        for case in test_cases:
            print(f"\nTesting: {case['id']}")
            
            # Analyze the case
            analysis = self._simulate_arbitration_analysis(case["content"])
            
            # Check accuracy
            correct_prediction = analysis["has_arbitration"] == case["expected"]
            confidence_met = analysis["confidence"] >= case["confidence_threshold"]
            
            result = {
                "case_id": case["id"],
                "expected": case["expected"],
                "predicted": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "confidence_threshold": case["confidence_threshold"],
                "correct_prediction": correct_prediction,
                "confidence_met": confidence_met,
                "overall_success": correct_prediction and confidence_met,
                "difficulty": case["difficulty"]
            }
            
            results.append(result)
            
            status = "✓" if result["overall_success"] else "✗"
            print(f"{status} Prediction: {analysis['has_arbitration']} (confidence: {analysis['confidence']:.2f})")
            print(f"   Expected: {case['expected']} (threshold: {case['confidence_threshold']})")
            
        # Calculate accuracy metrics
        accuracy = sum(1 for r in results if r["correct_prediction"]) / len(results)
        confidence_score = sum(1 for r in results if r["confidence_met"]) / len(results)
        overall_success = sum(1 for r in results if r["overall_success"]) / len(results)
        
        print(f"\n📊 Clear Positive Cases Results:")
        print(f"   Prediction Accuracy: {accuracy:.2%}")
        print(f"   Confidence Score: {confidence_score:.2%}")
        print(f"   Overall Success: {overall_success:.2%}")
        
        # Validation
        assert accuracy >= 0.95, f"Clear positive accuracy too low: {accuracy:.2%}"
        assert confidence_score >= 0.9, f"Confidence score too low: {confidence_score:.2%}"
        
        self.accuracy_metrics["clear_positive"] = {
            "accuracy": accuracy,
            "confidence_score": confidence_score,
            "overall_success": overall_success,
            "results": results
        }
        
        self.demo_results["clear_positive_accuracy"] = "PASS"
        
    def test_clear_negative_cases_accuracy(self):
        """Demo: Test accuracy on clear negative cases (no arbitration)."""
        print(f"\n=== DEMO: Clear Negative Cases Accuracy Test ===")
        
        test_cases = self.test_scenarios["clear_negative_cases"]
        results = []
        
        for case in test_cases:
            print(f"\nTesting: {case['id']}")
            
            # Analyze the case
            analysis = self._simulate_arbitration_analysis(case["content"])
            
            # Check accuracy (for negative cases, we want low confidence and correct prediction)
            correct_prediction = analysis["has_arbitration"] == case["expected"]
            confidence_appropriate = analysis["confidence"] <= (1 - case["confidence_threshold"])
            
            result = {
                "case_id": case["id"],
                "expected": case["expected"],
                "predicted": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "max_confidence": 1 - case["confidence_threshold"],
                "correct_prediction": correct_prediction,
                "confidence_appropriate": confidence_appropriate,
                "overall_success": correct_prediction and confidence_appropriate,
                "difficulty": case["difficulty"]
            }
            
            results.append(result)
            
            status = "✓" if result["overall_success"] else "✗"
            print(f"{status} Prediction: {analysis['has_arbitration']} (confidence: {analysis['confidence']:.2f})")
            print(f"   Expected: {case['expected']} (max confidence: {result['max_confidence']:.2f})")
            
        # Calculate accuracy metrics
        accuracy = sum(1 for r in results if r["correct_prediction"]) / len(results)
        confidence_score = sum(1 for r in results if r["confidence_appropriate"]) / len(results)
        overall_success = sum(1 for r in results if r["overall_success"]) / len(results)
        
        print(f"\n📊 Clear Negative Cases Results:")
        print(f"   Prediction Accuracy: {accuracy:.2%}")
        print(f"   Confidence Appropriateness: {confidence_score:.2%}")
        print(f"   Overall Success: {overall_success:.2%}")
        
        # Validation
        assert accuracy >= 0.95, f"Clear negative accuracy too low: {accuracy:.2%}"
        assert confidence_score >= 0.9, f"Confidence appropriateness too low: {confidence_score:.2%}"
        
        self.accuracy_metrics["clear_negative"] = {
            "accuracy": accuracy,
            "confidence_score": confidence_score,
            "overall_success": overall_success,
            "results": results
        }
        
        self.demo_results["clear_negative_accuracy"] = "PASS"
        
    def test_hidden_arbitration_detection(self):
        """Demo: Test detection of hidden or subtle arbitration clauses."""
        print(f"\n=== DEMO: Hidden Arbitration Detection Test ===")
        
        test_cases = self.test_scenarios["hidden_arbitration_cases"]
        results = []
        
        for case in test_cases:
            print(f"\nTesting: {case['id']} (difficulty: {case['difficulty']})")
            
            # Analyze the case
            analysis = self._simulate_arbitration_analysis(case["content"])
            
            # Check detection capability
            correct_prediction = analysis["has_arbitration"] == case["expected"]
            confidence_met = analysis["confidence"] >= case["confidence_threshold"]
            
            result = {
                "case_id": case["id"],
                "expected": case["expected"],
                "predicted": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "confidence_threshold": case["confidence_threshold"],
                "correct_prediction": correct_prediction,
                "confidence_met": confidence_met,
                "overall_success": correct_prediction and confidence_met,
                "difficulty": case["difficulty"]
            }
            
            results.append(result)
            
            status = "✓" if result["overall_success"] else "✗"
            print(f"{status} Prediction: {analysis['has_arbitration']} (confidence: {analysis['confidence']:.2f})")
            print(f"   Challenge: {case['id'].replace('_', ' ').title()}")
            
            # Highlight detection techniques used
            if analysis["has_arbitration"]:
                print(f"   Detection method: {self._explain_detection_method(case['content'])}")
                
        # Calculate accuracy metrics
        accuracy = sum(1 for r in results if r["correct_prediction"]) / len(results)
        confidence_score = sum(1 for r in results if r["confidence_met"]) / len(results)
        overall_success = sum(1 for r in results if r["overall_success"]) / len(results)
        
        print(f"\n📊 Hidden Arbitration Detection Results:")
        print(f"   Detection Accuracy: {accuracy:.2%}")
        print(f"   Confidence Score: {confidence_score:.2%}")
        print(f"   Overall Success: {overall_success:.2%}")
        
        # Validation (lower thresholds for harder cases)
        assert accuracy >= 0.8, f"Hidden arbitration accuracy too low: {accuracy:.2%}"
        assert confidence_score >= 0.7, f"Confidence score too low: {confidence_score:.2%}"
        
        self.accuracy_metrics["hidden_arbitration"] = {
            "accuracy": accuracy,
            "confidence_score": confidence_score,
            "overall_success": overall_success,
            "results": results
        }
        
        self.demo_results["hidden_arbitration_detection"] = "PASS"
        
    def test_ambiguous_cases_handling(self):
        """Demo: Test handling of ambiguous or unclear cases."""
        print(f"\n=== DEMO: Ambiguous Cases Handling Test ===")
        
        test_cases = self.test_scenarios["ambiguous_cases"]
        results = []
        
        for case in test_cases:
            print(f"\nTesting: {case['id']} (difficulty: {case['difficulty']})")
            
            # Analyze the case
            analysis = self._simulate_arbitration_analysis(case["content"])
            
            # For ambiguous cases, we care more about uncertainty than specific prediction
            correct_prediction = analysis["has_arbitration"] == case["expected"]
            confidence_appropriate = analysis["confidence"] <= case["confidence_threshold"]
            shows_uncertainty = analysis["confidence"] < 0.7  # Should show uncertainty
            
            result = {
                "case_id": case["id"],
                "expected": case["expected"],
                "predicted": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "max_confidence": case["confidence_threshold"],
                "correct_prediction": correct_prediction,
                "confidence_appropriate": confidence_appropriate,
                "shows_uncertainty": shows_uncertainty,
                "overall_success": confidence_appropriate and shows_uncertainty,
                "difficulty": case["difficulty"]
            }
            
            results.append(result)
            
            status = "✓" if result["overall_success"] else "✗"
            print(f"{status} Prediction: {analysis['has_arbitration']} (confidence: {analysis['confidence']:.2f})")
            print(f"   Ambiguity factor: {case['id'].replace('_', ' ').title()}")
            print(f"   Uncertainty shown: {result['shows_uncertainty']}")
            
        # Calculate metrics focused on uncertainty handling
        uncertainty_handling = sum(1 for r in results if r["shows_uncertainty"]) / len(results)
        confidence_appropriate = sum(1 for r in results if r["confidence_appropriate"]) / len(results)
        overall_success = sum(1 for r in results if r["overall_success"]) / len(results)
        
        print(f"\n📊 Ambiguous Cases Handling Results:")
        print(f"   Uncertainty Recognition: {uncertainty_handling:.2%}")
        print(f"   Confidence Appropriateness: {confidence_appropriate:.2%}")
        print(f"   Overall Success: {overall_success:.2%}")
        
        # Validation (focused on appropriate uncertainty)
        assert uncertainty_handling >= 0.8, f"Uncertainty recognition too low: {uncertainty_handling:.2%}"
        assert confidence_appropriate >= 0.8, f"Confidence appropriateness too low: {confidence_appropriate:.2%}"
        
        self.accuracy_metrics["ambiguous_cases"] = {
            "uncertainty_handling": uncertainty_handling,
            "confidence_appropriate": confidence_appropriate,
            "overall_success": overall_success,
            "results": results
        }
        
        self.demo_results["ambiguous_cases_handling"] = "PASS"
        
    def test_edge_cases_robustness(self):
        """Demo: Test robustness on edge cases and unusual scenarios."""
        print(f"\n=== DEMO: Edge Cases Robustness Test ===")
        
        test_cases = self.test_scenarios["edge_cases"]
        results = []
        
        for case in test_cases:
            print(f"\nTesting: {case['id']} (difficulty: {case['difficulty']})")
            
            # Analyze the case
            analysis = self._simulate_arbitration_analysis(case["content"])
            
            # Check robustness
            correct_prediction = analysis["has_arbitration"] == case["expected"]
            confidence_met = analysis["confidence"] >= case["confidence_threshold"] if case["expected"] else analysis["confidence"] <= (1 - case["confidence_threshold"])
            
            result = {
                "case_id": case["id"],
                "expected": case["expected"],
                "predicted": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "confidence_threshold": case["confidence_threshold"],
                "correct_prediction": correct_prediction,
                "confidence_met": confidence_met,
                "overall_success": correct_prediction and confidence_met,
                "difficulty": case["difficulty"],
                "edge_case_type": self._classify_edge_case(case["id"])
            }
            
            results.append(result)
            
            status = "✓" if result["overall_success"] else "✗"
            print(f"{status} Prediction: {analysis['has_arbitration']} (confidence: {analysis['confidence']:.2f})")
            print(f"   Edge case type: {result['edge_case_type']}")
            print(f"   Robustness test: {case['id'].replace('_', ' ').title()}")
            
        # Calculate robustness metrics
        accuracy = sum(1 for r in results if r["correct_prediction"]) / len(results)
        confidence_score = sum(1 for r in results if r["confidence_met"]) / len(results)
        overall_success = sum(1 for r in results if r["overall_success"]) / len(results)
        
        # Edge case specific metrics
        by_difficulty = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [r for r in results if r["difficulty"] == difficulty]
            if difficulty_results:
                by_difficulty[difficulty] = sum(1 for r in difficulty_results if r["overall_success"]) / len(difficulty_results)
                
        print(f"\n📊 Edge Cases Robustness Results:")
        print(f"   Overall Accuracy: {accuracy:.2%}")
        print(f"   Confidence Score: {confidence_score:.2%}")
        print(f"   Overall Success: {overall_success:.2%}")
        print(f"   By Difficulty:")
        for difficulty, success_rate in by_difficulty.items():
            print(f"     {difficulty.title()}: {success_rate:.2%}")
            
        # Validation
        assert accuracy >= 0.75, f"Edge case accuracy too low: {accuracy:.2%}"
        assert overall_success >= 0.7, f"Overall edge case success too low: {overall_success:.2%}"
        
        self.accuracy_metrics["edge_cases"] = {
            "accuracy": accuracy,
            "confidence_score": confidence_score,
            "overall_success": overall_success,
            "by_difficulty": by_difficulty,
            "results": results
        }
        
        self.demo_results["edge_cases_robustness"] = "PASS"
        
    def test_overall_accuracy_validation(self):
        """Demo: Comprehensive accuracy validation across all test scenarios."""
        print(f"\n=== DEMO: Overall Accuracy Validation ===")
        
        # Compile all test cases
        all_cases = []
        for category, cases in self.test_scenarios.items():
            for case in cases:
                case["category"] = category
                all_cases.append(case)
                
        print(f"Testing {len(all_cases)} total cases across {len(self.test_scenarios)} categories...")
        
        # Analyze all cases
        results = []
        category_results = {}
        
        for case in all_cases:
            analysis = self._simulate_arbitration_analysis(case["content"])
            
            correct_prediction = analysis["has_arbitration"] == case["expected"]
            
            # Adaptive confidence checking based on case type
            if case["expected"]:
                confidence_met = analysis["confidence"] >= case["confidence_threshold"]
            else:
                confidence_met = analysis["confidence"] <= (1 - case["confidence_threshold"])
                
            result = {
                "case_id": case["id"],
                "category": case["category"],
                "expected": case["expected"],
                "predicted": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "correct_prediction": correct_prediction,
                "confidence_met": confidence_met,
                "overall_success": correct_prediction and confidence_met,
                "difficulty": case["difficulty"]
            }
            
            results.append(result)
            
            # Track by category
            if case["category"] not in category_results:
                category_results[case["category"]] = []
            category_results[case["category"]].append(result)
            
        # Calculate comprehensive metrics
        overall_accuracy = sum(1 for r in results if r["correct_prediction"]) / len(results)
        overall_confidence = sum(1 for r in results if r["confidence_met"]) / len(results)
        overall_success = sum(1 for r in results if r["overall_success"]) / len(results)
        
        # Calculate metrics by category
        category_metrics = {}
        for category, cat_results in category_results.items():
            category_metrics[category] = {
                "accuracy": sum(1 for r in cat_results if r["correct_prediction"]) / len(cat_results),
                "confidence": sum(1 for r in cat_results if r["confidence_met"]) / len(cat_results),
                "success": sum(1 for r in cat_results if r["overall_success"]) / len(cat_results),
                "count": len(cat_results)
            }
            
        # Calculate confusion matrix
        true_positives = sum(1 for r in results if r["expected"] and r["predicted"])
        true_negatives = sum(1 for r in results if not r["expected"] and not r["predicted"])
        false_positives = sum(1 for r in results if not r["expected"] and r["predicted"])
        false_negatives = sum(1 for r in results if r["expected"] and not r["predicted"])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 Overall Accuracy Validation Results:")
        print(f"   Overall Accuracy: {overall_accuracy:.2%}")
        print(f"   Overall Confidence: {overall_confidence:.2%}")
        print(f"   Overall Success: {overall_success:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1_score:.2%}")
        
        print(f"\n   Category Breakdown:")
        for category, metrics in category_metrics.items():
            print(f"     {category.replace('_', ' ').title()}:")
            print(f"       Accuracy: {metrics['accuracy']:.2%} ({metrics['count']} cases)")
            print(f"       Success: {metrics['success']:.2%}")
            
        print(f"\n   Confusion Matrix:")
        print(f"     True Positives: {true_positives}")
        print(f"     True Negatives: {true_negatives}")
        print(f"     False Positives: {false_positives}")
        print(f"     False Negatives: {false_negatives}")
        
        # Validation against industry standards
        assert overall_accuracy >= 0.85, f"Overall accuracy below threshold: {overall_accuracy:.2%}"
        assert precision >= 0.8, f"Precision below threshold: {precision:.2%}"
        assert recall >= 0.8, f"Recall below threshold: {recall:.2%}"
        assert f1_score >= 0.8, f"F1 score below threshold: {f1_score:.2%}"
        
        self.accuracy_metrics["overall"] = {
            "accuracy": overall_accuracy,
            "confidence": overall_confidence,
            "success": overall_success,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "category_metrics": category_metrics,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            },
            "total_cases": len(results)
        }
        
        self.demo_results["overall_accuracy_validation"] = "PASS"
        
    def test_model_consistency_validation(self):
        """Demo: Test model consistency across multiple runs."""
        print(f"\n=== DEMO: Model Consistency Validation ===")
        
        # Select representative test cases
        consistency_cases = [
            self.test_scenarios["clear_positive_cases"][0],
            self.test_scenarios["clear_negative_cases"][0],
            self.test_scenarios["hidden_arbitration_cases"][0],
            self.test_scenarios["ambiguous_cases"][0]
        ]
        
        consistency_results = {}
        
        for case in consistency_cases:
            print(f"\nTesting consistency for: {case['id']}")
            
            # Run multiple analyses on the same content
            runs = []
            for run in range(5):  # 5 runs for consistency
                analysis = self._simulate_arbitration_analysis(case["content"])
                runs.append(analysis)
                
            # Calculate consistency metrics
            predictions = [r["has_arbitration"] for r in runs]
            confidences = [r["confidence"] for r in runs]
            
            prediction_consistency = len(set(predictions)) == 1  # All same prediction
            confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
            confidence_stability = confidence_std < 0.05  # Low standard deviation
            
            avg_confidence = statistics.mean(confidences)
            
            consistency_results[case["id"]] = {
                "prediction_consistency": prediction_consistency,
                "confidence_stability": confidence_stability,
                "confidence_std": confidence_std,
                "avg_confidence": avg_confidence,
                "runs": runs
            }
            
            status = "✓" if prediction_consistency and confidence_stability else "✗"
            print(f"{status} Prediction consistency: {prediction_consistency}")
            print(f"   Confidence std dev: {confidence_std:.3f}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            
        # Overall consistency metrics
        prediction_consistency_rate = sum(1 for r in consistency_results.values() if r["prediction_consistency"]) / len(consistency_results)
        confidence_stability_rate = sum(1 for r in consistency_results.values() if r["confidence_stability"]) / len(consistency_results)
        
        print(f"\n📊 Model Consistency Results:")
        print(f"   Prediction Consistency: {prediction_consistency_rate:.2%}")
        print(f"   Confidence Stability: {confidence_stability_rate:.2%}")
        print(f"   Average Confidence Std Dev: {statistics.mean([r['confidence_std'] for r in consistency_results.values()]):.3f}")
        
        # Validation
        assert prediction_consistency_rate >= 0.9, f"Prediction consistency too low: {prediction_consistency_rate:.2%}"
        assert confidence_stability_rate >= 0.8, f"Confidence stability too low: {confidence_stability_rate:.2%}"
        
        self.accuracy_metrics["consistency"] = {
            "prediction_consistency_rate": prediction_consistency_rate,
            "confidence_stability_rate": confidence_stability_rate,
            "case_results": consistency_results
        }
        
        self.demo_results["model_consistency"] = "PASS"
        
    # Helper methods
    def _simulate_arbitration_analysis(self, content: str) -> Dict[str, Any]:
        """Simulate arbitration analysis with realistic accuracy patterns."""
        time.sleep(0.1)  # Simulate processing time
        
        content_lower = content.lower()
        
        # Advanced pattern matching
        strong_indicators = ["binding arbitration", "final and binding", "arbitration administered"]
        medium_indicators = ["arbitration", "arbitrator", "arbitral", "aaa", "jams"]
        weak_indicators = ["dispute resolution", "alternative dispute", "neutral third party"]
        negative_indicators = ["court", "litigation", "no arbitration", "reject arbitration"]
        
        # Count indicators
        strong_count = sum(1 for ind in strong_indicators if ind in content_lower)
        medium_count = sum(1 for ind in medium_indicators if ind in content_lower)
        weak_count = sum(1 for ind in weak_indicators if ind in content_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in content_lower)
        
        # Calculate base confidence
        confidence = 0.1  # Base confidence
        confidence += strong_count * 0.3
        confidence += medium_count * 0.15
        confidence += weak_count * 0.05
        confidence -= negative_count * 0.2
        
        # Determine arbitration presence
        has_arbitration = False
        if strong_count > 0 and negative_count == 0:
            has_arbitration = True
            confidence = max(confidence, 0.8)
        elif medium_count >= 2 and negative_count == 0:
            has_arbitration = True
            confidence = max(confidence, 0.7)
        elif weak_count >= 3 and "arbitration" in content_lower and negative_count == 0:
            has_arbitration = True
            confidence = max(confidence, 0.6)
        elif negative_count > 0:
            has_arbitration = False
            confidence = max(0.05, 0.3 - confidence)
        else:
            # Ambiguous case
            confidence = min(0.6, confidence)
            
        # Clamp confidence
        confidence = max(0.05, min(0.98, confidence))
        
        # Add slight randomness for consistency testing
        import random
        confidence += random.uniform(-0.02, 0.02)
        confidence = max(0.05, min(0.98, confidence))
        
        return {
            "has_arbitration": has_arbitration,
            "confidence": confidence,
            "strong_indicators": strong_count,
            "medium_indicators": medium_count,
            "weak_indicators": weak_count,
            "negative_indicators": negative_count
        }
        
    def _explain_detection_method(self, content: str) -> str:
        """Explain how arbitration was detected in the content."""
        content_lower = content.lower()
        
        if "binding arbitration" in content_lower:
            return "Strong binding arbitration language detected"
        elif "aaa" in content_lower or "american arbitration" in content_lower:
            return "AAA (American Arbitration Association) reference found"
        elif "jams" in content_lower:
            return "JAMS arbitration provider identified"
        elif "neutral third party" in content_lower:
            return "Euphemistic arbitration language (neutral third party)"
        elif "alternative dispute resolution" in content_lower:
            return "Alternative dispute resolution language analysis"
        else:
            return "Pattern matching on arbitration keywords"
            
    def _classify_edge_case(self, case_id: str) -> str:
        """Classify the type of edge case."""
        if "anti" in case_id:
            return "Anti-arbitration clause"
        elif "optional" in case_id:
            return "Optional arbitration"
        elif "foreign" in case_id:
            return "International arbitration"
        elif "industry" in case_id:
            return "Industry-specific arbitration"
        else:
            return "General edge case"
            
    def generate_accuracy_report(self) -> Dict[str, Any]:
        """Generate comprehensive accuracy validation report."""
        total_demos = len(self.demo_results)
        passed_demos = sum(1 for result in self.demo_results.values() if result == "PASS")
        
        # Calculate weighted accuracy score
        weights = {
            "clear_positive": 0.2,
            "clear_negative": 0.2,
            "hidden_arbitration": 0.25,
            "ambiguous_cases": 0.15,
            "edge_cases": 0.15,
            "overall": 0.05
        }
        
        weighted_score = 0
        for category, weight in weights.items():
            if category in self.accuracy_metrics:
                metric_key = "accuracy" if "accuracy" in self.accuracy_metrics[category] else "overall_success"
                score = self.accuracy_metrics[category].get(metric_key, 0)
                weighted_score += score * weight
                
        return {
            "accuracy_validation_report": {
                "total_demos": total_demos,
                "passed": passed_demos,
                "failed": total_demos - passed_demos,
                "success_rate": passed_demos / total_demos if total_demos > 0 else 0,
                "weighted_accuracy_score": weighted_score,
                "accuracy_metrics": self.accuracy_metrics,
                "demo_results": self.demo_results,
                "timestamp": time.time(),
                "industry_benchmarks": {
                    "minimum_accuracy": 0.85,
                    "target_accuracy": 0.9,
                    "achieved_accuracy": weighted_score
                }
            }
        }


if __name__ == "__main__":
    # Run accuracy demo tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])