#!/usr/bin/env python3
"""
Comprehensive accuracy testing for Legal-BERT arbitration detection system.

This script tests the Legal-BERT detector and pattern matcher with various scenarios
to evaluate accuracy, identify false positives/negatives, and measure confidence scores.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from models.legal_bert_detector import LegalBERTDetector, DetectionResult
from models.pattern_matcher import ArbitrationPatternMatcher
from test_scenarios import get_all_test_scenarios, create_comprehensive_test_set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_detection_accuracy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AccuracyTester:
    """Comprehensive accuracy testing for arbitration detection."""
    
    def __init__(self):
        """Initialize the tester."""
        self.detector = None
        self.pattern_matcher = ArbitrationPatternMatcher()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'metrics': {},
            'false_positives': [],
            'false_negatives': [],
            'edge_cases': [],
            'summary': {}
        }
        
    def initialize_detector(self):
        """Initialize the Legal-BERT detector with error handling."""
        try:
            logger.info("Initializing Legal-BERT detector...")
            self.detector = LegalBERTDetector()
            logger.info("Legal-BERT detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Legal-BERT detector: {e}")
            return False
    
    def load_test_data(self) -> Dict[str, List[Dict]]:
        """Load positive and negative test examples."""
        test_data = {}
        
        # Load positive examples
        try:
            positive_path = Path(__file__).parent.parent.parent / 'data' / 'positive_examples.json'
            with open(positive_path, 'r') as f:
                positive_data = json.load(f)
                test_data['positive'] = positive_data.get('arbitration_clauses', [])
                logger.info(f"Loaded {len(test_data['positive'])} positive examples")
        except Exception as e:
            logger.error(f"Error loading positive examples: {e}")
            test_data['positive'] = []
        
        # Load negative examples
        try:
            negative_path = Path(__file__).parent.parent.parent / 'data' / 'negative_examples.json'
            with open(negative_path, 'r') as f:
                negative_data = json.load(f)
                test_data['negative'] = negative_data.get('non_arbitration_clauses', [])
                logger.info(f"Loaded {len(test_data['negative'])} negative examples")
        except Exception as e:
            logger.error(f"Error loading negative examples: {e}")
            test_data['negative'] = []
        
        # Add edge cases
        test_data['edge_cases'] = self.create_edge_cases()
        logger.info(f"Created {len(test_data['edge_cases'])} edge case examples")
        
        # Add comprehensive test scenarios
        try:
            comprehensive_scenarios = get_all_test_scenarios()
            for category_name, scenarios in comprehensive_scenarios.items():
                test_data[f'scenario_{category_name}'] = scenarios
                logger.info(f"Added {len(scenarios)} {category_name} scenarios")
        except Exception as e:
            logger.error(f"Error loading comprehensive test scenarios: {e}")
        
        return test_data
    
    def create_edge_cases(self) -> List[Dict]:
        """Create edge cases for testing."""
        return [
            {
                'id': 'edge_001',
                'text': 'The parties may choose to resolve disputes through arbitration if both agree.',
                'expected': False,  # Optional arbitration
                'description': 'Optional arbitration clause (should be negative)',
                'category': 'optional_arbitration'
            },
            {
                'id': 'edge_002',
                'text': 'This arbitration clause has been waived and is no longer applicable.',
                'expected': False,  # Waived arbitration
                'description': 'Waived arbitration clause',
                'category': 'waived_arbitration'
            },
            {
                'id': 'edge_003',
                'text': 'Any dispute shall be subject to the exclusive jurisdiction of courts, not arbitration.',
                'expected': False,  # Explicit rejection of arbitration
                'description': 'Explicit rejection of arbitration',
                'category': 'court_jurisdiction'
            },
            {
                'id': 'edge_004',
                'text': 'Arbitration is mentioned here but only as an example of alternative dispute resolution methods available.',
                'expected': False,  # Just mentioning arbitration
                'description': 'Mere mention of arbitration without requirement',
                'category': 'mention_only'
            },
            {
                'id': 'edge_005',
                'text': 'The arbitrator will decide all disputes arising under this agreement in accordance with applicable law.',
                'expected': True,  # Implicit mandatory arbitration
                'description': 'Implicit mandatory arbitration',
                'category': 'implicit_mandatory'
            },
            {
                'id': 'edge_006',
                'text': 'Mediation shall be attempted first, and if unsuccessful, binding arbitration shall follow under ICC rules.',
                'expected': True,  # Multi-step with arbitration
                'description': 'Multi-step dispute resolution with arbitration',
                'category': 'multi_step_adr'
            },
            {
                'id': 'edge_007',
                'text': 'Class actions are prohibited; all claims must be arbitrated individually.',
                'expected': True,  # Class action waiver implies arbitration
                'description': 'Class action waiver implying arbitration',
                'category': 'class_action_waiver'
            },
            {
                'id': 'edge_008',
                'text': 'The Federal Arbitration Act governs this agreement and supersedes state law.',
                'expected': True,  # FAA reference
                'description': 'Federal Arbitration Act reference',
                'category': 'faa_reference'
            },
            {
                'id': 'edge_009',
                'text': 'I understand that signing this agreement means I cannot sue in court and must use arbitration instead.',
                'expected': True,  # Clear arbitration requirement
                'description': 'Clear arbitration requirement with court waiver',
                'category': 'explicit_court_waiver'
            },
            {
                'id': 'edge_010',
                'text': 'Historical arbitration cases show that this type of dispute is typically resolved in court.',
                'expected': False,  # Academic discussion
                'description': 'Academic discussion of arbitration',
                'category': 'academic_discussion'
            },
            {
                'id': 'edge_011',
                'text': 'You have 30 days to opt out of this binding arbitration agreement by sending written notice to our legal department.',
                'expected': True,  # Mandatory with opt-out
                'description': 'Mandatory arbitration with opt-out provision',
                'category': 'opt_out_provision'
            },
            {
                'id': 'edge_012',
                'text': 'Small claims court actions are permitted, but all other disputes must be resolved through JAMS arbitration.',
                'expected': True,  # Exception clause but still mandatory
                'description': 'Mandatory arbitration with small claims exception',
                'category': 'small_claims_exception'
            },
            {
                'id': 'edge_013',
                'text': 'We recommend arbitration as a cost-effective alternative to litigation, but it is not required.',
                'expected': False,  # Recommendation only
                'description': 'Arbitration recommendation without requirement',
                'category': 'recommendation_only'
            },
            {
                'id': 'edge_014',
                'text': 'ARBITRATION AGREEMENT. BY USING OUR SERVICE, YOU AGREE TO RESOLVE ALL DISPUTES THROUGH BINDING ARBITRATION.',
                'expected': True,  # Capitalized emphasis
                'description': 'Emphasized arbitration agreement',
                'category': 'emphasized_mandatory'
            },
            {
                'id': 'edge_015',
                'text': 'This contract shall be governed by arbitration in accordance with the rules of the London Court of International Arbitration (LCIA).',
                'expected': True,  # International arbitration
                'description': 'International arbitration clause (LCIA)',
                'category': 'international_arbitration'
            },
            {
                'id': 'edge_016',
                'text': 'Disputes between parties may be resolved through mediation, litigation, or arbitration at the discretion of the complaining party.',
                'expected': False,  # Multiple options available
                'description': 'Multiple dispute resolution options',
                'category': 'multiple_options'
            },
            {
                'id': 'edge_017',
                'text': 'Any claims arising from this agreement shall be submitted to final and binding arbitration administered by the CPR Institute.',
                'expected': True,  # Binding with specific provider
                'description': 'CPR Institute arbitration clause',
                'category': 'cpr_arbitration'
            },
            {
                'id': 'edge_018',
                'text': 'Before pursuing arbitration, you must first attempt informal dispute resolution by contacting our customer service.',
                'expected': True,  # Pre-arbitration requirements
                'description': 'Pre-arbitration informal resolution requirement',
                'category': 'pre_arbitration_steps'
            },
            {
                'id': 'edge_019',
                'text': 'This arbitration provision survives termination of the agreement and remains in effect indefinitely.',
                'expected': True,  # Survival clause
                'description': 'Arbitration clause with survival provision',
                'category': 'survival_clause'
            },
            {
                'id': 'edge_020',
                'text': 'The parties agree that the arbitrator may not award punitive damages except where authorized by statute.',
                'expected': True,  # Arbitrator limitations
                'description': 'Arbitration with damages limitations',
                'category': 'damages_limitations'
            }
        ]
    
    def test_pattern_matching_only(self, test_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Test pattern matching separately from BERT detection."""
        logger.info("Testing pattern matching only...")
        
        pattern_results = {
            'positive': [],
            'negative': [],
            'edge_cases': []
        }
        
        for category, examples in test_data.items():
            logger.info(f"Testing pattern matching on {len(examples)} {category} examples")
            
            for example in examples:
                text = example['text']
                expected = example.get('expected', category == 'positive')
                
                # Test pattern matching
                pattern_result = self.pattern_matcher.match(text)
                
                result = {
                    'id': example['id'],
                    'expected': expected,
                    'pattern_confidence': pattern_result['confidence'],
                    'pattern_matches': pattern_result['matches'],
                    'predicted': pattern_result['confidence'] >= 0.7,
                    'correct': (pattern_result['confidence'] >= 0.7) == expected,
                    'description': example.get('description', '')
                }
                
                pattern_results[category].append(result)
        
        return pattern_results
    
    def test_bert_detection(self, test_data: Dict[str, List[Dict]], threshold: float = 0.7) -> Dict[str, Any]:
        """Test Legal-BERT detection with various thresholds."""
        if not self.detector:
            logger.error("BERT detector not initialized")
            return {}
        
        logger.info(f"Testing BERT detection with threshold {threshold}...")
        
        bert_results = {
            'positive': [],
            'negative': [],
            'edge_cases': []
        }
        
        for category, examples in test_data.items():
            logger.info(f"Testing BERT on {len(examples)} {category} examples")
            
            for example in examples:
                text = example['text']
                expected = example.get('expected', category == 'positive')
                
                try:
                    # Test BERT detection
                    detection = self.detector.detect(text, threshold)
                    
                    result = {
                        'id': example['id'],
                        'expected': expected,
                        'bert_confidence': detection.confidence,
                        'semantic_score': detection.semantic_score,
                        'predicted': detection.is_arbitration,
                        'correct': detection.is_arbitration == expected,
                        'pattern_matches': detection.pattern_matches,
                        'description': example.get('description', '')
                    }
                    
                    bert_results[category].append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing example {example['id']}: {e}")
                    bert_results[category].append({
                        'id': example['id'],
                        'error': str(e),
                        'expected': expected,
                        'predicted': False,
                        'correct': False
                    })
        
        return bert_results
    
    def calculate_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate accuracy metrics."""
        all_results = []
        for category_results in results.values():
            all_results.extend(category_results)
        
        if not all_results:
            return {}
        
        y_true = [r['expected'] for r in all_results if 'expected' in r]
        y_pred = [r['predicted'] for r in all_results if 'predicted' in r]
        
        if len(y_true) != len(y_pred):
            logger.warning("Mismatch in prediction arrays")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def identify_errors(self, results: Dict[str, List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
        """Identify false positives and false negatives."""
        false_positives = []
        false_negatives = []
        
        for category, category_results in results.items():
            for result in category_results:
                if not result.get('correct', True):
                    if result['predicted'] and not result['expected']:
                        # False positive
                        false_positives.append({
                            'id': result['id'],
                            'category': category,
                            'description': result.get('description', ''),
                            'confidence': result.get('bert_confidence', result.get('pattern_confidence', 0))
                        })
                    elif not result['predicted'] and result['expected']:
                        # False negative
                        false_negatives.append({
                            'id': result['id'],
                            'category': category,
                            'description': result.get('description', ''),
                            'confidence': result.get('bert_confidence', result.get('pattern_confidence', 0))
                        })
        
        return false_positives, false_negatives
    
    def test_threshold_analysis(self, test_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Test different confidence thresholds to find optimal balance."""
        if not self.detector:
            logger.warning("BERT detector not available for threshold analysis")
            return {}
        
        logger.info("Performing threshold analysis...")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_results = {}
        
        # Get all test examples
        all_examples = []
        for category, examples in test_data.items():
            for example in examples:
                example['category'] = category
                all_examples.append(example)
        
        # Test each threshold
        for threshold in thresholds:
            logger.info(f"Testing threshold: {threshold}")
            predictions = []
            actuals = []
            
            for example in all_examples:
                try:
                    result = self.detector.detect(example['text'], threshold)
                    predictions.append(result.is_arbitration)
                    actuals.append(example.get('expected', example['category'] == 'positive'))
                except Exception as e:
                    logger.error(f"Error testing threshold {threshold} on {example['id']}: {e}")
                    predictions.append(False)
                    actuals.append(example.get('expected', example['category'] == 'positive'))
            
            # Calculate metrics for this threshold
            try:
                metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy_score(actuals, predictions),
                    'precision': precision_score(actuals, predictions, zero_division=0),
                    'recall': recall_score(actuals, predictions, zero_division=0),
                    'f1_score': f1_score(actuals, predictions, zero_division=0)
                }
                threshold_results[threshold] = metrics
            except Exception as e:
                logger.error(f"Error calculating metrics for threshold {threshold}: {e}")
        
        return threshold_results
    
    def test_integration_accuracy(self, test_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Test integration between pattern matcher and BERT detector."""
        logger.info("Testing integration between pattern matcher and BERT detector...")
        
        if not self.detector:
            logger.warning("BERT detector not available for integration testing")
            return {}
        
        integration_results = {}
        all_examples = []
        for category, examples in test_data.items():
            for example in examples:
                all_examples.append({**example, 'category': category})
        
        # Test different combination strategies
        strategies = {
            'pattern_only': lambda p, b: p >= 0.7,
            'bert_only': lambda p, b: b >= 0.7,
            'max_confidence': lambda p, b: max(p, b) >= 0.7,
            'weighted_average': lambda p, b: (0.3 * p + 0.7 * b) >= 0.7,
            'conservative_and': lambda p, b: (p >= 0.6 and b >= 0.6),
            'liberal_or': lambda p, b: (p >= 0.8 or b >= 0.8)
        }
        
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"Testing strategy: {strategy_name}")
            predictions = []
            actuals = []
            
            for example in all_examples:
                try:
                    # Get pattern results
                    pattern_result = self.pattern_matcher.match(example['text'])
                    pattern_confidence = pattern_result['confidence']
                    
                    # Get BERT results
                    bert_result = self.detector.detect(example['text'])
                    bert_confidence = bert_result.semantic_score
                    
                    # Apply strategy
                    prediction = strategy_func(pattern_confidence, bert_confidence)
                    predictions.append(prediction)
                    actuals.append(example.get('expected', example['category'] == 'positive'))
                    
                except Exception as e:
                    logger.error(f"Error in integration test for {example['id']}: {e}")
                    predictions.append(False)
                    actuals.append(example.get('expected', example['category'] == 'positive'))
            
            # Calculate metrics
            try:
                metrics = {
                    'accuracy': accuracy_score(actuals, predictions),
                    'precision': precision_score(actuals, predictions, zero_division=0),
                    'recall': recall_score(actuals, predictions, zero_division=0),
                    'f1_score': f1_score(actuals, predictions, zero_division=0)
                }
                integration_results[strategy_name] = metrics
            except Exception as e:
                logger.error(f"Error calculating metrics for strategy {strategy_name}: {e}")
        
        return integration_results
    
    def analyze_category_performance(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Analyze performance by category/type of arbitration clause."""
        category_performance = {}
        
        for category, category_results in results.items():
            if not category_results:
                continue
                
            correct_predictions = sum(1 for r in category_results if r.get('correct', False))
            total_predictions = len(category_results)
            
            # Group by subcategory if available
            subcategories = {}
            for result in category_results:
                subcategory = result.get('category', 'unknown')
                if subcategory not in subcategories:
                    subcategories[subcategory] = {'correct': 0, 'total': 0}
                
                subcategories[subcategory]['total'] += 1
                if result.get('correct', False):
                    subcategories[subcategory]['correct'] += 1
            
            category_performance[category] = {
                'overall_accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
                'total_cases': total_predictions,
                'correct_cases': correct_predictions,
                'subcategories': {
                    subcat: {
                        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                        'cases': stats['total']
                    }
                    for subcat, stats in subcategories.items()
                }
            }
        
        return category_performance
    
    def plot_comprehensive_analysis(self, results: Dict[str, List[Dict]], 
                                   threshold_results: Dict[str, Any],
                                   integration_results: Dict[str, Any],
                                   output_dir: str):
        """Create comprehensive analysis plots."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Confidence distribution
            positive_confidences = []
            negative_confidences = []
            
            for category, category_results in results.items():
                for result in category_results:
                    confidence = result.get('bert_confidence', result.get('pattern_confidence', 0))
                    if result['expected']:
                        positive_confidences.append(confidence)
                    else:
                        negative_confidences.append(confidence)
            
            axes[0, 0].hist(positive_confidences, bins=15, alpha=0.7, label='Arbitration Clauses', color='green')
            axes[0, 0].hist(negative_confidences, bins=15, alpha=0.7, label='Non-Arbitration', color='red')
            axes[0, 0].set_xlabel('Confidence Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Confidence Score Distribution')
            axes[0, 0].legend()
            
            # 2. Threshold analysis
            if threshold_results:
                thresholds = list(threshold_results.keys())
                accuracies = [threshold_results[t]['accuracy'] for t in thresholds]
                precisions = [threshold_results[t]['precision'] for t in thresholds]
                recalls = [threshold_results[t]['recall'] for t in thresholds]
                
                axes[0, 1].plot(thresholds, accuracies, 'o-', label='Accuracy')
                axes[0, 1].plot(thresholds, precisions, 's-', label='Precision')
                axes[0, 1].plot(thresholds, recalls, '^-', label='Recall')
                axes[0, 1].set_xlabel('Threshold')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_title('Threshold Analysis')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Integration strategies comparison
            if integration_results:
                strategies = list(integration_results.keys())
                f1_scores = [integration_results[s]['f1_score'] for s in strategies]
                
                axes[0, 2].bar(range(len(strategies)), f1_scores, color='skyblue')
                axes[0, 2].set_xlabel('Integration Strategy')
                axes[0, 2].set_ylabel('F1 Score')
                axes[0, 2].set_title('Integration Strategy Performance')
                axes[0, 2].set_xticks(range(len(strategies)))
                axes[0, 2].set_xticklabels(strategies, rotation=45, ha='right')
            
            # 4. Confusion matrix
            if results:
                all_results = []
                for category_results in results.values():
                    all_results.extend(category_results)
                
                if all_results:
                    y_true = [r['expected'] for r in all_results if 'expected' in r]
                    y_pred = [r['predicted'] for r in all_results if 'predicted' in r]
                    
                    if len(y_true) == len(y_pred) and y_true:
                        cm = confusion_matrix(y_true, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
                        axes[1, 0].set_xlabel('Predicted')
                        axes[1, 0].set_ylabel('Actual')
                        axes[1, 0].set_title('Confusion Matrix')
            
            # 5. Performance by category
            category_performance = self.analyze_category_performance(results)
            if category_performance:
                categories = list(category_performance.keys())
                accuracies = [category_performance[c]['overall_accuracy'] for c in categories]
                
                axes[1, 1].bar(categories, accuracies, color='lightcoral')
                axes[1, 1].set_xlabel('Category')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].set_title('Performance by Category')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 6. Error analysis
            false_positives = []
            false_negatives = []
            
            for category, category_results in results.items():
                for result in category_results:
                    if not result.get('correct', True):
                        if result['predicted'] and not result['expected']:
                            false_positives.append(result.get('bert_confidence', result.get('pattern_confidence', 0)))
                        elif not result['predicted'] and result['expected']:
                            false_negatives.append(result.get('bert_confidence', result.get('pattern_confidence', 0)))
            
            if false_positives or false_negatives:
                axes[1, 2].hist(false_positives, bins=10, alpha=0.7, label=f'False Positives ({len(false_positives)})', color='red')
                axes[1, 2].hist(false_negatives, bins=10, alpha=0.7, label=f'False Negatives ({len(false_negatives)})', color='orange')
                axes[1, 2].set_xlabel('Confidence Score')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_title('Error Analysis')
                axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Comprehensive analysis plots saved")
            
        except Exception as e:
            logger.error(f"Error creating comprehensive analysis plots: {e}")
    
    def plot_confidence_distribution(self, results: Dict[str, List[Dict]], output_dir: str):
        """Plot confidence score distributions."""
        try:
            # Extract confidence scores
            positive_confidences = []
            negative_confidences = []
            
            for category, category_results in results.items():
                for result in category_results:
                    confidence = result.get('bert_confidence', result.get('pattern_confidence', 0))
                    if result['expected']:
                        positive_confidences.append(confidence)
                    else:
                        negative_confidences.append(confidence)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(positive_confidences, bins=20, alpha=0.7, label='True Positives', color='green')
            plt.hist(negative_confidences, bins=20, alpha=0.7, label='True Negatives', color='red')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Score Distribution')
            plt.legend()
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot([positive_confidences, negative_confidences], 
                       labels=['Arbitration Clauses', 'Non-Arbitration Text'])
            plt.ylabel('Confidence Score')
            plt.title('Confidence Score Box Plot')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Confidence distribution plot saved")
            
        except Exception as e:
            logger.error(f"Error creating confidence distribution plot: {e}")
    
    def generate_report(self, output_dir: str = "accuracy_test_results"):
        """Generate comprehensive accuracy report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_data = self.load_test_data()
        
        # Test pattern matching only
        logger.info("Testing pattern matching...")
        pattern_results = self.test_pattern_matching_only(test_data)
        pattern_metrics = self.calculate_metrics(pattern_results)
        pattern_fp, pattern_fn = self.identify_errors(pattern_results)
        
        self.results['tests']['pattern_matching'] = {
            'results': pattern_results,
            'metrics': pattern_metrics,
            'false_positives': pattern_fp,
            'false_negatives': pattern_fn
        }
        
        # Test BERT detection if available
        bert_available = self.initialize_detector()
        threshold_results = {}
        integration_results = {}
        
        if bert_available:
            logger.info("Testing BERT detection...")
            bert_results = self.test_bert_detection(test_data)
            bert_metrics = self.calculate_metrics(bert_results)
            bert_fp, bert_fn = self.identify_errors(bert_results)
            
            self.results['tests']['bert_detection'] = {
                'results': bert_results,
                'metrics': bert_metrics,
                'false_positives': bert_fp,
                'false_negatives': bert_fn
            }
            
            # Threshold analysis
            logger.info("Performing threshold analysis...")
            threshold_results = self.test_threshold_analysis(test_data)
            self.results['tests']['threshold_analysis'] = threshold_results
            
            # Integration testing
            logger.info("Testing integration strategies...")
            integration_results = self.test_integration_accuracy(test_data)
            self.results['tests']['integration_analysis'] = integration_results
            
            # Create comprehensive plots
            self.plot_comprehensive_analysis(bert_results, threshold_results, integration_results, output_dir)
            self.plot_confidence_distribution(bert_results, output_dir)
            
        else:
            logger.warning("BERT detection not available, skipping BERT tests")
        
        # Category performance analysis
        if self.results['tests']:
            category_performance = {}
            for test_type, test_results in self.results['tests'].items():
                if 'results' in test_results:
                    category_performance[test_type] = self.analyze_category_performance(test_results['results'])
            self.results['category_performance'] = category_performance
        
        # Generate summary
        self.results['summary'] = self.create_summary()
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'detailed_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate human-readable report
        self.generate_human_readable_report(output_dir, threshold_results, integration_results)
        
        logger.info(f"Comprehensive report generated in {output_dir}")
        return self.results
    
    def create_summary(self) -> Dict[str, Any]:
        """Create summary of test results."""
        summary = {
            'total_tests': 0,
            'pattern_matching': {},
            'bert_detection': {}
        }
        
        # Pattern matching summary
        if 'pattern_matching' in self.results['tests']:
            pm_results = self.results['tests']['pattern_matching']
            pm_metrics = pm_results['metrics']
            
            total_pattern_tests = sum(len(results) for results in pm_results['results'].values())
            summary['total_tests'] = total_pattern_tests
            
            summary['pattern_matching'] = {
                'accuracy': pm_metrics.get('accuracy', 0),
                'precision': pm_metrics.get('precision', 0),
                'recall': pm_metrics.get('recall', 0),
                'f1_score': pm_metrics.get('f1_score', 0),
                'false_positives': len(pm_results['false_positives']),
                'false_negatives': len(pm_results['false_negatives'])
            }
        
        # BERT detection summary
        if 'bert_detection' in self.results['tests']:
            bert_results = self.results['tests']['bert_detection']
            bert_metrics = bert_results['metrics']
            
            summary['bert_detection'] = {
                'accuracy': bert_metrics.get('accuracy', 0),
                'precision': bert_metrics.get('precision', 0),
                'recall': bert_metrics.get('recall', 0),
                'f1_score': bert_metrics.get('f1_score', 0),
                'false_positives': len(bert_results['false_positives']),
                'false_negatives': len(bert_results['false_negatives'])
            }
        
        return summary
    
    def generate_human_readable_report(self, output_dir: str, threshold_results: Dict = None, integration_results: Dict = None):
        """Generate a human-readable report."""
        report_file = os.path.join(output_dir, 'accuracy_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# Legal-BERT Arbitration Detection Accuracy Report\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            # Summary
            summary = self.results['summary']
            f.write("## Executive Summary\n\n")
            f.write(f"Total test cases: {summary['total_tests']}\n")
            f.write(f"Edge cases tested: {len(self.create_edge_cases())}\n\n")
            
            # Pattern matching results
            if 'pattern_matching' in summary:
                pm = summary['pattern_matching']
                f.write("### Pattern Matching Performance\n\n")
                f.write(f"- **Accuracy**: {pm['accuracy']:.1%}\n")
                f.write(f"- **Precision**: {pm['precision']:.1%}\n")
                f.write(f"- **Recall**: {pm['recall']:.1%}\n")
                f.write(f"- **F1 Score**: {pm['f1_score']:.1%}\n")
                f.write(f"- **False Positives**: {pm['false_positives']}\n")
                f.write(f"- **False Negatives**: {pm['false_negatives']}\n\n")
            
            # BERT detection results
            if 'bert_detection' in summary:
                bert = summary['bert_detection']
                f.write("### BERT Detection Performance\n\n")
                f.write(f"- **Accuracy**: {bert['accuracy']:.1%}\n")
                f.write(f"- **Precision**: {bert['precision']:.1%}\n")
                f.write(f"- **Recall**: {bert['recall']:.1%}\n")
                f.write(f"- **F1 Score**: {bert['f1_score']:.1%}\n")
                f.write(f"- **False Positives**: {bert['false_positives']}\n")
                f.write(f"- **False Negatives**: {bert['false_negatives']}\n\n")
            
            # Threshold analysis
            if threshold_results:
                f.write("## Threshold Analysis\n\n")
                f.write("Performance across different confidence thresholds:\n\n")
                f.write("| Threshold | Accuracy | Precision | Recall | F1 Score |\n")
                f.write("|-----------|----------|-----------|--------|----------|\n")
                
                for threshold in sorted(threshold_results.keys()):
                    metrics = threshold_results[threshold]
                    f.write(f"| {threshold:.1f} | {metrics['accuracy']:.1%} | {metrics['precision']:.1%} | {metrics['recall']:.1%} | {metrics['f1_score']:.1%} |\n")
                
                # Find optimal threshold
                optimal_threshold = max(threshold_results.keys(), 
                                      key=lambda t: threshold_results[t]['f1_score'])
                f.write(f"\n**Optimal threshold**: {optimal_threshold} (F1: {threshold_results[optimal_threshold]['f1_score']:.1%})\n\n")
            
            # Integration analysis
            if integration_results:
                f.write("## Integration Strategy Analysis\n\n")
                f.write("Performance of different pattern-BERT integration approaches:\n\n")
                f.write("| Strategy | Accuracy | Precision | Recall | F1 Score |\n")
                f.write("|----------|----------|-----------|--------|----------|\n")
                
                for strategy, metrics in integration_results.items():
                    f.write(f"| {strategy.replace('_', ' ').title()} | {metrics['accuracy']:.1%} | {metrics['precision']:.1%} | {metrics['recall']:.1%} | {metrics['f1_score']:.1%} |\n")
                
                # Find best strategy
                best_strategy = max(integration_results.keys(), 
                                  key=lambda s: integration_results[s]['f1_score'])
                f.write(f"\n**Best integration strategy**: {best_strategy.replace('_', ' ').title()} (F1: {integration_results[best_strategy]['f1_score']:.1%})\n\n")
            
            # Category performance
            if 'category_performance' in self.results:
                f.write("## Performance by Category\n\n")
                for test_type, categories in self.results['category_performance'].items():
                    f.write(f"### {test_type.replace('_', ' ').title()}\n\n")
                    for category, performance in categories.items():
                        f.write(f"- **{category.replace('_', ' ').title()}**: {performance['overall_accuracy']:.1%} accuracy ({performance['correct_cases']}/{performance['total_cases']} cases)\n")
                    f.write("\n")
            
            # Edge case analysis
            f.write("## Edge Case Analysis\n\n")
            edge_cases = self.create_edge_cases()
            edge_categories = {}
            for case in edge_cases:
                category = case['category']
                if category not in edge_categories:
                    edge_categories[category] = []
                edge_categories[category].append(case)
            
            for category, cases in edge_categories.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for case in cases:
                    f.write(f"- **{case['id']}**: {case['description']}\n")
                f.write("\n")
            
            # False positives analysis
            if 'pattern_matching' in self.results['tests']:
                pm_fp = self.results['tests']['pattern_matching']['false_positives']
                if pm_fp:
                    f.write("## Pattern Matching False Positives\n\n")
                    for fp in pm_fp:
                        f.write(f"- **{fp['id']}**: {fp['description']} (Confidence: {fp['confidence']:.3f})\n")
                    f.write("\n")
            
            if 'bert_detection' in self.results['tests']:
                bert_fp = self.results['tests']['bert_detection']['false_positives']
                if bert_fp:
                    f.write("## BERT Detection False Positives\n\n")
                    for fp in bert_fp:
                        f.write(f"- **{fp['id']}**: {fp['description']} (Confidence: {fp['confidence']:.3f})\n")
                    f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            self.add_recommendations(f, threshold_results, integration_results)
    
    def add_recommendations(self, f):
        """Add recommendations to the report."""
        summary = self.results['summary']
        
        if 'pattern_matching' in summary:
            pm_accuracy = summary['pattern_matching']['accuracy']
            if pm_accuracy < 0.8:
                f.write("- **Pattern Matching**: Consider refining regex patterns and keyword weights\n")
        
        if 'bert_detection' in summary:
            bert_accuracy = summary['bert_detection']['accuracy']
            if bert_accuracy < 0.85:
                f.write("- **BERT Detection**: Consider fine-tuning the model with more training data\n")
            
            # Compare pattern vs BERT
            pm_accuracy = summary.get('pattern_matching', {}).get('accuracy', 0)
            if bert_accuracy > pm_accuracy + 0.1:
                f.write("- **Hybrid Approach**: BERT significantly outperforms pattern matching alone\n")
            elif pm_accuracy > bert_accuracy + 0.1:
                f.write("- **Pattern Focus**: Pattern matching outperforms BERT, consider pattern-first approach\n")
        
        f.write("- **Threshold Tuning**: Experiment with different confidence thresholds for optimal precision/recall balance\n")
        f.write("- **Training Data**: Collect more edge cases for model improvement\n")

def main():
    """Main function to run accuracy tests."""
    logger.info("Starting Legal-BERT detection accuracy testing...")
    
    tester = AccuracyTester()
    
    # Generate comprehensive report
    results = tester.generate_report()
    
    # Print summary to console
    print("\n" + "="*60)
    print("LEGAL-BERT DETECTION ACCURACY TEST RESULTS")
    print("="*60)
    
    summary = results['summary']
    print(f"Total test cases: {summary['total_tests']}")
    
    if 'pattern_matching' in summary:
        pm = summary['pattern_matching']
        print(f"\nPattern Matching:")
        print(f"  Accuracy:  {pm['accuracy']:.1%}")
        print(f"  Precision: {pm['precision']:.1%}")
        print(f"  Recall:    {pm['recall']:.1%}")
        print(f"  F1 Score:  {pm['f1_score']:.1%}")
    
    if 'bert_detection' in summary:
        bert = summary['bert_detection']
        print(f"\nBERT Detection:")
        print(f"  Accuracy:  {bert['accuracy']:.1%}")
        print(f"  Precision: {bert['precision']:.1%}")
        print(f"  Recall:    {bert['recall']:.1%}")
        print(f"  F1 Score:  {bert['f1_score']:.1%}")
    
    print(f"\nDetailed results saved in: accuracy_test_results/")
    print("="*60)
    
    logger.info("Accuracy testing completed successfully")

if __name__ == "__main__":
    main()