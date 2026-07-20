#!/usr/bin/env python3
"""
Simplified accuracy testing for Legal-BERT arbitration detection system.
This version doesn't require sklearn and can run with basic Python libraries.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleAccuracyTester:
    """Simplified accuracy testing for arbitration detection."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        
        # Initialize pattern matcher (doesn't require heavy dependencies)
        try:
            from models.pattern_matcher import ArbitrationPatternMatcher
            self.pattern_matcher = ArbitrationPatternMatcher()
            logger.info("Pattern matcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pattern matcher: {e}")
            self.pattern_matcher = None
    
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
                'text': 'Any dispute shall be subject to the exclusive jurisdiction of courts, not arbitration.',
                'expected': False,  # Explicit rejection of arbitration
                'description': 'Explicit rejection of arbitration',
                'category': 'court_jurisdiction'
            },
            {
                'id': 'edge_003',
                'text': 'The arbitrator will decide all disputes arising under this agreement in accordance with applicable law.',
                'expected': True,  # Implicit mandatory arbitration
                'description': 'Implicit mandatory arbitration',
                'category': 'implicit_mandatory'
            },
            {
                'id': 'edge_004',
                'text': 'Class actions are prohibited; all claims must be arbitrated individually.',
                'expected': True,  # Class action waiver implies arbitration
                'description': 'Class action waiver implying arbitration', 
                'category': 'class_action_waiver'
            },
            {
                'id': 'edge_005',
                'text': 'You have 30 days to opt out of this binding arbitration agreement by sending written notice to our legal department.',
                'expected': True,  # Mandatory with opt-out
                'description': 'Mandatory arbitration with opt-out provision',
                'category': 'opt_out_provision'
            }
        ]
    
    def test_pattern_matching_only(self, test_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Test pattern matching separately."""
        if not self.pattern_matcher:
            logger.error("Pattern matcher not available")
            return {}
            
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
                
                try:
                    # Test pattern matching
                    pattern_result = self.pattern_matcher.match(text)
                    
                    predicted = pattern_result['confidence'] >= 0.7
                    correct = predicted == expected
                    
                    result = {
                        'id': example['id'],
                        'expected': expected,
                        'predicted': predicted,
                        'correct': correct,
                        'pattern_confidence': pattern_result['confidence'],
                        'pattern_matches': pattern_result['matches'],
                        'description': example.get('description', ''),
                        'category': example.get('category', category)
                    }
                    
                    pattern_results[category].append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing example {example['id']}: {e}")
                    pattern_results[category].append({
                        'id': example['id'],
                        'error': str(e),
                        'expected': expected,
                        'predicted': False,
                        'correct': False
                    })
        
        return pattern_results
    
    def calculate_simple_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate simple accuracy metrics without sklearn."""
        all_results = []
        for category_results in results.values():
            all_results.extend([r for r in category_results if 'error' not in r])
        
        if not all_results:
            return {}
        
        # Calculate basic metrics
        total = len(all_results)
        correct = sum(1 for r in all_results if r.get('correct', False))
        
        # True positives, false positives, etc.
        tp = sum(1 for r in all_results if r.get('predicted', False) and r.get('expected', False))
        fp = sum(1 for r in all_results if r.get('predicted', False) and not r.get('expected', False))
        tn = sum(1 for r in all_results if not r.get('predicted', False) and not r.get('expected', False))
        fn = sum(1 for r in all_results if not r.get('predicted', False) and r.get('expected', False))
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # True positive rate and false positive rate
        tpr = recall  # Same as recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'total_cases': total,
            'correct_predictions': correct,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def analyze_errors(self, results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Analyze false positives and false negatives."""
        false_positives = []
        false_negatives = []
        
        for category, category_results in results.items():
            for result in category_results:
                if 'error' in result:
                    continue
                    
                if not result.get('correct', True):
                    error_info = {
                        'id': result['id'],
                        'category': category,
                        'description': result.get('description', ''),
                        'confidence': result.get('pattern_confidence', 0),
                        'text_preview': result.get('text', '')[:100] + '...' if 'text' in result else ''
                    }
                    
                    if result['predicted'] and not result['expected']:
                        # False positive
                        false_positives.append(error_info)
                    elif not result['predicted'] and result['expected']:
                        # False negative  
                        false_negatives.append(error_info)
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def generate_report(self, output_dir: str = "simple_accuracy_results"):
        """Generate simplified accuracy report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_data = self.load_test_data()
        
        # Test pattern matching
        pattern_results = self.test_pattern_matching_only(test_data)
        
        if not pattern_results:
            logger.error("No pattern matching results available")
            return {}
        
        # Calculate metrics
        metrics = self.calculate_simple_metrics(pattern_results)
        
        # Analyze errors
        error_analysis = self.analyze_errors(pattern_results)
        
        # Store results
        self.results['tests']['pattern_matching'] = {
            'results': pattern_results,
            'metrics': metrics,
            **error_analysis
        }
        
        # Generate summary
        self.results['summary'] = {
            'total_tests': metrics.get('total_cases', 0),
            'pattern_matching': metrics
        }
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'simple_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate human-readable report
        self.generate_text_report(output_dir)
        
        logger.info(f"Simple accuracy report generated in {output_dir}")
        return self.results
    
    def generate_text_report(self, output_dir: str):
        """Generate a simple text report."""
        report_file = os.path.join(output_dir, 'accuracy_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("LEGAL-BERT ARBITRATION DETECTION - SIMPLE ACCURACY REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            if 'pattern_matching' in self.results['tests']:
                metrics = self.results['tests']['pattern_matching']['metrics']
                
                f.write("PATTERN MATCHING PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total test cases: {metrics.get('total_cases', 0)}\n")
                f.write(f"Correct predictions: {metrics.get('correct_predictions', 0)}\n")
                f.write(f"Accuracy: {metrics.get('accuracy', 0):.1%}\n")
                f.write(f"Precision: {metrics.get('precision', 0):.1%}\n")
                f.write(f"Recall: {metrics.get('recall', 0):.1%}\n")
                f.write(f"F1 Score: {metrics.get('f1_score', 0):.1%}\n")
                f.write(f"True Positive Rate: {metrics.get('true_positive_rate', 0):.1%}\n")
                f.write(f"False Positive Rate: {metrics.get('false_positive_rate', 0):.1%}\n\n")
                
                f.write("DETAILED BREAKDOWN\n")
                f.write("-" * 18 + "\n")
                f.write(f"True Positives: {metrics.get('true_positives', 0)}\n")
                f.write(f"False Positives: {metrics.get('false_positives', 0)}\n")
                f.write(f"True Negatives: {metrics.get('true_negatives', 0)}\n")
                f.write(f"False Negatives: {metrics.get('false_negatives', 0)}\n\n")
                
                # Error analysis
                errors = self.results['tests']['pattern_matching']
                if errors.get('false_positives'):
                    f.write("FALSE POSITIVES (incorrectly detected as arbitration)\n")
                    f.write("-" * 50 + "\n")
                    for fp in errors['false_positives']:
                        f.write(f"ID: {fp['id']}\n")
                        f.write(f"Description: {fp['description']}\n")
                        f.write(f"Confidence: {fp['confidence']:.3f}\n")
                        f.write("\n")
                
                if errors.get('false_negatives'):
                    f.write("FALSE NEGATIVES (missed arbitration clauses)\n")
                    f.write("-" * 45 + "\n")
                    for fn in errors['false_negatives']:
                        f.write(f"ID: {fn['id']}\n")
                        f.write(f"Description: {fn['description']}\n")
                        f.write(f"Confidence: {fn['confidence']:.3f}\n")
                        f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                
                accuracy = metrics.get('accuracy', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                fpr = metrics.get('false_positive_rate', 0)
                
                if accuracy < 0.8:
                    f.write("- Overall accuracy is below 80%. Consider refining pattern matching rules.\n")
                
                if precision < 0.8:
                    f.write("- Precision is low. Reduce false positives by tightening patterns.\n")
                
                if recall < 0.8:
                    f.write("- Recall is low. Add more comprehensive patterns to catch missed clauses.\n")
                
                if fpr > 0.2:
                    f.write("- False positive rate is high. Review patterns that trigger on non-arbitration text.\n")
                
                f.write("- Test with Legal-BERT for semantic understanding improvements.\n")
                f.write("- Collect more diverse training examples for better coverage.\n")
                f.write("- Consider different confidence thresholds based on use case requirements.\n")

def main():
    """Main function to run simple accuracy tests."""
    logger.info("Starting simple arbitration detection accuracy testing...")
    
    tester = SimpleAccuracyTester()
    
    # Generate report
    results = tester.generate_report()
    
    # Print summary to console
    if results and 'summary' in results:
        print("\n" + "="*60)
        print("ARBITRATION DETECTION ACCURACY TEST RESULTS")
        print("="*60)
        
        summary = results['summary']
        print(f"Total test cases: {summary.get('total_tests', 0)}")
        
        if 'pattern_matching' in summary:
            pm = summary['pattern_matching']
            print(f"\nPattern Matching Performance:")
            print(f"  Accuracy:  {pm.get('accuracy', 0):.1%}")
            print(f"  Precision: {pm.get('precision', 0):.1%}")
            print(f"  Recall:    {pm.get('recall', 0):.1%}")
            print(f"  F1 Score:  {pm.get('f1_score', 0):.1%}")
            print(f"  True Positive Rate:  {pm.get('true_positive_rate', 0):.1%}")
            print(f"  False Positive Rate: {pm.get('false_positive_rate', 0):.1%}")
        
        print(f"\nDetailed results saved in: simple_accuracy_results/")
        print("="*60)
    
    logger.info("Simple accuracy testing completed successfully")

if __name__ == "__main__":
    main()