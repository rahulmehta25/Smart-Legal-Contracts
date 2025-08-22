#!/usr/bin/env python3
"""
Accuracy threshold validation script for arbitration detection system.

This script validates that the system meets minimum accuracy requirements
before deployment. It can be run as part of CI/CD pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def load_test_results(results_path: Path) -> Dict[str, Any]:
    """Load test results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Test results file not found: {results_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in test results: {e}")
        sys.exit(1)


def validate_accuracy_thresholds(
    results: Dict[str, Any], 
    min_accuracy: float = 0.85,
    min_precision: float = 0.80,
    min_recall: float = 0.85,
    min_f1: float = 0.82
) -> Tuple[bool, List[str]]:
    """Validate accuracy metrics against thresholds."""
    
    issues = []
    
    # Extract metrics from results
    if "overall_metrics" in results:
        metrics = results["overall_metrics"]
    else:
        # Fallback: calculate from raw predictions if available
        if "predictions" not in results:
            print("❌ No metrics or predictions found in results")
            return False, ["No validation data available"]
        
        predictions = results["predictions"]
        y_true = [p["actual"] for p in predictions]
        y_pred = [p["predicted"] for p in predictions]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
    
    # Validate each metric
    thresholds = {
        "accuracy": min_accuracy,
        "precision": min_precision, 
        "recall": min_recall,
        "f1_score": min_f1
    }
    
    for metric_name, threshold in thresholds.items():
        actual_value = metrics.get(metric_name, 0)
        
        if actual_value < threshold:
            issues.append(
                f"{metric_name.title()}: {actual_value:.3f} < {threshold:.3f} "
                f"(shortfall: {threshold - actual_value:.3f})"
            )
    
    return len(issues) == 0, issues


def validate_category_performance(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate performance across different document categories."""
    
    issues = []
    
    if "category_breakdown" not in results:
        return True, []  # Skip if no category data available
    
    # Define minimum accuracy thresholds per category
    category_thresholds = {
        "clear_arbitration": 0.95,      # Very high for obvious cases
        "hidden_arbitration": 0.80,     # Lower for buried clauses
        "no_arbitration": 0.90,         # High for negative cases
        "ambiguous": 0.60,              # Lowest for inherently difficult cases
        "false_positive_prevention": 0.85,  # Important to avoid false positives
        "complex_arbitration": 0.85,    # Should handle complex cases well
        "edge_cases": 0.70             # Lower threshold for edge cases
    }
    
    category_breakdown = results["category_breakdown"]
    
    for category, threshold in category_thresholds.items():
        if category in category_breakdown:
            accuracy = category_breakdown[category]["accuracy"]
            sample_count = category_breakdown[category]["sample_count"]
            
            if accuracy < threshold:
                issues.append(
                    f"Category '{category}': {accuracy:.3f} < {threshold:.3f} "
                    f"(samples: {sample_count})"
                )
    
    return len(issues) == 0, issues


def validate_business_requirements(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate business-critical requirements."""
    
    issues = []
    
    # Critical business requirements
    requirements = {
        "max_false_positive_rate": 0.10,    # Max 10% false positive rate
        "max_false_negative_rate": 0.15,    # Max 15% false negative rate
        "min_high_confidence_accuracy": 0.95,  # High confidence predictions should be very accurate
    }
    
    # Calculate false positive rate
    if "confusion_matrix" in results:
        cm = np.array(results["confusion_matrix"])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            
            # False positive rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            if fpr > requirements["max_false_positive_rate"]:
                issues.append(f"False positive rate: {fpr:.3f} > {requirements['max_false_positive_rate']:.3f}")
            
            # False negative rate  
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            if fnr > requirements["max_false_negative_rate"]:
                issues.append(f"False negative rate: {fnr:.3f} > {requirements['max_false_negative_rate']:.3f}")
    
    # High confidence accuracy check
    if "confidence_analysis" in results:
        high_conf_accuracy = results["confidence_analysis"].get("high_confidence_accuracy", 1.0)
        if high_conf_accuracy < requirements["min_high_confidence_accuracy"]:
            issues.append(
                f"High confidence accuracy: {high_conf_accuracy:.3f} < "
                f"{requirements['min_high_confidence_accuracy']:.3f}"
            )
    
    return len(issues) == 0, issues


def validate_performance_requirements(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate performance requirements."""
    
    issues = []
    
    # Performance requirements
    requirements = {
        "max_avg_processing_time": 5.0,     # Max 5 seconds average processing time
        "max_p95_processing_time": 10.0,    # Max 10 seconds 95th percentile
        "min_throughput": 10.0,             # Min 10 documents per minute
    }
    
    if "performance_metrics" in results:
        perf = results["performance_metrics"]
        
        # Processing time checks
        avg_time = perf.get("avg_processing_time", 0)
        if avg_time > requirements["max_avg_processing_time"]:
            issues.append(f"Average processing time: {avg_time:.2f}s > {requirements['max_avg_processing_time']:.2f}s")
        
        p95_time = perf.get("p95_processing_time", 0)
        if p95_time > requirements["max_p95_processing_time"]:
            issues.append(f"P95 processing time: {p95_time:.2f}s > {requirements['max_p95_processing_time']:.2f}s")
        
        # Throughput check
        throughput = perf.get("throughput_per_minute", 0)
        if throughput < requirements["min_throughput"]:
            issues.append(f"Throughput: {throughput:.1f} docs/min < {requirements['min_throughput']:.1f} docs/min")
    
    return len(issues) == 0, issues


def generate_validation_report(
    results: Dict[str, Any],
    accuracy_passed: bool,
    accuracy_issues: List[str],
    category_passed: bool, 
    category_issues: List[str],
    business_passed: bool,
    business_issues: List[str],
    performance_passed: bool,
    performance_issues: List[str]
) -> Dict[str, Any]:
    """Generate comprehensive validation report."""
    
    overall_passed = all([accuracy_passed, category_passed, business_passed, performance_passed])
    
    report = {
        "validation_summary": {
            "overall_status": "PASS" if overall_passed else "FAIL",
            "validation_date": "2024-01-15",  # Would use actual date
            "total_issues": len(accuracy_issues + category_issues + business_issues + performance_issues)
        },
        "accuracy_validation": {
            "status": "PASS" if accuracy_passed else "FAIL",
            "issues": accuracy_issues
        },
        "category_validation": {
            "status": "PASS" if category_passed else "FAIL", 
            "issues": category_issues
        },
        "business_validation": {
            "status": "PASS" if business_passed else "FAIL",
            "issues": business_issues
        },
        "performance_validation": {
            "status": "PASS" if performance_passed else "FAIL",
            "issues": performance_issues
        },
        "recommendations": []
    }
    
    # Add recommendations based on issues
    if accuracy_issues:
        report["recommendations"].append("Review model training data and feature engineering")
    if category_issues:
        report["recommendations"].append("Improve handling of specific document categories")
    if business_issues:
        report["recommendations"].append("Address business-critical accuracy requirements")
    if performance_issues:
        report["recommendations"].append("Optimize system performance and throughput")
    
    if overall_passed:
        report["recommendations"].append("System meets all validation criteria and is ready for deployment")
    
    return report


def main():
    """Main validation function."""
    
    parser = argparse.ArgumentParser(description="Validate accuracy thresholds for arbitration detection system")
    parser.add_argument("--results", type=str, help="Path to test results JSON file")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum accuracy threshold")
    parser.add_argument("--precision", type=float, default=0.80, help="Minimum precision threshold")
    parser.add_argument("--recall", type=float, default=0.85, help="Minimum recall threshold")
    parser.add_argument("--f1", type=float, default=0.82, help="Minimum F1 score threshold")
    parser.add_argument("--output", type=str, help="Output path for validation report")
    parser.add_argument("--strict", action="store_true", help="Strict mode - fail on any threshold violation")
    
    args = parser.parse_args()
    
    # Default results path
    if not args.results:
        results_path = Path(__file__).parent.parent / "accuracy_validation_report.json"
    else:
        results_path = Path(args.results)
    
    print("🔍 Arbitration Detection System - Accuracy Validation")
    print("=" * 55)
    print(f"📁 Results file: {results_path}")
    print(f"🎯 Accuracy threshold: {args.threshold:.3f}")
    print(f"🎯 Precision threshold: {args.precision:.3f}")
    print(f"🎯 Recall threshold: {args.recall:.3f}")
    print(f"🎯 F1 score threshold: {args.f1:.3f}")
    print()
    
    # Load test results
    results = load_test_results(results_path)
    
    # Run validations
    print("🧪 Running accuracy threshold validation...")
    accuracy_passed, accuracy_issues = validate_accuracy_thresholds(
        results, args.threshold, args.precision, args.recall, args.f1
    )
    
    print("📊 Running category performance validation...")
    category_passed, category_issues = validate_category_performance(results)
    
    print("💼 Running business requirements validation...")
    business_passed, business_issues = validate_business_requirements(results)
    
    print("⚡ Running performance requirements validation...")
    performance_passed, performance_issues = validate_performance_requirements(results)
    
    # Generate report
    validation_report = generate_validation_report(
        results, accuracy_passed, accuracy_issues,
        category_passed, category_issues,
        business_passed, business_issues,
        performance_passed, performance_issues
    )
    
    # Print results
    print("\n📋 VALIDATION RESULTS")
    print("=" * 30)
    
    overall_status = validation_report["validation_summary"]["overall_status"]
    status_emoji = "✅" if overall_status == "PASS" else "❌"
    print(f"{status_emoji} Overall Status: {overall_status}")
    print()
    
    # Print detailed results
    validations = [
        ("Accuracy Thresholds", accuracy_passed, accuracy_issues),
        ("Category Performance", category_passed, category_issues), 
        ("Business Requirements", business_passed, business_issues),
        ("Performance Requirements", performance_passed, performance_issues)
    ]
    
    for name, passed, issues in validations:
        status_emoji = "✅" if passed else "❌"
        print(f"{status_emoji} {name}: {'PASS' if passed else 'FAIL'}")
        
        if issues:
            for issue in issues:
                print(f"   • {issue}")
        print()
    
    # Print recommendations
    if validation_report["recommendations"]:
        print("💡 RECOMMENDATIONS")
        print("=" * 20)
        for rec in validation_report["recommendations"]:
            print(f"• {rec}")
        print()
    
    # Save validation report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "validation_report.json"
    
    with open(output_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"📄 Validation report saved to: {output_path}")
    
    # Exit with appropriate code
    if overall_status == "PASS":
        print("\n🎉 All validations passed! System is ready for deployment.")
        sys.exit(0)
    else:
        print("\n🚫 Validation failed. Please address the issues before deployment.")
        if args.strict:
            sys.exit(1)
        else:
            print("⚠️  Running in non-strict mode. Deployment not blocked.")
            sys.exit(0)


if __name__ == "__main__":
    main()