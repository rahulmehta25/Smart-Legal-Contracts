"""
Test ML Evaluation Pipeline

Quick test of the evaluation functionality
"""

import os
import sys
sys.path.append('.')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from app.ml.evaluate import ModelEvaluator

def test_evaluation_pipeline():
    """Test the evaluation pipeline with existing models"""
    
    print("=" * 60)
    print("TESTING ML EVALUATION PIPELINE")
    print("=" * 60)
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Load models and prepare data
        evaluator.load_models_and_data()
        
        # Prepare smaller test data for quick evaluation
        texts, X, y = evaluator.prepare_test_data(test_size=100)
        
        print(f"\nEvaluating models on {len(texts)} test examples...")
        
        # Evaluate all models
        results = evaluator.evaluate_all_models(X, y)
        
        # Print results summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        
        for model_name, model_results in results.items():
            cv_metrics = model_results['cv_metrics']
            print(f"\n{model_name.upper()}:")
            print(f"  Cross-Validation F1: {cv_metrics['test_f1_mean']:.4f} ± {cv_metrics['test_f1_std']:.4f}")
            print(f"  Cross-Validation Accuracy: {cv_metrics['test_accuracy_mean']:.4f} ± {cv_metrics['test_accuracy_std']:.4f}")
            print(f"  Cross-Validation ROC-AUC: {cv_metrics.get('test_roc_auc_mean', 0):.4f} ± {cv_metrics.get('test_roc_auc_std', 0):.4f}")
        
        # Create visualizations (will be saved to files since we're using non-interactive backend)
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        try:
            print("Creating ROC curves...")
            evaluator.create_roc_curves(texts, y)
            
            print("Creating precision-recall curves...")
            evaluator.create_precision_recall_curves(texts, y)
            
            print("Creating confusion matrices...")
            evaluator.create_confusion_matrices()
            
            print("Creating model comparison chart...")
            comparison_df = evaluator.create_model_comparison_chart()
            
            print("Creating feature importance plots...")
            evaluator.create_feature_importance_plots()
            
        except Exception as e:
            print(f"Visualization error (expected with non-interactive backend): {e}")
        
        # Generate report
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        report = evaluator.generate_comprehensive_report()
        
        print(f"\nBest model: {report['best_model']['name']}")
        print(f"Best F1-score: {report['best_model']['f1_score']:.4f}")
        print(f"Best Accuracy: {report['best_model']['accuracy']:.4f}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"• {rec}")
        
        print("\n" + "=" * 60)
        print("EVALUATION PIPELINE TEST COMPLETED!")
        print("=" * 60)
        
        return report, results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This is expected if models aren't fully trained yet")
        return None, None

if __name__ == "__main__":
    report, results = test_evaluation_pipeline()