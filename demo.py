#!/usr/bin/env python3
"""
Demo script for the Arbitration Clause Detection RAG System

This script demonstrates the core functionality of the system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.arbitration_detector import ArbitrationDetectionPipeline
from comparison.comparison_engine import ClauseComparisonEngine
from explainability.explainer import ArbitrationExplainer

def demo_detection():
    """Demonstrate arbitration clause detection"""
    print("🔍 Arbitration Clause Detection Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ArbitrationDetectionPipeline(cache_enabled=False)
    
    # Test with sample document
    sample_path = "tests/fixtures/sample_tos.txt"
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample document not found: {sample_path}")
        print("Please run the tests first to generate sample documents.")
        return
    
    print(f"📄 Analyzing document: {sample_path}")
    
    # Detect arbitration clause
    result = pipeline.detect_arbitration_clause(sample_path)
    
    if result:
        print(f"✅ Arbitration clause detected!")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Type: {result.clause_type}")
        print(f"   Location: {result.location['section_title']}")
        print(f"   Key provisions: {', '.join(result.key_provisions)}")
        print(f"   Summary: {result.summary[:100]}...")
    else:
        print("❌ No arbitration clause detected")

def demo_comparison():
    """Demonstrate clause comparison"""
    print("\n🔍 Clause Comparison Demo")
    print("=" * 50)
    
    # Initialize comparison engine
    engine = ClauseComparisonEngine()
    
    # Add a sample clause to the database
    sample_clause = {
        'text': 'Any dispute shall be resolved through binding arbitration under AAA rules.',
        'company': 'Demo Corp',
        'industry': 'Technology',
        'doc_type': 'TOS',
        'summary': 'Standard arbitration clause with AAA',
        'provisions': ['Binding arbitration', 'AAA rules'],
        'enforceability': 0.8,
        'risk': 0.6,
        'jurisdiction': 'US'
    }
    
    print("📝 Adding sample clause to database...")
    clause_id = engine.add_clause_to_database(sample_clause)
    print(f"   Added clause with ID: {clause_id}")
    
    # Compare with another clause
    test_clause = "Disputes will be settled by binding arbitration administered by JAMS."
    print(f"\n📊 Comparing clause: {test_clause}")
    
    comparison = engine.compare_clause(test_clause)
    
    print(f"   Risk assessment: {comparison['analysis']['risk_assessment']}")
    print(f"   Similar clauses found: {len(comparison['similar_clauses'])}")
    
    if comparison['analysis']['recommendations']:
        print("   Recommendations:")
        for rec in comparison['analysis']['recommendations']:
            print(f"     • {rec}")

def demo_explainability():
    """Demonstrate explainability features"""
    print("\n🔍 Explainability Demo")
    print("=" * 50)
    
    # Initialize components
    pipeline = ArbitrationDetectionPipeline(cache_enabled=False)
    explainer = ArbitrationExplainer(pipeline.bert_detector)
    
    # Test text
    test_text = "Any dispute arising from this agreement shall be finally settled by binding arbitration."
    
    print(f"📝 Analyzing text: {test_text}")
    
    # Get detection result
    detection_result = pipeline.bert_detector.detect(test_text)
    
    # Generate explanation
    explanation = explainer.explain_detection(test_text, detection_result)
    
    print(f"   Overall confidence: {explanation['confidence_breakdown']['overall_confidence']:.1%}")
    print(f"   Semantic confidence: {explanation['confidence_breakdown']['semantic_confidence']:.1%}")
    
    print("   Decision path:")
    for step in explanation['decision_path']:
        print(f"     • {step}")
    
    if explanation['key_indicators']:
        print("   Key indicators:")
        for indicator in explanation['key_indicators'][:3]:
            print(f"     • {indicator['description']}")

def main():
    """Run all demos"""
    print("🚀 Arbitration Clause Detection RAG System Demo")
    print("=" * 60)
    
    try:
        demo_detection()
        demo_comparison()
        demo_explainability()
        
        print("\n✅ Demo completed successfully!")
        print("\nTo run the full system:")
        print("  • API: uvicorn src.api.main:app --reload")
        print("  • CLI: python src/cli.py detect tests/fixtures/sample_tos.txt --explain")
        print("  • Tests: python -m pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or models.")
        print("Please ensure all requirements are installed and models are downloaded.")

if __name__ == "__main__":
    main()
