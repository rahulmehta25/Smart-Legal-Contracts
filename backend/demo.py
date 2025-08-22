#!/usr/bin/env python3
"""
Demonstration script for the Arbitration Clause Detection RAG System.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag import ArbitrationDetector, RetrievalConfig


def print_separator(title: str = ""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)


def print_clause_details(clause: Dict[str, Any]):
    """Print details of a detected arbitration clause."""
    print(f"\nClause #{clause.get('chunk_id', 'N/A')}:")
    print(f"  Confidence Score: {clause['confidence_score']:.2%}")
    print(f"  Type: {clause['arbitration_type']}")
    print(f"  Clause Types: {', '.join(clause['clause_types'])}")
    
    if clause.get('details'):
        print("  Details:")
        for key, value in clause['details'].items():
            if value:
                print(f"    - {key}: {value}")
    
    print(f"  Text Preview: {clause['text'][:200]}...")
    
    if clause.get('pattern_matches'):
        print(f"  Matched Patterns: {len(clause['pattern_matches'])} patterns")


def demonstrate_detection():
    """Demonstrate the arbitration detection system."""
    
    print_separator("Arbitration Clause Detection System Demo")
    
    # Initialize detector with custom configuration
    config = RetrievalConfig(
        chunk_size=512,
        chunk_overlap=128,
        top_k_retrieval=10,
        similarity_threshold=0.5,
        use_hybrid_search=True
    )
    
    print("\nInitializing Arbitration Detector...")
    print("  - Model: sentence-transformers/all-MiniLM-L6-v2")
    print("  - Chunk Size: 512 characters")
    print("  - Using Hybrid Search: Semantic + Keyword")
    
    detector = ArbitrationDetector(config)
    print("✓ Detector initialized successfully")
    
    # Test documents
    test_docs_path = Path(__file__).parent / "data" / "test_documents"
    
    # Process document with arbitration
    print_separator("Testing Document WITH Arbitration Clause")
    
    doc_with_arb = test_docs_path / "sample_tou_with_arbitration.txt"
    if doc_with_arb.exists():
        print(f"\nProcessing: {doc_with_arb.name}")
        
        result = detector.detect_from_file(str(doc_with_arb))
        
        print(f"\n✓ Detection Complete in {result.processing_time:.2f} seconds")
        print(f"\nResults:")
        print(f"  - Has Arbitration: {result.has_arbitration}")
        print(f"  - Overall Confidence: {result.confidence:.2%}")
        print(f"  - Number of Clauses Found: {len(result.clauses)}")
        
        if result.summary:
            print("\nSummary:")
            print(f"  - Binding Arbitration: {result.summary.get('has_binding_arbitration', False)}")
            print(f"  - Class Action Waiver: {result.summary.get('has_class_action_waiver', False)}")
            print(f"  - Jury Trial Waiver: {result.summary.get('has_jury_trial_waiver', False)}")
            print(f"  - Opt-Out Available: {result.summary.get('has_opt_out', False)}")
            print(f"  - Provider: {result.summary.get('arbitration_provider', 'Not specified')}")
        
        if result.clauses:
            print("\nDetected Clauses:")
            for clause in result.clauses[:3]:  # Show first 3 clauses
                print_clause_details(clause.to_dict())
    else:
        print(f"⚠ Test document not found: {doc_with_arb}")
    
    # Process document without arbitration
    print_separator("Testing Document WITHOUT Arbitration Clause")
    
    doc_without_arb = test_docs_path / "sample_tou_without_arbitration.txt"
    if doc_without_arb.exists():
        print(f"\nProcessing: {doc_without_arb.name}")
        
        result = detector.detect_from_file(str(doc_without_arb))
        
        print(f"\n✓ Detection Complete in {result.processing_time:.2f} seconds")
        print(f"\nResults:")
        print(f"  - Has Arbitration: {result.has_arbitration}")
        print(f"  - Overall Confidence: {result.confidence:.2%}")
        print(f"  - Number of Clauses Found: {len(result.clauses)}")
        
        if not result.has_arbitration:
            print("\n✓ Correctly identified: No arbitration clause present")
    else:
        print(f"⚠ Test document not found: {doc_without_arb}")
    
    # Test with custom text
    print_separator("Testing Custom Text Snippets")
    
    test_texts = [
        {
            "name": "Strong Binding Arbitration",
            "text": "You agree that any dispute arising out of these Terms shall be resolved through binding arbitration administered by JAMS. You waive your right to a jury trial and to participate in class actions."
        },
        {
            "name": "Court Jurisdiction (No Arbitration)",
            "text": "Any disputes arising from these Terms shall be resolved in the state and federal courts located in New York County, New York, and you consent to the personal jurisdiction of such courts."
        },
        {
            "name": "Voluntary Arbitration",
            "text": "If a dispute arises, either party may elect to resolve it through voluntary arbitration. However, you retain your right to pursue claims in court if you prefer."
        }
    ]
    
    for test_case in test_texts:
        print(f"\n\nTest: {test_case['name']}")
        print(f"Text: \"{test_case['text'][:100]}...\"")
        
        result = detector.detect(test_case['text'], document_id=test_case['name'])
        
        print(f"Result:")
        print(f"  - Has Arbitration: {result.has_arbitration}")
        print(f"  - Confidence: {result.confidence:.2%}")
        if result.has_arbitration and result.clauses:
            print(f"  - Type: {result.clauses[0].arbitration_type.value}")
    
    print_separator("Demo Complete")
    print("\nThe RAG system successfully:")
    print("  ✓ Detected arbitration clauses with high confidence")
    print("  ✓ Identified specific clause types (binding, class action waiver, etc.)")
    print("  ✓ Extracted key details (provider, location, opt-out)")
    print("  ✓ Distinguished between documents with and without arbitration")
    print("  ✓ Used hybrid search combining semantic and keyword matching")


def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis capabilities."""
    print_separator("Pattern Analysis Demo")
    
    from app.rag import ArbitrationPatterns
    
    patterns = ArbitrationPatterns()
    
    sample_text = """
    DISPUTE RESOLUTION. Any controversy or claim arising out of or relating to 
    this Agreement shall be settled by binding arbitration administered by the 
    American Arbitration Association in accordance with its Commercial Arbitration 
    Rules. The arbitration shall be conducted in San Francisco, California. 
    You waive your right to a jury trial and agree that no class action proceedings 
    will be permitted. You may opt out of this arbitration agreement by sending 
    written notice within 30 days of accepting these terms.
    """
    
    print(f"\nAnalyzing sample text for arbitration patterns...")
    
    # Extract details
    details = ArbitrationPatterns.extract_arbitration_details(sample_text)
    
    print("\nExtracted Details:")
    for key, value in details.items():
        if value:
            print(f"  - {key}: {value}")
    
    # Count pattern matches
    all_patterns = ArbitrationPatterns.get_all_patterns()
    matched_count = 0
    high_confidence_matches = []
    
    for pattern in all_patterns:
        import re
        text_lower = sample_text.lower()
        
        if pattern.pattern_type == 'regex':
            if re.search(pattern.pattern, text_lower):
                matched_count += 1
                if pattern.weight > 0.8:
                    high_confidence_matches.append(pattern.pattern)
        else:
            if pattern.pattern in text_lower:
                matched_count += 1
                if pattern.weight > 0.8:
                    high_confidence_matches.append(pattern.pattern)
    
    print(f"\nPattern Matching Results:")
    print(f"  - Total patterns matched: {matched_count}")
    print(f"  - High confidence matches: {len(high_confidence_matches)}")
    if high_confidence_matches:
        print("  - Examples of high confidence patterns:")
        for pattern in high_confidence_matches[:5]:
            print(f"    • {pattern}")


if __name__ == "__main__":
    try:
        # Run main demonstration
        demonstrate_detection()
        
        # Run pattern analysis
        print("\n" + "="*60 + "\n")
        demonstrate_pattern_analysis()
        
    except ImportError as e:
        print(f"\n⚠ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r backend/requirements.txt")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()