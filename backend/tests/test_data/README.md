# Test Data Documentation

This directory contains test data for the arbitration clause detection system.

## File Structure

### sample_documents.json
Contains categorized test documents for functional testing:

- **clear_arbitration_documents**: Documents with obvious arbitration clauses
- **hidden_arbitration_documents**: Documents with arbitration clauses buried in text
- **no_arbitration_documents**: Documents that do not contain arbitration clauses
- **ambiguous_documents**: Documents with unclear or optional arbitration language
- **complex_arbitration_documents**: Documents with multi-tier or complex arbitration procedures
- **false_positive_prevention**: Documents designed to test false positive prevention

### performance_test_documents.json
Contains documents and benchmarks for performance testing:

- **large_documents**: Very large documents for testing processing speed
- **stress_test_documents**: Documents designed to stress-test the system
- **encoding_test_documents**: Documents with various encoding challenges
- **performance_benchmarks**: Target metrics for different document sizes
- **concurrent_processing_tests**: Specifications for load testing
- **memory_usage_tests**: Memory usage thresholds and limits
- **accuracy_test_suite**: Target accuracy metrics

## Usage in Tests

### Loading Test Data
```python
import json
from pathlib import Path

def load_test_documents():
    test_data_dir = Path(__file__).parent / "test_data"
    
    with open(test_data_dir / "sample_documents.json") as f:
        sample_docs = json.load(f)
    
    with open(test_data_dir / "performance_test_documents.json") as f:
        perf_docs = json.load(f)
    
    return sample_docs, perf_docs
```

### Document Categories

1. **Functional Testing**: Use `sample_documents.json` for:
   - Unit tests of detection accuracy
   - Testing different clause types
   - Edge case handling
   - False positive prevention

2. **Performance Testing**: Use `performance_test_documents.json` for:
   - Load testing
   - Memory usage monitoring
   - Processing speed benchmarks
   - Concurrent request handling

### Expected Results Format

Each test document includes:
```json
{
  "id": "unique_identifier",
  "filename": "descriptive_name.txt",
  "content": "actual document text...",
  "expected_result": {
    "has_arbitration": true/false,
    "confidence": 0.0-1.0,
    "clause_type": "mandatory_binding|optional|multi_tier|etc",
    "keywords": ["list", "of", "key", "terms"],
    "additional_metadata": "varies by document type"
  }
}
```

### Confidence Score Guidelines

- **0.90-1.00**: Very clear arbitration language
- **0.70-0.89**: Clear arbitration with some ambiguity
- **0.50-0.69**: Ambiguous or optional arbitration
- **0.20-0.49**: Weak arbitration indicators
- **0.00-0.19**: No arbitration or false positives

### Performance Benchmarks

- **Small documents** (< 500 words): < 100ms processing
- **Medium documents** (500-2000 words): < 500ms processing  
- **Large documents** (2000-10000 words): < 2s processing
- **Very large documents** (> 10000 words): < 8s processing

### Accuracy Targets

- **Precision**: ≥ 85% (minimize false positives)
- **Recall**: ≥ 88% (minimize false negatives)
- **F1 Score**: ≥ 86% (balanced accuracy)

## Adding New Test Cases

When adding new test cases:

1. Choose appropriate category based on document characteristics
2. Include realistic legal language and structure
3. Provide comprehensive expected results
4. Consider edge cases and boundary conditions
5. Update this documentation if adding new categories

## Document Sources

Test documents are synthesized examples created for testing purposes. They are designed to represent realistic arbitration clause patterns found in:

- Commercial contracts
- Employment agreements
- Consumer terms of service
- International trade agreements
- Merger and acquisition documents
- Service provider agreements

No actual confidential documents are included in this test suite.