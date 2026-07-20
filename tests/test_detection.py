import pytest
from pathlib import Path
import tempfile
import os

def test_pattern_matcher():
    """Test pattern matching functionality"""
    from src.models.pattern_matcher import ArbitrationPatternMatcher
    
    matcher = ArbitrationPatternMatcher()
    
    # Test arbitration text
    arbitration_text = "Any dispute shall be resolved through binding arbitration under JAMS rules."
    result = matcher.match(arbitration_text)
    
    assert result['confidence'] > 0.5
    assert len(result['matches']) > 0
    
    # Test non-arbitration text
    non_arbitration_text = "This is a simple contract about services."
    result = matcher.match(non_arbitration_text)
    
    assert result['confidence'] < 0.5

def test_document_section_detector():
    """Test document section detection"""
    from src.document.section_detector import DocumentStructureAnalyzer
    
    analyzer = DocumentStructureAnalyzer()
    
    # Create a test document
    test_content = """
    SECTION 1. INTRODUCTION
    This is the introduction section.
    
    SECTION 2. ARBITRATION
    Any disputes shall be resolved through arbitration.
    
    SECTION 3. CONCLUSION
    This concludes the document.
    """
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        sections = analyzer.analyze_document(temp_path)
        
        # Should find sections
        assert len(sections) > 0
        
        # Should identify arbitration section
        arbitration_sections = analyzer.find_arbitration_sections(temp_path, threshold=0.3)
        assert len(arbitration_sections) > 0
        
    finally:
        os.unlink(temp_path)

def test_arbitration_detection_pipeline():
    """Test the complete detection pipeline"""
    from src.core.arbitration_detector import ArbitrationDetectionPipeline
    
    pipeline = ArbitrationDetectionPipeline(cache_enabled=False)
    
    # Create test document with arbitration clause
    test_content = """
    TERMS OF SERVICE
    
    DISPUTE RESOLUTION
    
    Any dispute arising from this agreement shall be finally settled by binding arbitration 
    administered by JAMS in accordance with its Comprehensive Arbitration Rules and Procedures. 
    The parties agree to waive any right to a jury trial or to bring or participate in any 
    class action lawsuit.
    
    The arbitration shall be conducted in English and the seat of arbitration shall be 
    New York, New York.
    """
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        result = pipeline.detect_arbitration_clause(temp_path)
        
        # Should detect arbitration clause
        assert result is not None
        assert result.confidence > 0.7
        assert result.clause_type == 'mandatory'
        assert len(result.key_provisions) > 0
        
    finally:
        os.unlink(temp_path)

def test_comparison_engine():
    """Test clause comparison functionality"""
    from src.comparison.comparison_engine import ClauseComparisonEngine
    
    engine = ClauseComparisonEngine()
    
    # Test clause comparison
    test_clause = "Any dispute shall be resolved through binding arbitration under JAMS rules."
    
    comparison = engine.compare_clause(test_clause)
    
    assert 'similar_clauses' in comparison
    assert 'analysis' in comparison
    assert 'statistics' in comparison

def test_explainability():
    """Test explainability features"""
    from src.explainability.explainer import ArbitrationExplainer
    
    # Mock detector for testing
    class MockDetector:
        def detect(self, text):
            class MockResult:
                def __init__(self):
                    self.confidence = 0.8
                    self.semantic_score = 0.7
                    self.pattern_matches = ['mandatory_arbitration: shall be resolved by arbitration']
                    self.is_arbitration = True
            return MockResult()
    
    detector = MockDetector()
    explainer = ArbitrationExplainer(detector)
    
    test_text = "Any dispute shall be resolved through binding arbitration."
    explanation = explainer.explain_detection(test_text, detector.detect(test_text))
    
    assert 'confidence_breakdown' in explanation
    assert 'key_indicators' in explanation
    assert 'decision_path' in explanation

def test_api_endpoints():
    """Test API endpoint functionality"""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    client = TestClient(app)
    
    # Test health check
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert "Arbitration Clause Detection API" in response.json()["message"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

