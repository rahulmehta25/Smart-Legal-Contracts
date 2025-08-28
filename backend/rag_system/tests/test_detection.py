"""Comprehensive test suite for RAG arbitration detection system."""
import pytest
from pathlib import Path
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.arbitration_detector import ArbitrationDetectionPipeline, ArbitrationClause
from models.legal_bert_detector import LegalBERTDetector, DetectionResult
from models.pattern_matcher import ArbitrationPatternMatcher
from document.section_detector import DocumentStructureAnalyzer, DocumentSection
from comparison.comparison_engine import ClauseComparisonEngine
from database.schema import DatabaseManager, VectorStore

# Sample test data
SAMPLE_ARBITRATION_TEXT = """
DISPUTE RESOLUTION AND ARBITRATION

Any dispute arising out of or relating to this Agreement shall be finally settled 
by binding arbitration administered by the American Arbitration Association (AAA) 
under its Commercial Arbitration Rules. The arbitration shall be conducted in 
New York, New York. The parties waive any right to a jury trial and agree that 
class action lawsuits are prohibited. You may opt out of this arbitration agreement 
within 30 days of accepting these terms.
"""

SAMPLE_NON_ARBITRATION_TEXT = """
PRIVACY POLICY

We respect your privacy and are committed to protecting your personal data. 
This privacy policy will inform you about how we look after your personal data 
when you visit our website and tell you about your privacy rights and how the 
law protects you.
"""

@pytest.fixture
def pipeline():
    """Create a test pipeline instance."""
    with patch('core.arbitration_detector.redis.Redis'):
        return ArbitrationDetectionPipeline(cache_enabled=False)

@pytest.fixture
def pattern_matcher():
    """Create a pattern matcher instance."""
    return ArbitrationPatternMatcher()

@pytest.fixture
def bert_detector():
    """Create a BERT detector instance."""
    with patch('models.legal_bert_detector.AutoModel'):
        with patch('models.legal_bert_detector.AutoTokenizer'):
            return LegalBERTDetector()

@pytest.fixture
def comparison_engine():
    """Create a comparison engine instance."""
    return ClauseComparisonEngine(db_url="sqlite:///:memory:")

class TestPatternMatcher:
    """Test pattern matching functionality."""
    
    def test_detect_arbitration_patterns(self, pattern_matcher):
        """Test detection of arbitration patterns."""
        result = pattern_matcher.match(SAMPLE_ARBITRATION_TEXT)
        
        assert result['confidence'] > 0.7
        assert len(result['matches']) > 0
        assert result['num_patterns'] > 0
        assert 'mandatory_arbitration' in str(result['matches'])
    
    def test_no_arbitration_patterns(self, pattern_matcher):
        """Test non-detection in non-arbitration text."""
        result = pattern_matcher.match(SAMPLE_NON_ARBITRATION_TEXT)
        
        assert result['confidence'] < 0.3
        assert result['num_patterns'] == 0
    
    def test_clause_type_analysis(self, pattern_matcher):
        """Test clause type analysis."""
        mandatory_text = "You shall submit to binding arbitration"
        optional_text = "You may choose to arbitrate disputes"
        opt_out_text = "You can opt-out of this arbitration"
        
        assert pattern_matcher.analyze_clause_type(mandatory_text) == 'mandatory'
        assert pattern_matcher.analyze_clause_type(optional_text) == 'optional'
        assert pattern_matcher.analyze_clause_type(opt_out_text) == 'mandatory_with_opt_out'

class TestLegalBERTDetector:
    """Test Legal-BERT detection functionality."""
    
    @patch('models.legal_bert_detector.torch.no_grad')
    def test_detect_arbitration(self, mock_no_grad, bert_detector):
        """Test arbitration detection with BERT."""
        # Mock the embedding and classifier output
        bert_detector._get_embedding = Mock(return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([1.0]))))))
        bert_detector.classifier = Mock(return_value=Mock())
        
        with patch('torch.softmax', return_value=Mock(item=Mock(return_value=0.8))):
            result = bert_detector.detect(SAMPLE_ARBITRATION_TEXT)
        
        assert isinstance(result, DetectionResult)
        assert result.is_arbitration is True
        assert result.confidence > 0.5
    
    def test_batch_detection(self, bert_detector):
        """Test batch detection functionality."""
        texts = [SAMPLE_ARBITRATION_TEXT, SAMPLE_NON_ARBITRATION_TEXT]
        
        bert_detector.detect = Mock(side_effect=[
            DetectionResult(True, 0.9, "", 0, 100, [], 0.9),
            DetectionResult(False, 0.2, "", 0, 100, [], 0.2)
        ])
        
        results = bert_detector.batch_detect(texts)
        
        assert len(results) == 2
        assert results[0].is_arbitration is True
        assert results[1].is_arbitration is False

class TestDocumentStructureAnalyzer:
    """Test document structure analysis."""
    
    def test_section_detection(self):
        """Test section detection in text."""
        analyzer = DocumentStructureAnalyzer()
        
        test_text = """
        1. TERMS AND CONDITIONS
        This is the terms section.
        
        2. DISPUTE RESOLUTION
        All disputes shall be resolved through arbitration.
        
        3. PRIVACY POLICY
        We protect your privacy.
        """
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_text)
            temp_path = f.name
        
        try:
            sections = analyzer.analyze_document(temp_path)
            
            assert len(sections) >= 2
            assert any('DISPUTE' in s.title.upper() for s in sections)
        finally:
            Path(temp_path).unlink()
    
    def test_arbitration_section_scoring(self):
        """Test scoring of sections for arbitration likelihood."""
        analyzer = DocumentStructureAnalyzer()
        
        sections = [
            DocumentSection("Dispute Resolution", "arbitration mandatory", 1, 1),
            DocumentSection("Privacy", "personal data protection", 2, 2),
            DocumentSection("Payment", "credit card processing", 3, 3)
        ]
        
        scored_sections = analyzer._score_sections(sections)
        
        assert scored_sections[0].confidence > scored_sections[1].confidence
        assert scored_sections[0].confidence > scored_sections[2].confidence

class TestArbitrationDetectionPipeline:
    """Test the complete detection pipeline."""
    
    def test_detect_from_text(self, pipeline):
        """Test detection directly from text."""
        with patch.object(pipeline.bert_detector, 'detect') as mock_detect:
            mock_detect.return_value = DetectionResult(
                is_arbitration=True,
                confidence=0.85,
                text_span="test",
                start_idx=0,
                end_idx=100,
                pattern_matches=["mandatory_arbitration"],
                semantic_score=0.8
            )
            
            result = pipeline.detect_from_text(SAMPLE_ARBITRATION_TEXT)
            
            assert result is not None
            assert isinstance(result, ArbitrationClause)
            assert result.confidence == 0.85
    
    def test_cache_functionality(self):
        """Test caching of detection results."""
        with patch('redis.Redis') as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            mock_redis_instance.get.return_value = None
            
            pipeline = ArbitrationDetectionPipeline(cache_enabled=True)
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                f.write(b"test content")
                temp_path = f.name
            
            try:
                # Mock the detection
                with patch.object(pipeline.structure_analyzer, 'find_arbitration_sections') as mock_find:
                    mock_find.return_value = []
                    
                    result = pipeline.detect_arbitration_clause(temp_path)
                    
                    # Check cache was checked
                    assert mock_redis_instance.get.called
            finally:
                Path(temp_path).unlink()

class TestComparisonEngine:
    """Test clause comparison functionality."""
    
    def test_add_clause_to_database(self, comparison_engine):
        """Test adding a clause to the database."""
        clause_data = {
            'text': SAMPLE_ARBITRATION_TEXT,
            'company': 'Test Corp',
            'industry': 'Technology',
            'document_type': 'TOS',
            'risk_score': 0.7
        }
        
        with patch.object(comparison_engine.bert_detector, '_get_embedding') as mock_embed:
            mock_embed.return_value = Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.random.randn(768)))))
            
            clause_id = comparison_engine.add_clause_to_database(clause_data)
            
            assert clause_id is not None
    
    def test_compare_clause(self, comparison_engine):
        """Test clause comparison."""
        # Add some test clauses first
        test_clauses = [
            {
                'text': 'Mandatory binding arbitration under AAA rules',
                'company': 'Company A',
                'industry': 'Tech',
                'risk_score': 0.8
            },
            {
                'text': 'Disputes resolved through mediation',
                'company': 'Company B',
                'industry': 'Finance',
                'risk_score': 0.3
            }
        ]
        
        with patch.object(comparison_engine.bert_detector, '_get_embedding') as mock_embed:
            mock_embed.return_value = Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.random.randn(768)))))
            
            for clause in test_clauses:
                comparison_engine.add_clause_to_database(clause)
            
            # Now compare
            result = comparison_engine.compare_clause(SAMPLE_ARBITRATION_TEXT, top_k=5)
            
            assert 'similar_clauses' in result
            assert 'analysis' in result
            assert 'statistics' in result

class TestVectorStore:
    """Test vector store functionality."""
    
    def test_vector_store_operations(self):
        """Test vector store add and search operations."""
        vector_store = VectorStore(dimension=768)
        
        # Add vectors
        vector1 = np.random.randn(768).astype('float32')
        vector2 = np.random.randn(768).astype('float32')
        
        assert vector_store.add_clause("clause1", vector1)
        assert vector_store.add_clause("clause2", vector2)
        
        # Search
        query_vector = np.random.randn(768).astype('float32')
        results = vector_store.search_similar(query_vector, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r[1], float) for r in results)
    
    def test_vector_store_persistence(self, tmp_path):
        """Test saving and loading vector store."""
        vector_store = VectorStore(dimension=768)
        
        # Add vectors
        vector1 = np.random.randn(768).astype('float32')
        vector_store.add_clause("test_clause", vector1)
        
        # Save
        save_path = str(tmp_path / "test_vectors")
        vector_store.save(save_path)
        
        # Load in new instance
        new_store = VectorStore(dimension=768)
        new_store.load(save_path)
        
        assert new_store.get_stats()['total_vectors'] == 1
        assert new_store.get_stats()['mapped_ids'] == 1

class TestDatabaseManager:
    """Test database management functionality."""
    
    def test_database_operations(self):
        """Test database CRUD operations."""
        db_manager = DatabaseManager(db_url="sqlite:///:memory:")
        
        # Add clause
        clause_data = {
            'company_name': 'Test Company',
            'industry': 'Technology',
            'document_type': 'TOS',
            'clause_text': 'Test arbitration clause',
            'clause_summary': 'Test summary',
            'key_provisions': ['provision1', 'provision2'],
            'enforceability_score': 0.8,
            'risk_score': 0.6,
            'jurisdiction': 'US',
            'vector_id': 'test_vector_id'
        }
        
        clause_id = db_manager.add_clause(clause_data)
        assert clause_id is not None
        
        # Get clause
        retrieved = db_manager.get_clause(clause_id)
        assert retrieved is not None
        assert retrieved['company_name'] == 'Test Company'
        
        # Search clauses
        results = db_manager.search_clauses({'industry': 'Technology'})
        assert len(results) == 1
        assert results[0]['industry'] == 'Technology'

@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_detection(self):
        """Test end-to-end detection flow."""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_ARBITRATION_TEXT)
            temp_path = f.name
        
        try:
            # Run detection
            pipeline = ArbitrationDetectionPipeline(cache_enabled=False)
            
            with patch.object(pipeline.bert_detector, '_get_embedding') as mock_embed:
                mock_embed.return_value = Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.random.randn(768)))))
                
                with patch('torch.softmax', return_value=torch.tensor([[0.2, 0.8]])):
                    result = pipeline.detect_arbitration_clause(temp_path)
            
            # Verify result
            assert result is not None
            assert result.confidence > 0.5
            assert len(result.key_provisions) > 0
        finally:
            Path(temp_path).unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])