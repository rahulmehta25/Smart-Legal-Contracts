"""
Pytest configuration and shared fixtures for arbitration clause detection tests.
"""

import pytest
import os
import tempfile
import json
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Returns the path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_documents():
    """Sample documents for testing arbitration clause detection."""
    return {
        "clear_arbitration": """
        ARBITRATION AGREEMENT
        Any dispute, claim, or controversy arising out of or relating to this Agreement 
        shall be settled by binding arbitration administered by the American Arbitration 
        Association (AAA) in accordance with its Commercial Arbitration Rules.
        """,
        
        "hidden_arbitration": """
        Terms and Conditions
        
        Section 15.3: In the event of any disagreement between the parties concerning 
        the interpretation of this contract, the matter shall be resolved through 
        mandatory arbitration proceedings conducted under JAMS rules. The parties 
        waive their right to a jury trial.
        """,
        
        "no_arbitration": """
        DISPUTE RESOLUTION
        Any disputes arising under this agreement shall be resolved in the courts 
        of New York State. The parties agree to submit to the jurisdiction of 
        such courts for all legal proceedings.
        """,
        
        "ambiguous_text": """
        MEDIATION CLAUSE
        The parties agree to first attempt resolution through mediation before 
        pursuing other legal remedies. If mediation fails, parties may pursue 
        any available legal remedies in accordance with applicable law.
        """,
        
        "complex_arbitration": """
        DISPUTE RESOLUTION PROCEDURES
        
        Step 1: Direct negotiation between parties (30 days)
        Step 2: Mediation through AAA (60 days)
        Step 3: If mediation fails, binding arbitration shall be conducted by a 
        three-arbitrator panel selected under ICC International Arbitration Rules. 
        The arbitration shall be seated in London, England, and conducted in English.
        Each party shall bear its own costs except the arbitration fees which shall 
        be split equally. The arbitrator's decision shall be final and binding.
        """
    }


@pytest.fixture
def expected_detection_results():
    """Expected results for arbitration clause detection."""
    return {
        "clear_arbitration": {
            "has_arbitration": True,
            "confidence": 0.95,
            "clause_type": "mandatory_binding",
            "keywords": ["binding arbitration", "American Arbitration Association", "AAA"]
        },
        "hidden_arbitration": {
            "has_arbitration": True,
            "confidence": 0.85,
            "clause_type": "mandatory_binding",
            "keywords": ["mandatory arbitration", "JAMS", "waive", "jury trial"]
        },
        "no_arbitration": {
            "has_arbitration": False,
            "confidence": 0.05,
            "clause_type": None,
            "keywords": []
        },
        "ambiguous_text": {
            "has_arbitration": False,
            "confidence": 0.2,
            "clause_type": None,
            "keywords": ["mediation"]
        },
        "complex_arbitration": {
            "has_arbitration": True,
            "confidence": 0.98,
            "clause_type": "multi_step_binding",
            "keywords": ["binding arbitration", "three-arbitrator panel", "ICC", "final and binding"]
        }
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing RAG pipeline."""
    mock_store = Mock()
    mock_store.similarity_search.return_value = [
        Mock(page_content="Sample arbitration clause", metadata={"source": "doc1.pdf"}),
        Mock(page_content="Another arbitration example", metadata={"source": "doc2.pdf"})
    ]
    return mock_store


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = Mock()
    mock_llm.predict.return_value = json.dumps({
        "has_arbitration": True,
        "confidence": 0.9,
        "clause_type": "mandatory_binding",
        "explanation": "The document contains a clear arbitration clause."
    })
    return mock_llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in range(5)]
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock_embeddings


@pytest.fixture
def temp_upload_dir():
    """Temporary directory for file uploads during testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_pdf_path(temp_upload_dir):
    """Creates a sample PDF file for testing."""
    # Create a simple text file (in real scenario, you'd create a proper PDF)
    pdf_path = os.path.join(temp_upload_dir, "sample_contract.pdf")
    with open(pdf_path, "w") as f:
        f.write("Sample contract with arbitration clause for testing.")
    return pdf_path


@pytest.fixture
def performance_metrics():
    """Expected performance metrics for benchmarking."""
    return {
        "processing_time_per_page": 0.5,  # seconds
        "max_memory_usage": 500,  # MB
        "accuracy_threshold": 0.85,
        "precision_threshold": 0.80,
        "recall_threshold": 0.85
    }


@pytest.fixture
def test_api_client():
    """Test client for API testing."""
    from fastapi.testclient import TestClient
    # This would import your actual FastAPI app
    # from backend.app import app
    # return TestClient(app)
    
    # For now, return a mock client
    return Mock()


@pytest.fixture
def test_database():
    """Test database setup and teardown."""
    # Setup test database
    test_db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "arbitration_test",
        "user": "test_user",
        "password": "test_password"
    }
    
    # In real implementation, you would:
    # 1. Create test database
    # 2. Run migrations
    # 3. Yield the connection
    # 4. Cleanup after tests
    
    yield test_db_config


class MockArbitrationDetector:
    """Mock arbitration detector for testing."""
    
    def __init__(self):
        self.detection_results = {}
    
    def detect_arbitration_clause(self, text: str) -> Dict[str, Any]:
        """Mock detection method."""
        return {
            "has_arbitration": "arbitration" in text.lower(),
            "confidence": 0.8 if "arbitration" in text.lower() else 0.2,
            "clause_type": "mock_type",
            "keywords": ["arbitration"] if "arbitration" in text.lower() else []
        }


@pytest.fixture
def mock_arbitration_detector():
    """Mock arbitration detector instance."""
    return MockArbitrationDetector()


# Test data constants
TEST_DOCUMENTS_COUNT = 100
BENCHMARK_DOCUMENTS_COUNT = 1000

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_processing_time": 10.0,  # seconds per document
    "max_memory_usage": 1024,     # MB
    "min_accuracy": 0.85,
    "min_precision": 0.80,
    "min_recall": 0.85
}