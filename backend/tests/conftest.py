"""
Comprehensive pytest configuration and shared fixtures for Smart Legal Contracts.

Provides fixtures for:
- Mock AI services (embeddings, LLM)
- Mock vector store (Qdrant)
- Test database setup
- Sample documents (PDF, DOCX, TXT)
- FastAPI test client
- Performance benchmarking utilities
"""

import pytest
import os
import io
import tempfile
import json
import asyncio
from typing import Dict, List, Any, Optional, Generator
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path
from dataclasses import dataclass
import hashlib
import time

# Optional imports with graceful fallbacks
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    TestClient = Mock
    FASTAPI_AVAILABLE = False

try:
    from httpx import AsyncClient, ASGITransport
    HTTPX_AVAILABLE = True
except ImportError:
    AsyncClient = Mock
    ASGITransport = Mock
    HTTPX_AVAILABLE = False

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    create_engine = Mock
    sessionmaker = Mock
    Session = Mock
    StaticPool = Mock
    SQLALCHEMY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = Mock()
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = Mock()
    NUMPY_AVAILABLE = False


# ============================================================================
# Configuration Constants
# ============================================================================

TEST_DATA_DIR = Path(__file__).parent / "test_data"
SAMPLE_DOCUMENTS_DIR = TEST_DATA_DIR / "sample_documents"

PERFORMANCE_THRESHOLDS = {
    "max_processing_time": 10.0,  # seconds per document
    "max_memory_usage": 1024,     # MB
    "min_accuracy": 0.85,
    "min_precision": 0.80,
    "min_recall": 0.85,
    "max_api_response_time": 2.0,  # seconds
}

TEST_DOCUMENTS_COUNT = 100
BENCHMARK_DOCUMENTS_COUNT = 1000


# ============================================================================
# Session-scoped Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Returns the path to test data directory."""
    path = Path(__file__).parent / "test_data"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def sample_documents_dir(test_data_dir) -> Path:
    """Returns the path to sample documents directory."""
    path = test_data_dir / "sample_documents"
    path.mkdir(exist_ok=True)
    return path


# ============================================================================
# Sample Document Fixtures
# ============================================================================

@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Sample documents for testing arbitration clause detection."""
    return {
        "clear_arbitration": """
        ARBITRATION AGREEMENT
        Any dispute, claim, or controversy arising out of or relating to this Agreement
        shall be settled by binding arbitration administered by the American Arbitration
        Association (AAA) in accordance with its Commercial Arbitration Rules.
        The arbitration shall take place in New York, NY. The arbitrator's decision
        shall be final and binding on both parties.
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
        """,

        "class_action_waiver": """
        CLASS ACTION WAIVER
        You agree that any dispute resolution proceedings will be conducted only
        on an individual basis and not in a class, consolidated, or representative
        action. You waive any right to participate in class action lawsuits or
        class-wide arbitration.

        Binding arbitration will be administered by AAA under its Consumer
        Arbitration Rules.
        """,

        "opt_out_provision": """
        ARBITRATION AGREEMENT WITH OPT-OUT

        All disputes will be resolved through binding arbitration. However, you
        may opt out of this arbitration agreement by sending written notice to
        legal@company.com within 30 days of accepting these terms.
        """
    }


@pytest.fixture
def expected_detection_results() -> Dict[str, Dict[str, Any]]:
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
        },
        "class_action_waiver": {
            "has_arbitration": True,
            "confidence": 0.90,
            "clause_type": "mandatory_binding",
            "keywords": ["class action waiver", "binding arbitration", "AAA"]
        },
        "opt_out_provision": {
            "has_arbitration": True,
            "confidence": 0.85,
            "clause_type": "binding_with_opt_out",
            "keywords": ["binding arbitration", "opt out"]
        }
    }


# ============================================================================
# Mock AI Services Fixtures
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Mock embeddings model that returns consistent fake embeddings."""
    mock = Mock()

    def embed_documents(texts: List[str]) -> List[List[float]]:
        """Generate deterministic fake embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Generate reproducible embeddings from text hash
            text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            if NUMPY_AVAILABLE:
                np.random.seed(text_hash)
                embedding = np.random.randn(384).tolist()
            else:
                import random
                random.seed(text_hash)
                embedding = [random.gauss(0, 1) for _ in range(384)]
            embeddings.append(embedding)
        return embeddings

    def embed_query(text: str) -> List[float]:
        """Generate deterministic fake embedding for query."""
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        if NUMPY_AVAILABLE:
            np.random.seed(text_hash)
            return np.random.randn(384).tolist()
        else:
            import random
            random.seed(text_hash)
            return [random.gauss(0, 1) for _ in range(384)]

    mock.embed_documents = embed_documents
    mock.embed_query = embed_query
    mock.dimension = 384
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()

    def predict(prompt: str) -> str:
        """Return structured response based on prompt content."""
        prompt_lower = prompt.lower()

        if "arbitration" in prompt_lower:
            return json.dumps({
                "has_arbitration": True,
                "confidence": 0.9,
                "clause_type": "mandatory_binding",
                "explanation": "The document contains a clear arbitration clause.",
                "key_phrases": ["binding arbitration", "AAA"]
            })
        else:
            return json.dumps({
                "has_arbitration": False,
                "confidence": 0.1,
                "clause_type": None,
                "explanation": "No arbitration clause detected.",
                "key_phrases": []
            })

    async def apredict(prompt: str) -> str:
        return predict(prompt)

    mock.predict = predict
    mock.apredict = apredict
    mock.invoke = predict
    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=json.dumps({
        "has_arbitration": True,
        "confidence": 0.85,
        "clause_type": "binding",
        "explanation": "Test response"
    })))]
    mock_client.chat.completions.create = Mock(return_value=mock_response)
    return mock_client


# ============================================================================
# Mock Vector Store Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing RAG pipeline."""
    mock_store = Mock()

    # Store documents in memory
    mock_store._documents = {}
    mock_store._embeddings = {}

    def add_texts(texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """Add texts to mock store."""
        if ids is None:
            ids = [str(i) for i in range(len(mock_store._documents), len(mock_store._documents) + len(texts))]
        if metadatas is None:
            metadatas = [{}] * len(texts)

        for id_, text, metadata in zip(ids, texts, metadatas):
            mock_store._documents[id_] = {"text": text, "metadata": metadata}

        return ids

    def similarity_search(query: str, k: int = 4) -> List[Mock]:
        """Return mock search results."""
        results = []
        for id_, doc in list(mock_store._documents.items())[:k]:
            result = Mock()
            result.page_content = doc["text"]
            result.metadata = doc.get("metadata", {"source": "test.pdf"})
            results.append(result)

        if not results:
            # Return default results if store is empty
            result = Mock()
            result.page_content = "Sample arbitration clause for testing"
            result.metadata = {"source": "default.pdf"}
            results.append(result)

        return results

    def similarity_search_with_score(query: str, k: int = 4) -> List[tuple]:
        """Return mock search results with scores."""
        results = similarity_search(query, k)
        return [(r, 0.8 - i * 0.1) for i, r in enumerate(results)]

    mock_store.add_texts = add_texts
    mock_store.similarity_search = similarity_search
    mock_store.similarity_search_with_score = similarity_search_with_score
    mock_store.delete = Mock(return_value=True)

    return mock_store


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for vector store operations."""
    mock_client = Mock()

    # In-memory storage
    mock_client._collections = {}
    mock_client._points = {}

    def create_collection(collection_name: str, vectors_config: Any):
        mock_client._collections[collection_name] = vectors_config
        mock_client._points[collection_name] = []
        return True

    def get_collection(collection_name: str):
        if collection_name not in mock_client._collections:
            raise Exception(f"Collection {collection_name} not found")
        return Mock(
            vectors_count=len(mock_client._points.get(collection_name, [])),
            points_count=len(mock_client._points.get(collection_name, []))
        )

    def upsert(collection_name: str, points: List[Any]):
        if collection_name not in mock_client._points:
            mock_client._points[collection_name] = []
        mock_client._points[collection_name].extend(points)
        return Mock(status="completed")

    def search(collection_name: str, query_vector: List[float], limit: int = 10, **kwargs):
        points = mock_client._points.get(collection_name, [])[:limit]
        results = []
        for i, point in enumerate(points):
            result = Mock()
            result.id = getattr(point, 'id', str(i))
            result.score = 0.9 - i * 0.05
            result.payload = getattr(point, 'payload', {"text": "Sample text"})
            results.append(result)
        return results

    def delete(collection_name: str, points_selector: Any):
        return Mock(status="completed")

    mock_client.create_collection = create_collection
    mock_client.get_collection = get_collection
    mock_client.upsert = upsert
    mock_client.search = search
    mock_client.delete = delete
    mock_client.delete_collection = Mock(return_value=True)
    mock_client.get_collections = Mock(return_value=Mock(collections=[]))

    return mock_client


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def test_db_engine():
    """Create an in-memory SQLite database engine for testing."""
    if not SQLALCHEMY_AVAILABLE:
        pytest.skip("SQLAlchemy not available")

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


@pytest.fixture(scope="function")
def test_db_session(test_db_engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)

    # Create tables
    try:
        from app.db.database import Base
        Base.metadata.create_all(bind=test_db_engine)
    except ImportError:
        pass  # Base not available

    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_database():
    """Test database configuration for integration tests."""
    return {
        "host": os.getenv("TEST_DB_HOST", "localhost"),
        "port": int(os.getenv("TEST_DB_PORT", "5432")),
        "database": os.getenv("TEST_DB_NAME", "arbitration_test"),
        "user": os.getenv("TEST_DB_USER", "test_user"),
        "password": os.getenv("TEST_DB_PASSWORD", "test_password")
    }


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================

@pytest.fixture
def mock_app():
    """Create a minimal FastAPI app for testing."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Test API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/")
    async def root():
        return {"message": "Test API"}

    return app


@pytest.fixture
def test_client(mock_app) -> TestClient:
    """Create a test client for the mock app."""
    return TestClient(mock_app)


@pytest.fixture
def app_client():
    """Create a test client for the real application."""
    try:
        from app.main import app

        # Override dependencies for testing
        def override_get_db():
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy.pool import StaticPool

            engine = create_engine(
                "sqlite:///:memory:",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

            try:
                from app.db.database import Base
                Base.metadata.create_all(bind=engine)
            except ImportError:
                pass

            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        # Apply override
        try:
            from app.db.database import get_db
            app.dependency_overrides[get_db] = override_get_db
        except ImportError:
            pass

        with TestClient(app) as client:
            yield client

        # Clean up
        app.dependency_overrides.clear()

    except ImportError:
        # Return mock client if app not available
        yield Mock()


@pytest.fixture
async def async_client(mock_app):
    """Create an async test client."""
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not available")

    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================================================
# File and Document Fixtures
# ============================================================================

@pytest.fixture
def temp_upload_dir() -> Generator[str, None, None]:
    """Temporary directory for file uploads during testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_txt_file(temp_upload_dir) -> str:
    """Create a sample text file for testing."""
    content = """
    TERMS OF SERVICE AGREEMENT

    This Terms of Service Agreement ("Agreement") is entered into between
    User and Company.

    1. ACCEPTANCE OF TERMS
    By using our services, you agree to these terms.

    2. ARBITRATION CLAUSE
    Any dispute arising from this agreement shall be resolved through
    binding arbitration administered by the American Arbitration Association.

    3. CLASS ACTION WAIVER
    You agree to resolve disputes individually and waive any right to
    participate in class action lawsuits.
    """
    file_path = os.path.join(temp_upload_dir, "sample_terms.txt")
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create minimal PDF bytes for testing (mock PDF structure)."""
    # This is a minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_pdf_file(temp_upload_dir, sample_pdf_bytes) -> str:
    """Create a sample PDF file for testing."""
    file_path = os.path.join(temp_upload_dir, "sample_contract.pdf")
    with open(file_path, "wb") as f:
        f.write(sample_pdf_bytes)
    return file_path


@pytest.fixture
def sample_docx_bytes() -> bytes:
    """Create minimal DOCX bytes for testing."""
    # DOCX is a ZIP file with specific structure
    # This creates a minimal valid DOCX
    import zipfile

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
        zf.writestr('[Content_Types].xml', content_types)

        # _rels/.rels
        rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        zf.writestr('_rels/.rels', rels)

        # word/document.xml
        document = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p>
            <w:r>
                <w:t>Test DOCX content with arbitration clause for testing.</w:t>
            </w:r>
        </w:p>
    </w:body>
</w:document>"""
        zf.writestr('word/document.xml', document)

    return buffer.getvalue()


@pytest.fixture
def sample_docx_file(temp_upload_dir, sample_docx_bytes) -> str:
    """Create a sample DOCX file for testing."""
    file_path = os.path.join(temp_upload_dir, "sample_contract.docx")
    with open(file_path, "wb") as f:
        f.write(sample_docx_bytes)
    return file_path


# ============================================================================
# Mock Service Fixtures
# ============================================================================

class MockArbitrationDetector:
    """Mock arbitration detector for testing."""

    def __init__(self):
        self.detection_results = {}

    def detect_arbitration_clause(self, text: str) -> Dict[str, Any]:
        """Mock detection method."""
        text_lower = text.lower()
        has_arbitration = "arbitration" in text_lower

        return {
            "has_arbitration": has_arbitration,
            "confidence": 0.85 if has_arbitration else 0.15,
            "clause_type": "binding" if has_arbitration else None,
            "keywords": ["arbitration"] if has_arbitration else [],
            "clauses": [{"text": text[:200], "score": 0.85}] if has_arbitration else []
        }

    def detect(self, text: str, document_id: str = None) -> Dict[str, Any]:
        """Mock detect method compatible with ArbitrationDetector."""
        result = self.detect_arbitration_clause(text)
        return {
            "document_id": document_id or "test_doc",
            "has_arbitration": result["has_arbitration"],
            "confidence": result["confidence"],
            "clauses": result.get("clauses", []),
            "summary": {
                "has_binding_arbitration": result["has_arbitration"],
                "has_class_action_waiver": "class action" in text.lower(),
                "has_opt_out": "opt out" in text.lower() or "opt-out" in text.lower()
            },
            "processing_time": 0.1
        }


@pytest.fixture
def mock_arbitration_detector():
    """Mock arbitration detector instance."""
    return MockArbitrationDetector()


class MockDocumentProcessor:
    """Mock document processor for testing."""

    def process_document(self, file_path, filename=None, password=None):
        """Mock document processing."""
        @dataclass
        class MockResult:
            text: str = "Sample extracted text from document."
            structured_content: List = None
            tables: List = None
            images: List = None
            metadata: Dict = None
            processing_time: float = 0.5
            extraction_method: str = "mock"
            errors: List = None
            warnings: List = None

            def __post_init__(self):
                self.structured_content = self.structured_content or []
                self.tables = self.tables or []
                self.images = self.images or []
                self.metadata = self.metadata or {"word_count": 6}
                self.errors = self.errors or []
                self.warnings = self.warnings or []

        return MockResult()


@pytest.fixture
def mock_document_processor():
    """Mock document processor instance."""
    return MockDocumentProcessor()


# ============================================================================
# Performance and Benchmarking Fixtures
# ============================================================================

@pytest.fixture
def performance_metrics() -> Dict[str, float]:
    """Expected performance metrics for benchmarking."""
    return PERFORMANCE_THRESHOLDS.copy()


@pytest.fixture
def benchmark_timer():
    """Timer utility for performance benchmarks."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = 0

        def start(self):
            self.start_time = time.perf_counter()
            return self

        def stop(self):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
            return self.elapsed

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, *args):
            self.stop()

    return Timer()


@pytest.fixture
def large_document_text() -> str:
    """Generate a large document for performance testing."""
    base_text = """
    This is a paragraph of legal text that may or may not contain
    arbitration clauses. It is used for performance testing purposes.
    The document discusses various legal matters including contracts,
    agreements, and dispute resolution mechanisms.
    """
    # Generate ~100KB of text
    return base_text * 500


# ============================================================================
# Pytest Hooks and Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "rag: marks tests as RAG pipeline tests")
    config.addinivalue_line("markers", "document: marks tests as document processing tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip slow tests by default unless explicitly requested
    if not config.getoption("-m"):
        skip_slow = pytest.mark.skip(reason="need -m slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def cleanup_files(temp_upload_dir):
    """Cleanup utility for test files."""
    created_files = []

    def track_file(path):
        created_files.append(path)
        return path

    yield track_file

    # Cleanup
    for file_path in created_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    test_vars = {
        "OPENAI_API_KEY": "test-api-key",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379",
        "DEBUG": "true",
        "TESTING": "true"
    }

    os.environ.update(test_vars)

    yield test_vars

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
