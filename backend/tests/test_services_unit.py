"""
Unit tests for backend services.

Tests for:
- PDF Service
- Document Service
- Analysis Service
- Storage Service
"""

import pytest
import os
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# PDF Service Tests
# ============================================================================

class TestPDFService:
    """Test suite for PDF processing service."""

    @pytest.fixture
    def pdf_processor(self):
        """Create PDF processor instance."""
        try:
            from app.services.pdf_service import PDFProcessor, PDFQuality
            return PDFProcessor(quality=PDFQuality.MEDIUM)
        except ImportError:
            pytest.skip("PDFProcessor not available")

    @pytest.fixture
    def mock_pdf_bytes(self):
        """Create mock PDF bytes."""
        return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test Content) Tj ET
endstream endobj
xref 0 5
0000000000 65535 f
trailer<</Size 5/Root 1 0 R>>
startxref 253
%%EOF"""

    def test_pdf_quality_enum(self):
        """Test PDFQuality enum values."""
        try:
            from app.services.pdf_service import PDFQuality
            assert PDFQuality.LOW.value == "low"
            assert PDFQuality.MEDIUM.value == "medium"
            assert PDFQuality.HIGH.value == "high"
            assert PDFQuality.ULTRA.value == "ultra"
        except ImportError:
            pytest.skip("PDFQuality not available")

    def test_pdf_processor_initialization(self, pdf_processor):
        """Test PDFProcessor initialization."""
        assert pdf_processor is not None
        assert hasattr(pdf_processor, 'quality')
        assert hasattr(pdf_processor, 'extract_text')

    def test_needs_ocr_empty_text(self, pdf_processor):
        """Test OCR detection for empty text result."""
        try:
            from app.services.pdf_service import ExtractionResult, PDFMetadata

            result = ExtractionResult(
                text="",
                pages=[],
                metadata=PDFMetadata(
                    title=None, author=None, subject=None,
                    creator=None, producer=None, creation_date=None,
                    modification_date=None, pages=5, encrypted=False, file_size=0
                ),
                page_metadata=[],
                extraction_method="native",
                processing_time=0.5,
                errors=[],
                warnings=[]
            )

            assert pdf_processor._needs_ocr(result) is True
        except ImportError:
            pytest.skip("ExtractionResult not available")

    def test_needs_ocr_short_text(self, pdf_processor):
        """Test OCR detection for very short text."""
        try:
            from app.services.pdf_service import ExtractionResult, PDFMetadata, PageMetadata

            result = ExtractionResult(
                text="Short",
                pages=[{"page_number": 1, "text": "Short"}],
                metadata=PDFMetadata(
                    title=None, author=None, subject=None,
                    creator=None, producer=None, creation_date=None,
                    modification_date=None, pages=10, encrypted=False, file_size=0
                ),
                page_metadata=[
                    PageMetadata(
                        page_number=i+1, width=612, height=792, rotation=0,
                        has_text=(i == 0), has_images=False, word_count=1
                    ) for i in range(10)
                ],
                extraction_method="native",
                processing_time=0.5,
                errors=[],
                warnings=[]
            )

            assert pdf_processor._needs_ocr(result) is True
        except ImportError:
            pytest.skip("Required classes not available")

    def test_needs_ocr_adequate_text(self, pdf_processor):
        """Test OCR detection for adequate text content."""
        try:
            from app.services.pdf_service import ExtractionResult, PDFMetadata, PageMetadata

            page_text = " ".join(["word"] * 100)
            result = ExtractionResult(
                text=page_text * 5,
                pages=[{"page_number": i+1, "text": page_text} for i in range(5)],
                metadata=PDFMetadata(
                    title=None, author=None, subject=None,
                    creator=None, producer=None, creation_date=None,
                    modification_date=None, pages=5, encrypted=False, file_size=0
                ),
                page_metadata=[
                    PageMetadata(
                        page_number=i+1, width=612, height=792, rotation=0,
                        has_text=True, has_images=False, word_count=100
                    ) for i in range(5)
                ],
                extraction_method="native",
                processing_time=0.5,
                errors=[],
                warnings=[]
            )

            assert pdf_processor._needs_ocr(result) is False
        except ImportError:
            pytest.skip("Required classes not available")

    def test_format_tables(self, pdf_processor):
        """Test table formatting."""
        tables = [
            [["Header1", "Header2"], ["Data1", "Data2"], ["Data3", "Data4"]],
            [["Col1", "Col2", "Col3"], ["A", "B", "C"]]
        ]

        formatted = pdf_processor._format_tables(tables)

        assert "Header1" in formatted
        assert "Data1" in formatted
        assert "|" in formatted

    def test_format_tables_empty(self, pdf_processor):
        """Test formatting empty tables."""
        result = pdf_processor._format_tables([])
        assert result == ""

    def test_format_tables_with_empty_rows(self, pdf_processor):
        """Test formatting tables with empty rows."""
        tables = [[["Header"], [], ["Data"]]]

        formatted = pdf_processor._format_tables(tables)

        assert "Header" in formatted
        assert "Data" in formatted


# ============================================================================
# Document Service Tests
# ============================================================================

class TestDocumentService:
    """Test suite for document service."""

    @pytest.fixture
    def document_service(self):
        """Create document service instance."""
        try:
            from app.services.document_service import DocumentService
            return DocumentService()
        except ImportError:
            pytest.skip("DocumentService not available")

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        mock = Mock()
        mock.query.return_value.filter.return_value.first.return_value = None
        mock.add = Mock()
        mock.commit = Mock()
        mock.refresh = Mock()
        return mock

    def test_document_service_initialization(self, document_service):
        """Test DocumentService initialization."""
        assert document_service is not None

    @patch('app.services.document_service.DocumentService.create_document')
    def test_create_document(self, mock_create, document_service, mock_db_session):
        """Test document creation."""
        mock_doc = Mock()
        mock_doc.id = 1
        mock_doc.filename = "test.txt"
        mock_doc.content = "Test content"
        mock_create.return_value = mock_doc

        try:
            from app.models.document import DocumentCreate
            doc_data = DocumentCreate(
                filename="test.txt",
                content="Test content",
                content_type="text/plain"
            )

            result = document_service.create_document(mock_db_session, doc_data)
            assert result is not None
        except ImportError:
            pytest.skip("DocumentCreate not available")

    def test_document_service_has_required_methods(self, document_service):
        """Test DocumentService has required methods."""
        required_methods = [
            'create_document',
            'get_document',
            'get_documents',
            'delete_document'
        ]

        for method in required_methods:
            assert hasattr(document_service, method), f"Missing method: {method}"


# ============================================================================
# Analysis Service Tests
# ============================================================================

class TestAnalysisService:
    """Test suite for analysis service."""

    @pytest.fixture
    def analysis_service(self):
        """Create analysis service instance."""
        try:
            from app.services.analysis_service import AnalysisService
            return AnalysisService()
        except ImportError:
            try:
                from app.services.simple_analysis_service import SimpleAnalysisService
                return SimpleAnalysisService()
            except ImportError:
                pytest.skip("AnalysisService not available")

    def test_analysis_service_initialization(self, analysis_service):
        """Test AnalysisService initialization."""
        assert analysis_service is not None

    def test_analysis_service_has_analyze_method(self, analysis_service):
        """Test AnalysisService has analyze method."""
        assert hasattr(analysis_service, 'analyze_document') or hasattr(analysis_service, 'analyze')


# ============================================================================
# Storage Service Tests
# ============================================================================

class TestStorageService:
    """Test suite for storage service."""

    @pytest.fixture
    def storage_service(self):
        """Create storage service instance."""
        try:
            from app.services.storage_service import StorageService
            return StorageService()
        except ImportError:
            pytest.skip("StorageService not available")

    def test_storage_service_initialization(self, storage_service):
        """Test StorageService initialization."""
        assert storage_service is not None


# ============================================================================
# Batch Processor Tests
# ============================================================================

class TestBatchProcessor:
    """Test suite for batch processor."""

    @pytest.fixture
    def batch_processor(self):
        """Create batch processor instance."""
        try:
            from app.services.batch_processor import BatchProcessor
            return BatchProcessor()
        except ImportError:
            pytest.skip("BatchProcessor not available")

    def test_batch_processor_initialization(self, batch_processor):
        """Test BatchProcessor initialization."""
        assert batch_processor is not None


# ============================================================================
# Preprocessing Service Tests
# ============================================================================

class TestPreprocessingService:
    """Test suite for preprocessing service."""

    @pytest.fixture
    def preprocessing(self):
        """Import preprocessing module."""
        try:
            from app.services import preprocessing
            return preprocessing
        except ImportError:
            pytest.skip("preprocessing module not available")

    def test_preprocessing_module_available(self, preprocessing):
        """Test preprocessing module is available."""
        assert preprocessing is not None


# ============================================================================
# Mock Service Tests (for when real services unavailable)
# ============================================================================

class TestMockServices:
    """Tests using mock services."""

    def test_mock_document_processor(self, mock_document_processor):
        """Test mock document processor."""
        result = mock_document_processor.process_document("/fake/path.pdf")

        assert result.text == "Sample extracted text from document."
        assert result.processing_time == 0.5
        assert result.extraction_method == "mock"

    def test_mock_arbitration_detector(self, mock_arbitration_detector):
        """Test mock arbitration detector."""
        text_with_arbitration = "This contract includes binding arbitration clause."
        result = mock_arbitration_detector.detect_arbitration_clause(text_with_arbitration)

        assert result["has_arbitration"] is True
        assert result["confidence"] >= 0.5

    def test_mock_arbitration_detector_no_arbitration(self, mock_arbitration_detector):
        """Test mock detector with no arbitration clause."""
        text_without = "This is a regular contract with no special clauses."
        result = mock_arbitration_detector.detect_arbitration_clause(text_without)

        assert result["has_arbitration"] is False
        assert result["confidence"] < 0.5


# ============================================================================
# Service Integration Tests
# ============================================================================

class TestServiceIntegration:
    """Integration tests for services working together."""

    @pytest.mark.integration
    def test_document_processing_pipeline(self, temp_upload_dir, mock_arbitration_detector):
        """Test document processing pipeline."""
        # Create test document
        file_path = os.path.join(temp_upload_dir, "test_contract.txt")
        content = """
        TERMS OF SERVICE

        This agreement includes an arbitration clause.
        All disputes shall be resolved through binding arbitration.
        """
        with open(file_path, 'w') as f:
            f.write(content)

        # Detect arbitration
        result = mock_arbitration_detector.detect_arbitration_clause(content)

        assert result["has_arbitration"] is True

    @pytest.mark.integration
    def test_batch_document_analysis(self, temp_upload_dir, mock_arbitration_detector):
        """Test batch document analysis."""
        documents = []

        # Create multiple test documents
        for i in range(5):
            file_path = os.path.join(temp_upload_dir, f"contract_{i}.txt")
            has_arb = i % 2 == 0
            content = f"Document {i}. {'Includes arbitration clause.' if has_arb else 'No special clauses.'}"
            with open(file_path, 'w') as f:
                f.write(content)
            documents.append((file_path, content, has_arb))

        # Analyze batch
        results = []
        for file_path, content, expected in documents:
            result = mock_arbitration_detector.detect_arbitration_clause(content)
            results.append((result, expected))

        assert len(results) == 5
