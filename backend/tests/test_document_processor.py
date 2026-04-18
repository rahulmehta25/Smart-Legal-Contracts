"""
Unit tests for document processor service.

Tests document processing for multiple formats:
- PDF extraction
- DOCX extraction
- TXT processing
- HTML processing
- Error handling
"""

import pytest
import os
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import test targets
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.services.document_processor import (
        DocumentProcessor,
        DocumentType,
        StructureType,
        DocumentElement,
        ProcessingResult,
        DocumentMetadata,
        process_document
    )
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DOCUMENT_PROCESSOR_AVAILABLE,
    reason="DocumentProcessor not available"
)


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor(
            extract_images=True,
            preserve_formatting=True,
            detect_structure=True
        )

    @pytest.fixture
    def sample_text_content(self):
        """Sample text content for testing."""
        return """
        TERMS OF SERVICE

        1. INTRODUCTION
        This Terms of Service document outlines the agreement between User and Company.

        2. ARBITRATION CLAUSE
        Any disputes shall be resolved through binding arbitration administered by AAA.

        3. ACCEPTANCE
        By using our services, you accept these terms.
        """

    @pytest.fixture
    def sample_html_content(self):
        """Sample HTML content for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Terms of Service</title></head>
        <body>
            <h1>Terms of Service</h1>
            <p>This document outlines the terms.</p>
            <h2>Arbitration</h2>
            <p>Disputes will be resolved through arbitration.</p>
            <table>
                <tr><th>Section</th><th>Description</th></tr>
                <tr><td>1</td><td>Introduction</td></tr>
            </table>
        </body>
        </html>
        """

    # ========================================================================
    # File Type Detection Tests
    # ========================================================================

    def test_detect_txt_file_type(self, processor, temp_upload_dir, sample_text_content):
        """Test TXT file type detection."""
        file_path = os.path.join(temp_upload_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write(sample_text_content)

        doc_type = processor._detect_file_type(file_path)
        assert doc_type == DocumentType.TXT

    def test_detect_html_file_type(self, processor, temp_upload_dir, sample_html_content):
        """Test HTML file type detection."""
        file_path = os.path.join(temp_upload_dir, "test.html")
        with open(file_path, 'w') as f:
            f.write(sample_html_content)

        doc_type = processor._detect_file_type(file_path)
        assert doc_type in [DocumentType.HTML, DocumentType.TXT]

    def test_detect_file_type_from_bytes(self, processor, sample_pdf_bytes):
        """Test file type detection from bytes."""
        doc_type = processor._detect_file_type(sample_pdf_bytes, filename="test.pdf")
        # May detect as PDF or unknown depending on content
        assert doc_type in [DocumentType.PDF, DocumentType.UNKNOWN]

    def test_detect_file_type_with_filename_hint(self, processor):
        """Test file type detection uses filename as hint."""
        doc_type = processor._detect_file_type(b"plain text content", filename="document.docx")
        assert doc_type == DocumentType.DOCX

    # ========================================================================
    # Text Processing Tests
    # ========================================================================

    def test_process_text_file(self, processor, temp_upload_dir, sample_text_content):
        """Test processing plain text file."""
        file_path = os.path.join(temp_upload_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write(sample_text_content)

        result = processor.process_document(file_path)

        assert isinstance(result, ProcessingResult)
        assert result.text.strip() != ""
        assert result.metadata.word_count > 0
        assert result.extraction_method in ["text_native", "failed"]
        assert len(result.errors) == 0

    def test_process_text_from_bytes(self, processor, sample_text_content):
        """Test processing text from bytes."""
        text_bytes = sample_text_content.encode('utf-8')
        result = processor._process_text(text_bytes)

        assert result.text.strip() != ""
        assert result.metadata.encoding == "utf-8"

    def test_process_text_structure_detection(self, processor, sample_text_content):
        """Test structure detection in text files."""
        text_bytes = sample_text_content.encode('utf-8')
        result = processor._process_text(text_bytes)

        assert len(result.structured_content) > 0
        assert all(isinstance(elem, DocumentElement) for elem in result.structured_content)
        assert all(elem.type == StructureType.PARAGRAPH for elem in result.structured_content)

    def test_process_text_encoding_detection(self, processor, temp_upload_dir):
        """Test encoding detection for text files."""
        # UTF-8 with BOM
        content = "\ufeffUTF-8 content with special chars: \u00e9\u00e8\u00ea"
        file_path = os.path.join(temp_upload_dir, "utf8_bom.txt")
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(content)

        result = processor.process_document(file_path)
        assert "UTF-8" in result.text or "\u00e9" in result.text

    # ========================================================================
    # HTML Processing Tests
    # ========================================================================

    def test_process_html_file(self, processor, temp_upload_dir, sample_html_content):
        """Test processing HTML file."""
        file_path = os.path.join(temp_upload_dir, "test.html")
        with open(file_path, 'w') as f:
            f.write(sample_html_content)

        result = processor._process_html(file_path)

        assert result.text.strip() != ""
        assert "Terms of Service" in result.text
        assert result.metadata.title == "Terms of Service"

    def test_process_html_extracts_headings(self, processor, sample_html_content):
        """Test HTML heading extraction."""
        result = processor._process_html(sample_html_content.encode())

        headings = [elem for elem in result.structured_content
                   if elem.type == StructureType.HEADING]
        assert len(headings) >= 2  # h1 and h2

    def test_process_html_extracts_tables(self, processor, sample_html_content):
        """Test HTML table extraction."""
        result = processor._process_html(sample_html_content.encode())

        assert len(result.tables) >= 1
        assert result.tables[0].headers == ["Section", "Description"]

    def test_process_html_extracts_paragraphs(self, processor, sample_html_content):
        """Test HTML paragraph extraction."""
        result = processor._process_html(sample_html_content.encode())

        paragraphs = [elem for elem in result.structured_content
                     if elem.type == StructureType.PARAGRAPH]
        assert len(paragraphs) >= 2

    # ========================================================================
    # Document Metadata Tests
    # ========================================================================

    def test_metadata_word_count(self, processor, sample_text_content):
        """Test word count in metadata."""
        result = processor._process_text(sample_text_content.encode())

        expected_words = len(sample_text_content.split())
        assert abs(result.metadata.word_count - expected_words) < 5

    def test_metadata_character_count(self, processor, sample_text_content):
        """Test character count in metadata."""
        result = processor._process_text(sample_text_content.encode())

        assert result.metadata.character_count == len(sample_text_content)

    def test_metadata_checksum_generation(self, processor, temp_upload_dir):
        """Test checksum generation for documents."""
        content = "Test content for checksum"
        file_path = os.path.join(temp_upload_dir, "checksum_test.txt")
        with open(file_path, 'w') as f:
            f.write(content)

        result = processor.process_document(file_path)

        assert result.metadata.checksum is not None
        assert len(result.metadata.checksum) == 32  # MD5 hex length

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_process_nonexistent_file(self, processor):
        """Test handling of non-existent file."""
        result = processor.process_document("/nonexistent/path/file.txt")

        assert len(result.errors) > 0
        assert result.extraction_method == "failed"

    def test_process_empty_file(self, processor, temp_upload_dir):
        """Test handling of empty file."""
        file_path = os.path.join(temp_upload_dir, "empty.txt")
        with open(file_path, 'w') as f:
            pass  # Create empty file

        result = processor.process_document(file_path)

        assert result.text == ""
        assert result.metadata.word_count == 0

    def test_process_binary_file_as_text(self, processor, temp_upload_dir):
        """Test handling of binary content as text."""
        file_path = os.path.join(temp_upload_dir, "binary.txt")
        with open(file_path, 'wb') as f:
            f.write(bytes(range(256)))  # Binary content

        result = processor.process_document(file_path)

        # Should handle gracefully with warnings or errors
        assert result is not None

    # ========================================================================
    # Processing Time Tests
    # ========================================================================

    def test_processing_time_recorded(self, processor, temp_upload_dir):
        """Test that processing time is recorded."""
        file_path = os.path.join(temp_upload_dir, "timing_test.txt")
        with open(file_path, 'w') as f:
            f.write("Test content" * 100)

        result = processor.process_document(file_path)

        assert result.processing_time >= 0
        assert result.processing_time < 60  # Should be fast for small files


class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()

    def test_process_document_convenience_function(self, temp_upload_dir):
        """Test the convenience function."""
        file_path = os.path.join(temp_upload_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("Test document content for convenience function.")

        result = process_document(file_path)

        assert isinstance(result, ProcessingResult)
        assert result.text.strip() != ""

    def test_process_multiple_documents(self, processor, temp_upload_dir):
        """Test processing multiple documents in sequence."""
        files = []
        for i in range(5):
            file_path = os.path.join(temp_upload_dir, f"doc_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Document {i} content with unique text.")
            files.append(file_path)

        results = [processor.process_document(f) for f in files]

        assert len(results) == 5
        assert all(r.text.strip() != "" for r in results)

    @pytest.mark.parametrize("encoding", ["utf-8", "latin-1", "ascii"])
    def test_process_various_encodings(self, processor, temp_upload_dir, encoding):
        """Test processing files with various encodings."""
        try:
            content = "Test content for encoding test"
            file_path = os.path.join(temp_upload_dir, f"test_{encoding}.txt")
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)

            result = processor.process_document(file_path)

            assert content in result.text
        except UnicodeEncodeError:
            pytest.skip(f"Content cannot be encoded with {encoding}")


class TestTableDetection:
    """Tests for table header detection."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()

    def test_looks_like_header_short_cells(self, processor):
        """Test header detection with short cell content."""
        row_cells = ["Name", "Age", "City"]
        assert processor._looks_like_header(row_cells) is True

    def test_looks_like_header_long_cells(self, processor):
        """Test header detection with long cell content."""
        row_cells = [
            "This is a very long description that would not typically be a header",
            "Another long description with lots of details",
            "Yet another long description"
        ]
        assert processor._looks_like_header(row_cells) is False

    def test_looks_like_header_empty_cells(self, processor):
        """Test header detection with mostly empty cells."""
        row_cells = ["Header", "", "", ""]
        assert processor._looks_like_header(row_cells) is False

    def test_looks_like_header_empty_list(self, processor):
        """Test header detection with empty list."""
        assert processor._looks_like_header([]) is False


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_document_type_values(self):
        """Test DocumentType enum values."""
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.DOCX.value == "docx"
        assert DocumentType.TXT.value == "txt"
        assert DocumentType.HTML.value == "html"
        assert DocumentType.UNKNOWN.value == "unknown"

    def test_document_type_comparison(self):
        """Test DocumentType enum comparison."""
        assert DocumentType.PDF != DocumentType.DOCX
        assert DocumentType.TXT == DocumentType.TXT


class TestStructureType:
    """Tests for StructureType enum."""

    def test_structure_type_values(self):
        """Test StructureType enum values."""
        assert StructureType.PARAGRAPH.value == "paragraph"
        assert StructureType.HEADING.value == "heading"
        assert StructureType.TABLE.value == "table"
        assert StructureType.LIST.value == "list"


class TestDocumentElement:
    """Tests for DocumentElement dataclass."""

    def test_document_element_creation(self):
        """Test DocumentElement creation."""
        elem = DocumentElement(
            type=StructureType.PARAGRAPH,
            content="Test paragraph content",
            level=0,
            position=0
        )

        assert elem.type == StructureType.PARAGRAPH
        assert elem.content == "Test paragraph content"
        assert elem.level == 0
        assert elem.position == 0
        assert elem.metadata == {}

    def test_document_element_with_metadata(self):
        """Test DocumentElement with metadata."""
        elem = DocumentElement(
            type=StructureType.HEADING,
            content="Chapter 1",
            level=1,
            position=0,
            metadata={"tag": "h1", "id": "chapter-1"}
        )

        assert elem.metadata["tag"] == "h1"
        assert elem.metadata["id"] == "chapter-1"


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_processing_result_creation(self):
        """Test ProcessingResult creation."""
        result = ProcessingResult(
            text="Test content",
            structured_content=[],
            tables=[],
            images=[],
            metadata=DocumentMetadata(),
            processing_time=0.5,
            extraction_method="test"
        )

        assert result.text == "Test content"
        assert result.processing_time == 0.5
        assert result.errors == []
        assert result.warnings == []

    def test_processing_result_with_errors(self):
        """Test ProcessingResult with errors."""
        result = ProcessingResult(
            text="",
            structured_content=[],
            tables=[],
            images=[],
            metadata=DocumentMetadata(),
            processing_time=0.1,
            extraction_method="failed",
            errors=["File not found", "Permission denied"]
        )

        assert len(result.errors) == 2
        assert "File not found" in result.errors
