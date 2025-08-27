"""
PDF Processing Service

Provides comprehensive PDF processing capabilities including:
- Text extraction from native PDFs
- OCR for scanned/image-based PDFs  
- Layout preservation and structure detection
- Multi-page handling with page-level metadata
- Error handling and validation
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import fitz  # PyMuPDF for advanced PDF processing
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PDFQuality(Enum):
    """PDF quality levels for OCR processing"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class PageMetadata:
    """Metadata for a single PDF page"""
    page_number: int
    width: float
    height: float
    rotation: int
    has_text: bool
    has_images: bool
    word_count: int
    confidence_score: Optional[float] = None
    processing_method: str = "native"


@dataclass
class PDFMetadata:
    """Complete PDF document metadata"""
    title: Optional[str]
    author: Optional[str]
    subject: Optional[str]
    creator: Optional[str]
    producer: Optional[str]
    creation_date: Optional[str]
    modification_date: Optional[str]
    pages: int
    encrypted: bool
    file_size: int
    language: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of PDF text extraction"""
    text: str
    pages: List[Dict]
    metadata: PDFMetadata
    page_metadata: List[PageMetadata]
    extraction_method: str
    processing_time: float
    errors: List[str]
    warnings: List[str]


class PDFProcessor:
    """Advanced PDF processing with multiple extraction methods"""
    
    def __init__(self, 
                 tesseract_config: Optional[str] = None,
                 quality: PDFQuality = PDFQuality.MEDIUM):
        """
        Initialize PDF processor
        
        Args:
            tesseract_config: Custom Tesseract configuration
            quality: OCR quality level
        """
        self.tesseract_config = tesseract_config or '--oem 3 --psm 6'
        self.quality = quality
        self.dpi_settings = {
            PDFQuality.LOW: 150,
            PDFQuality.MEDIUM: 200,
            PDFQuality.HIGH: 300,
            PDFQuality.ULTRA: 600
        }
    
    def extract_text(self, 
                    pdf_path: Union[str, Path, bytes], 
                    password: Optional[str] = None,
                    preserve_layout: bool = True,
                    extract_images: bool = False) -> ExtractionResult:
        """
        Extract text from PDF using multiple methods
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            password: PDF password if encrypted
            preserve_layout: Whether to preserve text layout
            extract_images: Whether to extract images from PDF
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        import time
        start_time = time.time()
        
        errors = []
        warnings = []
        
        try:
            # Try native extraction first
            result = self._extract_native(pdf_path, password, preserve_layout)
            
            # If native extraction yields poor results, try OCR
            if self._needs_ocr(result):
                warnings.append("Native extraction yielded poor results, falling back to OCR")
                ocr_result = self._extract_ocr(pdf_path, password)
                
                # Use OCR result if it's better
                if len(ocr_result.text.strip()) > len(result.text.strip()):
                    result = ocr_result
                    result.extraction_method = "ocr_fallback"
            
            # Extract images if requested
            if extract_images:
                try:
                    images = self._extract_images(pdf_path, password)
                    result.images = images
                except Exception as e:
                    warnings.append(f"Image extraction failed: {str(e)}")
            
            result.processing_time = time.time() - start_time
            result.errors = errors
            result.warnings = warnings
            
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            errors.append(str(e))
            
            # Return minimal result on failure
            return ExtractionResult(
                text="",
                pages=[],
                metadata=PDFMetadata(
                    title=None, author=None, subject=None, creator=None,
                    producer=None, creation_date=None, modification_date=None,
                    pages=0, encrypted=False, file_size=0
                ),
                page_metadata=[],
                extraction_method="failed",
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings
            )
    
    def _extract_native(self, 
                       pdf_path: Union[str, Path, bytes], 
                       password: Optional[str] = None,
                       preserve_layout: bool = True) -> ExtractionResult:
        """Extract text using native PDF text content"""
        
        pages = []
        page_metadata = []
        all_text = []
        
        # Handle both file paths and bytes
        if isinstance(pdf_path, (str, Path)):
            with open(pdf_path, 'rb') as file:
                pdf_data = file.read()
        else:
            pdf_data = pdf_path
        
        # Use pdfplumber for better layout preservation
        if preserve_layout:
            with io.BytesIO(pdf_data) as pdf_file:
                with pdfplumber.open(pdf_file, password=password) as pdf:
                    metadata = self._extract_metadata_pdfplumber(pdf)
                    
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or ""
                        
                        # Extract tables separately if present
                        tables = page.extract_tables()
                        if tables:
                            table_text = self._format_tables(tables)
                            page_text += "\n\n" + table_text
                        
                        pages.append({
                            'page_number': page_num + 1,
                            'text': page_text,
                            'tables': len(tables),
                            'bbox': page.bbox
                        })
                        
                        page_metadata.append(PageMetadata(
                            page_number=page_num + 1,
                            width=page.width,
                            height=page.height,
                            rotation=page.rotation or 0,
                            has_text=bool(page_text.strip()),
                            has_images=bool(page.images),
                            word_count=len(page_text.split()),
                            processing_method="native_pdfplumber"
                        ))
                        
                        all_text.append(page_text)
        
        else:
            # Use PyPDF2 for simple text extraction
            with io.BytesIO(pdf_data) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                
                if reader.is_encrypted and password:
                    reader.decrypt(password)
                
                metadata = self._extract_metadata_pypdf2(reader)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        pages.append({
                            'page_number': page_num + 1,
                            'text': page_text,
                            'mediabox': page.mediabox
                        })
                        
                        page_metadata.append(PageMetadata(
                            page_number=page_num + 1,
                            width=float(page.mediabox.width),
                            height=float(page.mediabox.height),
                            rotation=page.rotation,
                            has_text=bool(page_text.strip()),
                            has_images=False,  # PyPDF2 doesn't easily detect images
                            word_count=len(page_text.split()),
                            processing_method="native_pypdf2"
                        ))
                        
                        all_text.append(page_text)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        pages.append({
                            'page_number': page_num + 1,
                            'text': "",
                            'error': str(e)
                        })
        
        return ExtractionResult(
            text="\n\n".join(all_text),
            pages=pages,
            metadata=metadata,
            page_metadata=page_metadata,
            extraction_method="native",
            processing_time=0,  # Will be set by caller
            errors=[],
            warnings=[]
        )
    
    def _extract_ocr(self, 
                    pdf_path: Union[str, Path, bytes], 
                    password: Optional[str] = None) -> ExtractionResult:
        """Extract text using OCR for scanned PDFs"""
        
        pages = []
        page_metadata = []
        all_text = []
        
        try:
            # Convert PDF pages to images
            if isinstance(pdf_path, (str, Path)):
                images = convert_from_path(
                    pdf_path, 
                    dpi=self.dpi_settings[self.quality],
                    fmt='jpeg'
                )
            else:
                images = convert_from_bytes(
                    pdf_path,
                    dpi=self.dpi_settings[self.quality],
                    fmt='jpeg'
                )
            
            # Get metadata using PyMuPDF
            if isinstance(pdf_path, (str, Path)):
                doc = fitz.open(pdf_path)
            else:
                doc = fitz.open(stream=pdf_path, filetype="pdf")
            
            metadata = self._extract_metadata_pymupdf(doc)
            
            for page_num, image in enumerate(images):
                try:
                    # Preprocess image for better OCR
                    processed_image = self._preprocess_image(image)
                    
                    # Perform OCR
                    ocr_data = pytesseract.image_to_data(
                        processed_image, 
                        config=self.tesseract_config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text and confidence
                    page_text = pytesseract.image_to_string(
                        processed_image,
                        config=self.tesseract_config
                    )
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    pages.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'confidence': avg_confidence,
                        'word_count': len([w for w in ocr_data['text'] if w.strip()])
                    })
                    
                    # Get page dimensions from original PDF
                    pdf_page = doc.load_page(page_num)
                    rect = pdf_page.rect
                    
                    page_metadata.append(PageMetadata(
                        page_number=page_num + 1,
                        width=rect.width,
                        height=rect.height,
                        rotation=pdf_page.rotation,
                        has_text=bool(page_text.strip()),
                        has_images=True,  # OCR assumes image-based content
                        word_count=len(page_text.split()),
                        confidence_score=avg_confidence,
                        processing_method="ocr"
                    ))
                    
                    all_text.append(page_text)
                    
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    pages.append({
                        'page_number': page_num + 1,
                        'text': "",
                        'error': str(e)
                    })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise
        
        return ExtractionResult(
            text="\n\n".join(all_text),
            pages=pages,
            metadata=metadata,
            page_metadata=page_metadata,
            extraction_method="ocr",
            processing_time=0,
            errors=[],
            warnings=[]
        )
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Apply adaptive thresholding for better text contrast
        thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        return Image.fromarray(thresh)
    
    def _needs_ocr(self, result: ExtractionResult) -> bool:
        """Determine if OCR is needed based on native extraction results"""
        
        # If no text extracted at all
        if not result.text.strip():
            return True
        
        # If text is very short compared to number of pages
        words_per_page = len(result.text.split()) / max(1, result.metadata.pages)
        if words_per_page < 10:
            return True
        
        # If many pages have no text
        pages_with_text = sum(1 for pm in result.page_metadata if pm.has_text)
        if pages_with_text / max(1, result.metadata.pages) < 0.5:
            return True
        
        return False
    
    def _extract_images(self, 
                       pdf_path: Union[str, Path, bytes], 
                       password: Optional[str] = None) -> List[Dict]:
        """Extract images from PDF"""
        
        images = []
        
        try:
            if isinstance(pdf_path, (str, Path)):
                doc = fitz.open(pdf_path)
            else:
                doc = fitz.open(stream=pdf_path, filetype="pdf")
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    images.append({
                        'page_number': page_num + 1,
                        'image_index': img_index,
                        'width': base_image['width'],
                        'height': base_image['height'],
                        'ext': base_image['ext'],
                        'size': len(base_image['image']),
                        'xref': xref
                    })
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
        
        return images
    
    def _format_tables(self, tables: List[List[List[str]]]) -> str:
        """Format extracted tables as text"""
        formatted_tables = []
        
        for table in tables:
            if not table:
                continue
            
            # Convert table to string format
            table_text = []
            for row in table:
                if row:  # Skip empty rows
                    clean_row = [cell or "" for cell in row]
                    table_text.append(" | ".join(clean_row))
            
            if table_text:
                formatted_tables.append("\n".join(table_text))
        
        return "\n\n".join(formatted_tables)
    
    def _extract_metadata_pdfplumber(self, pdf) -> PDFMetadata:
        """Extract metadata using pdfplumber"""
        metadata = pdf.metadata or {}
        
        return PDFMetadata(
            title=metadata.get('Title'),
            author=metadata.get('Author'),
            subject=metadata.get('Subject'),
            creator=metadata.get('Creator'),
            producer=metadata.get('Producer'),
            creation_date=str(metadata.get('CreationDate')) if metadata.get('CreationDate') else None,
            modification_date=str(metadata.get('ModDate')) if metadata.get('ModDate') else None,
            pages=len(pdf.pages),
            encrypted=False,  # pdfplumber handles decryption
            file_size=0  # Not available in pdfplumber
        )
    
    def _extract_metadata_pypdf2(self, reader: PyPDF2.PdfReader) -> PDFMetadata:
        """Extract metadata using PyPDF2"""
        metadata = reader.metadata or {}
        
        return PDFMetadata(
            title=metadata.get('/Title'),
            author=metadata.get('/Author'),
            subject=metadata.get('/Subject'),
            creator=metadata.get('/Creator'),
            producer=metadata.get('/Producer'),
            creation_date=str(metadata.get('/CreationDate')) if metadata.get('/CreationDate') else None,
            modification_date=str(metadata.get('/ModDate')) if metadata.get('/ModDate') else None,
            pages=len(reader.pages),
            encrypted=reader.is_encrypted,
            file_size=0
        )
    
    def _extract_metadata_pymupdf(self, doc) -> PDFMetadata:
        """Extract metadata using PyMuPDF"""
        metadata = doc.metadata
        
        return PDFMetadata(
            title=metadata.get('title'),
            author=metadata.get('author'),
            subject=metadata.get('subject'),
            creator=metadata.get('creator'),
            producer=metadata.get('producer'),
            creation_date=metadata.get('creationDate'),
            modification_date=metadata.get('modDate'),
            pages=doc.page_count,
            encrypted=doc.needs_pass,
            file_size=0
        )


# Convenience functions
def extract_pdf_text(pdf_path: Union[str, Path, bytes], 
                    password: Optional[str] = None,
                    quality: PDFQuality = PDFQuality.MEDIUM) -> ExtractionResult:
    """
    Extract text from PDF using the best available method
    
    Args:
        pdf_path: Path to PDF file or PDF bytes
        password: PDF password if encrypted
        quality: OCR quality level
        
    Returns:
        ExtractionResult with extracted text and metadata
    """
    processor = PDFProcessor(quality=quality)
    return processor.extract_text(pdf_path, password)


def extract_pdf_text_simple(pdf_path: Union[str, Path, bytes]) -> str:
    """
    Simple text extraction returning only the text content
    
    Args:
        pdf_path: Path to PDF file or PDF bytes
        
    Returns:
        Extracted text as string
    """
    result = extract_pdf_text(pdf_path)
    return result.text