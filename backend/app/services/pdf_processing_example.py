"""
PDF Processing Example and Testing Script

Demonstrates the complete PDF processing pipeline with examples
for all implemented services:
- PDF text extraction with OCR fallback
- Document processing for multiple formats
- Text preprocessing and analysis
- File storage and retrieval
- Batch processing with Celery
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any
import tempfile
import logging

# Import our services
from .pdf_service import PDFProcessor, PDFQuality, extract_pdf_text
from .document_processor import DocumentProcessor, process_document
from .preprocessing import TextPreprocessor, preprocess_text
from .storage_service import StorageService, StorageConfig, StorageBackend, create_storage_service
from .batch_processor import BatchProcessor, BatchJobConfig, get_batch_processor

logger = logging.getLogger(__name__)


class PDFProcessingDemo:
    """Comprehensive demonstration of PDF processing capabilities"""
    
    def __init__(self):
        """Initialize demo with all services"""
        # Initialize services
        self.pdf_processor = PDFProcessor(quality=PDFQuality.MEDIUM)
        self.document_processor = DocumentProcessor(
            extract_images=True,
            preserve_formatting=True,
            detect_structure=True
        )
        self.text_preprocessor = TextPreprocessor(
            target_language='en',
            clean_ocr_artifacts=True,
            normalize_whitespace=True
        )
        
        # Initialize storage service (local for demo)
        self.storage_service = create_storage_service(
            backend=StorageBackend.LOCAL,
            local_base_path="./demo_storage",
            enable_thumbnails=True,
            max_file_size=50 * 1024 * 1024  # 50MB
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            broker_url='redis://localhost:6379/0',
            backend_url='redis://localhost:6379/0'
        )
    
    def run_comprehensive_demo(self):
        """Run complete demonstration of all features"""
        
        print("🚀 PDF Processing System - Comprehensive Demo")
        print("=" * 60)
        
        # Create sample PDF content for testing
        sample_files = self._create_sample_files()
        
        # Demo 1: Basic PDF text extraction
        print("\n📄 Demo 1: Basic PDF Text Extraction")
        self._demo_pdf_extraction(sample_files.get('pdf'))
        
        # Demo 2: Multi-format document processing
        print("\n📚 Demo 2: Multi-format Document Processing")
        self._demo_document_processing(sample_files)
        
        # Demo 3: Text preprocessing and analysis
        print("\n🔧 Demo 3: Text Preprocessing and Analysis")
        self._demo_text_preprocessing()
        
        # Demo 4: File storage operations
        print("\n💾 Demo 4: File Storage and Management")
        self._demo_storage_operations(sample_files)
        
        # Demo 5: Batch processing
        print("\n⚡ Demo 5: Batch Processing")
        self._demo_batch_processing(sample_files)
        
        print("\n✅ Demo completed successfully!")
        print("Check the generated files and logs for detailed results.")
    
    def _create_sample_files(self) -> Dict[str, str]:
        """Create sample files for testing"""
        
        sample_files = {}
        
        # Sample PDF content (we'll create a simple text file for demo)
        pdf_content = """
        TERMS OF SERVICE AGREEMENT
        
        1. ARBITRATION CLAUSE
        Any disputes arising under this agreement shall be resolved through binding arbitration.
        The arbitration shall be conducted under the rules of the American Arbitration Association.
        
        2. USER RESPONSIBILITIES
        Users must comply with all applicable laws and regulations when using our service.
        
        3. LIMITATION OF LIABILITY
        Our liability is limited to the maximum extent permitted by law.
        
        4. GOVERNING LAW
        This agreement shall be governed by the laws of California.
        """
        
        # Create temporary PDF file (in real scenario, this would be actual PDF)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False, encoding='utf-8') as f:
            f.write(pdf_content)
            sample_files['pdf'] = f.name
        
        # Sample Word document content
        docx_content = """
        PRIVACY POLICY
        
        Data Collection
        We collect personal information to provide our services.
        
        Dispute Resolution
        All disputes will be resolved through mandatory arbitration in Delaware.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(docx_content)
            sample_files['docx'] = f.name
        
        # Sample HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Software License</title></head>
        <body>
        <h1>SOFTWARE LICENSE AGREEMENT</h1>
        <h2>Terms and Conditions</h2>
        <p>This software is licensed under the following terms:</p>
        <ul>
            <li>Users may not reverse engineer the software</li>
            <li>Disputes shall be resolved through arbitration</li>
            <li>The license is non-transferable</li>
        </ul>
        <h2>Arbitration Provision</h2>
        <p>Any legal disputes arising from the use of this software shall be settled through binding arbitration administered by JAMS.</p>
        </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            sample_files['html'] = f.name
        
        return sample_files
    
    def _demo_pdf_extraction(self, pdf_file: str):
        """Demonstrate PDF text extraction capabilities"""
        
        if not pdf_file:
            print("⚠️  No PDF file available for demo")
            return
        
        print(f"Processing file: {pdf_file}")
        
        try:
            # Basic extraction
            start_time = time.time()
            result = self.pdf_processor.extract_text(pdf_file, preserve_layout=True)
            processing_time = time.time() - start_time
            
            print(f"✅ Text extracted successfully")
            print(f"   Method: {result.extraction_method}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Pages processed: {len(result.pages)}")
            print(f"   Text length: {len(result.text)} characters")
            print(f"   Word count: {len(result.text.split())}")
            
            if result.warnings:
                print(f"   Warnings: {', '.join(result.warnings)}")
            
            if result.errors:
                print(f"   Errors: {', '.join(result.errors)}")
            
            # Show extract of text
            preview = result.text[:200].replace('\n', ' ').strip()
            print(f"   Preview: {preview}...")
            
            # Test different quality levels
            print("\n🔍 Testing OCR quality levels:")
            for quality in [PDFQuality.LOW, PDFQuality.HIGH]:
                processor = PDFProcessor(quality=quality)
                result_qual = processor.extract_text(pdf_file)
                print(f"   {quality.value}: {len(result_qual.text)} chars, {result_qual.processing_time:.2f}s")
        
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
    
    def _demo_document_processing(self, sample_files: Dict[str, str]):
        """Demonstrate multi-format document processing"""
        
        for file_type, file_path in sample_files.items():
            if not file_path:
                continue
            
            print(f"\nProcessing {file_type.upper()} file...")
            
            try:
                start_time = time.time()
                result = self.document_processor.process_document(file_path)
                processing_time = time.time() - start_time
                
                print(f"✅ {file_type.upper()} processed successfully")
                print(f"   File type detected: {result.metadata.file_type.value}")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Text length: {len(result.text)} characters")
                print(f"   Sections found: {len(result.structured_content)}")
                print(f"   Tables found: {len(result.tables)}")
                print(f"   Images found: {len(result.images)}")
                
                # Show structure analysis
                if result.structured_content:
                    print("   Document structure:")
                    for i, section in enumerate(result.structured_content[:3]):  # Show first 3 sections
                        print(f"     {i+1}. {section.type.value}: {section.content[:50]}...")
                
                if result.warnings:
                    print(f"   Warnings: {', '.join(result.warnings)}")
            
            except Exception as e:
                print(f"❌ {file_type.upper()} processing failed: {e}")
    
    def _demo_text_preprocessing(self):
        """Demonstrate text preprocessing and analysis"""
        
        sample_text = """
        This is a sample legal document with various issues that need preprocessing.
        
        It contains    multiple   spaces,  OCR  errors  like  'rn'  instead  of  'in',
        and inconsistent formatting.
        
        The document discusses arbitration clauses and dispute resolution mechanisms.
        Users should understand that binding arbitration may limit their legal rights.
        
        This  text  also  has  some  encoding  issues  like  â€™  and  other  artifacts
        from poor OCR processing.
        """
        
        print("Processing sample text with common issues...")
        
        try:
            start_time = time.time()
            result = self.text_preprocessor.preprocess(sample_text)
            processing_time = time.time() - start_time
            
            print(f"✅ Text preprocessing completed")
            print(f"   Processing time: {processing_time:.3f}s")
            print(f"   Detected language: {result.detected_language} ({result.statistics.language_confidence:.2f})")
            print(f"   Content type: {result.content_type.value}")
            print(f"   Text quality: {result.text_quality.value}")
            
            print(f"\n📊 Text Statistics:")
            stats = result.statistics
            print(f"   Words: {stats.word_count}")
            print(f"   Sentences: {stats.sentence_count}")
            print(f"   Paragraphs: {stats.paragraph_count}")
            print(f"   Avg words/sentence: {stats.avg_words_per_sentence:.1f}")
            print(f"   Readability score: {stats.readability_score:.1f}")
            print(f"   Lexical diversity: {stats.lexical_diversity:.2f}")
            
            if stats.most_common_words:
                common_words = [f"{word}({count})" for word, count in stats.most_common_words[:5]]
                print(f"   Most common words: {', '.join(common_words)}")
            
            print(f"\n🔧 Preprocessing Actions:")
            if result.encoding_issues_fixed:
                print(f"   Encoding fixes: {', '.join(result.encoding_issues_fixed)}")
            if result.warnings:
                print(f"   Warnings: {', '.join(result.warnings)}")
            
            print(f"\n📝 Sections Detected: {len(result.sections)}")
            for i, section in enumerate(result.sections[:3]):
                print(f"   {i+1}. {section.section_type} (Level {section.level})")
                if section.title:
                    print(f"      Title: {section.title}")
                print(f"      Content: {section.content[:80]}...")
        
        except Exception as e:
            print(f"❌ Text preprocessing failed: {e}")
    
    def _demo_storage_operations(self, sample_files: Dict[str, str]):
        """Demonstrate file storage and management"""
        
        print("Testing file storage operations...")
        
        uploaded_files = []
        
        # Upload files
        for file_type, file_path in sample_files.items():
            if not file_path:
                continue
            
            try:
                print(f"\nUploading {file_type} file...")
                
                # Read file content
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Upload to storage
                filename = f"demo_{file_type}_document.{file_type}"
                result = self.storage_service.upload_file(
                    file_data=file_content,
                    filename=filename,
                    generate_thumbnails=True
                )
                
                if result.errors:
                    print(f"❌ Upload failed: {', '.join(result.errors)}")
                    continue
                
                uploaded_files.append(result.file_id)
                
                print(f"✅ File uploaded successfully")
                print(f"   File ID: {result.file_id}")
                print(f"   Storage path: {result.file_metadata.storage_path}")
                print(f"   File size: {result.file_metadata.file_size} bytes")
                print(f"   Thumbnails: {len(result.thumbnails)} generated")
                print(f"   Checksum: {result.file_metadata.checksum[:16]}...")
                
                if result.secure_url:
                    print(f"   Secure URL: {result.secure_url}")
            
            except Exception as e:
                print(f"❌ Storage operation failed for {file_type}: {e}")
        
        # List files
        print(f"\n📁 Storage Summary:")
        try:
            files = self.storage_service.list_files(limit=10)
            print(f"   Total files in storage: {len(files)}")
            
            for file_meta in files:
                print(f"   - {file_meta.original_filename} ({file_meta.file_size} bytes)")
        
        except Exception as e:
            print(f"❌ Failed to list files: {e}")
        
        # Cleanup (delete uploaded files)
        print(f"\n🗑️  Cleaning up uploaded files...")
        for file_id in uploaded_files:
            try:
                success = self.storage_service.delete_file(file_id)
                if success:
                    print(f"   ✅ Deleted {file_id}")
                else:
                    print(f"   ❌ Failed to delete {file_id}")
            except Exception as e:
                print(f"   ❌ Error deleting {file_id}: {e}")
    
    def _demo_batch_processing(self, sample_files: Dict[str, str]):
        """Demonstrate batch processing capabilities"""
        
        print("Testing batch processing with Celery...")
        
        try:
            # Prepare file list for batch processing
            file_list = []
            for file_type, file_path in sample_files.items():
                if file_path:
                    file_list.append({
                        'file_path': file_path,
                        'filename': f"batch_{file_type}_document.{file_type}"
                    })
            
            if not file_list:
                print("⚠️  No files available for batch processing")
                return
            
            print(f"Submitting batch job with {len(file_list)} files...")
            
            # Submit batch job
            config = BatchJobConfig(
                max_retries=2,
                timeout=300,  # 5 minutes
                priority=5
            )
            
            job_id = self.batch_processor.submit_job(
                'batch_process_documents',
                config,
                file_list=file_list
            )
            
            print(f"✅ Batch job submitted: {job_id}")
            
            # Monitor job progress
            print("Monitoring job progress...")
            
            max_wait_time = 60  # Maximum wait time in seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                job_status = self.batch_processor.get_job_status(job_id)
                
                if not job_status:
                    print("❌ Job not found")
                    break
                
                print(f"   Status: {job_status.status.value}")
                print(f"   Progress: {job_status.progress.percentage:.1f}% - {job_status.progress.message}")
                
                if job_status.status.value in ['success', 'failure']:
                    break
                
                time.sleep(2)  # Wait 2 seconds before checking again
            
            # Get final results
            final_status = self.batch_processor.get_job_status(job_id)
            if final_status:
                print(f"\n📊 Final Job Results:")
                print(f"   Status: {final_status.status.value}")
                print(f"   Execution time: {final_status.execution_time:.2f}s")
                
                if final_status.result:
                    result = final_status.result
                    print(f"   Files processed: {result.get('total_files', 0)}")
                    print(f"   Successful: {result.get('successful', 0)}")
                    print(f"   Failed: {result.get('failed', 0)}")
                    print(f"   Success rate: {result.get('success_rate', 0):.1f}%")
                
                if final_status.error:
                    print(f"   Error: {final_status.error}")
        
        except Exception as e:
            print(f"❌ Batch processing demo failed: {e}")
    
    def cleanup_demo_files(self, sample_files: Dict[str, str]):
        """Clean up temporary files created during demo"""
        
        print("\n🧹 Cleaning up demo files...")
        
        for file_path in sample_files.values():
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    print(f"   ✅ Deleted {file_path}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {file_path}: {e}")


def run_pdf_processing_demo():
    """Run the complete PDF processing demonstration"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        demo = PDFProcessingDemo()
        demo.run_comprehensive_demo()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


def run_performance_benchmark():
    """Run performance benchmarks on PDF processing"""
    
    print("🏃‍♂️ PDF Processing Performance Benchmark")
    print("=" * 50)
    
    # Create test files of different sizes
    test_cases = [
        ("Small text", "A" * 1000),
        ("Medium text", "B" * 10000), 
        ("Large text", "C" * 100000),
    ]
    
    services = {
        "PDF Processor": PDFProcessor(),
        "Document Processor": DocumentProcessor(),
        "Text Preprocessor": TextPreprocessor()
    }
    
    results = {}
    
    for name, text_content in test_cases:
        print(f"\n📏 Testing {name} ({len(text_content)} characters):")
        results[name] = {}
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text_content)
            temp_path = f.name
        
        try:
            # Test Document Processor
            start = time.time()
            doc_result = services["Document Processor"].process_document(temp_path)
            doc_time = time.time() - start
            results[name]["Document Processor"] = doc_time
            print(f"   Document Processor: {doc_time:.3f}s")
            
            # Test Text Preprocessor
            start = time.time()
            preproc_result = services["Text Preprocessor"].preprocess(text_content)
            preproc_time = time.time() - start
            results[name]["Text Preprocessor"] = preproc_time
            print(f"   Text Preprocessor: {preproc_time:.3f}s")
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    # Print summary
    print(f"\n📊 Performance Summary:")
    print(f"{'Test Case':<15} {'Doc Processor':<15} {'Preprocessor':<15}")
    print("-" * 45)
    
    for test_name, timings in results.items():
        doc_time = timings.get("Document Processor", 0)
        prep_time = timings.get("Text Preprocessor", 0)
        print(f"{test_name:<15} {doc_time:<15.3f} {prep_time:<15.3f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_performance_benchmark()
    else:
        run_pdf_processing_demo()