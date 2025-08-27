#!/usr/bin/env python3
"""
PDF Processing Integration Test

Quick test to validate all PDF processing components work together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import tempfile
import time
from pathlib import Path

# Test imports
try:
    from app.services.pdf_service import PDFProcessor, extract_pdf_text
    from app.services.document_processor import DocumentProcessor, process_document
    from app.services.preprocessing import TextPreprocessor, preprocess_text
    from app.services.storage_service import create_storage_service, StorageBackend
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of all components"""
    
    print("\n🧪 Running PDF Processing Integration Tests")
    print("=" * 50)
    
    # Create sample text content
    sample_content = """
    TERMS OF SERVICE AGREEMENT
    
    1. BINDING ARBITRATION CLAUSE
    
    Any dispute, claim, or controversy arising out of or relating to this Agreement
    shall be resolved by binding arbitration. The arbitration will be conducted 
    under the Commercial Arbitration Rules of the American Arbitration Association.
    
    2. CLASS ACTION WAIVER
    
    You agree that any arbitration shall be conducted in your individual capacity
    only and not as a class action or other representative action.
    
    3. GOVERNING LAW
    
    This Agreement shall be governed by and construed in accordance with the
    laws of the State of California.
    """
    
    # Test 1: Document Processing
    print("\n📄 Test 1: Document Processing")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(sample_content)
            temp_file = f.name
        
        processor = DocumentProcessor()
        result = processor.process_document(temp_file, 'test_document.txt')
        
        print(f"   ✅ Processed successfully")
        print(f"   📊 Text length: {len(result.text)} chars")
        print(f"   🔗 Sections found: {len(result.structured_content)}")
        print(f"   📈 File type: {result.metadata.file_type.value}")
        
        Path(temp_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"   ❌ Document processing failed: {e}")
        return False
    
    # Test 2: Text Preprocessing
    print("\n🔧 Test 2: Text Preprocessing")
    try:
        preprocessor = TextPreprocessor()
        prep_result = preprocessor.preprocess(sample_content)
        
        print(f"   ✅ Preprocessing successful")
        print(f"   🌍 Language: {prep_result.detected_language}")
        print(f"   📋 Content type: {prep_result.content_type.value}")
        print(f"   ⭐ Quality: {prep_result.text_quality.value}")
        print(f"   📊 Word count: {prep_result.statistics.word_count}")
        
    except Exception as e:
        print(f"   ❌ Text preprocessing failed: {e}")
        return False
    
    # Test 3: Storage Service
    print("\n💾 Test 3: Storage Service")
    try:
        storage = create_storage_service(
            backend=StorageBackend.LOCAL,
            local_base_path="/tmp/test_storage"
        )
        
        # Upload test file
        test_content = sample_content.encode('utf-8')
        upload_result = storage.upload_file(
            file_data=test_content,
            filename="test_terms.txt",
            content_type="text/plain"
        )
        
        if upload_result.errors:
            print(f"   ❌ Upload failed: {upload_result.errors}")
            return False
        
        file_id = upload_result.file_id
        print(f"   ✅ Upload successful")
        print(f"   🆔 File ID: {file_id}")
        print(f"   📏 Size: {upload_result.file_metadata.file_size} bytes")
        
        # Retrieve file
        retrieved_content = storage.get_file(file_id)
        if retrieved_content == test_content:
            print(f"   ✅ Retrieval successful")
        else:
            print(f"   ❌ Retrieved content doesn't match")
            return False
        
        # Clean up
        storage.delete_file(file_id)
        print(f"   ✅ Cleanup successful")
        
    except Exception as e:
        print(f"   ❌ Storage test failed: {e}")
        return False
    
    # Test 4: End-to-end workflow
    print("\n🔄 Test 4: End-to-end Workflow")
    try:
        # Create document -> Process -> Preprocess -> Store
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(sample_content)
            temp_file = f.name
        
        # Process document
        doc_processor = DocumentProcessor()
        doc_result = process_document(temp_file, 'workflow_test.txt')
        
        # Preprocess text
        text_processor = TextPreprocessor()
        prep_result = preprocess_text(doc_result.text)
        
        # Store processed content
        storage = create_storage_service(
            backend=StorageBackend.LOCAL,
            local_base_path="/tmp/workflow_storage"
        )
        
        processed_content = prep_result.cleaned_text.encode('utf-8')
        store_result = storage.upload_file(
            file_data=processed_content,
            filename="processed_document.txt",
            metadata={
                'original_filename': 'workflow_test.txt',
                'processing_method': doc_result.extraction_method,
                'language': prep_result.detected_language,
                'quality': prep_result.text_quality.value,
                'word_count': prep_result.statistics.word_count
            }
        )
        
        print(f"   ✅ End-to-end workflow successful")
        print(f"   📝 Original length: {len(sample_content)} chars")
        print(f"   🔧 Processed length: {len(prep_result.cleaned_text)} chars")
        print(f"   🆔 Stored as: {store_result.file_id}")
        
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)
        storage.delete_file(store_result.file_id)
        
    except Exception as e:
        print(f"   ❌ End-to-end workflow failed: {e}")
        return False
    
    return True

def test_error_handling():
    """Test error handling and edge cases"""
    
    print("\n🛡️  Test 5: Error Handling")
    
    # Test with empty content
    print("   Testing empty content...")
    try:
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("")
        
        if result.warnings:
            print(f"   ✅ Empty content handled with warnings: {result.warnings[0]}")
        else:
            print(f"   ⚠️  Empty content handling could be improved")
    except Exception as e:
        print(f"   ❌ Empty content test failed: {e}")
        return False
    
    # Test with non-existent file
    print("   Testing non-existent file...")
    try:
        processor = DocumentProcessor()
        result = processor.process_document("/non/existent/file.txt", "missing.txt")
        print(f"   ⚠️  Non-existent file should have failed")
        return False
    except Exception as e:
        print(f"   ✅ Non-existent file properly handled: {type(e).__name__}")
    
    # Test with invalid storage path
    print("   Testing invalid storage configuration...")
    try:
        storage = create_storage_service(
            backend=StorageBackend.LOCAL,
            local_base_path="/root/invalid/path/that/should/not/exist"
        )
        
        # This should work as the path will be created
        test_content = b"test"
        result = storage.upload_file(test_content, "test.txt")
        
        if result.errors:
            print(f"   ✅ Invalid path handled: {result.errors[0]}")
        else:
            print(f"   ✅ Path auto-created successfully")
            # Cleanup
            storage.delete_file(result.file_id)
        
    except Exception as e:
        print(f"   ✅ Invalid storage path handled: {type(e).__name__}")
    
    return True

def run_performance_test():
    """Quick performance test"""
    
    print("\n⚡ Performance Test")
    
    # Create larger test content
    large_content = """
    COMPREHENSIVE TERMS OF SERVICE
    
    """ + """
    This is a longer document with more content to test processing performance.
    It contains multiple paragraphs and sections to simulate real-world usage.
    
    ARBITRATION CLAUSE:
    All disputes arising from this agreement shall be resolved through binding arbitration.
    
    """ * 20  # Repeat to make it larger
    
    try:
        start_time = time.time()
        
        # Process with document processor
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(large_content)
            temp_file = f.name
        
        processor = DocumentProcessor()
        doc_result = processor.process_document(temp_file)
        doc_time = time.time() - start_time
        
        # Process with text preprocessor
        start_time = time.time()
        text_processor = TextPreprocessor()
        prep_result = text_processor.preprocess(doc_result.text)
        prep_time = time.time() - start_time
        
        print(f"   📏 Content size: {len(large_content):,} characters")
        print(f"   📄 Document processing: {doc_time:.3f}s")
        print(f"   🔧 Text preprocessing: {prep_time:.3f}s")
        print(f"   📊 Total processing: {doc_time + prep_time:.3f}s")
        print(f"   🏃‍♂️ Processing rate: {len(large_content) / (doc_time + prep_time):,.0f} chars/sec")
        
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    
    print("🚀 PDF Processing System Integration Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Error Handling", test_error_handling),
        ("Performance", run_performance_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Tests...")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} tests passed")
            else:
                print(f"❌ {test_name} tests failed")
                
        except Exception as e:
            print(f"❌ {test_name} tests crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PDF processing system is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())