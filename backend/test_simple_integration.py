#!/usr/bin/env python3
"""
Simple integration test to isolate import issues.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path}")

def test_step_by_step():
    """Test imports step by step to identify the exact issue."""
    
    print("\n=== STEP 1: Testing basic RAG system path ===")
    rag_path = current_dir / "rag_system" / "src"
    print(f"RAG path: {rag_path}")
    print(f"RAG path exists: {rag_path.exists()}")
    
    if rag_path.exists():
        sys.path.insert(0, str(rag_path))
        print(f"Added RAG path to sys.path")
    else:
        print("❌ RAG system path does not exist!")
        return False
    
    print("\n=== STEP 2: Testing individual module imports ===")
    
    # Test core module
    try:
        print("Testing core module...")
        import rag_system.src.core
        print("✅ Core module imported")
    except Exception as e:
        print(f"❌ Core module failed: {e}")
        return False
    
    # Test arbitration detector
    try:
        print("Testing arbitration detector...")
        from rag_system.src.core import arbitration_detector
        print("✅ Arbitration detector module imported")
    except Exception as e:
        print(f"❌ Arbitration detector failed: {e}")
        print(f"Error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test classes
    try:
        print("Testing ArbitrationDetectionPipeline class...")
        from rag_system.src.core.arbitration_detector import ArbitrationDetectionPipeline
        print("✅ ArbitrationDetectionPipeline imported")
    except Exception as e:
        print(f"❌ ArbitrationDetectionPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("Testing ArbitrationClause class...")
        from rag_system.src.core.arbitration_detector import ArbitrationClause
        print("✅ ArbitrationClause imported")
    except Exception as e:
        print(f"❌ ArbitrationClause failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== STEP 3: Testing comparison engine ===")
    try:
        print("Testing comparison engine...")
        from rag_system.src.comparison.comparison_engine import ClauseComparisonEngine
        print("✅ ClauseComparisonEngine imported")
    except Exception as e:
        print(f"❌ ClauseComparisonEngine failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== STEP 4: Testing database schema ===")
    try:
        print("Testing database schema...")
        from rag_system.src.database.schema import DatabaseManager, VectorStore
        print("✅ Database schema imported")
    except Exception as e:
        print(f"❌ Database schema failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== STEP 5: Testing integration_rag ===")
    try:
        print("Testing integration_rag...")
        import integration_rag
        print("✅ integration_rag imported")
        
        print("Testing RAGIntegration class...")
        rag = integration_rag.RAGIntegration()
        print("✅ RAGIntegration instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ integration_rag failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_requirements():
    """Check if required packages are installed."""
    print("\n=== REQUIREMENTS CHECK ===")
    
    required_packages = [
        'sqlalchemy',
        'numpy',
        'typing',
        'dataclasses',
        'pathlib',
        'logging',
        'json',
        'hashlib',
        'os'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        return False
    else:
        print("\n✅ All basic packages available")
        return True

def test_optional_packages():
    """Test optional packages."""
    print("\n=== OPTIONAL PACKAGES CHECK ===")
    
    optional_packages = {
        'redis': 'For caching',
        'torch': 'For BERT models',
        'transformers': 'For BERT models',
        'faiss': 'For vector similarity search',
        'sentence_transformers': 'For embeddings'
    }
    
    for package, purpose in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {purpose}")
        except ImportError:
            print(f"⚠️  {package} - {purpose} (optional)")

def main():
    """Run simple integration test."""
    print("🚀 Simple RAG Integration Test")
    print("=" * 50)
    
    # Check requirements first
    if not test_requirements():
        print("❌ Missing required packages")
        return False
    
    # Check optional packages
    test_optional_packages()
    
    # Test step by step imports
    if test_step_by_step():
        print("\n🎉 SUCCESS: Basic integration test passed!")
        return True
    else:
        print("\n❌ FAILURE: Integration test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)