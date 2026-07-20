#!/usr/bin/env python3
"""
Startup script for the Arbitration Clause Detection RAG System

This script provides an easy way to start the system in different modes.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting Arbitration Detection API...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API: {e}")

def start_cli():
    """Start the CLI interface"""
    print("🖥️  Starting Arbitration Detection CLI...")
    try:
        subprocess.run([
            sys.executable, "src/cli.py", "--help"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start CLI: {e}")

def run_demo():
    """Run the demo script"""
    print("🎭 Running Arbitration Detection Demo...")
    try:
        subprocess.run([
            sys.executable, "demo.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run demo: {e}")

def run_tests():
    """Run the test suite"""
    print("🧪 Running Arbitration Detection Tests...")
    try:
        subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'spacy', 'fastapi', 'uvicorn',
        'click', 'rich', 'pydantic', 'sqlalchemy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Arbitration Clause Detection RAG System"
    )
    
    parser.add_argument(
        'mode',
        choices=['api', 'cli', 'demo', 'test', 'check'],
        help='Mode to run the system in'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='Port for API server (default: 8000)'
    )
    
    args = parser.parse_args()
    
    print("🔍 Arbitration Clause Detection RAG System")
    print("=" * 50)
    
    if args.mode == 'api':
        start_api()
    elif args.mode == 'cli':
        start_cli()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'test':
        run_tests()
    elif args.mode == 'check':
        check_dependencies()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
