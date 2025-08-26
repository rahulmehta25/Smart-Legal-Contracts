#!/usr/bin/env python3
"""
Deployment validation script for FastAPI backend
Tests local deployment configuration before pushing to Railway
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient

# Add app to Python path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_configuration():
    """Test deployment configuration files"""
    print("🔧 Testing deployment configuration...")
    
    required_files = [
        "railway.json",
        "Dockerfile", 
        ".dockerignore",
        "requirements-production.txt",
        ".env.production",
        "start.sh",
        "deploy.sh",
        "DEPLOYMENT_GUIDE.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All deployment files present")
    return True

def test_environment_config():
    """Test environment configuration"""
    print("🔧 Testing environment configuration...")
    
    # Set test environment variables
    os.environ.update({
        "ENVIRONMENT": "production",
        "DATABASE_URL": "sqlite:///./test.db",
        "SECRET_KEY": "test-secret-key-for-deployment-validation-64-chars-minimum"
    })
    
    try:
        from app.core.config import settings
        
        # Test configuration values
        assert settings.port == int(os.getenv("PORT", "8000"))
        assert settings.database_url == os.getenv("DATABASE_URL")
        assert settings.secret_key == os.getenv("SECRET_KEY")
        
        # Test CORS configuration
        if os.getenv("ENVIRONMENT") == "production":
            assert "smart-legal-contracts.vercel.app" in settings.allowed_origins
        
        print("✅ Environment configuration valid")
        return True
        
    except Exception as e:
        print(f"❌ Environment configuration error: {e}")
        return False

def test_application_startup():
    """Test application can start successfully"""
    print("🔧 Testing application startup...")
    
    try:
        from app.main import app
        
        # Create test client
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["service"] == "Arbitration RAG API"
        
        print("✅ Application starts successfully")
        print(f"   Health check: {health_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ Application startup error: {e}")
        return False

def test_api_endpoints():
    """Test core API endpoints"""
    print("🔧 Testing API endpoints...")
    
    try:
        from app.main import app
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        root_data = response.json()
        assert "Arbitration RAG API" in root_data["message"]
        
        # Test API overview
        response = client.get("/api/v1")
        assert response.status_code == 200
        api_data = response.json()
        assert "endpoints" in api_data
        
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        print("✅ Core API endpoints working")
        return True
        
    except Exception as e:
        print(f"❌ API endpoints error: {e}")
        return False

def test_requirements():
    """Test that all required packages can be imported"""
    print("🔧 Testing package imports...")
    
    try:
        # Core dependencies
        import fastapi
        import uvicorn
        import pydantic
        
        # ML dependencies
        import torch
        import numpy as np
        import pandas as pd
        
        # Vector store
        try:
            import chromadb
            print("   ✅ ChromaDB available")
        except ImportError:
            print("   ⚠️ ChromaDB not available (may install during deployment)")
        
        print("✅ Core packages can be imported")
        return True
        
    except ImportError as e:
        print(f"❌ Package import error: {e}")
        return False

def test_production_readiness():
    """Test production readiness checks"""
    print("🔧 Testing production readiness...")
    
    checks = []
    
    # Check secret key
    secret_key = os.getenv("SECRET_KEY", "")
    if secret_key and secret_key != "your-secret-key-change-in-production" and len(secret_key) >= 32:
        checks.append("✅ Secret key configured")
    else:
        checks.append("❌ Secret key not properly configured")
    
    # Check database URL format
    db_url = os.getenv("DATABASE_URL", "")
    if db_url and (db_url.startswith("postgresql://") or db_url.startswith("sqlite:///")):
        checks.append("✅ Database URL format valid")
    else:
        checks.append("⚠️ Database URL should use PostgreSQL in production")
    
    # Check environment
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        checks.append("✅ Production environment set")
    else:
        checks.append("⚠️ Environment should be 'production' for deployment")
    
    for check in checks:
        print(f"   {check}")
    
    return all("✅" in check for check in checks)

async def test_deployment_url(url: str):
    """Test deployed application URL"""
    print(f"🔧 Testing deployment URL: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test health endpoint
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Health check: {health_data['status']}")
                
                # Test API docs
                docs_response = await client.get(f"{url}/docs")
                if docs_response.status_code == 200:
                    print("   ✅ API documentation accessible")
                
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

def main():
    """Run all deployment tests"""
    print("🚀 FastAPI Backend Deployment Validation")
    print("=" * 50)
    
    tests = [
        ("Configuration Files", test_configuration),
        ("Environment Config", test_environment_config), 
        ("Package Imports", test_requirements),
        ("Application Startup", test_application_startup),
        ("API Endpoints", test_api_endpoints),
        ("Production Readiness", test_production_readiness),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for Railway deployment.")
        print("\nNext steps:")
        print("1. Go to https://railway.app and create a new project")
        print("2. Connect your GitHub repository")
        print("3. Add environment variables (see .env.production template)")
        print("4. Deploy will start automatically")
    else:
        print("❌ Some tests failed. Please fix issues before deploying.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)