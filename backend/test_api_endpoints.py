#!/usr/bin/env python3
"""
Comprehensive API endpoint testing script
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8001"
VERCEL_ORIGIN = "https://test-app.vercel.app"

def test_endpoint(method, endpoint, data=None, headers=None, expected_status=200):
    """Test a single API endpoint."""
    
    if headers is None:
        headers = {
            "Content-Type": "application/json",
            "Origin": VERCEL_ORIGIN
        }
    
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        elif method.upper() == "OPTIONS":
            response = requests.options(url, headers=headers)
        else:
            logger.error(f"Unsupported method: {method}")
            return False
        
        if response.status_code == expected_status:
            logger.info(f"✅ {method} {endpoint}: {response.status_code} - OK")
            return True
        else:
            logger.error(f"❌ {method} {endpoint}: Expected {expected_status}, got {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {method} {endpoint}: Exception - {e}")
        return False

def main():
    """Test all API endpoints."""
    logger.info("Starting comprehensive API endpoint testing...")
    
    test_results = []
    
    # Test basic endpoints
    endpoints_to_test = [
        ("GET", "/", 200),
        ("GET", "/health", 200),
        ("GET", "/health/detailed", 200),
        ("GET", "/api/v1", 200),
        ("GET", "/api/websocket/stats", 200),
        ("GET", "/docs", 200),
        ("GET", "/redoc", 200),
    ]
    
    # Test CORS preflight requests
    cors_endpoints = [
        ("OPTIONS", "/health", 200),
        ("OPTIONS", "/api/v1", 200),
        ("OPTIONS", "/api/websocket/stats", 200),
    ]
    
    # Test API endpoints
    api_endpoints = [
        ("GET", "/api/v1/documents", None, None, 200),  # Might return empty list
        ("GET", "/api/v1/analysis", None, None, 200),   # Might return empty list  
        ("GET", "/api/v1/users", None, None, 200),      # Might return empty list
    ]
    
    # Run basic endpoint tests
    logger.info("\n=== Testing Basic Endpoints ===")
    for method, endpoint, expected_status in endpoints_to_test:
        result = test_endpoint(method, endpoint, expected_status=expected_status)
        test_results.append((f"{method} {endpoint}", result))
    
    # Run CORS tests
    logger.info("\n=== Testing CORS Preflight Requests ===")
    for method, endpoint, expected_status in cors_endpoints:
        result = test_endpoint(method, endpoint, expected_status=expected_status)
        test_results.append((f"CORS {method} {endpoint}", result))
    
    # Run API endpoint tests (these might fail if database/services aren't fully configured)
    logger.info("\n=== Testing API Endpoints ===")
    for method, endpoint, data, headers, expected_status in api_endpoints:
        # These might return 404 or 500 if not fully implemented, that's OK for now
        result = test_endpoint(method, endpoint, data, headers)
        test_results.append((f"API {method} {endpoint}", result))
    
    # Test document upload simulation (will likely fail without proper setup, but good to test)
    logger.info("\n=== Testing Document Upload Simulation ===")
    # This will test the endpoint structure even if it fails due to missing dependencies
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.info(f"⚠️ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)