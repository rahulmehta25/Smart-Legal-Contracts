#!/usr/bin/env python3
"""
Simple API test script using only standard library
"""
import json
import urllib.request
import urllib.error
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(url, description):
    """Test a single endpoint."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            print(f"✅ {description}: OK")
            return True, data
    except urllib.error.HTTPError as e:
        print(f"❌ {description}: HTTP {e.code}")
        return False, None
    except urllib.error.URLError as e:
        print(f"❌ {description}: Connection error - {e.reason}")
        return False, None
    except Exception as e:
        print(f"❌ {description}: Error - {str(e)}")
        return False, None

def main():
    """Run basic connectivity tests."""
    print("🚀 Basic API Connectivity Test")
    print("="*40)
    
    # Test server availability
    success, data = test_endpoint(f"{BASE_URL}/health", "Health Check")
    if not success:
        print("\n❌ Server is not running or not accessible")
        print("To start the server:")
        print("1. Install dependencies: pip3 install -r requirements.txt")
        print("2. Start server: uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Test API overview
    test_endpoint(f"{BASE_URL}/api/v1", "API Overview")
    
    # Test root endpoint
    test_endpoint(f"{BASE_URL}/", "Root Endpoint")
    
    print("\n✅ Basic connectivity tests completed")
    print("Server appears to be running correctly")

if __name__ == "__main__":
    main()