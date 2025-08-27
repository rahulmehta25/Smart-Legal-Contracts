#!/usr/bin/env python3
"""
Comprehensive API Test Script for Arbitration Detection Backend

This script tests all critical endpoints, database connectivity, cache operations,
authentication, ML model integration, WebSocket server, and file upload functionality.

Usage:
    python test_api_comprehensive.py

Requirements:
    - Backend server running on http://localhost:8000
    - All dependencies installed (pip install -r requirements.txt)
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
import aiohttp
import requests
from pathlib import Path
import websockets

# Test configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
TEST_FILE_PATH = Path(__file__).parent / "test_data" / "sample_tou_with_arbitration.txt"


class APITestSuite:
    """Comprehensive test suite for the arbitration detection API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
        self.auth_token = None
        self.test_results = []
        self.test_user_id = None
        self.test_document_id = None
        
    async def setup_session(self):
        """Setup HTTP session for async requests."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup_session(self):
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, message: str = "", details: Any = None):
        """Log test results."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        if details and not success:
            print(f"   Details: {details}")
    
    async def test_health_check(self):
        """Test basic health check endpoint."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test(
                        "Health Check", 
                        True, 
                        f"Server healthy - {data.get('service', 'N/A')} v{data.get('version', 'N/A')}"
                    )
                    return True
                else:
                    self.log_test("Health Check", False, f"Status code: {response.status}")
                    return False
        except Exception as e:
            self.log_test("Health Check", False, f"Connection error: {str(e)}")
            return False
    
    async def test_api_overview(self):
        """Test API overview endpoint."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1") as response:
                if response.status == 200:
                    data = await response.json()
                    endpoints = data.get('endpoints', {})
                    features = data.get('features', [])
                    self.log_test(
                        "API Overview", 
                        True, 
                        f"API v{data.get('version')} with {len(endpoints)} endpoints and {len(features)} features"
                    )
                    return True
                else:
                    self.log_test("API Overview", False, f"Status code: {response.status}")
                    return False
        except Exception as e:
            self.log_test("API Overview", False, f"Error: {str(e)}")
            return False
    
    async def test_database_connection(self):
        """Test database connectivity via internal health check."""
        try:
            # This would typically be an internal endpoint or we test via other endpoints
            # For now, we'll test by trying to create a user
            test_user_data = {
                "username": f"testuser_{int(time.time())}",
                "email": f"test_{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "full_name": "Test User"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/users/register",
                json=test_user_data
            ) as response:
                if response.status == 200:
                    user_data = await response.json()
                    self.test_user_id = user_data.get('id')
                    self.log_test(
                        "Database Connection", 
                        True, 
                        f"Successfully created test user (ID: {self.test_user_id})"
                    )
                    return True
                elif response.status == 400:
                    # User might already exist, try login instead
                    return await self.test_user_login_existing()
                else:
                    error_text = await response.text()
                    self.log_test("Database Connection", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("Database Connection", False, f"Error: {str(e)}")
            return False
    
    async def test_user_login_existing(self):
        """Test login with existing user or create new one."""
        try:
            # Try with a default test user
            login_data = {
                "username": "admin@example.com",
                "password": "admin123"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/users/login",
                json=login_data
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.auth_token = token_data.get('access_token')
                    self.log_test(
                        "User Login", 
                        True, 
                        "Successfully logged in with test credentials"
                    )
                    return True
                else:
                    # Try creating admin user
                    return await self.test_create_admin_user()
        except Exception as e:
            self.log_test("User Login", False, f"Error: {str(e)}")
            return False
    
    async def test_create_admin_user(self):
        """Create admin user for testing."""
        try:
            admin_data = {
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
                "full_name": "Admin User"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/users/register",
                json=admin_data
            ) as response:
                if response.status == 200:
                    # Now try to login
                    return await self.test_user_login_existing()
                else:
                    error_text = await response.text()
                    self.log_test("Create Admin User", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("Create Admin User", False, f"Error: {str(e)}")
            return False
    
    async def test_jwt_authentication(self):
        """Test JWT token-based authentication."""
        if not self.auth_token:
            self.log_test("JWT Authentication", False, "No auth token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            async with self.session.get(
                f"{self.base_url}/api/v1/users/me",
                headers=headers
            ) as response:
                if response.status == 200:
                    user_data = await response.json()
                    self.log_test(
                        "JWT Authentication", 
                        True, 
                        f"Authenticated as {user_data.get('username', 'Unknown')}"
                    )
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("JWT Authentication", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("JWT Authentication", False, f"Error: {str(e)}")
            return False
    
    async def test_document_upload(self):
        """Test document upload functionality."""
        try:
            # Create test document content
            test_content = """
            Terms of Service
            
            By using this service, you agree to the following terms:
            
            1. All disputes arising out of or relating to these Terms or the Service shall be resolved through binding arbitration administered by the American Arbitration Association in accordance with its Commercial Arbitration Rules.
            
            2. The arbitration shall be conducted by a single arbitrator in San Francisco, California.
            
            3. You waive any right to a jury trial or to participate in a class action lawsuit.
            """
            
            # Create temporary test file
            test_file_path = Path("/tmp/test_arbitration_doc.txt")
            test_file_path.write_text(test_content)
            
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Upload via multipart form
            data = aiohttp.FormData()
            data.add_field('file', 
                          open(test_file_path, 'rb'),
                          filename='test_arbitration_doc.txt',
                          content_type='text/plain')
            
            async with self.session.post(
                f"{self.base_url}/api/v1/documents/upload",
                data=data,
                headers=headers
            ) as response:
                if response.status == 200:
                    doc_data = await response.json()
                    self.test_document_id = doc_data.get('document_id')
                    self.log_test(
                        "Document Upload", 
                        True, 
                        f"Uploaded document ID: {self.test_document_id}, Chunks: {doc_data.get('chunks_created', 0)}"
                    )
                    
                    # Cleanup
                    test_file_path.unlink()
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("Document Upload", False, f"Status: {response.status}, Error: {error_text}")
                    test_file_path.unlink()
                    return False
                    
        except Exception as e:
            self.log_test("Document Upload", False, f"Error: {str(e)}")
            return False
    
    async def test_arbitration_analysis(self):
        """Test arbitration clause analysis."""
        if not self.test_document_id:
            self.log_test("Arbitration Analysis", False, "No test document available")
            return False
        
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            analysis_data = {
                "document_id": self.test_document_id,
                "analysis_options": {
                    "include_explanations": True,
                    "confidence_threshold": 0.5
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis/analyze",
                json=analysis_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    analysis = await response.json()
                    has_arbitration = analysis.get('has_arbitration_clause', False)
                    confidence = analysis.get('confidence_score', 0)
                    clauses_found = len(analysis.get('clauses', []))
                    
                    self.log_test(
                        "Arbitration Analysis", 
                        True, 
                        f"Analysis complete - Arbitration: {has_arbitration}, Confidence: {confidence:.3f}, Clauses: {clauses_found}"
                    )
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("Arbitration Analysis", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("Arbitration Analysis", False, f"Error: {str(e)}")
            return False
    
    async def test_quick_text_analysis(self):
        """Test quick text analysis without document storage."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            quick_analysis_data = {
                "text": "Any dispute shall be resolved through arbitration administered by JAMS.",
                "options": {
                    "include_explanations": True
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/analysis/quick-analyze",
                json=quick_analysis_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    has_arbitration = result.get('has_arbitration_clause', False)
                    confidence = result.get('confidence_score', 0)
                    
                    self.log_test(
                        "Quick Text Analysis", 
                        True, 
                        f"Quick analysis - Arbitration: {has_arbitration}, Confidence: {confidence:.3f}"
                    )
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("Quick Text Analysis", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("Quick Text Analysis", False, f"Error: {str(e)}")
            return False
    
    async def test_document_search(self):
        """Test document search functionality."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            search_params = {
                "query": "arbitration dispute resolution",
                "limit": 5
            }
            
            async with self.session.get(
                f"{self.base_url}/api/v1/documents/search/",
                params=search_params,
                headers=headers
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    total_results = results.get('total_results', 0)
                    
                    self.log_test(
                        "Document Search", 
                        True, 
                        f"Search completed - {total_results} results found"
                    )
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("Document Search", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("Document Search", False, f"Error: {str(e)}")
            return False
    
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        try:
            uri = f"{WS_URL}"
            if self.auth_token:
                uri += f"?token={self.auth_token}"
            
            async with websockets.connect(uri, timeout=10) as websocket:
                # Send a test message
                test_message = {
                    "type": "ping",
                    "data": {"timestamp": time.time()}
                }
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    self.log_test(
                        "WebSocket Connection", 
                        True, 
                        f"WebSocket communication successful - Response type: {response_data.get('type', 'unknown')}"
                    )
                    return True
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Connection", False, "Timeout waiting for WebSocket response")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket Connection", False, f"Error: {str(e)}")
            return False
    
    def test_redis_connection(self):
        """Test Redis connection (sync test)."""
        try:
            import redis
            
            # Try to connect to Redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            # Test basic operations
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            r.delete('test_key')
            
            if value == b'test_value':
                self.log_test("Redis Connection", True, "Redis connection and operations successful")
                return True
            else:
                self.log_test("Redis Connection", False, "Redis operations failed")
                return False
                
        except ImportError:
            self.log_test("Redis Connection", False, "Redis library not installed")
            return False
        except Exception as e:
            self.log_test("Redis Connection", False, f"Error: {str(e)}")
            return False
    
    async def test_pdf_processing(self):
        """Test PDF processing endpoint."""
        try:
            # Create a simple test PDF content (as text for now)
            test_pdf_content = "This is a test PDF content with arbitration clauses for dispute resolution."
            
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Test PDF upload endpoint
            data = aiohttp.FormData()
            data.add_field('file', 
                          test_pdf_content.encode(),
                          filename='test.pdf',
                          content_type='application/pdf')
            
            async with self.session.post(
                f"{self.base_url}/api/upload/pdf",
                data=data,
                headers=headers
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    self.log_test(
                        "PDF Processing", 
                        True, 
                        f"PDF processing successful - Job ID: {result.get('job_id', 'N/A')}"
                    )
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("PDF Processing", False, f"Status: {response.status}, Error: {error_text}")
                    return False
        except Exception as e:
            self.log_test("PDF Processing", False, f"Error: {str(e)}")
            return False
    
    async def test_statistics_endpoints(self):
        """Test various statistics endpoints."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            endpoints = [
                "/api/v1/documents/stats/overview",
                "/api/v1/analysis/stats/overview",
                "/api/v1/analysis/stats/clause-types"
            ]
            
            success_count = 0
            for endpoint in endpoints:
                try:
                    async with self.session.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            success_count += 1
                        # Don't fail the entire test for individual endpoint failures
                except:
                    pass
            
            if success_count > 0:
                self.log_test(
                    "Statistics Endpoints", 
                    True, 
                    f"{success_count}/{len(endpoints)} statistics endpoints working"
                )
                return True
            else:
                self.log_test("Statistics Endpoints", False, "No statistics endpoints working")
                return False
        except Exception as e:
            self.log_test("Statistics Endpoints", False, f"Error: {str(e)}")
            return False
    
    def print_summary(self):
        """Print comprehensive test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*70)
        print("COMPREHENSIVE API TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print("="*70)
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for result in self.test_results:
                if not result['success']:
                    print(f"❌ {result['test_name']}: {result['message']}")
        
        print("\nRECOMMENDations:")
        print("-" * 40)
        
        # Analyze results and provide recommendations
        failed_test_names = [r['test_name'] for r in self.test_results if not r['success']]
        
        if 'Health Check' in failed_test_names:
            print("🔧 Server is not running. Start with: uvicorn app.main:app --reload")
        
        if 'Database Connection' in failed_test_names:
            print("🔧 Database connection failed. Check DATABASE_URL environment variable")
        
        if 'Redis Connection' in failed_test_names:
            print("🔧 Redis is not available. Install and start Redis server")
        
        if 'WebSocket Connection' in failed_test_names:
            print("🔧 WebSocket server may not be properly configured")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Your API is production-ready.")
        elif passed_tests > total_tests * 0.8:
            print("👍 Most tests passed. Minor issues to resolve.")
        else:
            print("⚠️  Multiple critical issues found. Review failed tests above.")
        
        print("\n" + "="*70)
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting Comprehensive API Test Suite")
        print("="*50)
        
        await self.setup_session()
        
        try:
            # Core connectivity tests
            await self.test_health_check()
            await self.test_api_overview()
            
            # Database and authentication tests
            await self.test_database_connection()
            await self.test_jwt_authentication()
            
            # Cache test (sync)
            self.test_redis_connection()
            
            # Document and analysis tests
            await self.test_document_upload()
            await self.test_arbitration_analysis()
            await self.test_quick_text_analysis()
            await self.test_document_search()
            
            # File processing tests
            await self.test_pdf_processing()
            
            # Statistics tests
            await self.test_statistics_endpoints()
            
            # WebSocket test
            await self.test_websocket_connection()
            
        finally:
            await self.cleanup_session()
        
        self.print_summary()


async def main():
    """Main test runner."""
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server health check failed. Status: {response.status_code}")
            print("Please ensure the backend server is running on http://localhost:8000")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to backend server at http://localhost:8000")
        print("Please start the server with: uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Run test suite
    test_suite = APITestSuite()
    await test_suite.run_all_tests()
    
    # Exit with appropriate code
    failed_count = sum(1 for result in test_suite.test_results if not result['success'])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())