"""
Comprehensive Integration Tests for Frontend-Backend Connectivity

This test suite verifies all API endpoints and connections between frontend and backend,
ensuring proper communication, data flow, and error handling.
"""

import asyncio
import json
import time
import aiohttp
import websockets
import io
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
TEST_TIMEOUT = 30
MAX_RETRIES = 3

class IntegrationTestSuite:
    """Comprehensive integration test suite for frontend-backend connectivity."""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        self.auth_token = None
        self.test_user_id = None
        self.test_document_id = None
        self.test_analysis_id = None
        
        # Test data
        self.test_user_data = {
            "username": f"testuser_{int(time.time())}",
            "email": f"testuser_{int(time.time())}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User"
        }
        
        self.test_document_content = """
        Terms of Service Agreement
        
        1. Acceptance of Terms
        By using this service, you agree to be bound by these terms.
        
        2. Dispute Resolution
        Any disputes arising from this agreement shall be resolved through binding arbitration
        in accordance with the rules of the American Arbitration Association. You waive your
        right to participate in a class action lawsuit or class-wide arbitration.
        
        3. Governing Law
        This agreement shall be governed by the laws of the State of California.
        """
    
    async def setup(self):
        """Setup test session and authentication."""
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=TEST_TIMEOUT)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        logger.info("Test session initialized")
    
    async def cleanup(self):
        """Cleanup test session and resources."""
        if self.session:
            await self.session.close()
        logger.info("Test session cleaned up")
    
    def log_result(self, test_name: str, status: str, details: Dict[str, Any] = None, error: str = None):
        """Log test result."""
        result = {
            "test_name": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "error": error
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        logger.info(f"{status_emoji} {test_name}: {status}" + (f" - {error}" if error else ""))
        
        if details:
            logger.info(f"   Details: {details}")
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[int, Dict[str, Any], str]:
        """Make HTTP request with error handling and retries."""
        url = f"{BASE_URL}{endpoint}"
        headers = kwargs.pop('headers', {})
        
        # Add auth header if token is available
        if self.auth_token:
            headers['Authorization'] = f"Bearer {self.auth_token}"
        
        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.request(method, url, headers=headers, **kwargs) as response:
                    status = response.status
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'application/json' in content_type:
                        data = await response.json()
                        error_msg = None
                    else:
                        text = await response.text()
                        data = {"text": text}
                        error_msg = None
                        
                    return status, data, error_msg
                    
            except asyncio.TimeoutError:
                error_msg = f"Request timeout (attempt {attempt + 1})"
                if attempt == MAX_RETRIES - 1:
                    return 0, {}, error_msg
                await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = f"Request error: {str(e)} (attempt {attempt + 1})"
                if attempt == MAX_RETRIES - 1:
                    return 0, {}, error_msg
                await asyncio.sleep(1)
    
    async def test_health_check(self):
        """Test health check endpoint."""
        logger.info("Testing health check endpoint...")
        
        status, data, error = await self.make_request('GET', '/health')
        
        if error:
            self.log_result("Health Check", "FAIL", error=error)
            return False
        
        if status == 200:
            expected_keys = ["status", "service", "version"]
            has_all_keys = all(key in data for key in expected_keys)
            
            if has_all_keys and data.get("status") == "healthy":
                self.log_result("Health Check", "PASS", {
                    "status_code": status,
                    "service": data.get("service"),
                    "version": data.get("version")
                })
                return True
            else:
                self.log_result("Health Check", "FAIL", {
                    "status_code": status,
                    "missing_keys": [k for k in expected_keys if k not in data],
                    "response": data
                }, "Invalid response structure or status")
                return False
        else:
            self.log_result("Health Check", "FAIL", {
                "status_code": status,
                "response": data
            }, f"Unexpected status code: {status}")
            return False
    
    async def test_api_overview(self):
        """Test API overview endpoints."""
        logger.info("Testing API overview endpoints...")
        
        # Test root endpoint
        status, data, error = await self.make_request('GET', '/')
        if status == 200:
            self.log_result("Root Endpoint", "PASS", {
                "status_code": status,
                "message": data.get("message"),
                "version": data.get("version")
            })
        else:
            self.log_result("Root Endpoint", "FAIL", {"status_code": status, "response": data}, error)
        
        # Test API v1 overview
        status, data, error = await self.make_request('GET', '/api/v1')
        if status == 200 and "endpoints" in data:
            self.log_result("API v1 Overview", "PASS", {
                "status_code": status,
                "endpoints_count": len(data.get("endpoints", {})),
                "features_count": len(data.get("features", []))
            })
            return True
        else:
            self.log_result("API v1 Overview", "FAIL", {"status_code": status, "response": data}, error)
            return False
    
    async def test_user_registration(self):
        """Test user registration endpoint."""
        logger.info("Testing user registration...")
        
        status, data, error = await self.make_request(
            'POST', 
            '/api/v1/users/register',
            json=self.test_user_data
        )
        
        if status == 200 and "id" in data:
            self.test_user_id = data["id"]
            self.log_result("User Registration", "PASS", {
                "status_code": status,
                "user_id": self.test_user_id,
                "username": data.get("username")
            })
            return True
        elif status == 400 and "already exists" in str(data):
            # User might already exist, try with different credentials
            self.test_user_data["username"] += "_alt"
            self.test_user_data["email"] = self.test_user_data["email"].replace("@", "_alt@")
            
            status, data, error = await self.make_request(
                'POST', 
                '/api/v1/users/register',
                json=self.test_user_data
            )
            
            if status == 200 and "id" in data:
                self.test_user_id = data["id"]
                self.log_result("User Registration", "PASS", {
                    "status_code": status,
                    "user_id": self.test_user_id,
                    "username": data.get("username"),
                    "note": "Used alternative username"
                })
                return True
        
        self.log_result("User Registration", "FAIL", {
            "status_code": status,
            "response": data
        }, error)
        return False
    
    async def test_user_login(self):
        """Test user login endpoint."""
        logger.info("Testing user login...")
        
        login_data = {
            "username": self.test_user_data["username"],
            "password": self.test_user_data["password"]
        }
        
        status, data, error = await self.make_request(
            'POST',
            '/api/v1/users/login',
            json=login_data
        )
        
        if status == 200 and "access_token" in data:
            self.auth_token = data["access_token"]
            self.log_result("User Login", "PASS", {
                "status_code": status,
                "token_type": data.get("token_type"),
                "has_token": bool(self.auth_token)
            })
            return True
        else:
            self.log_result("User Login", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_token_verification(self):
        """Test token verification endpoint."""
        logger.info("Testing token verification...")
        
        if not self.auth_token:
            self.log_result("Token Verification", "SKIP", error="No auth token available")
            return False
        
        status, data, error = await self.make_request(
            'POST',
            '/api/v1/users/verify-token',
            json={"token": self.auth_token}
        )
        
        if status == 200 and data.get("valid"):
            self.log_result("Token Verification", "PASS", {
                "status_code": status,
                "valid": data.get("valid"),
                "user_id": data.get("user_id")
            })
            return True
        else:
            self.log_result("Token Verification", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_user_profile(self):
        """Test user profile endpoint."""
        logger.info("Testing user profile endpoint...")
        
        if not self.auth_token:
            self.log_result("User Profile", "SKIP", error="No auth token available")
            return False
        
        status, data, error = await self.make_request('GET', '/api/v1/users/me')
        
        if status == 200 and "id" in data:
            self.log_result("User Profile", "PASS", {
                "status_code": status,
                "user_id": data.get("id"),
                "username": data.get("username")
            })
            return True
        else:
            self.log_result("User Profile", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_document_upload(self):
        """Test document upload endpoint."""
        logger.info("Testing document upload...")
        
        if not self.auth_token:
            self.log_result("Document Upload", "SKIP", error="No auth token available")
            return False
        
        # Create form data for file upload
        data = aiohttp.FormData()
        data.add_field('file', 
                      io.StringIO(self.test_document_content), 
                      filename='test_document.txt',
                      content_type='text/plain')
        
        headers = {'Authorization': f'Bearer {self.auth_token}'}
        
        try:
            async with self.session.post(f"{BASE_URL}/api/v1/documents/upload", 
                                       data=data, headers=headers) as response:
                status = response.status
                response_data = await response.json()
                
                if status == 200 and "document_id" in response_data:
                    self.test_document_id = response_data["document_id"]
                    self.log_result("Document Upload", "PASS", {
                        "status_code": status,
                        "document_id": self.test_document_id,
                        "filename": response_data.get("filename"),
                        "chunks_created": response_data.get("chunks_created")
                    })
                    return True
                else:
                    self.log_result("Document Upload", "FAIL", {
                        "status_code": status,
                        "response": response_data
                    })
                    return False
                    
        except Exception as e:
            self.log_result("Document Upload", "FAIL", error=f"Upload error: {str(e)}")
            return False
    
    async def test_document_retrieval(self):
        """Test document retrieval endpoints."""
        logger.info("Testing document retrieval...")
        
        if not self.auth_token or not self.test_document_id:
            self.log_result("Document Retrieval", "SKIP", error="No auth token or document ID")
            return False
        
        # Test get specific document
        status, data, error = await self.make_request(
            'GET', 
            f'/api/v1/documents/{self.test_document_id}'
        )
        
        if status == 200 and data.get("id") == self.test_document_id:
            self.log_result("Document Retrieval", "PASS", {
                "status_code": status,
                "document_id": data.get("id"),
                "filename": data.get("filename"),
                "processed": data.get("processed")
            })
            return True
        else:
            self.log_result("Document Retrieval", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_document_list(self):
        """Test document list endpoint."""
        logger.info("Testing document list...")
        
        if not self.auth_token:
            self.log_result("Document List", "SKIP", error="No auth token available")
            return False
        
        status, data, error = await self.make_request('GET', '/api/v1/documents/')
        
        if status == 200 and isinstance(data, list):
            self.log_result("Document List", "PASS", {
                "status_code": status,
                "documents_count": len(data),
                "has_test_doc": any(doc.get("id") == self.test_document_id for doc in data)
            })
            return True
        else:
            self.log_result("Document List", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_quick_text_analysis(self):
        """Test quick text analysis endpoint."""
        logger.info("Testing quick text analysis...")
        
        if not self.auth_token:
            self.log_result("Quick Text Analysis", "SKIP", error="No auth token available")
            return False
        
        analysis_data = {
            "text": self.test_document_content,
            "analysis_type": "arbitration_detection"
        }
        
        status, data, error = await self.make_request(
            'POST',
            '/api/v1/analysis/quick-analyze',
            json=analysis_data
        )
        
        if status == 200 and "has_arbitration_clause" in data:
            self.log_result("Quick Text Analysis", "PASS", {
                "status_code": status,
                "has_arbitration": data.get("has_arbitration_clause"),
                "confidence_score": data.get("confidence_score"),
                "clauses_found": len(data.get("clauses_found", [])),
                "processing_time_ms": data.get("processing_time_ms")
            })
            return True
        else:
            self.log_result("Quick Text Analysis", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_document_analysis(self):
        """Test document analysis endpoint."""
        logger.info("Testing document analysis...")
        
        if not self.auth_token or not self.test_document_id:
            self.log_result("Document Analysis", "SKIP", error="No auth token or document ID")
            return False
        
        analysis_data = {
            "document_id": self.test_document_id,
            "analysis_type": "arbitration_detection"
        }
        
        status, data, error = await self.make_request(
            'POST',
            '/api/v1/analysis/analyze',
            json=analysis_data
        )
        
        if status == 200 and "id" in data:
            self.test_analysis_id = data["id"]
            self.log_result("Document Analysis", "PASS", {
                "status_code": status,
                "analysis_id": self.test_analysis_id,
                "has_arbitration": data.get("has_arbitration_clause"),
                "confidence_score": data.get("confidence_score"),
                "clauses_count": len(data.get("clauses", []))
            })
            return True
        else:
            self.log_result("Document Analysis", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_analysis_retrieval(self):
        """Test analysis retrieval endpoints."""
        logger.info("Testing analysis retrieval...")
        
        if not self.auth_token or not self.test_analysis_id:
            self.log_result("Analysis Retrieval", "SKIP", error="No auth token or analysis ID")
            return False
        
        # Test get specific analysis
        status, data, error = await self.make_request(
            'GET',
            f'/api/v1/analysis/{self.test_analysis_id}'
        )
        
        if status == 200 and data.get("id") == self.test_analysis_id:
            self.log_result("Analysis Retrieval", "PASS", {
                "status_code": status,
                "analysis_id": data.get("id"),
                "document_id": data.get("document_id"),
                "has_arbitration": data.get("has_arbitration_clause"),
                "processed_at": data.get("analyzed_at")
            })
            return True
        else:
            self.log_result("Analysis Retrieval", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_analysis_statistics(self):
        """Test analysis statistics endpoints."""
        logger.info("Testing analysis statistics...")
        
        if not self.auth_token:
            self.log_result("Analysis Statistics", "SKIP", error="No auth token available")
            return False
        
        status, data, error = await self.make_request('GET', '/api/v1/analysis/stats/overview')
        
        if status == 200 and isinstance(data, dict):
            self.log_result("Analysis Statistics", "PASS", {
                "status_code": status,
                "total_analyses": data.get("total_analyses", 0),
                "with_arbitration": data.get("analyses_with_arbitration", 0),
                "avg_confidence": data.get("average_confidence_score", 0)
            })
            return True
        else:
            self.log_result("Analysis Statistics", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        logger.info("Testing WebSocket connection...")
        
        if not self.auth_token:
            self.log_result("WebSocket Connection", "SKIP", error="No auth token available")
            return False
        
        ws_url = f"{WS_URL}?token={self.auth_token}"
        
        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Send a test message
                test_message = {
                    "event_type": "ping",
                    "data": {"timestamp": datetime.now().isoformat()}
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response_data = json.loads(response)
                    
                    self.log_result("WebSocket Connection", "PASS", {
                        "connected": True,
                        "response_received": True,
                        "response_type": response_data.get("event_type"),
                        "connection_id": response_data.get("data", {}).get("connection_id")
                    })
                    return True
                    
                except asyncio.TimeoutError:
                    self.log_result("WebSocket Connection", "PARTIAL", {
                        "connected": True,
                        "response_received": False
                    }, "No response received within timeout")
                    return True
                    
        except websockets.exceptions.ConnectionClosed as e:
            self.log_result("WebSocket Connection", "FAIL", error=f"Connection closed: {e}")
            return False
        except websockets.exceptions.InvalidHandshake as e:
            self.log_result("WebSocket Connection", "FAIL", error=f"Invalid handshake: {e}")
            return False
        except Exception as e:
            self.log_result("WebSocket Connection", "FAIL", error=f"Connection error: {str(e)}")
            return False
    
    async def test_websocket_stats_endpoint(self):
        """Test WebSocket statistics endpoint."""
        logger.info("Testing WebSocket statistics endpoint...")
        
        if not self.auth_token:
            self.log_result("WebSocket Statistics", "SKIP", error="No auth token available")
            return False
        
        status, data, error = await self.make_request('GET', '/api/websocket/stats')
        
        if status == 200 and "server" in data:
            self.log_result("WebSocket Statistics", "PASS", {
                "status_code": status,
                "active_connections": data.get("connections", {}).get("active_connections", 0),
                "active_users": data.get("connections", {}).get("active_users", 0),
                "server_uptime": data.get("server", {}).get("uptime_seconds", 0)
            })
            return True
        else:
            self.log_result("WebSocket Statistics", "FAIL", {
                "status_code": status,
                "response": data
            }, error)
            return False
    
    async def test_cors_headers(self):
        """Test CORS headers on API endpoints."""
        logger.info("Testing CORS headers...")
        
        try:
            async with self.session.options(f"{BASE_URL}/api/v1") as response:
                cors_headers = {
                    "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
                    "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
                    "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers"),
                }
                
                has_cors = any(cors_headers.values())
                
                self.log_result("CORS Headers", "PASS" if has_cors else "FAIL", {
                    "status_code": response.status,
                    "cors_headers_present": has_cors,
                    "allow_origin": cors_headers["Access-Control-Allow-Origin"],
                    "allow_methods": cors_headers["Access-Control-Allow-Methods"]
                })
                return has_cors
                
        except Exception as e:
            self.log_result("CORS Headers", "FAIL", error=f"CORS test error: {str(e)}")
            return False
    
    async def test_error_handling(self):
        """Test API error handling."""
        logger.info("Testing API error handling...")
        
        # Test invalid endpoint
        status, data, error = await self.make_request('GET', '/api/v1/nonexistent')
        
        if status == 404:
            self.log_result("Error Handling - 404", "PASS", {
                "status_code": status,
                "error_format": "proper" if "detail" in data else "basic"
            })
        else:
            self.log_result("Error Handling - 404", "FAIL", {
                "status_code": status,
                "expected": 404,
                "response": data
            })
        
        # Test invalid JSON
        try:
            async with self.session.post(f"{BASE_URL}/api/v1/analysis/quick-analyze",
                                       data="invalid json",
                                       headers={"content-type": "application/json"}) as response:
                status = response.status
                if status >= 400:
                    self.log_result("Error Handling - Invalid JSON", "PASS", {
                        "status_code": status,
                        "properly_handled": True
                    })
                else:
                    self.log_result("Error Handling - Invalid JSON", "FAIL", {
                        "status_code": status,
                        "expected": "4xx error"
                    })
        except Exception as e:
            self.log_result("Error Handling - Invalid JSON", "PARTIAL", 
                          error=f"Exception during invalid JSON test: {str(e)}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting comprehensive integration test suite...")
        start_time = time.time()
        
        await self.setup()
        
        try:
            # Core connectivity tests
            await self.test_health_check()
            await self.test_api_overview()
            await self.test_cors_headers()
            await self.test_error_handling()
            
            # Authentication tests
            await self.test_user_registration()
            await self.test_user_login()
            await self.test_token_verification()
            await self.test_user_profile()
            
            # Document management tests
            await self.test_document_upload()
            await self.test_document_retrieval()
            await self.test_document_list()
            
            # Analysis tests
            await self.test_quick_text_analysis()
            await self.test_document_analysis()
            await self.test_analysis_retrieval()
            await self.test_analysis_statistics()
            
            # WebSocket tests
            await self.test_websocket_connection()
            await self.test_websocket_stats_endpoint()
            
        finally:
            await self.cleanup()
        
        end_time = time.time()
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r["status"] == "SKIP"])
        partial_tests = len([r for r in self.test_results if r["status"] == "PARTIAL"])
        
        summary = {
            "test_suite": "Frontend-Backend Integration Tests",
            "execution_time_seconds": round(end_time - start_time, 2),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "partial": partial_tests,
            "success_rate": round((passed_tests / max(total_tests - skipped_tests, 1)) * 100, 2),
            "timestamp": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "test_results": self.test_results
        }
        
        return summary


async def main():
    """Main test execution function."""
    test_suite = IntegrationTestSuite()
    results = await test_suite.run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("FRONTEND-BACKEND INTEGRATION TEST RESULTS")
    print("="*80)
    print(f"Execution Time: {results['execution_time_seconds']} seconds")
    print(f"Total Tests: {results['total_tests']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⚠️  Partial: {results['partial']}")
    print(f"⏭️  Skipped: {results['skipped']}")
    print(f"Success Rate: {results['success_rate']}%")
    print("\nDetailed Results:")
    print("-"*40)
    
    # Group results by status
    for status in ["FAIL", "PARTIAL", "SKIP", "PASS"]:
        status_results = [r for r in results['test_results'] if r['status'] == status]
        if status_results:
            status_emoji = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "⚠️", "SKIP": "⏭️"}[status]
            print(f"\n{status_emoji} {status} ({len(status_results)} tests):")
            for result in status_results:
                print(f"  • {result['test_name']}")
                if result.get('error'):
                    print(f"    Error: {result['error']}")
                if result.get('details'):
                    key_details = {k: v for k, v in result['details'].items() 
                                 if k in ['status_code', 'has_arbitration', 'connected', 'user_id', 'document_id']}
                    if key_details:
                        print(f"    Details: {key_details}")
    
    # Save detailed results to file
    import json
    results_file = Path(__file__).parent / "integration_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())