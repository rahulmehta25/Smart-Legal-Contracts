#!/usr/bin/env python3
"""
Basic API Endpoint Testing Script

Tests basic connectivity and non-database endpoints to validate frontend-backend communication.
This script focuses on endpoints that don't require database connectivity.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import websockets
import sys

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

class BasicEndpointTester:
    def __init__(self):
        self.session = None
        self.results = []
    
    async def setup(self):
        """Setup HTTP session."""
        self.session = aiohttp.ClientSession()
        print("🔧 Test session initialized")
    
    async def cleanup(self):
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
        print("🧹 Test session cleaned up")
    
    def log_result(self, endpoint, method, status, success, details=None, error=None):
        """Log test result."""
        result = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "error": error
        }
        self.results.append(result)
        
        status_emoji = "✅" if success else "❌"
        print(f"{status_emoji} {method} {endpoint} - Status: {status}" + (f" - {error}" if error else ""))
        
        if details:
            print(f"   📊 {details}")
    
    async def test_endpoint(self, method, endpoint, **kwargs):
        """Test a single endpoint."""
        url = f"{BASE_URL}{endpoint}"
        
        try:
            async with self.session.request(method, url, timeout=aiohttp.ClientTimeout(total=10), **kwargs) as response:
                status = response.status
                
                try:
                    if response.content_type == 'application/json':
                        data = await response.json()
                    else:
                        text = await response.text()
                        data = {"text": text[:200] + "..." if len(text) > 200 else text}
                except:
                    data = {"raw_response": "Unable to parse response"}
                
                success = 200 <= status < 400
                self.log_result(endpoint, method, status, success, details=data)
                return success, status, data
                
        except asyncio.TimeoutError:
            self.log_result(endpoint, method, 0, False, error="Timeout")
            return False, 0, {}
        except Exception as e:
            self.log_result(endpoint, method, 0, False, error=str(e))
            return False, 0, {}
    
    async def test_websocket_basic(self):
        """Test basic WebSocket connectivity (without auth)."""
        print("\n🔌 Testing WebSocket basic connectivity...")
        
        try:
            # Try to connect without token first to see if endpoint exists
            async with websockets.connect(WS_URL, timeout=5) as websocket:
                self.log_result("/ws", "WebSocket", 200, True, {"connected": True})
                return True
        except websockets.exceptions.ConnectionClosed as e:
            # This is expected without auth token
            if "4xx" in str(e) or "401" in str(e) or "403" in str(e):
                self.log_result("/ws", "WebSocket", 401, True, {"auth_required": True, "endpoint_exists": True})
                return True
            else:
                self.log_result("/ws", "WebSocket", 0, False, error=f"Connection closed: {e}")
                return False
        except Exception as e:
            self.log_result("/ws", "WebSocket", 0, False, error=str(e))
            return False
    
    async def run_all_tests(self):
        """Run all basic endpoint tests."""
        print("🚀 Starting Basic API Endpoint Tests")
        print("="*60)
        
        start_time = time.time()
        await self.setup()
        
        # Core endpoints that don't require database
        print("\n📍 Testing Core Endpoints...")
        await self.test_endpoint("GET", "/health")
        await self.test_endpoint("GET", "/")
        await self.test_endpoint("GET", "/api/v1")
        
        # Test 404 handling
        print("\n🚫 Testing Error Handling...")
        await self.test_endpoint("GET", "/nonexistent")
        await self.test_endpoint("GET", "/api/v1/nonexistent")
        
        # Test CORS preflight
        print("\n🌐 Testing CORS...")
        await self.test_endpoint("OPTIONS", "/api/v1", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # Test endpoints that might work without DB
        print("\n📊 Testing API Documentation...")
        await self.test_endpoint("GET", "/docs")
        await self.test_endpoint("GET", "/redoc")
        await self.test_endpoint("GET", "/openapi.json")
        
        # Test WebSocket
        await self.test_websocket_basic()
        
        # Test database-dependent endpoints (expect failures)
        print("\n🗄️ Testing Database-Dependent Endpoints (Expected Failures)...")
        await self.test_endpoint("GET", "/api/v1/documents/")
        await self.test_endpoint("POST", "/api/v1/users/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        })
        
        await self.cleanup()
        end_time = time.time()
        
        # Generate summary
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r["success"]])
        failed_tests = total_tests - successful_tests
        
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        print(f"⏱️  Execution Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Categorize results
        working_endpoints = [r for r in self.results if r["success"] and 200 <= r["status_code"] < 300]
        auth_required = [r for r in self.results if r["status_code"] in [401, 403]]
        not_found = [r for r in self.results if r["status_code"] == 404]
        server_errors = [r for r in self.results if r["status_code"] >= 500]
        connection_errors = [r for r in self.results if r["status_code"] == 0]
        
        if working_endpoints:
            print(f"\n✅ WORKING ENDPOINTS ({len(working_endpoints)}):")
            for result in working_endpoints:
                print(f"   • {result['method']} {result['endpoint']} - {result['status_code']}")
        
        if auth_required:
            print(f"\n🔒 AUTHENTICATION REQUIRED ({len(auth_required)}):")
            for result in auth_required:
                print(f"   • {result['method']} {result['endpoint']} - {result['status_code']}")
        
        if server_errors:
            print(f"\n🚨 SERVER ERRORS (Need Fixes) ({len(server_errors)}):")
            for result in server_errors:
                print(f"   • {result['method']} {result['endpoint']} - {result['status_code']} - {result.get('error', 'Internal Server Error')}")
        
        if connection_errors:
            print(f"\n🔌 CONNECTION ERRORS ({len(connection_errors)}):")
            for result in connection_errors:
                print(f"   • {result['method']} {result['endpoint']} - {result.get('error', 'Connection failed')}")
        
        if not_found:
            print(f"\n🔍 NOT FOUND (Expected) ({len(not_found)}):")
            for result in not_found:
                print(f"   • {result['method']} {result['endpoint']}")
        
        print("\n" + "="*60)
        print("🔧 RECOMMENDATIONS:")
        print("="*60)
        
        recommendations = []
        
        if server_errors:
            recommendations.append("🗄️  Database connectivity issues detected - install PostgreSQL dependencies")
            recommendations.append("   Run: pip install psycopg2-binary")
            
        if not any(r["endpoint"] == "/ws" and r["success"] for r in self.results):
            recommendations.append("🔌 WebSocket endpoint may need authentication token")
            
        cors_test = next((r for r in self.results if r["method"] == "OPTIONS"), None)
        if cors_test and not cors_test["success"]:
            recommendations.append("🌐 CORS preflight may need additional configuration")
            
        if not recommendations:
            recommendations.append("✨ Basic connectivity looks good! Database setup needed for full functionality.")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)
        
        # Save results
        results_file = "basic_endpoint_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "successful": successful_tests,
                    "failed": failed_tests,
                    "success_rate": (successful_tests/total_tests)*100,
                    "execution_time": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                },
                "results": self.results,
                "recommendations": recommendations
            }, f, indent=2)
        
        print(f"📁 Detailed results saved to: {results_file}")
        
        return self.results


async def main():
    tester = BasicEndpointTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())