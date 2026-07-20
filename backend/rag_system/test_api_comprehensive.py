#!/usr/bin/env python3
"""
Comprehensive API Testing Suite for RAG Arbitration Detection System

This test suite covers all API endpoints with various scenarios including:
- Health checks and connectivity
- File upload detection 
- Text-based detection
- Clause comparison
- Database operations
- Batch processing
- Error handling
- Performance testing
- Concurrent requests
"""

import asyncio
import aiofiles
import httpx
import time
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import statistics

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30.0

class APITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {}
        self.test_files = {}
        
    async def setup_test_files(self):
        """Create test files for upload testing."""
        # Sample arbitration clause text
        arbitration_text = """
        Any dispute, claim or controversy arising out of or relating to this Agreement or 
        the breach, termination, enforcement, interpretation or validity thereof, including 
        the determination of the scope or applicability of this agreement to arbitrate, 
        shall be determined by arbitration in New York, New York before one arbitrator. 
        The arbitration shall be administered by JAMS pursuant to its Comprehensive 
        Arbitration Rules and Procedures and in accordance with the Expedited Procedures 
        in those Rules. Judgment on the Award may be entered in any court having jurisdiction.
        """
        
        # Create temporary test files
        test_files = {
            'arbitration_clause.txt': arbitration_text,
            'contract_with_arbitration.txt': f"TERMS OF SERVICE\n\n{arbitration_text}\n\nOther contract terms...",
            'no_arbitration.txt': "This is a simple contract without any arbitration clauses. Standard terms apply.",
            'empty.txt': "",
            'invalid.xyz': "Invalid file type content"
        }
        
        for filename, content in test_files.items():
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False)
            temp_file.write(content)
            temp_file.close()
            self.test_files[filename] = temp_file.name
            
    def cleanup_test_files(self):
        """Clean up temporary test files."""
        for filepath in self.test_files.values():
            try:
                os.unlink(filepath)
            except:
                pass
                
    async def test_health_endpoint(self) -> Dict:
        """Test the health check endpoint."""
        print("Testing GET /health endpoint...")
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                response_time = time.time() - start_time
                
                result = {
                    "endpoint": "GET /health",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200,
                    "response_data": response.json() if response.status_code == 200 else response.text
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result["components_status"] = data.get("components", {})
                    
            except Exception as e:
                result = {
                    "endpoint": "GET /health",
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                
        return result
        
    async def test_text_detection(self) -> Dict:
        """Test text-based arbitration detection."""
        print("Testing POST /detect/text endpoint...")
        
        test_cases = [
            {
                "name": "clear_arbitration_clause",
                "text": "Any disputes shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.",
                "expected_detected": True
            },
            {
                "name": "no_arbitration", 
                "text": "Standard terms and conditions apply. Disputes will be resolved in court.",
                "expected_detected": False
            },
            {
                "name": "empty_text",
                "text": "",
                "expected_detected": False
            }
        ]
        
        results = []
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for test_case in test_cases:
                start_time = time.time()
                try:
                    payload = {
                        "text": test_case["text"],
                        "threshold": 0.7,
                        "explain": True,
                        "compare": True
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/detect/text",
                        json=payload
                    )
                    response_time = time.time() - start_time
                    
                    result = {
                        "test_case": test_case["name"],
                        "endpoint": "POST /detect/text",
                        "status_code": response.status_code,
                        "response_time_ms": round(response_time * 1000, 2),
                        "success": response.status_code == 200,
                        "text_length": len(test_case["text"])
                    }
                    
                    if response.status_code == 200:
                        data = response.json()
                        result.update({
                            "detected": data.get("detected"),
                            "confidence": data.get("confidence"),
                            "expected_detected": test_case["expected_detected"],
                            "prediction_correct": data.get("detected") == test_case["expected_detected"]
                        })
                    else:
                        result["error"] = response.text
                        
                except Exception as e:
                    result = {
                        "test_case": test_case["name"],
                        "endpoint": "POST /detect/text", 
                        "success": False,
                        "error": str(e),
                        "response_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                    
                results.append(result)
                
        return {"test_results": results}
        
    async def test_file_upload_detection(self) -> Dict:
        """Test file upload detection endpoint."""
        print("Testing POST /detect endpoint (file upload)...")
        
        results = []
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for filename, filepath in self.test_files.items():
                start_time = time.time()
                try:
                    with open(filepath, 'rb') as file:
                        files = {"file": (filename, file, "text/plain")}
                        params = {"threshold": 0.7, "explain": True, "compare": True}
                        
                        response = await client.post(
                            f"{self.base_url}/detect",
                            files=files,
                            params=params
                        )
                        
                    response_time = time.time() - start_time
                    
                    result = {
                        "test_file": filename,
                        "endpoint": "POST /detect",
                        "status_code": response.status_code,
                        "response_time_ms": round(response_time * 1000, 2),
                        "success": response.status_code in [200, 400]  # 400 expected for invalid files
                    }
                    
                    if response.status_code == 200:
                        data = response.json()
                        result.update({
                            "detected": data.get("detected"),
                            "confidence": data.get("confidence"),
                            "has_explanation": "explanation" in data,
                            "has_similar_clauses": "similar_clauses" in data
                        })
                    elif response.status_code == 400:
                        result["error"] = response.text
                        result["expected_error"] = filename.endswith('.xyz')  # Invalid file type
                    else:
                        result["error"] = response.text
                        
                except Exception as e:
                    result = {
                        "test_file": filename,
                        "endpoint": "POST /detect",
                        "success": False,
                        "error": str(e),
                        "response_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                    
                results.append(result)
                
        return {"test_results": results}
        
    async def test_clause_comparison(self) -> Dict:
        """Test clause comparison endpoint."""
        print("Testing POST /compare endpoint...")
        
        test_clause = "Disputes shall be resolved by binding arbitration under AAA rules."
        
        start_time = time.time()
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                payload = {
                    "clause_text": test_clause,
                    "top_k": 10
                }
                
                response = await client.post(
                    f"{self.base_url}/compare",
                    json=payload
                )
                response_time = time.time() - start_time
                
                result = {
                    "endpoint": "POST /compare",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200,
                    "clause_length": len(test_clause)
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result.update({
                        "has_similar_clauses": "similar_clauses" in data,
                        "similar_count": len(data.get("similar_clauses", [])) if data.get("similar_clauses") else 0
                    })
                else:
                    result["error"] = response.text
                    
            except Exception as e:
                result = {
                    "endpoint": "POST /compare",
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                
        return result
        
    async def test_database_operations(self) -> Dict:
        """Test database-related endpoints."""
        print("Testing database operations...")
        
        results = []
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test database stats
            start_time = time.time()
            try:
                response = await client.get(f"{self.base_url}/database/stats")
                response_time = time.time() - start_time
                
                stats_result = {
                    "endpoint": "GET /database/stats",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    data = response.json()
                    stats_result["stats_data"] = data
                else:
                    stats_result["error"] = response.text
                    
                results.append(stats_result)
                
            except Exception as e:
                results.append({
                    "endpoint": "GET /database/stats",
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                })
                
            # Test adding clause to database
            start_time = time.time()
            try:
                clause_data = {
                    "text": "Test arbitration clause for database",
                    "company": "Test Corp",
                    "industry": "Technology", 
                    "document_type": "TOS",
                    "jurisdiction": "US",
                    "enforceability": 0.8,
                    "risk_score": 0.6,
                    "metadata": {"test": True}
                }
                
                response = await client.post(
                    f"{self.base_url}/database/add",
                    json=clause_data
                )
                response_time = time.time() - start_time
                
                add_result = {
                    "endpoint": "POST /database/add",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    data = response.json()
                    add_result["clause_added"] = data.get("success", False)
                    add_result["clause_id"] = data.get("clause_id")
                else:
                    add_result["error"] = response.text
                    
                results.append(add_result)
                
            except Exception as e:
                results.append({
                    "endpoint": "POST /database/add",
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                })
                
            # Test database search
            start_time = time.time()
            try:
                params = {
                    "company": "Test Corp",
                    "industry": "Technology"
                }
                
                response = await client.get(
                    f"{self.base_url}/database/search",
                    params=params
                )
                response_time = time.time() - start_time
                
                search_result = {
                    "endpoint": "GET /database/search",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    data = response.json()
                    search_result["results_count"] = data.get("count", 0)
                else:
                    search_result["error"] = response.text
                    
                results.append(search_result)
                
            except Exception as e:
                results.append({
                    "endpoint": "GET /database/search",
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                })
                
        return {"test_results": results}
        
    async def test_batch_analysis(self) -> Dict:
        """Test batch analysis endpoint."""
        print("Testing POST /analyze/batch endpoint...")
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                files = []
                for filename, filepath in list(self.test_files.items())[:3]:  # Limit to 3 files
                    with open(filepath, 'rb') as f:
                        files.append(("files", (filename, f.read(), "text/plain")))
                
                response = await client.post(
                    f"{self.base_url}/analyze/batch",
                    files=files
                )
                response_time = time.time() - start_time
                
                result = {
                    "endpoint": "POST /analyze/batch",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200,
                    "files_sent": len(files)
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result.update({
                        "total_files": data.get("total_files"),
                        "processed_files": data.get("processed"),
                        "results_count": len(data.get("results", []))
                    })
                else:
                    result["error"] = response.text
                    
            except Exception as e:
                result = {
                    "endpoint": "POST /analyze/batch",
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                
        return result
        
    async def test_error_handling(self) -> Dict:
        """Test error handling scenarios."""
        print("Testing error handling scenarios...")
        
        results = []
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test invalid endpoint
            try:
                response = await client.get(f"{self.base_url}/invalid-endpoint")
                results.append({
                    "test": "invalid_endpoint",
                    "status_code": response.status_code,
                    "success": response.status_code == 404
                })
            except Exception as e:
                results.append({
                    "test": "invalid_endpoint",
                    "success": False,
                    "error": str(e)
                })
                
            # Test malformed JSON
            try:
                response = await client.post(
                    f"{self.base_url}/detect/text",
                    content="invalid json",
                    headers={"Content-Type": "application/json"}
                )
                results.append({
                    "test": "malformed_json",
                    "status_code": response.status_code,
                    "success": response.status_code == 422  # Unprocessable entity
                })
            except Exception as e:
                results.append({
                    "test": "malformed_json",
                    "success": False,
                    "error": str(e)
                })
                
            # Test missing required fields
            try:
                response = await client.post(
                    f"{self.base_url}/detect/text",
                    json={"threshold": 0.7}  # Missing 'text' field
                )
                results.append({
                    "test": "missing_required_field",
                    "status_code": response.status_code,
                    "success": response.status_code == 422
                })
            except Exception as e:
                results.append({
                    "test": "missing_required_field",
                    "success": False,
                    "error": str(e)
                })
                
        return {"test_results": results}
        
    async def test_performance_and_concurrent_requests(self) -> Dict:
        """Test performance and concurrent request handling."""
        print("Testing performance and concurrent requests...")
        
        # Test concurrent requests
        concurrent_results = []
        
        async def make_concurrent_request(client, request_id):
            start_time = time.time()
            try:
                payload = {
                    "text": f"Test arbitration clause {request_id} for concurrent testing",
                    "threshold": 0.7
                }
                response = await client.post(f"{self.base_url}/detect/text", json=payload)
                response_time = time.time() - start_time
                
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2),
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "response_time_ms": round((time.time() - start_time) * 1000, 2)
                }
        
        # Send 10 concurrent requests
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = [make_concurrent_request(client, i) for i in range(10)]
            concurrent_results = await asyncio.gather(*tasks)
            
        # Calculate performance metrics
        response_times = [r["response_time_ms"] for r in concurrent_results if "response_time_ms" in r]
        successful_requests = sum(1 for r in concurrent_results if r.get("success", False))
        
        performance_metrics = {
            "total_concurrent_requests": len(concurrent_results),
            "successful_requests": successful_requests,
            "success_rate": round(successful_requests / len(concurrent_results) * 100, 2),
            "avg_response_time_ms": round(statistics.mean(response_times), 2) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "median_response_time_ms": round(statistics.median(response_times), 2) if response_times else 0
        }
        
        return {
            "performance_metrics": performance_metrics,
            "concurrent_results": concurrent_results
        }
        
    async def test_cors_and_middleware(self) -> Dict:
        """Test CORS and middleware functionality."""
        print("Testing CORS and middleware...")
        
        results = []
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test CORS preflight
            try:
                response = await client.options(
                    f"{self.base_url}/health",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "GET"
                    }
                )
                
                results.append({
                    "test": "cors_preflight",
                    "status_code": response.status_code,
                    "has_cors_headers": "access-control-allow-origin" in response.headers,
                    "success": response.status_code in [200, 204]
                })
                
            except Exception as e:
                results.append({
                    "test": "cors_preflight",
                    "success": False,
                    "error": str(e)
                })
                
            # Test actual CORS request
            try:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers={"Origin": "http://localhost:3000"}
                )
                
                results.append({
                    "test": "cors_actual_request",
                    "status_code": response.status_code,
                    "has_cors_headers": "access-control-allow-origin" in response.headers,
                    "cors_origin": response.headers.get("access-control-allow-origin"),
                    "success": response.status_code == 200
                })
                
            except Exception as e:
                results.append({
                    "test": "cors_actual_request",
                    "success": False,
                    "error": str(e)
                })
                
        return {"test_results": results}
        
    async def run_all_tests(self) -> Dict:
        """Run all tests and compile results."""
        print("Starting comprehensive API testing...")
        print("=" * 60)
        
        await self.setup_test_files()
        
        all_results = {
            "test_start_time": datetime.now().isoformat(),
            "base_url": self.base_url
        }
        
        try:
            # Run all test categories
            all_results["health_check"] = await self.test_health_endpoint()
            all_results["text_detection"] = await self.test_text_detection()
            all_results["file_upload"] = await self.test_file_upload_detection()
            all_results["clause_comparison"] = await self.test_clause_comparison()
            all_results["database_operations"] = await self.test_database_operations()
            all_results["batch_analysis"] = await self.test_batch_analysis()
            all_results["error_handling"] = await self.test_error_handling()
            all_results["performance_concurrent"] = await self.test_performance_and_concurrent_requests()
            all_results["cors_middleware"] = await self.test_cors_and_middleware()
            
        finally:
            self.cleanup_test_files()
            
        all_results["test_end_time"] = datetime.now().isoformat()
        
        # Generate summary
        summary = self.generate_test_summary(all_results)
        all_results["summary"] = summary
        
        return all_results
        
    def generate_test_summary(self, results: Dict) -> Dict:
        """Generate a summary of test results."""
        total_tests = 0
        successful_tests = 0
        
        # Count tests from each category
        for category, data in results.items():
            if category in ["test_start_time", "test_end_time", "base_url"]:
                continue
                
            if isinstance(data, dict):
                if "test_results" in data:
                    for test in data["test_results"]:
                        total_tests += 1
                        if test.get("success", False):
                            successful_tests += 1
                elif "success" in data:
                    total_tests += 1
                    if data.get("success", False):
                        successful_tests += 1
                        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": round(successful_tests / total_tests * 100, 2) if total_tests > 0 else 0,
            "test_categories": len([k for k in results.keys() if k not in ["test_start_time", "test_end_time", "base_url", "summary"]])
        }

async def main():
    """Main function to run the comprehensive API tests."""
    tester = APITester()
    
    print("RAG System API Comprehensive Testing Suite")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")
    print()
    
    try:
        results = await tester.run_all_tests()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Test Categories: {summary['test_categories']}")
        
        print(f"\nDetailed results saved to: {filename}")
        print(f"Test Duration: {results['test_start_time']} to {results['test_end_time']}")
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        
if __name__ == "__main__":
    asyncio.run(main())