"""
Locust Load Testing for RAG Legal Analysis System

This script simulates realistic user load patterns including:
- Document uploads (various sizes)
- Arbitration analysis requests
- Concurrent user sessions
- API endpoint stress testing
- WebSocket connections
"""

from locust import HttpUser, task, between, TaskSet, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import json
import random
import time
import os
import base64
from datetime import datetime
import gevent
import websocket
from typing import Dict, List, Any


class DocumentGenerator:
    """Generate test documents for load testing"""
    
    @staticmethod
    def generate_test_document(doc_type: str = "medium") -> str:
        """Generate a test document with arbitration content"""
        
        templates = {
            "small": """
                TERMS OF SERVICE
                
                1. ARBITRATION AGREEMENT
                By using our service, you agree to resolve any disputes through binding arbitration.
                You waive your right to jury trial and class action lawsuits.
                
                2. GENERAL TERMS
                These terms govern your use of our platform and services.
            """,
            "medium": """
                TERMS OF SERVICE AND USER AGREEMENT
                
                1. INTRODUCTION
                Welcome to our platform. By accessing or using our services, you agree to be bound by these terms.
                
                2. ARBITRATION CLAUSE
                Any dispute, claim or controversy arising out of or relating to this Agreement or the breach, 
                termination, enforcement, interpretation or validity thereof, including the determination of 
                the scope or applicability of this agreement to arbitrate, shall be determined by arbitration 
                in California before one arbitrator. The arbitration shall be administered by JAMS pursuant 
                to its Comprehensive Arbitration Rules and Procedures. Judgment on the Award may be entered 
                in any court having jurisdiction. This clause shall not preclude parties from seeking 
                provisional remedies in aid of arbitration from a court of appropriate jurisdiction.
                
                3. CLASS ACTION WAIVER
                You agree that any arbitration or proceeding shall be limited to the Dispute between us and 
                you individually. To the full extent permitted by law, (a) no arbitration or proceeding shall 
                be joined with any other; (b) there is no right or authority for any Dispute to be arbitrated 
                or resolved on a class action-basis or to utilize class action procedures; and (c) there is 
                no right or authority for any Dispute to be brought in a purported representative capacity 
                on behalf of the general public or any other persons.
                
                4. LIMITATION OF LIABILITY
                In no event shall we be liable for any indirect, incidental, special, consequential or 
                punitive damages, or any loss of profits or revenues.
                
                5. GOVERNING LAW
                These Terms shall be governed by the laws of the State of California without regard to its 
                conflict of law provisions.
            """ * 5,  # Repeat to make it larger
            "large": """
                COMPREHENSIVE SERVICE AGREEMENT AND TERMS OF USE
                
                [Large document content with multiple arbitration clauses...]
            """ * 100  # Very large document
        }
        
        return templates.get(doc_type, templates["medium"])
    
    @staticmethod
    def generate_pdf_content() -> bytes:
        """Generate mock PDF content for testing"""
        # Simple PDF-like binary content (not a real PDF)
        content = b"%PDF-1.4\n" + b"Mock PDF content with arbitration clause\n" * 100
        return content


class ArbitrationAnalysisTasks(TaskSet):
    """Task set for arbitration analysis operations"""
    
    def on_start(self):
        """Initialize user session"""
        # Simulate user login/authentication if needed
        self.user_token = None
        self.document_ids = []
    
    @task(10)
    def analyze_text_document(self):
        """Submit text document for analysis"""
        doc_size = random.choice(["small", "medium", "medium", "large"])  # Weight towards medium
        document = DocumentGenerator.generate_test_document(doc_size)
        
        with self.client.post(
            "/api/v1/arbitration/analyze",
            json={
                "text": document,
                "document_type": "terms_of_service",
                "options": {
                    "detailed_analysis": True,
                    "extract_clauses": True
                }
            },
            catch_response=True,
            name=f"analyze_text_{doc_size}"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "has_arbitration_clause" in result:
                    response.success()
                    # Store document ID for later operations
                    if "document_id" in result:
                        self.document_ids.append(result["document_id"])
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def upload_pdf_document(self):
        """Upload PDF document for analysis"""
        pdf_content = DocumentGenerator.generate_pdf_content()
        
        with self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
            data={"document_type": "contract"},
            catch_response=True,
            name="upload_pdf"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
                result = response.json()
                if "document_id" in result:
                    self.document_ids.append(result["document_id"])
            else:
                response.failure(f"Upload failed: {response.status_code}")
    
    @task(3)
    def batch_analysis(self):
        """Submit batch of documents for analysis"""
        batch_size = random.randint(2, 5)
        documents = [
            {
                "id": f"doc_{i}",
                "text": DocumentGenerator.generate_test_document("small")
            }
            for i in range(batch_size)
        ]
        
        with self.client.post(
            "/api/v1/arbitration/batch",
            json={"documents": documents},
            catch_response=True,
            name=f"batch_analysis_{batch_size}"
        ) as response:
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list) and len(results) == batch_size:
                    response.success()
                else:
                    response.failure("Batch size mismatch")
            else:
                response.failure(f"Batch failed: {response.status_code}")
    
    @task(8)
    def check_analysis_status(self):
        """Check status of previous analysis"""
        if self.document_ids:
            doc_id = random.choice(self.document_ids)
            
            with self.client.get(
                f"/api/v1/documents/{doc_id}/status",
                catch_response=True,
                name="check_status"
            ) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 404:
                    response.failure("Document not found")
                    self.document_ids.remove(doc_id)
                else:
                    response.failure(f"Status check failed: {response.status_code}")
    
    @task(6)
    def retrieve_analysis_results(self):
        """Retrieve detailed analysis results"""
        if self.document_ids:
            doc_id = random.choice(self.document_ids)
            
            with self.client.get(
                f"/api/v1/arbitration/results/{doc_id}",
                catch_response=True,
                name="get_results"
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    if "confidence_score" in result:
                        response.success()
                    else:
                        response.failure("Missing confidence score")
                else:
                    response.failure(f"Retrieval failed: {response.status_code}")
    
    @task(2)
    def search_similar_documents(self):
        """Search for similar documents"""
        query = random.choice([
            "binding arbitration clause",
            "class action waiver",
            "dispute resolution",
            "mandatory arbitration"
        ])
        
        with self.client.post(
            "/api/v1/search/similar",
            json={
                "query": query,
                "limit": 10,
                "filters": {"document_type": "terms_of_service"}
            },
            catch_response=True,
            name="search_similar"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")
    
    @task(4)
    def pattern_matching(self):
        """Test pattern matching endpoint"""
        text = DocumentGenerator.generate_test_document("small")
        
        with self.client.post(
            "/api/v1/arbitration/patterns",
            json={"text": text},
            catch_response=True,
            name="pattern_match"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "patterns" in result:
                    response.success()
                else:
                    response.failure("No patterns in response")
            else:
                response.failure(f"Pattern matching failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Periodic health check"""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class WebSocketUser(HttpUser):
    """User class for WebSocket testing"""
    
    def on_start(self):
        """Initialize WebSocket connection"""
        self.ws = None
        self.connect_websocket()
    
    def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            # Convert HTTP URL to WebSocket URL
            ws_url = self.host.replace("http://", "ws://").replace("https://", "wss://")
            ws_url = f"{ws_url}/ws"
            
            self.ws = websocket.WebSocket()
            self.ws.connect(ws_url)
            
            # Send initial handshake
            self.ws.send(json.dumps({
                "type": "connect",
                "client_id": f"locust_{self.locust_instance_id}"
            }))
            
            events.request_success.fire(
                request_type="WebSocket",
                name="ws_connect",
                response_time=0,
                response_length=0
            )
        except Exception as e:
            events.request_failure.fire(
                request_type="WebSocket",
                name="ws_connect",
                response_time=0,
                exception=e
            )
    
    @task
    def send_analysis_request(self):
        """Send analysis request over WebSocket"""
        if self.ws:
            try:
                start_time = time.time()
                
                message = {
                    "type": "analyze",
                    "data": {
                        "text": DocumentGenerator.generate_test_document("small"),
                        "request_id": f"req_{time.time()}"
                    }
                }
                
                self.ws.send(json.dumps(message))
                response = self.ws.recv()
                
                elapsed = (time.time() - start_time) * 1000
                
                if response:
                    events.request_success.fire(
                        request_type="WebSocket",
                        name="ws_analyze",
                        response_time=elapsed,
                        response_length=len(response)
                    )
                else:
                    raise Exception("Empty response")
                    
            except Exception as e:
                events.request_failure.fire(
                    request_type="WebSocket",
                    name="ws_analyze",
                    response_time=0,
                    exception=e
                )
    
    def on_stop(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()


class StandardUser(HttpUser):
    """Standard HTTP user for API testing"""
    
    tasks = [ArbitrationAnalysisTasks]
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session"""
        # Optionally authenticate the user
        pass


class PowerUser(HttpUser):
    """Power user with faster request rate"""
    
    tasks = [ArbitrationAnalysisTasks]
    wait_time = between(0.5, 1)  # Faster request rate
    
    def on_start(self):
        """Initialize power user session"""
        pass


class MobileUser(HttpUser):
    """Mobile user with different usage patterns"""
    
    wait_time = between(2, 5)  # Slower, more deliberate actions
    
    @task(10)
    def quick_analysis(self):
        """Quick analysis for mobile"""
        text = "This agreement contains a binding arbitration clause."
        
        with self.client.post(
            "/api/v1/arbitration/quick",
            json={"text": text},
            catch_response=True,
            name="mobile_quick_analysis"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Quick analysis failed: {response.status_code}")
    
    @task(5)
    def check_cached_result(self):
        """Check for cached results"""
        doc_hash = random.randint(1000, 9999)
        
        with self.client.get(
            f"/api/v1/cache/document/{doc_hash}",
            catch_response=True,
            name="mobile_cache_check"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Cache check failed: {response.status_code}")


class StressTestUser(HttpUser):
    """User for stress testing with aggressive patterns"""
    
    wait_time = between(0.1, 0.5)  # Very aggressive
    
    @task
    def rapid_fire_analysis(self):
        """Rapid fire analysis requests"""
        for _ in range(5):  # Burst of 5 requests
            text = f"Test document {random.randint(1, 1000)}"
            
            self.client.post(
                "/api/v1/arbitration/analyze",
                json={"text": text, "quick": True},
                name="stress_rapid_fire"
            )
            
            gevent.sleep(0.1)  # Small delay between burst requests


# Custom event handlers for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test metrics"""
    print("="*60)
    print("Starting RAG Legal Analysis Load Test")
    print(f"Target Host: {environment.host}")
    print(f"Total Users: {environment.runner.target_user_count if environment.runner else 'N/A'}")
    print("="*60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate final report"""
    print("\n" + "="*60)
    print("Load Test Complete - Summary Report")
    print("="*60)
    
    # Calculate statistics
    stats = environment.stats
    
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median Response Time: {stats.total.median_response_time:.2f}ms")
    print(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    
    # Generate detailed report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"load_test_report_{timestamp}.json"
    
    report_data = {
        "timestamp": timestamp,
        "total_requests": stats.total.num_requests,
        "total_failures": stats.total.num_failures,
        "failure_rate": stats.total.fail_ratio,
        "response_times": {
            "average": stats.total.avg_response_time,
            "median": stats.total.median_response_time,
            "p95": stats.total.get_response_time_percentile(0.95),
            "p99": stats.total.get_response_time_percentile(0.99),
            "min": stats.total.min_response_time,
            "max": stats.total.max_response_time
        },
        "rps": stats.total.current_rps,
        "endpoints": {}
    }
    
    # Add per-endpoint statistics
    for name, entry in stats.entries.items():
        if name != "Aggregated":
            report_data["endpoints"][name] = {
                "requests": entry.num_requests,
                "failures": entry.num_failures,
                "avg_response_time": entry.avg_response_time,
                "median_response_time": entry.median_response_time,
                "p95": entry.get_response_time_percentile(0.95),
                "p99": entry.get_response_time_percentile(0.99)
            }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")


# Configuration for different load test scenarios
class LoadTestScenarios:
    """Predefined load test scenarios"""
    
    @staticmethod
    def normal_load():
        """Normal daily load pattern"""
        return {
            "users": 50,
            "spawn_rate": 2,
            "duration": "5m",
            "user_classes": [StandardUser, MobileUser]
        }
    
    @staticmethod
    def peak_load():
        """Peak hour load pattern"""
        return {
            "users": 200,
            "spawn_rate": 10,
            "duration": "15m",
            "user_classes": [StandardUser, PowerUser, MobileUser]
        }
    
    @staticmethod
    def stress_test():
        """Stress test to find breaking point"""
        return {
            "users": 500,
            "spawn_rate": 20,
            "duration": "10m",
            "user_classes": [StressTestUser, StandardUser]
        }
    
    @staticmethod
    def endurance_test():
        """Long-running endurance test"""
        return {
            "users": 100,
            "spawn_rate": 5,
            "duration": "1h",
            "user_classes": [StandardUser, MobileUser]
        }
    
    @staticmethod
    def spike_test():
        """Sudden spike in traffic"""
        return {
            "users": 300,
            "spawn_rate": 50,
            "duration": "5m",
            "user_classes": [StandardUser, StressTestUser]
        }


if __name__ == "__main__":
    # Run with: locust -f locustfile.py --host http://localhost:8000
    print("RAG Legal Analysis Load Testing Script")
    print("Run with: locust -f locustfile.py --host http://your-api-host")
    print("\nAvailable test scenarios:")
    print("  - Normal Load: 50 users, moderate activity")
    print("  - Peak Load: 200 users, high activity")
    print("  - Stress Test: 500 users, find breaking point")
    print("  - Endurance Test: 100 users for 1 hour")
    print("  - Spike Test: 300 users sudden spike")