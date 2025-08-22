"""
Locust load testing script for arbitration detection API.

This script defines load testing scenarios for:
- Document upload endpoints
- Arbitration analysis endpoints
- Batch processing
- Concurrent user simulation
"""

import io
import json
import random
import time
from locust import HttpUser, task, between, SequentialTaskSet
from faker import Faker

fake = Faker()


class DocumentUploadTaskSet(SequentialTaskSet):
    """Sequential task set for document upload and analysis workflow."""
    
    def on_start(self):
        """Set up user session."""
        self.auth_token = None
        self.document_ids = []
        self.login()
    
    def login(self):
        """Authenticate user."""
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "testpassword123"
        })
        
        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
    
    @task
    def upload_document(self):
        """Upload a test document."""
        # Create a mock PDF file
        pdf_content = self.generate_mock_pdf_content()
        files = {
            "file": ("test_contract.pdf", io.BytesIO(pdf_content.encode()), "application/pdf")
        }
        
        with self.client.post(
            "/api/v1/documents/upload",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.document_ids.append(data["document_id"])
                response.success()
            else:
                response.failure(f"Upload failed: {response.status_code}")
    
    @task
    def analyze_document(self):
        """Analyze an uploaded document."""
        if not self.document_ids:
            return
        
        document_id = random.choice(self.document_ids)
        
        with self.client.post(
            "/api/v1/arbitration/detect-by-id",
            json={"document_id": document_id},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analysis failed: {response.status_code}")
    
    @task
    def batch_analyze(self):
        """Perform batch analysis on multiple documents."""
        if len(self.document_ids) < 2:
            return
        
        batch_documents = random.sample(self.document_ids, min(3, len(self.document_ids)))
        
        with self.client.post(
            "/api/v1/arbitration/detect-batch",
            json={
                "documents": [{"id": doc_id, "text": self.generate_mock_document_text()} 
                             for doc_id in batch_documents]
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Batch analysis failed: {response.status_code}")
    
    def generate_mock_pdf_content(self):
        """Generate mock PDF content with potential arbitration clauses."""
        templates = [
            f"""
            CONTRACT AGREEMENT
            
            This agreement is entered into between {fake.company()} and {fake.company()}.
            
            Section 1: Services
            The contractor shall provide consulting services as outlined below.
            
            Section 2: Payment
            Payment terms are net 30 days from invoice date.
            
            Section 15: Dispute Resolution
            Any dispute arising under this agreement shall be resolved through binding 
            arbitration administered by the American Arbitration Association.
            
            Signed this day: {fake.date()}
            """,
            f"""
            EMPLOYMENT AGREEMENT
            
            Employee: {fake.name()}
            Position: {fake.job()}
            
            Terms and conditions of employment are as follows:
            
            1. Compensation: ${fake.random_int(50000, 150000)} annually
            2. Benefits: Health insurance, dental, vision
            3. Disputes shall be resolved in state court
            
            Date: {fake.date()}
            """,
            f"""
            TERMS OF SERVICE
            
            Welcome to {fake.company()}'s services.
            
            By using our service, you agree to the following terms:
            
            1. Service description
            2. User obligations  
            3. Privacy policy
            4. Any disputes may be resolved through arbitration at the option of either party
            
            Last updated: {fake.date()}
            """
        ]
        return random.choice(templates)
    
    def generate_mock_document_text(self):
        """Generate mock document text for analysis."""
        return fake.text(max_nb_chars=1000)


class ArbitrationAPIUser(HttpUser):
    """Simulated user for arbitration detection API."""
    
    tasks = [DocumentUploadTaskSet]
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    weight = 3


class HighVolumeUser(HttpUser):
    """High-volume user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Minimal wait time for stress testing
    weight = 1
    
    def on_start(self):
        """Initialize high-volume user."""
        self.auth_token = None
        self.login()
    
    def login(self):
        """Authenticate user."""
        response = self.client.post("/api/v1/auth/login", json={
            "email": "stress_test@example.com",
            "password": "testpassword123"
        })
        
        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")
            self.client.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
    
    @task(3)
    def quick_text_analysis(self):
        """Perform quick text-based arbitration analysis."""
        test_texts = [
            "Any disputes shall be resolved through binding arbitration.",
            "Disputes will be handled in state court.",
            "This agreement may be terminated by either party.",
            "Arbitration clause: All claims must go to arbitration."
        ]
        
        with self.client.post(
            "/api/v1/arbitration/detect",
            json={"text": random.choice(test_texts)},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Quick analysis failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Perform health check."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_user_stats(self):
        """Get user statistics."""
        with self.client.get("/api/v1/user/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stats failed: {response.status_code}")


class ReadOnlyUser(HttpUser):
    """Read-only user for testing read endpoints."""
    
    wait_time = between(0.5, 2)
    weight = 2
    
    @task(3)
    def list_documents(self):
        """List user documents."""
        with self.client.get("/api/v1/documents/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Document list failed: {response.status_code}")
    
    @task(2)
    def get_recent_analyses(self):
        """Get recent analyses."""
        with self.client.get("/api/v1/analyses/recent", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recent analyses failed: {response.status_code}")
    
    @task(1)
    def api_info(self):
        """Get API information."""
        with self.client.get("/api/v1/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"API info failed: {response.status_code}")


# Custom load testing scenarios for different use cases
class PeakLoadTest(HttpUser):
    """Simulate peak load conditions."""
    
    wait_time = between(0, 1)
    weight = 1
    
    def on_start(self):
        """Initialize peak load test."""
        self.start_time = time.time()
    
    @task
    def peak_analysis_load(self):
        """Simulate peak analysis load."""
        # Generate large document text
        large_text = fake.text(max_nb_chars=5000) + """
        
        ARBITRATION AGREEMENT
        
        This comprehensive arbitration agreement requires all disputes to be 
        resolved through binding arbitration administered by the International 
        Chamber of Commerce under its Rules of Arbitration.
        """
        
        with self.client.post(
            "/api/v1/arbitration/detect",
            json={
                "text": large_text,
                "options": {
                    "detailed_analysis": True,
                    "use_rag": True
                }
            },
            catch_response=True
        ) as response:
            # Track response time
            response_time = response.elapsed.total_seconds()
            
            if response.status_code == 200:
                if response_time > 10:  # Fail if response time > 10 seconds
                    response.failure(f"Response too slow: {response_time}s")
                else:
                    response.success()
            else:
                response.failure(f"Peak load analysis failed: {response.status_code}")
    
    def on_stop(self):
        """Clean up after peak load test."""
        duration = time.time() - self.start_time
        print(f"Peak load test completed in {duration:.2f} seconds")


# Event hooks for custom metrics
from locust import events

@events.request.add_listener
def request_handler(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Custom request handler for additional metrics."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response_time > 5000:  # Log slow requests (>5 seconds)
        print(f"Slow request detected: {name} - {response_time}ms")


@events.test_start.add_listener
def test_start_handler(environment, **kwargs):
    """Handler for test start event."""
    print("Load test starting...")
    print(f"Target host: {environment.host}")
    print(f"Number of users: {environment.runner.target_user_count}")


@events.test_stop.add_listener
def test_stop_handler(environment, **kwargs):
    """Handler for test stop event."""
    print("Load test completed.")
    
    # Calculate and log final statistics
    stats = environment.runner.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")
    
    # Fail the test if error rate is too high
    if stats.total.num_failures > 0:
        failure_rate = stats.total.num_failures / stats.total.num_requests
        if failure_rate > 0.05:  # Fail if >5% error rate
            print(f"❌ Test failed: Error rate too high ({failure_rate:.1%})")
            environment.process_exit_code = 1
        else:
            print(f"✅ Test passed: Error rate acceptable ({failure_rate:.1%})")


# Performance thresholds
class PerformanceThresholds:
    """Define performance thresholds for different endpoints."""
    
    THRESHOLDS = {
        "/api/v1/arbitration/detect": {"max_response_time": 5000, "max_failure_rate": 0.01},
        "/api/v1/documents/upload": {"max_response_time": 10000, "max_failure_rate": 0.02},
        "/api/v1/arbitration/detect-batch": {"max_response_time": 15000, "max_failure_rate": 0.03},
    }
    
    @classmethod
    def check_thresholds(cls, stats):
        """Check if performance thresholds are met."""
        failures = []
        
        for endpoint, thresholds in cls.THRESHOLDS.items():
            if endpoint in stats.entries:
                entry = stats.entries[endpoint]
                
                # Check response time
                if entry.avg_response_time > thresholds["max_response_time"]:
                    failures.append(f"{endpoint}: Average response time {entry.avg_response_time}ms > {thresholds['max_response_time']}ms")
                
                # Check failure rate
                if entry.num_requests > 0:
                    failure_rate = entry.num_failures / entry.num_requests
                    if failure_rate > thresholds["max_failure_rate"]:
                        failures.append(f"{endpoint}: Failure rate {failure_rate:.1%} > {thresholds['max_failure_rate']:.1%}")
        
        return failures