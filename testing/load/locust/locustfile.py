"""
Locust Load Testing Configuration
Simulates 10,000+ concurrent users for enterprise-scale testing
"""

import random
import json
import time
from locust import HttpUser, task, between, events, LoadTestShape
from locust.runners import MasterRunner, WorkerRunner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnterpriseUser(HttpUser):
    """Simulates enterprise application user behavior"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when user starts - login"""
        self.login()
        self.user_id = random.randint(1, 10000)
        self.session_data = {}
    
    def login(self):
        """Authenticate user"""
        response = self.client.post("/api/auth/login", json={
            "username": f"user{random.randint(1, 10000)}",
            "password": "password123"
        })
        
        if response.status_code == 200:
            data = response.json()
            self.client.headers.update({
                "Authorization": f"Bearer {data.get('token', '')}"
            })
            self.session_data['user'] = data.get('user', {})
    
    @task(30)
    def view_dashboard(self):
        """View main dashboard - high frequency task"""
        with self.client.get(
            "/api/dashboard",
            catch_response=True,
            name="Dashboard"
        ) as response:
            if response.status_code == 200:
                response.success()
                # Parse response time
                if response.elapsed.total_seconds() > 2:
                    response.failure(f"Dashboard too slow: {response.elapsed.total_seconds()}s")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(20)
    def search_data(self):
        """Search functionality"""
        search_terms = ["product", "user", "order", "report", "analytics"]
        query = random.choice(search_terms)
        
        self.client.get(
            f"/api/search?q={query}&limit=50",
            name="Search"
        )
    
    @task(15)
    def view_reports(self):
        """View analytics reports"""
        report_types = ["sales", "performance", "users", "inventory"]
        report_type = random.choice(report_types)
        
        self.client.get(
            f"/api/reports/{report_type}",
            name=f"Report:{report_type}"
        )
    
    @task(10)
    def crud_operations(self):
        """CRUD operations on entities"""
        operations = [
            self.create_entity,
            self.read_entity,
            self.update_entity,
            self.delete_entity
        ]
        
        operation = random.choice(operations)
        operation()
    
    def create_entity(self):
        """Create new entity"""
        entity_data = {
            "name": f"Entity_{random.randint(1, 10000)}",
            "type": random.choice(["A", "B", "C"]),
            "value": random.uniform(100, 10000),
            "metadata": {
                "created_by": self.user_id,
                "timestamp": time.time()
            }
        }
        
        response = self.client.post(
            "/api/entities",
            json=entity_data,
            name="Create Entity"
        )
        
        if response.status_code == 201:
            self.session_data['last_entity_id'] = response.json().get('id')
    
    def read_entity(self):
        """Read entity details"""
        entity_id = self.session_data.get('last_entity_id', random.randint(1, 1000))
        
        self.client.get(
            f"/api/entities/{entity_id}",
            name="Read Entity"
        )
    
    def update_entity(self):
        """Update entity"""
        entity_id = self.session_data.get('last_entity_id', random.randint(1, 1000))
        
        update_data = {
            "value": random.uniform(100, 10000),
            "updated_at": time.time()
        }
        
        self.client.put(
            f"/api/entities/{entity_id}",
            json=update_data,
            name="Update Entity"
        )
    
    def delete_entity(self):
        """Delete entity"""
        entity_id = random.randint(1, 1000)
        
        self.client.delete(
            f"/api/entities/{entity_id}",
            name="Delete Entity"
        )
    
    @task(8)
    def file_operations(self):
        """File upload/download operations"""
        if random.random() > 0.5:
            self.upload_file()
        else:
            self.download_file()
    
    def upload_file(self):
        """Upload file"""
        files = {
            'file': ('test.txt', b'Test file content ' * 100, 'text/plain')
        }
        
        self.client.post(
            "/api/files/upload",
            files=files,
            name="Upload File"
        )
    
    def download_file(self):
        """Download file"""
        file_id = random.randint(1, 100)
        
        self.client.get(
            f"/api/files/{file_id}/download",
            name="Download File"
        )
    
    @task(5)
    def complex_aggregation(self):
        """Complex data aggregation query"""
        params = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "group_by": random.choice(["day", "week", "month"]),
            "metrics": ["revenue", "users", "transactions"]
        }
        
        self.client.post(
            "/api/analytics/aggregate",
            json=params,
            name="Complex Aggregation"
        )
    
    @task(3)
    def batch_operations(self):
        """Batch processing operations"""
        batch_size = random.randint(10, 100)
        batch_data = [
            {
                "id": i,
                "operation": random.choice(["create", "update", "delete"]),
                "data": {"value": random.uniform(1, 1000)}
            }
            for i in range(batch_size)
        ]
        
        self.client.post(
            "/api/batch",
            json=batch_data,
            name=f"Batch Operation ({batch_size} items)"
        )
    
    @task(2)
    def websocket_simulation(self):
        """Simulate WebSocket-like real-time operations"""
        # Simulate long-polling for real-time updates
        self.client.get(
            "/api/realtime/poll",
            timeout=30,
            name="Real-time Poll"
        )
    
    def on_stop(self):
        """Called when user stops - logout"""
        self.client.post("/api/auth/logout")

class PowerUser(EnterpriseUser):
    """Simulates power users with more intensive operations"""
    
    wait_time = between(0.5, 1.5)  # Faster interactions
    
    @task(40)
    def intensive_operations(self):
        """More resource-intensive operations"""
        self.complex_aggregation()
        self.batch_operations()

class MobileUser(HttpUser):
    """Simulates mobile app users with different patterns"""
    
    wait_time = between(2, 5)  # Slower interactions on mobile
    
    def on_start(self):
        """Mobile app initialization"""
        self.client.headers.update({
            "User-Agent": "MobileApp/1.0",
            "X-Platform": "iOS"
        })
        self.login()
    
    def login(self):
        """Mobile authentication"""
        response = self.client.post("/api/mobile/auth", json={
            "device_id": f"device_{random.randint(1, 5000)}",
            "pin": "1234"
        })
        
        if response.status_code == 200:
            data = response.json()
            self.client.headers["Authorization"] = f"Bearer {data.get('token', '')}"
    
    @task(50)
    def mobile_dashboard(self):
        """Mobile optimized dashboard"""
        self.client.get("/api/mobile/dashboard", name="Mobile Dashboard")
    
    @task(30)
    def push_notification_check(self):
        """Check for push notifications"""
        self.client.get("/api/mobile/notifications", name="Check Notifications")
    
    @task(20)
    def sync_offline_data(self):
        """Sync offline data"""
        sync_data = {
            "last_sync": time.time() - 3600,
            "changes": [
                {"id": i, "data": {"field": "value"}} 
                for i in range(random.randint(1, 10))
            ]
        }
        
        self.client.post(
            "/api/mobile/sync",
            json=sync_data,
            name="Sync Offline Data"
        )

class SteppedLoadShape(LoadTestShape):
    """
    Custom load shape for gradual ramp-up to 10,000 users
    Implements a stepped load increase pattern
    """
    
    step_time = 60  # Time for each step in seconds
    step_users = 500  # Users to add each step
    max_users = 10000  # Maximum number of users
    time_limit = 1200  # Total test time in seconds (20 minutes)
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None  # Stop the test
        
        current_step = run_time // self.step_time
        target_users = min(self.max_users, (current_step + 1) * self.step_users)
        
        # Spawn rate increases with load
        spawn_rate = max(10, target_users // 10)
        
        return (target_users, spawn_rate)

class SpikeLoadShape(LoadTestShape):
    """
    Spike test - sudden increase in load
    """
    
    stages = [
        {"duration": 60, "users": 100, "spawn_rate": 10},
        {"duration": 30, "users": 5000, "spawn_rate": 100},  # Spike
        {"duration": 120, "users": 5000, "spawn_rate": 100},  # Sustained
        {"duration": 30, "users": 100, "spawn_rate": 50},  # Recovery
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
            run_time -= stage["duration"]
        
        return None

# Event handlers for monitoring
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    """Log request metrics"""
    if response_time > 2000:  # Log slow requests (>2s)
        logger.warning(f"Slow request: {name} took {response_time}ms")
    
    # Track percentiles
    if hasattr(on_request, 'response_times'):
        on_request.response_times.append(response_time)
    else:
        on_request.response_times = [response_time]

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test metrics"""
    logger.info("Load test starting...")
    logger.info(f"Target host: {environment.host}")
    
    # Initialize custom metrics
    environment.stats.custom_metrics = {
        "errors_by_type": {},
        "slow_requests": 0,
        "test_start_time": time.time()
    }

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report"""
    logger.info("Load test completed!")
    
    # Calculate percentiles
    if hasattr(on_request, 'response_times'):
        response_times = sorted(on_request.response_times)
        if response_times:
            p50 = response_times[int(len(response_times) * 0.5)]
            p95 = response_times[int(len(response_times) * 0.95)]
            p99 = response_times[int(len(response_times) * 0.99)]
            
            logger.info(f"Response time percentiles - P50: {p50}ms, P95: {p95}ms, P99: {p99}ms")
    
    # Generate summary report
    total_requests = environment.stats.total.num_requests
    total_failures = environment.stats.total.num_failures
    
    if total_requests > 0:
        error_rate = (total_failures / total_requests) * 100
        logger.info(f"Error rate: {error_rate:.2f}%")
    
    # Save detailed report
    report = {
        "total_requests": total_requests,
        "total_failures": total_failures,
        "response_time_percentiles": {
            "p50": p50 if 'p50' in locals() else 0,
            "p95": p95 if 'p95' in locals() else 0,
            "p99": p99 if 'p99' in locals() else 0
        },
        "test_duration": time.time() - environment.stats.custom_metrics.get("test_start_time", 0)
    }
    
    with open("locust_report.json", "w") as f:
        json.dump(report, f, indent=2)

# Distributed testing support
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize distributed testing"""
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running as master node")
    elif isinstance(environment.runner, WorkerRunner):
        logger.info("Running as worker node")

# Custom failure handler
@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    """Handle request failures"""
    error_type = type(exception).__name__
    
    # Track errors by type
    if hasattr(events, 'custom_metrics'):
        if error_type not in events.custom_metrics['errors_by_type']:
            events.custom_metrics['errors_by_type'][error_type] = 0
        events.custom_metrics['errors_by_type'][error_type] += 1

if __name__ == "__main__":
    # Can be run with: locust -f locustfile.py --host=http://localhost:8000
    pass