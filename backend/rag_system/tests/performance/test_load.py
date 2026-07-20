"""Load testing script using locust for the RAG system API."""
import os
import time
import random
import json
from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import gevent
import pandas as pd
import numpy as np
from datetime import datetime

setup_logging("INFO", None)

# Sample test data
SAMPLE_TEXTS = [
    """This Agreement contains a binding arbitration provision which may be enforced by the parties. 
    All disputes arising out of or relating to this Agreement shall be resolved through binding arbitration 
    in accordance with the Commercial Arbitration Rules of the American Arbitration Association.""",
    
    """The parties agree that any dispute arising from this contract shall be settled through 
    mandatory arbitration proceedings. The arbitrator's decision shall be final and binding.""",
    
    """In the event of any dispute, the parties shall first attempt to resolve the matter through 
    good faith negotiations before proceeding to court.""",
    
    """All claims must be brought through individual arbitration. Class action lawsuits are waived. 
    The arbitration shall be conducted in New York under AAA rules.""",
    
    """This is a standard service agreement without any arbitration requirements. 
    Disputes shall be resolved in state court."""
]

SAMPLE_PDFS = [
    "/Users/rahulmehta/Desktop/Test/backend/rag_system/data/test/sample_contract_1.pdf",
    "/Users/rahulmehta/Desktop/Test/backend/rag_system/data/test/sample_contract_2.pdf",
    "/Users/rahulmehta/Desktop/Test/backend/rag_system/data/test/sample_tos.pdf"
]

class RAGSystemUser(HttpUser):
    """Simulated user for load testing the RAG system."""
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session."""
        self.client.verify = False  # Disable SSL verification for testing
        
    @task(3)
    def detect_text(self):
        """Test text detection endpoint."""
        text = random.choice(SAMPLE_TEXTS)
        
        with self.client.post(
            "/detect/text",
            json={
                "text": text,
                "threshold": random.uniform(0.5, 0.9),
                "explain": random.choice([True, False]),
                "compare": random.choice([True, False])
            },
            catch_response=True,
            name="Text Detection"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                # Validate response structure
                if "detected" in result and "confidence" in result:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2)
    def detect_file(self):
        """Test file detection endpoint."""
        # For testing, we'll simulate file upload with sample PDFs
        if SAMPLE_PDFS and os.path.exists(SAMPLE_PDFS[0]):
            pdf_path = random.choice(SAMPLE_PDFS)
            
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                params = {
                    'threshold': random.uniform(0.5, 0.9),
                    'explain': random.choice(['true', 'false']),
                    'compare': random.choice(['true', 'false'])
                }
                
                with self.client.post(
                    "/detect",
                    files=files,
                    params=params,
                    catch_response=True,
                    name="PDF Detection"
                ) as response:
                    if response.status_code == 200:
                        response.success()
                    else:
                        response.failure(f"Status {response.status_code}")
    
    @task(1)
    def compare_clause(self):
        """Test clause comparison endpoint."""
        clause = random.choice(SAMPLE_TEXTS[:3])  # Use arbitration clauses
        
        with self.client.post(
            "/compare",
            json={
                "clause_text": clause,
                "top_k": random.randint(5, 20)
            },
            catch_response=True,
            name="Clause Comparison"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def database_stats(self):
        """Test database stats endpoint."""
        with self.client.get(
            "/database/stats",
            catch_response=True,
            name="Database Stats"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service unhealthy")
            else:
                response.failure(f"Status {response.status_code}")

class PerformanceMetrics:
    """Collect and analyze performance metrics."""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        
    def start_collection(self):
        """Start collecting metrics."""
        self.start_time = time.time()
        
    def collect_stats(self, stats):
        """Collect current statistics."""
        current_time = time.time() - self.start_time
        
        for name, entry in stats.entries.items():
            self.metrics.append({
                'time': current_time,
                'name': name,
                'num_requests': entry.num_requests,
                'num_failures': entry.num_failures,
                'median_response_time': entry.median_response_time,
                'average_response_time': entry.average_response_time,
                'min_response_time': entry.min_response_time or 0,
                'max_response_time': entry.max_response_time or 0,
                'current_rps': entry.current_rps,
                'current_fail_per_sec': entry.current_fail_per_sec,
                'p95': entry.get_response_time_percentile(0.95) or 0,
                'p99': entry.get_response_time_percentile(0.99) or 0
            })
    
    def generate_report(self, output_file="performance_report.json"):
        """Generate performance report."""
        if not self.metrics:
            print("No metrics collected")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.metrics)
        
        # Calculate summary statistics
        summary = {
            'test_duration': time.time() - self.start_time,
            'total_requests': df.groupby('name')['num_requests'].max().sum(),
            'total_failures': df.groupby('name')['num_failures'].max().sum(),
            'endpoints': {}
        }
        
        # Per-endpoint statistics
        for endpoint in df['name'].unique():
            endpoint_data = df[df['name'] == endpoint]
            latest = endpoint_data.iloc[-1] if not endpoint_data.empty else None
            
            if latest is not None:
                summary['endpoints'][endpoint] = {
                    'total_requests': int(latest['num_requests']),
                    'total_failures': int(latest['num_failures']),
                    'failure_rate': (latest['num_failures'] / latest['num_requests'] * 100) if latest['num_requests'] > 0 else 0,
                    'avg_response_time': float(latest['average_response_time']),
                    'median_response_time': float(latest['median_response_time']),
                    'min_response_time': float(latest['min_response_time']),
                    'max_response_time': float(latest['max_response_time']),
                    'p95_response_time': float(latest['p95']),
                    'p99_response_time': float(latest['p99']),
                    'avg_rps': float(endpoint_data['current_rps'].mean())
                }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def run_load_test(
    host="http://localhost:8000",
    users=10,
    spawn_rate=2,
    run_time=60,
    output_file="load_test_results.json"
):
    """
    Run load test with specified parameters.
    
    Args:
        host: Target host URL
        users: Number of concurrent users to simulate
        spawn_rate: Users spawned per second
        run_time: Test duration in seconds
        output_file: Output file for results
    """
    # Setup Environment
    env = Environment(user_classes=[RAGSystemUser], host=host)
    
    # Initialize metrics collector
    metrics = PerformanceMetrics()
    
    # Start test
    print(f"\n{'='*60}")
    print(f"Starting load test:")
    print(f"  Host: {host}")
    print(f"  Users: {users}")
    print(f"  Spawn rate: {spawn_rate}/second")
    print(f"  Duration: {run_time} seconds")
    print(f"{'='*60}\n")
    
    # Spawn users
    env.runner.start(users, spawn_rate=spawn_rate)
    
    # Start metrics collection
    metrics.start_collection()
    
    # Collect metrics every 2 seconds
    gevent.spawn(stats_printer(env.stats))
    
    for i in range(run_time // 2):
        gevent.sleep(2)
        metrics.collect_stats(env.stats)
    
    # Stop test
    env.runner.quit()
    
    # Generate report
    report = metrics.generate_report(output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Load Test Results:")
    print(f"{'='*60}")
    print(f"Total Requests: {report['total_requests']}")
    print(f"Total Failures: {report['total_failures']}")
    print(f"Failure Rate: {(report['total_failures'] / report['total_requests'] * 100):.2f}%")
    print(f"\nPer-Endpoint Statistics:")
    
    for endpoint, stats in report['endpoints'].items():
        print(f"\n{endpoint}:")
        print(f"  Requests: {stats['total_requests']}")
        print(f"  Failures: {stats['total_failures']} ({stats['failure_rate']:.2f}%)")
        print(f"  Avg Response: {stats['avg_response_time']:.2f}ms")
        print(f"  P95 Response: {stats['p95_response_time']:.2f}ms")
        print(f"  P99 Response: {stats['p99_response_time']:.2f}ms")
        print(f"  Avg RPS: {stats['avg_rps']:.2f}")
    
    print(f"\nDetailed report saved to: {output_file}")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test the RAG system")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=2, help="Users spawned per second")
    parser.add_argument("--run-time", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--output", default="load_test_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    run_load_test(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
        output_file=args.output
    )