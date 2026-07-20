"""API endpoint performance testing for the RAG system."""
import asyncio
import aiohttp
import time
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import tempfile
from pathlib import Path

class APIPerformanceTester:
    """Comprehensive API performance testing for RAG system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API tester."""
        self.base_url = base_url
        self.session = None
        self.results = {
            'endpoint_latency': [],
            'throughput': [],
            'error_rates': [],
            'response_sizes': [],
            'concurrent_performance': []
        }
        
        # Sample test data
        self.test_texts = [
            """This Agreement contains a binding arbitration provision which may be enforced by the parties.
            All disputes shall be resolved through binding arbitration in accordance with AAA rules.""",
            
            """The parties agree to submit any dispute to mandatory arbitration proceedings.
            The arbitrator's decision shall be final and binding on both parties.""",
            
            """Any controversy or claim arising out of this contract shall be settled by arbitration
            administered by the American Arbitration Association.""",
            
            """Disputes will be resolved through negotiation and if necessary, litigation in state court.""",
            
            """This is a standard service agreement without arbitration requirements."""
        ]
        
        # Create sample test files
        self.test_files = self._create_test_files()
    
    def _create_test_files(self) -> List[str]:
        """Create sample PDF test files."""
        files = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            for i, text in enumerate(self.test_texts[:3]):
                pdf_path = os.path.join(temp_dir, f"test_contract_{i}.pdf")
                c = canvas.Canvas(pdf_path, pagesize=letter)
                
                # Add title
                c.drawString(100, 750, f"Test Contract {i+1}")
                
                # Add content
                y_position = 700
                lines = text.split('\n')
                for line in lines:
                    c.drawString(100, y_position, line.strip())
                    y_position -= 20
                
                c.save()
                files.append(pdf_path)
        except ImportError:
            print("Warning: reportlab not installed, using text files instead")
            for i, text in enumerate(self.test_texts[:3]):
                txt_path = os.path.join(temp_dir, f"test_contract_{i}.txt")
                with open(txt_path, 'w') as f:
                    f.write(text)
                files.append(txt_path)
        
        return files
    
    async def test_endpoint_latency(self):
        """Test latency for all API endpoints."""
        print("\n" + "="*60)
        print("TESTING ENDPOINT LATENCY")
        print("="*60)
        
        async with aiohttp.ClientSession() as session:
            endpoints = [
                ('GET', '/health', None, None),
                ('GET', '/database/stats', None, None),
                ('POST', '/detect/text', {'json': {
                    'text': random.choice(self.test_texts),
                    'threshold': 0.7,
                    'explain': True,
                    'compare': True
                }}, None),
                ('POST', '/compare', {'json': {
                    'clause_text': random.choice(self.test_texts[:3]),
                    'top_k': 10
                }}, None)
            ]
            
            for method, endpoint, data, files in endpoints:
                print(f"\nTesting {method} {endpoint}...")
                
                latencies = []
                response_sizes = []
                errors = 0
                
                for i in range(20):  # 20 requests per endpoint
                    try:
                        start = time.perf_counter()
                        
                        if method == 'GET':
                            async with session.get(f"{self.base_url}{endpoint}") as response:
                                content = await response.text()
                                status = response.status
                        else:  # POST
                            if data and 'json' in data:
                                async with session.post(
                                    f"{self.base_url}{endpoint}",
                                    json=data['json']
                                ) as response:
                                    content = await response.text()
                                    status = response.status
                        
                        latency = time.perf_counter() - start
                        latencies.append(latency)
                        response_sizes.append(len(content))
                        
                        if status != 200:
                            errors += 1
                            
                    except Exception as e:
                        print(f"  Error: {e}")
                        errors += 1
                
                if latencies:
                    # Calculate statistics
                    result = {
                        'endpoint': endpoint,
                        'method': method,
                        'requests': len(latencies),
                        'errors': errors,
                        'error_rate': errors / 20 * 100,
                        'avg_latency_ms': np.mean(latencies) * 1000,
                        'median_latency_ms': np.median(latencies) * 1000,
                        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                        'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                        'min_latency_ms': np.min(latencies) * 1000,
                        'max_latency_ms': np.max(latencies) * 1000,
                        'avg_response_size_bytes': np.mean(response_sizes)
                    }
                    
                    self.results['endpoint_latency'].append(result)
                    
                    print(f"  Avg latency: {result['avg_latency_ms']:.2f}ms")
                    print(f"  P95 latency: {result['p95_latency_ms']:.2f}ms")
                    print(f"  P99 latency: {result['p99_latency_ms']:.2f}ms")
                    print(f"  Error rate: {result['error_rate']:.1f}%")
    
    async def test_file_upload_performance(self):
        """Test file upload endpoint performance."""
        print("\n" + "="*60)
        print("TESTING FILE UPLOAD PERFORMANCE")
        print("="*60)
        
        if not self.test_files:
            print("No test files available, skipping file upload tests")
            return
        
        async with aiohttp.ClientSession() as session:
            for file_path in self.test_files:
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                
                print(f"\nTesting upload of {file_name} ({file_size/1024:.1f}KB)...")
                
                upload_times = []
                process_times = []
                errors = 0
                
                for i in range(5):  # 5 uploads per file
                    try:
                        # Prepare file upload
                        with open(file_path, 'rb') as f:
                            data = aiohttp.FormData()
                            data.add_field('file',
                                         f,
                                         filename=file_name,
                                         content_type='application/pdf')
                            
                            # Measure upload time
                            start = time.perf_counter()
                            
                            async with session.post(
                                f"{self.base_url}/detect",
                                data=data,
                                params={'threshold': 0.7, 'explain': 'true', 'compare': 'true'}
                            ) as response:
                                result = await response.json()
                                status = response.status
                            
                            total_time = time.perf_counter() - start
                            
                            if status == 200:
                                upload_times.append(total_time)
                            else:
                                errors += 1
                                
                    except Exception as e:
                        print(f"  Error: {e}")
                        errors += 1
                
                if upload_times:
                    print(f"  Avg time: {np.mean(upload_times)*1000:.2f}ms")
                    print(f"  P95 time: {np.percentile(upload_times, 95)*1000:.2f}ms")
                    print(f"  Throughput: {file_size/np.mean(upload_times)/1024:.1f}KB/s")
                    print(f"  Error rate: {errors/5*100:.1f}%")
    
    async def test_concurrent_requests(self, num_concurrent: int = 10):
        """Test API performance under concurrent load."""
        print("\n" + "="*60)
        print(f"TESTING CONCURRENT REQUESTS ({num_concurrent} concurrent)")
        print("="*60)
        
        async def make_request(session, endpoint_config):
            """Make a single request."""
            method, endpoint, data = endpoint_config
            
            try:
                start = time.perf_counter()
                
                if method == 'POST':
                    async with session.post(
                        f"{self.base_url}{endpoint}",
                        json=data
                    ) as response:
                        result = await response.json()
                        status = response.status
                else:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        result = await response.json()
                        status = response.status
                
                latency = time.perf_counter() - start
                
                return {
                    'endpoint': endpoint,
                    'status': status,
                    'latency': latency,
                    'success': status == 200
                }
                
            except Exception as e:
                return {
                    'endpoint': endpoint,
                    'error': str(e),
                    'success': False
                }
        
        # Prepare requests
        requests = []
        for _ in range(num_concurrent):
            # Mix of different endpoints
            choice = random.random()
            if choice < 0.5:
                # Text detection
                requests.append(('POST', '/detect/text', {
                    'text': random.choice(self.test_texts),
                    'threshold': random.uniform(0.5, 0.9),
                    'explain': random.choice([True, False]),
                    'compare': random.choice([True, False])
                }))
            elif choice < 0.8:
                # Comparison
                requests.append(('POST', '/compare', {
                    'clause_text': random.choice(self.test_texts[:3]),
                    'top_k': random.randint(5, 20)
                }))
            else:
                # Health check
                requests.append(('GET', '/health', None))
        
        # Execute concurrent requests
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            tasks = [make_request(session, req) for req in requests]
            results = await asyncio.gather(*tasks)
            
            total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r.get('success', False))
        latencies = [r.get('latency', 0) for r in results if 'latency' in r]
        
        concurrent_result = {
            'concurrent_requests': num_concurrent,
            'total_time': total_time,
            'successful_requests': successful,
            'failed_requests': num_concurrent - successful,
            'success_rate': successful / num_concurrent * 100,
            'throughput_rps': num_concurrent / total_time,
            'avg_latency_ms': np.mean(latencies) * 1000 if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000 if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000 if latencies else 0
        }
        
        self.results['concurrent_performance'].append(concurrent_result)
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Success rate: {concurrent_result['success_rate']:.1f}%")
        print(f"  Throughput: {concurrent_result['throughput_rps']:.1f} req/s")
        print(f"  Avg latency: {concurrent_result['avg_latency_ms']:.2f}ms")
        print(f"  P95 latency: {concurrent_result['p95_latency_ms']:.2f}ms")
    
    async def test_throughput_limits(self):
        """Test maximum throughput capacity."""
        print("\n" + "="*60)
        print("TESTING THROUGHPUT LIMITS")
        print("="*60)
        
        concurrent_levels = [1, 5, 10, 20, 50]
        
        for level in concurrent_levels:
            print(f"\nTesting with {level} concurrent requests...")
            await self.test_concurrent_requests(level)
            
            # Small delay between tests
            await asyncio.sleep(1)
    
    async def test_response_time_under_load(self, duration_seconds: int = 30):
        """Test response time stability under sustained load."""
        print("\n" + "="*60)
        print(f"TESTING RESPONSE TIME UNDER LOAD ({duration_seconds}s)")
        print("="*60)
        
        async def worker(session, worker_id):
            """Worker that continuously makes requests."""
            results = []
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                try:
                    req_start = time.perf_counter()
                    
                    # Make request
                    async with session.post(
                        f"{self.base_url}/detect/text",
                        json={
                            'text': random.choice(self.test_texts),
                            'threshold': 0.7
                        }
                    ) as response:
                        await response.json()
                        status = response.status
                    
                    latency = time.perf_counter() - req_start
                    
                    results.append({
                        'timestamp': time.time() - start_time,
                        'latency': latency,
                        'status': status,
                        'worker_id': worker_id
                    })
                    
                except Exception as e:
                    results.append({
                        'timestamp': time.time() - start_time,
                        'error': str(e),
                        'worker_id': worker_id
                    })
                
                # Small delay
                await asyncio.sleep(0.1)
            
            return results
        
        # Run workers
        num_workers = 5
        async with aiohttp.ClientSession() as session:
            tasks = [worker(session, i) for i in range(num_workers)]
            all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        flat_results = []
        for worker_results in all_results:
            flat_results.extend(worker_results)
        
        # Analyze time series
        df = pd.DataFrame(flat_results)
        
        if 'latency' in df.columns:
            # Group by time windows
            df['time_window'] = (df['timestamp'] // 5).astype(int)  # 5-second windows
            
            windowed_stats = df.groupby('time_window')['latency'].agg([
                'count', 'mean', 'std',
                lambda x: x.quantile(0.95),
                lambda x: x.quantile(0.99)
            ])
            windowed_stats.columns = ['count', 'mean', 'std', 'p95', 'p99']
            
            print("\nResponse Time Over Time (5-second windows):")
            for idx, row in windowed_stats.iterrows():
                print(f"  Window {idx}: avg={row['mean']*1000:.2f}ms, "
                      f"p95={row['p95']*1000:.2f}ms, count={row['count']:.0f}")
            
            # Check for degradation
            first_window_avg = windowed_stats.iloc[0]['mean']
            last_window_avg = windowed_stats.iloc[-1]['mean']
            
            if last_window_avg > first_window_avg * 1.5:
                print("\n⚠️ WARNING: Response time degradation detected under load!")
                print(f"  Start: {first_window_avg*1000:.2f}ms")
                print(f"  End: {last_window_avg*1000:.2f}ms")
                print(f"  Degradation: {(last_window_avg/first_window_avg - 1)*100:.1f}%")
    
    async def test_error_recovery(self):
        """Test API error handling and recovery."""
        print("\n" + "="*60)
        print("TESTING ERROR RECOVERY")
        print("="*60)
        
        error_scenarios = [
            {
                'name': 'Invalid JSON',
                'endpoint': '/detect/text',
                'data': 'invalid json',
                'expected_status': [400, 422]
            },
            {
                'name': 'Missing Required Field',
                'endpoint': '/detect/text',
                'data': {'threshold': 0.7},  # Missing 'text' field
                'expected_status': [400, 422]
            },
            {
                'name': 'Invalid File Type',
                'endpoint': '/detect',
                'file': 'test.exe',  # Invalid file type
                'expected_status': [400]
            },
            {
                'name': 'Oversized Request',
                'endpoint': '/detect/text',
                'data': {'text': 'x' * 1000000},  # 1MB of text
                'expected_status': [200, 413, 500]  # May succeed or fail based on limits
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for scenario in error_scenarios:
                print(f"\nTesting: {scenario['name']}")
                
                try:
                    if 'file' in scenario:
                        # File upload test
                        data = aiohttp.FormData()
                        data.add_field('file', b'test content', 
                                     filename=scenario['file'])
                        
                        async with session.post(
                            f"{self.base_url}{scenario['endpoint']}",
                            data=data
                        ) as response:
                            status = response.status
                            content = await response.text()
                    else:
                        # JSON request test
                        async with session.post(
                            f"{self.base_url}{scenario['endpoint']}",
                            json=scenario.get('data')
                        ) as response:
                            status = response.status
                            content = await response.text()
                    
                    if status in scenario['expected_status']:
                        print(f"  ✓ Correct error handling (status: {status})")
                    else:
                        print(f"  ✗ Unexpected status: {status} (expected: {scenario['expected_status']})")
                    
                    # Test recovery - make valid request after error
                    async with session.post(
                        f"{self.base_url}/detect/text",
                        json={'text': 'Valid test text', 'threshold': 0.7}
                    ) as response:
                        recovery_status = response.status
                    
                    if recovery_status == 200:
                        print(f"  ✓ API recovered successfully after error")
                    else:
                        print(f"  ✗ API failed to recover (status: {recovery_status})")
                        
                except Exception as e:
                    print(f"  Error: {e}")
    
    def generate_report(self, output_file: str = "api_performance_report.json"):
        """Generate comprehensive API performance report."""
        print("\n" + "="*60)
        print("GENERATING API PERFORMANCE REPORT")
        print("="*60)
        
        # Save raw results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nReport saved to: {output_file}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Print summary
        self._print_summary()
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _generate_visualizations(self):
        """Generate performance visualization plots."""
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Endpoint latency comparison
        if self.results['endpoint_latency']:
            df = pd.DataFrame(self.results['endpoint_latency'])
            
            endpoints = df['endpoint'].values
            avg_latencies = df['avg_latency_ms'].values
            p95_latencies = df['p95_latency_ms'].values
            
            x = np.arange(len(endpoints))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, avg_latencies, width, label='Average')
            axes[0, 0].bar(x + width/2, p95_latencies, width, label='P95')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(endpoints, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Latency (ms)')
            axes[0, 0].set_title('Endpoint Latency Comparison')
            axes[0, 0].legend()
        
        # 2. Throughput vs Concurrency
        if self.results['concurrent_performance']:
            df = pd.DataFrame(self.results['concurrent_performance'])
            
            axes[0, 1].plot(df['concurrent_requests'], 
                          df['throughput_rps'], 'bo-')
            axes[0, 1].set_xlabel('Concurrent Requests')
            axes[0, 1].set_ylabel('Throughput (req/s)')
            axes[0, 1].set_title('Throughput Scaling')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Success Rate vs Load
        if self.results['concurrent_performance']:
            df = pd.DataFrame(self.results['concurrent_performance'])
            
            axes[1, 0].plot(df['concurrent_requests'], 
                          df['success_rate'], 'go-')
            axes[1, 0].set_xlabel('Concurrent Requests')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_title('Success Rate Under Load')
            axes[1, 0].set_ylim([0, 105])
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Latency vs Load
        if self.results['concurrent_performance']:
            df = pd.DataFrame(self.results['concurrent_performance'])
            
            axes[1, 1].plot(df['concurrent_requests'], 
                          df['avg_latency_ms'], 'b-', label='Average')
            axes[1, 1].plot(df['concurrent_requests'], 
                          df['p95_latency_ms'], 'r-', label='P95')
            axes[1, 1].plot(df['concurrent_requests'], 
                          df['p99_latency_ms'], 'orange', label='P99')
            axes[1, 1].set_xlabel('Concurrent Requests')
            axes[1, 1].set_ylabel('Latency (ms)')
            axes[1, 1].set_title('Latency Under Load')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('api_performance_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to: api_performance_metrics.png")
    
    def _print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("API PERFORMANCE SUMMARY")
        print("="*60)
        
        # Endpoint performance
        if self.results['endpoint_latency']:
            df = pd.DataFrame(self.results['endpoint_latency'])
            
            print("\nEndpoint Performance:")
            for _, row in df.iterrows():
                print(f"\n{row['method']} {row['endpoint']}:")
                print(f"  Avg latency: {row['avg_latency_ms']:.2f}ms")
                print(f"  P95 latency: {row['p95_latency_ms']:.2f}ms")
                print(f"  P99 latency: {row['p99_latency_ms']:.2f}ms")
                print(f"  Error rate: {row['error_rate']:.1f}%")
        
        # Concurrent performance
        if self.results['concurrent_performance']:
            df = pd.DataFrame(self.results['concurrent_performance'])
            
            print("\nConcurrency Scaling:")
            max_throughput_idx = df['throughput_rps'].idxmax()
            optimal_concurrency = df.loc[max_throughput_idx]
            
            print(f"  Optimal concurrency: {optimal_concurrency['concurrent_requests']}")
            print(f"  Max throughput: {optimal_concurrency['throughput_rps']:.1f} req/s")
            print(f"  Latency at optimal: {optimal_concurrency['avg_latency_ms']:.2f}ms")
    
    def _generate_recommendations(self):
        """Generate performance recommendations."""
        print("\n" + "="*60)
        print("PERFORMANCE RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Check endpoint latencies
        if self.results['endpoint_latency']:
            df = pd.DataFrame(self.results['endpoint_latency'])
            
            slow_endpoints = df[df['p95_latency_ms'] > 1000]
            if not slow_endpoints.empty:
                for _, endpoint in slow_endpoints.iterrows():
                    recommendations.append(
                        f"Optimize {endpoint['endpoint']}: P95 latency is {endpoint['p95_latency_ms']:.0f}ms (>1s)"
                    )
            
            high_error_endpoints = df[df['error_rate'] > 5]
            if not high_error_endpoints.empty:
                for _, endpoint in high_error_endpoints.iterrows():
                    recommendations.append(
                        f"Fix errors in {endpoint['endpoint']}: Error rate is {endpoint['error_rate']:.1f}%"
                    )
        
        # Check concurrent performance
        if self.results['concurrent_performance']:
            df = pd.DataFrame(self.results['concurrent_performance'])
            
            # Check if throughput plateaus or decreases
            if len(df) > 2:
                max_throughput = df['throughput_rps'].max()
                last_throughput = df.iloc[-1]['throughput_rps']
                
                if last_throughput < max_throughput * 0.8:
                    recommendations.append(
                        f"System shows throughput degradation at high concurrency. "
                        f"Consider connection pooling or rate limiting."
                    )
            
            # Check success rates
            low_success = df[df['success_rate'] < 95]
            if not low_success.empty:
                min_success = low_success['success_rate'].min()
                recommendations.append(
                    f"Success rate drops to {min_success:.1f}% under load. "
                    f"Investigate timeout settings and error handling."
                )
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec}")
        else:
            print("\n✓ API is performing within acceptable parameters")
        
        print("\n" + "="*60)

async def main():
    """Run API performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test API performance")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Run quick tests")
    
    args = parser.parse_args()
    
    tester = APIPerformanceTester(base_url=args.url)
    
    try:
        # Run tests
        await tester.test_endpoint_latency()
        await tester.test_file_upload_performance()
        
        if not args.quick:
            await tester.test_throughput_limits()
            await tester.test_response_time_under_load(duration_seconds=30)
            await tester.test_error_recovery()
        else:
            await tester.test_concurrent_requests(num_concurrent=10)
        
        # Generate report
        tester.generate_report()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))