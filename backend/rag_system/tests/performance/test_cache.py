"""Redis cache performance testing for the RAG system."""
import time
import json
import hashlib
import random
import string
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import redis
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class CachePerformanceTester:
    """Test Redis cache performance for RAG system."""
    
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize cache tester."""
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.results = {
            'set_operations': [],
            'get_operations': [],
            'hit_miss_ratio': [],
            'memory_usage': [],
            'eviction_stats': [],
            'concurrent_access': []
        }
        
        # Test connection
        try:
            self.redis_client.ping()
            print(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError:
            raise Exception(f"Cannot connect to Redis at {host}:{port}")
    
    def generate_cache_key(self, content: str) -> str:
        """Generate cache key from content."""
        return f"rag:detect:{hashlib.md5(content.encode()).hexdigest()}"
    
    def generate_test_data(self, size: int) -> str:
        """Generate random test data of specified size in bytes."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=size))
    
    def test_basic_operations(self, num_operations: int = 1000):
        """Test basic cache SET and GET operations."""
        print("\n" + "="*60)
        print("TESTING BASIC CACHE OPERATIONS")
        print("="*60)
        
        # Test data sizes (in bytes)
        data_sizes = [100, 1000, 10000, 100000, 1000000]  # 100B to 1MB
        
        for size in data_sizes:
            print(f"\nTesting with {size/1000:.1f}KB data...")
            
            set_times = []
            get_times = []
            
            for i in range(min(num_operations, 100)):
                # Generate test data
                data = self.generate_test_data(size)
                key = self.generate_cache_key(data + str(i))
                
                # Test SET operation
                start = time.perf_counter()
                self.redis_client.set(key, data, ex=300)  # 5 min expiry
                set_time = time.perf_counter() - start
                set_times.append(set_time)
                
                # Test GET operation
                start = time.perf_counter()
                retrieved = self.redis_client.get(key)
                get_time = time.perf_counter() - start
                get_times.append(get_time)
                
                # Verify data integrity
                assert retrieved == data, "Data integrity check failed"
            
            # Store results
            self.results['set_operations'].append({
                'data_size_kb': size/1000,
                'avg_set_time_ms': np.mean(set_times) * 1000,
                'p95_set_time_ms': np.percentile(set_times, 95) * 1000,
                'p99_set_time_ms': np.percentile(set_times, 99) * 1000
            })
            
            self.results['get_operations'].append({
                'data_size_kb': size/1000,
                'avg_get_time_ms': np.mean(get_times) * 1000,
                'p95_get_time_ms': np.percentile(get_times, 95) * 1000,
                'p99_get_time_ms': np.percentile(get_times, 99) * 1000
            })
            
            print(f"  SET: avg={np.mean(set_times)*1000:.2f}ms, p95={np.percentile(set_times, 95)*1000:.2f}ms")
            print(f"  GET: avg={np.mean(get_times)*1000:.2f}ms, p95={np.percentile(get_times, 95)*1000:.2f}ms")
            
            # Clean up
            self.redis_client.delete(key)
    
    def test_hit_miss_ratio(self, cache_size: int = 100, access_pattern: str = 'zipf'):
        """Test cache hit/miss ratio with different access patterns."""
        print("\n" + "="*60)
        print("TESTING CACHE HIT/MISS RATIO")
        print("="*60)
        
        print(f"Cache size: {cache_size} items")
        print(f"Access pattern: {access_pattern}")
        
        # Populate cache
        cache_keys = []
        for i in range(cache_size):
            key = f"rag:test:{i}"
            data = self.generate_test_data(1000)
            self.redis_client.set(key, data, ex=300)
            cache_keys.append(key)
        
        # Generate access pattern
        num_accesses = cache_size * 10
        
        if access_pattern == 'uniform':
            # Uniform random access
            access_indices = np.random.randint(0, cache_size, num_accesses)
        elif access_pattern == 'zipf':
            # Zipf distribution (hot/cold pattern)
            access_indices = np.random.zipf(1.5, num_accesses) - 1
            access_indices = access_indices % cache_size
        elif access_pattern == 'sequential':
            # Sequential access
            access_indices = [i % cache_size for i in range(num_accesses)]
        else:
            raise ValueError(f"Unknown access pattern: {access_pattern}")
        
        # Simulate accesses
        hits = 0
        misses = 0
        access_times = []
        
        for idx in access_indices:
            key = cache_keys[int(idx)]
            
            start = time.perf_counter()
            result = self.redis_client.get(key)
            access_time = time.perf_counter() - start
            access_times.append(access_time)
            
            if result:
                hits += 1
            else:
                misses += 1
                # Re-populate on miss
                data = self.generate_test_data(1000)
                self.redis_client.set(key, data, ex=300)
        
        hit_ratio = hits / (hits + misses)
        
        self.results['hit_miss_ratio'].append({
            'cache_size': cache_size,
            'access_pattern': access_pattern,
            'total_accesses': num_accesses,
            'hits': hits,
            'misses': misses,
            'hit_ratio': hit_ratio,
            'avg_access_time_ms': np.mean(access_times) * 1000,
            'p95_access_time_ms': np.percentile(access_times, 95) * 1000
        })
        
        print(f"\nResults:")
        print(f"  Hit ratio: {hit_ratio:.2%}")
        print(f"  Hits: {hits}, Misses: {misses}")
        print(f"  Avg access time: {np.mean(access_times)*1000:.2f}ms")
        
        # Clean up
        for key in cache_keys:
            self.redis_client.delete(key)
    
    def test_concurrent_access(self, num_threads: int = 10, operations_per_thread: int = 100):
        """Test cache performance under concurrent access."""
        print("\n" + "="*60)
        print("TESTING CONCURRENT CACHE ACCESS")
        print("="*60)
        
        print(f"Threads: {num_threads}")
        print(f"Operations per thread: {operations_per_thread}")
        
        # Shared cache keys
        shared_keys = [f"rag:shared:{i}" for i in range(20)]
        
        # Pre-populate some keys
        for key in shared_keys[:10]:
            self.redis_client.set(key, self.generate_test_data(1000), ex=300)
        
        def worker(thread_id: int) -> Dict:
            """Worker function for concurrent access."""
            local_results = {
                'thread_id': thread_id,
                'operations': [],
                'conflicts': 0
            }
            
            for _ in range(operations_per_thread):
                key = random.choice(shared_keys)
                operation = random.choice(['get', 'set'])
                
                start = time.perf_counter()
                
                if operation == 'get':
                    result = self.redis_client.get(key)
                    success = result is not None
                else:  # set
                    data = self.generate_test_data(1000)
                    try:
                        self.redis_client.set(key, data, ex=300)
                        success = True
                    except:
                        success = False
                        local_results['conflicts'] += 1
                
                elapsed = time.perf_counter() - start
                
                local_results['operations'].append({
                    'type': operation,
                    'time': elapsed,
                    'success': success
                })
            
            return local_results
        
        # Run concurrent test
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            thread_results = [f.result() for f in futures]
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate results
        all_operations = []
        total_conflicts = 0
        
        for result in thread_results:
            all_operations.extend(result['operations'])
            total_conflicts += result['conflicts']
        
        # Calculate statistics
        operation_times = [op['time'] for op in all_operations]
        successful_ops = sum(1 for op in all_operations if op['success'])
        
        self.results['concurrent_access'].append({
            'num_threads': num_threads,
            'operations_per_thread': operations_per_thread,
            'total_operations': len(all_operations),
            'successful_operations': successful_ops,
            'total_conflicts': total_conflicts,
            'total_time': total_time,
            'throughput_ops_per_sec': len(all_operations) / total_time,
            'avg_operation_time_ms': np.mean(operation_times) * 1000,
            'p95_operation_time_ms': np.percentile(operation_times, 95) * 1000,
            'p99_operation_time_ms': np.percentile(operation_times, 99) * 1000
        })
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {len(all_operations)/total_time:.1f} ops/sec")
        print(f"  Avg operation time: {np.mean(operation_times)*1000:.2f}ms")
        print(f"  P95 operation time: {np.percentile(operation_times, 95)*1000:.2f}ms")
        print(f"  Conflicts: {total_conflicts}")
        
        # Clean up
        for key in shared_keys:
            self.redis_client.delete(key)
    
    def test_memory_efficiency(self, num_items: int = 1000):
        """Test memory efficiency and usage patterns."""
        print("\n" + "="*60)
        print("TESTING MEMORY EFFICIENCY")
        print("="*60)
        
        # Clear Redis first
        self.redis_client.flushdb()
        
        # Get initial memory
        info = self.redis_client.info('memory')
        initial_memory = info['used_memory']
        
        print(f"Initial memory: {initial_memory/1024/1024:.2f}MB")
        
        # Test with different data sizes
        test_configs = [
            {'count': num_items, 'size': 100, 'name': 'Small (100B)'},
            {'count': num_items // 10, 'size': 10000, 'name': 'Medium (10KB)'},
            {'count': num_items // 100, 'size': 100000, 'name': 'Large (100KB)'}
        ]
        
        for config in test_configs:
            print(f"\nTesting {config['name']} items...")
            
            # Clear previous data
            self.redis_client.flushdb()
            
            # Add items
            keys_added = []
            total_data_size = 0
            
            for i in range(config['count']):
                key = f"rag:mem_test:{i}"
                data = self.generate_test_data(config['size'])
                self.redis_client.set(key, data, ex=300)
                keys_added.append(key)
                total_data_size += config['size']
            
            # Get memory after adding
            info = self.redis_client.info('memory')
            used_memory = info['used_memory']
            memory_overhead = used_memory - initial_memory
            
            # Calculate efficiency
            efficiency = (total_data_size / memory_overhead) * 100 if memory_overhead > 0 else 0
            
            self.results['memory_usage'].append({
                'item_count': config['count'],
                'item_size_bytes': config['size'],
                'total_data_size_mb': total_data_size / 1024 / 1024,
                'redis_memory_used_mb': memory_overhead / 1024 / 1024,
                'memory_efficiency_percent': efficiency,
                'avg_memory_per_item_bytes': memory_overhead / config['count'] if config['count'] > 0 else 0
            })
            
            print(f"  Items: {config['count']}")
            print(f"  Total data: {total_data_size/1024/1024:.2f}MB")
            print(f"  Redis memory: {memory_overhead/1024/1024:.2f}MB")
            print(f"  Efficiency: {efficiency:.1f}%")
    
    def test_eviction_policies(self, max_memory_mb: int = 10):
        """Test different eviction policies."""
        print("\n" + "="*60)
        print("TESTING EVICTION POLICIES")
        print("="*60)
        
        # Note: This test requires Redis to be configured with maxmemory
        # For testing, we'll simulate eviction by monitoring memory usage
        
        print(f"Testing with simulated {max_memory_mb}MB limit...")
        
        # Clear Redis
        self.redis_client.flushdb()
        
        # Fill cache until "full"
        keys_added = []
        total_size = 0
        target_size = max_memory_mb * 1024 * 1024  # Convert to bytes
        
        item_size = 10000  # 10KB per item
        
        while total_size < target_size:
            key = f"rag:evict:{len(keys_added)}"
            data = self.generate_test_data(item_size)
            self.redis_client.set(key, data, ex=300)
            keys_added.append(key)
            total_size += item_size
        
        print(f"Added {len(keys_added)} items ({total_size/1024/1024:.1f}MB)")
        
        # Now add more items and check what gets evicted
        eviction_test_results = []
        
        for i in range(20):
            new_key = f"rag:evict:new_{i}"
            new_data = self.generate_test_data(item_size)
            self.redis_client.set(new_key, new_data, ex=300)
            
            # Check how many original keys remain
            remaining = sum(1 for key in keys_added if self.redis_client.exists(key))
            evicted = len(keys_added) - remaining
            
            eviction_test_results.append({
                'iteration': i + 1,
                'original_keys_remaining': remaining,
                'keys_evicted': evicted
            })
        
        # Store results
        self.results['eviction_stats'] = eviction_test_results
        
        if eviction_test_results:
            final_evicted = eviction_test_results[-1]['keys_evicted']
            print(f"\nEviction Results:")
            print(f"  Original keys: {len(keys_added)}")
            print(f"  Keys evicted: {final_evicted}")
            print(f"  Eviction rate: {final_evicted/len(keys_added)*100:.1f}%")
        
        # Clean up
        self.redis_client.flushdb()
    
    def test_pipeline_performance(self, num_operations: int = 1000):
        """Test Redis pipeline for batch operations."""
        print("\n" + "="*60)
        print("TESTING PIPELINE PERFORMANCE")
        print("="*60)
        
        # Test data
        test_data = [(f"rag:pipe:{i}", self.generate_test_data(1000)) 
                     for i in range(num_operations)]
        
        # Test without pipeline
        print(f"\nWithout pipeline ({num_operations} operations)...")
        start = time.perf_counter()
        for key, value in test_data:
            self.redis_client.set(key, value, ex=300)
        no_pipeline_time = time.perf_counter() - start
        
        # Clear for next test
        for key, _ in test_data:
            self.redis_client.delete(key)
        
        # Test with pipeline
        print(f"With pipeline ({num_operations} operations)...")
        start = time.perf_counter()
        pipe = self.redis_client.pipeline()
        for key, value in test_data:
            pipe.set(key, value, ex=300)
        pipe.execute()
        pipeline_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = no_pipeline_time / pipeline_time if pipeline_time > 0 else 0
        
        print(f"\nResults:")
        print(f"  Without pipeline: {no_pipeline_time:.3f}s")
        print(f"  With pipeline: {pipeline_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Throughput (no pipeline): {num_operations/no_pipeline_time:.1f} ops/s")
        print(f"  Throughput (pipeline): {num_operations/pipeline_time:.1f} ops/s")
        
        # Clean up
        for key, _ in test_data:
            self.redis_client.delete(key)
        
        return {
            'no_pipeline_time': no_pipeline_time,
            'pipeline_time': pipeline_time,
            'speedup': speedup
        }
    
    def generate_report(self, output_file: str = "cache_performance_report.json"):
        """Generate comprehensive cache performance report."""
        print("\n" + "="*60)
        print("GENERATING CACHE PERFORMANCE REPORT")
        print("="*60)
        
        # Save raw results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nReport saved to: {output_file}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Print summary
        self._print_summary()
    
    def _generate_visualizations(self):
        """Generate performance visualization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. SET/GET operation times vs data size
        if self.results['set_operations'] and self.results['get_operations']:
            set_df = pd.DataFrame(self.results['set_operations'])
            get_df = pd.DataFrame(self.results['get_operations'])
            
            axes[0, 0].plot(set_df['data_size_kb'], set_df['avg_set_time_ms'], 
                          'bo-', label='SET')
            axes[0, 0].plot(get_df['data_size_kb'], get_df['avg_get_time_ms'], 
                          'ro-', label='GET')
            axes[0, 0].set_xlabel('Data Size (KB)')
            axes[0, 0].set_ylabel('Time (ms)')
            axes[0, 0].set_title('Cache Operation Performance')
            axes[0, 0].legend()
            axes[0, 0].set_xscale('log')
        
        # 2. Hit ratio by access pattern
        if self.results['hit_miss_ratio']:
            hit_df = pd.DataFrame(self.results['hit_miss_ratio'])
            patterns = hit_df['access_pattern'].unique()
            x = np.arange(len(patterns))
            
            axes[0, 1].bar(x, [hit_df[hit_df['access_pattern'] == p]['hit_ratio'].mean() 
                              for p in patterns])
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(patterns)
            axes[0, 1].set_ylabel('Hit Ratio')
            axes[0, 1].set_title('Cache Hit Ratio by Access Pattern')
            axes[0, 1].set_ylim([0, 1])
        
        # 3. Concurrent access performance
        if self.results['concurrent_access']:
            conc_df = pd.DataFrame(self.results['concurrent_access'])
            axes[1, 0].plot(conc_df['num_threads'], 
                          conc_df['throughput_ops_per_sec'], 'go-')
            axes[1, 0].set_xlabel('Number of Threads')
            axes[1, 0].set_ylabel('Throughput (ops/sec)')
            axes[1, 0].set_title('Concurrent Access Throughput')
        
        # 4. Memory efficiency
        if self.results['memory_usage']:
            mem_df = pd.DataFrame(self.results['memory_usage'])
            axes[1, 1].bar(range(len(mem_df)), 
                          mem_df['memory_efficiency_percent'])
            axes[1, 1].set_xlabel('Test Configuration')
            axes[1, 1].set_ylabel('Memory Efficiency (%)')
            axes[1, 1].set_title('Memory Usage Efficiency')
        
        plt.tight_layout()
        plt.savefig('cache_performance_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to: cache_performance_metrics.png")
    
    def _print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("CACHE PERFORMANCE SUMMARY")
        print("="*60)
        
        # SET/GET performance
        if self.results['set_operations']:
            set_df = pd.DataFrame(self.results['set_operations'])
            print("\nSET Operation Performance:")
            print(f"  Small (0.1KB): {set_df.iloc[0]['avg_set_time_ms']:.2f}ms")
            if len(set_df) > 2:
                print(f"  Large (100KB): {set_df.iloc[2]['avg_set_time_ms']:.2f}ms")
        
        if self.results['get_operations']:
            get_df = pd.DataFrame(self.results['get_operations'])
            print("\nGET Operation Performance:")
            print(f"  Small (0.1KB): {get_df.iloc[0]['avg_get_time_ms']:.2f}ms")
            if len(get_df) > 2:
                print(f"  Large (100KB): {get_df.iloc[2]['avg_get_time_ms']:.2f}ms")
        
        # Hit ratio
        if self.results['hit_miss_ratio']:
            hit_df = pd.DataFrame(self.results['hit_miss_ratio'])
            print("\nCache Hit Ratios:")
            for pattern in hit_df['access_pattern'].unique():
                ratio = hit_df[hit_df['access_pattern'] == pattern]['hit_ratio'].mean()
                print(f"  {pattern}: {ratio:.2%}")
        
        # Concurrent access
        if self.results['concurrent_access']:
            conc_df = pd.DataFrame(self.results['concurrent_access'])
            max_throughput = conc_df['throughput_ops_per_sec'].max()
            print(f"\nMax Concurrent Throughput: {max_throughput:.1f} ops/sec")
        
        print("\n" + "="*60)

def main():
    """Run cache performance tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Redis cache performance")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--quick", action="store_true", help="Run quick tests")
    
    args = parser.parse_args()
    
    try:
        tester = CachePerformanceTester(host=args.host, port=args.port)
        
        # Run tests
        if args.quick:
            tester.test_basic_operations(num_operations=100)
            tester.test_hit_miss_ratio(cache_size=50)
            tester.test_concurrent_access(num_threads=5, operations_per_thread=20)
        else:
            tester.test_basic_operations(num_operations=1000)
            tester.test_hit_miss_ratio(cache_size=100, access_pattern='uniform')
            tester.test_hit_miss_ratio(cache_size=100, access_pattern='zipf')
            tester.test_hit_miss_ratio(cache_size=100, access_pattern='sequential')
            tester.test_concurrent_access(num_threads=10, operations_per_thread=100)
            tester.test_memory_efficiency(num_items=1000)
            tester.test_eviction_policies(max_memory_mb=10)
            tester.test_pipeline_performance(num_operations=1000)
        
        # Generate report
        tester.generate_report()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())