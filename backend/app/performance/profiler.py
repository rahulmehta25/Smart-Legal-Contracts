"""
Application Performance Profiler
Provides CPU, memory, and execution time profiling for code optimization
"""

import cProfile
import pstats
import io
import tracemalloc
import functools
import time
import psutil
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import inspect
from pathlib import Path
import linecache
import sys

@dataclass
class ProfileResult:
    """Container for profiling results"""
    function_name: str
    execution_time: float
    cpu_percent: float
    memory_used: float
    memory_peak: float
    call_count: int
    timestamp: datetime
    cpu_profile: Optional[str] = None
    memory_profile: Optional[Dict] = None
    flamegraph_data: Optional[Dict] = None
    
class PerformanceProfiler:
    """Advanced performance profiler with multiple profiling modes"""
    
    def __init__(self, output_dir: str = "./profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ProfileResult] = []
        self.is_profiling = False
        self.cpu_baseline = None
        self.memory_baseline = None
        
    def profile_cpu(self, func: Callable) -> Callable:
        """Decorator for CPU profiling"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.perf_counter()
            start_cpu = psutil.cpu_percent(interval=0.1)
            
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                
                end_time = time.perf_counter()
                end_cpu = psutil.cpu_percent(interval=0.1)
                
                # Generate profile stats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(50)
                
                profile_result = ProfileResult(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    cpu_percent=end_cpu - start_cpu,
                    memory_used=0,
                    memory_peak=0,
                    call_count=1,
                    timestamp=datetime.now(),
                    cpu_profile=s.getvalue()
                )
                
                self.results.append(profile_result)
                self._save_result(profile_result)
                
            return result
        return wrapper
    
    def profile_memory(self, func: Callable) -> Callable:
        """Decorator for memory profiling"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            
            start_memory = tracemalloc.get_traced_memory()
            
            result = func(*args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            memory_profile = {
                'top_allocations': [
                    {
                        'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                        'size': stat.size,
                        'count': stat.count
                    }
                    for stat in top_stats
                ]
            }
            
            profile_result = ProfileResult(
                function_name=func.__name__,
                execution_time=0,
                cpu_percent=0,
                memory_used=(current - start_memory[0]) / 1024 / 1024,  # MB
                memory_peak=peak / 1024 / 1024,  # MB
                call_count=1,
                timestamp=datetime.now(),
                memory_profile=memory_profile
            )
            
            self.results.append(profile_result)
            self._save_result(profile_result)
            
            return result
        return wrapper
    
    def profile_async(self, func: Callable) -> Callable:
        """Decorator for async function profiling"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = await func(*args, **kwargs)
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            profile_result = ProfileResult(
                function_name=func.__name__,
                execution_time=end_time - start_time,
                cpu_percent=process.cpu_percent(),
                memory_used=end_memory - start_memory,
                memory_peak=end_memory,
                call_count=1,
                timestamp=datetime.now()
            )
            
            self.results.append(profile_result)
            self._save_result(profile_result)
            
            return result
        return wrapper
    
    def line_profiler(self, func: Callable) -> Callable:
        """Line-by-line profiling decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get source lines
            source_lines = inspect.getsourcelines(func)[0]
            line_times = {}
            
            # Profile each line (simplified version)
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # For actual line profiling, we'd need to use sys.settrace
            # This is a simplified version showing the concept
            
            profile_data = {
                'function': func.__name__,
                'total_time': end_time - start_time,
                'source': ''.join(source_lines)
            }
            
            self._save_line_profile(profile_data)
            
            return result
        return wrapper
    
    def generate_flamegraph(self, profile_data: str) -> Dict:
        """Generate flamegraph data from profile results"""
        flamegraph = {
            'name': 'root',
            'value': 0,
            'children': []
        }
        
        # Parse profile data and build tree structure
        lines = profile_data.split('\n')
        stack = []
        
        for line in lines:
            if 'function calls' in line or not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                try:
                    calls = int(parts[0])
                    time_val = float(parts[1])
                    func_name = ' '.join(parts[5:])
                    
                    node = {
                        'name': func_name,
                        'value': time_val,
                        'calls': calls,
                        'children': []
                    }
                    
                    flamegraph['children'].append(node)
                    flamegraph['value'] += time_val
                except (ValueError, IndexError):
                    continue
        
        return flamegraph
    
    def benchmark(self, func: Callable, iterations: int = 100) -> Dict:
        """Benchmark a function with multiple iterations"""
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            process = psutil.Process()
            
            start_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.perf_counter()
            
            func()
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        # Calculate statistics
        import statistics
        
        return {
            'function': func.__name__,
            'iterations': iterations,
            'timing': {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
                'p95': sorted(times)[int(0.95 * len(times))],
                'p99': sorted(times)[int(0.99 * len(times))]
            },
            'memory': {
                'mean': statistics.mean(memory_usage),
                'median': statistics.median(memory_usage),
                'max': max(memory_usage)
            }
        }
    
    def compare_implementations(self, implementations: Dict[str, Callable], 
                              test_data: Any, iterations: int = 100) -> Dict:
        """Compare performance of different implementations"""
        results = {}
        
        for name, impl in implementations.items():
            times = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                impl(test_data)
                times.append(time.perf_counter() - start)
            
            results[name] = {
                'mean_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        # Find best performer
        best = min(results.items(), key=lambda x: x[1]['mean_time'])
        
        # Calculate relative performance
        for name in results:
            results[name]['relative_speed'] = (
                results[name]['mean_time'] / best[1]['mean_time']
            )
        
        return {
            'comparison': results,
            'best_performer': best[0],
            'rankings': sorted(
                results.items(), 
                key=lambda x: x[1]['mean_time']
            )
        }
    
    def profile_database_query(self, query_func: Callable) -> Callable:
        """Profile database query performance"""
        @functools.wraps(query_func)
        def wrapper(*args, **kwargs):
            import sqlalchemy.event
            from sqlalchemy.engine import Engine
            
            query_times = []
            
            @sqlalchemy.event.listens_for(Engine, "before_cursor_execute")
            def before_cursor_execute(conn, cursor, statement, parameters, 
                                     context, executemany):
                conn.info.setdefault('query_start_time', []).append(
                    time.perf_counter()
                )
            
            @sqlalchemy.event.listens_for(Engine, "after_cursor_execute")
            def after_cursor_execute(conn, cursor, statement, parameters, 
                                    context, executemany):
                total = time.perf_counter() - conn.info['query_start_time'].pop()
                query_times.append({
                    'query': statement[:100],
                    'time': total,
                    'parameters': str(parameters)[:100]
                })
            
            result = query_func(*args, **kwargs)
            
            # Analyze query performance
            if query_times:
                slow_queries = [q for q in query_times if q['time'] > 0.1]
                
                profile_data = {
                    'function': query_func.__name__,
                    'total_queries': len(query_times),
                    'total_time': sum(q['time'] for q in query_times),
                    'slow_queries': slow_queries,
                    'average_time': sum(q['time'] for q in query_times) / len(query_times)
                }
                
                self._save_query_profile(profile_data)
            
            return result
        return wrapper
    
    def continuous_profiling(self, interval: int = 60):
        """Continuous profiling in background"""
        def profile_loop():
            while self.is_profiling:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'disk_percent': disk.percent,
                    'process_count': len(psutil.pids())
                }
                
                # Save metrics
                self._save_continuous_metrics(metrics)
                
                time.sleep(interval)
        
        self.is_profiling = True
        thread = threading.Thread(target=profile_loop, daemon=True)
        thread.start()
    
    def stop_profiling(self):
        """Stop continuous profiling"""
        self.is_profiling = False
    
    def _save_result(self, result: ProfileResult):
        """Save profiling result to file"""
        filename = self.output_dir / f"profile_{result.function_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'function_name': result.function_name,
            'execution_time': result.execution_time,
            'cpu_percent': result.cpu_percent,
            'memory_used': result.memory_used,
            'memory_peak': result.memory_peak,
            'call_count': result.call_count,
            'timestamp': result.timestamp.isoformat(),
            'cpu_profile': result.cpu_profile,
            'memory_profile': result.memory_profile,
            'flamegraph_data': result.flamegraph_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_line_profile(self, data: Dict):
        """Save line profiling data"""
        filename = self.output_dir / f"line_profile_{data['function']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_query_profile(self, data: Dict):
        """Save database query profiling data"""
        filename = self.output_dir / f"query_profile_{data['function']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_continuous_metrics(self, metrics: Dict):
        """Save continuous profiling metrics"""
        filename = self.output_dir / f"continuous_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(filename, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def generate_report(self) -> str:
        """Generate performance profiling report"""
        if not self.results:
            return "No profiling results available"
        
        report = ["Performance Profiling Report", "=" * 50, ""]
        
        # Summary statistics
        total_time = sum(r.execution_time for r in self.results)
        total_memory = sum(r.memory_used for r in self.results)
        
        report.append(f"Total Functions Profiled: {len(self.results)}")
        report.append(f"Total Execution Time: {total_time:.4f} seconds")
        report.append(f"Total Memory Used: {total_memory:.2f} MB")
        report.append("")
        
        # Function details
        report.append("Function Performance Details:")
        report.append("-" * 40)
        
        for result in sorted(self.results, key=lambda x: x.execution_time, reverse=True):
            report.append(f"\nFunction: {result.function_name}")
            report.append(f"  Execution Time: {result.execution_time:.4f}s")
            report.append(f"  CPU Usage: {result.cpu_percent:.2f}%")
            report.append(f"  Memory Used: {result.memory_used:.2f} MB")
            report.append(f"  Memory Peak: {result.memory_peak:.2f} MB")
            report.append(f"  Timestamp: {result.timestamp}")
        
        # Bottlenecks
        report.append("\n" + "=" * 50)
        report.append("Performance Bottlenecks:")
        report.append("-" * 40)
        
        # Find slowest functions
        slowest = sorted(self.results, key=lambda x: x.execution_time, reverse=True)[:5]
        report.append("\nSlowest Functions:")
        for result in slowest:
            report.append(f"  - {result.function_name}: {result.execution_time:.4f}s")
        
        # Find memory-intensive functions
        memory_intensive = sorted(self.results, key=lambda x: x.memory_peak, reverse=True)[:5]
        report.append("\nMemory-Intensive Functions:")
        for result in memory_intensive:
            report.append(f"  - {result.function_name}: {result.memory_peak:.2f} MB")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    profiler = PerformanceProfiler()
    
    # Example function to profile
    @profiler.profile_cpu
    @profiler.profile_memory
    def expensive_operation():
        """Simulate an expensive operation"""
        import numpy as np
        
        # CPU-intensive operation
        data = np.random.rand(1000, 1000)
        result = np.linalg.inv(data)
        
        # Memory-intensive operation
        large_list = [i for i in range(1000000)]
        
        return result
    
    # Run profiling
    expensive_operation()
    
    # Generate report
    print(profiler.generate_report())
    
    # Benchmark example
    benchmark_result = profiler.benchmark(expensive_operation, iterations=10)
    print(f"\nBenchmark Results: {json.dumps(benchmark_result, indent=2)}")