"""
Performance profiling system for identifying bottlenecks and optimization opportunities.
Provides CPU, memory, I/O, and custom profiling capabilities.
"""

import time
import cProfile
import pstats
import io
import tracemalloc
import asyncio
import functools
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import sys
import gc
import objgraph
import numpy as np
from memory_profiler import memory_usage
from line_profiler import LineProfiler


@dataclass
class ProfileResult:
    """Result of a profiling session"""
    profile_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    summary: Dict[str, Any]
    details: Any = None
    recommendations: List[str] = field(default_factory=list)


class PerformanceProfiler:
    """
    Comprehensive performance profiling system.
    """
    
    def __init__(self, enable_auto_profiling: bool = False):
        """
        Initialize performance profiler.
        
        Args:
            enable_auto_profiling: Enable automatic profiling of slow operations
        """
        self.enable_auto_profiling = enable_auto_profiling
        self.profile_results: List[ProfileResult] = []
        self.max_results = 100
        
        # Thresholds for auto-profiling
        self.thresholds = {
            'slow_function_seconds': 1.0,
            'high_memory_mb': 100,
            'high_cpu_percent': 80
        }
        
        # Memory tracking
        self.memory_snapshots = []
        
        # CPU profiling
        self.cpu_profiler = None
        
        # Line profiler for detailed analysis
        self.line_profiler = LineProfiler()
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile a function's performance.
        
        Args:
            func: Function to profile
        
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start profiling
            start_time = datetime.utcnow()
            start_perf = time.perf_counter()
            
            # Memory before
            tracemalloc.start()
            memory_before = tracemalloc.get_traced_memory()[0]
            
            # CPU profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Stop profiling
                profiler.disable()
                duration = time.perf_counter() - start_perf
                
                # Memory after
                memory_after = tracemalloc.get_traced_memory()[0]
                memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
                
                # Generate profile report
                profile_result = self._analyze_cpu_profile(
                    profiler, func.__name__, start_time, duration
                )
                
                # Add memory information
                profile_result.summary['memory_used_mb'] = memory_used
                
                # Check if should auto-profile
                if self.enable_auto_profiling and duration > self.thresholds['slow_function_seconds']:
                    self._store_profile_result(profile_result)
                    self._generate_recommendations(profile_result)
                
                return result
                
            finally:
                tracemalloc.stop()
        
        return wrapper
    
    def profile_async_function(self, func: Callable) -> Callable:
        """
        Decorator to profile an async function's performance.
        
        Args:
            func: Async function to profile
        
        Returns:
            Wrapped async function with profiling
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Start profiling
            start_time = datetime.utcnow()
            start_perf = time.perf_counter()
            
            # Memory before
            tracemalloc.start()
            memory_before = tracemalloc.get_traced_memory()[0]
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                duration = time.perf_counter() - start_perf
                
                # Memory after
                memory_after = tracemalloc.get_traced_memory()[0]
                memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
                
                # Create profile result
                profile_result = ProfileResult(
                    profile_type='async_function',
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    duration_seconds=duration,
                    summary={
                        'function_name': func.__name__,
                        'memory_used_mb': memory_used,
                        'async': True
                    }
                )
                
                # Check if should auto-profile
                if self.enable_auto_profiling and duration > self.thresholds['slow_function_seconds']:
                    self._store_profile_result(profile_result)
                    self._generate_recommendations(profile_result)
                
                return result
                
            finally:
                tracemalloc.stop()
        
        return wrapper
    
    def start_cpu_profiling(self):
        """Start CPU profiling session"""
        self.cpu_profiler = cProfile.Profile()
        self.cpu_profiler.enable()
        self.cpu_start_time = datetime.utcnow()
    
    def stop_cpu_profiling(self) -> ProfileResult:
        """
        Stop CPU profiling and return results.
        
        Returns:
            ProfileResult with CPU profiling data
        """
        if not self.cpu_profiler:
            raise RuntimeError("CPU profiling not started")
        
        self.cpu_profiler.disable()
        duration = (datetime.utcnow() - self.cpu_start_time).total_seconds()
        
        result = self._analyze_cpu_profile(
            self.cpu_profiler, 
            "cpu_session",
            self.cpu_start_time,
            duration
        )
        
        self.cpu_profiler = None
        self._store_profile_result(result)
        
        return result
    
    def _analyze_cpu_profile(self, profiler: cProfile.Profile, 
                            name: str, start_time: datetime,
                            duration: float) -> ProfileResult:
        """
        Analyze CPU profile data.
        
        Args:
            profiler: cProfile.Profile object
            name: Name of the profiled operation
            start_time: Start time of profiling
            duration: Duration in seconds
        
        Returns:
            ProfileResult with analysis
        """
        # Get statistics
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        # Get top functions by time
        stats.print_stats(20)
        profile_output = stream.getvalue()
        
        # Extract key metrics
        total_calls = sum(stats.stats[func][0] for func in stats.stats)
        
        # Find hotspots (functions taking >5% of time)
        hotspots = []
        for func, (ncalls, tottime, cumtime, callers) in stats.stats.items():
            if cumtime / duration > 0.05:  # >5% of total time
                hotspots.append({
                    'function': f"{func[0]}:{func[1]} {func[2]}",
                    'calls': ncalls,
                    'total_time': tottime,
                    'cumulative_time': cumtime,
                    'percent_time': (cumtime / duration) * 100
                })
        
        return ProfileResult(
            profile_type='cpu',
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration_seconds=duration,
            summary={
                'name': name,
                'total_calls': total_calls,
                'hotspot_count': len(hotspots),
                'top_hotspots': hotspots[:5]
            },
            details=profile_output
        )
    
    def profile_memory(self, func: Callable, *args, **kwargs) -> Tuple[Any, ProfileResult]:
        """
        Profile memory usage of a function.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (function result, ProfileResult)
        """
        start_time = datetime.utcnow()
        
        # Track memory allocations
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        # Get memory usage over time
        mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=60)
        
        # Execute function and get result
        result = func(*args, **kwargs)
        
        # Take snapshot after
        snapshot_after = tracemalloc.take_snapshot()
        
        # Analyze memory changes
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        
        # Get top memory consumers
        memory_consumers = []
        for stat in top_stats[:10]:
            memory_consumers.append({
                'file': stat.traceback[0].filename,
                'line': stat.traceback[0].lineno,
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count_diff': stat.count_diff
            })
        
        # Memory statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        profile_result = ProfileResult(
            profile_type='memory',
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            summary={
                'function_name': func.__name__,
                'peak_memory_mb': max(mem_usage),
                'avg_memory_mb': np.mean(mem_usage),
                'memory_allocated_mb': (peak - current) / 1024 / 1024,
                'top_consumers': memory_consumers[:5]
            },
            details={
                'memory_usage_timeline': mem_usage,
                'all_consumers': memory_consumers
            }
        )
        
        self._store_profile_result(profile_result)
        self._generate_memory_recommendations(profile_result)
        
        return result, profile_result
    
    def analyze_memory_leaks(self) -> ProfileResult:
        """
        Analyze potential memory leaks.
        
        Returns:
            ProfileResult with memory leak analysis
        """
        start_time = datetime.utcnow()
        
        # Force garbage collection
        gc.collect()
        
        # Get object counts
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Sort by count
        sorted_counts = sorted(
            object_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        # Get largest objects
        largest_objects = []
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > 1024 * 1024:  # Objects larger than 1MB
                    largest_objects.append({
                        'type': type(obj).__name__,
                        'size_mb': size / 1024 / 1024,
                        'id': id(obj)
                    })
            except:
                pass
        
        largest_objects.sort(key=lambda x: x['size_mb'], reverse=True)
        
        # Check for circular references
        circular_refs = []
        for obj in gc.garbage:
            circular_refs.append({
                'type': type(obj).__name__,
                'id': id(obj)
            })
        
        profile_result = ProfileResult(
            profile_type='memory_leak',
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            summary={
                'total_objects': sum(object_counts.values()),
                'object_types': len(object_counts),
                'largest_objects_count': len(largest_objects),
                'circular_references': len(circular_refs),
                'top_object_types': dict(sorted_counts[:10])
            },
            details={
                'all_object_counts': dict(sorted_counts),
                'largest_objects': largest_objects[:10],
                'circular_references': circular_refs[:10]
            }
        )
        
        self._store_profile_result(profile_result)
        
        # Generate recommendations
        if len(circular_refs) > 0:
            profile_result.recommendations.append(
                f"Found {len(circular_refs)} circular references. Consider using weakref."
            )
        
        if len(largest_objects) > 10:
            profile_result.recommendations.append(
                f"Found {len(largest_objects)} large objects (>1MB). Consider optimizing memory usage."
            )
        
        return profile_result
    
    def profile_io_operations(self, func: Callable, *args, **kwargs) -> Tuple[Any, ProfileResult]:
        """
        Profile I/O operations of a function.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (function result, ProfileResult)
        """
        start_time = datetime.utcnow()
        
        # Get initial I/O counters
        process = psutil.Process()
        io_before = process.io_counters()
        
        # Execute function
        start_perf = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_perf
        
        # Get final I/O counters
        io_after = process.io_counters()
        
        # Calculate I/O metrics
        read_bytes = io_after.read_bytes - io_before.read_bytes
        write_bytes = io_after.write_bytes - io_before.write_bytes
        read_count = io_after.read_count - io_before.read_count
        write_count = io_after.write_count - io_before.write_count
        
        profile_result = ProfileResult(
            profile_type='io',
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration_seconds=duration,
            summary={
                'function_name': func.__name__,
                'read_mb': read_bytes / 1024 / 1024,
                'write_mb': write_bytes / 1024 / 1024,
                'read_operations': read_count,
                'write_operations': write_count,
                'read_throughput_mbps': (read_bytes / 1024 / 1024) / duration if duration > 0 else 0,
                'write_throughput_mbps': (write_bytes / 1024 / 1024) / duration if duration > 0 else 0
            }
        )
        
        self._store_profile_result(profile_result)
        self._generate_io_recommendations(profile_result)
        
        return result, profile_result
    
    def profile_database_queries(self, queries: List[Dict[str, Any]]) -> ProfileResult:
        """
        Analyze database query performance.
        
        Args:
            queries: List of query execution data
        
        Returns:
            ProfileResult with query analysis
        """
        start_time = datetime.utcnow()
        
        # Analyze queries
        total_time = sum(q.get('duration', 0) for q in queries)
        slow_queries = [q for q in queries if q.get('duration', 0) > 0.1]
        
        # Group by query type
        query_types = {}
        for q in queries:
            qtype = q.get('type', 'unknown')
            if qtype not in query_types:
                query_types[qtype] = {'count': 0, 'total_time': 0}
            query_types[qtype]['count'] += 1
            query_types[qtype]['total_time'] += q.get('duration', 0)
        
        # Find N+1 query patterns
        n_plus_one_candidates = []
        query_counts = {}
        for q in queries:
            query_template = q.get('template', '')
            query_counts[query_template] = query_counts.get(query_template, 0) + 1
        
        for template, count in query_counts.items():
            if count > 10:  # Potential N+1 if same query executed many times
                n_plus_one_candidates.append({
                    'template': template[:100],
                    'count': count
                })
        
        profile_result = ProfileResult(
            profile_type='database',
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            summary={
                'total_queries': len(queries),
                'total_time_seconds': total_time,
                'slow_queries': len(slow_queries),
                'avg_query_time': total_time / len(queries) if queries else 0,
                'query_types': query_types,
                'n_plus_one_candidates': len(n_plus_one_candidates)
            },
            details={
                'slow_queries': slow_queries[:10],
                'n_plus_one_patterns': n_plus_one_candidates[:5]
            }
        )
        
        self._store_profile_result(profile_result)
        
        # Generate recommendations
        if len(slow_queries) > 0:
            profile_result.recommendations.append(
                f"Found {len(slow_queries)} slow queries (>100ms). Consider adding indexes."
            )
        
        if len(n_plus_one_candidates) > 0:
            profile_result.recommendations.append(
                f"Potential N+1 query problem detected. Consider using eager loading."
            )
        
        return profile_result
    
    def analyze_concurrency(self) -> ProfileResult:
        """
        Analyze concurrency and threading issues.
        
        Returns:
            ProfileResult with concurrency analysis
        """
        start_time = datetime.utcnow()
        
        # Get thread information
        threads = threading.enumerate()
        thread_info = []
        
        for thread in threads:
            thread_info.append({
                'name': thread.name,
                'daemon': thread.daemon,
                'alive': thread.is_alive(),
                'ident': thread.ident
            })
        
        # Get asyncio task information
        try:
            loop = asyncio.get_event_loop()
            tasks = asyncio.all_tasks(loop)
            task_info = []
            
            for task in tasks:
                task_info.append({
                    'name': task.get_name(),
                    'done': task.done(),
                    'cancelled': task.cancelled()
                })
        except:
            task_info = []
        
        profile_result = ProfileResult(
            profile_type='concurrency',
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            summary={
                'thread_count': len(threads),
                'daemon_threads': sum(1 for t in threads if t.daemon),
                'asyncio_tasks': len(task_info),
                'completed_tasks': sum(1 for t in task_info if t['done'])
            },
            details={
                'threads': thread_info,
                'tasks': task_info[:20]
            }
        )
        
        self._store_profile_result(profile_result)
        
        # Generate recommendations
        if len(threads) > 100:
            profile_result.recommendations.append(
                f"High thread count ({len(threads)}). Consider using thread pools."
            )
        
        if len(task_info) > 1000:
            profile_result.recommendations.append(
                f"High async task count ({len(task_info)}). Check for task leaks."
            )
        
        return profile_result
    
    def get_flamegraph_data(self, profile_result: ProfileResult) -> Dict[str, Any]:
        """
        Generate flamegraph data from profile result.
        
        Args:
            profile_result: ProfileResult to convert
        
        Returns:
            Flamegraph-compatible data structure
        """
        if profile_result.profile_type != 'cpu':
            raise ValueError("Flamegraph only available for CPU profiles")
        
        # Parse profile details to generate flamegraph structure
        # This would integrate with flamegraph visualization tools
        
        return {
            'name': 'root',
            'value': profile_result.duration_seconds,
            'children': self._parse_profile_for_flamegraph(profile_result.details)
        }
    
    def _parse_profile_for_flamegraph(self, profile_output: str) -> List[Dict[str, Any]]:
        """Parse profile output for flamegraph visualization"""
        # Implementation would parse the profile output
        # and create hierarchical structure for flamegraph
        return []
    
    def _store_profile_result(self, result: ProfileResult):
        """Store profile result with size limit"""
        self.profile_results.append(result)
        
        # Trim old results
        if len(self.profile_results) > self.max_results:
            self.profile_results = self.profile_results[-self.max_results:]
    
    def _generate_recommendations(self, result: ProfileResult):
        """Generate performance recommendations based on profile"""
        if result.duration_seconds > 5:
            result.recommendations.append(
                "Function execution time >5 seconds. Consider optimization or async processing."
            )
        
        if result.summary.get('memory_used_mb', 0) > 100:
            result.recommendations.append(
                "High memory usage detected. Consider streaming or chunking data."
            )
    
    def _generate_memory_recommendations(self, result: ProfileResult):
        """Generate memory-specific recommendations"""
        peak_memory = result.summary.get('peak_memory_mb', 0)
        
        if peak_memory > 500:
            result.recommendations.append(
                f"Peak memory usage {peak_memory:.1f}MB. Consider memory optimization."
            )
        
        top_consumers = result.summary.get('top_consumers', [])
        if top_consumers:
            largest = top_consumers[0]
            if largest['size_diff_mb'] > 50:
                result.recommendations.append(
                    f"Large memory allocation at {largest['file']}:{largest['line']}"
                )
    
    def _generate_io_recommendations(self, result: ProfileResult):
        """Generate I/O-specific recommendations"""
        read_mb = result.summary.get('read_mb', 0)
        write_mb = result.summary.get('write_mb', 0)
        
        if read_mb > 100:
            result.recommendations.append(
                f"High I/O read volume ({read_mb:.1f}MB). Consider caching."
            )
        
        if write_mb > 100:
            result.recommendations.append(
                f"High I/O write volume ({write_mb:.1f}MB). Consider batching writes."
            )
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Get summary of all profiling results.
        
        Returns:
            Summary statistics of profiling sessions
        """
        if not self.profile_results:
            return {'message': 'No profiling data available'}
        
        # Group by type
        by_type = {}
        for result in self.profile_results:
            ptype = result.profile_type
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(result)
        
        # Calculate statistics
        summary = {}
        for ptype, results in by_type.items():
            durations = [r.duration_seconds for r in results]
            summary[ptype] = {
                'count': len(results),
                'avg_duration': np.mean(durations),
                'max_duration': np.max(durations),
                'min_duration': np.min(durations)
            }
        
        return {
            'total_profiles': len(self.profile_results),
            'by_type': summary,
            'latest_profile': {
                'type': self.profile_results[-1].profile_type,
                'time': self.profile_results[-1].start_time.isoformat(),
                'duration': self.profile_results[-1].duration_seconds
            }
        }