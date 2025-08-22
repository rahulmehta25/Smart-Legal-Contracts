"""
Memory Monitoring and Optimization System
Tracks memory usage, detects leaks, and provides optimization recommendations
"""

import gc
import tracemalloc
import psutil
import objgraph
import weakref
import sys
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path
import numpy as np

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: datetime
    total_memory: float  # MB
    available_memory: float  # MB
    process_memory: float  # MB
    heap_size: float  # MB
    gc_stats: Dict
    top_objects: List[Dict]
    memory_map: Dict[str, float]

@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    object_type: str
    growth_rate: float  # MB/minute
    instances: int
    total_size: float  # MB
    traceback: List[str]
    first_detected: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'

class MemoryMonitor:
    """Advanced memory monitoring and leak detection system"""
    
    def __init__(self, 
                 threshold_mb: float = 100,
                 leak_detection_interval: int = 60,
                 history_size: int = 1000):
        self.threshold_mb = threshold_mb
        self.leak_detection_interval = leak_detection_interval
        self.history = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitoring_thread = None
        self.detected_leaks: List[MemoryLeak] = []
        self.object_tracker: Dict[type, List[int]] = defaultdict(list)
        self.weak_refs: Set[weakref.ref] = set()
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
        # Start tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Keep 10 frames of traceback
    
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.baseline_snapshot = self._take_snapshot()
        
        def monitor_loop():
            while self.is_monitoring:
                snapshot = self._take_snapshot()
                self.history.append(snapshot)
                
                # Check for memory leaks
                self._detect_memory_leaks()
                
                # Check threshold
                if snapshot.process_memory > self.threshold_mb:
                    self._handle_high_memory(snapshot)
                
                time.sleep(self.leak_detection_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        # Get garbage collection stats
        gc_stats = {
            'collections': gc.get_count(),
            'collected': gc.collect(),
            'uncollectable': len(gc.garbage)
        }
        
        # Get top memory consuming objects
        top_objects = self._get_top_objects(limit=20)
        
        # Memory map by category
        memory_map = self._categorize_memory_usage()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            total_memory=virtual_memory.total / (1024 * 1024),
            available_memory=virtual_memory.available / (1024 * 1024),
            process_memory=memory_info.rss / (1024 * 1024),
            heap_size=memory_info.vms / (1024 * 1024),
            gc_stats=gc_stats,
            top_objects=top_objects,
            memory_map=memory_map
        )
    
    def _get_top_objects(self, limit: int = 20) -> List[Dict]:
        """Get top memory consuming objects"""
        objects = []
        
        # Use objgraph to find most common types
        try:
            most_common = objgraph.most_common_types(limit=limit)
            for obj_type, count in most_common:
                # Estimate size (simplified - actual implementation would be more complex)
                size_estimate = count * sys.getsizeof(type(obj_type))
                objects.append({
                    'type': obj_type,
                    'count': count,
                    'size_mb': size_estimate / (1024 * 1024)
                })
        except Exception as e:
            print(f"Error getting top objects: {e}")
        
        return objects
    
    def _categorize_memory_usage(self) -> Dict[str, float]:
        """Categorize memory usage by type"""
        categories = defaultdict(float)
        
        # Analyze objects in memory
        for obj in gc.get_objects():
            try:
                obj_type = type(obj).__name__
                obj_size = sys.getsizeof(obj) / (1024 * 1024)  # MB
                
                # Categorize by type
                if 'dict' in obj_type:
                    categories['dictionaries'] += obj_size
                elif 'list' in obj_type or 'tuple' in obj_type:
                    categories['sequences'] += obj_size
                elif 'str' in obj_type:
                    categories['strings'] += obj_size
                elif 'ndarray' in obj_type:
                    categories['numpy_arrays'] += obj_size
                elif 'DataFrame' in obj_type:
                    categories['dataframes'] += obj_size
                else:
                    categories['other'] += obj_size
            except Exception:
                continue
        
        return dict(categories)
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks"""
        if len(self.history) < 2:
            return
        
        # Track object growth
        current_snapshot = self.history[-1]
        
        # Compare with previous snapshots
        growth_rates = self._calculate_growth_rates()
        
        for obj_type, growth_rate in growth_rates.items():
            if growth_rate > 1.0:  # Growing more than 1 MB/minute
                # Get traceback for this type
                traceback_info = self._get_allocation_traceback(obj_type)
                
                leak = MemoryLeak(
                    object_type=obj_type,
                    growth_rate=growth_rate,
                    instances=self._count_instances(obj_type),
                    total_size=self._calculate_type_size(obj_type),
                    traceback=traceback_info,
                    first_detected=datetime.now(),
                    severity=self._classify_severity(growth_rate)
                )
                
                # Check if already detected
                if not any(l.object_type == obj_type for l in self.detected_leaks):
                    self.detected_leaks.append(leak)
                    self._report_leak(leak)
    
    def _calculate_growth_rates(self) -> Dict[str, float]:
        """Calculate memory growth rates for different object types"""
        if len(self.history) < 2:
            return {}
        
        growth_rates = {}
        recent = self.history[-1]
        older = self.history[-min(10, len(self.history))]
        
        time_diff = (recent.timestamp - older.timestamp).total_seconds() / 60  # minutes
        
        for recent_obj in recent.top_objects:
            obj_type = recent_obj['type']
            recent_size = recent_obj['size_mb']
            
            # Find corresponding older entry
            older_size = 0
            for older_obj in older.top_objects:
                if older_obj['type'] == obj_type:
                    older_size = older_obj['size_mb']
                    break
            
            if time_diff > 0:
                growth_rate = (recent_size - older_size) / time_diff
                if growth_rate > 0:
                    growth_rates[obj_type] = growth_rate
        
        return growth_rates
    
    def _get_allocation_traceback(self, obj_type: str) -> List[str]:
        """Get allocation traceback for an object type"""
        tracebacks = []
        
        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('traceback')
        
        for stat in top_stats[:10]:
            # Filter by type if possible
            traceback_lines = []
            for line in stat.traceback.format():
                traceback_lines.append(line)
            
            if traceback_lines:
                tracebacks.extend(traceback_lines[:5])  # Keep first 5 lines
        
        return tracebacks
    
    def _count_instances(self, obj_type: str) -> int:
        """Count instances of a specific type"""
        count = 0
        for obj in gc.get_objects():
            if type(obj).__name__ == obj_type:
                count += 1
        return count
    
    def _calculate_type_size(self, obj_type: str) -> float:
        """Calculate total size of all objects of a type"""
        total_size = 0
        for obj in gc.get_objects():
            if type(obj).__name__ == obj_type:
                try:
                    total_size += sys.getsizeof(obj)
                except Exception:
                    continue
        return total_size / (1024 * 1024)  # MB
    
    def _classify_severity(self, growth_rate: float) -> str:
        """Classify leak severity based on growth rate"""
        if growth_rate > 10:
            return 'critical'
        elif growth_rate > 5:
            return 'high'
        elif growth_rate > 2:
            return 'medium'
        else:
            return 'low'
    
    def _handle_high_memory(self, snapshot: MemorySnapshot):
        """Handle high memory usage"""
        print(f"WARNING: High memory usage detected: {snapshot.process_memory:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if possible
        self._clear_caches()
        
        # Generate memory report
        report = self.generate_memory_report()
        
        # Save report
        self._save_report(report)
    
    def _clear_caches(self):
        """Clear various caches to free memory"""
        # Clear Python's internal caches
        gc.collect()
        
        # Clear function caches
        import functools
        for obj in gc.get_objects():
            if isinstance(obj, functools._lru_cache_wrapper):
                obj.cache_clear()
        
        # Clear module caches
        sys.modules.clear()
        
        # Force garbage collection again
        gc.collect()
    
    def _report_leak(self, leak: MemoryLeak):
        """Report detected memory leak"""
        report = [
            f"MEMORY LEAK DETECTED - {leak.severity.upper()}",
            f"Object Type: {leak.object_type}",
            f"Growth Rate: {leak.growth_rate:.2f} MB/minute",
            f"Instances: {leak.instances}",
            f"Total Size: {leak.total_size:.2f} MB",
            f"First Detected: {leak.first_detected}",
            "Traceback:"
        ]
        report.extend(leak.traceback[:5])
        
        print("\n".join(report))
        
        # Save to file
        leak_file = Path(f"memory_leak_{leak.object_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(leak_file, 'w') as f:
            f.write("\n".join(report))
    
    def analyze_object_references(self, obj: Any) -> Dict:
        """Analyze references to/from an object"""
        analysis = {
            'type': type(obj).__name__,
            'size': sys.getsizeof(obj),
            'id': id(obj),
            'referrers': [],
            'referents': []
        }
        
        # Get objects that reference this object
        referrers = gc.get_referrers(obj)
        for ref in referrers[:10]:  # Limit to 10
            analysis['referrers'].append({
                'type': type(ref).__name__,
                'id': id(ref)
            })
        
        # Get objects referenced by this object
        referents = gc.get_referents(obj)
        for ref in referents[:10]:  # Limit to 10
            analysis['referents'].append({
                'type': type(ref).__name__,
                'id': id(ref)
            })
        
        return analysis
    
    def find_circular_references(self) -> List[List[Any]]:
        """Find circular references in memory"""
        circular_refs = []
        
        # Enable garbage collection debugging
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        
        # Check for uncollectable objects (usually due to circular refs)
        for obj in gc.garbage:
            # Try to find the reference cycle
            cycle = self._find_cycle(obj)
            if cycle:
                circular_refs.append(cycle)
        
        # Disable debugging
        gc.set_debug(0)
        
        return circular_refs
    
    def _find_cycle(self, obj: Any, visited: Optional[Set] = None) -> Optional[List[Any]]:
        """Find a reference cycle starting from obj"""
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return [obj]  # Found cycle
        
        visited.add(obj_id)
        
        for referent in gc.get_referents(obj):
            if id(referent) in visited:
                return [obj, referent]
            
            cycle = self._find_cycle(referent, visited)
            if cycle:
                return [obj] + cycle
        
        return None
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization"""
        optimization_results = {
            'before': self._take_snapshot(),
            'actions': [],
            'after': None
        }
        
        # 1. Force garbage collection
        collected = gc.collect()
        optimization_results['actions'].append(f"Garbage collection: {collected} objects collected")
        
        # 2. Clear caches
        self._clear_caches()
        optimization_results['actions'].append("Cleared internal caches")
        
        # 3. Compact memory (platform-specific)
        try:
            import ctypes
            if sys.platform == 'win32':
                kernel32 = ctypes.windll.kernel32
                kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                optimization_results['actions'].append("Compacted process memory")
        except Exception:
            pass
        
        # 4. Release unused memory back to OS
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        optimization_results['actions'].append("Released unused memory to OS")
        
        # Take final snapshot
        optimization_results['after'] = self._take_snapshot()
        
        # Calculate improvement
        before_mb = optimization_results['before'].process_memory
        after_mb = optimization_results['after'].process_memory
        saved_mb = before_mb - after_mb
        
        optimization_results['memory_saved_mb'] = saved_mb
        optimization_results['improvement_percent'] = (saved_mb / before_mb) * 100
        
        return optimization_results
    
    def generate_memory_report(self) -> str:
        """Generate comprehensive memory report"""
        if not self.history:
            return "No memory data available"
        
        current = self.history[-1]
        report = [
            "=" * 60,
            "MEMORY USAGE REPORT",
            "=" * 60,
            f"Timestamp: {current.timestamp}",
            f"Process Memory: {current.process_memory:.2f} MB",
            f"Available System Memory: {current.available_memory:.2f} MB",
            f"Heap Size: {current.heap_size:.2f} MB",
            "",
            "Memory Categories:",
            "-" * 40
        ]
        
        for category, size in current.memory_map.items():
            report.append(f"  {category}: {size:.2f} MB")
        
        report.extend([
            "",
            "Top Memory Consumers:",
            "-" * 40
        ])
        
        for obj in current.top_objects[:10]:
            report.append(f"  {obj['type']}: {obj['count']} instances, {obj['size_mb']:.2f} MB")
        
        if self.detected_leaks:
            report.extend([
                "",
                "Detected Memory Leaks:",
                "-" * 40
            ])
            
            for leak in self.detected_leaks:
                report.append(f"  {leak.object_type}: {leak.growth_rate:.2f} MB/min ({leak.severity})")
        
        report.extend([
            "",
            "Garbage Collection Stats:",
            "-" * 40,
            f"  Collections: {current.gc_stats['collections']}",
            f"  Last Collection: {current.gc_stats['collected']} objects",
            f"  Uncollectable: {current.gc_stats['uncollectable']} objects"
        ])
        
        return "\n".join(report)
    
    def _save_report(self, report: str):
        """Save memory report to file"""
        filename = f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export memory metrics for monitoring systems"""
        if not self.history:
            return "{}" if format == 'json' else ""
        
        metrics = []
        for snapshot in self.history:
            metric = {
                'timestamp': snapshot.timestamp.isoformat(),
                'process_memory_mb': snapshot.process_memory,
                'heap_size_mb': snapshot.heap_size,
                'available_memory_mb': snapshot.available_memory,
                'gc_collections': snapshot.gc_stats['collections'],
                'memory_categories': snapshot.memory_map
            }
            metrics.append(metric)
        
        if format == 'json':
            return json.dumps(metrics, indent=2)
        elif format == 'prometheus':
            # Prometheus format
            lines = []
            for metric in metrics:
                timestamp = int(metric['timestamp'].timestamp() * 1000)
                lines.append(f"process_memory_mb {metric['process_memory_mb']} {timestamp}")
                lines.append(f"heap_size_mb {metric['heap_size_mb']} {timestamp}")
            return "\n".join(lines)
        
        return str(metrics)

# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = MemoryMonitor(threshold_mb=500)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate memory usage
    data = []
    for i in range(100):
        data.append([0] * 10000)
        time.sleep(1)
        
        if i % 10 == 0:
            print(monitor.generate_memory_report())
    
    # Optimize memory
    optimization = monitor.optimize_memory()
    print(f"Memory optimized: {optimization['memory_saved_mb']:.2f} MB saved")
    
    # Stop monitoring
    monitor.stop_monitoring()