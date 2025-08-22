//! Cache performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_core::{LockFreeCache, HighPerformanceCache, CacheConfig, EvictionPolicy};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");
    
    let cache_sizes = [100, 1000, 10000, 100000];
    
    for &size in &cache_sizes {
        let config = CacheConfig {
            max_capacity: size,
            ..Default::default()
        };
        let cache = LockFreeCache::new(config);
        
        // Pre-populate cache
        for i in 0..size / 2 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i).into_bytes();
            let _ = cache.put(key, value);
        }
        
        group.throughput(Throughput::Elements(1));
        
        // Benchmark GET operations
        group.bench_with_input(
            BenchmarkId::new("get", size),
            &cache,
            |b, cache| {
                let mut counter = 0usize;
                b.iter(|| {
                    let key = format!("key_{}", counter % (size / 2));
                    counter += 1;
                    let _ = cache.get(black_box(&key));
                });
            },
        );
        
        // Benchmark PUT operations
        group.bench_with_input(
            BenchmarkId::new("put", size),
            &cache,
            |b, cache| {
                let mut counter = 0usize;
                b.iter(|| {
                    let key = format!("new_key_{}", counter);
                    let value = format!("new_value_{}", counter).into_bytes();
                    counter += 1;
                    let _ = cache.put(black_box(key), black_box(value));
                });
            },
        );
        
        // Benchmark mixed operations (70% get, 30% put)
        group.bench_with_input(
            BenchmarkId::new("mixed", size),
            &cache,
            |b, cache| {
                let mut counter = 0usize;
                b.iter(|| {
                    counter += 1;
                    if counter % 10 < 7 {
                        // GET operation
                        let key = format!("key_{}", counter % (size / 2));
                        let _ = cache.get(black_box(&key));
                    } else {
                        // PUT operation
                        let key = format!("mixed_key_{}", counter);
                        let value = format!("mixed_value_{}", counter).into_bytes();
                        let _ = cache.put(black_box(key), black_box(value));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different eviction policies
fn bench_eviction_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction_policies");
    
    let policies = [
        EvictionPolicy::LRU,
        EvictionPolicy::LFU,
        EvictionPolicy::FIFO,
        EvictionPolicy::Random,
    ];
    
    let cache_size = 1000;
    let operations = 5000; // More operations than cache size to trigger evictions
    
    for policy in policies.iter() {
        let config = CacheConfig {
            max_capacity: cache_size,
            eviction_policy: policy.clone(),
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("eviction", format!("{:?}", policy)),
            &config,
            |b, config| {
                b.iter_with_setup(
                    || LockFreeCache::new(config.clone()),
                    |cache| {
                        for i in 0..operations {
                            let key = format!("key_{}", i);
                            let value = format!("value_{}", i).into_bytes();
                            let _ = cache.put(black_box(key), black_box(value));
                            
                            // Occasionally access old keys to test LRU/LFU
                            if i % 10 == 0 && i > 0 {
                                let old_key = format!("key_{}", i - 100);
                                let _ = cache.get(black_box(&old_key));
                            }
                        }
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent access
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    group.measurement_time(Duration::from_secs(10));
    
    let thread_counts = [1, 2, 4, 8, 16];
    let operations_per_thread = 10000;
    
    for &thread_count in &thread_counts {
        let config = CacheConfig {
            max_capacity: 50000,
            ..Default::default()
        };
        let cache = Arc::new(LockFreeCache::new(config));
        
        group.throughput(Throughput::Elements((thread_count * operations_per_thread) as u64));
        
        // Read-heavy workload (90% reads)
        group.bench_with_input(
            BenchmarkId::new("read_heavy", thread_count),
            &thread_count,
            |b, &thread_count| {
                // Pre-populate cache
                for i in 0..10000 {
                    let key = format!("key_{}", i);
                    let value = format!("value_{}", i).into_bytes();
                    let _ = cache.put(key, value);
                }
                
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|thread_id| {
                            let cache = cache.clone();
                            thread::spawn(move || {
                                let mut counter = thread_id * operations_per_thread;
                                for _ in 0..operations_per_thread {
                                    counter += 1;
                                    if counter % 10 < 9 {
                                        // Read operation
                                        let key = format!("key_{}", counter % 10000);
                                        let _ = cache.get(&key);
                                    } else {
                                        // Write operation
                                        let key = format!("new_key_{}_{}", thread_id, counter);
                                        let value = format!("new_value_{}", counter).into_bytes();
                                        let _ = cache.put(key, value);
                                    }
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
        
        // Write-heavy workload (70% writes)
        group.bench_with_input(
            BenchmarkId::new("write_heavy", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let cache = Arc::new(LockFreeCache::new(CacheConfig {
                        max_capacity: 50000,
                        ..Default::default()
                    }));
                    
                    let handles: Vec<_> = (0..thread_count)
                        .map(|thread_id| {
                            let cache = cache.clone();
                            thread::spawn(move || {
                                let mut counter = thread_id * operations_per_thread;
                                for _ in 0..operations_per_thread {
                                    counter += 1;
                                    if counter % 10 < 3 {
                                        // Read operation
                                        let key = format!("key_{}", counter);
                                        let _ = cache.get(&key);
                                    } else {
                                        // Write operation
                                        let key = format!("key_{}_{}", thread_id, counter);
                                        let value = format!("value_{}", counter).into_bytes();
                                        let _ = cache.put(key, value);
                                    }
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different value sizes
fn bench_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_sizes");
    
    let value_sizes = [
        64,      // 64B
        1024,    // 1KB
        10240,   // 10KB
        102400,  // 100KB
        1048576, // 1MB
    ];
    
    for &size in &value_sizes {
        let config = CacheConfig {
            max_capacity: 1000,
            max_memory: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        };
        let cache = LockFreeCache::new(config);
        
        let test_value = vec![42u8; size];
        
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark PUT with different value sizes
        group.bench_with_input(
            BenchmarkId::new("put", format!("{}B", size)),
            &test_value,
            |b, value| {
                let mut counter = 0usize;
                b.iter(|| {
                    let key = format!("key_{}", counter);
                    counter += 1;
                    let _ = cache.put(black_box(key), black_box(value.clone()));
                });
            },
        );
        
        // Benchmark GET with different value sizes (pre-populate first)
        for i in 0..100 {
            let key = format!("get_key_{}", i);
            let _ = cache.put(key, test_value.clone());
        }
        
        group.bench_with_input(
            BenchmarkId::new("get", format!("{}B", size)),
            &100,
            |b, &key_count| {
                let mut counter = 0usize;
                b.iter(|| {
                    let key = format!("get_key_{}", counter % key_count);
                    counter += 1;
                    let _ = cache.get(black_box(&key));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache with TTL (Time To Live)
fn bench_ttl_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttl_operations");
    
    let config = CacheConfig {
        max_capacity: 10000,
        ttl: Duration::from_secs(1), // Short TTL for testing
        ..Default::default()
    };
    let cache = LockFreeCache::new(config);
    
    // Benchmark operations with TTL
    group.bench_function("with_ttl", |b| {
        b.iter(|| {
            // Add entries
            for i in 0..1000 {
                let key = format!("ttl_key_{}", i);
                let value = format!("ttl_value_{}", i).into_bytes();
                let _ = cache.put(black_box(key), black_box(value));
            }
            
            // Try to access them (some may have expired)
            for i in 0..1000 {
                let key = format!("ttl_key_{}", i);
                let _ = cache.get(black_box(&key));
            }
        });
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let memory_limits = [
        1024 * 1024,      // 1MB
        10 * 1024 * 1024, // 10MB
        100 * 1024 * 1024, // 100MB
    ];
    
    for &limit in &memory_limits {
        let config = CacheConfig {
            max_capacity: usize::MAX, // No capacity limit
            max_memory: limit,
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("memory_limit", format!("{}MB", limit / 1024 / 1024)),
            &config,
            |b, config| {
                b.iter_with_setup(
                    || LockFreeCache::new(config.clone()),
                    |cache| {
                        let mut counter = 0;
                        loop {
                            let key = format!("mem_key_{}", counter);
                            let value = vec![42u8; 1024]; // 1KB values
                            
                            match cache.put(key, value) {
                                Ok(_) => counter += 1,
                                Err(_) => break, // Memory limit reached
                            }
                            
                            if counter > 200000 {
                                break; // Safety limit
                            }
                        }
                        
                        black_box(counter);
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache statistics overhead
fn bench_stats_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_overhead");
    
    // With statistics enabled
    let config_with_stats = CacheConfig {
        max_capacity: 10000,
        stats_enabled: true,
        ..Default::default()
    };
    let cache_with_stats = LockFreeCache::new(config_with_stats);
    
    // With statistics disabled (simulated)
    let config_no_stats = CacheConfig {
        max_capacity: 10000,
        stats_enabled: false,
        ..Default::default()
    };
    let cache_no_stats = LockFreeCache::new(config_no_stats);
    
    group.bench_function("with_stats", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let key = format!("stats_key_{}", i);
                let value = format!("stats_value_{}", i).into_bytes();
                let _ = cache_with_stats.put(black_box(key), black_box(value));
            }
            
            for i in 0..1000 {
                let key = format!("stats_key_{}", i);
                let _ = cache_with_stats.get(black_box(&key));
            }
        });
    });
    
    group.bench_function("without_stats", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let key = format!("no_stats_key_{}", i);
                let value = format!("no_stats_value_{}", i).into_bytes();
                let _ = cache_no_stats.put(black_box(key), black_box(value));
            }
            
            for i in 0..1000 {
                let key = format!("no_stats_key_{}", i);
                let _ = cache_no_stats.get(black_box(&key));
            }
        });
    });
    
    group.finish();
}

/// Benchmark async cache operations
fn bench_async_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_cache");
    
    let rt = Runtime::new().unwrap();
    
    group.bench_function("async_operations", |b| {
        b.to_async(&rt).iter(|| async {
            use rust_core::AsyncCache;
            
            let config = CacheConfig {
                max_capacity: 10000,
                ..Default::default()
            };
            let cache = AsyncCache::new(config);
            
            // Async PUT operations
            for i in 0..1000 {
                let key = format!("async_key_{}", i);
                let value = format!("async_value_{}", i).into_bytes();
                let _ = cache.put(black_box(key), black_box(value)).await;
            }
            
            // Async GET operations
            for i in 0..1000 {
                let key = format!("async_key_{}", i);
                let _ = cache.get(black_box(&key)).await;
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_cache_operations,
    bench_eviction_policies,
    bench_concurrent_access,
    bench_value_sizes,
    bench_ttl_operations,
    bench_memory_usage,
    bench_stats_overhead,
    bench_async_cache
);
criterion_main!(benches);