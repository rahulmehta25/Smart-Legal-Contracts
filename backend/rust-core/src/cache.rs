//! Lock-free caching system with advanced features

use crate::{CoreError, CoreResult};
use dashmap::DashMap;
use moka::future::Cache as MokaCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock as AsyncRwLock;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_capacity: usize,
    pub ttl: Duration,
    pub max_memory: usize,
    pub eviction_policy: EvictionPolicy,
    pub enable_compression: bool,
    pub enable_persistence: bool,
    pub persistence_interval: Duration,
    pub stats_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 10_000,
            ttl: Duration::from_secs(3600), // 1 hour
            max_memory: 100 * 1024 * 1024, // 100MB
            eviction_policy: EvictionPolicy::LRU,
            enable_compression: true,
            enable_persistence: false,
            persistence_interval: Duration::from_secs(300), // 5 minutes
            stats_enabled: true,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,          // Least Recently Used
    LFU,          // Least Frequently Used
    FIFO,         // First In, First Out
    TTL,          // Time To Live
    Random,       // Random eviction
    None,         // No eviction (manual only)
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    last_accessed: AtomicU64, // nanoseconds since epoch
    access_count: AtomicUsize,
    size: usize,
    compressed: bool,
    ttl_override: Option<Duration>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub memory_usage: usize,
    pub hit_rate: f64,
    pub average_access_time: Duration,
    pub last_updated: SystemTime,
}

/// High-performance cache trait
pub trait HighPerformanceCache<K, V>: Send + Sync {
    /// Get value from cache
    fn get(&self, key: &K) -> Option<V>;
    
    /// Put value into cache
    fn put(&self, key: K, value: V) -> CoreResult<()>;
    
    /// Remove value from cache
    fn remove(&self, key: &K) -> Option<V>;
    
    /// Clear all entries
    fn clear(&self);
    
    /// Get cache statistics
    fn stats(&self) -> CacheStats;
    
    /// Check if key exists
    fn contains_key(&self, key: &K) -> bool;
    
    /// Get cache size
    fn size(&self) -> usize;
}

/// Lock-free cache implementation
pub struct LockFreeCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    data: DashMap<K, CacheEntry<V>>,
    config: CacheConfig,
    stats: Arc<CacheStatistics>,
    cleanup_running: Arc<AtomicU64>,
}

/// Advanced cache with async support
pub struct AsyncCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    inner: MokaCache<K, CacheEntry<V>>,
    config: CacheConfig,
    stats: Arc<CacheStatistics>,
}

/// Cache statistics tracking
#[derive(Debug)]
struct CacheStatistics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    total_access_time: AtomicU64,
    access_count: AtomicU64,
    memory_usage: AtomicUsize,
}

/// Multi-level cache with L1, L2, and persistent storage
pub struct MultiLevelCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    l1_cache: LockFreeCache<K, V>,          // Fast in-memory cache
    l2_cache: AsyncCache<K, V>,             // Larger async cache
    persistent_cache: Option<PersistentCache<K, V>>, // Disk-based cache
    promotion_threshold: usize,
    config: CacheConfig,
}

/// Persistent cache for disk storage
pub struct PersistentCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    storage_path: std::path::PathBuf,
    index: AsyncRwLock<HashMap<K, PersistentEntry>>,
    compression: CompressionEngine,
}

#[derive(Debug, Clone)]
struct PersistentEntry {
    offset: u64,
    size: usize,
    compressed_size: usize,
    created_at: SystemTime,
    last_accessed: SystemTime,
}

/// Compression engine for cache entries
struct CompressionEngine {
    enabled: bool,
    algorithm: CompressionAlgorithm,
}

#[derive(Debug, Clone)]
enum CompressionAlgorithm {
    Lz4,
    Zstd,
    Gzip,
}

impl<K, V> LockFreeCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new lock-free cache
    pub fn new(config: CacheConfig) -> Self {
        let data = DashMap::with_capacity(config.max_capacity);
        let stats = Arc::new(CacheStatistics::new());
        
        let cache = Self {
            data,
            config,
            stats,
            cleanup_running: Arc::new(AtomicU64::new(0)),
        };
        
        // Start cleanup task
        cache.start_cleanup_task();
        
        cache
    }

    /// Start background cleanup task
    fn start_cleanup_task(&self) {
        let data = self.data.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let cleanup_running = self.cleanup_running.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                if cleanup_running.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    Self::cleanup_expired_entries(&data, &config, &stats);
                    cleanup_running.store(0, Ordering::Release);
                }
            }
        });
    }

    /// Cleanup expired entries
    fn cleanup_expired_entries(
        data: &DashMap<K, CacheEntry<V>>,
        config: &CacheConfig,
        stats: &Arc<CacheStatistics>,
    ) {
        let now = Instant::now();
        let mut evicted = 0;

        data.retain(|_key, entry| {
            let ttl = entry.ttl_override.unwrap_or(config.ttl);
            let is_expired = now.duration_since(entry.created_at) > ttl;
            
            if is_expired {
                stats.evictions.fetch_add(1, Ordering::Relaxed);
                stats.memory_usage.fetch_sub(entry.size, Ordering::Relaxed);
                evicted += 1;
                false
            } else {
                true
            }
        });

        if evicted > 0 {
            tracing::debug!("Evicted {} expired cache entries", evicted);
        }
    }

    /// Evict entries based on policy
    fn evict_by_policy(&self) {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::LFU => self.evict_lfu(),
            EvictionPolicy::FIFO => self.evict_fifo(),
            EvictionPolicy::Random => self.evict_random(),
            _ => {}
        }
    }

    /// Evict least recently used entries
    fn evict_lru(&self) {
        let target_size = (self.config.max_capacity * 80) / 100; // Evict to 80%
        
        if self.data.len() <= target_size {
            return;
        }

        let mut entries: Vec<_> = self.data
            .iter()
            .map(|entry| {
                let last_accessed = entry.value().last_accessed.load(Ordering::Relaxed);
                (entry.key().clone(), last_accessed)
            })
            .collect();

        entries.sort_by_key(|(_, last_accessed)| *last_accessed);
        
        let to_remove = self.data.len() - target_size;
        for (key, _) in entries.into_iter().take(to_remove) {
            if let Some((_k, entry)) = self.data.remove(&key) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats.memory_usage.fetch_sub(entry.size, Ordering::Relaxed);
            }
        }
    }

    /// Evict least frequently used entries
    fn evict_lfu(&self) {
        let target_size = (self.config.max_capacity * 80) / 100;
        
        if self.data.len() <= target_size {
            return;
        }

        let mut entries: Vec<_> = self.data
            .iter()
            .map(|entry| {
                let access_count = entry.value().access_count.load(Ordering::Relaxed);
                (entry.key().clone(), access_count)
            })
            .collect();

        entries.sort_by_key(|(_, count)| *count);
        
        let to_remove = self.data.len() - target_size;
        for (key, _) in entries.into_iter().take(to_remove) {
            if let Some((_k, entry)) = self.data.remove(&key) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats.memory_usage.fetch_sub(entry.size, Ordering::Relaxed);
            }
        }
    }

    /// Evict first in, first out
    fn evict_fifo(&self) {
        let target_size = (self.config.max_capacity * 80) / 100;
        
        if self.data.len() <= target_size {
            return;
        }

        let mut entries: Vec<_> = self.data
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().created_at))
            .collect();

        entries.sort_by_key(|(_, created_at)| *created_at);
        
        let to_remove = self.data.len() - target_size;
        for (key, _) in entries.into_iter().take(to_remove) {
            if let Some((_k, entry)) = self.data.remove(&key) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats.memory_usage.fetch_sub(entry.size, Ordering::Relaxed);
            }
        }
    }

    /// Evict random entries
    fn evict_random(&self) {
        let target_size = (self.config.max_capacity * 80) / 100;
        let to_remove = self.data.len().saturating_sub(target_size);
        
        let keys: Vec<_> = self.data.iter().take(to_remove).map(|entry| entry.key().clone()).collect();
        
        for key in keys {
            if let Some((_k, entry)) = self.data.remove(&key) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats.memory_usage.fetch_sub(entry.size, Ordering::Relaxed);
            }
        }
    }
}

impl<K, V> HighPerformanceCache<K, V> for LockFreeCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        let start_time = Instant::now();
        
        let result = self.data.get(key).map(|entry| {
            // Update access statistics
            let now_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            
            entry.last_accessed.store(now_nanos, Ordering::Relaxed);
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            
            entry.value.clone()
        });

        // Update statistics
        if result.is_some() {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
        }
        
        let access_time = start_time.elapsed().as_nanos() as u64;
        self.stats.total_access_time.fetch_add(access_time, Ordering::Relaxed);
        self.stats.access_count.fetch_add(1, Ordering::Relaxed);

        result
    }

    fn put(&self, key: K, value: V) -> CoreResult<()> {
        // Check capacity and evict if necessary
        if self.data.len() >= self.config.max_capacity {
            self.evict_by_policy();
        }

        // Estimate size (simplified)
        let size = std::mem::size_of::<V>() + std::mem::size_of::<K>();
        
        // Check memory usage
        let current_memory = self.stats.memory_usage.load(Ordering::Relaxed);
        if current_memory + size > self.config.max_memory {
            return Err(CoreError::CacheError("Memory limit exceeded".to_string()));
        }

        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_accessed: AtomicU64::new(now_nanos),
            access_count: AtomicUsize::new(0),
            size,
            compressed: false,
            ttl_override: None,
        };

        self.data.insert(key, entry);
        self.stats.memory_usage.fetch_add(size, Ordering::Relaxed);

        Ok(())
    }

    fn remove(&self, key: &K) -> Option<V> {
        self.data.remove(key).map(|(_, entry)| {
            self.stats.memory_usage.fetch_sub(entry.size, Ordering::Relaxed);
            entry.value
        })
    }

    fn clear(&self) {
        self.data.clear();
        self.stats.memory_usage.store(0, Ordering::Relaxed);
    }

    fn stats(&self) -> CacheStats {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total_access_time = self.stats.total_access_time.load(Ordering::Relaxed);
        let access_count = self.stats.access_count.load(Ordering::Relaxed);

        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        let avg_access_time = if access_count > 0 {
            Duration::from_nanos(total_access_time / access_count)
        } else {
            Duration::default()
        };

        CacheStats {
            hits,
            misses,
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            size: self.data.len(),
            memory_usage: self.stats.memory_usage.load(Ordering::Relaxed),
            hit_rate,
            average_access_time: avg_access_time,
            last_updated: SystemTime::now(),
        }
    }

    fn contains_key(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<K, V> AsyncCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new async cache
    pub fn new(config: CacheConfig) -> Self {
        let cache = MokaCache::builder()
            .max_capacity(config.max_capacity as u64)
            .time_to_live(config.ttl)
            .build();

        Self {
            inner: cache,
            config,
            stats: Arc::new(CacheStatistics::new()),
        }
    }

    /// Get value asynchronously
    pub async fn get(&self, key: &K) -> Option<V> {
        let start_time = Instant::now();
        
        let result = self.inner.get(key).await.map(|entry| {
            // Update access statistics
            let now_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            
            entry.last_accessed.store(now_nanos, Ordering::Relaxed);
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            
            entry.value.clone()
        });

        // Update statistics
        if result.is_some() {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
        }
        
        let access_time = start_time.elapsed().as_nanos() as u64;
        self.stats.total_access_time.fetch_add(access_time, Ordering::Relaxed);
        self.stats.access_count.fetch_add(1, Ordering::Relaxed);

        result
    }

    /// Put value asynchronously
    pub async fn put(&self, key: K, value: V) -> CoreResult<()> {
        let size = std::mem::size_of::<V>() + std::mem::size_of::<K>();
        
        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_accessed: AtomicU64::new(now_nanos),
            access_count: AtomicUsize::new(0),
            size,
            compressed: false,
            ttl_override: None,
        };

        self.inner.insert(key, entry).await;
        self.stats.memory_usage.fetch_add(size, Ordering::Relaxed);

        Ok(())
    }
}

impl<K, V> MultiLevelCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>,
    V: Clone + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>,
{
    /// Create new multi-level cache
    pub fn new(config: CacheConfig) -> CoreResult<Self> {
        let l1_config = CacheConfig {
            max_capacity: config.max_capacity / 10, // 10% for L1
            ..config.clone()
        };
        
        let l2_config = CacheConfig {
            max_capacity: config.max_capacity - l1_config.max_capacity,
            ..config.clone()
        };

        let l1_cache = LockFreeCache::new(l1_config);
        let l2_cache = AsyncCache::new(l2_config);
        
        let persistent_cache = if config.enable_persistence {
            Some(PersistentCache::new(std::path::PathBuf::from("cache_storage"))?)
        } else {
            None
        };

        Ok(Self {
            l1_cache,
            l2_cache,
            persistent_cache,
            promotion_threshold: 3,
            config,
        })
    }

    /// Get value from multi-level cache
    pub async fn get(&self, key: &K) -> Option<V> {
        // Try L1 first
        if let Some(value) = self.l1_cache.get(key) {
            return Some(value);
        }

        // Try L2
        if let Some(value) = self.l2_cache.get(key).await {
            // Promote to L1 if accessed frequently
            if let Some(entry) = self.l2_cache.inner.get(key).await {
                let access_count = entry.access_count.load(Ordering::Relaxed);
                if access_count >= self.promotion_threshold {
                    let _ = self.l1_cache.put(key.clone(), value.clone());
                }
            }
            return Some(value);
        }

        // Try persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            if let Ok(Some(value)) = persistent.get(key).await {
                // Load into L2
                let _ = self.l2_cache.put(key.clone(), value.clone()).await;
                return Some(value);
            }
        }

        None
    }

    /// Put value into multi-level cache
    pub async fn put(&self, key: K, value: V) -> CoreResult<()> {
        // Always put in L2
        self.l2_cache.put(key.clone(), value.clone()).await?;

        // Put in persistent cache if enabled
        if let Some(ref persistent) = self.persistent_cache {
            persistent.put(key, value).await?;
        }

        Ok(())
    }
}

impl<K, V> PersistentCache<K, V>
where
    K: Clone + std::hash::Hash + Eq + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>,
    V: Clone + Send + Sync + 'static + Serialize + for<'de> Deserialize<'de>,
{
    /// Create new persistent cache
    pub fn new(storage_path: std::path::PathBuf) -> CoreResult<Self> {
        std::fs::create_dir_all(&storage_path)?;
        
        Ok(Self {
            storage_path,
            index: AsyncRwLock::new(HashMap::new()),
            compression: CompressionEngine::new(true, CompressionAlgorithm::Lz4),
        })
    }

    /// Get value from persistent storage
    pub async fn get(&self, key: &K) -> CoreResult<Option<V>> {
        let index = self.index.read().await;
        
        if let Some(entry) = index.get(key) {
            let file_path = self.storage_path.join(format!("{:?}.cache", key));
            let data = tokio::fs::read(&file_path).await?;
            
            let decompressed = if entry.compressed_size != entry.size {
                self.compression.decompress(&data)?
            } else {
                data
            };
            
            let value: V = bincode::deserialize(&decompressed)
                .map_err(|e| CoreError::CacheError(e.to_string()))?;
            
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Put value into persistent storage
    pub async fn put(&self, key: K, value: V) -> CoreResult<()> {
        let serialized = bincode::serialize(&value)
            .map_err(|e| CoreError::CacheError(e.to_string()))?;
        
        let (data, compressed_size) = if self.compression.enabled {
            let compressed = self.compression.compress(&serialized)?;
            (compressed.clone(), compressed.len())
        } else {
            (serialized.clone(), serialized.len())
        };
        
        let file_path = self.storage_path.join(format!("{:?}.cache", key));
        tokio::fs::write(&file_path, &data).await?;
        
        let entry = PersistentEntry {
            offset: 0, // Simplified - would track actual file offsets
            size: serialized.len(),
            compressed_size,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
        };
        
        let mut index = self.index.write().await;
        index.insert(key, entry);
        
        Ok(())
    }
}

impl CompressionEngine {
    fn new(enabled: bool, algorithm: CompressionAlgorithm) -> Self {
        Self { enabled, algorithm }
    }

    fn compress(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        if !self.enabled {
            return Ok(data.to_vec());
        }

        match self.algorithm {
            CompressionAlgorithm::Lz4 => {
                lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(12)), false)
                    .map_err(|e| CoreError::CacheError(format!("LZ4 compression error: {}", e)))
            }
            _ => Ok(data.to_vec()), // Simplified - would implement other algorithms
        }
    }

    fn decompress(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        if !self.enabled {
            return Ok(data.to_vec());
        }

        match self.algorithm {
            CompressionAlgorithm::Lz4 => {
                lz4::block::decompress(data, None)
                    .map_err(|e| CoreError::CacheError(format!("LZ4 decompression error: {}", e)))
            }
            _ => Ok(data.to_vec()),
        }
    }
}

impl CacheStatistics {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            total_access_time: AtomicU64::new(0),
            access_count: AtomicU64::new(0),
            memory_usage: AtomicUsize::new(0),
        }
    }
}

/// Initialize global cache instance
static GLOBAL_CACHE: once_cell::sync::OnceCell<LockFreeCache<String, Vec<u8>>> = once_cell::sync::OnceCell::new();

/// Initialize global cache
pub fn init_global_cache(capacity: usize) -> CoreResult<()> {
    let config = CacheConfig {
        max_capacity: capacity,
        ..Default::default()
    };
    
    let cache = LockFreeCache::new(config);
    GLOBAL_CACHE.set(cache).map_err(|_| CoreError::CacheError("Global cache already initialized".to_string()))?;
    
    Ok(())
}

/// Get global cache instance
pub fn global_cache() -> &'static LockFreeCache<String, Vec<u8>> {
    GLOBAL_CACHE.get().expect("Global cache not initialized")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lock_free_cache_basic() {
        let cache = LockFreeCache::new(CacheConfig::default());
        
        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(cache.get(&"nonexistent".to_string()), None);
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_rate > 0.0);
    }
    
    #[tokio::test]
    async fn test_async_cache() {
        let cache = AsyncCache::new(CacheConfig::default());
        
        cache.put("key1".to_string(), "value1".to_string()).await.unwrap();
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));
        assert_eq!(cache.get(&"nonexistent".to_string()).await, None);
    }
    
    #[tokio::test]
    async fn test_multi_level_cache() {
        let cache = MultiLevelCache::new(CacheConfig::default()).unwrap();
        
        cache.put("key1".to_string(), "value1".to_string()).await.unwrap();
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));
    }
    
    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_capacity: 2,
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        
        let cache = LockFreeCache::new(config);
        
        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        cache.put("key2".to_string(), "value2".to_string()).unwrap();
        cache.put("key3".to_string(), "value3".to_string()).unwrap(); // Should trigger eviction
        
        assert!(cache.size() <= 2);
    }
}