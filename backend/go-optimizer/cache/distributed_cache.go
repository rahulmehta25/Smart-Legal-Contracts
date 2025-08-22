package cache

import (
	"context"
	"crypto/xxhash"
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/bits-and-blooms/bloom/v3"
	"github.com/dgraph-io/ristretto"
	"github.com/go-redis/redis/v8"
	"github.com/patrickmn/go-cache"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// CacheLevel represents different cache tiers
type CacheLevel int

const (
	L1Cache CacheLevel = iota // In-memory, fastest
	L2Cache                   // Distributed in-memory
	L3Cache                   // Persistent storage
)

// CacheMetrics tracks cache performance
type CacheMetrics struct {
	Hits        int64
	Misses      int64
	Evictions   int64
	LoadTime    int64 // nanoseconds
	TotalMemory int64 // bytes
}

// BloomFilter provides existence checks to prevent cache stampede
type BloomFilter struct {
	filter *bloom.BloomFilter
	mutex  sync.RWMutex
}

// NewBloomFilter creates a new bloom filter
func NewBloomFilter(expectedItems uint, falsePositiveRate float64) *BloomFilter {
	return &BloomFilter{
		filter: bloom.NewWithEstimates(expectedItems, falsePositiveRate),
	}
}

// Add adds an item to the bloom filter
func (bf *BloomFilter) Add(key string) {
	bf.mutex.Lock()
	defer bf.mutex.Unlock()
	bf.filter.Add([]byte(key))
}

// Test checks if an item might exist
func (bf *BloomFilter) Test(key string) bool {
	bf.mutex.RLock()
	defer bf.mutex.RUnlock()
	return bf.filter.Test([]byte(key))
}

// CacheEntry represents a cache entry with metadata
type CacheEntry struct {
	Value      interface{}
	CreatedAt  time.Time
	AccessedAt time.Time
	TTL        time.Duration
	Size       int64
	Compressed bool
	Version    uint64
}

// DistributedCache implements a high-performance multi-tier caching system
type DistributedCache struct {
	// L1: Local in-memory cache (fastest)
	l1Cache *ristretto.Cache

	// L2: Distributed in-memory cache
	l2Cache *cache.Cache

	// L3: Redis cluster (persistent)
	redisCluster *redis.ClusterClient

	// Bloom filter for existence checks
	bloomFilter *BloomFilter

	// Singleflight for preventing cache stampede
	singleflight singleflight.Group

	// Consistent hashing ring for distribution
	hashRing *ConsistentHashRing

	// Configuration
	config *CacheConfig

	// Metrics
	metrics *CacheMetrics

	// Logger
	logger *zap.Logger

	// Shutdown channel
	shutdown chan struct{}

	// Background workers
	workers sync.WaitGroup

	// Preloader for cache warming
	preloader *CachePreloader
}

// CacheConfig holds configuration for the distributed cache
type CacheConfig struct {
	L1MaxSize      int64         // Maximum size for L1 cache in bytes
	L2MaxSize      int64         // Maximum size for L2 cache in bytes
	DefaultTTL     time.Duration // Default TTL for cache entries
	CleanupInterval time.Duration // Cleanup interval for expired entries
	MaxKeyLength   int           // Maximum key length
	CompressThreshold int64      // Compress values larger than this threshold
	BloomFPRate    float64       // Bloom filter false positive rate
	WarmupEnabled  bool          // Enable cache warming
	ShardCount     int           // Number of shards for consistent hashing
}

// DefaultCacheConfig returns a default cache configuration
func DefaultCacheConfig() *CacheConfig {
	return &CacheConfig{
		L1MaxSize:         100 << 20, // 100MB
		L2MaxSize:         500 << 20, // 500MB
		DefaultTTL:        time.Hour,
		CleanupInterval:   time.Minute * 5,
		MaxKeyLength:      250,
		CompressThreshold: 1024, // 1KB
		BloomFPRate:       0.01, // 1% false positive rate
		WarmupEnabled:     true,
		ShardCount:        64,
	}
}

// NewDistributedCache creates a new distributed cache instance
func NewDistributedCache(config *CacheConfig, redisAddrs []string, logger *zap.Logger) (*DistributedCache, error) {
	if config == nil {
		config = DefaultCacheConfig()
	}

	// Initialize L1 cache (Ristretto)
	l1Cache, err := ristretto.NewCache(&ristretto.Config{
		NumCounters: 1e7,     // 10M counters
		MaxCost:     config.L1MaxSize,
		BufferItems: 64,
		Metrics:     true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create L1 cache: %w", err)
	}

	// Initialize L2 cache (in-memory with TTL)
	l2Cache := cache.New(config.DefaultTTL, config.CleanupInterval)

	// Initialize Redis cluster for L3
	var redisCluster *redis.ClusterClient
	if len(redisAddrs) > 0 {
		redisCluster = redis.NewClusterClient(&redis.ClusterOptions{
			Addrs:        redisAddrs,
			PoolSize:     50,
			MinIdleConns: 10,
			MaxRetries:   3,
			ReadTimeout:  time.Second * 3,
			WriteTimeout: time.Second * 3,
		})
	}

	// Initialize bloom filter
	bloomFilter := NewBloomFilter(1000000, config.BloomFPRate)

	// Initialize consistent hash ring
	hashRing := NewConsistentHashRing(config.ShardCount)

	dc := &DistributedCache{
		l1Cache:      l1Cache,
		l2Cache:      l2Cache,
		redisCluster: redisCluster,
		bloomFilter:  bloomFilter,
		hashRing:     hashRing,
		config:       config,
		metrics:      &CacheMetrics{},
		logger:       logger,
		shutdown:     make(chan struct{}),
		preloader:    NewCachePreloader(logger),
	}

	// Start background workers
	dc.startBackgroundWorkers()

	return dc, nil
}

// Get retrieves a value from the cache, checking L1 -> L2 -> L3
func (dc *DistributedCache) Get(ctx context.Context, key string) (interface{}, bool) {
	start := time.Now()
	defer func() {
		atomic.AddInt64(&dc.metrics.LoadTime, time.Since(start).Nanoseconds())
	}()

	// Validate key length
	if len(key) > dc.config.MaxKeyLength {
		dc.logger.Warn("Key too long", zap.String("key", key), zap.Int("length", len(key)))
		return nil, false
	}

	// Check bloom filter first to avoid unnecessary lookups
	if !dc.bloomFilter.Test(key) {
		atomic.AddInt64(&dc.metrics.Misses, 1)
		return nil, false
	}

	// Try L1 cache first
	if value, found := dc.l1Cache.Get(key); found {
		atomic.AddInt64(&dc.metrics.Hits, 1)
		dc.logger.Debug("L1 cache hit", zap.String("key", key))
		return value, true
	}

	// Try L2 cache
	if value, found := dc.l2Cache.Get(key); found {
		atomic.AddInt64(&dc.metrics.Hits, 1)
		dc.logger.Debug("L2 cache hit", zap.String("key", key))
		
		// Promote to L1
		dc.l1Cache.Set(key, value, 1)
		return value, true
	}

	// Try L3 cache (Redis) with singleflight to prevent stampede
	if dc.redisCluster != nil {
		result, err, _ := dc.singleflight.Do(key, func() (interface{}, error) {
			return dc.getFromRedis(ctx, key)
		})

		if err == nil && result != nil {
			atomic.AddInt64(&dc.metrics.Hits, 1)
			dc.logger.Debug("L3 cache hit", zap.String("key", key))
			
			// Promote to L2 and L1
			dc.l2Cache.Set(key, result, dc.config.DefaultTTL)
			dc.l1Cache.Set(key, result, 1)
			return result, true
		}
	}

	atomic.AddInt64(&dc.metrics.Misses, 1)
	return nil, false
}

// Set stores a value in all cache levels
func (dc *DistributedCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	if len(key) > dc.config.MaxKeyLength {
		return fmt.Errorf("key too long: %d > %d", len(key), dc.config.MaxKeyLength)
	}

	// Add to bloom filter
	dc.bloomFilter.Add(key)

	// Calculate value size
	size := int64(unsafe.Sizeof(value))

	// Set in L1 cache
	dc.l1Cache.Set(key, value, size)

	// Set in L2 cache
	if ttl == 0 {
		ttl = dc.config.DefaultTTL
	}
	dc.l2Cache.Set(key, value, ttl)

	// Set in L3 cache (Redis) if available
	if dc.redisCluster != nil {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
			defer cancel()
			
			if err := dc.setInRedis(ctx, key, value, ttl); err != nil {
				dc.logger.Error("Failed to set in Redis", zap.String("key", key), zap.Error(err))
			}
		}()
	}

	dc.logger.Debug("Cache set", zap.String("key", key), zap.Duration("ttl", ttl))
	return nil
}

// Delete removes a value from all cache levels
func (dc *DistributedCache) Delete(ctx context.Context, key string) error {
	// Delete from L1
	dc.l1Cache.Del(key)

	// Delete from L2
	dc.l2Cache.Delete(key)

	// Delete from L3 (Redis)
	if dc.redisCluster != nil {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
			defer cancel()
			
			if err := dc.redisCluster.Del(ctx, key).Err(); err != nil {
				dc.logger.Error("Failed to delete from Redis", zap.String("key", key), zap.Error(err))
			}
		}()
	}

	dc.logger.Debug("Cache delete", zap.String("key", key))
	return nil
}

// GetOrLoad retrieves a value from cache or loads it using the provided function
func (dc *DistributedCache) GetOrLoad(ctx context.Context, key string, loadFunc func() (interface{}, error), ttl time.Duration) (interface{}, error) {
	// Try to get from cache first
	if value, found := dc.Get(ctx, key); found {
		return value, nil
	}

	// Use singleflight to prevent multiple concurrent loads
	result, err, shared := dc.singleflight.Do(key, func() (interface{}, error) {
		value, err := loadFunc()
		if err != nil {
			return nil, err
		}

		// Set in cache
		if setErr := dc.Set(ctx, key, value, ttl); setErr != nil {
			dc.logger.Warn("Failed to set loaded value in cache", zap.String("key", key), zap.Error(setErr))
		}

		return value, nil
	})

	if shared {
		dc.logger.Debug("Shared singleflight result", zap.String("key", key))
	}

	return result, err
}

// GetMetrics returns current cache metrics
func (dc *DistributedCache) GetMetrics() CacheMetrics {
	return CacheMetrics{
		Hits:        atomic.LoadInt64(&dc.metrics.Hits),
		Misses:      atomic.LoadInt64(&dc.metrics.Misses),
		Evictions:   atomic.LoadInt64(&dc.metrics.Evictions),
		LoadTime:    atomic.LoadInt64(&dc.metrics.LoadTime),
		TotalMemory: atomic.LoadInt64(&dc.metrics.TotalMemory),
	}
}

// WarmCache preloads frequently accessed keys
func (dc *DistributedCache) WarmCache(ctx context.Context, keys []string, loadFunc func(key string) (interface{}, error)) error {
	if !dc.config.WarmupEnabled {
		return nil
	}

	return dc.preloader.Warmup(ctx, dc, keys, loadFunc)
}

// Close gracefully shuts down the cache
func (dc *DistributedCache) Close() error {
	close(dc.shutdown)
	dc.workers.Wait()

	if dc.l1Cache != nil {
		dc.l1Cache.Close()
	}

	if dc.redisCluster != nil {
		return dc.redisCluster.Close()
	}

	return nil
}

// getFromRedis retrieves a value from Redis
func (dc *DistributedCache) getFromRedis(ctx context.Context, key string) (interface{}, error) {
	result, err := dc.redisCluster.Get(ctx, key).Result()
	if err != nil {
		return nil, err
	}

	// Deserialize the value (implement based on your serialization strategy)
	return result, nil
}

// setInRedis stores a value in Redis
func (dc *DistributedCache) setInRedis(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// Serialize the value (implement based on your serialization strategy)
	serialized := fmt.Sprintf("%v", value)
	
	return dc.redisCluster.Set(ctx, key, serialized, ttl).Err()
}

// startBackgroundWorkers starts background maintenance tasks
func (dc *DistributedCache) startBackgroundWorkers() {
	// Metrics collector
	dc.workers.Add(1)
	go func() {
		defer dc.workers.Done()
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				dc.collectMetrics()
			case <-dc.shutdown:
				return
			}
		}
	}()

	// Cache cleanup worker
	dc.workers.Add(1)
	go func() {
		defer dc.workers.Done()
		ticker := time.NewTicker(dc.config.CleanupInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				dc.cleanup()
			case <-dc.shutdown:
				return
			}
		}
	}()
}

// collectMetrics updates cache metrics
func (dc *DistributedCache) collectMetrics() {
	if dc.l1Cache != nil {
		metrics := dc.l1Cache.Metrics
		atomic.StoreInt64(&dc.metrics.TotalMemory, int64(metrics.CostAdded()))
	}
}

// cleanup performs cache cleanup tasks
func (dc *DistributedCache) cleanup() {
	// L2 cache cleanup is handled automatically by go-cache
	// Additional cleanup logic can be added here
	dc.logger.Debug("Cache cleanup completed")
}

// ConsistentHashRing implements consistent hashing for cache distribution
type ConsistentHashRing struct {
	nodes    map[uint64]string
	sortedKeys []uint64
	mutex    sync.RWMutex
}

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing(replicas int) *ConsistentHashRing {
	ring := &ConsistentHashRing{
		nodes: make(map[uint64]string),
	}

	// Add virtual nodes for better distribution
	for i := 0; i < replicas; i++ {
		key := xxhash.Sum64String(fmt.Sprintf("node-%d", i))
		ring.nodes[key] = fmt.Sprintf("node-%d", i)
		ring.sortedKeys = append(ring.sortedKeys, key)
	}

	return ring
}

// GetNode returns the node responsible for a given key
func (chr *ConsistentHashRing) GetNode(key string) string {
	chr.mutex.RLock()
	defer chr.mutex.RUnlock()

	if len(chr.sortedKeys) == 0 {
		return ""
	}

	hash := xxhash.Sum64String(key)
	
	// Binary search for the appropriate node
	idx := 0
	for i, nodeKey := range chr.sortedKeys {
		if hash <= nodeKey {
			idx = i
			break
		}
	}

	return chr.nodes[chr.sortedKeys[idx]]
}

// CachePreloader handles cache warming strategies
type CachePreloader struct {
	logger *zap.Logger
}

// NewCachePreloader creates a new cache preloader
func NewCachePreloader(logger *zap.Logger) *CachePreloader {
	return &CachePreloader{
		logger: logger,
	}
}

// Warmup preloads cache with frequently accessed data
func (cp *CachePreloader) Warmup(ctx context.Context, cache *DistributedCache, keys []string, loadFunc func(key string) (interface{}, error)) error {
	semaphore := make(chan struct{}, 10) // Limit concurrent warmup operations
	
	for _, key := range keys {
		select {
		case semaphore <- struct{}{}:
			go func(k string) {
				defer func() { <-semaphore }()
				
				if _, found := cache.Get(ctx, k); !found {
					if value, err := loadFunc(k); err == nil {
						cache.Set(ctx, k, value, cache.config.DefaultTTL)
						cp.logger.Debug("Cache warmed", zap.String("key", k))
					} else {
						cp.logger.Warn("Failed to warm cache", zap.String("key", k), zap.Error(err))
					}
				}
			}(key)
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}