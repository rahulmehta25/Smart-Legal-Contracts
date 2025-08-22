package benchmarks

import (
	"context"
	"crypto/rand"
	"fmt"
	"net"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"../cache"
	"../compress"
	"../loadbalancer"
	"../monitoring"
	"../pool"
	"../queue"
	"../ratelimit"

	"go.uber.org/zap"
)

// BenchmarkResult represents the result of a performance benchmark
type BenchmarkResult struct {
	Name              string        `json:"name"`
	Operations        int64         `json:"operations"`
	Duration          time.Duration `json:"duration"`
	ThroughputPerSec  float64       `json:"throughput_per_sec"`
	LatencyP50        time.Duration `json:"latency_p50"`
	LatencyP95        time.Duration `json:"latency_p95"`
	LatencyP99        time.Duration `json:"latency_p99"`
	MemoryAllocated   int64         `json:"memory_allocated"`
	GoroutinesCreated int           `json:"goroutines_created"`
	ErrorRate         float64       `json:"error_rate"`
}

// PerformanceSuite runs comprehensive performance tests
type PerformanceSuite struct {
	logger   *zap.Logger
	results  []BenchmarkResult
	baseline BenchmarkResult
}

// NewPerformanceSuite creates a new performance test suite
func NewPerformanceSuite() *PerformanceSuite {
	logger, _ := zap.NewProduction()
	return &PerformanceSuite{
		logger:  logger,
		results: make([]BenchmarkResult, 0),
	}
}

// RunAllBenchmarks runs all performance benchmarks
func (ps *PerformanceSuite) RunAllBenchmarks(b *testing.B) {
	ps.logger.Info("Starting comprehensive performance benchmarks")

	// Cache benchmarks
	ps.runCacheBenchmarks(b)

	// Queue benchmarks
	ps.runQueueBenchmarks(b)

	// Rate limiter benchmarks
	ps.runRateLimiterBenchmarks(b)

	// Load balancer benchmarks
	ps.runLoadBalancerBenchmarks(b)

	// Compression benchmarks
	ps.runCompressionBenchmarks(b)

	// Connection pool benchmarks
	ps.runConnectionPoolBenchmarks(b)

	// Generate performance report
	ps.generateReport()
}

// BenchmarkDistributedCache tests cache performance
func BenchmarkDistributedCache(b *testing.B) {
	logger, _ := zap.NewProduction()
	config := cache.DefaultCacheConfig()
	config.L1MaxSize = 10 << 20 // 10MB
	config.L2MaxSize = 50 << 20 // 50MB

	distributedCache, err := cache.NewDistributedCache(config, nil, logger)
	if err != nil {
		b.Fatal(err)
	}
	defer distributedCache.Close()

	ctx := context.Background()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("key-%d", i%1000)
			value := fmt.Sprintf("value-%d", i)
			
			// Set operation
			distributedCache.Set(ctx, key, value, time.Minute)
			
			// Get operation
			distributedCache.Get(ctx, key)
			
			i++
		}
	})
	
	metrics := distributedCache.GetMetrics()
	b.Logf("Cache metrics: Hits=%d, Misses=%d, Hit Rate=%.2f%%", 
		metrics.Hits, metrics.Misses, 
		float64(metrics.Hits)/float64(metrics.Hits+metrics.Misses)*100)
}

// BenchmarkTaskQueue tests queue performance
func BenchmarkTaskQueue(b *testing.B) {
	logger, _ := zap.NewProduction()
	config := queue.DefaultQueueConfig()
	config.MaxWorkers = runtime.NumCPU() * 2

	taskQueue, err := queue.NewTaskQueue(config, logger)
	if err != nil {
		b.Fatal(err)
	}
	defer taskQueue.Stop(time.Second * 5)

	if err := taskQueue.Start(); err != nil {
		b.Fatal(err)
	}

	var completedTasks int64
	taskFunc := func(ctx context.Context, args interface{}) (interface{}, error) {
		atomic.AddInt64(&completedTasks, 1)
		// Simulate work
		time.Sleep(time.Microsecond * 100)
		return "result", nil
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			task := queue.NewTask(
				fmt.Sprintf("task-%d", i),
				"benchmark-task",
				queue.PriorityNormal,
				taskFunc,
				i,
			)
			taskQueue.Submit(task)
			i++
		}
	})

	// Wait for all tasks to complete
	for atomic.LoadInt64(&completedTasks) < int64(b.N) {
		time.Sleep(time.Millisecond * 10)
	}

	metrics := taskQueue.GetMetrics()
	b.Logf("Queue metrics: Total=%d, Completed=%d, Failed=%d", 
		metrics.TotalTasks, metrics.CompletedTasks, metrics.FailedTasks)
}

// BenchmarkRateLimiter tests rate limiter performance
func BenchmarkRateLimiter(b *testing.B) {
	logger, _ := zap.NewProduction()
	config := ratelimit.DefaultLimiterConfig()
	config.Rate = 100000 // 100k requests per minute
	config.Type = ratelimit.SlidingWindow

	limiter := ratelimit.NewDistributedRateLimiter(config, nil, logger)
	defer limiter.Close()

	ctx := context.Background()
	var allowed, denied int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("user-%d", i%100)
			result, _ := limiter.Allow(ctx, key)
			
			if result.Allowed {
				atomic.AddInt64(&allowed, 1)
			} else {
				atomic.AddInt64(&denied, 1)
			}
			i++
		}
	})

	metrics := limiter.GetMetrics()
	b.Logf("Rate limiter metrics: Allowed=%d, Denied=%d, Error Rate=%.2f%%", 
		allowed, denied, metrics.ErrorRate*100)
}

// BenchmarkLoadBalancer tests load balancer performance
func BenchmarkLoadBalancer(b *testing.B) {
	logger, _ := zap.NewProduction()
	config := loadbalancer.DefaultLoadBalancerConfig()
	config.Algorithm = loadbalancer.RoundRobin

	lb := loadbalancer.NewSmartLoadBalancer(config, logger)
	defer lb.Close()

	// Add test backends
	for i := 0; i < 5; i++ {
		backend, _ := loadbalancer.NewBackend(
			fmt.Sprintf("backend-%d", i),
			fmt.Sprintf("http://backend-%d:8080", i),
			100,
		)
		lb.AddBackend(backend)
	}

	ctx := context.Background()
	var successful, failed int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			clientIP := fmt.Sprintf("192.168.1.%d", i%254+1)
			backend, err := lb.GetBackend(ctx, clientIP, "")
			
			if err != nil || backend == nil {
				atomic.AddInt64(&failed, 1)
			} else {
				atomic.AddInt64(&successful, 1)
			}
			i++
		}
	})

	metrics := lb.GetMetrics()
	b.Logf("Load balancer metrics: Successful=%d, Failed=%d, Total Requests=%d", 
		successful, failed, metrics.TotalRequests)
}

// BenchmarkCompressor tests compression performance
func BenchmarkCompressor(b *testing.B) {
	logger, _ := zap.NewProduction()
	config := compress.DefaultCompressorConfig()
	config.DefaultType = compress.Zstandard

	compressor, err := compress.NewRealTimeCompressor(config, logger)
	if err != nil {
		b.Fatal(err)
	}
	defer compressor.Close()

	// Generate test data
	testData := make([]byte, 1024) // 1KB test data
	rand.Read(testData)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Compress
			compressed, err := compressor.Compress(testData, compress.Zstandard, compress.LevelDefault)
			if err != nil {
				b.Error(err)
				continue
			}

			// Decompress
			_, err = compressor.Decompress(compressed)
			if err != nil {
				b.Error(err)
			}
		}
	})

	metrics := compressor.GetMetrics()
	b.Logf("Compression metrics: Ops=%d, Ratio=%.2f, Throughput=%.2f MB/s", 
		metrics.TotalOperations, metrics.CompressionRatio, metrics.ThroughputMBps)
}

// BenchmarkConnectionPool tests connection pool performance
func BenchmarkConnectionPool(b *testing.B) {
	logger, _ := zap.NewProduction()
	config := pool.DefaultPoolConfig()
	config.MaxConnections = 100

	// Mock dial function
	dialFunc := func() (net.Conn, error) {
		// Return a mock connection for testing
		return &MockConn{}, nil
	}

	connectionPool := pool.NewConnectionPool(config, dialFunc, logger)
	defer connectionPool.Close()

	ctx := context.Background()
	var successful, failed int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Get connection
			conn, err := connectionPool.Get(ctx)
			if err != nil {
				atomic.AddInt64(&failed, 1)
				continue
			}

			// Simulate work
			time.Sleep(time.Microsecond * 10)

			// Return connection
			connectionPool.Put(conn)
			atomic.AddInt64(&successful, 1)
		}
	})

	metrics := connectionPool.GetMetrics()
	b.Logf("Connection pool metrics: Successful=%d, Failed=%d, Active=%d, Idle=%d", 
		successful, failed, metrics.ActiveConnections, metrics.IdleConnections)
}

// Stress tests

// StressTestHighThroughput tests system under high load
func BenchmarkHighThroughput(b *testing.B) {
	ps := NewPerformanceSuite()
	
	// Setup all components
	logger, _ := zap.NewProduction()
	
	// Cache
	cacheConfig := cache.DefaultCacheConfig()
	distributedCache, _ := cache.NewDistributedCache(cacheConfig, nil, logger)
	defer distributedCache.Close()

	// Queue
	queueConfig := queue.DefaultQueueConfig()
	queueConfig.MaxWorkers = runtime.NumCPU() * 4
	taskQueue, _ := queue.NewTaskQueue(queueConfig, logger)
	taskQueue.Start()
	defer taskQueue.Stop(time.Second * 5)

	// Rate limiter
	limiterConfig := ratelimit.DefaultLimiterConfig()
	limiterConfig.Rate = 1000000 // 1M requests per minute
	rateLimiter := ratelimit.NewDistributedRateLimiter(limiterConfig, nil, logger)
	defer rateLimiter.Close()

	ctx := context.Background()
	var operations int64
	var errors int64

	numWorkers := runtime.NumCPU() * 8
	var wg sync.WaitGroup

	start := time.Now()

	// Simulate high-throughput workload
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for j := 0; j < b.N/numWorkers; j++ {
				// Rate limiting check
				key := fmt.Sprintf("user-%d", j%1000)
				result, err := rateLimiter.Allow(ctx, key)
				if err != nil || !result.Allowed {
					atomic.AddInt64(&errors, 1)
					continue
				}

				// Cache operation
				cacheKey := fmt.Sprintf("cache-key-%d-%d", workerID, j)
				value := fmt.Sprintf("value-%d", j)
				distributedCache.Set(ctx, cacheKey, value, time.Minute)
				distributedCache.Get(ctx, cacheKey)

				// Queue operation
				taskFunc := func(ctx context.Context, args interface{}) (interface{}, error) {
					return "processed", nil
				}
				task := queue.NewTask(
					fmt.Sprintf("task-%d-%d", workerID, j),
					"stress-test",
					queue.PriorityNormal,
					taskFunc,
					j,
				)
				taskQueue.Submit(task)

				atomic.AddInt64(&operations, 1)
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	throughput := float64(operations) / duration.Seconds()
	errorRate := float64(errors) / float64(operations) * 100

	b.Logf("High-throughput test: %d operations in %v (%.2f ops/sec, %.2f%% errors)",
		operations, duration, throughput, errorRate)

	ps.results = append(ps.results, BenchmarkResult{
		Name:             "HighThroughputStress",
		Operations:       operations,
		Duration:         duration,
		ThroughputPerSec: throughput,
		ErrorRate:        errorRate,
	})
}

// LoadTestConcurrency tests system under concurrent load
func BenchmarkConcurrency(b *testing.B) {
	logger, _ := zap.NewProduction()
	
	// Setup monitoring
	monitoringConfig := monitoring.DefaultProfilerConfig()
	metricsCollector := monitoring.NewMetricsCollector(monitoringConfig, logger)
	metricsCollector.Start()
	defer metricsCollector.Stop()

	concurrencyLevels := []int{10, 50, 100, 500, 1000}
	
	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrency-%d", concurrency), func(b *testing.B) {
			var operations int64
			var wg sync.WaitGroup

			start := time.Now()

			for i := 0; i < concurrency; i++ {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					
					for j := 0; j < b.N/concurrency; j++ {
						// Simulate work
						time.Sleep(time.Microsecond * 100)
						
						// Record request
						requestStart := time.Now()
						duration := time.Since(requestStart)
						metricsCollector.RecordRequest(duration, true)
						
						atomic.AddInt64(&operations, 1)
					}
				}(i)
			}

			wg.Wait()
			totalDuration := time.Since(start)

			throughput := float64(operations) / totalDuration.Seconds()
			
			// Get system metrics
			systemMetrics := metricsCollector.GetSystemMetrics()
			
			b.Logf("Concurrency-%d: %d operations, %.2f ops/sec, %d goroutines, %.2f MB memory",
				concurrency, operations, throughput, systemMetrics.NumGoroutines, systemMetrics.MemoryUsageMB)
		})
	}
}

// Memory and GC pressure tests
func BenchmarkMemoryPressure(b *testing.B) {
	logger, _ := zap.NewProduction()
	
	cacheConfig := cache.DefaultCacheConfig()
	cacheConfig.L1MaxSize = 100 << 20 // 100MB
	distributedCache, _ := cache.NewDistributedCache(cacheConfig, nil, logger)
	defer distributedCache.Close()

	ctx := context.Background()
	
	// Create large objects to induce GC pressure
	largeData := make([]byte, 10240) // 10KB per object
	rand.Read(largeData)

	var gcCount uint32
	var allocCount int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			// Allocate memory
			data := make([]byte, len(largeData))
			copy(data, largeData)
			atomic.AddInt64(&allocCount, 1)

			// Cache the data
			key := fmt.Sprintf("large-object-%d", i)
			distributedCache.Set(ctx, key, data, time.Minute)

			// Trigger GC occasionally
			if i%1000 == 0 {
				runtime.GC()
				atomic.AddUint32(&gcCount, 1)
			}

			i++
		}
	})

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	b.Logf("Memory pressure test: %d allocations, %d GCs, %.2f MB allocated, %d objects",
		allocCount, gcCount, float64(m.Alloc)/1024/1024, m.HeapObjects)
}

// Helper types and functions

// MockConn implements net.Conn for testing
type MockConn struct{}

func (mc *MockConn) Read(b []byte) (n int, err error)   { return len(b), nil }
func (mc *MockConn) Write(b []byte) (n int, err error) { return len(b), nil }
func (mc *MockConn) Close() error                      { return nil }
func (mc *MockConn) LocalAddr() net.Addr              { return &MockAddr{} }
func (mc *MockConn) RemoteAddr() net.Addr             { return &MockAddr{} }
func (mc *MockConn) SetDeadline(t time.Time) error    { return nil }
func (mc *MockConn) SetReadDeadline(t time.Time) error  { return nil }
func (mc *MockConn) SetWriteDeadline(t time.Time) error { return nil }

// MockAddr implements net.Addr
type MockAddr struct{}

func (ma *MockAddr) Network() string { return "tcp" }
func (ma *MockAddr) String() string  { return "127.0.0.1:8080" }

// Implementation methods for PerformanceSuite

func (ps *PerformanceSuite) runCacheBenchmarks(b *testing.B) {
	// Implement cache-specific benchmarks
	ps.logger.Info("Running cache benchmarks")
}

func (ps *PerformanceSuite) runQueueBenchmarks(b *testing.B) {
	// Implement queue-specific benchmarks
	ps.logger.Info("Running queue benchmarks")
}

func (ps *PerformanceSuite) runRateLimiterBenchmarks(b *testing.B) {
	// Implement rate limiter-specific benchmarks
	ps.logger.Info("Running rate limiter benchmarks")
}

func (ps *PerformanceSuite) runLoadBalancerBenchmarks(b *testing.B) {
	// Implement load balancer-specific benchmarks
	ps.logger.Info("Running load balancer benchmarks")
}

func (ps *PerformanceSuite) runCompressionBenchmarks(b *testing.B) {
	// Implement compression-specific benchmarks
	ps.logger.Info("Running compression benchmarks")
}

func (ps *PerformanceSuite) runConnectionPoolBenchmarks(b *testing.B) {
	// Implement connection pool-specific benchmarks
	ps.logger.Info("Running connection pool benchmarks")
}

func (ps *PerformanceSuite) generateReport() {
	ps.logger.Info("Generating performance report",
		zap.Int("total_benchmarks", len(ps.results)))
	
	// Generate detailed performance report
	for _, result := range ps.results {
		ps.logger.Info("Benchmark result",
			zap.String("name", result.Name),
			zap.Int64("operations", result.Operations),
			zap.Duration("duration", result.Duration),
			zap.Float64("throughput", result.ThroughputPerSec),
			zap.Float64("error_rate", result.ErrorRate))
	}
}

// Performance target validation
func ValidatePerformanceTargets(results []BenchmarkResult) bool {
	targets := map[string]float64{
		"DistributedCache":    50000,  // 50k ops/sec
		"TaskQueue":          10000,   // 10k tasks/sec
		"RateLimiter":        100000,  // 100k checks/sec
		"LoadBalancer":       50000,   // 50k selections/sec
		"Compressor":         1000,    // 1k compress/decompress/sec
		"ConnectionPool":     20000,   // 20k get/put/sec
	}

	allPassed := true
	for _, result := range results {
		if target, exists := targets[result.Name]; exists {
			if result.ThroughputPerSec < target {
				fmt.Printf("PERFORMANCE ISSUE: %s achieved %.2f ops/sec, target is %.2f ops/sec\n",
					result.Name, result.ThroughputPerSec, target)
				allPassed = false
			}
		}
	}

	return allPassed
}