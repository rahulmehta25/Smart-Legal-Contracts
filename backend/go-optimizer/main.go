package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"runtime"
	"sync"
	"time"

	"./cache"
	"./compress"
	"./loadbalancer"
	"./monitoring"
	"./pool"
	"./queue"
	"./ratelimit"

	"go.uber.org/zap"
)

// OptimizationLayer represents the complete high-performance optimization layer
type OptimizationLayer struct {
	Cache        *cache.DistributedCache
	Queue        *queue.TaskQueue
	RateLimiter  *ratelimit.DistributedRateLimiter
	LoadBalancer *loadbalancer.SmartLoadBalancer
	Compressor   *compress.RealTimeCompressor
	ConnPool     *pool.ConnectionPool
	Monitor      *monitoring.MetricsCollector
	Logger       *zap.Logger
}

// NewOptimizationLayer creates a new optimization layer with all components
func NewOptimizationLayer() (*OptimizationLayer, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}

	// Initialize distributed cache
	cacheConfig := cache.DefaultCacheConfig()
	cacheConfig.L1MaxSize = 100 << 20 // 100MB
	cacheConfig.L2MaxSize = 500 << 20 // 500MB
	distributedCache, err := cache.NewDistributedCache(cacheConfig, nil, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache: %w", err)
	}

	// Initialize task queue
	queueConfig := queue.DefaultQueueConfig()
	queueConfig.MaxWorkers = runtime.NumCPU() * 4
	taskQueue, err := queue.NewTaskQueue(queueConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create queue: %w", err)
	}

	// Initialize rate limiter
	limiterConfig := ratelimit.DefaultLimiterConfig()
	limiterConfig.Rate = 100000 // 100k requests per minute
	limiterConfig.Type = ratelimit.SlidingWindow
	rateLimiter := ratelimit.NewDistributedRateLimiter(limiterConfig, nil, logger)

	// Initialize load balancer
	lbConfig := loadbalancer.DefaultLoadBalancerConfig()
	lbConfig.Algorithm = loadbalancer.AdaptiveLoad
	loadBalancer := loadbalancer.NewSmartLoadBalancer(lbConfig, logger)

	// Add backend servers
	backends := []string{
		"http://backend-1:8080",
		"http://backend-2:8080",
		"http://backend-3:8080",
		"http://backend-4:8080",
		"http://backend-5:8080",
	}
	
	for i, backendURL := range backends {
		backend, err := loadbalancer.NewBackend(
			fmt.Sprintf("backend-%d", i+1),
			backendURL,
			100, // weight
		)
		if err != nil {
			logger.Warn("Failed to create backend", zap.String("url", backendURL), zap.Error(err))
			continue
		}
		loadBalancer.AddBackend(backend)
	}

	// Initialize compressor
	compressorConfig := compress.DefaultCompressorConfig()
	compressorConfig.DefaultType = compress.Zstandard
	compressorConfig.EnableAsyncMode = true
	compressor, err := compress.NewRealTimeCompressor(compressorConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create compressor: %w", err)
	}

	// Initialize connection pool
	connPoolConfig := pool.DefaultPoolConfig()
	connPoolConfig.MaxConnections = 1000
	dialFunc := func() (net.Conn, error) {
		return net.Dial("tcp", "localhost:8080")
	}
	connPool := pool.NewConnectionPool(connPoolConfig, dialFunc, logger)

	// Initialize monitoring
	monitoringConfig := monitoring.DefaultProfilerConfig()
	monitoringConfig.EnableProfiling = true
	monitoringConfig.EnablePrometheus = true
	monitor := monitoring.NewMetricsCollector(monitoringConfig, logger)

	ol := &OptimizationLayer{
		Cache:        distributedCache,
		Queue:        taskQueue,
		RateLimiter:  rateLimiter,
		LoadBalancer: loadBalancer,
		Compressor:   compressor,
		ConnPool:     connPool,
		Monitor:      monitor,
		Logger:       logger,
	}

	return ol, nil
}

// Start initializes and starts all optimization components
func (ol *OptimizationLayer) Start() error {
	ol.Logger.Info("Starting optimization layer")

	// Start task queue
	if err := ol.Queue.Start(); err != nil {
		return fmt.Errorf("failed to start task queue: %w", err)
	}

	// Start monitoring
	if err := ol.Monitor.Start(); err != nil {
		return fmt.Errorf("failed to start monitoring: %w", err)
	}

	ol.Logger.Info("Optimization layer started successfully")
	return nil
}

// Stop gracefully shuts down all components
func (ol *OptimizationLayer) Stop() error {
	ol.Logger.Info("Stopping optimization layer")

	var wg sync.WaitGroup
	var errors []error
	var errorsMutex sync.Mutex

	addError := func(err error) {
		if err != nil {
			errorsMutex.Lock()
			errors = append(errors, err)
			errorsMutex.Unlock()
		}
	}

	// Stop all components concurrently
	components := []func() error{
		func() error { return ol.Queue.Stop(time.Second * 10) },
		func() error { return ol.Monitor.Stop() },
		func() error { return ol.Cache.Close() },
		func() error { return ol.RateLimiter.Close() },
		func() error { return ol.LoadBalancer.Close() },
		func() error { return ol.Compressor.Close() },
		func() error { return ol.ConnPool.Close() },
	}

	for _, stopFunc := range components {
		wg.Add(1)
		go func(fn func() error) {
			defer wg.Done()
			addError(fn())
		}(stopFunc)
	}

	wg.Wait()

	if len(errors) > 0 {
		ol.Logger.Error("Errors during shutdown", zap.Int("error_count", len(errors)))
		return fmt.Errorf("shutdown completed with %d errors", len(errors))
	}

	ol.Logger.Info("Optimization layer stopped successfully")
	return nil
}

// ProcessRequest demonstrates a complete request processing pipeline
func (ol *OptimizationLayer) ProcessRequest(ctx context.Context, userID, requestData string) (string, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		ol.Monitor.RecordRequest(duration, true)
	}()

	// 1. Rate limiting
	limitResult, err := ol.RateLimiter.Allow(ctx, userID)
	if err != nil {
		return "", fmt.Errorf("rate limiter error: %w", err)
	}
	if !limitResult.Allowed {
		return "", fmt.Errorf("rate limit exceeded, try again in %v", limitResult.RetryAfter)
	}

	// 2. Cache check
	cacheKey := fmt.Sprintf("user:%s:request:%s", userID, requestData)
	if cachedResult, found := ol.Cache.Get(ctx, cacheKey); found {
		ol.Logger.Debug("Cache hit", zap.String("key", cacheKey))
		return cachedResult.(string), nil
	}

	// 3. Load balancing
	backend, err := ol.LoadBalancer.GetBackend(ctx, "client-ip", userID)
	if err != nil {
		return "", fmt.Errorf("load balancer error: %w", err)
	}

	// 4. Async task processing
	taskFunc := func(ctx context.Context, args interface{}) (interface{}, error) {
		// Simulate processing
		time.Sleep(time.Millisecond * 100)
		return fmt.Sprintf("processed-%s", args), nil
	}

	task := queue.NewTask(
		fmt.Sprintf("req-%s-%d", userID, time.Now().UnixNano()),
		"process-request",
		queue.PriorityNormal,
		taskFunc,
		requestData,
	)

	if err := ol.Queue.Submit(task); err != nil {
		return "", fmt.Errorf("failed to submit task: %w", err)
	}

	// 5. Data compression for storage
	result := fmt.Sprintf("processed-result-for-%s-via-%s", requestData, backend.ID)
	compressed, err := ol.Compressor.Compress(result, compress.Zstandard, compress.LevelDefault)
	if err != nil {
		ol.Logger.Warn("Compression failed", zap.Error(err))
	} else {
		ol.Logger.Debug("Data compressed",
			zap.Float64("ratio", compressed.CompressionRatio))
	}

	// 6. Cache the result
	if err := ol.Cache.Set(ctx, cacheKey, result, time.Hour); err != nil {
		ol.Logger.Warn("Failed to cache result", zap.Error(err))
	}

	return result, nil
}

// GetMetrics returns comprehensive system metrics
func (ol *OptimizationLayer) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"cache":         ol.Cache.GetMetrics(),
		"queue":         ol.Queue.GetMetrics(),
		"rate_limiter":  ol.RateLimiter.GetMetrics(),
		"load_balancer": ol.LoadBalancer.GetMetrics(),
		"compressor":    ol.Compressor.GetMetrics(),
		"conn_pool":     ol.ConnPool.GetMetrics(),
		"system":        ol.Monitor.GetSystemMetrics(),
		"application":   ol.Monitor.GetApplicationMetrics(),
	}
}

// HealthCheck performs a comprehensive health check
func (ol *OptimizationLayer) HealthCheck() map[string]string {
	health := map[string]string{
		"cache":         "healthy",
		"queue":         "healthy",
		"rate_limiter":  "healthy",
		"load_balancer": "healthy",
		"compressor":    "healthy",
		"conn_pool":     "healthy",
		"monitoring":    "healthy",
	}

	// Check queue metrics
	queueMetrics := ol.Queue.GetMetrics()
	if queueMetrics.ErrorRate > 0.1 { // 10% error rate threshold
		health["queue"] = "degraded"
	}

	// Check system metrics
	systemMetrics := ol.Monitor.GetSystemMetrics()
	if systemMetrics.CPUUsagePercent > 90 {
		health["system_cpu"] = "critical"
	}
	if systemMetrics.MemoryUsageMB > 2048 { // 2GB threshold
		health["system_memory"] = "warning"
	}

	// Check for goroutine leaks
	if systemMetrics.NumGoroutines > 10000 {
		health["goroutines"] = "warning"
	}

	return health
}

// Example HTTP handler demonstrating the optimization layer
func (ol *OptimizationLayer) HTTPHandler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		userID = "anonymous"
	}

	requestData := r.URL.Query().Get("data")
	if requestData == "" {
		requestData = "default"
	}

	result, err := ol.ProcessRequest(ctx, userID, requestData)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"result": "%s", "timestamp": "%s"}`, result, time.Now().Format(time.RFC3339))
}

// Example usage and demonstration
func main() {
	// Create optimization layer
	ol, err := NewOptimizationLayer()
	if err != nil {
		log.Fatal("Failed to create optimization layer:", err)
	}

	// Start all components
	if err := ol.Start(); err != nil {
		log.Fatal("Failed to start optimization layer:", err)
	}

	// Graceful shutdown
	defer func() {
		if err := ol.Stop(); err != nil {
			ol.Logger.Error("Shutdown error", zap.Error(err))
		}
	}()

	// Set up HTTP routes
	http.HandleFunc("/process", ol.HTTPHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		health := ol.HealthCheck()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"health": %v}`, health)
	})
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		metrics := ol.GetMetrics()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"metrics": %v}`, metrics)
	})

	// Demonstration: Process some requests
	go func() {
		time.Sleep(time.Second * 2) // Wait for startup
		
		ctx := context.Background()
		
		ol.Logger.Info("Starting demonstration workload")

		// Simulate high-load scenario
		var wg sync.WaitGroup
		numWorkers := 10
		requestsPerWorker := 100

		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				
				for j := 0; j < requestsPerWorker; j++ {
					userID := fmt.Sprintf("user-%d", j%50) // 50 different users
					requestData := fmt.Sprintf("data-%d-%d", workerID, j)
					
					result, err := ol.ProcessRequest(ctx, userID, requestData)
					if err != nil {
						ol.Logger.Error("Request failed",
							zap.String("user_id", userID),
							zap.Error(err))
					} else {
						ol.Logger.Debug("Request processed",
							zap.String("user_id", userID),
							zap.String("result", result))
					}
					
					// Small delay to simulate realistic load
					time.Sleep(time.Millisecond * 10)
				}
			}(i)
		}

		wg.Wait()
		
		// Print final metrics
		metrics := ol.GetMetrics()
		ol.Logger.Info("Demonstration completed",
			zap.Any("final_metrics", metrics))
	}()

	// Start HTTP server
	ol.Logger.Info("Starting HTTP server on :8080")
	ol.Logger.Info("Endpoints available:",
		zap.Strings("endpoints", []string{
			"GET /process?data=<data> - Process a request",
			"GET /health - Health check",
			"GET /metrics - System metrics",
			"GET /debug/pprof/ - Profiling (if enabled)",
		}))

	log.Fatal(http.ListenAndServe(":8080", nil))
}

// Performance test runner
func runPerformanceTest(ol *OptimizationLayer) {
	ol.Logger.Info("Running performance test targeting 100,000+ RPS with <10ms P99 latency")
	
	ctx := context.Background()
	duration := time.Minute * 1 // 1-minute test
	numWorkers := runtime.NumCPU() * 10
	
	var totalRequests int64
	var totalErrors int64
	var latencies []time.Duration
	var latencyMutex sync.Mutex
	
	startTime := time.Now()
	
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			requestID := 0
			for time.Since(startTime) < duration {
				start := time.Now()
				
				userID := fmt.Sprintf("user-%d", requestID%1000)
				requestData := fmt.Sprintf("perf-test-%d-%d", workerID, requestID)
				
				_, err := ol.ProcessRequest(ctx, userID, requestData)
				
				latency := time.Since(start)
				
				atomic.AddInt64(&totalRequests, 1)
				if err != nil {
					atomic.AddInt64(&totalErrors, 1)
				}
				
				// Sample latencies for percentile calculation
				if requestID%100 == 0 { // Sample every 100th request
					latencyMutex.Lock()
					latencies = append(latencies, latency)
					latencyMutex.Unlock()
				}
				
				requestID++
			}
		}(i)
	}
	
	wg.Wait()
	
	// Calculate results
	totalDuration := time.Since(startTime)
	rps := float64(totalRequests) / totalDuration.Seconds()
	errorRate := float64(totalErrors) / float64(totalRequests) * 100
	
	// Calculate latency percentiles
	if len(latencies) > 0 {
		// Sort latencies
		for i := 0; i < len(latencies)-1; i++ {
			for j := i + 1; j < len(latencies); j++ {
				if latencies[i] > latencies[j] {
					latencies[i], latencies[j] = latencies[j], latencies[i]
				}
			}
		}
		
		p50 := latencies[len(latencies)*50/100]
		p95 := latencies[len(latencies)*95/100]
		p99 := latencies[len(latencies)*99/100]
		
		ol.Logger.Info("Performance test results",
			zap.Int64("total_requests", totalRequests),
			zap.Float64("requests_per_second", rps),
			zap.Float64("error_rate_percent", errorRate),
			zap.Duration("p50_latency", p50),
			zap.Duration("p95_latency", p95),
			zap.Duration("p99_latency", p99),
			zap.Duration("test_duration", totalDuration))
		
		// Validate performance targets
		targetRPS := 100000.0
		targetP99 := time.Millisecond * 10
		
		if rps >= targetRPS {
			ol.Logger.Info("✅ RPS Target achieved", zap.Float64("target", targetRPS), zap.Float64("actual", rps))
		} else {
			ol.Logger.Warn("❌ RPS Target missed", zap.Float64("target", targetRPS), zap.Float64("actual", rps))
		}
		
		if p99 <= targetP99 {
			ol.Logger.Info("✅ P99 Latency Target achieved", zap.Duration("target", targetP99), zap.Duration("actual", p99))
		} else {
			ol.Logger.Warn("❌ P99 Latency Target missed", zap.Duration("target", targetP99), zap.Duration("actual", p99))
		}
	}
}