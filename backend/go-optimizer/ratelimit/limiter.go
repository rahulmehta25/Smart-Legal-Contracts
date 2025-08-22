package ratelimit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// LimitResult represents the result of a rate limit check
type LimitResult struct {
	Allowed      bool          `json:"allowed"`
	Remaining    int64         `json:"remaining"`
	Limit        int64         `json:"limit"`
	ResetTime    time.Time     `json:"reset_time"`
	RetryAfter   time.Duration `json:"retry_after,omitempty"`
	WindowStart  time.Time     `json:"window_start"`
	WindowEnd    time.Time     `json:"window_end"`
}

// LimiterType represents different rate limiting algorithms
type LimiterType int

const (
	TokenBucket LimiterType = iota
	SlidingWindow
	FixedWindow
	SlidingWindowLog
	AdaptiveBucket
)

// String returns string representation of limiter type
func (lt LimiterType) String() string {
	switch lt {
	case TokenBucket:
		return "token_bucket"
	case SlidingWindow:
		return "sliding_window"
	case FixedWindow:
		return "fixed_window"
	case SlidingWindowLog:
		return "sliding_window_log"
	case AdaptiveBucket:
		return "adaptive_bucket"
	default:
		return "unknown"
	}
}

// LimiterConfig holds configuration for rate limiting
type LimiterConfig struct {
	Type           LimiterType   `json:"type"`
	Rate           int64         `json:"rate"`           // Requests per window
	Window         time.Duration `json:"window"`         // Time window
	Burst          int64         `json:"burst"`          // Burst capacity
	EnableRedis    bool          `json:"enable_redis"`   // Use Redis for distributed limiting
	KeyPrefix      string        `json:"key_prefix"`     // Redis key prefix
	GCInterval     time.Duration `json:"gc_interval"`    // Garbage collection interval
	MaxMemoryUsage int64         `json:"max_memory_usage"` // Maximum memory usage in bytes
	
	// Adaptive limiting
	EnableAdaptive bool    `json:"enable_adaptive"`
	TargetLatency  time.Duration `json:"target_latency"`
	LatencyWindow  time.Duration `json:"latency_window"`
	AdaptionRate   float64 `json:"adaption_rate"`
	
	// Circuit breaker integration
	EnableCircuitBreaker bool `json:"enable_circuit_breaker"`
	FailureThreshold     int  `json:"failure_threshold"`
	RecoveryTimeout      time.Duration `json:"recovery_timeout"`
}

// DefaultLimiterConfig returns a default limiter configuration
func DefaultLimiterConfig() *LimiterConfig {
	return &LimiterConfig{
		Type:           SlidingWindow,
		Rate:           1000,
		Window:         time.Minute,
		Burst:          100,
		KeyPrefix:      "rate_limit:",
		GCInterval:     time.Minute * 5,
		MaxMemoryUsage: 100 << 20, // 100MB
		
		EnableAdaptive: false,
		TargetLatency:  time.Millisecond * 100,
		LatencyWindow:  time.Minute,
		AdaptionRate:   0.1,
		
		EnableCircuitBreaker: false,
		FailureThreshold:     10,
		RecoveryTimeout:      time.Minute,
	}
}

// WindowEntry represents an entry in a sliding window
type WindowEntry struct {
	Timestamp time.Time
	Count     int64
}

// SlidingWindowCounter implements sliding window rate limiting
type SlidingWindowCounter struct {
	entries   []WindowEntry
	mutex     sync.RWMutex
	rate      int64
	window    time.Duration
	subWindow time.Duration
	buckets   int
}

// NewSlidingWindowCounter creates a new sliding window counter
func NewSlidingWindowCounter(rate int64, window time.Duration) *SlidingWindowCounter {
	buckets := 60 // 60 sub-windows for precision
	return &SlidingWindowCounter{
		entries:   make([]WindowEntry, 0, buckets),
		rate:      rate,
		window:    window,
		subWindow: window / time.Duration(buckets),
		buckets:   buckets,
	}
}

// Check checks if a request is allowed
func (swc *SlidingWindowCounter) Check(now time.Time, count int64) bool {
	swc.mutex.Lock()
	defer swc.mutex.Unlock()

	// Clean old entries
	cutoff := now.Add(-swc.window)
	swc.cleanOldEntries(cutoff)

	// Calculate current usage
	currentUsage := swc.getCurrentUsage(now)

	// Check if request would exceed limit
	if currentUsage+count > swc.rate {
		return false
	}

	// Add new entry
	bucketTime := now.Truncate(swc.subWindow)
	
	// Find or create bucket
	for i := range swc.entries {
		if swc.entries[i].Timestamp.Equal(bucketTime) {
			swc.entries[i].Count += count
			return true
		}
	}

	// Create new bucket
	swc.entries = append(swc.entries, WindowEntry{
		Timestamp: bucketTime,
		Count:     count,
	})

	return true
}

// getCurrentUsage calculates current usage within the window
func (swc *SlidingWindowCounter) getCurrentUsage(now time.Time) int64 {
	cutoff := now.Add(-swc.window)
	usage := int64(0)

	for _, entry := range swc.entries {
		if entry.Timestamp.After(cutoff) {
			// Calculate weight based on time overlap
			weight := swc.calculateWeight(entry.Timestamp, now)
			usage += int64(float64(entry.Count) * weight)
		}
	}

	return usage
}

// calculateWeight calculates the weight of an entry based on time overlap
func (swc *SlidingWindowCounter) calculateWeight(entryTime, now time.Time) float64 {
	windowStart := now.Add(-swc.window)
	
	if entryTime.Before(windowStart) {
		return 0.0
	}
	
	if entryTime.Add(swc.subWindow).After(now) {
		// Partial overlap
		overlap := now.Sub(entryTime)
		return float64(overlap) / float64(swc.subWindow)
	}
	
	return 1.0
}

// cleanOldEntries removes entries older than cutoff
func (swc *SlidingWindowCounter) cleanOldEntries(cutoff time.Time) {
	n := 0
	for _, entry := range swc.entries {
		if entry.Timestamp.After(cutoff) {
			swc.entries[n] = entry
			n++
		}
	}
	swc.entries = swc.entries[:n]
}

// GetRemaining returns remaining capacity
func (swc *SlidingWindowCounter) GetRemaining(now time.Time) int64 {
	swc.mutex.RLock()
	defer swc.mutex.RUnlock()
	
	currentUsage := swc.getCurrentUsage(now)
	remaining := swc.rate - currentUsage
	if remaining < 0 {
		return 0
	}
	return remaining
}

// AdaptiveCounter implements adaptive rate limiting based on latency
type AdaptiveCounter struct {
	baseRate      int64
	currentRate   int64
	targetLatency time.Duration
	latencyWindow time.Duration
	adaptionRate  float64
	latencies     []time.Duration
	mutex         sync.RWMutex
	lastUpdate    time.Time
}

// NewAdaptiveCounter creates a new adaptive counter
func NewAdaptiveCounter(baseRate int64, targetLatency, latencyWindow time.Duration, adaptionRate float64) *AdaptiveCounter {
	return &AdaptiveCounter{
		baseRate:      baseRate,
		currentRate:   baseRate,
		targetLatency: targetLatency,
		latencyWindow: latencyWindow,
		adaptionRate:  adaptionRate,
		latencies:     make([]time.Duration, 0, 1000),
		lastUpdate:    time.Now(),
	}
}

// RecordLatency records a latency measurement
func (ac *AdaptiveCounter) RecordLatency(latency time.Duration) {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()
	
	ac.latencies = append(ac.latencies, latency)
	
	// Clean old latencies
	cutoff := time.Now().Add(-ac.latencyWindow)
	n := 0
	for i, lat := range ac.latencies {
		if time.Now().Add(-time.Duration(len(ac.latencies)-i)*time.Millisecond).After(cutoff) {
			ac.latencies[n] = lat
			n++
		}
	}
	ac.latencies = ac.latencies[:n]
}

// GetCurrentRate returns the current adaptive rate
func (ac *AdaptiveCounter) GetCurrentRate() int64 {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()
	
	if time.Since(ac.lastUpdate) < time.Second {
		return ac.currentRate
	}
	
	ac.lastUpdate = time.Now()
	
	if len(ac.latencies) == 0 {
		return ac.currentRate
	}
	
	// Calculate average latency
	total := time.Duration(0)
	for _, lat := range ac.latencies {
		total += lat
	}
	avgLatency := total / time.Duration(len(ac.latencies))
	
	// Adjust rate based on latency
	if avgLatency > ac.targetLatency {
		// Decrease rate
		adjustment := float64(ac.currentRate) * ac.adaptionRate
		ac.currentRate -= int64(adjustment)
		if ac.currentRate < ac.baseRate/10 {
			ac.currentRate = ac.baseRate / 10
		}
	} else if avgLatency < ac.targetLatency/2 {
		// Increase rate
		adjustment := float64(ac.currentRate) * ac.adaptionRate
		ac.currentRate += int64(adjustment)
		if ac.currentRate > ac.baseRate*2 {
			ac.currentRate = ac.baseRate * 2
		}
	}
	
	return ac.currentRate
}

// CircuitBreaker implements circuit breaker pattern for rate limiting
type CircuitBreaker struct {
	state            int32 // 0=closed, 1=open, 2=half-open
	failures         int64
	threshold        int64
	timeout          time.Duration
	lastFailureTime  time.Time
	mutex            sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(threshold int64, timeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		threshold: threshold,
		timeout:   timeout,
	}
}

// Allow checks if the circuit breaker allows requests
func (cb *CircuitBreaker) Allow() bool {
	cb.mutex.RLock()
	state := atomic.LoadInt32(&cb.state)
	cb.mutex.RUnlock()

	switch state {
	case 0: // Closed
		return true
	case 1: // Open
		cb.mutex.Lock()
		defer cb.mutex.Unlock()
		if time.Since(cb.lastFailureTime) > cb.timeout {
			atomic.StoreInt32(&cb.state, 2) // Half-open
			return true
		}
		return false
	case 2: // Half-open
		return true
	default:
		return false
	}
}

// RecordSuccess records a successful request
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	atomic.StoreInt64(&cb.failures, 0)
	atomic.StoreInt32(&cb.state, 0) // Closed
}

// RecordFailure records a failed request
func (cb *CircuitBreaker) RecordFailure() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	failures := atomic.AddInt64(&cb.failures, 1)
	cb.lastFailureTime = time.Now()
	
	if failures >= cb.threshold {
		atomic.StoreInt32(&cb.state, 1) // Open
	}
}

// RateLimiterMetrics tracks rate limiter performance
type RateLimiterMetrics struct {
	TotalRequests   int64     `json:"total_requests"`
	AllowedRequests int64     `json:"allowed_requests"`
	DeniedRequests  int64     `json:"denied_requests"`
	CurrentRate     int64     `json:"current_rate"`
	AverageLatency  time.Duration `json:"average_latency"`
	ErrorRate       float64   `json:"error_rate"`
	LastReset       time.Time `json:"last_reset"`
}

// DistributedRateLimiter implements a high-performance distributed rate limiter
type DistributedRateLimiter struct {
	config         *LimiterConfig
	redis          *redis.Client
	localCounters  map[string]*SlidingWindowCounter
	tokenBuckets   map[string]*rate.Limiter
	adaptiveCounters map[string]*AdaptiveCounter
	circuitBreakers map[string]*CircuitBreaker
	metrics        *RateLimiterMetrics
	mutex          sync.RWMutex
	gcTicker       *time.Ticker
	shutdown       chan struct{}
	logger         *zap.Logger
}

// NewDistributedRateLimiter creates a new distributed rate limiter
func NewDistributedRateLimiter(config *LimiterConfig, redisClient *redis.Client, logger *zap.Logger) *DistributedRateLimiter {
	if config == nil {
		config = DefaultLimiterConfig()
	}

	limiter := &DistributedRateLimiter{
		config:           config,
		redis:           redisClient,
		localCounters:   make(map[string]*SlidingWindowCounter),
		tokenBuckets:    make(map[string]*rate.Limiter),
		adaptiveCounters: make(map[string]*AdaptiveCounter),
		circuitBreakers: make(map[string]*CircuitBreaker),
		metrics:         &RateLimiterMetrics{LastReset: time.Now()},
		shutdown:        make(chan struct{}),
		logger:          logger,
	}

	// Start garbage collection
	limiter.startGC()

	return limiter
}

// Allow checks if a request is allowed for the given key
func (drl *DistributedRateLimiter) Allow(ctx context.Context, key string) (*LimitResult, error) {
	return drl.AllowN(ctx, key, 1)
}

// AllowN checks if N requests are allowed for the given key
func (drl *DistributedRateLimiter) AllowN(ctx context.Context, key string, n int64) (*LimitResult, error) {
	atomic.AddInt64(&drl.metrics.TotalRequests, n)

	// Check circuit breaker if enabled
	if drl.config.EnableCircuitBreaker {
		cb := drl.getCircuitBreaker(key)
		if !cb.Allow() {
			atomic.AddInt64(&drl.metrics.DeniedRequests, n)
			return &LimitResult{
				Allowed:     false,
				Remaining:   0,
				Limit:       drl.getCurrentRate(key),
				ResetTime:   time.Now().Add(drl.config.RecoveryTimeout),
				RetryAfter:  drl.config.RecoveryTimeout,
				WindowStart: time.Now().Add(-drl.config.Window),
				WindowEnd:   time.Now(),
			}, nil
		}
	}

	now := time.Now()
	
	switch drl.config.Type {
	case TokenBucket:
		return drl.allowTokenBucket(key, n, now)
	case SlidingWindow:
		return drl.allowSlidingWindow(ctx, key, n, now)
	case FixedWindow:
		return drl.allowFixedWindow(ctx, key, n, now)
	case SlidingWindowLog:
		return drl.allowSlidingWindowLog(ctx, key, n, now)
	case AdaptiveBucket:
		return drl.allowAdaptiveBucket(ctx, key, n, now)
	default:
		return nil, fmt.Errorf("unsupported limiter type: %s", drl.config.Type)
	}
}

// allowTokenBucket implements token bucket algorithm
func (drl *DistributedRateLimiter) allowTokenBucket(key string, n int64, now time.Time) (*LimitResult, error) {
	bucket := drl.getTokenBucket(key)
	
	if bucket.AllowN(now, int(n)) {
		atomic.AddInt64(&drl.metrics.AllowedRequests, n)
		return &LimitResult{
			Allowed:     true,
			Remaining:   int64(bucket.Tokens()),
			Limit:       drl.config.Rate,
			ResetTime:   now.Add(drl.config.Window),
			WindowStart: now.Add(-drl.config.Window),
			WindowEnd:   now,
		}, nil
	}

	atomic.AddInt64(&drl.metrics.DeniedRequests, n)
	reservation := bucket.Reserve()
	delay := reservation.Delay()
	reservation.Cancel()

	return &LimitResult{
		Allowed:     false,
		Remaining:   0,
		Limit:       drl.config.Rate,
		ResetTime:   now.Add(delay),
		RetryAfter:  delay,
		WindowStart: now.Add(-drl.config.Window),
		WindowEnd:   now,
	}, nil
}

// allowSlidingWindow implements sliding window algorithm
func (drl *DistributedRateLimiter) allowSlidingWindow(ctx context.Context, key string, n int64, now time.Time) (*LimitResult, error) {
	if drl.config.EnableRedis && drl.redis != nil {
		return drl.allowSlidingWindowRedis(ctx, key, n, now)
	}

	counter := drl.getSlidingWindowCounter(key)
	rate := drl.getCurrentRate(key)
	
	if counter.Check(now, n) {
		atomic.AddInt64(&drl.metrics.AllowedRequests, n)
		remaining := counter.GetRemaining(now)
		
		return &LimitResult{
			Allowed:     true,
			Remaining:   remaining,
			Limit:       rate,
			ResetTime:   now.Add(drl.config.Window),
			WindowStart: now.Add(-drl.config.Window),
			WindowEnd:   now,
		}, nil
	}

	atomic.AddInt64(&drl.metrics.DeniedRequests, n)
	return &LimitResult{
		Allowed:     false,
		Remaining:   0,
		Limit:       rate,
		ResetTime:   now.Add(drl.config.Window),
		RetryAfter:  drl.config.Window,
		WindowStart: now.Add(-drl.config.Window),
		WindowEnd:   now,
	}, nil
}

// allowSlidingWindowRedis implements distributed sliding window using Redis
func (drl *DistributedRateLimiter) allowSlidingWindowRedis(ctx context.Context, key string, n int64, now time.Time) (*LimitResult, error) {
	redisKey := drl.config.KeyPrefix + key
	windowStart := now.Add(-drl.config.Window)
	rate := drl.getCurrentRate(key)

	// Lua script for atomic sliding window check
	script := `
		local key = KEYS[1]
		local window_start = tonumber(ARGV[1])
		local now = tonumber(ARGV[2])
		local limit = tonumber(ARGV[3])
		local count = tonumber(ARGV[4])
		local window_size = tonumber(ARGV[5])

		-- Remove old entries
		redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

		-- Count current entries
		local current = redis.call('ZCARD', key)

		if current + count > limit then
			-- Rate limit exceeded
			local remaining = math.max(0, limit - current)
			local reset_time = now + window_size
			return {0, remaining, reset_time}
		else
			-- Add new entry
			redis.call('ZADD', key, now, now .. ':' .. count)
			redis.call('EXPIRE', key, math.ceil(window_size / 1000))
			
			local remaining = limit - current - count
			local reset_time = now + window_size
			return {1, remaining, reset_time}
		end
	`

	result, err := drl.redis.Eval(ctx, script, []string{redisKey},
		windowStart.UnixNano(),
		now.UnixNano(),
		rate,
		n,
		drl.config.Window.Nanoseconds(),
	).Result()

	if err != nil {
		drl.logger.Error("Redis sliding window error", zap.Error(err))
		// Fallback to local counter
		return drl.allowSlidingWindow(ctx, key, n, now)
	}

	res := result.([]interface{})
	allowed := res[0].(int64) == 1
	remaining := res[1].(int64)
	resetTime := time.Unix(0, res[2].(int64))

	if allowed {
		atomic.AddInt64(&drl.metrics.AllowedRequests, n)
	} else {
		atomic.AddInt64(&drl.metrics.DeniedRequests, n)
	}

	return &LimitResult{
		Allowed:     allowed,
		Remaining:   remaining,
		Limit:       rate,
		ResetTime:   resetTime,
		RetryAfter:  time.Until(resetTime),
		WindowStart: windowStart,
		WindowEnd:   now,
	}, nil
}

// allowFixedWindow implements fixed window algorithm
func (drl *DistributedRateLimiter) allowFixedWindow(ctx context.Context, key string, n int64, now time.Time) (*LimitResult, error) {
	windowStart := now.Truncate(drl.config.Window)
	windowEnd := windowStart.Add(drl.config.Window)
	windowKey := fmt.Sprintf("%s:%d", key, windowStart.Unix())

	if drl.config.EnableRedis && drl.redis != nil {
		return drl.allowFixedWindowRedis(ctx, windowKey, n, windowStart, windowEnd)
	}

	// Local implementation
	counter := drl.getSlidingWindowCounter(windowKey)
	rate := drl.getCurrentRate(key)
	
	if counter.Check(now, n) {
		atomic.AddInt64(&drl.metrics.AllowedRequests, n)
		remaining := counter.GetRemaining(now)
		
		return &LimitResult{
			Allowed:     true,
			Remaining:   remaining,
			Limit:       rate,
			ResetTime:   windowEnd,
			WindowStart: windowStart,
			WindowEnd:   windowEnd,
		}, nil
	}

	atomic.AddInt64(&drl.metrics.DeniedRequests, n)
	return &LimitResult{
		Allowed:     false,
		Remaining:   0,
		Limit:       rate,
		ResetTime:   windowEnd,
		RetryAfter:  time.Until(windowEnd),
		WindowStart: windowStart,
		WindowEnd:   windowEnd,
	}, nil
}

// allowFixedWindowRedis implements distributed fixed window using Redis
func (drl *DistributedRateLimiter) allowFixedWindowRedis(ctx context.Context, windowKey string, n int64, windowStart, windowEnd time.Time) (*LimitResult, error) {
	redisKey := drl.config.KeyPrefix + windowKey
	rate := drl.config.Rate

	// Lua script for atomic fixed window check
	script := `
		local key = KEYS[1]
		local limit = tonumber(ARGV[1])
		local count = tonumber(ARGV[2])
		local expire = tonumber(ARGV[3])

		local current = tonumber(redis.call('GET', key) or 0)

		if current + count > limit then
			return {0, math.max(0, limit - current)}
		else
			local new_value = redis.call('INCRBY', key, count)
			redis.call('EXPIRE', key, expire)
			return {1, limit - new_value}
		end
	`

	result, err := drl.redis.Eval(ctx, script, []string{redisKey},
		rate,
		n,
		int(drl.config.Window.Seconds())+1,
	).Result()

	if err != nil {
		drl.logger.Error("Redis fixed window error", zap.Error(err))
		// Fallback to local counter
		return drl.allowFixedWindow(ctx, windowKey, n, time.Now())
	}

	res := result.([]interface{})
	allowed := res[0].(int64) == 1
	remaining := res[1].(int64)

	if allowed {
		atomic.AddInt64(&drl.metrics.AllowedRequests, n)
	} else {
		atomic.AddInt64(&drl.metrics.DeniedRequests, n)
	}

	return &LimitResult{
		Allowed:     allowed,
		Remaining:   remaining,
		Limit:       rate,
		ResetTime:   windowEnd,
		RetryAfter:  time.Until(windowEnd),
		WindowStart: windowStart,
		WindowEnd:   windowEnd,
	}, nil
}

// allowSlidingWindowLog implements sliding window log algorithm
func (drl *DistributedRateLimiter) allowSlidingWindowLog(ctx context.Context, key string, n int64, now time.Time) (*LimitResult, error) {
	// This is memory-intensive but provides exact rate limiting
	// Similar to sliding window but tracks exact timestamps
	return drl.allowSlidingWindow(ctx, key, n, now)
}

// allowAdaptiveBucket implements adaptive rate limiting
func (drl *DistributedRateLimiter) allowAdaptiveBucket(ctx context.Context, key string, n int64, now time.Time) (*LimitResult, error) {
	adaptiveCounter := drl.getAdaptiveCounter(key)
	currentRate := adaptiveCounter.GetCurrentRate()
	
	// Use token bucket with adaptive rate
	bucket := rate.NewLimiter(rate.Limit(currentRate), int(drl.config.Burst))
	
	if bucket.AllowN(now, int(n)) {
		atomic.AddInt64(&drl.metrics.AllowedRequests, n)
		return &LimitResult{
			Allowed:     true,
			Remaining:   int64(bucket.Tokens()),
			Limit:       currentRate,
			ResetTime:   now.Add(drl.config.Window),
			WindowStart: now.Add(-drl.config.Window),
			WindowEnd:   now,
		}, nil
	}

	atomic.AddInt64(&drl.metrics.DeniedRequests, n)
	reservation := bucket.Reserve()
	delay := reservation.Delay()
	reservation.Cancel()

	return &LimitResult{
		Allowed:     false,
		Remaining:   0,
		Limit:       currentRate,
		ResetTime:   now.Add(delay),
		RetryAfter:  delay,
		WindowStart: now.Add(-drl.config.Window),
		WindowEnd:   now,
	}, nil
}

// RecordLatency records latency for adaptive rate limiting
func (drl *DistributedRateLimiter) RecordLatency(key string, latency time.Duration) {
	if drl.config.EnableAdaptive {
		adaptiveCounter := drl.getAdaptiveCounter(key)
		adaptiveCounter.RecordLatency(latency)
	}
}

// RecordSuccess records a successful request for circuit breaker
func (drl *DistributedRateLimiter) RecordSuccess(key string) {
	if drl.config.EnableCircuitBreaker {
		cb := drl.getCircuitBreaker(key)
		cb.RecordSuccess()
	}
}

// RecordFailure records a failed request for circuit breaker
func (drl *DistributedRateLimiter) RecordFailure(key string) {
	if drl.config.EnableCircuitBreaker {
		cb := drl.getCircuitBreaker(key)
		cb.RecordFailure()
	}
}

// GetMetrics returns current rate limiter metrics
func (drl *DistributedRateLimiter) GetMetrics() RateLimiterMetrics {
	total := atomic.LoadInt64(&drl.metrics.TotalRequests)
	denied := atomic.LoadInt64(&drl.metrics.DeniedRequests)
	
	errorRate := 0.0
	if total > 0 {
		errorRate = float64(denied) / float64(total)
	}

	return RateLimiterMetrics{
		TotalRequests:   total,
		AllowedRequests: atomic.LoadInt64(&drl.metrics.AllowedRequests),
		DeniedRequests:  denied,
		CurrentRate:     drl.config.Rate,
		AverageLatency:  drl.metrics.AverageLatency,
		ErrorRate:       errorRate,
		LastReset:       drl.metrics.LastReset,
	}
}

// Close shuts down the rate limiter
func (drl *DistributedRateLimiter) Close() error {
	close(drl.shutdown)
	if drl.gcTicker != nil {
		drl.gcTicker.Stop()
	}
	return nil
}

// Helper methods

func (drl *DistributedRateLimiter) getTokenBucket(key string) *rate.Limiter {
	drl.mutex.Lock()
	defer drl.mutex.Unlock()
	
	bucket, exists := drl.tokenBuckets[key]
	if !exists {
		bucket = rate.NewLimiter(rate.Limit(drl.config.Rate), int(drl.config.Burst))
		drl.tokenBuckets[key] = bucket
	}
	return bucket
}

func (drl *DistributedRateLimiter) getSlidingWindowCounter(key string) *SlidingWindowCounter {
	drl.mutex.Lock()
	defer drl.mutex.Unlock()
	
	counter, exists := drl.localCounters[key]
	if !exists {
		counter = NewSlidingWindowCounter(drl.config.Rate, drl.config.Window)
		drl.localCounters[key] = counter
	}
	return counter
}

func (drl *DistributedRateLimiter) getAdaptiveCounter(key string) *AdaptiveCounter {
	drl.mutex.Lock()
	defer drl.mutex.Unlock()
	
	counter, exists := drl.adaptiveCounters[key]
	if !exists {
		counter = NewAdaptiveCounter(
			drl.config.Rate,
			drl.config.TargetLatency,
			drl.config.LatencyWindow,
			drl.config.AdaptionRate,
		)
		drl.adaptiveCounters[key] = counter
	}
	return counter
}

func (drl *DistributedRateLimiter) getCircuitBreaker(key string) *CircuitBreaker {
	drl.mutex.Lock()
	defer drl.mutex.Unlock()
	
	cb, exists := drl.circuitBreakers[key]
	if !exists {
		cb = NewCircuitBreaker(
			int64(drl.config.FailureThreshold),
			drl.config.RecoveryTimeout,
		)
		drl.circuitBreakers[key] = cb
	}
	return cb
}

func (drl *DistributedRateLimiter) getCurrentRate(key string) int64 {
	if drl.config.EnableAdaptive {
		adaptiveCounter := drl.getAdaptiveCounter(key)
		return adaptiveCounter.GetCurrentRate()
	}
	return drl.config.Rate
}

func (drl *DistributedRateLimiter) startGC() {
	drl.gcTicker = time.NewTicker(drl.config.GCInterval)
	
	go func() {
		for {
			select {
			case <-drl.gcTicker.C:
				drl.performGC()
			case <-drl.shutdown:
				return
			}
		}
	}()
}

func (drl *DistributedRateLimiter) performGC() {
	drl.mutex.Lock()
	defer drl.mutex.Unlock()

	// Clean up old counters to prevent memory leaks
	cutoff := time.Now().Add(-drl.config.Window * 2)
	
	for key, counter := range drl.localCounters {
		counter.mutex.Lock()
		counter.cleanOldEntries(cutoff)
		if len(counter.entries) == 0 {
			delete(drl.localCounters, key)
		}
		counter.mutex.Unlock()
	}

	drl.logger.Debug("Rate limiter GC completed",
		zap.Int("local_counters", len(drl.localCounters)),
		zap.Int("token_buckets", len(drl.tokenBuckets)))
}

// HashKey creates a consistent hash of the key for distribution
func HashKey(key string) string {
	hash := sha256.Sum256([]byte(key))
	return hex.EncodeToString(hash[:])[:16]
}