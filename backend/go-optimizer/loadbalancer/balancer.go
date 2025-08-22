package loadbalancer

import (
	"context"
	"crypto/tls"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
)

// BackendState represents the state of a backend server
type BackendState int

const (
	StateHealthy BackendState = iota
	StateUnhealthy
	StateDraining
	StateMaintenance
)

// String returns string representation of backend state
func (bs BackendState) String() string {
	switch bs {
	case StateHealthy:
		return "healthy"
	case StateUnhealthy:
		return "unhealthy"
	case StateDraining:
		return "draining"
	case StateMaintenance:
		return "maintenance"
	default:
		return "unknown"
	}
}

// LoadBalancingAlgorithm represents different load balancing algorithms
type LoadBalancingAlgorithm int

const (
	RoundRobin LoadBalancingAlgorithm = iota
	WeightedRoundRobin
	LeastConnections
	WeightedLeastConnections
	IPHash
	ConsistentHash
	RandomChoice
	PowerOfTwoChoices
	AdaptiveLoad
)

// String returns string representation of load balancing algorithm
func (lba LoadBalancingAlgorithm) String() string {
	switch lba {
	case RoundRobin:
		return "round_robin"
	case WeightedRoundRobin:
		return "weighted_round_robin"
	case LeastConnections:
		return "least_connections"
	case WeightedLeastConnections:
		return "weighted_least_connections"
	case IPHash:
		return "ip_hash"
	case ConsistentHash:
		return "consistent_hash"
	case RandomChoice:
		return "random_choice"
	case PowerOfTwoChoices:
		return "power_of_two_choices"
	case AdaptiveLoad:
		return "adaptive_load"
	default:
		return "unknown"
	}
}

// BackendMetrics tracks performance metrics for a backend
type BackendMetrics struct {
	TotalRequests    int64         `json:"total_requests"`
	SuccessfulRequests int64       `json:"successful_requests"`
	FailedRequests   int64         `json:"failed_requests"`
	ActiveConnections int64        `json:"active_connections"`
	AverageLatency   time.Duration `json:"average_latency"`
	P95Latency       time.Duration `json:"p95_latency"`
	P99Latency       time.Duration `json:"p99_latency"`
	ErrorRate        float64       `json:"error_rate"`
	Throughput       float64       `json:"throughput"` // requests per second
	CPUUsage         float64       `json:"cpu_usage"`
	MemoryUsage      float64       `json:"memory_usage"`
	LastHealthCheck  time.Time     `json:"last_health_check"`
	HealthCheckCount int64         `json:"health_check_count"`
	ResponseTimes    []time.Duration `json:"-"` // For percentile calculations
}

// HealthCheckConfig holds configuration for health checks
type HealthCheckConfig struct {
	Interval        time.Duration `json:"interval"`
	Timeout         time.Duration `json:"timeout"`
	HealthyThreshold int          `json:"healthy_threshold"`
	UnhealthyThreshold int        `json:"unhealthy_threshold"`
	Path            string        `json:"path"`
	ExpectedStatus  int           `json:"expected_status"`
	ExpectedBody    string        `json:"expected_body,omitempty"`
	Method          string        `json:"method"`
	Headers         map[string]string `json:"headers,omitempty"`
}

// DefaultHealthCheckConfig returns a default health check configuration
func DefaultHealthCheckConfig() *HealthCheckConfig {
	return &HealthCheckConfig{
		Interval:           time.Second * 30,
		Timeout:            time.Second * 5,
		HealthyThreshold:   2,
		UnhealthyThreshold: 3,
		Path:               "/health",
		ExpectedStatus:     200,
		Method:             "GET",
	}
}

// Backend represents a backend server
type Backend struct {
	ID              string                 `json:"id"`
	URL             *url.URL              `json:"url"`
	Weight          int                   `json:"weight"`
	MaxConnections  int                   `json:"max_connections"`
	State           BackendState          `json:"state"`
	Metrics         *BackendMetrics       `json:"metrics"`
	HealthCheck     *HealthCheckConfig    `json:"health_check"`
	Tags            map[string]string     `json:"tags,omitempty"`
	
	// Internal fields
	mutex           sync.RWMutex
	healthyChecks   int32
	unhealthyChecks int32
	client          *http.Client
	lastSelected    time.Time
	loadScore       float64 // Adaptive load score
}

// NewBackend creates a new backend
func NewBackend(id string, rawURL string, weight int) (*Backend, error) {
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}

	client := &http.Client{
		Timeout: time.Second * 30,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   time.Second * 10,
				KeepAlive: time.Second * 30,
			}).DialContext,
			TLSClientConfig: &tls.Config{InsecureSkipVerify: false},
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     time.Second * 90,
		},
	}

	return &Backend{
		ID:             id,
		URL:            parsedURL,
		Weight:         weight,
		MaxConnections: 1000,
		State:          StateHealthy,
		Metrics:        &BackendMetrics{ResponseTimes: make([]time.Duration, 0, 1000)},
		HealthCheck:    DefaultHealthCheckConfig(),
		Tags:           make(map[string]string),
		client:         client,
		loadScore:      1.0,
	}, nil
}

// IsHealthy returns true if the backend is healthy
func (b *Backend) IsHealthy() bool {
	b.mutex.RLock()
	defer b.mutex.RUnlock()
	return b.State == StateHealthy
}

// IsAvailable returns true if the backend can accept requests
func (b *Backend) IsAvailable() bool {
	b.mutex.RLock()
	defer b.mutex.RUnlock()
	return b.State == StateHealthy && atomic.LoadInt64(&b.Metrics.ActiveConnections) < int64(b.MaxConnections)
}

// GetState returns the current backend state
func (b *Backend) GetState() BackendState {
	b.mutex.RLock()
	defer b.mutex.RUnlock()
	return b.State
}

// SetState updates the backend state
func (b *Backend) SetState(state BackendState) {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	b.State = state
}

// RecordRequest records request metrics
func (b *Backend) RecordRequest(duration time.Duration, success bool) {
	atomic.AddInt64(&b.Metrics.TotalRequests, 1)
	
	if success {
		atomic.AddInt64(&b.Metrics.SuccessfulRequests, 1)
	} else {
		atomic.AddInt64(&b.Metrics.FailedRequests, 1)
	}

	// Record response time for latency calculations
	b.mutex.Lock()
	b.Metrics.ResponseTimes = append(b.Metrics.ResponseTimes, duration)
	
	// Keep only recent response times (last 1000)
	if len(b.Metrics.ResponseTimes) > 1000 {
		b.Metrics.ResponseTimes = b.Metrics.ResponseTimes[500:]
	}
	b.mutex.Unlock()

	// Update average latency
	b.updateLatencyMetrics()
}

// updateLatencyMetrics calculates latency percentiles
func (b *Backend) updateLatencyMetrics() {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	if len(b.Metrics.ResponseTimes) == 0 {
		return
	}

	// Sort response times for percentile calculation
	times := make([]time.Duration, len(b.Metrics.ResponseTimes))
	copy(times, b.Metrics.ResponseTimes)
	sort.Slice(times, func(i, j int) bool {
		return times[i] < times[j]
	})

	// Calculate average
	total := time.Duration(0)
	for _, t := range times {
		total += t
	}
	b.Metrics.AverageLatency = total / time.Duration(len(times))

	// Calculate percentiles
	if len(times) > 0 {
		p95Index := int(float64(len(times)) * 0.95)
		p99Index := int(float64(len(times)) * 0.99)
		
		if p95Index >= len(times) {
			p95Index = len(times) - 1
		}
		if p99Index >= len(times) {
			p99Index = len(times) - 1
		}
		
		b.Metrics.P95Latency = times[p95Index]
		b.Metrics.P99Latency = times[p99Index]
	}

	// Calculate error rate
	total := atomic.LoadInt64(&b.Metrics.TotalRequests)
	failed := atomic.LoadInt64(&b.Metrics.FailedRequests)
	if total > 0 {
		b.Metrics.ErrorRate = float64(failed) / float64(total)
	}
}

// GetLoadScore returns the current load score for adaptive load balancing
func (b *Backend) GetLoadScore() float64 {
	b.mutex.RLock()
	defer b.mutex.RUnlock()
	return b.loadScore
}

// UpdateLoadScore updates the load score based on current metrics
func (b *Backend) UpdateLoadScore() {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	// Base score on multiple factors
	score := 1.0

	// Factor in error rate (higher error rate = lower score)
	if b.Metrics.ErrorRate > 0 {
		score *= (1.0 - b.Metrics.ErrorRate)
	}

	// Factor in latency (higher latency = lower score)
	if b.Metrics.AverageLatency > 0 {
		latencyFactor := 1.0 / (1.0 + float64(b.Metrics.AverageLatency.Milliseconds())/100.0)
		score *= latencyFactor
	}

	// Factor in connection load
	connectionLoad := float64(atomic.LoadInt64(&b.Metrics.ActiveConnections)) / float64(b.MaxConnections)
	score *= (1.0 - connectionLoad)

	// Ensure score is within bounds
	if score < 0.01 {
		score = 0.01
	}
	if score > 1.0 {
		score = 1.0
	}

	b.loadScore = score
}

// LoadBalancerConfig holds configuration for the load balancer
type LoadBalancerConfig struct {
	Algorithm              LoadBalancingAlgorithm `json:"algorithm"`
	HealthCheckConfig      *HealthCheckConfig     `json:"health_check"`
	EnableStickySession    bool                   `json:"enable_sticky_session"`
	SessionCookieName      string                 `json:"session_cookie_name"`
	SessionTimeout         time.Duration          `json:"session_timeout"`
	MaxRetries            int                    `json:"max_retries"`
	RetryBackoff          time.Duration          `json:"retry_backoff"`
	CircuitBreakerEnabled bool                   `json:"circuit_breaker_enabled"`
	CircuitBreakerThreshold int                  `json:"circuit_breaker_threshold"`
	MetricsInterval       time.Duration          `json:"metrics_interval"`
}

// DefaultLoadBalancerConfig returns a default load balancer configuration
func DefaultLoadBalancerConfig() *LoadBalancerConfig {
	return &LoadBalancerConfig{
		Algorithm:               RoundRobin,
		HealthCheckConfig:       DefaultHealthCheckConfig(),
		EnableStickySession:     false,
		SessionCookieName:       "lb_session",
		SessionTimeout:          time.Hour,
		MaxRetries:             3,
		RetryBackoff:           time.Millisecond * 100,
		CircuitBreakerEnabled:  true,
		CircuitBreakerThreshold: 5,
		MetricsInterval:        time.Second * 30,
	}
}

// LoadBalancerMetrics tracks overall load balancer performance
type LoadBalancerMetrics struct {
	TotalRequests     int64            `json:"total_requests"`
	SuccessfulRequests int64           `json:"successful_requests"`
	FailedRequests    int64            `json:"failed_requests"`
	AverageLatency    time.Duration    `json:"average_latency"`
	BackendMetrics    map[string]*BackendMetrics `json:"backend_metrics"`
	RequestsPerSecond float64          `json:"requests_per_second"`
	ErrorRate         float64          `json:"error_rate"`
}

// SmartLoadBalancer implements a high-performance smart load balancer
type SmartLoadBalancer struct {
	config         *LoadBalancerConfig
	backends       map[string]*Backend
	healthyBackends []*Backend
	mutex          sync.RWMutex
	roundRobinIndex int32
	consistentHash  *ConsistentHashRing
	stickySessions map[string]string // session -> backend ID
	sessionMutex   sync.RWMutex
	metrics        *LoadBalancerMetrics
	logger         *zap.Logger
	shutdown       chan struct{}
	workers        sync.WaitGroup
	semaphore      *semaphore.Weighted
}

// NewSmartLoadBalancer creates a new smart load balancer
func NewSmartLoadBalancer(config *LoadBalancerConfig, logger *zap.Logger) *SmartLoadBalancer {
	if config == nil {
		config = DefaultLoadBalancerConfig()
	}

	lb := &SmartLoadBalancer{
		config:          config,
		backends:        make(map[string]*Backend),
		healthyBackends: make([]*Backend, 0),
		consistentHash:  NewConsistentHashRing(100),
		stickySessions:  make(map[string]string),
		metrics:         &LoadBalancerMetrics{BackendMetrics: make(map[string]*BackendMetrics)},
		logger:          logger,
		shutdown:        make(chan struct{}),
		semaphore:       semaphore.NewWeighted(10000), // Limit concurrent health checks
	}

	// Start background workers
	lb.startBackgroundWorkers()

	return lb
}

// AddBackend adds a backend to the load balancer
func (lb *SmartLoadBalancer) AddBackend(backend *Backend) error {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	if _, exists := lb.backends[backend.ID]; exists {
		return fmt.Errorf("backend %s already exists", backend.ID)
	}

	lb.backends[backend.ID] = backend
	
	if backend.IsHealthy() {
		lb.healthyBackends = append(lb.healthyBackends, backend)
	}

	// Add to consistent hash ring
	lb.consistentHash.AddNode(backend.ID, backend.Weight)

	lb.logger.Info("Backend added", 
		zap.String("id", backend.ID),
		zap.String("url", backend.URL.String()),
		zap.Int("weight", backend.Weight))

	return nil
}

// RemoveBackend removes a backend from the load balancer
func (lb *SmartLoadBalancer) RemoveBackend(backendID string) error {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	backend, exists := lb.backends[backendID]
	if !exists {
		return fmt.Errorf("backend %s not found", backendID)
	}

	// Set backend to draining state
	backend.SetState(StateDraining)

	// Remove from healthy backends
	for i, b := range lb.healthyBackends {
		if b.ID == backendID {
			lb.healthyBackends = append(lb.healthyBackends[:i], lb.healthyBackends[i+1:]...)
			break
		}
	}

	// Remove from consistent hash ring
	lb.consistentHash.RemoveNode(backendID)

	// Clean up sticky sessions
	lb.sessionMutex.Lock()
	for sessionID, bID := range lb.stickySessions {
		if bID == backendID {
			delete(lb.stickySessions, sessionID)
		}
	}
	lb.sessionMutex.Unlock()

	delete(lb.backends, backendID)

	lb.logger.Info("Backend removed", zap.String("id", backendID))
	return nil
}

// GetBackend selects a backend using the configured algorithm
func (lb *SmartLoadBalancer) GetBackend(ctx context.Context, clientIP string, sessionID string) (*Backend, error) {
	// Check for sticky session
	if lb.config.EnableStickySession && sessionID != "" {
		if backend := lb.getStickyBackend(sessionID); backend != nil {
			return backend, nil
		}
	}

	lb.mutex.RLock()
	if len(lb.healthyBackends) == 0 {
		lb.mutex.RUnlock()
		return nil, fmt.Errorf("no healthy backends available")
	}
	
	backends := make([]*Backend, len(lb.healthyBackends))
	copy(backends, lb.healthyBackends)
	lb.mutex.RUnlock()

	var backend *Backend
	var err error

	switch lb.config.Algorithm {
	case RoundRobin:
		backend = lb.roundRobin(backends)
	case WeightedRoundRobin:
		backend = lb.weightedRoundRobin(backends)
	case LeastConnections:
		backend = lb.leastConnections(backends)
	case WeightedLeastConnections:
		backend = lb.weightedLeastConnections(backends)
	case IPHash:
		backend = lb.ipHash(backends, clientIP)
	case ConsistentHash:
		backend = lb.consistentHashSelection(clientIP)
	case RandomChoice:
		backend = lb.randomChoice(backends)
	case PowerOfTwoChoices:
		backend = lb.powerOfTwoChoices(backends)
	case AdaptiveLoad:
		backend = lb.adaptiveLoad(backends)
	default:
		err = fmt.Errorf("unsupported algorithm: %s", lb.config.Algorithm)
	}

	if err != nil {
		return nil, err
	}

	if backend == nil {
		return nil, fmt.Errorf("no backend selected")
	}

	// Set sticky session if enabled
	if lb.config.EnableStickySession && sessionID != "" {
		lb.setStickySession(sessionID, backend.ID)
	}

	return backend, nil
}

// RequestWithRetry executes a request with retry logic
func (lb *SmartLoadBalancer) RequestWithRetry(ctx context.Context, clientIP string, sessionID string, requestFunc func(*Backend) error) error {
	var lastErr error

	for attempt := 0; attempt < lb.config.MaxRetries; attempt++ {
		backend, err := lb.GetBackend(ctx, clientIP, sessionID)
		if err != nil {
			return err
		}

		// Increment active connections
		atomic.AddInt64(&backend.Metrics.ActiveConnections, 1)
		start := time.Now()

		err = requestFunc(backend)
		duration := time.Since(start)

		// Decrement active connections
		atomic.AddInt64(&backend.Metrics.ActiveConnections, -1)

		// Record metrics
		success := err == nil
		backend.RecordRequest(duration, success)
		atomic.AddInt64(&lb.metrics.TotalRequests, 1)

		if success {
			atomic.AddInt64(&lb.metrics.SuccessfulRequests, 1)
			return nil
		}

		atomic.AddInt64(&lb.metrics.FailedRequests, 1)
		lastErr = err

		// Mark backend as unhealthy if circuit breaker is enabled
		if lb.config.CircuitBreakerEnabled {
			failedRequests := atomic.LoadInt64(&backend.Metrics.FailedRequests)
			if failedRequests >= int64(lb.config.CircuitBreakerThreshold) {
				backend.SetState(StateUnhealthy)
				lb.removeFromHealthyBackends(backend.ID)
			}
		}

		// Wait before retrying
		if attempt < lb.config.MaxRetries-1 {
			select {
			case <-time.After(lb.config.RetryBackoff * time.Duration(attempt+1)):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	return fmt.Errorf("all retry attempts failed, last error: %w", lastErr)
}

// GetMetrics returns current load balancer metrics
func (lb *SmartLoadBalancer) GetMetrics() LoadBalancerMetrics {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()

	metrics := LoadBalancerMetrics{
		TotalRequests:      atomic.LoadInt64(&lb.metrics.TotalRequests),
		SuccessfulRequests: atomic.LoadInt64(&lb.metrics.SuccessfulRequests),
		FailedRequests:     atomic.LoadInt64(&lb.metrics.FailedRequests),
		BackendMetrics:     make(map[string]*BackendMetrics),
	}

	// Calculate error rate
	if metrics.TotalRequests > 0 {
		metrics.ErrorRate = float64(metrics.FailedRequests) / float64(metrics.TotalRequests)
	}

	// Copy backend metrics
	for id, backend := range lb.backends {
		metrics.BackendMetrics[id] = &BackendMetrics{
			TotalRequests:      atomic.LoadInt64(&backend.Metrics.TotalRequests),
			SuccessfulRequests: atomic.LoadInt64(&backend.Metrics.SuccessfulRequests),
			FailedRequests:     atomic.LoadInt64(&backend.Metrics.FailedRequests),
			ActiveConnections:  atomic.LoadInt64(&backend.Metrics.ActiveConnections),
			AverageLatency:     backend.Metrics.AverageLatency,
			P95Latency:         backend.Metrics.P95Latency,
			P99Latency:         backend.Metrics.P99Latency,
			ErrorRate:          backend.Metrics.ErrorRate,
			LastHealthCheck:    backend.Metrics.LastHealthCheck,
			HealthCheckCount:   atomic.LoadInt64(&backend.Metrics.HealthCheckCount),
		}
	}

	return metrics
}

// Close shuts down the load balancer
func (lb *SmartLoadBalancer) Close() error {
	close(lb.shutdown)
	lb.workers.Wait()
	
	lb.logger.Info("Load balancer stopped")
	return nil
}

// Load balancing algorithms

func (lb *SmartLoadBalancer) roundRobin(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}
	
	index := atomic.AddInt32(&lb.roundRobinIndex, 1) - 1
	return backends[int(index)%len(backends)]
}

func (lb *SmartLoadBalancer) weightedRoundRobin(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	// Simple weighted round robin implementation
	totalWeight := 0
	for _, backend := range backends {
		totalWeight += backend.Weight
	}

	if totalWeight == 0 {
		return lb.roundRobin(backends)
	}

	target := int(atomic.AddInt32(&lb.roundRobinIndex, 1)-1) % totalWeight
	currentWeight := 0

	for _, backend := range backends {
		currentWeight += backend.Weight
		if target < currentWeight {
			return backend
		}
	}

	return backends[0]
}

func (lb *SmartLoadBalancer) leastConnections(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	minConnections := int64(math.MaxInt64)

	for _, backend := range backends {
		connections := atomic.LoadInt64(&backend.Metrics.ActiveConnections)
		if connections < minConnections {
			minConnections = connections
			selected = backend
		}
	}

	return selected
}

func (lb *SmartLoadBalancer) weightedLeastConnections(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	minScore := float64(math.MaxFloat64)

	for _, backend := range backends {
		connections := atomic.LoadInt64(&backend.Metrics.ActiveConnections)
		weight := float64(backend.Weight)
		if weight == 0 {
			weight = 1
		}
		
		score := float64(connections) / weight
		if score < minScore {
			minScore = score
			selected = backend
		}
	}

	return selected
}

func (lb *SmartLoadBalancer) ipHash(backends []*Backend, clientIP string) *Backend {
	if len(backends) == 0 {
		return nil
	}

	hash := lb.hashString(clientIP)
	index := hash % uint32(len(backends))
	return backends[index]
}

func (lb *SmartLoadBalancer) consistentHashSelection(key string) *Backend {
	backendID := lb.consistentHash.GetNode(key)
	if backendID == "" {
		return nil
	}

	lb.mutex.RLock()
	backend := lb.backends[backendID]
	lb.mutex.RUnlock()

	if backend != nil && backend.IsAvailable() {
		return backend
	}

	return nil
}

func (lb *SmartLoadBalancer) randomChoice(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}
	
	index := rand.Intn(len(backends))
	return backends[index]
}

func (lb *SmartLoadBalancer) powerOfTwoChoices(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}
	
	if len(backends) == 1 {
		return backends[0]
	}

	// Randomly select two backends and choose the one with fewer connections
	i := rand.Intn(len(backends))
	j := rand.Intn(len(backends))
	for i == j {
		j = rand.Intn(len(backends))
	}

	connectionsI := atomic.LoadInt64(&backends[i].Metrics.ActiveConnections)
	connectionsJ := atomic.LoadInt64(&backends[j].Metrics.ActiveConnections)

	if connectionsI <= connectionsJ {
		return backends[i]
	}
	return backends[j]
}

func (lb *SmartLoadBalancer) adaptiveLoad(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	maxScore := float64(0)

	for _, backend := range backends {
		score := backend.GetLoadScore()
		if score > maxScore {
			maxScore = score
			selected = backend
		}
	}

	return selected
}

// Helper methods

func (lb *SmartLoadBalancer) getStickyBackend(sessionID string) *Backend {
	lb.sessionMutex.RLock()
	backendID, exists := lb.stickySessions[sessionID]
	lb.sessionMutex.RUnlock()

	if !exists {
		return nil
	}

	lb.mutex.RLock()
	backend := lb.backends[backendID]
	lb.mutex.RUnlock()

	if backend != nil && backend.IsAvailable() {
		return backend
	}

	// Backend is not available, remove sticky session
	lb.sessionMutex.Lock()
	delete(lb.stickySessions, sessionID)
	lb.sessionMutex.Unlock()

	return nil
}

func (lb *SmartLoadBalancer) setStickySession(sessionID, backendID string) {
	lb.sessionMutex.Lock()
	defer lb.sessionMutex.Unlock()
	lb.stickySessions[sessionID] = backendID
}

func (lb *SmartLoadBalancer) removeFromHealthyBackends(backendID string) {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	for i, backend := range lb.healthyBackends {
		if backend.ID == backendID {
			lb.healthyBackends = append(lb.healthyBackends[:i], lb.healthyBackends[i+1:]...)
			break
		}
	}
}

func (lb *SmartLoadBalancer) hashString(s string) uint32 {
	h := uint32(0)
	for i := 0; i < len(s); i++ {
		h = h*31 + uint32(s[i])
	}
	return h
}

// Background workers

func (lb *SmartLoadBalancer) startBackgroundWorkers() {
	// Health checker
	lb.workers.Add(1)
	go lb.healthChecker()

	// Metrics updater
	lb.workers.Add(1)
	go lb.metricsUpdater()

	// Session cleaner
	if lb.config.EnableStickySession {
		lb.workers.Add(1)
		go lb.sessionCleaner()
	}
}

func (lb *SmartLoadBalancer) healthChecker() {
	defer lb.workers.Done()
	ticker := time.NewTicker(lb.config.HealthCheckConfig.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			lb.performHealthChecks()
		case <-lb.shutdown:
			return
		}
	}
}

func (lb *SmartLoadBalancer) performHealthChecks() {
	lb.mutex.RLock()
	backends := make([]*Backend, 0, len(lb.backends))
	for _, backend := range lb.backends {
		backends = append(backends, backend)
	}
	lb.mutex.RUnlock()

	for _, backend := range backends {
		go lb.checkBackendHealth(backend)
	}
}

func (lb *SmartLoadBalancer) checkBackendHealth(backend *Backend) {
	if err := lb.semaphore.Acquire(context.Background(), 1); err != nil {
		return
	}
	defer lb.semaphore.Release(1)

	ctx, cancel := context.WithTimeout(context.Background(), backend.HealthCheck.Timeout)
	defer cancel()

	healthCheckURL := backend.URL.ResolveReference(&url.URL{Path: backend.HealthCheck.Path})
	
	req, err := http.NewRequestWithContext(ctx, backend.HealthCheck.Method, healthCheckURL.String(), nil)
	if err != nil {
		lb.handleHealthCheckFailure(backend)
		return
	}

	// Add custom headers
	for key, value := range backend.HealthCheck.Headers {
		req.Header.Set(key, value)
	}

	start := time.Now()
	resp, err := backend.client.Do(req)
	duration := time.Since(start)

	if err != nil {
		lb.handleHealthCheckFailure(backend)
		return
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode != backend.HealthCheck.ExpectedStatus {
		lb.handleHealthCheckFailure(backend)
		return
	}

	// Health check passed
	lb.handleHealthCheckSuccess(backend, duration)
}

func (lb *SmartLoadBalancer) handleHealthCheckSuccess(backend *Backend, duration time.Duration) {
	atomic.AddInt64(&backend.Metrics.HealthCheckCount, 1)
	backend.Metrics.LastHealthCheck = time.Now()
	
	healthyChecks := atomic.AddInt32(&backend.healthyChecks, 1)
	atomic.StoreInt32(&backend.unhealthyChecks, 0)

	// Mark as healthy if it passes threshold
	if healthyChecks >= int32(backend.HealthCheck.HealthyThreshold) && !backend.IsHealthy() {
		backend.SetState(StateHealthy)
		
		lb.mutex.Lock()
		lb.healthyBackends = append(lb.healthyBackends, backend)
		lb.mutex.Unlock()

		lb.logger.Info("Backend marked as healthy", 
			zap.String("id", backend.ID),
			zap.Duration("response_time", duration))
	}
}

func (lb *SmartLoadBalancer) handleHealthCheckFailure(backend *Backend) {
	atomic.AddInt64(&backend.Metrics.HealthCheckCount, 1)
	backend.Metrics.LastHealthCheck = time.Now()
	
	unhealthyChecks := atomic.AddInt32(&backend.unhealthyChecks, 1)
	atomic.StoreInt32(&backend.healthyChecks, 0)

	// Mark as unhealthy if it fails threshold
	if unhealthyChecks >= int32(backend.HealthCheck.UnhealthyThreshold) && backend.IsHealthy() {
		backend.SetState(StateUnhealthy)
		lb.removeFromHealthyBackends(backend.ID)

		lb.logger.Warn("Backend marked as unhealthy", zap.String("id", backend.ID))
	}
}

func (lb *SmartLoadBalancer) metricsUpdater() {
	defer lb.workers.Done()
	ticker := time.NewTicker(lb.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			lb.updateMetrics()
		case <-lb.shutdown:
			return
		}
	}
}

func (lb *SmartLoadBalancer) updateMetrics() {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()

	// Update load scores for adaptive load balancing
	for _, backend := range lb.backends {
		backend.UpdateLoadScore()
	}

	// Calculate overall metrics
	totalRequests := atomic.LoadInt64(&lb.metrics.TotalRequests)
	if totalRequests > 0 {
		failed := atomic.LoadInt64(&lb.metrics.FailedRequests)
		lb.metrics.ErrorRate = float64(failed) / float64(totalRequests)
	}
}

func (lb *SmartLoadBalancer) sessionCleaner() {
	defer lb.workers.Done()
	ticker := time.NewTicker(time.Minute * 10) // Clean every 10 minutes
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			lb.cleanExpiredSessions()
		case <-lb.shutdown:
			return
		}
	}
}

func (lb *SmartLoadBalancer) cleanExpiredSessions() {
	// This is a simplified implementation
	// In a real system, you'd track session timestamps
	lb.sessionMutex.Lock()
	defer lb.sessionMutex.Unlock()

	// For now, just periodically clear all sessions
	// In production, implement proper session expiration tracking
	if len(lb.stickySessions) > 10000 { // Prevent memory bloat
		lb.stickySessions = make(map[string]string)
		lb.logger.Debug("Cleared sticky sessions to prevent memory bloat")
	}
}

// ConsistentHashRing implements consistent hashing for load balancing
type ConsistentHashRing struct {
	nodes       map[uint32]string
	sortedKeys  []uint32
	replicas    int
	mutex       sync.RWMutex
}

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing(replicas int) *ConsistentHashRing {
	return &ConsistentHashRing{
		nodes:    make(map[uint32]string),
		replicas: replicas,
	}
}

// AddNode adds a node to the hash ring
func (chr *ConsistentHashRing) AddNode(nodeID string, weight int) {
	chr.mutex.Lock()
	defer chr.mutex.Unlock()

	// Add virtual nodes based on weight
	virtualNodes := chr.replicas * weight
	for i := 0; i < virtualNodes; i++ {
		virtualKey := fmt.Sprintf("%s:%d", nodeID, i)
		hash := chr.hashString(virtualKey)
		chr.nodes[hash] = nodeID
		chr.sortedKeys = append(chr.sortedKeys, hash)
	}

	sort.Slice(chr.sortedKeys, func(i, j int) bool {
		return chr.sortedKeys[i] < chr.sortedKeys[j]
	})
}

// RemoveNode removes a node from the hash ring
func (chr *ConsistentHashRing) RemoveNode(nodeID string) {
	chr.mutex.Lock()
	defer chr.mutex.Unlock()

	// Remove all virtual nodes for this node
	newSortedKeys := make([]uint32, 0)
	for _, key := range chr.sortedKeys {
		if chr.nodes[key] != nodeID {
			newSortedKeys = append(newSortedKeys, key)
		} else {
			delete(chr.nodes, key)
		}
	}
	chr.sortedKeys = newSortedKeys
}

// GetNode returns the node responsible for a given key
func (chr *ConsistentHashRing) GetNode(key string) string {
	chr.mutex.RLock()
	defer chr.mutex.RUnlock()

	if len(chr.sortedKeys) == 0 {
		return ""
	}

	hash := chr.hashString(key)
	
	// Binary search for the appropriate node
	idx := sort.Search(len(chr.sortedKeys), func(i int) bool {
		return chr.sortedKeys[i] >= hash
	})

	// Wrap around to the beginning if necessary
	if idx == len(chr.sortedKeys) {
		idx = 0
	}

	return chr.nodes[chr.sortedKeys[idx]]
}

func (chr *ConsistentHashRing) hashString(s string) uint32 {
	h := uint32(0)
	for i := 0; i < len(s); i++ {
		h = h*31 + uint32(s[i])
	}
	return h
}