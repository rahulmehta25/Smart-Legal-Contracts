package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

// MetricType represents different types of metrics
type MetricType int

const (
	Counter MetricType = iota
	Gauge
	Histogram
	Summary
)

// String returns string representation of metric type
func (mt MetricType) String() string {
	switch mt {
	case Counter:
		return "counter"
	case Gauge:
		return "gauge"
	case Histogram:
		return "histogram"
	case Summary:
		return "summary"
	default:
		return "unknown"
	}
}

// SystemMetrics represents system-level performance metrics
type SystemMetrics struct {
	Timestamp         time.Time `json:"timestamp"`
	CPUUsagePercent   float64   `json:"cpu_usage_percent"`
	MemoryUsageMB     float64   `json:"memory_usage_mb"`
	MemoryAllocMB     float64   `json:"memory_alloc_mb"`
	MemoryTotalAllocMB float64  `json:"memory_total_alloc_mb"`
	NumGoroutines     int       `json:"num_goroutines"`
	NumGC             uint32    `json:"num_gc"`
	GCPauseMs         float64   `json:"gc_pause_ms"`
	HeapObjects       uint64    `json:"heap_objects"`
	StackInUseMB      float64   `json:"stack_in_use_mb"`
	NumCgoCall        int64     `json:"num_cgo_call"`
	OpenFileDescriptors int     `json:"open_file_descriptors"`
}

// ApplicationMetrics represents application-level metrics
type ApplicationMetrics struct {
	Timestamp          time.Time              `json:"timestamp"`
	RequestsPerSecond  float64               `json:"requests_per_second"`
	AverageLatency     time.Duration         `json:"average_latency"`
	P95Latency         time.Duration         `json:"p95_latency"`
	P99Latency         time.Duration         `json:"p99_latency"`
	ErrorRate          float64               `json:"error_rate"`
	ThroughputMBps     float64               `json:"throughput_mbps"`
	ActiveConnections  int64                 `json:"active_connections"`
	CacheHitRate       float64               `json:"cache_hit_rate"`
	QueueLength        int64                 `json:"queue_length"`
	CustomMetrics      map[string]interface{} `json:"custom_metrics"`
}

// PerformanceAlert represents a performance alert
type PerformanceAlert struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	Level       AlertLevel             `json:"level"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	MetricName  string                 `json:"metric_name"`
	Threshold   float64                `json:"threshold"`
	CurrentValue float64               `json:"current_value"`
	Tags        map[string]string      `json:"tags"`
}

// AlertLevel represents alert severity levels
type AlertLevel int

const (
	AlertInfo AlertLevel = iota
	AlertWarning
	AlertCritical
)

// String returns string representation of alert level
func (al AlertLevel) String() string {
	switch al {
	case AlertInfo:
		return "info"
	case AlertWarning:
		return "warning"
	case AlertCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// ProfilerConfig holds configuration for the profiler
type ProfilerConfig struct {
	EnableProfiling       bool          `json:"enable_profiling"`
	ProfilingPort         int           `json:"profiling_port"`
	MetricsInterval       time.Duration `json:"metrics_interval"`
	EnablePrometheus      bool          `json:"enable_prometheus"`
	PrometheusPort        int           `json:"prometheus_port"`
	EnableGoroutineLeakDetection bool   `json:"enable_goroutine_leak_detection"`
	GoroutineThreshold    int           `json:"goroutine_threshold"`
	EnableMemoryProfiling bool          `json:"enable_memory_profiling"`
	MemoryThresholdMB     int           `json:"memory_threshold_mb"`
	EnableCPUProfiling    bool          `json:"enable_cpu_profiling"`
	CPUThresholdPercent   float64       `json:"cpu_threshold_percent"`
	AlertsEnabled         bool          `json:"alerts_enabled"`
	MaxMetricsHistory     int           `json:"max_metrics_history"`
	EnableTracing         bool          `json:"enable_tracing"`
	TracingSampleRate     float64       `json:"tracing_sample_rate"`
}

// DefaultProfilerConfig returns a default profiler configuration
func DefaultProfilerConfig() *ProfilerConfig {
	return &ProfilerConfig{
		EnableProfiling:              true,
		ProfilingPort:               6060,
		MetricsInterval:             time.Second * 30,
		EnablePrometheus:            true,
		PrometheusPort:              8080,
		EnableGoroutineLeakDetection: true,
		GoroutineThreshold:          10000,
		EnableMemoryProfiling:       true,
		MemoryThresholdMB:           1024, // 1GB
		EnableCPUProfiling:          true,
		CPUThresholdPercent:         80.0,
		AlertsEnabled:               true,
		MaxMetricsHistory:           1000,
		EnableTracing:               true,
		TracingSampleRate:           0.1, // 10%
	}
}

// MetricsCollector handles metrics collection and monitoring
type MetricsCollector struct {
	config               *ProfilerConfig
	systemMetrics        []SystemMetrics
	applicationMetrics   []ApplicationMetrics
	alerts               []PerformanceAlert
	mutex                sync.RWMutex
	logger               *zap.Logger
	shutdown             chan struct{}
	workers              sync.WaitGroup
	
	// Prometheus metrics
	prometheusRegistry   prometheus.Registerer
	requestsTotal        prometheus.Counter
	requestDuration      prometheus.Histogram
	goroutinesGauge      prometheus.Gauge
	memoryGauge          prometheus.Gauge
	cpuGauge             prometheus.Gauge
	
	// Performance tracking
	lastCPUTime          int64
	lastCPUMeasurement   time.Time
	requestCount         int64
	errorCount           int64
	totalLatency         int64
	latencyMeasurements  []time.Duration
	latencyMutex         sync.Mutex
	
	// Alert handlers
	alertHandlers        []func(PerformanceAlert)
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(config *ProfilerConfig, logger *zap.Logger) *MetricsCollector {
	if config == nil {
		config = DefaultProfilerConfig()
	}

	mc := &MetricsCollector{
		config:              config,
		systemMetrics:       make([]SystemMetrics, 0, config.MaxMetricsHistory),
		applicationMetrics:  make([]ApplicationMetrics, 0, config.MaxMetricsHistory),
		alerts:              make([]PerformanceAlert, 0),
		logger:              logger,
		shutdown:            make(chan struct{}),
		prometheusRegistry:  prometheus.NewRegistry(),
		latencyMeasurements: make([]time.Duration, 0, 1000),
		alertHandlers:       make([]func(PerformanceAlert), 0),
	}

	// Initialize Prometheus metrics
	if config.EnablePrometheus {
		mc.initPrometheusMetrics()
	}

	return mc
}

// Start begins metrics collection
func (mc *MetricsCollector) Start() error {
	mc.logger.Info("Starting metrics collector",
		zap.Bool("profiling_enabled", mc.config.EnableProfiling),
		zap.Bool("prometheus_enabled", mc.config.EnablePrometheus))

	// Start pprof server
	if mc.config.EnableProfiling {
		go func() {
			addr := fmt.Sprintf(":%d", mc.config.ProfilingPort)
			mc.logger.Info("Starting pprof server", zap.String("address", addr))
			if err := http.ListenAndServe(addr, nil); err != nil {
				mc.logger.Error("Failed to start pprof server", zap.Error(err))
			}
		}()
	}

	// Start Prometheus metrics server
	if mc.config.EnablePrometheus {
		go func() {
			http.Handle("/metrics", promhttp.HandlerFor(
				prometheus.Gatherers{mc.prometheusRegistry.(*prometheus.Registry)},
				promhttp.HandlerOpts{},
			))
			addr := fmt.Sprintf(":%d", mc.config.PrometheusPort)
			mc.logger.Info("Starting Prometheus metrics server", zap.String("address", addr))
			if err := http.ListenAndServe(addr, nil); err != nil {
				mc.logger.Error("Failed to start Prometheus server", zap.Error(err))
			}
		}()
	}

	// Start metrics collection workers
	mc.startWorkers()

	return nil
}

// Stop stops metrics collection
func (mc *MetricsCollector) Stop() error {
	close(mc.shutdown)
	mc.workers.Wait()
	mc.logger.Info("Metrics collector stopped")
	return nil
}

// RecordRequest records a request for metrics tracking
func (mc *MetricsCollector) RecordRequest(duration time.Duration, success bool) {
	atomic.AddInt64(&mc.requestCount, 1)
	atomic.AddInt64(&mc.totalLatency, duration.Nanoseconds())

	if !success {
		atomic.AddInt64(&mc.errorCount, 1)
	}

	// Record latency for percentile calculations
	mc.latencyMutex.Lock()
	mc.latencyMeasurements = append(mc.latencyMeasurements, duration)
	if len(mc.latencyMeasurements) > 1000 {
		// Keep only recent measurements
		mc.latencyMeasurements = mc.latencyMeasurements[500:]
	}
	mc.latencyMutex.Unlock()

	// Update Prometheus metrics
	if mc.config.EnablePrometheus {
		if mc.requestsTotal != nil {
			mc.requestsTotal.Inc()
		}
		if mc.requestDuration != nil {
			mc.requestDuration.Observe(duration.Seconds())
		}
	}
}

// GetSystemMetrics returns current system metrics
func (mc *MetricsCollector) GetSystemMetrics() SystemMetrics {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	var gcStats debug.GCStats
	debug.ReadGCStats(&gcStats)

	metrics := SystemMetrics{
		Timestamp:          time.Now(),
		MemoryUsageMB:      float64(m.Sys) / 1024 / 1024,
		MemoryAllocMB:      float64(m.Alloc) / 1024 / 1024,
		MemoryTotalAllocMB: float64(m.TotalAlloc) / 1024 / 1024,
		NumGoroutines:      runtime.NumGoroutine(),
		NumGC:              m.NumGC,
		HeapObjects:        m.HeapObjects,
		StackInUseMB:       float64(m.StackInuse) / 1024 / 1024,
		NumCgoCall:         runtime.NumCgoCall(),
	}

	// Calculate GC pause time
	if len(gcStats.Pause) > 0 {
		metrics.GCPauseMs = float64(gcStats.Pause[0]) / float64(time.Millisecond)
	}

	// Calculate CPU usage (simplified)
	metrics.CPUUsagePercent = mc.calculateCPUUsage()

	return metrics
}

// GetApplicationMetrics returns current application metrics
func (mc *MetricsCollector) GetApplicationMetrics() ApplicationMetrics {
	requestCount := atomic.LoadInt64(&mc.requestCount)
	errorCount := atomic.LoadInt64(&mc.errorCount)
	totalLatency := atomic.LoadInt64(&mc.totalLatency)

	metrics := ApplicationMetrics{
		Timestamp:     time.Now(),
		CustomMetrics: make(map[string]interface{}),
	}

	// Calculate error rate
	if requestCount > 0 {
		metrics.ErrorRate = float64(errorCount) / float64(requestCount)
	}

	// Calculate average latency
	if requestCount > 0 {
		metrics.AverageLatency = time.Duration(totalLatency / requestCount)
	}

	// Calculate percentiles
	mc.latencyMutex.Lock()
	if len(mc.latencyMeasurements) > 0 {
		latencies := make([]time.Duration, len(mc.latencyMeasurements))
		copy(latencies, mc.latencyMeasurements)
		mc.latencyMutex.Unlock()

		// Sort latencies for percentile calculation
		mc.sortLatencies(latencies)
		
		if len(latencies) > 0 {
			p95Index := int(float64(len(latencies)) * 0.95)
			p99Index := int(float64(len(latencies)) * 0.99)
			
			if p95Index >= len(latencies) {
				p95Index = len(latencies) - 1
			}
			if p99Index >= len(latencies) {
				p99Index = len(latencies) - 1
			}
			
			metrics.P95Latency = latencies[p95Index]
			metrics.P99Latency = latencies[p99Index]
		}
	} else {
		mc.latencyMutex.Unlock()
	}

	return metrics
}

// GetMetricsHistory returns historical metrics
func (mc *MetricsCollector) GetMetricsHistory() ([]SystemMetrics, []ApplicationMetrics) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	systemHistory := make([]SystemMetrics, len(mc.systemMetrics))
	appHistory := make([]ApplicationMetrics, len(mc.applicationMetrics))

	copy(systemHistory, mc.systemMetrics)
	copy(appHistory, mc.applicationMetrics)

	return systemHistory, appHistory
}

// GetAlerts returns current alerts
func (mc *MetricsCollector) GetAlerts() []PerformanceAlert {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	alerts := make([]PerformanceAlert, len(mc.alerts))
	copy(alerts, mc.alerts)
	return alerts
}

// AddAlertHandler adds a handler for performance alerts
func (mc *MetricsCollector) AddAlertHandler(handler func(PerformanceAlert)) {
	mc.alertHandlers = append(mc.alertHandlers, handler)
}

// DetectGoroutineLeak checks for potential goroutine leaks
func (mc *MetricsCollector) DetectGoroutineLeak() bool {
	if !mc.config.EnableGoroutineLeakDetection {
		return false
	}

	numGoroutines := runtime.NumGoroutine()
	if numGoroutines > mc.config.GoroutineThreshold {
		alert := PerformanceAlert{
			ID:           fmt.Sprintf("goroutine-leak-%d", time.Now().Unix()),
			Timestamp:    time.Now(),
			Level:        AlertCritical,
			Title:        "Potential Goroutine Leak Detected",
			Description:  fmt.Sprintf("Number of goroutines (%d) exceeds threshold (%d)", numGoroutines, mc.config.GoroutineThreshold),
			MetricName:   "goroutines",
			Threshold:    float64(mc.config.GoroutineThreshold),
			CurrentValue: float64(numGoroutines),
			Tags:         map[string]string{"type": "goroutine_leak"},
		}

		mc.triggerAlert(alert)
		return true
	}

	return false
}

// DetectMemoryLeak checks for potential memory leaks
func (mc *MetricsCollector) DetectMemoryLeak() bool {
	if !mc.config.EnableMemoryProfiling {
		return false
	}

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	memoryUsageMB := float64(m.Sys) / 1024 / 1024
	if memoryUsageMB > float64(mc.config.MemoryThresholdMB) {
		alert := PerformanceAlert{
			ID:           fmt.Sprintf("memory-leak-%d", time.Now().Unix()),
			Timestamp:    time.Now(),
			Level:        AlertCritical,
			Title:        "High Memory Usage Detected",
			Description:  fmt.Sprintf("Memory usage (%.2f MB) exceeds threshold (%d MB)", memoryUsageMB, mc.config.MemoryThresholdMB),
			MetricName:   "memory_usage",
			Threshold:    float64(mc.config.MemoryThresholdMB),
			CurrentValue: memoryUsageMB,
			Tags:         map[string]string{"type": "memory_usage"},
		}

		mc.triggerAlert(alert)
		return true
	}

	return false
}

// ExportMetrics exports metrics in JSON format
func (mc *MetricsCollector) ExportMetrics() ([]byte, error) {
	systemMetrics := mc.GetSystemMetrics()
	appMetrics := mc.GetApplicationMetrics()

	export := struct {
		SystemMetrics      SystemMetrics      `json:"system_metrics"`
		ApplicationMetrics ApplicationMetrics `json:"application_metrics"`
		Timestamp          time.Time          `json:"timestamp"`
	}{
		SystemMetrics:      systemMetrics,
		ApplicationMetrics: appMetrics,
		Timestamp:          time.Now(),
	}

	return json.Marshal(export)
}

// Internal methods

func (mc *MetricsCollector) initPrometheusMetrics() {
	// Request metrics
	mc.requestsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "requests_total",
		Help: "Total number of requests",
	})

	mc.requestDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "request_duration_seconds",
		Help:    "Request duration in seconds",
		Buckets: prometheus.DefBuckets,
	})

	// System metrics
	mc.goroutinesGauge = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "goroutines_total",
		Help: "Number of goroutines",
	})

	mc.memoryGauge = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "memory_usage_bytes",
		Help: "Memory usage in bytes",
	})

	mc.cpuGauge = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "cpu_usage_percent",
		Help: "CPU usage percentage",
	})

	// Register metrics
	registry := mc.prometheusRegistry.(*prometheus.Registry)
	registry.MustRegister(mc.requestsTotal)
	registry.MustRegister(mc.requestDuration)
	registry.MustRegister(mc.goroutinesGauge)
	registry.MustRegister(mc.memoryGauge)
	registry.MustRegister(mc.cpuGauge)
}

func (mc *MetricsCollector) startWorkers() {
	// Metrics collection worker
	mc.workers.Add(1)
	go mc.metricsCollectionWorker()

	// Alert monitoring worker
	if mc.config.AlertsEnabled {
		mc.workers.Add(1)
		go mc.alertMonitoringWorker()
	}
}

func (mc *MetricsCollector) metricsCollectionWorker() {
	defer mc.workers.Done()
	ticker := time.NewTicker(mc.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mc.collectAndStoreMetrics()
		case <-mc.shutdown:
			return
		}
	}
}

func (mc *MetricsCollector) alertMonitoringWorker() {
	defer mc.workers.Done()
	ticker := time.NewTicker(time.Minute) // Check alerts every minute
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mc.checkAlerts()
		case <-mc.shutdown:
			return
		}
	}
}

func (mc *MetricsCollector) collectAndStoreMetrics() {
	// Collect system metrics
	systemMetrics := mc.GetSystemMetrics()
	
	// Collect application metrics
	appMetrics := mc.GetApplicationMetrics()

	// Store metrics
	mc.mutex.Lock()
	mc.systemMetrics = append(mc.systemMetrics, systemMetrics)
	mc.applicationMetrics = append(mc.applicationMetrics, appMetrics)

	// Limit history size
	if len(mc.systemMetrics) > mc.config.MaxMetricsHistory {
		mc.systemMetrics = mc.systemMetrics[1:]
	}
	if len(mc.applicationMetrics) > mc.config.MaxMetricsHistory {
		mc.applicationMetrics = mc.applicationMetrics[1:]
	}
	mc.mutex.Unlock()

	// Update Prometheus metrics
	if mc.config.EnablePrometheus {
		if mc.goroutinesGauge != nil {
			mc.goroutinesGauge.Set(float64(systemMetrics.NumGoroutines))
		}
		if mc.memoryGauge != nil {
			mc.memoryGauge.Set(systemMetrics.MemoryUsageMB * 1024 * 1024)
		}
		if mc.cpuGauge != nil {
			mc.cpuGauge.Set(systemMetrics.CPUUsagePercent)
		}
	}

	mc.logger.Debug("Metrics collected",
		zap.Int("goroutines", systemMetrics.NumGoroutines),
		zap.Float64("memory_mb", systemMetrics.MemoryUsageMB),
		zap.Float64("cpu_percent", systemMetrics.CPUUsagePercent))
}

func (mc *MetricsCollector) checkAlerts() {
	// Check goroutine leaks
	mc.DetectGoroutineLeak()

	// Check memory leaks
	mc.DetectMemoryLeak()

	// Check CPU usage
	systemMetrics := mc.GetSystemMetrics()
	if systemMetrics.CPUUsagePercent > mc.config.CPUThresholdPercent {
		alert := PerformanceAlert{
			ID:           fmt.Sprintf("high-cpu-%d", time.Now().Unix()),
			Timestamp:    time.Now(),
			Level:        AlertWarning,
			Title:        "High CPU Usage",
			Description:  fmt.Sprintf("CPU usage (%.2f%%) exceeds threshold (%.2f%%)", systemMetrics.CPUUsagePercent, mc.config.CPUThresholdPercent),
			MetricName:   "cpu_usage",
			Threshold:    mc.config.CPUThresholdPercent,
			CurrentValue: systemMetrics.CPUUsagePercent,
			Tags:         map[string]string{"type": "cpu_usage"},
		}

		mc.triggerAlert(alert)
	}
}

func (mc *MetricsCollector) triggerAlert(alert PerformanceAlert) {
	mc.mutex.Lock()
	mc.alerts = append(mc.alerts, alert)
	
	// Limit alerts history
	if len(mc.alerts) > 100 {
		mc.alerts = mc.alerts[1:]
	}
	mc.mutex.Unlock()

	// Call alert handlers
	for _, handler := range mc.alertHandlers {
		go handler(alert)
	}

	mc.logger.Warn("Performance alert triggered",
		zap.String("id", alert.ID),
		zap.String("title", alert.Title),
		zap.String("level", alert.Level.String()),
		zap.Float64("threshold", alert.Threshold),
		zap.Float64("current_value", alert.CurrentValue))
}

func (mc *MetricsCollector) calculateCPUUsage() float64 {
	// Simplified CPU usage calculation
	// In production, use more sophisticated methods or external libraries
	currentTime := time.Now()
	
	if mc.lastCPUMeasurement.IsZero() {
		mc.lastCPUMeasurement = currentTime
		return 0.0
	}

	// This is a placeholder implementation
	// Real CPU usage calculation would involve system calls
	numGoroutines := runtime.NumGoroutine()
	return math.Min(float64(numGoroutines)/100.0*10.0, 100.0)
}

func (mc *MetricsCollector) sortLatencies(latencies []time.Duration) {
	// Simple bubble sort for small arrays
	n := len(latencies)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if latencies[j] > latencies[j+1] {
				latencies[j], latencies[j+1] = latencies[j+1], latencies[j]
			}
		}
	}
}

// TraceProfiler provides distributed tracing capabilities
type TraceProfiler struct {
	config     *ProfilerConfig
	traces     map[string]*Trace
	mutex      sync.RWMutex
	logger     *zap.Logger
	sampleRate float64
}

// Trace represents a distributed trace
type Trace struct {
	ID        string                 `json:"id"`
	ParentID  string                 `json:"parent_id,omitempty"`
	Operation string                 `json:"operation"`
	StartTime time.Time              `json:"start_time"`
	EndTime   time.Time              `json:"end_time"`
	Duration  time.Duration          `json:"duration"`
	Tags      map[string]interface{} `json:"tags"`
	Logs      []TraceLog             `json:"logs"`
}

// TraceLog represents a log entry within a trace
type TraceLog struct {
	Timestamp time.Time              `json:"timestamp"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Fields    map[string]interface{} `json:"fields"`
}

// NewTraceProfiler creates a new trace profiler
func NewTraceProfiler(config *ProfilerConfig, logger *zap.Logger) *TraceProfiler {
	return &TraceProfiler{
		config:     config,
		traces:     make(map[string]*Trace),
		logger:     logger,
		sampleRate: config.TracingSampleRate,
	}
}

// StartTrace starts a new trace
func (tp *TraceProfiler) StartTrace(id, operation string, parentID string) *Trace {
	// Sample traces based on sample rate
	if tp.shouldSample() {
		trace := &Trace{
			ID:        id,
			ParentID:  parentID,
			Operation: operation,
			StartTime: time.Now(),
			Tags:      make(map[string]interface{}),
			Logs:      make([]TraceLog, 0),
		}

		tp.mutex.Lock()
		tp.traces[id] = trace
		tp.mutex.Unlock()

		return trace
	}

	return nil
}

// FinishTrace completes a trace
func (tp *TraceProfiler) FinishTrace(traceID string) {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()

	if trace, exists := tp.traces[traceID]; exists {
		trace.EndTime = time.Now()
		trace.Duration = trace.EndTime.Sub(trace.StartTime)

		tp.logger.Debug("Trace completed",
			zap.String("trace_id", traceID),
			zap.String("operation", trace.Operation),
			zap.Duration("duration", trace.Duration))
	}
}

// AddTraceLog adds a log entry to a trace
func (tp *TraceProfiler) AddTraceLog(traceID, level, message string, fields map[string]interface{}) {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()

	if trace, exists := tp.traces[traceID]; exists {
		log := TraceLog{
			Timestamp: time.Now(),
			Level:     level,
			Message:   message,
			Fields:    fields,
		}
		trace.Logs = append(trace.Logs, log)
	}
}

// GetTrace retrieves a trace by ID
func (tp *TraceProfiler) GetTrace(traceID string) (*Trace, bool) {
	tp.mutex.RLock()
	defer tp.mutex.RUnlock()
	
	trace, exists := tp.traces[traceID]
	return trace, exists
}

func (tp *TraceProfiler) shouldSample() bool {
	return tp.sampleRate > 0 && tp.sampleRate >= 1.0 || (tp.sampleRate > 0 && tp.sampleRate > 0.0)
}

