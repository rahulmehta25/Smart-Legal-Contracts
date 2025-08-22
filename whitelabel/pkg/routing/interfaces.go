package routing

import (
	"time"

	"github.com/enterprise/whitelabel/pkg/tenant"
)

// Cache defines the interface for tenant caching
type Cache interface {
	// Tenant operations
	GetTenant(key string) *tenant.Tenant
	SetTenant(key string, tenant *tenant.Tenant, ttl time.Duration) error
	InvalidateTenant(tenantID string) error
	
	// General cache operations
	Get(key string) (interface{}, error)
	Set(key string, value interface{}, ttl time.Duration) error
	Delete(key string) error
	Clear() error
	
	// Health and stats
	GetHealthStatus() map[string]interface{}
	GetStats() map[string]interface{}
}

// RateLimiter defines the interface for rate limiting
type RateLimiter interface {
	// Rate limiting operations
	Allow(key string, config RateLimitConfig) bool
	GetRemaining(key string) int64
	GetResetTime(key string) time.Time
	
	// Configuration
	UpdateConfig(config RateLimitConfig) error
	
	// Health and stats
	GetHealthStatus() map[string]interface{}
	GetStats() map[string]interface{}
}

// MetricsCollector defines the interface for metrics collection
type MetricsCollector interface {
	// Request metrics
	RecordRequest(tenantID string, duration time.Duration, status int) error
	RecordBandwidth(tenantID string, bytes int64) error
	RecordCacheHit(tenantID string, hit bool) error
	RecordError(tenantID string, errorType string) error
	
	// Aggregate metrics
	GetRequestCount(tenantID string, from, to time.Time) (int64, error)
	GetAverageResponseTime(tenantID string, from, to time.Time) (time.Duration, error)
	GetErrorRate(tenantID string, from, to time.Time) (float64, error)
	GetBandwidthUsage(tenantID string, from, to time.Time) (int64, error)
	
	// Export metrics
	Export(format string) ([]byte, error)
}