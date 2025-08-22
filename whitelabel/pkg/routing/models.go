package routing

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/enterprise/whitelabel/pkg/tenant"
)

// RouteType defines the type of route
type RouteType string

const (
	RouteTypeSubdomain    RouteType = "subdomain"
	RouteTypeCustomDomain RouteType = "custom_domain"
	RouteTypePath         RouteType = "path"
)

// RoutingStrategy defines how requests are routed to tenants
type RoutingStrategy struct {
	Type              RouteType `json:"type"`
	SubdomainPattern  string    `json:"subdomain_pattern,omitempty"`  // e.g., "{tenant}.example.com"
	PathPrefix        string    `json:"path_prefix,omitempty"`        // e.g., "/tenant/{tenant}"
	CustomDomainEnabled bool    `json:"custom_domain_enabled"`
}

// TenantContext holds tenant information in request context
type TenantContext struct {
	Tenant       *tenant.Tenant
	Subdomain    string
	CustomDomain string
	DetectedBy   RouteType
	RequestTime  time.Time
}

// RouteConfig defines routing configuration
type RouteConfig struct {
	Strategy            RoutingStrategy     `json:"strategy"`
	DefaultTenant       string              `json:"default_tenant,omitempty"`
	FallbackDomain      string              `json:"fallback_domain"`
	SSLRequired         bool                `json:"ssl_required"`
	WWWRedirect         bool                `json:"www_redirect"`
	CacheConfig         CacheConfig         `json:"cache_config"`
	RateLimitConfig     RateLimitConfig     `json:"rate_limit_config"`
	SecurityHeaders     map[string]string   `json:"security_headers"`
	CORSConfig          CORSConfig          `json:"cors_config"`
}

// CacheConfig defines caching configuration for routing
type CacheConfig struct {
	Enabled         bool          `json:"enabled"`
	TTL             time.Duration `json:"ttl"`
	MaxSize         int           `json:"max_size"`
	RefreshInterval time.Duration `json:"refresh_interval"`
}

// RateLimitConfig defines rate limiting per tenant
type RateLimitConfig struct {
	Enabled       bool          `json:"enabled"`
	RequestsPerHour int          `json:"requests_per_hour"`
	BurstLimit    int           `json:"burst_limit"`
	WindowSize    time.Duration `json:"window_size"`
	ByIP          bool          `json:"by_ip"`
	ByTenant      bool          `json:"by_tenant"`
}

// CORSConfig defines CORS configuration
type CORSConfig struct {
	Enabled          bool     `json:"enabled"`
	AllowedOrigins   []string `json:"allowed_origins"`
	AllowedMethods   []string `json:"allowed_methods"`
	AllowedHeaders   []string `json:"allowed_headers"`
	ExposedHeaders   []string `json:"exposed_headers"`
	AllowCredentials bool     `json:"allow_credentials"`
	MaxAge           int      `json:"max_age"`
}

// RouteMatch represents a successful route match
type RouteMatch struct {
	Tenant       *tenant.Tenant `json:"tenant"`
	RouteType    RouteType      `json:"route_type"`
	MatchedValue string         `json:"matched_value"` // subdomain, domain, or path
	IsCustomDomain bool         `json:"is_custom_domain"`
}

// HealthCheckConfig defines health check endpoints per tenant
type HealthCheckConfig struct {
	Enabled   bool              `json:"enabled"`
	Path      string            `json:"path"`
	Checks    []HealthCheck     `json:"checks"`
	Timeout   time.Duration     `json:"timeout"`
	Interval  time.Duration     `json:"interval"`
	Headers   map[string]string `json:"headers"`
}

// HealthCheck defines individual health checks
type HealthCheck struct {
	Name        string        `json:"name"`
	Type        string        `json:"type"` // database, redis, http, custom
	Target      string        `json:"target"`
	Timeout     time.Duration `json:"timeout"`
	Retries     int           `json:"retries"`
	Required    bool          `json:"required"`
}

// LoadBalancerConfig defines load balancing configuration
type LoadBalancerConfig struct {
	Enabled      bool              `json:"enabled"`
	Strategy     string            `json:"strategy"` // round_robin, least_connections, weighted
	Endpoints    []Endpoint        `json:"endpoints"`
	HealthCheck  HealthCheckConfig `json:"health_check"`
	StickySession bool             `json:"sticky_session"`
	SessionTTL   time.Duration     `json:"session_ttl"`
}

// Endpoint defines a backend endpoint
type Endpoint struct {
	URL       string            `json:"url"`
	Weight    int               `json:"weight"`
	Health    EndpointHealth    `json:"health"`
	Metadata  map[string]string `json:"metadata"`
	Active    bool              `json:"active"`
}

// EndpointHealth tracks endpoint health status
type EndpointHealth struct {
	Status       string    `json:"status"` // healthy, unhealthy, unknown
	LastCheck    time.Time `json:"last_check"`
	ResponseTime time.Duration `json:"response_time"`
	ErrorCount   int       `json:"error_count"`
	LastError    string    `json:"last_error,omitempty"`
}

// Middleware defines custom middleware for tenant routing
type Middleware struct {
	Name     string            `json:"name"`
	Type     string            `json:"type"` // auth, logging, metrics, custom
	Config   map[string]interface{} `json:"config"`
	Enabled  bool              `json:"enabled"`
	Order    int               `json:"order"`
}

// TenantRoutingRule defines routing rules per tenant
type TenantRoutingRule struct {
	TenantID      string              `json:"tenant_id"`
	CustomDomains []string            `json:"custom_domains"`
	Subdomains    []string            `json:"subdomains"`
	PathPrefixes  []string            `json:"path_prefixes"`
	Middlewares   []Middleware        `json:"middlewares"`
	LoadBalancer  LoadBalancerConfig  `json:"load_balancer"`
	RateLimit     RateLimitConfig     `json:"rate_limit"`
	CacheRules    []CacheRule         `json:"cache_rules"`
	Redirects     []RedirectRule      `json:"redirects"`
	RewriteRules  []RewriteRule       `json:"rewrite_rules"`
}

// CacheRule defines caching rules for specific paths
type CacheRule struct {
	Path        string        `json:"path"`
	Pattern     string        `json:"pattern"`
	TTL         time.Duration `json:"ttl"`
	VaryHeaders []string      `json:"vary_headers"`
	Conditions  []Condition   `json:"conditions"`
}

// RedirectRule defines redirect rules
type RedirectRule struct {
	From       string      `json:"from"`
	To         string      `json:"to"`
	Status     int         `json:"status"`
	Permanent  bool        `json:"permanent"`
	Conditions []Condition `json:"conditions"`
}

// RewriteRule defines URL rewrite rules
type RewriteRule struct {
	From       string      `json:"from"`
	To         string      `json:"to"`
	Conditions []Condition `json:"conditions"`
}

// Condition defines conditions for rules
type Condition struct {
	Type     string `json:"type"`     // header, query, path, host
	Key      string `json:"key"`
	Value    string `json:"value"`
	Operator string `json:"operator"` // equals, contains, starts_with, regex
}

// RoutingMetrics tracks routing performance
type RoutingMetrics struct {
	TenantID       string        `json:"tenant_id"`
	Timestamp      time.Time     `json:"timestamp"`
	RequestCount   int64         `json:"request_count"`
	ResponseTime   time.Duration `json:"response_time"`
	ErrorCount     int64         `json:"error_count"`
	CacheHitRate   float64       `json:"cache_hit_rate"`
	BandwidthBytes int64         `json:"bandwidth_bytes"`
}

// Router defines the main routing interface
type Router interface {
	// Route resolution
	ResolveRoute(r *http.Request) (*RouteMatch, error)
	GetTenantContext(r *http.Request) (*TenantContext, error)
	
	// Middleware management
	AddMiddleware(middleware Middleware) error
	RemoveMiddleware(name string) error
	GetMiddlewares() []Middleware
	
	// Route management
	AddTenantRoute(rule *TenantRoutingRule) error
	RemoveTenantRoute(tenantID string) error
	UpdateTenantRoute(rule *TenantRoutingRule) error
	GetTenantRoute(tenantID string) (*TenantRoutingRule, error)
	
	// Health and metrics
	GetHealthStatus() map[string]interface{}
	GetMetrics(tenantID string, from, to time.Time) (*RoutingMetrics, error)
	
	// Configuration
	UpdateConfig(config *RouteConfig) error
	GetConfig() *RouteConfig
}

// TenantResolver defines interface for tenant resolution
type TenantResolver interface {
	ResolveTenantBySubdomain(ctx context.Context, subdomain string) (*tenant.Tenant, error)
	ResolveTenantByDomain(ctx context.Context, domain string) (*tenant.Tenant, error)
	ResolveTenantByPath(ctx context.Context, path string) (*tenant.Tenant, error)
	CacheTenant(tenant *tenant.Tenant, ttl time.Duration) error
	InvalidateCache(tenantID string) error
}

// Helper functions for request handling

// ExtractSubdomain extracts subdomain from host header
func ExtractSubdomain(host, baseDomain string) string {
	if !strings.HasSuffix(host, baseDomain) {
		return ""
	}
	
	subdomain := strings.TrimSuffix(host, "."+baseDomain)
	if subdomain == baseDomain {
		return ""
	}
	
	// Handle www prefix
	if strings.HasPrefix(subdomain, "www.") {
		return ""
	}
	
	return subdomain
}

// IsCustomDomain checks if the host is a custom domain
func IsCustomDomain(host, baseDomain string) bool {
	if strings.HasSuffix(host, baseDomain) {
		return false
	}
	
	// Check if it's a subdomain of base domain
	if strings.Contains(host, baseDomain) {
		return false
	}
	
	return true
}

// ExtractPathTenant extracts tenant from path
func ExtractPathTenant(path, prefix string) string {
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	
	remaining := strings.TrimPrefix(path, prefix)
	parts := strings.Split(remaining, "/")
	
	if len(parts) > 0 && parts[0] != "" {
		return parts[0]
	}
	
	return ""
}

// BuildTenantURL builds URL for tenant
func BuildTenantURL(tenant *tenant.Tenant, baseDomain string, useSSL bool) string {
	scheme := "http"
	if useSSL {
		scheme = "https"
	}
	
	if tenant.CustomDomain != "" {
		return fmt.Sprintf("%s://%s", scheme, tenant.CustomDomain)
	}
	
	return fmt.Sprintf("%s://%s.%s", scheme, tenant.Subdomain, baseDomain)
}

// Context keys for tenant information
type contextKey string

const (
	TenantContextKey   contextKey = "tenant"
	TenantIDContextKey contextKey = "tenant_id"
	RouteTypeContextKey contextKey = "route_type"
)

// GetTenantFromContext retrieves tenant from request context
func GetTenantFromContext(ctx context.Context) (*tenant.Tenant, bool) {
	tenant, ok := ctx.Value(TenantContextKey).(*tenant.Tenant)
	return tenant, ok
}

// GetTenantIDFromContext retrieves tenant ID from request context
func GetTenantIDFromContext(ctx context.Context) (string, bool) {
	tenantID, ok := ctx.Value(TenantIDContextKey).(string)
	return tenantID, ok
}

// GetRouteTypeFromContext retrieves route type from request context
func GetRouteTypeFromContext(ctx context.Context) (RouteType, bool) {
	routeType, ok := ctx.Value(RouteTypeContextKey).(RouteType)
	return routeType, ok
}