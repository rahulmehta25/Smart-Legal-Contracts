package routing

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/enterprise/whitelabel/pkg/tenant"
	"go.uber.org/zap"
)

// Service implements the Router interface
type Service struct {
	config           *RouteConfig
	tenantResolver   TenantResolver
	middlewares      []Middleware
	tenantRoutes     map[string]*TenantRoutingRule
	metrics          map[string]*RoutingMetrics
	cache            Cache
	rateLimiter      RateLimiter
	logger           *zap.Logger
	mu               sync.RWMutex
	metricsCollector MetricsCollector
}

// NewService creates a new routing service
func NewService(config *RouteConfig, tenantResolver TenantResolver, cache Cache, rateLimiter RateLimiter, logger *zap.Logger) *Service {
	return &Service{
		config:         config,
		tenantResolver: tenantResolver,
		middlewares:    []Middleware{},
		tenantRoutes:   make(map[string]*TenantRoutingRule),
		metrics:        make(map[string]*RoutingMetrics),
		cache:          cache,
		rateLimiter:    rateLimiter,
		logger:         logger,
	}
}

// ResolveRoute resolves the tenant for an incoming request
func (s *Service) ResolveRoute(r *http.Request) (*RouteMatch, error) {
	startTime := time.Now()
	defer func() {
		s.logger.Debug("Route resolution completed",
			zap.Duration("duration", time.Since(startTime)),
			zap.String("host", r.Host),
			zap.String("path", r.URL.Path))
	}()

	host := strings.ToLower(r.Host)
	
	// Remove port if present
	if colonIndex := strings.Index(host, ":"); colonIndex != -1 {
		host = host[:colonIndex]
	}

	// Try different routing strategies
	switch s.config.Strategy.Type {
	case RouteTypeSubdomain:
		return s.resolveBySubdomain(r, host)
	case RouteTypeCustomDomain:
		return s.resolveByCustomDomain(r, host)
	case RouteTypePath:
		return s.resolveByPath(r)
	default:
		return s.resolveMultiStrategy(r, host)
	}
}

// resolveBySubdomain resolves tenant by subdomain
func (s *Service) resolveBySubdomain(r *http.Request, host string) (*RouteMatch, error) {
	baseDomain := s.extractBaseDomain()
	subdomain := ExtractSubdomain(host, baseDomain)
	
	if subdomain == "" {
		if s.config.DefaultTenant != "" {
			return s.getDefaultTenantMatch()
		}
		return nil, fmt.Errorf("no subdomain found in host: %s", host)
	}

	// Check cache first
	if s.cache != nil {
		if cachedTenant := s.cache.GetTenant(subdomain); cachedTenant != nil {
			return &RouteMatch{
				Tenant:         cachedTenant,
				RouteType:      RouteTypeSubdomain,
				MatchedValue:   subdomain,
				IsCustomDomain: false,
			}, nil
		}
	}

	// Resolve from tenant resolver
	tenantObj, err := s.tenantResolver.ResolveTenantBySubdomain(r.Context(), subdomain)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve tenant by subdomain %s: %w", subdomain, err)
	}

	// Cache the result
	if s.cache != nil {
		s.cache.SetTenant(subdomain, tenantObj, s.config.CacheConfig.TTL)
	}

	return &RouteMatch{
		Tenant:         tenantObj,
		RouteType:      RouteTypeSubdomain,
		MatchedValue:   subdomain,
		IsCustomDomain: false,
	}, nil
}

// resolveByCustomDomain resolves tenant by custom domain
func (s *Service) resolveByCustomDomain(r *http.Request, host string) (*RouteMatch, error) {
	if !s.config.Strategy.CustomDomainEnabled {
		return nil, fmt.Errorf("custom domains are not enabled")
	}

	// Check if it's actually a custom domain
	baseDomain := s.extractBaseDomain()
	if !IsCustomDomain(host, baseDomain) {
		// Fallback to subdomain resolution
		return s.resolveBySubdomain(r, host)
	}

	// Check cache first
	cacheKey := "domain:" + host
	if s.cache != nil {
		if cachedTenant := s.cache.GetTenant(cacheKey); cachedTenant != nil {
			return &RouteMatch{
				Tenant:         cachedTenant,
				RouteType:      RouteTypeCustomDomain,
				MatchedValue:   host,
				IsCustomDomain: true,
			}, nil
		}
	}

	// Resolve from tenant resolver
	tenantObj, err := s.tenantResolver.ResolveTenantByDomain(r.Context(), host)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve tenant by domain %s: %w", host, err)
	}

	// Cache the result
	if s.cache != nil {
		s.cache.SetTenant(cacheKey, tenantObj, s.config.CacheConfig.TTL)
	}

	return &RouteMatch{
		Tenant:         tenantObj,
		RouteType:      RouteTypeCustomDomain,
		MatchedValue:   host,
		IsCustomDomain: true,
	}, nil
}

// resolveByPath resolves tenant by path prefix
func (s *Service) resolveByPath(r *http.Request) (*RouteMatch, error) {
	if s.config.Strategy.PathPrefix == "" {
		return nil, fmt.Errorf("path prefix not configured")
	}

	tenantID := ExtractPathTenant(r.URL.Path, s.config.Strategy.PathPrefix)
	if tenantID == "" {
		if s.config.DefaultTenant != "" {
			return s.getDefaultTenantMatch()
		}
		return nil, fmt.Errorf("no tenant found in path: %s", r.URL.Path)
	}

	// Check cache first
	cacheKey := "path:" + tenantID
	if s.cache != nil {
		if cachedTenant := s.cache.GetTenant(cacheKey); cachedTenant != nil {
			return &RouteMatch{
				Tenant:         cachedTenant,
				RouteType:      RouteTypePath,
				MatchedValue:   tenantID,
				IsCustomDomain: false,
			}, nil
		}
	}

	// Resolve from tenant resolver
	tenantObj, err := s.tenantResolver.ResolveTenantByPath(r.Context(), tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve tenant by path %s: %w", tenantID, err)
	}

	// Cache the result
	if s.cache != nil {
		s.cache.SetTenant(cacheKey, tenantObj, s.config.CacheConfig.TTL)
	}

	return &RouteMatch{
		Tenant:         tenantObj,
		RouteType:      RouteTypePath,
		MatchedValue:   tenantID,
		IsCustomDomain: false,
	}, nil
}

// resolveMultiStrategy tries multiple strategies
func (s *Service) resolveMultiStrategy(r *http.Request, host string) (*RouteMatch, error) {
	baseDomain := s.extractBaseDomain()
	
	// Try custom domain first
	if IsCustomDomain(host, baseDomain) && s.config.Strategy.CustomDomainEnabled {
		if match, err := s.resolveByCustomDomain(r, host); err == nil {
			return match, nil
		}
	}
	
	// Try subdomain
	if subdomain := ExtractSubdomain(host, baseDomain); subdomain != "" {
		if match, err := s.resolveBySubdomain(r, host); err == nil {
			return match, nil
		}
	}
	
	// Try path-based routing
	if s.config.Strategy.PathPrefix != "" {
		if match, err := s.resolveByPath(r); err == nil {
			return match, nil
		}
	}
	
	// Use default tenant if configured
	if s.config.DefaultTenant != "" {
		return s.getDefaultTenantMatch()
	}
	
	return nil, fmt.Errorf("no tenant found for request")
}

// GetTenantContext creates a tenant context from the request
func (s *Service) GetTenantContext(r *http.Request) (*TenantContext, error) {
	match, err := s.ResolveRoute(r)
	if err != nil {
		return nil, err
	}

	// Check rate limiting
	if s.config.RateLimitConfig.Enabled {
		key := s.getRateLimitKey(match.Tenant.ID, r)
		if !s.rateLimiter.Allow(key, s.config.RateLimitConfig) {
			return nil, fmt.Errorf("rate limit exceeded for tenant: %s", match.Tenant.ID)
		}
	}

	// Check if tenant is active
	if !match.Tenant.IsActive() {
		return nil, fmt.Errorf("tenant is not active: %s", match.Tenant.ID)
	}

	// Check if trial has expired
	if match.Tenant.IsTrialExpired() {
		return nil, fmt.Errorf("tenant trial has expired: %s", match.Tenant.ID)
	}

	tenantCtx := &TenantContext{
		Tenant:      match.Tenant,
		DetectedBy:  match.RouteType,
		RequestTime: time.Now(),
	}

	if match.IsCustomDomain {
		tenantCtx.CustomDomain = match.MatchedValue
	} else if match.RouteType == RouteTypeSubdomain {
		tenantCtx.Subdomain = match.MatchedValue
	}

	// Record metrics
	s.recordMetrics(match.Tenant.ID, r)

	return tenantCtx, nil
}

// Middleware management methods

// AddMiddleware adds a middleware to the routing service
func (s *Service) AddMiddleware(middleware Middleware) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	// Check if middleware already exists
	for i, m := range s.middlewares {
		if m.Name == middleware.Name {
			s.middlewares[i] = middleware
			return nil
		}
	}
	
	s.middlewares = append(s.middlewares, middleware)
	s.sortMiddlewares()
	
	return nil
}

// RemoveMiddleware removes a middleware by name
func (s *Service) RemoveMiddleware(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	for i, m := range s.middlewares {
		if m.Name == name {
			s.middlewares = append(s.middlewares[:i], s.middlewares[i+1:]...)
			return nil
		}
	}
	
	return fmt.Errorf("middleware not found: %s", name)
}

// GetMiddlewares returns all middlewares
func (s *Service) GetMiddlewares() []Middleware {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	middlewares := make([]Middleware, len(s.middlewares))
	copy(middlewares, s.middlewares)
	return middlewares
}

// Route management methods

// AddTenantRoute adds a routing rule for a tenant
func (s *Service) AddTenantRoute(rule *TenantRoutingRule) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.tenantRoutes[rule.TenantID] = rule
	
	// Invalidate cache for this tenant
	if s.cache != nil {
		s.cache.InvalidateTenant(rule.TenantID)
	}
	
	s.logger.Info("Added tenant routing rule", zap.String("tenant_id", rule.TenantID))
	return nil
}

// RemoveTenantRoute removes a routing rule for a tenant
func (s *Service) RemoveTenantRoute(tenantID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	delete(s.tenantRoutes, tenantID)
	
	// Invalidate cache for this tenant
	if s.cache != nil {
		s.cache.InvalidateTenant(tenantID)
	}
	
	s.logger.Info("Removed tenant routing rule", zap.String("tenant_id", tenantID))
	return nil
}

// UpdateTenantRoute updates a routing rule for a tenant
func (s *Service) UpdateTenantRoute(rule *TenantRoutingRule) error {
	return s.AddTenantRoute(rule)
}

// GetTenantRoute gets a routing rule for a tenant
func (s *Service) GetTenantRoute(tenantID string) (*TenantRoutingRule, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	rule, exists := s.tenantRoutes[tenantID]
	if !exists {
		return nil, fmt.Errorf("routing rule not found for tenant: %s", tenantID)
	}
	
	return rule, nil
}

// Health and metrics methods

// GetHealthStatus returns the health status of the routing service
func (s *Service) GetHealthStatus() map[string]interface{} {
	status := map[string]interface{}{
		"status":        "healthy",
		"timestamp":     time.Now(),
		"tenant_count":  len(s.tenantRoutes),
		"middleware_count": len(s.middlewares),
	}
	
	// Check cache health
	if s.cache != nil {
		status["cache"] = s.cache.GetHealthStatus()
	}
	
	// Check rate limiter health
	if s.rateLimiter != nil {
		status["rate_limiter"] = s.rateLimiter.GetHealthStatus()
	}
	
	return status
}

// GetMetrics returns routing metrics for a tenant
func (s *Service) GetMetrics(tenantID string, from, to time.Time) (*RoutingMetrics, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	metrics, exists := s.metrics[tenantID]
	if !exists {
		return &RoutingMetrics{
			TenantID:  tenantID,
			Timestamp: time.Now(),
		}, nil
	}
	
	return metrics, nil
}

// Configuration methods

// UpdateConfig updates the routing configuration
func (s *Service) UpdateConfig(config *RouteConfig) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.config = config
	s.logger.Info("Updated routing configuration")
	
	return nil
}

// GetConfig returns the current routing configuration
func (s *Service) GetConfig() *RouteConfig {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	return s.config
}

// Helper methods

// extractBaseDomain extracts the base domain from configuration
func (s *Service) extractBaseDomain() string {
	if s.config.FallbackDomain != "" {
		return s.config.FallbackDomain
	}
	
	// Extract from subdomain pattern
	pattern := s.config.Strategy.SubdomainPattern
	if pattern != "" {
		// Pattern format: "{tenant}.example.com"
		if strings.Contains(pattern, ".") {
			parts := strings.Split(pattern, ".")
			if len(parts) > 1 {
				return strings.Join(parts[1:], ".")
			}
		}
	}
	
	return "localhost"
}

// getDefaultTenantMatch returns a match for the default tenant
func (s *Service) getDefaultTenantMatch() (*RouteMatch, error) {
	// This would typically resolve the default tenant from the database
	// For now, return an error to indicate implementation needed
	return nil, fmt.Errorf("default tenant resolution not implemented")
}

// getRateLimitKey generates a rate limiting key
func (s *Service) getRateLimitKey(tenantID string, r *http.Request) string {
	if s.config.RateLimitConfig.ByIP {
		ip := s.getClientIP(r)
		return fmt.Sprintf("%s:%s", tenantID, ip)
	}
	
	if s.config.RateLimitConfig.ByTenant {
		return tenantID
	}
	
	return "global"
}

// getClientIP extracts client IP from request
func (s *Service) getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}
	
	// Check X-Real-IP header
	xri := r.Header.Get("X-Real-IP")
	if xri != "" {
		return strings.TrimSpace(xri)
	}
	
	// Fall back to RemoteAddr
	ip := r.RemoteAddr
	if colonIndex := strings.LastIndex(ip, ":"); colonIndex != -1 {
		ip = ip[:colonIndex]
	}
	
	return ip
}

// recordMetrics records routing metrics for a tenant
func (s *Service) recordMetrics(tenantID string, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	metrics, exists := s.metrics[tenantID]
	if !exists {
		metrics = &RoutingMetrics{
			TenantID:  tenantID,
			Timestamp: time.Now(),
		}
		s.metrics[tenantID] = metrics
	}
	
	metrics.RequestCount++
	metrics.Timestamp = time.Now()
	
	// Record bandwidth if available
	if contentLength := r.Header.Get("Content-Length"); contentLength != "" {
		// Parse content length and add to bandwidth
		// Implementation depends on parsing logic
	}
}

// sortMiddlewares sorts middlewares by order
func (s *Service) sortMiddlewares() {
	// Simple bubble sort by order field
	n := len(s.middlewares)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if s.middlewares[j].Order > s.middlewares[j+1].Order {
				s.middlewares[j], s.middlewares[j+1] = s.middlewares[j+1], s.middlewares[j]
			}
		}
	}
}