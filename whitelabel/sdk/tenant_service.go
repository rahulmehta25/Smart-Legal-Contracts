package whitelabelsdk

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// TenantService provides methods for tenant operations
type TenantService struct {
	client *Client
}

// NewTenantService creates a new tenant service
func NewTenantService(client *Client) *TenantService {
	return &TenantService{client: client}
}

// Tenant represents a tenant in the white-label platform
type Tenant struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Subdomain         string            `json:"subdomain"`
	CustomDomain      string            `json:"custom_domain,omitempty"`
	Status            TenantStatus      `json:"status"`
	IsolationStrategy IsolationStrategy `json:"isolation_strategy"`
	Plan              string            `json:"plan"`
	ResourceQuota     ResourceQuota     `json:"resource_quota"`
	Config            TenantConfig      `json:"config"`
	CreatedAt         time.Time         `json:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at"`
	LastAccessedAt    *time.Time        `json:"last_accessed_at,omitempty"`
	
	// Admin contact information
	AdminEmail    string `json:"admin_email"`
	AdminName     string `json:"admin_name"`
	AdminPhone    string `json:"admin_phone,omitempty"`
	
	// Billing information
	BillingEmail  string     `json:"billing_email,omitempty"`
	BillingPlan   string     `json:"billing_plan,omitempty"`
	BillingCycle  string     `json:"billing_cycle,omitempty"`
	TrialEndsAt   *time.Time `json:"trial_ends_at,omitempty"`
	
	// Performance metrics
	CurrentUsers    int64 `json:"current_users"`
	StorageUsedMB   int64 `json:"storage_used_mb"`
	MonthlyRequests int64 `json:"monthly_requests"`
	BandwidthUsedMB int64 `json:"bandwidth_used_mb"`
}

// TenantStatus represents the status of a tenant
type TenantStatus string

const (
	TenantStatusActive    TenantStatus = "active"
	TenantStatusSuspended TenantStatus = "suspended"
	TenantStatusInactive  TenantStatus = "inactive"
	TenantStatusPending   TenantStatus = "pending"
)

// IsolationStrategy represents the data isolation strategy
type IsolationStrategy string

const (
	DatabasePerTenant IsolationStrategy = "database_per_tenant"
	SchemaPerTenant   IsolationStrategy = "schema_per_tenant"
	RowLevel          IsolationStrategy = "row_level"
)

// ResourceQuota defines resource limits for a tenant
type ResourceQuota struct {
	MaxUsers       int64   `json:"max_users"`
	MaxStorage     int64   `json:"max_storage_mb"`
	MaxAPIRequests int64   `json:"max_api_requests_per_hour"`
	MaxBandwidth   int64   `json:"max_bandwidth_mb"`
	CPULimit       float64 `json:"cpu_limit"`
	MemoryLimit    int64   `json:"memory_limit_mb"`
}

// TenantConfig holds tenant-specific configuration
type TenantConfig struct {
	Features         map[string]bool `json:"features"`
	APIKeys          []string        `json:"api_keys,omitempty"`
	WebhookURL       string          `json:"webhook_url,omitempty"`
	TimeZone         string          `json:"timezone"`
	Language         string          `json:"language"`
	CustomDomains    []string        `json:"custom_domains,omitempty"`
	SSLCertificates  []string        `json:"ssl_certificates,omitempty"`
	BackupSchedule   string          `json:"backup_schedule,omitempty"`
	RetentionPolicy  int             `json:"retention_policy_days"`
}

// TenantCreateRequest represents a request to create a new tenant
type TenantCreateRequest struct {
	Name              string            `json:"name" validate:"required"`
	Subdomain         string            `json:"subdomain" validate:"required"`
	CustomDomain      string            `json:"custom_domain,omitempty"`
	Plan              string            `json:"plan" validate:"required"`
	IsolationStrategy IsolationStrategy `json:"isolation_strategy" validate:"required"`
	AdminEmail        string            `json:"admin_email" validate:"required,email"`
	AdminName         string            `json:"admin_name" validate:"required"`
	AdminPhone        string            `json:"admin_phone,omitempty"`
	BillingEmail      string            `json:"billing_email,omitempty"`
	Config            TenantConfig      `json:"config,omitempty"`
}

// TenantUpdateRequest represents a request to update tenant information
type TenantUpdateRequest struct {
	Name           string         `json:"name,omitempty"`
	CustomDomain   string         `json:"custom_domain,omitempty"`
	Status         TenantStatus   `json:"status,omitempty"`
	Plan           string         `json:"plan,omitempty"`
	ResourceQuota  *ResourceQuota `json:"resource_quota,omitempty"`
	Config         *TenantConfig  `json:"config,omitempty"`
	AdminEmail     string         `json:"admin_email,omitempty"`
	AdminName      string         `json:"admin_name,omitempty"`
	AdminPhone     string         `json:"admin_phone,omitempty"`
	BillingEmail   string         `json:"billing_email,omitempty"`
	BillingPlan    string         `json:"billing_plan,omitempty"`
	BillingCycle   string         `json:"billing_cycle,omitempty"`
}

// TenantUsageMetrics tracks real-time usage for performance isolation
type TenantUsageMetrics struct {
	TenantID             string        `json:"tenant_id"`
	Timestamp            time.Time     `json:"timestamp"`
	ActiveUsers          int64         `json:"active_users"`
	APIRequests          int64         `json:"api_requests"`
	StorageUsedMB        int64         `json:"storage_used_mb"`
	BandwidthUsedMB      int64         `json:"bandwidth_used_mb"`
	CPUUsagePercent      float64       `json:"cpu_usage_percent"`
	MemoryUsageMB        int64         `json:"memory_usage_mb"`
	ResponseTimeMS       float64       `json:"response_time_ms"`
	ErrorRate            float64       `json:"error_rate"`
}

// TenantListOptions represents options for listing tenants
type TenantListOptions struct {
	ListOptions
	Status            []TenantStatus      `json:"status,omitempty"`
	Plan              []string            `json:"plan,omitempty"`
	IsolationStrategy []IsolationStrategy `json:"isolation_strategy,omitempty"`
	SubdomainSearch   string              `json:"subdomain_search,omitempty"`
}

// Create creates a new tenant
func (s *TenantService) Create(ctx context.Context, request *TenantCreateRequest) (*Tenant, error) {
	if request == nil {
		return nil, fmt.Errorf("create request cannot be nil")
	}
	
	var tenant Tenant
	err := s.client.makeRequest(ctx, http.MethodPost, "/api/v1/tenants", request, &tenant)
	if err != nil {
		return nil, fmt.Errorf("failed to create tenant: %w", err)
	}
	
	return &tenant, nil
}

// Get retrieves a tenant by ID
func (s *TenantService) Get(ctx context.Context, tenantID string) (*Tenant, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var tenant Tenant
	endpoint := fmt.Sprintf("/api/v1/tenants/%s", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &tenant)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	return &tenant, nil
}

// GetBySubdomain retrieves a tenant by subdomain
func (s *TenantService) GetBySubdomain(ctx context.Context, subdomain string) (*Tenant, error) {
	if subdomain == "" {
		return nil, fmt.Errorf("subdomain cannot be empty")
	}
	
	var tenant Tenant
	endpoint := fmt.Sprintf("/api/v1/tenants/subdomain/%s", subdomain)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &tenant)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant by subdomain: %w", err)
	}
	
	return &tenant, nil
}

// Update updates a tenant
func (s *TenantService) Update(ctx context.Context, tenantID string, request *TenantUpdateRequest) (*Tenant, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if request == nil {
		return nil, fmt.Errorf("update request cannot be nil")
	}
	
	var tenant Tenant
	endpoint := fmt.Sprintf("/api/v1/tenants/%s", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, request, &tenant)
	if err != nil {
		return nil, fmt.Errorf("failed to update tenant: %w", err)
	}
	
	return &tenant, nil
}

// Delete deletes a tenant
func (s *TenantService) Delete(ctx context.Context, tenantID string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s", tenantID)
	err := s.client.makeRequest(ctx, http.MethodDelete, endpoint, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to delete tenant: %w", err)
	}
	
	return nil
}

// List retrieves a list of tenants
func (s *TenantService) List(ctx context.Context, options *TenantListOptions) (*ListResponse[Tenant], error) {
	var response ListResponse[Tenant]
	
	req := s.client.httpClient.R().
		SetContext(ctx).
		SetResult(&response)
	
	// Add query parameters if options provided
	if options != nil {
		if options.Pagination != nil {
			if options.Pagination.Page > 0 {
				req.SetQueryParam("page", fmt.Sprintf("%d", options.Pagination.Page))
			}
			if options.Pagination.PageSize > 0 {
				req.SetQueryParam("page_size", fmt.Sprintf("%d", options.Pagination.PageSize))
			}
			if options.Pagination.Limit > 0 {
				req.SetQueryParam("limit", fmt.Sprintf("%d", options.Pagination.Limit))
			}
			if options.Pagination.Offset > 0 {
				req.SetQueryParam("offset", fmt.Sprintf("%d", options.Pagination.Offset))
			}
		}
		
		if options.Filter != nil {
			if options.Filter.Search != "" {
				req.SetQueryParam("search", options.Filter.Search)
			}
		}
		
		for _, status := range options.Status {
			req.SetQueryParam("status", string(status))
		}
		
		for _, plan := range options.Plan {
			req.SetQueryParam("plan", plan)
		}
		
		if options.SubdomainSearch != "" {
			req.SetQueryParam("subdomain_search", options.SubdomainSearch)
		}
	}
	
	resp, err := req.Get("/api/v1/tenants")
	if err != nil {
		return nil, fmt.Errorf("failed to list tenants: %w", err)
	}
	
	if resp.IsError() {
		return nil, fmt.Errorf("failed to list tenants with status %d: %s", resp.StatusCode(), resp.String())
	}
	
	return &response, nil
}

// Suspend suspends a tenant
func (s *TenantService) Suspend(ctx context.Context, tenantID, reason string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if reason == "" {
		return fmt.Errorf("reason cannot be empty")
	}
	
	request := map[string]string{"reason": reason}
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/suspend", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to suspend tenant: %w", err)
	}
	
	return nil
}

// Reactivate reactivates a suspended tenant
func (s *TenantService) Reactivate(ctx context.Context, tenantID string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/reactivate", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to reactivate tenant: %w", err)
	}
	
	return nil
}

// GetUsage retrieves usage metrics for a tenant
func (s *TenantService) GetUsage(ctx context.Context, tenantID string, from, to *time.Time) ([]*TenantUsageMetrics, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	req := s.client.httpClient.R().SetContext(ctx)
	
	if from != nil {
		req.SetQueryParam("from", from.Format(time.RFC3339))
	}
	if to != nil {
		req.SetQueryParam("to", to.Format(time.RFC3339))
	}
	
	var response struct {
		Usage []*TenantUsageMetrics `json:"usage"`
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/usage", tenantID)
	resp, err := req.SetResult(&response).Get(endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant usage: %w", err)
	}
	
	if resp.IsError() {
		return nil, fmt.Errorf("failed to get tenant usage with status %d: %s", resp.StatusCode(), resp.String())
	}
	
	return response.Usage, nil
}

// UpdateUsage updates real-time usage metrics
func (s *TenantService) UpdateUsage(ctx context.Context, metrics *TenantUsageMetrics) error {
	if metrics == nil {
		return fmt.Errorf("metrics cannot be nil")
	}
	if metrics.TenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/usage", metrics.TenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, metrics, nil)
	if err != nil {
		return fmt.Errorf("failed to update tenant usage: %w", err)
	}
	
	return nil
}

// CheckQuota checks if tenant has exceeded any quota limits
func (s *TenantService) CheckQuota(ctx context.Context, tenantID string) (*QuotaStatus, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var quotaStatus QuotaStatus
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/quota/check", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &quotaStatus)
	if err != nil {
		return nil, fmt.Errorf("failed to check tenant quota: %w", err)
	}
	
	return &quotaStatus, nil
}

// QuotaStatus represents the quota status for a tenant
type QuotaStatus struct {
	TenantID        string   `json:"tenant_id"`
	ExceedsQuota    bool     `json:"exceeds_quota"`
	Violations      []string `json:"violations"`
	CurrentUsage    ResourceUsage `json:"current_usage"`
	QuotaLimits     ResourceQuota `json:"quota_limits"`
	UtilizationPct  map[string]float64 `json:"utilization_percent"`
}

// ResourceUsage represents current resource usage
type ResourceUsage struct {
	Users       int64   `json:"users"`
	StorageMB   int64   `json:"storage_mb"`
	APIRequests int64   `json:"api_requests"`
	BandwidthMB int64   `json:"bandwidth_mb"`
	CPU         float64 `json:"cpu"`
	MemoryMB    int64   `json:"memory_mb"`
}

// IsActive checks if the tenant is active
func (t *Tenant) IsActive() bool {
	return t.Status == TenantStatusActive
}

// IsTrialExpired checks if the tenant trial has expired
func (t *Tenant) IsTrialExpired() bool {
	return t.TrialEndsAt != nil && time.Now().After(*t.TrialEndsAt)
}

// ExceedsQuota checks if tenant exceeds any resource quota
func (t *Tenant) ExceedsQuota() bool {
	quota := t.ResourceQuota
	return t.CurrentUsers > quota.MaxUsers ||
		   t.StorageUsedMB > quota.MaxStorage ||
		   t.BandwidthUsedMB > quota.MaxBandwidth
}

// GetUtilizationPercentage calculates resource utilization percentage
func (t *Tenant) GetUtilizationPercentage() map[string]float64 {
	quota := t.ResourceQuota
	utilization := make(map[string]float64)
	
	if quota.MaxUsers > 0 {
		utilization["users"] = float64(t.CurrentUsers) / float64(quota.MaxUsers) * 100
	}
	if quota.MaxStorage > 0 {
		utilization["storage"] = float64(t.StorageUsedMB) / float64(quota.MaxStorage) * 100
	}
	if quota.MaxBandwidth > 0 {
		utilization["bandwidth"] = float64(t.BandwidthUsedMB) / float64(quota.MaxBandwidth) * 100
	}
	
	return utilization
}