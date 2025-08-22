package tenant

import (
	"context"
	"time"
)

// TenantFilter represents filtering options for tenant queries
type TenantFilter struct {
	Status            []TenantStatus      `json:"status,omitempty"`
	Plan              []string            `json:"plan,omitempty"`
	IsolationStrategy []IsolationStrategy `json:"isolation_strategy,omitempty"`
	CreatedAfter      *time.Time          `json:"created_after,omitempty"`
	CreatedBefore     *time.Time          `json:"created_before,omitempty"`
	LastAccessedAfter *time.Time          `json:"last_accessed_after,omitempty"`
	Search            string              `json:"search,omitempty"` // Search in name, subdomain, admin_email
	Limit             int                 `json:"limit,omitempty"`
	Offset            int                 `json:"offset,omitempty"`
	SortBy            string              `json:"sort_by,omitempty"`
	SortOrder         string              `json:"sort_order,omitempty"` // ASC or DESC
}

// Repository defines the interface for tenant data operations
type Repository interface {
	// Tenant CRUD operations
	Create(ctx context.Context, tenant *Tenant) error
	GetByID(ctx context.Context, id string) (*Tenant, error)
	GetBySubdomain(ctx context.Context, subdomain string) (*Tenant, error)
	GetByCustomDomain(ctx context.Context, domain string) (*Tenant, error)
	Update(ctx context.Context, tenant *Tenant) error
	Delete(ctx context.Context, id string) error
	List(ctx context.Context, filter *TenantFilter) ([]*Tenant, error)
	Count(ctx context.Context, filter *TenantFilter) (int64, error)
	
	// Utility methods
	SubdomainExists(ctx context.Context, subdomain string) (bool, error)
	CustomDomainExists(ctx context.Context, domain string) (bool, error)
	UpdateLastAccessed(ctx context.Context, id string, timestamp time.Time) error
	
	// Usage metrics operations
	UpdateUsageMetrics(ctx context.Context, metrics *TenantUsageMetrics) error
	GetUsageMetrics(ctx context.Context, tenantID string, from, to time.Time) ([]*TenantUsageMetrics, error)
	GetAggregatedUsage(ctx context.Context, tenantID string, from, to time.Time) (*TenantUsageMetrics, error)
	
	// Event logging
	CreateEvent(ctx context.Context, event *TenantEvent) error
	GetEvents(ctx context.Context, tenantID string, limit, offset int) ([]*TenantEvent, error)
	
	// Bulk operations
	BulkUpdateStatus(ctx context.Context, tenantIDs []string, status TenantStatus) error
	BulkUpdateQuota(ctx context.Context, tenantIDs []string, quota ResourceQuota) error
	
	// Analytics
	GetTenantStats(ctx context.Context) (*TenantStats, error)
	GetTenantsByPlan(ctx context.Context) (map[string]int64, error)
	GetActiveTenantsCount(ctx context.Context) (int64, error)
	GetExpiredTrialsCount(ctx context.Context) (int64, error)
}

// TenantStats represents aggregate statistics about tenants
type TenantStats struct {
	TotalTenants    int64              `json:"total_tenants"`
	ActiveTenants   int64              `json:"active_tenants"`
	SuspendedTenants int64             `json:"suspended_tenants"`
	TrialTenants    int64              `json:"trial_tenants"`
	ExpiredTrials   int64              `json:"expired_trials"`
	TenantsByPlan   map[string]int64   `json:"tenants_by_plan"`
	TenantsByStrategy map[string]int64 `json:"tenants_by_strategy"`
	AverageUsers    float64            `json:"average_users_per_tenant"`
	TotalStorageGB  float64            `json:"total_storage_gb"`
	TotalBandwidthGB float64           `json:"total_bandwidth_gb"`
}