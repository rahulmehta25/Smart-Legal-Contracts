package tenant

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"time"
)

// IsolationStrategy defines how tenant data is isolated
type IsolationStrategy string

const (
	DatabasePerTenant IsolationStrategy = "database_per_tenant"
	SchemaPerTenant   IsolationStrategy = "schema_per_tenant"
	RowLevel          IsolationStrategy = "row_level"
)

// TenantStatus represents the current state of a tenant
type TenantStatus string

const (
	StatusActive    TenantStatus = "active"
	StatusSuspended TenantStatus = "suspended"
	StatusInactive  TenantStatus = "inactive"
	StatusPending   TenantStatus = "pending"
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

// Value implements driver.Valuer for database storage
func (rq ResourceQuota) Value() (driver.Value, error) {
	return json.Marshal(rq)
}

// Scan implements sql.Scanner for database retrieval
func (rq *ResourceQuota) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, rq)
}

// TenantConfig holds tenant-specific configuration
type TenantConfig struct {
	Features         map[string]bool `json:"features"`
	APIKeys          []string        `json:"api_keys"`
	WebhookURL       string          `json:"webhook_url"`
	TimeZone         string          `json:"timezone"`
	Language         string          `json:"language"`
	CustomDomains    []string        `json:"custom_domains"`
	SSLCertificates  []string        `json:"ssl_certificates"`
	BackupSchedule   string          `json:"backup_schedule"`
	RetentionPolicy  int             `json:"retention_policy_days"`
}

// Value implements driver.Valuer for database storage
func (tc TenantConfig) Value() (driver.Value, error) {
	return json.Marshal(tc)
}

// Scan implements sql.Scanner for database retrieval
func (tc *TenantConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, tc)
}

// Tenant represents a white-label client organization
type Tenant struct {
	ID                string            `json:"id" db:"id"`
	Name              string            `json:"name" db:"name"`
	Subdomain         string            `json:"subdomain" db:"subdomain"`
	CustomDomain      string            `json:"custom_domain" db:"custom_domain"`
	Status            TenantStatus      `json:"status" db:"status"`
	IsolationStrategy IsolationStrategy `json:"isolation_strategy" db:"isolation_strategy"`
	DatabaseName      string            `json:"database_name" db:"database_name"`
	SchemaName        string            `json:"schema_name" db:"schema_name"`
	Plan              string            `json:"plan" db:"plan"`
	ResourceQuota     ResourceQuota     `json:"resource_quota" db:"resource_quota"`
	Config            TenantConfig      `json:"config" db:"config"`
	CreatedAt         time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at" db:"updated_at"`
	LastAccessedAt    *time.Time        `json:"last_accessed_at" db:"last_accessed_at"`
	
	// Admin contact information
	AdminEmail    string `json:"admin_email" db:"admin_email"`
	AdminName     string `json:"admin_name" db:"admin_name"`
	AdminPhone    string `json:"admin_phone" db:"admin_phone"`
	
	// Billing information
	BillingEmail  string `json:"billing_email" db:"billing_email"`
	BillingPlan   string `json:"billing_plan" db:"billing_plan"`
	BillingCycle  string `json:"billing_cycle" db:"billing_cycle"`
	TrialEndsAt   *time.Time `json:"trial_ends_at" db:"trial_ends_at"`
	
	// Performance metrics
	CurrentUsers     int64 `json:"current_users" db:"current_users"`
	StorageUsedMB    int64 `json:"storage_used_mb" db:"storage_used_mb"`
	MonthlyRequests  int64 `json:"monthly_requests" db:"monthly_requests"`
	BandwidthUsedMB  int64 `json:"bandwidth_used_mb" db:"bandwidth_used_mb"`
}

// TenantUsageMetrics tracks real-time usage for performance isolation
type TenantUsageMetrics struct {
	TenantID          string    `json:"tenant_id" db:"tenant_id"`
	Timestamp         time.Time `json:"timestamp" db:"timestamp"`
	ActiveUsers       int64     `json:"active_users" db:"active_users"`
	APIRequests       int64     `json:"api_requests" db:"api_requests"`
	StorageUsedMB     int64     `json:"storage_used_mb" db:"storage_used_mb"`
	BandwidthUsedMB   int64     `json:"bandwidth_used_mb" db:"bandwidth_used_mb"`
	CPUUsagePercent   float64   `json:"cpu_usage_percent" db:"cpu_usage_percent"`
	MemoryUsageMB     int64     `json:"memory_usage_mb" db:"memory_usage_mb"`
	ResponseTimeMS    float64   `json:"response_time_ms" db:"response_time_ms"`
	ErrorRate         float64   `json:"error_rate" db:"error_rate"`
}

// TenantEvent represents events in tenant lifecycle
type TenantEvent struct {
	ID          string    `json:"id" db:"id"`
	TenantID    string    `json:"tenant_id" db:"tenant_id"`
	EventType   string    `json:"event_type" db:"event_type"`
	Description string    `json:"description" db:"description"`
	Metadata    string    `json:"metadata" db:"metadata"`
	CreatedAt   time.Time `json:"created_at" db:"created_at"`
	CreatedBy   string    `json:"created_by" db:"created_by"`
}

// IsActive checks if tenant is in active state
func (t *Tenant) IsActive() bool {
	return t.Status == StatusActive
}

// IsTrialExpired checks if tenant trial has expired
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

// GetDatabaseConnection returns database connection string based on isolation strategy
func (t *Tenant) GetDatabaseConnection(baseConn string) string {
	switch t.IsolationStrategy {
	case DatabasePerTenant:
		return fmt.Sprintf("%s dbname=%s", baseConn, t.DatabaseName)
	case SchemaPerTenant:
		return fmt.Sprintf("%s search_path=%s", baseConn, t.SchemaName)
	default:
		return baseConn
	}
}

// TenantCreateRequest represents a request to create a new tenant
type TenantCreateRequest struct {
	Name              string            `json:"name" validate:"required,min=2,max=100"`
	Subdomain         string            `json:"subdomain" validate:"required,min=3,max=50,alphanum"`
	CustomDomain      string            `json:"custom_domain,omitempty" validate:"omitempty,fqdn"`
	Plan              string            `json:"plan" validate:"required"`
	IsolationStrategy IsolationStrategy `json:"isolation_strategy" validate:"required"`
	AdminEmail        string            `json:"admin_email" validate:"required,email"`
	AdminName         string            `json:"admin_name" validate:"required,min=2,max=100"`
	AdminPhone        string            `json:"admin_phone,omitempty"`
	BillingEmail      string            `json:"billing_email,omitempty" validate:"omitempty,email"`
	Config            TenantConfig      `json:"config,omitempty"`
}

// TenantUpdateRequest represents a request to update tenant information
type TenantUpdateRequest struct {
	Name           string            `json:"name,omitempty" validate:"omitempty,min=2,max=100"`
	CustomDomain   string            `json:"custom_domain,omitempty" validate:"omitempty,fqdn"`
	Status         TenantStatus      `json:"status,omitempty"`
	Plan           string            `json:"plan,omitempty"`
	ResourceQuota  *ResourceQuota    `json:"resource_quota,omitempty"`
	Config         *TenantConfig     `json:"config,omitempty"`
	AdminEmail     string            `json:"admin_email,omitempty" validate:"omitempty,email"`
	AdminName      string            `json:"admin_name,omitempty" validate:"omitempty,min=2,max=100"`
	AdminPhone     string            `json:"admin_phone,omitempty"`
	BillingEmail   string            `json:"billing_email,omitempty" validate:"omitempty,email"`
	BillingPlan    string            `json:"billing_plan,omitempty"`
	BillingCycle   string            `json:"billing_cycle,omitempty"`
}