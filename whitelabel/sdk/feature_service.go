package whitelabelsdk

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// FeatureService provides methods for feature management operations
type FeatureService struct {
	client *Client
}

// NewFeatureService creates a new feature service
func NewFeatureService(client *Client) *FeatureService {
	return &FeatureService{client: client}
}

// FeatureToggle defines a feature that can be enabled/disabled per tenant
type FeatureToggle struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Type         FeatureType            `json:"type"`
	DefaultValue interface{}            `json:"default_value"`
	Config       map[string]interface{} `json:"config"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// FeatureType defines the type of feature toggle
type FeatureType string

const (
	FeatureTypeBoolean FeatureType = "boolean"
	FeatureTypeString  FeatureType = "string"
	FeatureTypeNumber  FeatureType = "number"
	FeatureTypeJSON    FeatureType = "json"
)

// TenantFeature represents a feature setting for a specific tenant
type TenantFeature struct {
	ID        string      `json:"id"`
	TenantID  string      `json:"tenant_id"`
	FeatureID string      `json:"feature_id"`
	Value     interface{} `json:"value"`
	Enabled   bool        `json:"enabled"`
	CreatedAt time.Time   `json:"created_at"`
	UpdatedAt time.Time   `json:"updated_at"`
}

// FeatureCreateRequest represents a request to create a new feature
type FeatureCreateRequest struct {
	Name         string                 `json:"name" validate:"required"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Type         FeatureType            `json:"type" validate:"required"`
	DefaultValue interface{}            `json:"default_value"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

// FeatureUpdateRequest represents a request to update a feature
type FeatureUpdateRequest struct {
	Name         string                 `json:"name,omitempty"`
	Description  string                 `json:"description,omitempty"`
	Category     string                 `json:"category,omitempty"`
	DefaultValue interface{}            `json:"default_value,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
}

// TenantFeatureUpdateRequest represents a request to update a tenant feature
type TenantFeatureUpdateRequest struct {
	Value   interface{} `json:"value"`
	Enabled bool        `json:"enabled"`
}

// BulkFeatureUpdateRequest represents a request to bulk update features
type BulkFeatureUpdateRequest struct {
	TenantIDs []string               `json:"tenant_ids" validate:"required"`
	Features  map[string]interface{} `json:"features" validate:"required"`
}

// FeatureListOptions represents options for listing features
type FeatureListOptions struct {
	ListOptions
	Category []string      `json:"category,omitempty"`
	Type     []FeatureType `json:"type,omitempty"`
}

// CreateFeature creates a new feature toggle
func (s *FeatureService) CreateFeature(ctx context.Context, request *FeatureCreateRequest) (*FeatureToggle, error) {
	if request == nil {
		return nil, fmt.Errorf("create request cannot be nil")
	}
	
	var feature FeatureToggle
	err := s.client.makeRequest(ctx, http.MethodPost, "/api/v1/features", request, &feature)
	if err != nil {
		return nil, fmt.Errorf("failed to create feature: %w", err)
	}
	
	return &feature, nil
}

// GetFeature retrieves a feature by ID
func (s *FeatureService) GetFeature(ctx context.Context, featureID string) (*FeatureToggle, error) {
	if featureID == "" {
		return nil, fmt.Errorf("feature ID cannot be empty")
	}
	
	var feature FeatureToggle
	endpoint := fmt.Sprintf("/api/v1/features/%s", featureID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &feature)
	if err != nil {
		return nil, fmt.Errorf("failed to get feature: %w", err)
	}
	
	return &feature, nil
}

// UpdateFeature updates a feature
func (s *FeatureService) UpdateFeature(ctx context.Context, featureID string, request *FeatureUpdateRequest) (*FeatureToggle, error) {
	if featureID == "" {
		return nil, fmt.Errorf("feature ID cannot be empty")
	}
	if request == nil {
		return nil, fmt.Errorf("update request cannot be nil")
	}
	
	var feature FeatureToggle
	endpoint := fmt.Sprintf("/api/v1/features/%s", featureID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, request, &feature)
	if err != nil {
		return nil, fmt.Errorf("failed to update feature: %w", err)
	}
	
	return &feature, nil
}

// DeleteFeature deletes a feature
func (s *FeatureService) DeleteFeature(ctx context.Context, featureID string) error {
	if featureID == "" {
		return fmt.Errorf("feature ID cannot be empty")
	}
	
	endpoint := fmt.Sprintf("/api/v1/features/%s", featureID)
	err := s.client.makeRequest(ctx, http.MethodDelete, endpoint, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to delete feature: %w", err)
	}
	
	return nil
}

// ListFeatures retrieves a list of features
func (s *FeatureService) ListFeatures(ctx context.Context, options *FeatureListOptions) (*ListResponse[FeatureToggle], error) {
	var response ListResponse[FeatureToggle]
	
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
		}
		
		if options.Filter != nil {
			if options.Filter.Search != "" {
				req.SetQueryParam("search", options.Filter.Search)
			}
		}
		
		for _, category := range options.Category {
			req.SetQueryParam("category", category)
		}
		
		for _, featureType := range options.Type {
			req.SetQueryParam("type", string(featureType))
		}
	}
	
	resp, err := req.Get("/api/v1/features")
	if err != nil {
		return nil, fmt.Errorf("failed to list features: %w", err)
	}
	
	if resp.IsError() {
		return nil, fmt.Errorf("failed to list features with status %d: %s", resp.StatusCode(), resp.String())
	}
	
	return &response, nil
}

// SetTenantFeature enables/disables a feature for a tenant
func (s *FeatureService) SetTenantFeature(ctx context.Context, tenantID, featureID string, enabled bool, value interface{}) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if featureID == "" {
		return fmt.Errorf("feature ID cannot be empty")
	}
	
	request := &TenantFeatureUpdateRequest{
		Value:   value,
		Enabled: enabled,
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features/%s", tenantID, featureID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to set tenant feature: %w", err)
	}
	
	return nil
}

// GetTenantFeature retrieves a feature setting for a tenant
func (s *FeatureService) GetTenantFeature(ctx context.Context, tenantID, featureID string) (*TenantFeature, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if featureID == "" {
		return nil, fmt.Errorf("feature ID cannot be empty")
	}
	
	var tenantFeature TenantFeature
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features/%s", tenantID, featureID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &tenantFeature)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant feature: %w", err)
	}
	
	return &tenantFeature, nil
}

// GetTenantFeatures retrieves all feature settings for a tenant
func (s *FeatureService) GetTenantFeatures(ctx context.Context, tenantID string) (map[string]*TenantFeature, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var response struct {
		Features map[string]*TenantFeature `json:"features"`
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &response)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant features: %w", err)
	}
	
	return response.Features, nil
}

// IsFeatureEnabled checks if a feature is enabled for a tenant
func (s *FeatureService) IsFeatureEnabled(ctx context.Context, tenantID, featureID string) (bool, error) {
	if tenantID == "" {
		return false, fmt.Errorf("tenant ID cannot be empty")
	}
	if featureID == "" {
		return false, fmt.Errorf("feature ID cannot be empty")
	}
	
	var response struct {
		Enabled bool `json:"enabled"`
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features/%s/enabled", tenantID, featureID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &response)
	if err != nil {
		return false, fmt.Errorf("failed to check if feature is enabled: %w", err)
	}
	
	return response.Enabled, nil
}

// GetFeatureValue gets the value of a feature for a tenant
func (s *FeatureService) GetFeatureValue(ctx context.Context, tenantID, featureID string) (interface{}, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if featureID == "" {
		return nil, fmt.Errorf("feature ID cannot be empty")
	}
	
	var response struct {
		Value interface{} `json:"value"`
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features/%s/value", tenantID, featureID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &response)
	if err != nil {
		return nil, fmt.Errorf("failed to get feature value: %w", err)
	}
	
	return response.Value, nil
}

// BulkEnableFeature enables a feature for multiple tenants
func (s *FeatureService) BulkEnableFeature(ctx context.Context, featureID string, tenantIDs []string) error {
	if featureID == "" {
		return fmt.Errorf("feature ID cannot be empty")
	}
	if len(tenantIDs) == 0 {
		return fmt.Errorf("tenant IDs cannot be empty")
	}
	
	request := map[string]interface{}{
		"tenant_ids": tenantIDs,
		"enabled":    true,
	}
	
	endpoint := fmt.Sprintf("/api/v1/features/%s/bulk-enable", featureID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to bulk enable feature: %w", err)
	}
	
	return nil
}

// BulkDisableFeature disables a feature for multiple tenants
func (s *FeatureService) BulkDisableFeature(ctx context.Context, featureID string, tenantIDs []string) error {
	if featureID == "" {
		return fmt.Errorf("feature ID cannot be empty")
	}
	if len(tenantIDs) == 0 {
		return fmt.Errorf("tenant IDs cannot be empty")
	}
	
	request := map[string]interface{}{
		"tenant_ids": tenantIDs,
		"enabled":    false,
	}
	
	endpoint := fmt.Sprintf("/api/v1/features/%s/bulk-disable", featureID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to bulk disable feature: %w", err)
	}
	
	return nil
}

// BulkUpdateTenantFeatures updates multiple features for a tenant
func (s *FeatureService) BulkUpdateTenantFeatures(ctx context.Context, tenantID string, features map[string]interface{}) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if len(features) == 0 {
		return fmt.Errorf("features cannot be empty")
	}
	
	request := map[string]interface{}{
		"features": features,
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features/bulk-update", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to bulk update tenant features: %w", err)
	}
	
	return nil
}

// CopyFeaturesFromTenant copies all feature settings from one tenant to another
func (s *FeatureService) CopyFeaturesFromTenant(ctx context.Context, sourceTenantID, targetTenantID string) error {
	if sourceTenantID == "" {
		return fmt.Errorf("source tenant ID cannot be empty")
	}
	if targetTenantID == "" {
		return fmt.Errorf("target tenant ID cannot be empty")
	}
	
	request := map[string]string{
		"source_tenant_id": sourceTenantID,
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/features/copy", targetTenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to copy features from tenant: %w", err)
	}
	
	return nil
}

// GetFeatureUsageStats retrieves usage statistics for a feature
func (s *FeatureService) GetFeatureUsageStats(ctx context.Context, featureID string) (*FeatureUsageStats, error) {
	if featureID == "" {
		return nil, fmt.Errorf("feature ID cannot be empty")
	}
	
	var stats FeatureUsageStats
	endpoint := fmt.Sprintf("/api/v1/features/%s/usage", featureID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &stats)
	if err != nil {
		return nil, fmt.Errorf("failed to get feature usage stats: %w", err)
	}
	
	return &stats, nil
}

// FeatureUsageStats represents usage statistics for a feature
type FeatureUsageStats struct {
	FeatureID         string    `json:"feature_id"`
	TotalTenants      int64     `json:"total_tenants"`
	EnabledTenants    int64     `json:"enabled_tenants"`
	DisabledTenants   int64     `json:"disabled_tenants"`
	EnabledPercent    float64   `json:"enabled_percent"`
	LastUsed          time.Time `json:"last_used"`
	UsageByPlan       map[string]int64 `json:"usage_by_plan"`
	UsageByCategory   map[string]int64 `json:"usage_by_category"`
	RecentlyEnabled   []string  `json:"recently_enabled"`
	RecentlyDisabled  []string  `json:"recently_disabled"`
}

// Feature flag helpers for common operations

// EnableFeatureForTenant enables a boolean feature for a tenant
func (s *FeatureService) EnableFeatureForTenant(ctx context.Context, tenantID, featureID string) error {
	return s.SetTenantFeature(ctx, tenantID, featureID, true, true)
}

// DisableFeatureForTenant disables a feature for a tenant
func (s *FeatureService) DisableFeatureForTenant(ctx context.Context, tenantID, featureID string) error {
	return s.SetTenantFeature(ctx, tenantID, featureID, false, false)
}

// SetFeatureString sets a string feature value for a tenant
func (s *FeatureService) SetFeatureString(ctx context.Context, tenantID, featureID, value string) error {
	return s.SetTenantFeature(ctx, tenantID, featureID, true, value)
}

// SetFeatureNumber sets a number feature value for a tenant
func (s *FeatureService) SetFeatureNumber(ctx context.Context, tenantID, featureID string, value float64) error {
	return s.SetTenantFeature(ctx, tenantID, featureID, true, value)
}

// SetFeatureJSON sets a JSON feature value for a tenant
func (s *FeatureService) SetFeatureJSON(ctx context.Context, tenantID, featureID string, value map[string]interface{}) error {
	return s.SetTenantFeature(ctx, tenantID, featureID, true, value)
}