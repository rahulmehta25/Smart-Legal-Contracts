// Package whitelabelsdk provides a Go SDK for the White-label Platform API
package whitelabelsdk

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/go-resty/resty/v2"
	"golang.org/x/oauth2"
)

// Client is the main SDK client for interacting with the White-label Platform API
type Client struct {
	httpClient   *resty.Client
	baseURL      string
	apiKey       string
	tenantID     string
	userAgent    string
	timeout      time.Duration
	retryCount   int
	retryWait    time.Duration
	debug        bool
	
	// Service clients
	Tenants       *TenantService
	Customization *CustomizationService
	Isolation     *IsolationService
	Admin         *AdminService
	Monitoring    *MonitoringService
	Backup        *BackupService
	Analytics     *AnalyticsService
	Webhooks      *WebhookService
	Users         *UserService
	Features      *FeatureService
}

// Config holds configuration options for the SDK client
type Config struct {
	// BaseURL is the base URL of the White-label Platform API
	BaseURL string
	
	// APIKey for authentication
	APIKey string
	
	// TenantID for tenant-specific operations
	TenantID string
	
	// HTTPClient allows providing a custom HTTP client
	HTTPClient *http.Client
	
	// OAuth2Config for OAuth2 authentication
	OAuth2Config *oauth2.Config
	
	// Timeout for HTTP requests
	Timeout time.Duration
	
	// RetryCount for failed requests
	RetryCount int
	
	// RetryWait time between retries
	RetryWait time.Duration
	
	// Debug enables debug logging
	Debug bool
	
	// UserAgent for HTTP requests
	UserAgent string
	
	// RateLimitConfig for rate limiting
	RateLimit *RateLimitConfig
	
	// Custom headers to be sent with requests
	Headers map[string]string
}

// RateLimitConfig defines rate limiting configuration
type RateLimitConfig struct {
	RequestsPerSecond int
	BurstLimit        int
}

// NewClient creates a new SDK client with the provided configuration
func NewClient(config *Config) (*Client, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	
	if config.BaseURL == "" {
		return nil, fmt.Errorf("baseURL is required")
	}
	
	// Validate base URL
	if _, err := url.Parse(config.BaseURL); err != nil {
		return nil, fmt.Errorf("invalid baseURL: %w", err)
	}
	
	// Set defaults
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.RetryCount == 0 {
		config.RetryCount = 3
	}
	if config.RetryWait == 0 {
		config.RetryWait = 1 * time.Second
	}
	if config.UserAgent == "" {
		config.UserAgent = "whitelabel-sdk-go/1.0.0"
	}
	
	// Create resty client
	httpClient := resty.New()
	
	// Use custom HTTP client if provided
	if config.HTTPClient != nil {
		httpClient.SetHTTPClient(config.HTTPClient)
	}
	
	// Configure client
	httpClient.
		SetBaseURL(config.BaseURL).
		SetTimeout(config.Timeout).
		SetRetryCount(config.RetryCount).
		SetRetryWaitTime(config.RetryWait).
		SetHeader("User-Agent", config.UserAgent).
		SetHeader("Content-Type", "application/json").
		SetHeader("Accept", "application/json")
	
	// Set API key if provided
	if config.APIKey != "" {
		httpClient.SetHeader("Authorization", "Bearer "+config.APIKey)
	}
	
	// Set tenant ID header if provided
	if config.TenantID != "" {
		httpClient.SetHeader("X-Tenant-ID", config.TenantID)
	}
	
	// Set custom headers
	for key, value := range config.Headers {
		httpClient.SetHeader(key, value)
	}
	
	// Enable debug if configured
	if config.Debug {
		httpClient.SetDebug(true)
	}
	
	// Create client
	client := &Client{
		httpClient: httpClient,
		baseURL:    config.BaseURL,
		apiKey:     config.APIKey,
		tenantID:   config.TenantID,
		userAgent:  config.UserAgent,
		timeout:    config.Timeout,
		retryCount: config.RetryCount,
		retryWait:  config.RetryWait,
		debug:      config.Debug,
	}
	
	// Initialize service clients
	client.initServices()
	
	return client, nil
}

// NewClientWithAPIKey creates a new SDK client with API key authentication
func NewClientWithAPIKey(baseURL, apiKey, tenantID string) (*Client, error) {
	config := &Config{
		BaseURL:  baseURL,
		APIKey:   apiKey,
		TenantID: tenantID,
	}
	return NewClient(config)
}

// NewClientWithOAuth2 creates a new SDK client with OAuth2 authentication
func NewClientWithOAuth2(baseURL string, oauth2Config *oauth2.Config, token *oauth2.Token) (*Client, error) {
	httpClient := oauth2Config.Client(context.Background(), token)
	
	config := &Config{
		BaseURL:      baseURL,
		HTTPClient:   httpClient,
		OAuth2Config: oauth2Config,
	}
	return NewClient(config)
}

// initServices initializes all service clients
func (c *Client) initServices() {
	c.Tenants = NewTenantService(c)
	c.Customization = NewCustomizationService(c)
	c.Isolation = NewIsolationService(c)
	c.Admin = NewAdminService(c)
	c.Monitoring = NewMonitoringService(c)
	c.Backup = NewBackupService(c)
	c.Analytics = NewAnalyticsService(c)
	c.Webhooks = NewWebhookService(c)
	c.Users = NewUserService(c)
	c.Features = NewFeatureService(c)
}

// SetAPIKey updates the API key for authentication
func (c *Client) SetAPIKey(apiKey string) {
	c.apiKey = apiKey
	c.httpClient.SetHeader("Authorization", "Bearer "+apiKey)
}

// SetTenantID updates the tenant ID
func (c *Client) SetTenantID(tenantID string) {
	c.tenantID = tenantID
	c.httpClient.SetHeader("X-Tenant-ID", tenantID)
}

// SetDebug enables or disables debug logging
func (c *Client) SetDebug(debug bool) {
	c.debug = debug
	c.httpClient.SetDebug(debug)
}

// GetBaseURL returns the base URL of the API
func (c *Client) GetBaseURL() string {
	return c.baseURL
}

// GetTenantID returns the current tenant ID
func (c *Client) GetTenantID() string {
	return c.tenantID
}

// Health checks the health of the API
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var response HealthResponse
	
	resp, err := c.httpClient.R().
		SetContext(ctx).
		SetResult(&response).
		Get("/health")
	
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	
	if resp.IsError() {
		return nil, fmt.Errorf("health check failed with status %d: %s", resp.StatusCode(), resp.String())
	}
	
	return &response, nil
}

// Version gets the API version information
func (c *Client) Version(ctx context.Context) (*VersionResponse, error) {
	var response VersionResponse
	
	resp, err := c.httpClient.R().
		SetContext(ctx).
		SetResult(&response).
		Get("/version")
	
	if err != nil {
		return nil, fmt.Errorf("version check failed: %w", err)
	}
	
	if resp.IsError() {
		return nil, fmt.Errorf("version check failed with status %d: %s", resp.StatusCode(), resp.String())
	}
	
	return &response, nil
}

// makeRequest is a helper method for making HTTP requests
func (c *Client) makeRequest(ctx context.Context, method, endpoint string, body interface{}, result interface{}) error {
	req := c.httpClient.R().SetContext(ctx)
	
	if body != nil {
		req.SetBody(body)
	}
	
	if result != nil {
		req.SetResult(result)
	}
	
	var resp *resty.Response
	var err error
	
	switch method {
	case http.MethodGet:
		resp, err = req.Get(endpoint)
	case http.MethodPost:
		resp, err = req.Post(endpoint)
	case http.MethodPut:
		resp, err = req.Put(endpoint)
	case http.MethodPatch:
		resp, err = req.Patch(endpoint)
	case http.MethodDelete:
		resp, err = req.Delete(endpoint)
	default:
		return fmt.Errorf("unsupported HTTP method: %s", method)
	}
	
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	
	if resp.IsError() {
		var apiError APIError
		if jsonErr := json.Unmarshal(resp.Body(), &apiError); jsonErr == nil {
			apiError.StatusCode = resp.StatusCode()
			return &apiError
		}
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode(), resp.String())
	}
	
	return nil
}

// Common response types

// HealthResponse represents the response from the health endpoint
type HealthResponse struct {
	Status    string                 `json:"status"`
	Timestamp time.Time              `json:"timestamp"`
	Version   string                 `json:"version"`
	Services  map[string]interface{} `json:"services,omitempty"`
}

// VersionResponse represents the response from the version endpoint
type VersionResponse struct {
	Version     string    `json:"version"`
	BuildTime   time.Time `json:"build_time"`
	GitCommit   string    `json:"git_commit"`
	APIVersion  string    `json:"api_version"`
	GoVersion   string    `json:"go_version"`
}

// APIError represents an error returned by the API
type APIError struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	Details    string `json:"details,omitempty"`
	StatusCode int    `json:"status_code"`
	RequestID  string `json:"request_id,omitempty"`
}

// Error implements the error interface
func (e *APIError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("API error %d: %s - %s", e.StatusCode, e.Message, e.Details)
	}
	return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Message)
}

// IsAPIError checks if an error is an API error
func IsAPIError(err error) bool {
	_, ok := err.(*APIError)
	return ok
}

// GetAPIError returns the API error if the error is an API error
func GetAPIError(err error) (*APIError, bool) {
	apiError, ok := err.(*APIError)
	return apiError, ok
}

// PaginationParams represents pagination parameters
type PaginationParams struct {
	Page     int `json:"page,omitempty"`
	PageSize int `json:"page_size,omitempty"`
	Limit    int `json:"limit,omitempty"`
	Offset   int `json:"offset,omitempty"`
}

// PaginationResponse represents a paginated response
type PaginationResponse struct {
	Page       int   `json:"page"`
	PageSize   int   `json:"page_size"`
	TotalPages int   `json:"total_pages"`
	TotalCount int64 `json:"total_count"`
	HasNext    bool  `json:"has_next"`
	HasPrev    bool  `json:"has_prev"`
}

// FilterParams represents common filter parameters
type FilterParams struct {
	Search    string    `json:"search,omitempty"`
	Status    []string  `json:"status,omitempty"`
	CreatedAt *TimeRange `json:"created_at,omitempty"`
	UpdatedAt *TimeRange `json:"updated_at,omitempty"`
	Tags      []string  `json:"tags,omitempty"`
}

// TimeRange represents a time range filter
type TimeRange struct {
	From *time.Time `json:"from,omitempty"`
	To   *time.Time `json:"to,omitempty"`
}

// SortParams represents sorting parameters
type SortParams struct {
	Field string `json:"field"`
	Order string `json:"order"` // asc or desc
}

// ListOptions represents common list operation options
type ListOptions struct {
	Pagination *PaginationParams `json:"pagination,omitempty"`
	Filter     *FilterParams     `json:"filter,omitempty"`
	Sort       []SortParams      `json:"sort,omitempty"`
}

// Response wrapper for consistent API responses
type Response[T any] struct {
	Data       T                   `json:"data"`
	Pagination *PaginationResponse `json:"pagination,omitempty"`
	Meta       map[string]interface{} `json:"meta,omitempty"`
}

// ListResponse represents a list response
type ListResponse[T any] struct {
	Items      []T                 `json:"items"`
	Pagination *PaginationResponse `json:"pagination,omitempty"`
	Meta       map[string]interface{} `json:"meta,omitempty"`
}