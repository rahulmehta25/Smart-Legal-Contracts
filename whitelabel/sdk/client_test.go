package whitelabelsdk

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name        string
		config      *Config
		expectError bool
		errorMsg    string
	}{
		{
			name:        "nil config",
			config:      nil,
			expectError: true,
			errorMsg:    "config cannot be nil",
		},
		{
			name: "missing base URL",
			config: &Config{
				APIKey: "test-key",
			},
			expectError: true,
			errorMsg:    "baseURL is required",
		},
		{
			name: "invalid base URL",
			config: &Config{
				BaseURL: "not-a-url",
				APIKey:  "test-key",
			},
			expectError: true,
			errorMsg:    "invalid baseURL",
		},
		{
			name: "valid config",
			config: &Config{
				BaseURL: "https://api.example.com",
				APIKey:  "test-key",
			},
			expectError: false,
		},
		{
			name: "valid config with tenant ID",
			config: &Config{
				BaseURL:  "https://api.example.com",
				APIKey:   "test-key",
				TenantID: "tenant-123",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)
			
			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
				assert.Equal(t, tt.config.BaseURL, client.baseURL)
				assert.Equal(t, tt.config.APIKey, client.apiKey)
				assert.Equal(t, tt.config.TenantID, client.tenantID)
			}
		})
	}
}

func TestNewClientWithAPIKey(t *testing.T) {
	baseURL := "https://api.example.com"
	apiKey := "test-key"
	tenantID := "tenant-123"
	
	client, err := NewClientWithAPIKey(baseURL, apiKey, tenantID)
	
	require.NoError(t, err)
	assert.NotNil(t, client)
	assert.Equal(t, baseURL, client.baseURL)
	assert.Equal(t, apiKey, client.apiKey)
	assert.Equal(t, tenantID, client.tenantID)
}

func TestClientSetters(t *testing.T) {
	client, err := NewClientWithAPIKey("https://api.example.com", "old-key", "old-tenant")
	require.NoError(t, err)
	
	// Test SetAPIKey
	newAPIKey := "new-key"
	client.SetAPIKey(newAPIKey)
	assert.Equal(t, newAPIKey, client.apiKey)
	
	// Test SetTenantID
	newTenantID := "new-tenant"
	client.SetTenantID(newTenantID)
	assert.Equal(t, newTenantID, client.tenantID)
	
	// Test SetDebug
	client.SetDebug(true)
	assert.True(t, client.debug)
}

func TestClientHealth(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/health", r.URL.Path)
		assert.Equal(t, "GET", r.Method)
		
		response := HealthResponse{
			Status:    "healthy",
			Timestamp: time.Now(),
			Version:   "1.0.0",
			Services: map[string]interface{}{
				"database": "ok",
				"cache":    "ok",
			},
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	
	client, err := NewClientWithAPIKey(server.URL, "test-key", "")
	require.NoError(t, err)
	
	ctx := context.Background()
	health, err := client.Health(ctx)
	
	require.NoError(t, err)
	assert.NotNil(t, health)
	assert.Equal(t, "healthy", health.Status)
	assert.Equal(t, "1.0.0", health.Version)
	assert.NotNil(t, health.Services)
}

func TestClientVersion(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/version", r.URL.Path)
		assert.Equal(t, "GET", r.Method)
		
		response := VersionResponse{
			Version:    "1.0.0",
			BuildTime:  time.Now(),
			GitCommit:  "abc123",
			APIVersion: "v1",
			GoVersion:  "1.21.0",
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	
	client, err := NewClientWithAPIKey(server.URL, "test-key", "")
	require.NoError(t, err)
	
	ctx := context.Background()
	version, err := client.Version(ctx)
	
	require.NoError(t, err)
	assert.NotNil(t, version)
	assert.Equal(t, "1.0.0", version.Version)
	assert.Equal(t, "abc123", version.GitCommit)
}

func TestClientHealthError(t *testing.T) {
	// Mock server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("Service unavailable"))
	}))
	defer server.Close()
	
	client, err := NewClientWithAPIKey(server.URL, "test-key", "")
	require.NoError(t, err)
	
	ctx := context.Background()
	health, err := client.Health(ctx)
	
	assert.Error(t, err)
	assert.Nil(t, health)
	assert.Contains(t, err.Error(), "health check failed")
}

func TestClientAPIError(t *testing.T) {
	// Mock server that returns API error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiError := APIError{
			Code:    "INVALID_API_KEY",
			Message: "The provided API key is invalid",
			Details: "Please check your API key",
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(apiError)
	}))
	defer server.Close()
	
	client, err := NewClientWithAPIKey(server.URL, "invalid-key", "")
	require.NoError(t, err)
	
	ctx := context.Background()
	err = client.makeRequest(ctx, "GET", "/test", nil, nil)
	
	require.Error(t, err)
	assert.True(t, IsAPIError(err))
	
	apiError, ok := GetAPIError(err)
	require.True(t, ok)
	assert.Equal(t, "INVALID_API_KEY", apiError.Code)
	assert.Equal(t, "The provided API key is invalid", apiError.Message)
	assert.Equal(t, 401, apiError.StatusCode)
}

func TestMakeRequest(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		endpoint       string
		body           interface{}
		serverResponse interface{}
		serverStatus   int
		expectError    bool
	}{
		{
			name:           "successful GET request",
			method:         "GET",
			endpoint:       "/test",
			body:           nil,
			serverResponse: map[string]string{"message": "success"},
			serverStatus:   http.StatusOK,
			expectError:    false,
		},
		{
			name:           "successful POST request",
			method:         "POST",
			endpoint:       "/test",
			body:           map[string]string{"name": "test"},
			serverResponse: map[string]string{"id": "123"},
			serverStatus:   http.StatusCreated,
			expectError:    false,
		},
		{
			name:         "unsupported method",
			method:       "PATCH2", // Invalid method
			endpoint:     "/test",
			expectError:  true,
		},
		{
			name:         "server error",
			method:       "GET",
			endpoint:     "/test",
			serverStatus: http.StatusInternalServerError,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.method == "PATCH2" {
				// Test unsupported method without server
				client, _ := NewClientWithAPIKey("https://api.example.com", "test-key", "")
				err := client.makeRequest(context.Background(), tt.method, tt.endpoint, tt.body, nil)
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unsupported HTTP method")
				return
			}
			
			// Mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tt.endpoint, r.URL.Path)
				assert.Equal(t, tt.method, r.Method)
				
				if tt.body != nil {
					var requestBody map[string]interface{}
					json.NewDecoder(r.Body).Decode(&requestBody)
					assert.NotNil(t, requestBody)
				}
				
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.serverStatus)
				
				if tt.serverResponse != nil {
					json.NewEncoder(w).Encode(tt.serverResponse)
				}
			}))
			defer server.Close()
			
			client, err := NewClientWithAPIKey(server.URL, "test-key", "")
			require.NoError(t, err)
			
			var result map[string]interface{}
			err = client.makeRequest(context.Background(), tt.method, tt.endpoint, tt.body, &result)
			
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.serverResponse != nil {
					assert.NotNil(t, result)
				}
			}
		})
	}
}

func TestAPIErrorInterface(t *testing.T) {
	apiError := &APIError{
		Code:       "TEST_ERROR",
		Message:    "This is a test error",
		Details:    "Additional details",
		StatusCode: 400,
		RequestID:  "req-123",
	}
	
	// Test Error() method
	errorMsg := apiError.Error()
	assert.Contains(t, errorMsg, "TEST_ERROR")
	assert.Contains(t, errorMsg, "This is a test error")
	assert.Contains(t, errorMsg, "Additional details")
	assert.Contains(t, errorMsg, "400")
	
	// Test without details
	apiErrorNoDetails := &APIError{
		Code:       "TEST_ERROR",
		Message:    "This is a test error",
		StatusCode: 400,
	}
	
	errorMsgNoDetails := apiErrorNoDetails.Error()
	assert.Contains(t, errorMsgNoDetails, "TEST_ERROR")
	assert.Contains(t, errorMsgNoDetails, "This is a test error")
	assert.NotContains(t, errorMsgNoDetails, " - ")
}

func TestClientTimeout(t *testing.T) {
	// Mock server with delay
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second) // Delay longer than client timeout
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("{}"))
	}))
	defer server.Close()
	
	config := &Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Timeout: 500 * time.Millisecond, // Short timeout
	}
	
	client, err := NewClient(config)
	require.NoError(t, err)
	
	ctx := context.Background()
	_, err = client.Health(ctx)
	
	assert.Error(t, err)
	// Should be a timeout error
}

func TestConfigDefaults(t *testing.T) {
	config := &Config{
		BaseURL: "https://api.example.com",
		APIKey:  "test-key",
	}
	
	client, err := NewClient(config)
	require.NoError(t, err)
	
	// Check defaults
	assert.Equal(t, 30*time.Second, client.timeout)
	assert.Equal(t, 3, client.retryCount)
	assert.Equal(t, 1*time.Second, client.retryWait)
	assert.Equal(t, "whitelabel-sdk-go/1.0.0", client.userAgent)
}

func TestClientHeaders(t *testing.T) {
	// Mock server to check headers
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		assert.Equal(t, "tenant-123", r.Header.Get("X-Tenant-ID"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "application/json", r.Header.Get("Accept"))
		assert.Equal(t, "custom-value", r.Header.Get("X-Custom-Header"))
		assert.Contains(t, r.Header.Get("User-Agent"), "whitelabel-sdk-go")
		
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
	}))
	defer server.Close()
	
	config := &Config{
		BaseURL:  server.URL,
		APIKey:   "test-key",
		TenantID: "tenant-123",
		Headers: map[string]string{
			"X-Custom-Header": "custom-value",
		},
	}
	
	client, err := NewClient(config)
	require.NoError(t, err)
	
	ctx := context.Background()
	_, err = client.Health(ctx)
	
	assert.NoError(t, err)
}

// Benchmark tests
func BenchmarkNewClient(b *testing.B) {
	config := &Config{
		BaseURL: "https://api.example.com",
		APIKey:  "test-key",
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client, err := NewClient(config)
		if err != nil {
			b.Fatal(err)
		}
		_ = client
	}
}

func BenchmarkMakeRequest(b *testing.B) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer server.Close()
	
	client, err := NewClientWithAPIKey(server.URL, "test-key", "")
	if err != nil {
		b.Fatal(err)
	}
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result map[string]interface{}
		err := client.makeRequest(ctx, "GET", "/test", nil, &result)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Example tests
func ExampleNewClientWithAPIKey() {
	client, err := NewClientWithAPIKey(
		"https://api.whitelabel-platform.com",
		"your-api-key-here",
		"tenant-123",
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	// Use the client
	ctx := context.Background()
	health, err := client.Health(ctx)
	if err != nil {
		fmt.Printf("Health check failed: %v\n", err)
		return
	}
	
	fmt.Printf("API Status: %s\n", health.Status)
}

func ExampleClient_Health() {
	client, _ := NewClientWithAPIKey("https://api.example.com", "key", "")
	
	ctx := context.Background()
	health, err := client.Health(ctx)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	fmt.Printf("Status: %s, Version: %s\n", health.Status, health.Version)
}