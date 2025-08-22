package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	whitelabelsdk "github.com/enterprise/whitelabel-sdk-go"
)

func main() {
	// Example 1: Basic client setup with API key
	fmt.Println("=== Basic Client Setup ===")
	basicClientExample()
	
	// Example 2: Tenant management
	fmt.Println("\n=== Tenant Management ===")
	tenantManagementExample()
	
	// Example 3: Customization operations
	fmt.Println("\n=== Customization Operations ===")
	customizationExample()
	
	// Example 4: Feature management
	fmt.Println("\n=== Feature Management ===")
	featureManagementExample()
	
	// Example 5: Error handling
	fmt.Println("\n=== Error Handling ===")
	errorHandlingExample()
}

func basicClientExample() {
	// Create client with API key
	client, err := whitelabelsdk.NewClientWithAPIKey(
		"https://api.whitelabel-platform.com",
		"your-api-key-here",
		"tenant-123",
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	
	// Check API health
	ctx := context.Background()
	health, err := client.Health(ctx)
	if err != nil {
		log.Printf("Health check failed: %v", err)
		return
	}
	
	fmt.Printf("API Status: %s\n", health.Status)
	fmt.Printf("API Version: %s\n", health.Version)
	
	// Get version information
	version, err := client.Version(ctx)
	if err != nil {
		log.Printf("Version check failed: %v", err)
		return
	}
	
	fmt.Printf("API Version: %s\n", version.Version)
	fmt.Printf("Build Time: %s\n", version.BuildTime.Format(time.RFC3339))
}

func tenantManagementExample() {
	client, err := whitelabelsdk.NewClientWithAPIKey(
		"https://api.whitelabel-platform.com",
		"your-api-key-here",
		"",
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	
	ctx := context.Background()
	
	// Create a new tenant
	createReq := &whitelabelsdk.TenantCreateRequest{
		Name:              "Acme Corporation",
		Subdomain:         "acme",
		CustomDomain:      "app.acme.com",
		Plan:              "professional",
		IsolationStrategy: whitelabelsdk.DatabasePerTenant,
		AdminEmail:        "admin@acme.com",
		AdminName:         "John Doe",
		AdminPhone:        "+1-555-123-4567",
		Config: whitelabelsdk.TenantConfig{
			Features: map[string]bool{
				"analytics":     true,
				"custom_domain": true,
				"sso":          true,
			},
			TimeZone: "America/New_York",
			Language: "en-US",
		},
	}
	
	tenant, err := client.Tenants.Create(ctx, createReq)
	if err != nil {
		log.Printf("Failed to create tenant: %v", err)
		return
	}
	
	fmt.Printf("Created tenant: %s (ID: %s)\n", tenant.Name, tenant.ID)
	fmt.Printf("Subdomain: %s\n", tenant.Subdomain)
	fmt.Printf("Status: %s\n", tenant.Status)
	
	// Get tenant by ID
	retrievedTenant, err := client.Tenants.Get(ctx, tenant.ID)
	if err != nil {
		log.Printf("Failed to get tenant: %v", err)
		return
	}
	
	fmt.Printf("Retrieved tenant: %s\n", retrievedTenant.Name)
	
	// Update tenant
	updateReq := &whitelabelsdk.TenantUpdateRequest{
		Name:         "Acme Corporation Ltd",
		BillingEmail: "billing@acme.com",
	}
	
	updatedTenant, err := client.Tenants.Update(ctx, tenant.ID, updateReq)
	if err != nil {
		log.Printf("Failed to update tenant: %v", err)
		return
	}
	
	fmt.Printf("Updated tenant name: %s\n", updatedTenant.Name)
	
	// List tenants with filters
	listOptions := &whitelabelsdk.TenantListOptions{
		Status: []whitelabelsdk.TenantStatus{whitelabelsdk.TenantStatusActive},
		Plan:   []string{"professional", "enterprise"},
		ListOptions: whitelabelsdk.ListOptions{
			Pagination: &whitelabelsdk.PaginationParams{
				Page:     1,
				PageSize: 10,
			},
		},
	}
	
	tenantList, err := client.Tenants.List(ctx, listOptions)
	if err != nil {
		log.Printf("Failed to list tenants: %v", err)
		return
	}
	
	fmt.Printf("Found %d tenants\n", len(tenantList.Items))
	for _, t := range tenantList.Items {
		fmt.Printf("- %s (%s)\n", t.Name, t.Subdomain)
	}
	
	// Get tenant usage metrics
	from := time.Now().AddDate(0, 0, -7) // Last 7 days
	to := time.Now()
	
	usage, err := client.Tenants.GetUsage(ctx, tenant.ID, &from, &to)
	if err != nil {
		log.Printf("Failed to get tenant usage: %v", err)
		return
	}
	
	if len(usage) > 0 {
		latest := usage[0]
		fmt.Printf("Latest usage - Users: %d, API Requests: %d, Storage: %dMB\n",
			latest.ActiveUsers, latest.APIRequests, latest.StorageUsedMB)
	}
}

func customizationExample() {
	client, err := whitelabelsdk.NewClientWithAPIKey(
		"https://api.whitelabel-platform.com",
		"your-api-key-here",
		"tenant-123",
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	
	ctx := context.Background()
	tenantID := "tenant-123"
	
	// Apply a custom theme
	theme := &whitelabelsdk.ThemeConfig{
		PrimaryColor:    "#007bff",
		SecondaryColor:  "#6c757d",
		AccentColor:     "#28a745",
		BackgroundColor: "#ffffff",
		TextColor:       "#333333",
		LinkColor:       "#007bff",
		FontFamily:      "Inter, sans-serif",
		FontSize:        "14px",
		BorderRadius:    "8px",
		BoxShadow:       "0 2px 4px rgba(0,0,0,0.1)",
		LogoURL:         "https://cdn.example.com/logo.png",
		FaviconURL:      "https://cdn.example.com/favicon.ico",
		DarkMode:        false,
		CustomProperties: map[string]string{
			"sidebar-width": "260px",
			"header-height": "64px",
		},
	}
	
	err = client.Customization.ApplyTheme(ctx, tenantID, theme)
	if err != nil {
		log.Printf("Failed to apply theme: %v", err)
		return
	}
	
	fmt.Printf("Theme applied successfully\n")
	
	// Apply branding configuration
	branding := &whitelabelsdk.BrandingConfig{
		CompanyName:        "Acme Corporation",
		CompanyDescription: "Leading provider of enterprise solutions",
		LogoURL:            "https://cdn.example.com/logo.png",
		LogoWidth:          120,
		LogoHeight:         40,
		HeaderText:         "Acme Platform",
		FooterText:         "© 2024 Acme Corporation. All rights reserved.",
		ContactEmail:       "support@acme.com",
		SupportURL:         "https://support.acme.com",
		TermsURL:           "https://acme.com/terms",
		PrivacyURL:         "https://acme.com/privacy",
		SocialMedia: whitelabelsdk.SocialMediaConfig{
			Twitter:  "https://twitter.com/acmecorp",
			LinkedIn: "https://linkedin.com/company/acme",
		},
	}
	
	err = client.Customization.ApplyBranding(ctx, tenantID, branding)
	if err != nil {
		log.Printf("Failed to apply branding: %v", err)
		return
	}
	
	fmt.Printf("Branding applied successfully\n")
	
	// Create a custom email template
	emailTemplate := &whitelabelsdk.EmailTemplate{
		TenantID:    tenantID,
		Type:        whitelabelsdk.EmailTypeWelcome,
		Subject:     "Welcome to {{.CompanyName}}!",
		HTMLContent: `<h1>Welcome {{.UserName}}!</h1><p>Thank you for joining {{.CompanyName}}.</p>`,
		TextContent: `Welcome {{.UserName}}! Thank you for joining {{.CompanyName}}.`,
		Variables:   []string{"UserName", "CompanyName"},
	}
	
	createdTemplate, err := client.Customization.CreateEmailTemplate(ctx, emailTemplate)
	if err != nil {
		log.Printf("Failed to create email template: %v", err)
		return
	}
	
	fmt.Printf("Email template created: %s\n", createdTemplate.ID)
	
	// Get all customization settings
	customization, err := client.Customization.GetCustomization(ctx, tenantID)
	if err != nil {
		log.Printf("Failed to get customization: %v", err)
		return
	}
	
	fmt.Printf("Current theme primary color: %s\n", customization.Theme.PrimaryColor)
	fmt.Printf("Current branding company name: %s\n", customization.Branding.CompanyName)
	
	// Generate CSS for the current theme
	css, err := client.Customization.GenerateCSS(ctx, tenantID)
	if err != nil {
		log.Printf("Failed to generate CSS: %v", err)
		return
	}
	
	fmt.Printf("Generated CSS length: %d characters\n", len(css))
}

func featureManagementExample() {
	client, err := whitelabelsdk.NewClientWithAPIKey(
		"https://api.whitelabel-platform.com",
		"your-api-key-here",
		"tenant-123",
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	
	ctx := context.Background()
	tenantID := "tenant-123"
	
	// Create a new feature toggle
	featureReq := &whitelabelsdk.FeatureCreateRequest{
		Name:         "Advanced Analytics",
		Description:  "Enable advanced analytics dashboard with custom reports",
		Category:     "analytics",
		Type:         whitelabelsdk.FeatureTypeBoolean,
		DefaultValue: false,
		Config: map[string]interface{}{
			"max_reports":    10,
			"data_retention": "90d",
		},
	}
	
	feature, err := client.Features.CreateFeature(ctx, featureReq)
	if err != nil {
		log.Printf("Failed to create feature: %v", err)
		return
	}
	
	fmt.Printf("Created feature: %s (ID: %s)\n", feature.Name, feature.ID)
	
	// Enable feature for a tenant
	err = client.Features.EnableFeatureForTenant(ctx, tenantID, feature.ID)
	if err != nil {
		log.Printf("Failed to enable feature: %v", err)
		return
	}
	
	fmt.Printf("Feature enabled for tenant\n")
	
	// Check if feature is enabled
	enabled, err := client.Features.IsFeatureEnabled(ctx, tenantID, feature.ID)
	if err != nil {
		log.Printf("Failed to check feature status: %v", err)
		return
	}
	
	fmt.Printf("Feature enabled: %t\n", enabled)
	
	// Set a string feature value
	err = client.Features.SetFeatureString(ctx, tenantID, "theme_variant", "dark")
	if err != nil {
		log.Printf("Failed to set feature string: %v", err)
		return
	}
	
	// Set a number feature value
	err = client.Features.SetFeatureNumber(ctx, tenantID, "api_rate_limit", 1000.0)
	if err != nil {
		log.Printf("Failed to set feature number: %v", err)
		return
	}
	
	// Get all tenant features
	tenantFeatures, err := client.Features.GetTenantFeatures(ctx, tenantID)
	if err != nil {
		log.Printf("Failed to get tenant features: %v", err)
		return
	}
	
	fmt.Printf("Tenant has %d features configured\n", len(tenantFeatures))
	for featureID, tenantFeature := range tenantFeatures {
		fmt.Printf("- %s: enabled=%t, value=%v\n", featureID, tenantFeature.Enabled, tenantFeature.Value)
	}
	
	// Bulk update features
	features := map[string]interface{}{
		"notifications":     true,
		"custom_branding":   true,
		"max_users":         100,
		"storage_limit_gb":  50,
	}
	
	err = client.Features.BulkUpdateTenantFeatures(ctx, tenantID, features)
	if err != nil {
		log.Printf("Failed to bulk update features: %v", err)
		return
	}
	
	fmt.Printf("Bulk feature update completed\n")
}

func errorHandlingExample() {
	client, err := whitelabelsdk.NewClientWithAPIKey(
		"https://api.whitelabel-platform.com",
		"invalid-api-key",
		"tenant-123",
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	
	ctx := context.Background()
	
	// This will likely fail with authentication error
	_, err = client.Tenants.Get(ctx, "non-existent-tenant")
	if err != nil {
		// Check if it's an API error
		if whitelabelsdk.IsAPIError(err) {
			apiError, _ := whitelabelsdk.GetAPIError(err)
			fmt.Printf("API Error: Code=%s, Status=%d, Message=%s\n", 
				apiError.Code, apiError.StatusCode, apiError.Message)
			
			// Handle specific error codes
			switch apiError.StatusCode {
			case 401:
				fmt.Println("Authentication failed - check your API key")
			case 403:
				fmt.Println("Access denied - insufficient permissions")
			case 404:
				fmt.Println("Resource not found")
			case 429:
				fmt.Println("Rate limit exceeded - please retry later")
			default:
				fmt.Printf("Unexpected error: %v\n", err)
			}
		} else {
			// Handle other types of errors (network, timeout, etc.)
			fmt.Printf("Request failed: %v\n", err)
		}
		return
	}
}

// Advanced configuration example
func advancedConfigExample() {
	// Custom HTTP client with timeout
	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}
	
	// Advanced client configuration
	config := &whitelabelsdk.Config{
		BaseURL:    "https://api.whitelabel-platform.com",
		APIKey:     "your-api-key-here",
		TenantID:   "tenant-123",
		HTTPClient: httpClient,
		Timeout:    30 * time.Second,
		RetryCount: 3,
		RetryWait:  2 * time.Second,
		Debug:      false,
		UserAgent:  "MyApp/1.0.0",
		Headers: map[string]string{
			"X-Custom-Header": "value",
		},
		RateLimit: &whitelabelsdk.RateLimitConfig{
			RequestsPerSecond: 10,
			BurstLimit:        20,
		},
	}
	
	client, err := whitelabelsdk.NewClient(config)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	
	// Use the configured client
	ctx := context.Background()
	health, err := client.Health(ctx)
	if err != nil {
		log.Printf("Health check failed: %v", err)
		return
	}
	
	fmt.Printf("API Status: %s\n", health.Status)
}