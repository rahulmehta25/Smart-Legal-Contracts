package api

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// SetupRoutes sets up all admin API routes
func SetupRoutes(handler *AdminHandler) *gin.Engine {
	// Create gin router with recovery and logging middleware
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(gin.Logger())
	
	// Add CORS middleware
	router.Use(handler.CORS())
	
	// Health check endpoint (no auth required)
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "healthy", "service": "whitelabel-admin"})
	})
	
	// API version group
	v1 := router.Group("/api/v1")
	
	// Admin authentication middleware
	v1.Use(handler.AdminAuth())
	
	// System endpoints
	system := v1.Group("/system")
	{
		system.GET("/stats", handler.GetSystemStats)
		system.GET("/health", handler.GetSystemHealth)
	}
	
	// Tenant management endpoints
	tenants := v1.Group("/tenants")
	{
		tenants.POST("", handler.CreateTenant)
		tenants.GET("", handler.ListTenants)
		tenants.GET("/:id", handler.GetTenant)
		tenants.PUT("/:id", handler.UpdateTenant)
		tenants.DELETE("/:id", handler.DeleteTenant)
		
		// Tenant actions
		tenants.POST("/:id/suspend", handler.SuspendTenant)
		tenants.POST("/:id/reactivate", handler.ReactivateTenant)
		
		// Tenant monitoring
		tenants.GET("/:id/usage", handler.GetTenantUsage)
		tenants.GET("/:id/metrics", handler.GetTenantMetrics)
		tenants.GET("/:id/health", handler.GetTenantHealth)
		tenants.GET("/:id/data-summary", handler.GetTenantDataSummary)
		
		// Tenant customization
		tenants.GET("/:id/customization", handler.GetTenantCustomization)
		tenants.POST("/:id/theme", handler.ApplyTenantTheme)
		tenants.PUT("/:id/features/:feature", handler.SetTenantFeature)
		
		// Tenant billing
		tenants.GET("/:id/billing", handler.GetTenantBilling)
		tenants.PUT("/:id/billing", handler.UpdateTenantBilling)
		
		// Tenant maintenance
		tenants.POST("/:id/backup", handler.BackupTenant)
		tenants.POST("/:id/restore", handler.RestoreTenant)
		tenants.POST("/:id/migrate", handler.MigrateTenant)
	}
	
	// Feature management endpoints
	features := v1.Group("/features")
	{
		features.GET("", handler.ListFeatures)
		features.POST("", handler.CreateFeature)
		features.GET("/:id", handler.GetFeature)
		features.PUT("/:id", handler.UpdateFeature)
		features.DELETE("/:id", handler.DeleteFeature)
		
		// Bulk feature operations
		features.POST("/bulk-enable", handler.BulkEnableFeature)
		features.POST("/bulk-disable", handler.BulkDisableFeature)
	}
	
	// Billing management endpoints
	billing := v1.Group("/billing")
	{
		billing.GET("/plans", handler.GetBillingPlans)
		billing.POST("/plans", handler.CreateBillingPlan)
		billing.PUT("/plans/:id", handler.UpdateBillingPlan)
		billing.DELETE("/plans/:id", handler.DeleteBillingPlan)
		
		billing.GET("/subscriptions", handler.GetSubscriptions)
		billing.GET("/invoices", handler.GetInvoices)
		billing.GET("/payments", handler.GetPayments)
		
		billing.GET("/revenue", handler.GetRevenueMetrics)
		billing.GET("/churn", handler.GetChurnAnalysis)
	}
	
	// Analytics endpoints
	analytics := v1.Group("/analytics")
	{
		analytics.GET("/overview", handler.GetAnalyticsOverview)
		analytics.GET("/usage", handler.GetUsageStats)
		analytics.GET("/growth", handler.GetGrowthMetrics)
		analytics.GET("/resource-utilization", handler.GetResourceUtilization)
		analytics.GET("/tenant-performance", handler.GetTenantPerformanceMetrics)
		analytics.GET("/api-usage", handler.GetAPIUsageStats)
		analytics.GET("/error-analysis", handler.GetErrorAnalysis)
		analytics.GET("/geographic-distribution", handler.GetGeographicDistribution)
	}
	
	// Support ticket management endpoints
	support := v1.Group("/support")
	{
		support.GET("/tickets", handler.GetSupportTickets)
		support.POST("/tickets", handler.CreateSupportTicket)
		support.GET("/tickets/:id", handler.GetSupportTicket)
		support.PUT("/tickets/:id", handler.UpdateSupportTicket)
		support.POST("/tickets/:id/messages", handler.AddTicketMessage)
		support.POST("/tickets/:id/assign", handler.AssignTicket)
		support.POST("/tickets/:id/close", handler.CloseTicket)
		
		support.GET("/stats", handler.GetSupportStats)
		support.GET("/agents", handler.GetSupportAgents)
	}
	
	// User management endpoints
	users := v1.Group("/users")
	{
		users.GET("", handler.GetUsers)
		users.POST("", handler.CreateUser)
		users.GET("/:id", handler.GetUser)
		users.PUT("/:id", handler.UpdateUser)
		users.DELETE("/:id", handler.DeleteUser)
		users.POST("/:id/suspend", handler.SuspendUser)
		users.POST("/:id/activate", handler.ActivateUser)
		users.POST("/:id/reset-password", handler.ResetUserPassword)
	}
	
	// Configuration management endpoints
	config := v1.Group("/config")
	{
		config.GET("/routing", handler.GetRoutingConfig)
		config.PUT("/routing", handler.UpdateRoutingConfig)
		
		config.GET("/isolation", handler.GetIsolationConfig)
		config.PUT("/isolation", handler.UpdateIsolationConfig)
		
		config.GET("/customization", handler.GetCustomizationConfig)
		config.PUT("/customization", handler.UpdateCustomizationConfig)
		
		config.GET("/security", handler.GetSecurityConfig)
		config.PUT("/security", handler.UpdateSecurityConfig)
	}
	
	// Monitoring and alerts endpoints
	monitoring := v1.Group("/monitoring")
	{
		monitoring.GET("/alerts", handler.GetAlerts)
		monitoring.POST("/alerts", handler.CreateAlert)
		monitoring.PUT("/alerts/:id", handler.UpdateAlert)
		monitoring.DELETE("/alerts/:id", handler.DeleteAlert)
		monitoring.POST("/alerts/:id/acknowledge", handler.AcknowledgeAlert)
		
		monitoring.GET("/metrics", handler.GetMonitoringMetrics)
		monitoring.GET("/logs", handler.GetSystemLogs)
		monitoring.GET("/incidents", handler.GetIncidents)
		monitoring.POST("/incidents", handler.CreateIncident)
		monitoring.PUT("/incidents/:id", handler.UpdateIncident)
	}
	
	// Backup and disaster recovery endpoints
	backup := v1.Group("/backup")
	{
		backup.GET("/jobs", handler.GetBackupJobs)
		backup.POST("/jobs", handler.CreateBackupJob)
		backup.GET("/jobs/:id", handler.GetBackupJob)
		backup.PUT("/jobs/:id", handler.UpdateBackupJob)
		backup.DELETE("/jobs/:id", handler.DeleteBackupJob)
		backup.POST("/jobs/:id/run", handler.RunBackupJob)
		
		backup.GET("/backups", handler.GetBackups)
		backup.GET("/backups/:id", handler.GetBackup)
		backup.DELETE("/backups/:id", handler.DeleteBackup)
		backup.POST("/backups/:id/verify", handler.VerifyBackup)
		
		backup.GET("/restore-points", handler.GetRestorePoints)
		backup.POST("/restore", handler.RestoreFromBackup)
		backup.GET("/restore-status/:id", handler.GetRestoreStatus)
	}
	
	// Integration management endpoints
	integrations := v1.Group("/integrations")
	{
		integrations.GET("", handler.GetIntegrations)
		integrations.POST("", handler.CreateIntegration)
		integrations.GET("/:id", handler.GetIntegration)
		integrations.PUT("/:id", handler.UpdateIntegration)
		integrations.DELETE("/:id", handler.DeleteIntegration)
		integrations.POST("/:id/test", handler.TestIntegration)
		integrations.POST("/:id/sync", handler.SyncIntegration)
		
		integrations.GET("/webhooks", handler.GetWebhooks)
		integrations.POST("/webhooks", handler.CreateWebhook)
		integrations.PUT("/webhooks/:id", handler.UpdateWebhook)
		integrations.DELETE("/webhooks/:id", handler.DeleteWebhook)
		integrations.POST("/webhooks/:id/test", handler.TestWebhook)
	}
	
	// API management endpoints
	api := v1.Group("/api-management")
	{
		api.GET("/keys", handler.GetAPIKeys)
		api.POST("/keys", handler.CreateAPIKey)
		api.PUT("/keys/:id", handler.UpdateAPIKey)
		api.DELETE("/keys/:id", handler.DeleteAPIKey)
		api.POST("/keys/:id/regenerate", handler.RegenerateAPIKey)
		
		api.GET("/rate-limits", handler.GetRateLimits)
		api.PUT("/rate-limits", handler.UpdateRateLimits)
		
		api.GET("/usage", handler.GetAPIUsageMetrics)
		api.GET("/errors", handler.GetAPIErrors)
		api.GET("/performance", handler.GetAPIPerformanceMetrics)
	}
	
	// Security and audit endpoints
	security := v1.Group("/security")
	{
		security.GET("/audit-logs", handler.GetAuditLogs)
		security.GET("/security-events", handler.GetSecurityEvents)
		security.GET("/compliance-reports", handler.GetComplianceReports)
		security.POST("/compliance-reports", handler.GenerateComplianceReport)
		security.GET("/vulnerabilities", handler.GetVulnerabilities)
		security.POST("/security-scan", handler.RunSecurityScan)
		
		security.GET("/access-policies", handler.GetAccessPolicies)
		security.POST("/access-policies", handler.CreateAccessPolicy)
		security.PUT("/access-policies/:id", handler.UpdateAccessPolicy)
		security.DELETE("/access-policies/:id", handler.DeleteAccessPolicy)
	}
	
	// Migration and deployment endpoints
	deployment := v1.Group("/deployment")
	{
		deployment.GET("/environments", handler.GetEnvironments)
		deployment.POST("/environments", handler.CreateEnvironment)
		deployment.PUT("/environments/:id", handler.UpdateEnvironment)
		deployment.DELETE("/environments/:id", handler.DeleteEnvironment)
		
		deployment.GET("/deployments", handler.GetDeployments)
		deployment.POST("/deployments", handler.CreateDeployment)
		deployment.GET("/deployments/:id", handler.GetDeployment)
		deployment.POST("/deployments/:id/promote", handler.PromoteDeployment)
		deployment.POST("/deployments/:id/rollback", handler.RollbackDeployment)
		
		deployment.GET("/migrations", handler.GetMigrations)
		deployment.POST("/migrations/run", handler.RunMigrations)
		deployment.GET("/migrations/status", handler.GetMigrationStatus)
	}
	
	// License management endpoints
	license := v1.Group("/license")
	{
		license.GET("/info", handler.GetLicenseInfo)
		license.PUT("/update", handler.UpdateLicense)
		license.GET("/usage", handler.GetLicenseUsage)
		license.GET("/compliance", handler.GetLicenseCompliance)
	}
	
	return router
}

// Additional handler methods for the new routes would be implemented
// in the AdminHandler struct. Here are some method signatures:

// Feature Management
func (h *AdminHandler) ListFeatures(c *gin.Context) {
	// Implementation for listing all features
}

func (h *AdminHandler) CreateFeature(c *gin.Context) {
	// Implementation for creating a new feature
}

func (h *AdminHandler) GetFeature(c *gin.Context) {
	// Implementation for getting a specific feature
}

func (h *AdminHandler) UpdateFeature(c *gin.Context) {
	// Implementation for updating a feature
}

func (h *AdminHandler) DeleteFeature(c *gin.Context) {
	// Implementation for deleting a feature
}

func (h *AdminHandler) BulkEnableFeature(c *gin.Context) {
	// Implementation for bulk enabling features
}

func (h *AdminHandler) BulkDisableFeature(c *gin.Context) {
	// Implementation for bulk disabling features
}

// Delete Tenant (missing from handlers.go)
func (h *AdminHandler) DeleteTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(400, gin.H{"error": "Tenant ID is required"})
		return
	}

	if err := h.tenantService.DeleteTenant(c.Request.Context(), tenantID); err != nil {
		h.logger.Error("Failed to delete tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(500, gin.H{"error": "Failed to delete tenant", "details": err.Error()})
		return
	}

	c.JSON(200, gin.H{"message": "Tenant deleted successfully"})
}

// Additional handler methods would continue here...
// For brevity, I'm not implementing all of them, but they would follow
// similar patterns to the existing handlers.