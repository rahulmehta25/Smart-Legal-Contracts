package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/enterprise/whitelabel/pkg/customization"
	"github.com/enterprise/whitelabel/pkg/isolation"
	"github.com/enterprise/whitelabel/pkg/tenant"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// AdminHandler handles admin portal operations
type AdminHandler struct {
	tenantService        *tenant.Service
	customizationService *customization.Service
	isolationService     *isolation.Service
	billingService       BillingService
	analyticsService     AnalyticsService
	logger               *zap.Logger
}

// NewAdminHandler creates a new admin handler
func NewAdminHandler(
	tenantService *tenant.Service,
	customizationService *customization.Service,
	isolationService *isolation.Service,
	billingService BillingService,
	analyticsService AnalyticsService,
	logger *zap.Logger,
) *AdminHandler {
	return &AdminHandler{
		tenantService:        tenantService,
		customizationService: customizationService,
		isolationService:     isolationService,
		billingService:       billingService,
		analyticsService:     analyticsService,
		logger:               logger,
	}
}

// Tenant Management Endpoints

// CreateTenant creates a new tenant
func (h *AdminHandler) CreateTenant(c *gin.Context) {
	var req tenant.TenantCreateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body", "details": err.Error()})
		return
	}

	tenantObj, err := h.tenantService.CreateTenant(c.Request.Context(), &req)
	if err != nil {
		h.logger.Error("Failed to create tenant", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create tenant", "details": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, tenantObj)
}

// GetTenant retrieves a tenant by ID
func (h *AdminHandler) GetTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	tenantObj, err := h.tenantService.GetTenant(c.Request.Context(), tenantID)
	if err != nil {
		h.logger.Error("Failed to get tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusNotFound, gin.H{"error": "Tenant not found", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, tenantObj)
}

// UpdateTenant updates a tenant
func (h *AdminHandler) UpdateTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	var req tenant.TenantUpdateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body", "details": err.Error()})
		return
	}

	tenantObj, err := h.tenantService.UpdateTenant(c.Request.Context(), tenantID, &req)
	if err != nil {
		h.logger.Error("Failed to update tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update tenant", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, tenantObj)
}

// SuspendTenant suspends a tenant
func (h *AdminHandler) SuspendTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	var req struct {
		Reason string `json:"reason" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Reason is required"})
		return
	}

	if err := h.tenantService.SuspendTenant(c.Request.Context(), tenantID, req.Reason); err != nil {
		h.logger.Error("Failed to suspend tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to suspend tenant", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Tenant suspended successfully"})
}

// ReactivateTenant reactivates a suspended tenant
func (h *AdminHandler) ReactivateTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	if err := h.tenantService.ReactivateTenant(c.Request.Context(), tenantID); err != nil {
		h.logger.Error("Failed to reactivate tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to reactivate tenant", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Tenant reactivated successfully"})
}

// ListTenants lists all tenants with filtering
func (h *AdminHandler) ListTenants(c *gin.Context) {
	filter := &tenant.TenantFilter{}
	
	// Parse query parameters
	if status := c.QueryArray("status"); len(status) > 0 {
		for _, s := range status {
			filter.Status = append(filter.Status, tenant.TenantStatus(s))
		}
	}
	
	if plans := c.QueryArray("plan"); len(plans) > 0 {
		filter.Plan = plans
	}
	
	if search := c.Query("search"); search != "" {
		filter.Search = search
	}
	
	if limitStr := c.Query("limit"); limitStr != "" {
		if limit, err := strconv.Atoi(limitStr); err == nil {
			filter.Limit = limit
		}
	}
	
	if offsetStr := c.Query("offset"); offsetStr != "" {
		if offset, err := strconv.Atoi(offsetStr); err == nil {
			filter.Offset = offset
		}
	}

	tenants, err := h.tenantService.ListTenants(c.Request.Context(), filter)
	if err != nil {
		h.logger.Error("Failed to list tenants", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list tenants", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"tenants": tenants,
		"total":   len(tenants),
	})
}

// Monitoring and Analytics Endpoints

// GetTenantUsage retrieves usage metrics for a tenant
func (h *AdminHandler) GetTenantUsage(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	// Parse time range
	from, to := h.parseTimeRange(c)

	usage, err := h.tenantService.GetTenantUsage(c.Request.Context(), tenantID, from, to)
	if err != nil {
		h.logger.Error("Failed to get tenant usage", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get tenant usage", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"usage": usage})
}

// GetTenantMetrics retrieves isolation metrics for a tenant
func (h *AdminHandler) GetTenantMetrics(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	from, to := h.parseTimeRange(c)

	metrics, err := h.isolationService.GetTenantMetrics(c.Request.Context(), tenantID, from, to)
	if err != nil {
		h.logger.Error("Failed to get tenant metrics", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get tenant metrics", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"metrics": metrics})
}

// GetTenantHealth performs health check on tenant
func (h *AdminHandler) GetTenantHealth(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	health, err := h.isolationService.HealthCheck(c.Request.Context(), tenantID)
	if err != nil {
		h.logger.Error("Failed to get tenant health", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get tenant health", "details": err.Error()})
		return
	}

	statusCode := http.StatusOK
	if health.Status != "healthy" {
		statusCode = http.StatusServiceUnavailable
	}

	c.JSON(statusCode, health)
}

// GetTenantDataSummary retrieves data summary for a tenant
func (h *AdminHandler) GetTenantDataSummary(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	summary, err := h.isolationService.GetTenantDataSummary(c.Request.Context(), tenantID)
	if err != nil {
		h.logger.Error("Failed to get tenant data summary", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get tenant data summary", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, summary)
}

// Customization Management Endpoints

// GetTenantCustomization retrieves customization settings for a tenant
func (h *AdminHandler) GetTenantCustomization(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	customization, err := h.customizationService.GetTenantCustomization(c.Request.Context(), tenantID)
	if err != nil {
		h.logger.Error("Failed to get tenant customization", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get tenant customization", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, customization)
}

// ApplyTenantTheme applies a theme to a tenant
func (h *AdminHandler) ApplyTenantTheme(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	var theme customization.ThemeConfig
	if err := c.ShouldBindJSON(&theme); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid theme configuration", "details": err.Error()})
		return
	}

	if err := h.customizationService.ApplyTheme(c.Request.Context(), tenantID, theme); err != nil {
		h.logger.Error("Failed to apply tenant theme", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to apply theme", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Theme applied successfully"})
}

// SetTenantFeature sets a feature toggle for a tenant
func (h *AdminHandler) SetTenantFeature(c *gin.Context) {
	tenantID := c.Param("id")
	featureID := c.Param("feature")
	if tenantID == "" || featureID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID and feature ID are required"})
		return
	}

	var req struct {
		Enabled bool        `json:"enabled"`
		Value   interface{} `json:"value"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body", "details": err.Error()})
		return
	}

	if err := h.customizationService.SetFeatureToggle(c.Request.Context(), tenantID, featureID, req.Enabled, req.Value); err != nil {
		h.logger.Error("Failed to set tenant feature", 
			zap.String("tenant_id", tenantID), 
			zap.String("feature_id", featureID), 
			zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to set feature", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Feature updated successfully"})
}

// Billing Management Endpoints

// GetTenantBilling retrieves billing information for a tenant
func (h *AdminHandler) GetTenantBilling(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	billing, err := h.billingService.GetTenantBilling(c.Request.Context(), tenantID)
	if err != nil {
		h.logger.Error("Failed to get tenant billing", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get billing information", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, billing)
}

// UpdateTenantBilling updates billing information for a tenant
func (h *AdminHandler) UpdateTenantBilling(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	var billing BillingInfo
	if err := c.ShouldBindJSON(&billing); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid billing information", "details": err.Error()})
		return
	}

	if err := h.billingService.UpdateTenantBilling(c.Request.Context(), tenantID, &billing); err != nil {
		h.logger.Error("Failed to update tenant billing", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update billing information", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Billing information updated successfully"})
}

// System Management Endpoints

// GetSystemStats retrieves overall system statistics
func (h *AdminHandler) GetSystemStats(c *gin.Context) {
	stats, err := h.analyticsService.GetSystemStats(c.Request.Context())
	if err != nil {
		h.logger.Error("Failed to get system stats", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get system statistics", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, stats)
}

// GetSystemHealth retrieves overall system health
func (h *AdminHandler) GetSystemHealth(c *gin.Context) {
	health := h.analyticsService.GetSystemHealth(c.Request.Context())
	
	statusCode := http.StatusOK
	if health["status"] != "healthy" {
		statusCode = http.StatusServiceUnavailable
	}

	c.JSON(statusCode, health)
}

// Backup and Maintenance Endpoints

// BackupTenant creates a backup for a tenant
func (h *AdminHandler) BackupTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	if err := h.isolationService.BackupTenantDatabase(c.Request.Context(), tenantID); err != nil {
		h.logger.Error("Failed to backup tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create backup", "details": err.Error()})
		return
	}

	c.JSON(http.StatusAccepted, gin.H{"message": "Backup initiated successfully"})
}

// RestoreTenant restores a tenant from backup
func (h *AdminHandler) RestoreTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	var req struct {
		BackupID string `json:"backup_id" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Backup ID is required"})
		return
	}

	if err := h.isolationService.RestoreTenantDatabase(c.Request.Context(), tenantID, req.BackupID); err != nil {
		h.logger.Error("Failed to restore tenant", zap.String("tenant_id", tenantID), zap.String("backup_id", req.BackupID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to restore from backup", "details": err.Error()})
		return
	}

	c.JSON(http.StatusAccepted, gin.H{"message": "Restore initiated successfully"})
}

// MigrateTenant runs migrations for a tenant
func (h *AdminHandler) MigrateTenant(c *gin.Context) {
	tenantID := c.Param("id")
	if tenantID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Tenant ID is required"})
		return
	}

	if err := h.isolationService.MigrateTenant(c.Request.Context(), tenantID); err != nil {
		h.logger.Error("Failed to migrate tenant", zap.String("tenant_id", tenantID), zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to run migrations", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Migrations completed successfully"})
}

// Helper methods

// parseTimeRange parses time range from query parameters
func (h *AdminHandler) parseTimeRange(c *gin.Context) (from, to time.Time) {
	now := time.Now()
	from = now.AddDate(0, 0, -7) // Default to last 7 days
	to = now

	if fromStr := c.Query("from"); fromStr != "" {
		if parsed, err := time.Parse(time.RFC3339, fromStr); err == nil {
			from = parsed
		}
	}

	if toStr := c.Query("to"); toStr != "" {
		if parsed, err := time.Parse(time.RFC3339, toStr); err == nil {
			to = parsed
		}
	}

	return from, to
}

// Middleware for admin authentication and authorization
func (h *AdminHandler) AdminAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Extract and validate admin token
		token := c.GetHeader("Authorization")
		if token == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
			c.Abort()
			return
		}

		// Validate token (implementation depends on your auth system)
		if !h.validateAdminToken(token) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid or expired token"})
			c.Abort()
			return
		}

		c.Next()
	}
}

// validateAdminToken validates admin authentication token
func (h *AdminHandler) validateAdminToken(token string) bool {
	// Implement token validation logic
	// This could involve JWT validation, database lookup, etc.
	return true // Simplified for example
}

// CORS middleware for admin API
func (h *AdminHandler) CORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}