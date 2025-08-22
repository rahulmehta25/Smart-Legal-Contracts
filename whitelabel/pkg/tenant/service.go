package tenant

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// Service handles tenant operations
type Service struct {
	repo        Repository
	provisioner Provisioner
	logger      *zap.Logger
}

// NewService creates a new tenant service
func NewService(repo Repository, provisioner Provisioner, logger *zap.Logger) *Service {
	return &Service{
		repo:        repo,
		provisioner: provisioner,
		logger:      logger,
	}
}

// CreateTenant creates a new tenant with full provisioning
func (s *Service) CreateTenant(ctx context.Context, req *TenantCreateRequest) (*Tenant, error) {
	s.logger.Info("Creating new tenant", zap.String("name", req.Name), zap.String("subdomain", req.Subdomain))
	
	// Validate subdomain availability
	if exists, err := s.repo.SubdomainExists(ctx, req.Subdomain); err != nil {
		return nil, fmt.Errorf("failed to check subdomain availability: %w", err)
	} else if exists {
		return nil, fmt.Errorf("subdomain '%s' is already taken", req.Subdomain)
	}
	
	// Create tenant model
	tenant := &Tenant{
		ID:                uuid.New().String(),
		Name:              req.Name,
		Subdomain:         req.Subdomain,
		CustomDomain:      req.CustomDomain,
		Status:            StatusPending,
		IsolationStrategy: req.IsolationStrategy,
		Plan:              req.Plan,
		AdminEmail:        req.AdminEmail,
		AdminName:         req.AdminName,
		AdminPhone:        req.AdminPhone,
		BillingEmail:      req.BillingEmail,
		Config:            req.Config,
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
	}
	
	// Set database/schema names based on isolation strategy
	switch req.IsolationStrategy {
	case DatabasePerTenant:
		tenant.DatabaseName = fmt.Sprintf("tenant_%s", tenant.ID)
	case SchemaPerTenant:
		tenant.SchemaName = fmt.Sprintf("tenant_%s", tenant.ID)
	}
	
	// Set default resource quota based on plan
	tenant.ResourceQuota = s.getDefaultQuotaForPlan(req.Plan)
	
	// Set trial period for new tenants
	if req.Plan == "trial" {
		trialEnd := time.Now().AddDate(0, 0, 30) // 30 days trial
		tenant.TrialEndsAt = &trialEnd
	}
	
	// Save tenant to database
	if err := s.repo.Create(ctx, tenant); err != nil {
		return nil, fmt.Errorf("failed to create tenant in database: %w", err)
	}
	
	// Provision tenant resources asynchronously
	go func() {
		if err := s.provisioner.ProvisionTenant(context.Background(), tenant); err != nil {
			s.logger.Error("Failed to provision tenant resources", 
				zap.String("tenant_id", tenant.ID), 
				zap.Error(err))
			
			// Update tenant status to indicate provisioning failure
			tenant.Status = StatusInactive
			if updateErr := s.repo.Update(context.Background(), tenant); updateErr != nil {
				s.logger.Error("Failed to update tenant status after provisioning failure",
					zap.String("tenant_id", tenant.ID),
					zap.Error(updateErr))
			}
			return
		}
		
		// Update tenant status to active after successful provisioning
		tenant.Status = StatusActive
		if err := s.repo.Update(context.Background(), tenant); err != nil {
			s.logger.Error("Failed to update tenant status after successful provisioning",
				zap.String("tenant_id", tenant.ID),
				zap.Error(err))
		}
		
		s.logger.Info("Tenant provisioned successfully", zap.String("tenant_id", tenant.ID))
	}()
	
	// Log tenant creation event
	s.logTenantEvent(ctx, tenant.ID, "tenant_created", "Tenant created and provisioning started", "")
	
	return tenant, nil
}

// GetTenant retrieves a tenant by ID
func (s *Service) GetTenant(ctx context.Context, id string) (*Tenant, error) {
	tenant, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	// Update last accessed timestamp
	now := time.Now()
	tenant.LastAccessedAt = &now
	if err := s.repo.UpdateLastAccessed(ctx, id, now); err != nil {
		s.logger.Warn("Failed to update last accessed timestamp", 
			zap.String("tenant_id", id), 
			zap.Error(err))
	}
	
	return tenant, nil
}

// GetTenantBySubdomain retrieves a tenant by subdomain
func (s *Service) GetTenantBySubdomain(ctx context.Context, subdomain string) (*Tenant, error) {
	tenant, err := s.repo.GetBySubdomain(ctx, subdomain)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant by subdomain: %w", err)
	}
	
	// Update last accessed timestamp
	now := time.Now()
	tenant.LastAccessedAt = &now
	if err := s.repo.UpdateLastAccessed(ctx, tenant.ID, now); err != nil {
		s.logger.Warn("Failed to update last accessed timestamp",
			zap.String("tenant_id", tenant.ID),
			zap.Error(err))
	}
	
	return tenant, nil
}

// UpdateTenant updates tenant information
func (s *Service) UpdateTenant(ctx context.Context, id string, req *TenantUpdateRequest) (*Tenant, error) {
	tenant, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	// Update fields if provided
	if req.Name != "" {
		tenant.Name = req.Name
	}
	if req.CustomDomain != "" {
		tenant.CustomDomain = req.CustomDomain
	}
	if req.Status != "" {
		oldStatus := tenant.Status
		tenant.Status = req.Status
		if oldStatus != req.Status {
			s.logTenantEvent(ctx, id, "status_changed", 
				fmt.Sprintf("Status changed from %s to %s", oldStatus, req.Status), "")
		}
	}
	if req.Plan != "" {
		tenant.Plan = req.Plan
		tenant.ResourceQuota = s.getDefaultQuotaForPlan(req.Plan)
	}
	if req.ResourceQuota != nil {
		tenant.ResourceQuota = *req.ResourceQuota
	}
	if req.Config != nil {
		tenant.Config = *req.Config
	}
	if req.AdminEmail != "" {
		tenant.AdminEmail = req.AdminEmail
	}
	if req.AdminName != "" {
		tenant.AdminName = req.AdminName
	}
	if req.AdminPhone != "" {
		tenant.AdminPhone = req.AdminPhone
	}
	if req.BillingEmail != "" {
		tenant.BillingEmail = req.BillingEmail
	}
	if req.BillingPlan != "" {
		tenant.BillingPlan = req.BillingPlan
	}
	if req.BillingCycle != "" {
		tenant.BillingCycle = req.BillingCycle
	}
	
	tenant.UpdatedAt = time.Now()
	
	if err := s.repo.Update(ctx, tenant); err != nil {
		return nil, fmt.Errorf("failed to update tenant: %w", err)
	}
	
	s.logTenantEvent(ctx, id, "tenant_updated", "Tenant information updated", "")
	
	return tenant, nil
}

// SuspendTenant suspends a tenant
func (s *Service) SuspendTenant(ctx context.Context, id, reason string) error {
	tenant, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to get tenant: %w", err)
	}
	
	if tenant.Status == StatusSuspended {
		return fmt.Errorf("tenant is already suspended")
	}
	
	tenant.Status = StatusSuspended
	tenant.UpdatedAt = time.Now()
	
	if err := s.repo.Update(ctx, tenant); err != nil {
		return fmt.Errorf("failed to suspend tenant: %w", err)
	}
	
	s.logTenantEvent(ctx, id, "tenant_suspended", fmt.Sprintf("Tenant suspended: %s", reason), "")
	
	return nil
}

// ReactivateTenant reactivates a suspended tenant
func (s *Service) ReactivateTenant(ctx context.Context, id string) error {
	tenant, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to get tenant: %w", err)
	}
	
	if tenant.Status != StatusSuspended {
		return fmt.Errorf("tenant is not suspended")
	}
	
	tenant.Status = StatusActive
	tenant.UpdatedAt = time.Now()
	
	if err := s.repo.Update(ctx, tenant); err != nil {
		return fmt.Errorf("failed to reactivate tenant: %w", err)
	}
	
	s.logTenantEvent(ctx, id, "tenant_reactivated", "Tenant reactivated", "")
	
	return nil
}

// DeleteTenant soft deletes a tenant
func (s *Service) DeleteTenant(ctx context.Context, id string) error {
	tenant, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to get tenant: %w", err)
	}
	
	// Deprovision tenant resources
	if err := s.provisioner.DeprovisionTenant(ctx, tenant); err != nil {
		s.logger.Error("Failed to deprovision tenant resources",
			zap.String("tenant_id", id),
			zap.Error(err))
		// Continue with deletion even if deprovisioning fails
	}
	
	if err := s.repo.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete tenant: %w", err)
	}
	
	s.logTenantEvent(ctx, id, "tenant_deleted", "Tenant deleted", "")
	
	return nil
}

// ListTenants retrieves all tenants with optional filtering
func (s *Service) ListTenants(ctx context.Context, filter *TenantFilter) ([]*Tenant, error) {
	return s.repo.List(ctx, filter)
}

// GetTenantUsage retrieves usage metrics for a tenant
func (s *Service) GetTenantUsage(ctx context.Context, tenantID string, from, to time.Time) ([]*TenantUsageMetrics, error) {
	return s.repo.GetUsageMetrics(ctx, tenantID, from, to)
}

// UpdateTenantUsage updates real-time usage metrics
func (s *Service) UpdateTenantUsage(ctx context.Context, metrics *TenantUsageMetrics) error {
	return s.repo.UpdateUsageMetrics(ctx, metrics)
}

// CheckQuotaExceeded checks if tenant has exceeded any quota limits
func (s *Service) CheckQuotaExceeded(ctx context.Context, tenantID string) (bool, []string, error) {
	tenant, err := s.repo.GetByID(ctx, tenantID)
	if err != nil {
		return false, nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	var violations []string
	quota := tenant.ResourceQuota
	
	if tenant.CurrentUsers > quota.MaxUsers {
		violations = append(violations, fmt.Sprintf("User limit exceeded: %d/%d", tenant.CurrentUsers, quota.MaxUsers))
	}
	
	if tenant.StorageUsedMB > quota.MaxStorage {
		violations = append(violations, fmt.Sprintf("Storage limit exceeded: %dMB/%dMB", tenant.StorageUsedMB, quota.MaxStorage))
	}
	
	if tenant.BandwidthUsedMB > quota.MaxBandwidth {
		violations = append(violations, fmt.Sprintf("Bandwidth limit exceeded: %dMB/%dMB", tenant.BandwidthUsedMB, quota.MaxBandwidth))
	}
	
	return len(violations) > 0, violations, nil
}

// getDefaultQuotaForPlan returns default resource quota for a plan
func (s *Service) getDefaultQuotaForPlan(plan string) ResourceQuota {
	switch plan {
	case "trial":
		return ResourceQuota{
			MaxUsers:       10,
			MaxStorage:     1024,  // 1GB
			MaxAPIRequests: 1000,  // per hour
			MaxBandwidth:   5120,  // 5GB
			CPULimit:       0.5,
			MemoryLimit:    512,   // 512MB
		}
	case "starter":
		return ResourceQuota{
			MaxUsers:       50,
			MaxStorage:     10240, // 10GB
			MaxAPIRequests: 10000,
			MaxBandwidth:   51200, // 50GB
			CPULimit:       1.0,
			MemoryLimit:    1024,  // 1GB
		}
	case "professional":
		return ResourceQuota{
			MaxUsers:       500,
			MaxStorage:     102400, // 100GB
			MaxAPIRequests: 100000,
			MaxBandwidth:   512000, // 500GB
			CPULimit:       4.0,
			MemoryLimit:    4096,   // 4GB
		}
	case "enterprise":
		return ResourceQuota{
			MaxUsers:       -1,    // unlimited
			MaxStorage:     -1,    // unlimited
			MaxAPIRequests: -1,    // unlimited
			MaxBandwidth:   -1,    // unlimited
			CPULimit:       8.0,
			MemoryLimit:    8192,  // 8GB
		}
	default:
		return ResourceQuota{
			MaxUsers:       10,
			MaxStorage:     1024,
			MaxAPIRequests: 1000,
			MaxBandwidth:   5120,
			CPULimit:       0.5,
			MemoryLimit:    512,
		}
	}
}

// logTenantEvent logs an event in tenant lifecycle
func (s *Service) logTenantEvent(ctx context.Context, tenantID, eventType, description, metadata string) {
	event := &TenantEvent{
		ID:          uuid.New().String(),
		TenantID:    tenantID,
		EventType:   eventType,
		Description: description,
		Metadata:    metadata,
		CreatedAt:   time.Now(),
		CreatedBy:   "system", // TODO: get from context
	}
	
	if err := s.repo.CreateEvent(ctx, event); err != nil {
		s.logger.Error("Failed to log tenant event",
			zap.String("tenant_id", tenantID),
			zap.String("event_type", eventType),
			zap.Error(err))
	}
}