package tenant

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"
	"go.uber.org/zap"
)

// PostgresRepository implements Repository using PostgreSQL
type PostgresRepository struct {
	db     *sqlx.DB
	logger *zap.Logger
}

// NewPostgresRepository creates a new PostgreSQL repository
func NewPostgresRepository(db *sqlx.DB, logger *zap.Logger) *PostgresRepository {
	return &PostgresRepository{
		db:     db,
		logger: logger,
	}
}

// Create creates a new tenant
func (r *PostgresRepository) Create(ctx context.Context, tenant *Tenant) error {
	query := `
		INSERT INTO tenants (
			id, name, subdomain, custom_domain, status, isolation_strategy,
			database_name, schema_name, plan, resource_quota, config,
			created_at, updated_at, admin_email, admin_name, admin_phone,
			billing_email, billing_plan, billing_cycle, trial_ends_at
		) VALUES (
			:id, :name, :subdomain, :custom_domain, :status, :isolation_strategy,
			:database_name, :schema_name, :plan, :resource_quota, :config,
			:created_at, :updated_at, :admin_email, :admin_name, :admin_phone,
			:billing_email, :billing_plan, :billing_cycle, :trial_ends_at
		)`
	
	_, err := r.db.NamedExecContext(ctx, query, tenant)
	if err != nil {
		r.logger.Error("Failed to create tenant", zap.String("tenant_id", tenant.ID), zap.Error(err))
		return fmt.Errorf("failed to create tenant: %w", err)
	}
	
	return nil
}

// GetByID retrieves a tenant by ID
func (r *PostgresRepository) GetByID(ctx context.Context, id string) (*Tenant, error) {
	query := `
		SELECT * FROM tenants WHERE id = $1 AND deleted_at IS NULL`
	
	var tenant Tenant
	err := r.db.GetContext(ctx, &tenant, query, id)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("tenant not found: %s", id)
		}
		r.logger.Error("Failed to get tenant by ID", zap.String("tenant_id", id), zap.Error(err))
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	return &tenant, nil
}

// GetBySubdomain retrieves a tenant by subdomain
func (r *PostgresRepository) GetBySubdomain(ctx context.Context, subdomain string) (*Tenant, error) {
	query := `
		SELECT * FROM tenants WHERE subdomain = $1 AND deleted_at IS NULL`
	
	var tenant Tenant
	err := r.db.GetContext(ctx, &tenant, query, subdomain)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("tenant not found for subdomain: %s", subdomain)
		}
		r.logger.Error("Failed to get tenant by subdomain", zap.String("subdomain", subdomain), zap.Error(err))
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	return &tenant, nil
}

// GetByCustomDomain retrieves a tenant by custom domain
func (r *PostgresRepository) GetByCustomDomain(ctx context.Context, domain string) (*Tenant, error) {
	query := `
		SELECT * FROM tenants WHERE custom_domain = $1 AND deleted_at IS NULL`
	
	var tenant Tenant
	err := r.db.GetContext(ctx, &tenant, query, domain)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("tenant not found for domain: %s", domain)
		}
		r.logger.Error("Failed to get tenant by custom domain", zap.String("domain", domain), zap.Error(err))
		return nil, fmt.Errorf("failed to get tenant: %w", err)
	}
	
	return &tenant, nil
}

// Update updates a tenant
func (r *PostgresRepository) Update(ctx context.Context, tenant *Tenant) error {
	query := `
		UPDATE tenants SET
			name = :name,
			custom_domain = :custom_domain,
			status = :status,
			plan = :plan,
			resource_quota = :resource_quota,
			config = :config,
			updated_at = :updated_at,
			admin_email = :admin_email,
			admin_name = :admin_name,
			admin_phone = :admin_phone,
			billing_email = :billing_email,
			billing_plan = :billing_plan,
			billing_cycle = :billing_cycle,
			trial_ends_at = :trial_ends_at,
			current_users = :current_users,
			storage_used_mb = :storage_used_mb,
			monthly_requests = :monthly_requests,
			bandwidth_used_mb = :bandwidth_used_mb
		WHERE id = :id AND deleted_at IS NULL`
	
	result, err := r.db.NamedExecContext(ctx, query, tenant)
	if err != nil {
		r.logger.Error("Failed to update tenant", zap.String("tenant_id", tenant.ID), zap.Error(err))
		return fmt.Errorf("failed to update tenant: %w", err)
	}
	
	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found or already deleted: %s", tenant.ID)
	}
	
	return nil
}

// Delete soft deletes a tenant
func (r *PostgresRepository) Delete(ctx context.Context, id string) error {
	query := `
		UPDATE tenants SET 
			deleted_at = NOW(),
			status = 'inactive'
		WHERE id = $1 AND deleted_at IS NULL`
	
	result, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		r.logger.Error("Failed to delete tenant", zap.String("tenant_id", id), zap.Error(err))
		return fmt.Errorf("failed to delete tenant: %w", err)
	}
	
	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", id)
	}
	
	return nil
}

// List retrieves tenants with optional filtering
func (r *PostgresRepository) List(ctx context.Context, filter *TenantFilter) ([]*Tenant, error) {
	query := "SELECT * FROM tenants WHERE deleted_at IS NULL"
	args := []interface{}{}
	argIndex := 1
	
	if filter != nil {
		if len(filter.Status) > 0 {
			placeholders := make([]string, len(filter.Status))
			for i, status := range filter.Status {
				placeholders[i] = fmt.Sprintf("$%d", argIndex)
				args = append(args, status)
				argIndex++
			}
			query += fmt.Sprintf(" AND status IN (%s)", strings.Join(placeholders, ","))
		}
		
		if len(filter.Plan) > 0 {
			placeholders := make([]string, len(filter.Plan))
			for i, plan := range filter.Plan {
				placeholders[i] = fmt.Sprintf("$%d", argIndex)
				args = append(args, plan)
				argIndex++
			}
			query += fmt.Sprintf(" AND plan IN (%s)", strings.Join(placeholders, ","))
		}
		
		if len(filter.IsolationStrategy) > 0 {
			placeholders := make([]string, len(filter.IsolationStrategy))
			for i, strategy := range filter.IsolationStrategy {
				placeholders[i] = fmt.Sprintf("$%d", argIndex)
				args = append(args, strategy)
				argIndex++
			}
			query += fmt.Sprintf(" AND isolation_strategy IN (%s)", strings.Join(placeholders, ","))
		}
		
		if filter.CreatedAfter != nil {
			query += fmt.Sprintf(" AND created_at >= $%d", argIndex)
			args = append(args, *filter.CreatedAfter)
			argIndex++
		}
		
		if filter.CreatedBefore != nil {
			query += fmt.Sprintf(" AND created_at <= $%d", argIndex)
			args = append(args, *filter.CreatedBefore)
			argIndex++
		}
		
		if filter.LastAccessedAfter != nil {
			query += fmt.Sprintf(" AND last_accessed_at >= $%d", argIndex)
			args = append(args, *filter.LastAccessedAfter)
			argIndex++
		}
		
		if filter.Search != "" {
			query += fmt.Sprintf(" AND (name ILIKE $%d OR subdomain ILIKE $%d OR admin_email ILIKE $%d)", argIndex, argIndex+1, argIndex+2)
			searchTerm := "%" + filter.Search + "%"
			args = append(args, searchTerm, searchTerm, searchTerm)
			argIndex += 3
		}
		
		// Sorting
		if filter.SortBy != "" {
			order := "ASC"
			if filter.SortOrder == "DESC" {
				order = "DESC"
			}
			query += fmt.Sprintf(" ORDER BY %s %s", filter.SortBy, order)
		} else {
			query += " ORDER BY created_at DESC"
		}
		
		// Pagination
		if filter.Limit > 0 {
			query += fmt.Sprintf(" LIMIT $%d", argIndex)
			args = append(args, filter.Limit)
			argIndex++
		}
		
		if filter.Offset > 0 {
			query += fmt.Sprintf(" OFFSET $%d", argIndex)
			args = append(args, filter.Offset)
		}
	} else {
		query += " ORDER BY created_at DESC"
	}
	
	var tenants []*Tenant
	err := r.db.SelectContext(ctx, &tenants, query, args...)
	if err != nil {
		r.logger.Error("Failed to list tenants", zap.Error(err))
		return nil, fmt.Errorf("failed to list tenants: %w", err)
	}
	
	return tenants, nil
}

// Count returns the total number of tenants matching the filter
func (r *PostgresRepository) Count(ctx context.Context, filter *TenantFilter) (int64, error) {
	query := "SELECT COUNT(*) FROM tenants WHERE deleted_at IS NULL"
	args := []interface{}{}
	argIndex := 1
	
	if filter != nil {
		if len(filter.Status) > 0 {
			placeholders := make([]string, len(filter.Status))
			for i, status := range filter.Status {
				placeholders[i] = fmt.Sprintf("$%d", argIndex)
				args = append(args, status)
				argIndex++
			}
			query += fmt.Sprintf(" AND status IN (%s)", strings.Join(placeholders, ","))
		}
		
		if len(filter.Plan) > 0 {
			placeholders := make([]string, len(filter.Plan))
			for i, plan := range filter.Plan {
				placeholders[i] = fmt.Sprintf("$%d", argIndex)
				args = append(args, plan)
				argIndex++
			}
			query += fmt.Sprintf(" AND plan IN (%s)", strings.Join(placeholders, ","))
		}
		
		if filter.Search != "" {
			query += fmt.Sprintf(" AND (name ILIKE $%d OR subdomain ILIKE $%d OR admin_email ILIKE $%d)", argIndex, argIndex+1, argIndex+2)
			searchTerm := "%" + filter.Search + "%"
			args = append(args, searchTerm, searchTerm, searchTerm)
		}
	}
	
	var count int64
	err := r.db.GetContext(ctx, &count, query, args...)
	if err != nil {
		r.logger.Error("Failed to count tenants", zap.Error(err))
		return 0, fmt.Errorf("failed to count tenants: %w", err)
	}
	
	return count, nil
}

// SubdomainExists checks if a subdomain is already taken
func (r *PostgresRepository) SubdomainExists(ctx context.Context, subdomain string) (bool, error) {
	query := "SELECT EXISTS(SELECT 1 FROM tenants WHERE subdomain = $1 AND deleted_at IS NULL)"
	
	var exists bool
	err := r.db.GetContext(ctx, &exists, query, subdomain)
	if err != nil {
		r.logger.Error("Failed to check subdomain existence", zap.String("subdomain", subdomain), zap.Error(err))
		return false, fmt.Errorf("failed to check subdomain existence: %w", err)
	}
	
	return exists, nil
}

// CustomDomainExists checks if a custom domain is already taken
func (r *PostgresRepository) CustomDomainExists(ctx context.Context, domain string) (bool, error) {
	query := "SELECT EXISTS(SELECT 1 FROM tenants WHERE custom_domain = $1 AND deleted_at IS NULL)"
	
	var exists bool
	err := r.db.GetContext(ctx, &exists, query, domain)
	if err != nil {
		r.logger.Error("Failed to check custom domain existence", zap.String("domain", domain), zap.Error(err))
		return false, fmt.Errorf("failed to check custom domain existence: %w", err)
	}
	
	return exists, nil
}

// UpdateLastAccessed updates the last accessed timestamp for a tenant
func (r *PostgresRepository) UpdateLastAccessed(ctx context.Context, id string, timestamp time.Time) error {
	query := "UPDATE tenants SET last_accessed_at = $1 WHERE id = $2"
	
	_, err := r.db.ExecContext(ctx, query, timestamp, id)
	if err != nil {
		return fmt.Errorf("failed to update last accessed timestamp: %w", err)
	}
	
	return nil
}

// UpdateUsageMetrics updates usage metrics for a tenant
func (r *PostgresRepository) UpdateUsageMetrics(ctx context.Context, metrics *TenantUsageMetrics) error {
	// Insert new metrics record
	query := `
		INSERT INTO tenant_usage_metrics (
			tenant_id, timestamp, active_users, api_requests, storage_used_mb,
			bandwidth_used_mb, cpu_usage_percent, memory_usage_mb, response_time_ms, error_rate
		) VALUES (
			:tenant_id, :timestamp, :active_users, :api_requests, :storage_used_mb,
			:bandwidth_used_mb, :cpu_usage_percent, :memory_usage_mb, :response_time_ms, :error_rate
		)`
	
	_, err := r.db.NamedExecContext(ctx, query, metrics)
	if err != nil {
		return fmt.Errorf("failed to update usage metrics: %w", err)
	}
	
	// Update current usage in tenants table
	updateQuery := `
		UPDATE tenants SET
			current_users = $1,
			storage_used_mb = $2,
			bandwidth_used_mb = $3,
			updated_at = NOW()
		WHERE id = $4`
	
	_, err = r.db.ExecContext(ctx, updateQuery, 
		metrics.ActiveUsers, metrics.StorageUsedMB, metrics.BandwidthUsedMB, metrics.TenantID)
	if err != nil {
		r.logger.Warn("Failed to update current usage in tenants table", 
			zap.String("tenant_id", metrics.TenantID), zap.Error(err))
	}
	
	return nil
}

// GetUsageMetrics retrieves usage metrics for a tenant in a time range
func (r *PostgresRepository) GetUsageMetrics(ctx context.Context, tenantID string, from, to time.Time) ([]*TenantUsageMetrics, error) {
	query := `
		SELECT * FROM tenant_usage_metrics
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp <= $3
		ORDER BY timestamp DESC`
	
	var metrics []*TenantUsageMetrics
	err := r.db.SelectContext(ctx, &metrics, query, tenantID, from, to)
	if err != nil {
		return nil, fmt.Errorf("failed to get usage metrics: %w", err)
	}
	
	return metrics, nil
}

// GetAggregatedUsage retrieves aggregated usage metrics for a tenant
func (r *PostgresRepository) GetAggregatedUsage(ctx context.Context, tenantID string, from, to time.Time) (*TenantUsageMetrics, error) {
	query := `
		SELECT 
			tenant_id,
			AVG(active_users) as active_users,
			SUM(api_requests) as api_requests,
			MAX(storage_used_mb) as storage_used_mb,
			SUM(bandwidth_used_mb) as bandwidth_used_mb,
			AVG(cpu_usage_percent) as cpu_usage_percent,
			AVG(memory_usage_mb) as memory_usage_mb,
			AVG(response_time_ms) as response_time_ms,
			AVG(error_rate) as error_rate
		FROM tenant_usage_metrics
		WHERE tenant_id = $1 AND timestamp >= $2 AND timestamp <= $3
		GROUP BY tenant_id`
	
	var metrics TenantUsageMetrics
	err := r.db.GetContext(ctx, &metrics, query, tenantID, from, to)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get aggregated usage: %w", err)
	}
	
	return &metrics, nil
}

// CreateEvent creates a tenant event
func (r *PostgresRepository) CreateEvent(ctx context.Context, event *TenantEvent) error {
	query := `
		INSERT INTO tenant_events (id, tenant_id, event_type, description, metadata, created_at, created_by)
		VALUES (:id, :tenant_id, :event_type, :description, :metadata, :created_at, :created_by)`
	
	_, err := r.db.NamedExecContext(ctx, query, event)
	if err != nil {
		return fmt.Errorf("failed to create tenant event: %w", err)
	}
	
	return nil
}

// GetEvents retrieves events for a tenant
func (r *PostgresRepository) GetEvents(ctx context.Context, tenantID string, limit, offset int) ([]*TenantEvent, error) {
	query := `
		SELECT * FROM tenant_events
		WHERE tenant_id = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3`
	
	var events []*TenantEvent
	err := r.db.SelectContext(ctx, &events, query, tenantID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant events: %w", err)
	}
	
	return events, nil
}

// BulkUpdateStatus updates status for multiple tenants
func (r *PostgresRepository) BulkUpdateStatus(ctx context.Context, tenantIDs []string, status TenantStatus) error {
	if len(tenantIDs) == 0 {
		return nil
	}
	
	placeholders := make([]string, len(tenantIDs))
	args := make([]interface{}, len(tenantIDs)+1)
	args[0] = status
	
	for i, id := range tenantIDs {
		placeholders[i] = fmt.Sprintf("$%d", i+2)
		args[i+1] = id
	}
	
	query := fmt.Sprintf("UPDATE tenants SET status = $1, updated_at = NOW() WHERE id IN (%s)", 
		strings.Join(placeholders, ","))
	
	_, err := r.db.ExecContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("failed to bulk update status: %w", err)
	}
	
	return nil
}

// BulkUpdateQuota updates resource quota for multiple tenants
func (r *PostgresRepository) BulkUpdateQuota(ctx context.Context, tenantIDs []string, quota ResourceQuota) error {
	if len(tenantIDs) == 0 {
		return nil
	}
	
	placeholders := make([]string, len(tenantIDs))
	args := make([]interface{}, len(tenantIDs)+1)
	args[0] = quota
	
	for i, id := range tenantIDs {
		placeholders[i] = fmt.Sprintf("$%d", i+2)
		args[i+1] = id
	}
	
	query := fmt.Sprintf("UPDATE tenants SET resource_quota = $1, updated_at = NOW() WHERE id IN (%s)",
		strings.Join(placeholders, ","))
	
	_, err := r.db.ExecContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("failed to bulk update quota: %w", err)
	}
	
	return nil
}

// GetTenantStats retrieves aggregate statistics about tenants
func (r *PostgresRepository) GetTenantStats(ctx context.Context) (*TenantStats, error) {
	query := `
		SELECT 
			COUNT(*) as total_tenants,
			COUNT(CASE WHEN status = 'active' THEN 1 END) as active_tenants,
			COUNT(CASE WHEN status = 'suspended' THEN 1 END) as suspended_tenants,
			COUNT(CASE WHEN plan = 'trial' THEN 1 END) as trial_tenants,
			COUNT(CASE WHEN trial_ends_at < NOW() THEN 1 END) as expired_trials,
			AVG(current_users) as average_users,
			SUM(storage_used_mb) / 1024.0 as total_storage_gb,
			SUM(bandwidth_used_mb) / 1024.0 as total_bandwidth_gb
		FROM tenants 
		WHERE deleted_at IS NULL`
	
	var stats TenantStats
	err := r.db.GetContext(ctx, &stats, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant stats: %w", err)
	}
	
	// Get tenants by plan
	planQuery := "SELECT plan, COUNT(*) as count FROM tenants WHERE deleted_at IS NULL GROUP BY plan"
	rows, err := r.db.QueryContext(ctx, planQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenants by plan: %w", err)
	}
	defer rows.Close()
	
	stats.TenantsByPlan = make(map[string]int64)
	for rows.Next() {
		var plan string
		var count int64
		if err := rows.Scan(&plan, &count); err != nil {
			return nil, err
		}
		stats.TenantsByPlan[plan] = count
	}
	
	// Get tenants by isolation strategy
	strategyQuery := "SELECT isolation_strategy, COUNT(*) as count FROM tenants WHERE deleted_at IS NULL GROUP BY isolation_strategy"
	rows2, err := r.db.QueryContext(ctx, strategyQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenants by strategy: %w", err)
	}
	defer rows2.Close()
	
	stats.TenantsByStrategy = make(map[string]int64)
	for rows2.Next() {
		var strategy string
		var count int64
		if err := rows2.Scan(&strategy, &count); err != nil {
			return nil, err
		}
		stats.TenantsByStrategy[strategy] = count
	}
	
	return &stats, nil
}

// GetTenantsByPlan returns count of tenants grouped by plan
func (r *PostgresRepository) GetTenantsByPlan(ctx context.Context) (map[string]int64, error) {
	query := "SELECT plan, COUNT(*) as count FROM tenants WHERE deleted_at IS NULL GROUP BY plan"
	
	rows, err := r.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenants by plan: %w", err)
	}
	defer rows.Close()
	
	result := make(map[string]int64)
	for rows.Next() {
		var plan string
		var count int64
		if err := rows.Scan(&plan, &count); err != nil {
			return nil, err
		}
		result[plan] = count
	}
	
	return result, nil
}

// GetActiveTenantsCount returns count of active tenants
func (r *PostgresRepository) GetActiveTenantsCount(ctx context.Context) (int64, error) {
	query := "SELECT COUNT(*) FROM tenants WHERE status = 'active' AND deleted_at IS NULL"
	
	var count int64
	err := r.db.GetContext(ctx, &count, query)
	if err != nil {
		return 0, fmt.Errorf("failed to get active tenants count: %w", err)
	}
	
	return count, nil
}

// GetExpiredTrialsCount returns count of tenants with expired trials
func (r *PostgresRepository) GetExpiredTrialsCount(ctx context.Context) (int64, error) {
	query := "SELECT COUNT(*) FROM tenants WHERE trial_ends_at < NOW() AND deleted_at IS NULL"
	
	var count int64
	err := r.db.GetContext(ctx, &count, query)
	if err != nil {
		return 0, fmt.Errorf("failed to get expired trials count: %w", err)
	}
	
	return count, nil
}