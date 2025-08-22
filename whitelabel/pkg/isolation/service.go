package isolation

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/enterprise/whitelabel/pkg/tenant"
	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
	"go.uber.org/zap"
)

// Service handles data isolation operations
type Service struct {
	config          *IsolationConfig
	masterDB        *sqlx.DB
	tenantDBs       map[string]*sqlx.DB
	connectionPools map[string]*ConnectionPool
	repo            Repository
	provisioner     DatabaseProvisioner
	migrationRunner MigrationRunner
	backupManager   BackupManager
	logger          *zap.Logger
	mu              sync.RWMutex
}

// NewService creates a new isolation service
func NewService(
	config *IsolationConfig,
	masterDB *sqlx.DB,
	repo Repository,
	provisioner DatabaseProvisioner,
	migrationRunner MigrationRunner,
	backupManager BackupManager,
	logger *zap.Logger,
) *Service {
	return &Service{
		config:          config,
		masterDB:        masterDB,
		tenantDBs:       make(map[string]*sqlx.DB),
		connectionPools: make(map[string]*ConnectionPool),
		repo:            repo,
		provisioner:     provisioner,
		migrationRunner: migrationRunner,
		backupManager:   backupManager,
		logger:          logger,
	}
}

// ProvisionTenantDatabase creates and configures a database for a tenant
func (s *Service) ProvisionTenantDatabase(ctx context.Context, tenantObj *tenant.Tenant) error {
	s.logger.Info("Provisioning tenant database",
		zap.String("tenant_id", tenantObj.ID),
		zap.String("isolation_strategy", string(tenantObj.IsolationStrategy)))

	// Create tenant database configuration
	tenantDB := &TenantDatabase{
		TenantID:       tenantObj.ID,
		IsolationLevel: s.mapIsolationStrategy(tenantObj.IsolationStrategy),
		Config:         *s.config,
		Status:         StatusProvisioning,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	// Set database/schema names based on isolation level
	switch tenantDB.IsolationLevel {
	case DatabaseLevel:
		tenantDB.DatabaseName = s.config.DatabasePrefix + tenantObj.ID
		if err := s.provisioner.CreateDatabase(ctx, tenantDB.DatabaseName); err != nil {
			return fmt.Errorf("failed to create database: %w", err)
		}
	case SchemaLevel:
		tenantDB.SchemaName = s.config.SchemaPrefix + tenantObj.ID
		if err := s.provisioner.CreateSchema(ctx, s.masterDB, tenantDB.SchemaName); err != nil {
			return fmt.Errorf("failed to create schema: %w", err)
		}
	case RowLevel:
		// Use master database with row-level security
		tenantDB.DatabaseName = "master"
		tenantDB.SchemaName = "public"
	}

	// Build connection string
	tenantDB.ConnectionString = s.buildConnectionString(tenantDB)

	// Save tenant database configuration
	if err := s.repo.CreateTenantDatabase(ctx, tenantDB); err != nil {
		return fmt.Errorf("failed to save tenant database config: %w", err)
	}

	// Create database connection
	if err := s.createTenantConnection(ctx, tenantDB); err != nil {
		tenantDB.Status = StatusError
		s.repo.UpdateTenantDatabase(ctx, tenantDB)
		return fmt.Errorf("failed to create tenant connection: %w", err)
	}

	// Run initial migrations
	if s.config.MigrationConfig.AutoMigrate {
		if err := s.runTenantMigrations(ctx, tenantObj.ID); err != nil {
			s.logger.Error("Failed to run tenant migrations",
				zap.String("tenant_id", tenantObj.ID),
				zap.Error(err))
		}
	}

	// Update status to active
	tenantDB.Status = StatusActive
	tenantDB.UpdatedAt = time.Now()
	if err := s.repo.UpdateTenantDatabase(ctx, tenantDB); err != nil {
		s.logger.Error("Failed to update tenant database status",
			zap.String("tenant_id", tenantObj.ID),
			zap.Error(err))
	}

	s.logger.Info("Successfully provisioned tenant database", zap.String("tenant_id", tenantObj.ID))
	return nil
}

// GetTenantConnection retrieves a database connection for a tenant
func (s *Service) GetTenantConnection(ctx context.Context, tenantID string) (*sqlx.DB, error) {
	s.mu.RLock()
	if db, exists := s.tenantDBs[tenantID]; exists {
		s.mu.RUnlock()
		return db, nil
	}
	s.mu.RUnlock()

	// Load tenant database configuration
	tenantDB, err := s.repo.GetTenantDatabase(ctx, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant database config: %w", err)
	}

	// Create connection if not exists
	if err := s.createTenantConnection(ctx, tenantDB); err != nil {
		return nil, fmt.Errorf("failed to create tenant connection: %w", err)
	}

	s.mu.RLock()
	db := s.tenantDBs[tenantID]
	s.mu.RUnlock()

	return db, nil
}

// ExecuteQuery executes a query with proper isolation
func (s *Service) ExecuteQuery(ctx context.Context, tenantID, query string, args ...interface{}) (*sql.Rows, error) {
	startTime := time.Now()
	
	// Get tenant connection
	db, err := s.GetTenantConnection(ctx, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant connection: %w", err)
	}

	// Apply row-level security if needed
	query = s.applyRowLevelSecurity(tenantID, query)

	// Execute query
	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		s.logQueryError(tenantID, query, err)
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}

	// Record performance metrics
	duration := time.Since(startTime)
	s.recordQueryPerformance(tenantID, query, duration, rows)

	// Log data access for auditing
	s.logDataAccess(ctx, tenantID, "SELECT", query, duration, err == nil)

	return rows, nil
}

// ExecuteTransaction executes queries within a transaction
func (s *Service) ExecuteTransaction(ctx context.Context, tenantID string, fn func(*sqlx.Tx) error) error {
	db, err := s.GetTenantConnection(ctx, tenantID)
	if err != nil {
		return fmt.Errorf("failed to get tenant connection: %w", err)
	}

	tx, err := db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if err != nil {
			if rollbackErr := tx.Rollback(); rollbackErr != nil {
				s.logger.Error("Failed to rollback transaction",
					zap.String("tenant_id", tenantID),
					zap.Error(rollbackErr))
			}
		}
	}()

	if err = fn(tx); err != nil {
		return err
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// MigrateTenant runs migrations for a specific tenant
func (s *Service) MigrateTenant(ctx context.Context, tenantID string) error {
	s.logger.Info("Running tenant migrations", zap.String("tenant_id", tenantID))
	
	return s.runTenantMigrations(ctx, tenantID)
}

// BackupTenantDatabase creates a backup for a tenant database
func (s *Service) BackupTenantDatabase(ctx context.Context, tenantID string) error {
	s.logger.Info("Starting tenant database backup", zap.String("tenant_id", tenantID))

	tenantDB, err := s.repo.GetTenantDatabase(ctx, tenantID)
	if err != nil {
		return fmt.Errorf("failed to get tenant database config: %w", err)
	}

	if !tenantDB.Config.BackupConfig.Enabled {
		return fmt.Errorf("backup is not enabled for tenant: %s", tenantID)
	}

	// Create backup
	backupSize, err := s.backupManager.CreateBackup(ctx, tenantDB)
	if err != nil {
		return fmt.Errorf("failed to create backup: %w", err)
	}

	// Update last backup timestamp
	now := time.Now()
	tenantDB.LastBackup = &now
	tenantDB.BackupSize = backupSize
	tenantDB.UpdatedAt = now

	if err := s.repo.UpdateTenantDatabase(ctx, tenantDB); err != nil {
		s.logger.Error("Failed to update backup timestamp",
			zap.String("tenant_id", tenantID),
			zap.Error(err))
	}

	s.logger.Info("Successfully backed up tenant database",
		zap.String("tenant_id", tenantID),
		zap.Int64("backup_size", backupSize))

	return nil
}

// RestoreTenantDatabase restores a tenant database from backup
func (s *Service) RestoreTenantDatabase(ctx context.Context, tenantID, backupID string) error {
	s.logger.Info("Restoring tenant database from backup",
		zap.String("tenant_id", tenantID),
		zap.String("backup_id", backupID))

	tenantDB, err := s.repo.GetTenantDatabase(ctx, tenantID)
	if err != nil {
		return fmt.Errorf("failed to get tenant database config: %w", err)
	}

	// Set status to maintenance
	tenantDB.Status = StatusMaintenance
	if err := s.repo.UpdateTenantDatabase(ctx, tenantDB); err != nil {
		return fmt.Errorf("failed to update database status: %w", err)
	}

	// Close existing connections
	s.closeTenantConnection(tenantID)

	// Restore from backup
	if err := s.backupManager.RestoreBackup(ctx, tenantDB, backupID); err != nil {
		tenantDB.Status = StatusError
		s.repo.UpdateTenantDatabase(ctx, tenantDB)
		return fmt.Errorf("failed to restore from backup: %w", err)
	}

	// Recreate connections
	if err := s.createTenantConnection(ctx, tenantDB); err != nil {
		tenantDB.Status = StatusError
		s.repo.UpdateTenantDatabase(ctx, tenantDB)
		return fmt.Errorf("failed to recreate connections: %w", err)
	}

	// Update status to active
	tenantDB.Status = StatusActive
	tenantDB.UpdatedAt = time.Now()
	if err := s.repo.UpdateTenantDatabase(ctx, tenantDB); err != nil {
		s.logger.Error("Failed to update database status after restore",
			zap.String("tenant_id", tenantID),
			zap.Error(err))
	}

	s.logger.Info("Successfully restored tenant database", zap.String("tenant_id", tenantID))
	return nil
}

// GetTenantMetrics retrieves isolation metrics for a tenant
func (s *Service) GetTenantMetrics(ctx context.Context, tenantID string, from, to time.Time) ([]*IsolationMetrics, error) {
	return s.repo.GetIsolationMetrics(ctx, tenantID, from, to)
}

// GetTenantDataSummary retrieves a summary of tenant data
func (s *Service) GetTenantDataSummary(ctx context.Context, tenantID string) (*TenantDataSummary, error) {
	return s.repo.GetTenantDataSummary(ctx, tenantID)
}

// HealthCheck performs a health check on tenant database
func (s *Service) HealthCheck(ctx context.Context, tenantID string) (*DatabaseHealth, error) {
	db, err := s.GetTenantConnection(ctx, tenantID)
	if err != nil {
		return &DatabaseHealth{
			Status:          "unhealthy",
			LastHealthCheck: time.Now(),
			Errors:          []string{err.Error()},
		}, nil
	}

	startTime := time.Now()
	
	// Test basic connectivity
	if err := db.PingContext(ctx); err != nil {
		return &DatabaseHealth{
			Status:          "unhealthy",
			LastHealthCheck: time.Now(),
			ResponseTime:    time.Since(startTime),
			Errors:          []string{err.Error()},
		}, nil
	}

	// Get connection stats
	stats := db.Stats()
	
	// Query performance metrics
	var activeQueries int
	db.GetContext(ctx, &activeQueries, "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")

	health := &DatabaseHealth{
		Status:             "healthy",
		LastHealthCheck:    time.Now(),
		ResponseTime:       time.Since(startTime),
		ConnectionCount:    stats.OpenConnections,
		ActiveQueries:      activeQueries,
		CacheHitRatio:      s.getCacheHitRatio(ctx, db),
		DiskUsagePercent:   s.getDiskUsagePercent(ctx, db),
		Errors:             []string{},
	}

	// Update tenant database health
	tenantDB, err := s.repo.GetTenantDatabase(ctx, tenantID)
	if err == nil {
		tenantDB.Health = *health
		tenantDB.ConnectionCount = stats.OpenConnections
		tenantDB.UpdatedAt = time.Now()
		s.repo.UpdateTenantDatabase(ctx, tenantDB)
	}

	return health, nil
}

// Helper methods

// mapIsolationStrategy maps tenant isolation strategy to isolation level
func (s *Service) mapIsolationStrategy(strategy tenant.IsolationStrategy) IsolationLevel {
	switch strategy {
	case tenant.DatabasePerTenant:
		return DatabaseLevel
	case tenant.SchemaPerTenant:
		return SchemaLevel
	case tenant.RowLevel:
		return RowLevel
	default:
		return SharedLevel
	}
}

// buildConnectionString builds connection string for tenant database
func (s *Service) buildConnectionString(tenantDB *TenantDatabase) string {
	baseConnStr := s.masterDB.DriverName()
	
	switch tenantDB.IsolationLevel {
	case DatabaseLevel:
		return strings.Replace(baseConnStr, "dbname=master", fmt.Sprintf("dbname=%s", tenantDB.DatabaseName), 1)
	case SchemaLevel:
		return fmt.Sprintf("%s search_path=%s", baseConnStr, tenantDB.SchemaName)
	default:
		return baseConnStr
	}
}

// createTenantConnection creates a database connection for tenant
func (s *Service) createTenantConnection(ctx context.Context, tenantDB *TenantDatabase) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Skip if connection already exists
	if _, exists := s.tenantDBs[tenantDB.TenantID]; exists {
		return nil
	}

	var db *sqlx.DB
	var err error

	switch tenantDB.IsolationLevel {
	case DatabaseLevel:
		// Create new connection to tenant database
		db, err = sqlx.ConnectContext(ctx, s.masterDB.DriverName(), tenantDB.ConnectionString)
	case SchemaLevel, RowLevel, SharedLevel:
		// Use master database but set search path or row-level security
		db = s.masterDB
	}

	if err != nil {
		return fmt.Errorf("failed to connect to tenant database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(tenantDB.Config.MaxConnections)
	db.SetMaxIdleConns(tenantDB.Config.ConnectionPoolSize)
	db.SetConnMaxLifetime(tenantDB.Config.ConnectionTimeout)

	s.tenantDBs[tenantDB.TenantID] = db

	// Initialize connection pool tracking
	s.connectionPools[tenantDB.TenantID] = &ConnectionPool{
		TenantID:          tenantDB.TenantID,
		DatabaseName:      tenantDB.DatabaseName,
		MaxConnections:    tenantDB.Config.MaxConnections,
		HealthStatus:      "healthy",
		LastActivity:      time.Now(),
	}

	return nil
}

// closeTenantConnection closes database connection for tenant
func (s *Service) closeTenantConnection(tenantID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if db, exists := s.tenantDBs[tenantID]; exists {
		if db != s.masterDB { // Don't close master connection
			db.Close()
		}
		delete(s.tenantDBs, tenantID)
	}

	delete(s.connectionPools, tenantID)
}

// applyRowLevelSecurity applies row-level security to queries
func (s *Service) applyRowLevelSecurity(tenantID, query string) string {
	if s.config.Level != RowLevel {
		return query
	}

	// Add tenant filter to WHERE clause
	// This is a simplified implementation - production would need more sophisticated query parsing
	if strings.Contains(strings.ToUpper(query), "WHERE") {
		return strings.Replace(query, "WHERE", fmt.Sprintf("WHERE %s = '%s' AND", s.config.RowLevelColumn, tenantID), 1)
	} else {
		// Add WHERE clause if none exists
		fromIndex := strings.Index(strings.ToUpper(query), "FROM")
		if fromIndex != -1 {
			// Find the end of FROM clause
			parts := strings.Split(query[fromIndex:], " ")
			if len(parts) >= 2 {
				tableName := parts[1]
				whereClause := fmt.Sprintf(" WHERE %s.%s = '%s'", tableName, s.config.RowLevelColumn, tenantID)
				return query + whereClause
			}
		}
	}

	return query
}

// runTenantMigrations runs database migrations for a tenant
func (s *Service) runTenantMigrations(ctx context.Context, tenantID string) error {
	db, err := s.GetTenantConnection(ctx, tenantID)
	if err != nil {
		return fmt.Errorf("failed to get tenant connection: %w", err)
	}

	return s.migrationRunner.RunMigrations(ctx, tenantID, db)
}

// recordQueryPerformance records query performance metrics
func (s *Service) recordQueryPerformance(tenantID, query string, duration time.Duration, rows *sql.Rows) {
	// Implementation for recording query performance metrics
	// This would typically involve extracting query patterns, counting rows, etc.
	
	performance := &QueryPerformance{
		TenantID:      tenantID,
		QueryHash:     s.hashQuery(query),
		QueryTemplate: s.extractQueryTemplate(query),
		ExecutionTime: duration,
		Timestamp:     time.Now(),
	}

	// Save to repository (async)
	go func() {
		if err := s.repo.CreateQueryPerformance(context.Background(), performance); err != nil {
			s.logger.Error("Failed to record query performance",
				zap.String("tenant_id", tenantID),
				zap.Error(err))
		}
	}()
}

// logDataAccess logs data access for auditing
func (s *Service) logDataAccess(ctx context.Context, tenantID, action, query string, duration time.Duration, success bool) {
	if !s.config.SecurityConfig.AuditLogging {
		return
	}

	log := &DataAccessLog{
		ID:           uuid.New().String(),
		TenantID:     tenantID,
		Action:       action,
		Query:        query,
		Timestamp:    time.Now(),
		Duration:     duration,
		Success:      success,
		RowsAffected: 0, // Would be populated from query result
	}

	// Save to repository (async)
	go func() {
		if err := s.repo.CreateDataAccessLog(context.Background(), log); err != nil {
			s.logger.Error("Failed to log data access",
				zap.String("tenant_id", tenantID),
				zap.Error(err))
		}
	}()
}

// logQueryError logs query errors
func (s *Service) logQueryError(tenantID, query string, err error) {
	s.logger.Error("Query execution failed",
		zap.String("tenant_id", tenantID),
		zap.String("query", query),
		zap.Error(err))
}

// getCacheHitRatio gets cache hit ratio from database
func (s *Service) getCacheHitRatio(ctx context.Context, db *sqlx.DB) float64 {
	var hitRatio float64
	query := "SELECT (blks_hit * 100.0) / (blks_hit + blks_read) as hit_ratio FROM pg_stat_database WHERE datname = current_database()"
	db.GetContext(ctx, &hitRatio, query)
	return hitRatio
}

// getDiskUsagePercent gets disk usage percentage
func (s *Service) getDiskUsagePercent(ctx context.Context, db *sqlx.DB) float64 {
	var usagePercent float64
	query := "SELECT (pg_database_size(current_database())::float / (1024*1024*1024)) as size_gb"
	db.GetContext(ctx, &usagePercent, query)
	return usagePercent
}

// hashQuery creates a hash of the query for grouping
func (s *Service) hashQuery(query string) string {
	// Simple implementation - production would use proper hashing
	return fmt.Sprintf("%x", len(query))
}

// extractQueryTemplate extracts a template from the query for pattern matching
func (s *Service) extractQueryTemplate(query string) string {
	// Replace literal values with placeholders
	// This is a simplified implementation
	template := query
	template = strings.ReplaceAll(template, "'[^']*'", "?")
	template = strings.ReplaceAll(template, "\\d+", "?")
	return template
}