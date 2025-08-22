package isolation

import (
	"context"
	"time"

	"github.com/jmoiron/sqlx"
)

// Repository defines the interface for isolation data operations
type Repository interface {
	// Tenant database operations
	CreateTenantDatabase(ctx context.Context, tenantDB *TenantDatabase) error
	GetTenantDatabase(ctx context.Context, tenantID string) (*TenantDatabase, error)
	UpdateTenantDatabase(ctx context.Context, tenantDB *TenantDatabase) error
	DeleteTenantDatabase(ctx context.Context, tenantID string) error
	ListTenantDatabases(ctx context.Context, filter *TenantDatabaseFilter) ([]*TenantDatabase, error)
	
	// Migration operations
	CreateMigration(ctx context.Context, migration *Migration) error
	GetMigration(ctx context.Context, tenantID, version string) (*Migration, error)
	ListMigrations(ctx context.Context, tenantID string) ([]*Migration, error)
	UpdateMigration(ctx context.Context, migration *Migration) error
	GetLatestMigrationVersion(ctx context.Context, tenantID string) (string, error)
	
	// Performance tracking
	CreateQueryPerformance(ctx context.Context, performance *QueryPerformance) error
	GetQueryPerformance(ctx context.Context, tenantID string, from, to time.Time) ([]*QueryPerformance, error)
	GetSlowQueries(ctx context.Context, tenantID string, threshold time.Duration) ([]*QueryPerformance, error)
	
	// Metrics operations
	CreateIsolationMetrics(ctx context.Context, metrics *IsolationMetrics) error
	GetIsolationMetrics(ctx context.Context, tenantID string, from, to time.Time) ([]*IsolationMetrics, error)
	GetAggregatedMetrics(ctx context.Context, tenantID string, from, to time.Time, groupBy string) ([]*IsolationMetrics, error)
	
	// Data access logging
	CreateDataAccessLog(ctx context.Context, log *DataAccessLog) error
	GetDataAccessLogs(ctx context.Context, tenantID string, from, to time.Time, limit int) ([]*DataAccessLog, error)
	GetDataAccessStatistics(ctx context.Context, tenantID string, from, to time.Time) (map[string]interface{}, error)
	
	// Tenant data summary
	GetTenantDataSummary(ctx context.Context, tenantID string) (*TenantDataSummary, error)
	UpdateTenantDataSummary(ctx context.Context, summary *TenantDataSummary) error
}

// TenantDatabaseFilter represents filtering options for tenant database queries
type TenantDatabaseFilter struct {
	Status         []DatabaseStatus  `json:"status,omitempty"`
	IsolationLevel []IsolationLevel  `json:"isolation_level,omitempty"`
	CreatedAfter   *time.Time        `json:"created_after,omitempty"`
	CreatedBefore  *time.Time        `json:"created_before,omitempty"`
	HealthStatus   []string          `json:"health_status,omitempty"`
	Limit          int               `json:"limit,omitempty"`
	Offset         int               `json:"offset,omitempty"`
}

// DatabaseProvisioner defines the interface for database provisioning
type DatabaseProvisioner interface {
	// Database operations
	CreateDatabase(ctx context.Context, databaseName string) error
	DropDatabase(ctx context.Context, databaseName string) error
	DatabaseExists(ctx context.Context, databaseName string) (bool, error)
	
	// Schema operations
	CreateSchema(ctx context.Context, db *sqlx.DB, schemaName string) error
	DropSchema(ctx context.Context, db *sqlx.DB, schemaName string) error
	SchemaExists(ctx context.Context, db *sqlx.DB, schemaName string) (bool, error)
	
	// User and permission management
	CreateDatabaseUser(ctx context.Context, username, password string) error
	GrantDatabaseAccess(ctx context.Context, username, databaseName string, permissions []string) error
	RevokeDatabaseAccess(ctx context.Context, username, databaseName string) error
	
	// Configuration
	SetDatabaseConfig(ctx context.Context, databaseName string, config map[string]string) error
	GetDatabaseConfig(ctx context.Context, databaseName string) (map[string]string, error)
}

// MigrationRunner defines the interface for running database migrations
type MigrationRunner interface {
	// Migration execution
	RunMigrations(ctx context.Context, tenantID string, db *sqlx.DB) error
	RunSpecificMigration(ctx context.Context, tenantID, version string, db *sqlx.DB) error
	RollbackMigration(ctx context.Context, tenantID, version string, db *sqlx.DB) error
	
	// Migration management
	LoadMigrations(migrationPath string) ([]*Migration, error)
	ValidateMigration(migration *Migration) error
	GetAppliedMigrations(ctx context.Context, tenantID string, db *sqlx.DB) ([]*Migration, error)
	
	// Migration status
	GetMigrationStatus(ctx context.Context, tenantID string, db *sqlx.DB) (map[string]MigrationStatus, error)
	IsMigrationTableCreated(ctx context.Context, db *sqlx.DB) (bool, error)
	CreateMigrationTable(ctx context.Context, db *sqlx.DB) error
}

// BackupManager defines the interface for database backups
type BackupManager interface {
	// Backup operations
	CreateBackup(ctx context.Context, tenantDB *TenantDatabase) (int64, error)
	RestoreBackup(ctx context.Context, tenantDB *TenantDatabase, backupID string) error
	DeleteBackup(ctx context.Context, backupID string) error
	
	// Backup management
	ListBackups(ctx context.Context, tenantID string) ([]*BackupInfo, error)
	GetBackupInfo(ctx context.Context, backupID string) (*BackupInfo, error)
	VerifyBackup(ctx context.Context, backupID string) error
	
	// Automated backups
	ScheduleBackup(ctx context.Context, tenantID, schedule string) error
	CancelScheduledBackup(ctx context.Context, tenantID string) error
	
	// Backup cleanup
	CleanupExpiredBackups(ctx context.Context) error
	GetBackupStorageUsage(ctx context.Context, tenantID string) (int64, error)
}

// BackupInfo represents information about a backup
type BackupInfo struct {
	ID            string                 `json:"id"`
	TenantID      string                 `json:"tenant_id"`
	DatabaseName  string                 `json:"database_name"`
	CreatedAt     time.Time              `json:"created_at"`
	Size          int64                  `json:"size"`
	Status        BackupStatus           `json:"status"`
	Location      string                 `json:"location"`
	Checksum      string                 `json:"checksum"`
	Metadata      map[string]interface{} `json:"metadata"`
	ExpiresAt     *time.Time             `json:"expires_at,omitempty"`
}

// BackupStatus represents the status of a backup
type BackupStatus string

const (
	BackupStatusCreating  BackupStatus = "creating"
	BackupStatusCompleted BackupStatus = "completed"
	BackupStatusFailed    BackupStatus = "failed"
	BackupStatusExpired   BackupStatus = "expired"
)

// ConnectionManager defines the interface for managing database connections
type ConnectionManager interface {
	// Connection pool management
	GetConnectionPool(tenantID string) (*ConnectionPool, error)
	UpdateConnectionPool(tenantID string, config ConnectionPoolConfig) error
	CloseConnectionPool(tenantID string) error
	
	// Connection monitoring
	GetConnectionStats(tenantID string) (*ConnectionStats, error)
	GetActiveConnections(tenantID string) ([]*ActiveConnection, error)
	KillConnection(tenantID, connectionID string) error
	
	// Health monitoring
	HealthCheck(tenantID string) (*ConnectionHealth, error)
	GetConnectionMetrics(tenantID string, from, to time.Time) ([]*ConnectionMetrics, error)
}

// ConnectionPoolConfig defines connection pool configuration
type ConnectionPoolConfig struct {
	MaxConnections    int           `json:"max_connections"`
	MinConnections    int           `json:"min_connections"`
	MaxIdleTime       time.Duration `json:"max_idle_time"`
	MaxLifetime       time.Duration `json:"max_lifetime"`
	ConnectTimeout    time.Duration `json:"connect_timeout"`
	QueryTimeout      time.Duration `json:"query_timeout"`
}

// ConnectionStats represents connection statistics
type ConnectionStats struct {
	TenantID          string    `json:"tenant_id"`
	MaxConnections    int       `json:"max_connections"`
	ActiveConnections int       `json:"active_connections"`
	IdleConnections   int       `json:"idle_connections"`
	WaitingConnections int      `json:"waiting_connections"`
	TotalConnections  int64     `json:"total_connections"`
	AverageWaitTime   time.Duration `json:"average_wait_time"`
	LastUpdated       time.Time `json:"last_updated"`
}

// ActiveConnection represents an active database connection
type ActiveConnection struct {
	ID            string        `json:"id"`
	TenantID      string        `json:"tenant_id"`
	UserID        string        `json:"user_id"`
	StartTime     time.Time     `json:"start_time"`
	State         string        `json:"state"`
	Query         string        `json:"query"`
	WaitEventType string        `json:"wait_event_type"`
	Duration      time.Duration `json:"duration"`
}

// ConnectionHealth represents connection health status
type ConnectionHealth struct {
	TenantID      string        `json:"tenant_id"`
	Status        string        `json:"status"`
	ResponseTime  time.Duration `json:"response_time"`
	ErrorCount    int           `json:"error_count"`
	LastError     string        `json:"last_error,omitempty"`
	CheckTime     time.Time     `json:"check_time"`
}

// ConnectionMetrics represents connection metrics over time
type ConnectionMetrics struct {
	TenantID      string        `json:"tenant_id"`
	Timestamp     time.Time     `json:"timestamp"`
	Connections   int           `json:"connections"`
	QueriesPerSec float64       `json:"queries_per_sec"`
	AvgWaitTime   time.Duration `json:"avg_wait_time"`
	ErrorRate     float64       `json:"error_rate"`
}

// DataMasker defines the interface for data masking
type DataMasker interface {
	// Masking operations
	MaskData(ctx context.Context, tenantID, table, column string, data interface{}) (interface{}, error)
	UnmaskData(ctx context.Context, tenantID, table, column string, maskedData interface{}) (interface{}, error)
	
	// Rule management
	AddMaskingRule(ctx context.Context, rule *MaskingRule) error
	RemoveMaskingRule(ctx context.Context, tenantID, table, column string) error
	GetMaskingRules(ctx context.Context, tenantID string) ([]*MaskingRule, error)
	
	// Algorithm management
	RegisterAlgorithm(name string, algorithm Algorithm) error
	GetAlgorithm(name string) (Algorithm, error)
	ListAlgorithms() []string
}

// SecurityAuditor defines the interface for security auditing
type SecurityAuditor interface {
	// Access auditing
	AuditDataAccess(ctx context.Context, log *DataAccessLog) error
	GetSecurityEvents(ctx context.Context, tenantID string, from, to time.Time) ([]*SecurityEvent, error)
	
	// Compliance checking
	CheckCompliance(ctx context.Context, tenantID string, standard string) (*ComplianceReport, error)
	GenerateComplianceReport(ctx context.Context, tenantID string, from, to time.Time) (*ComplianceReport, error)
	
	// Security monitoring
	DetectAnomalies(ctx context.Context, tenantID string) ([]*SecurityAnomaly, error)
	SetSecurityAlert(ctx context.Context, alert *SecurityAlert) error
	GetSecurityAlerts(ctx context.Context, tenantID string) ([]*SecurityAlert, error)
}

// SecurityEvent represents a security-related event
type SecurityEvent struct {
	ID        string                 `json:"id"`
	TenantID  string                 `json:"tenant_id"`
	Type      string                 `json:"type"`
	Severity  string                 `json:"severity"`
	Message   string                 `json:"message"`
	Metadata  map[string]interface{} `json:"metadata"`
	Timestamp time.Time              `json:"timestamp"`
	UserID    string                 `json:"user_id,omitempty"`
	IPAddress string                 `json:"ip_address,omitempty"`
}

// ComplianceReport represents a compliance assessment report
type ComplianceReport struct {
	TenantID       string                 `json:"tenant_id"`
	Standard       string                 `json:"standard"`
	Score          float64                `json:"score"`
	Passed         int                    `json:"passed"`
	Failed         int                    `json:"failed"`
	Issues         []ComplianceIssue      `json:"issues"`
	Recommendations []string              `json:"recommendations"`
	GeneratedAt    time.Time              `json:"generated_at"`
	ValidUntil     time.Time              `json:"valid_until"`
}

// ComplianceIssue represents a compliance issue
type ComplianceIssue struct {
	ID          string    `json:"id"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	Resource    string    `json:"resource"`
	Remediation string    `json:"remediation"`
	DetectedAt  time.Time `json:"detected_at"`
}

// SecurityAnomaly represents a detected security anomaly
type SecurityAnomaly struct {
	ID          string                 `json:"id"`
	TenantID    string                 `json:"tenant_id"`
	Type        string                 `json:"type"`
	Confidence  float64                `json:"confidence"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	DetectedAt  time.Time              `json:"detected_at"`
	Status      string                 `json:"status"`
}

// SecurityAlert represents a security alert
type SecurityAlert struct {
	ID        string    `json:"id"`
	TenantID  string    `json:"tenant_id"`
	Type      string    `json:"type"`
	Severity  string    `json:"severity"`
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
	Status    string    `json:"status"`
	Actions   []string  `json:"actions"`
}