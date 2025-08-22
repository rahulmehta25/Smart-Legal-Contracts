package isolation

import (
	"context"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// IsolationLevel defines the level of data isolation
type IsolationLevel string

const (
	DatabaseLevel IsolationLevel = "database"
	SchemaLevel   IsolationLevel = "schema"
	RowLevel      IsolationLevel = "row"
	SharedLevel   IsolationLevel = "shared"
)

// IsolationConfig defines configuration for data isolation
type IsolationConfig struct {
	Level                 IsolationLevel    `json:"level"`
	DatabasePrefix        string            `json:"database_prefix,omitempty"`
	SchemaPrefix          string            `json:"schema_prefix,omitempty"`
	RowLevelColumn        string            `json:"row_level_column,omitempty"`
	ConnectionPoolSize    int               `json:"connection_pool_size"`
	ConnectionTimeout     time.Duration     `json:"connection_timeout"`
	QueryTimeout          time.Duration     `json:"query_timeout"`
	MaxConnections        int               `json:"max_connections"`
	BackupConfig          BackupConfig      `json:"backup_config"`
	MigrationConfig       MigrationConfig   `json:"migration_config"`
	SecurityConfig        SecurityConfig    `json:"security_config"`
}

// BackupConfig defines backup configuration per isolation level
type BackupConfig struct {
	Enabled          bool          `json:"enabled"`
	Schedule         string        `json:"schedule"` // cron format
	RetentionDays    int           `json:"retention_days"`
	CompressionLevel int           `json:"compression_level"`
	EncryptionKey    string        `json:"encryption_key,omitempty"`
	StorageLocation  string        `json:"storage_location"`
	ParallelWorkers  int           `json:"parallel_workers"`
}

// MigrationConfig defines migration configuration
type MigrationConfig struct {
	AutoMigrate      bool     `json:"auto_migrate"`
	MigrationPath    string   `json:"migration_path"`
	VersionTable     string   `json:"version_table"`
	AllowedOperations []string `json:"allowed_operations"`
}

// SecurityConfig defines security configuration for isolation
type SecurityConfig struct {
	EncryptionAtRest     bool              `json:"encryption_at_rest"`
	EncryptionInTransit  bool              `json:"encryption_in_transit"`
	AccessControl        AccessControlConfig `json:"access_control"`
	AuditLogging         bool              `json:"audit_logging"`
	DataMasking          DataMaskingConfig  `json:"data_masking"`
}

// AccessControlConfig defines access control configuration
type AccessControlConfig struct {
	RoleBasedAccess bool              `json:"role_based_access"`
	Roles          []Role            `json:"roles"`
	DefaultRole    string            `json:"default_role"`
	SessionTimeout time.Duration     `json:"session_timeout"`
	IPWhitelist    []string          `json:"ip_whitelist"`
}

// Role defines a security role
type Role struct {
	Name        string       `json:"name"`
	Permissions []Permission `json:"permissions"`
	Resources   []string     `json:"resources"`
}

// Permission defines a permission
type Permission struct {
	Action   string   `json:"action"`   // read, write, delete, admin
	Resource string   `json:"resource"` // table name or resource type
	Columns  []string `json:"columns,omitempty"`
}

// DataMaskingConfig defines data masking configuration
type DataMaskingConfig struct {
	Enabled     bool                  `json:"enabled"`
	Rules       []MaskingRule         `json:"rules"`
	DefaultMask string                `json:"default_mask"`
	Algorithms  map[string]Algorithm  `json:"algorithms"`
}

// MaskingRule defines a data masking rule
type MaskingRule struct {
	Table     string    `json:"table"`
	Column    string    `json:"column"`
	Algorithm string    `json:"algorithm"`
	Condition string    `json:"condition,omitempty"`
}

// Algorithm defines a masking algorithm
type Algorithm struct {
	Type       string            `json:"type"` // hash, encrypt, substitute, partial
	Config     map[string]string `json:"config"`
	PreserveFormat bool          `json:"preserve_format"`
}

// TenantDatabase represents a tenant's database configuration
type TenantDatabase struct {
	TenantID         string            `json:"tenant_id" db:"tenant_id"`
	DatabaseName     string            `json:"database_name" db:"database_name"`
	SchemaName       string            `json:"schema_name" db:"schema_name"`
	ConnectionString string            `json:"connection_string" db:"connection_string"`
	IsolationLevel   IsolationLevel    `json:"isolation_level" db:"isolation_level"`
	Config           IsolationConfig   `json:"config" db:"config"`
	Status           DatabaseStatus    `json:"status" db:"status"`
	CreatedAt        time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at" db:"updated_at"`
	LastBackup       *time.Time        `json:"last_backup" db:"last_backup"`
	BackupSize       int64             `json:"backup_size" db:"backup_size"`
	StorageUsed      int64             `json:"storage_used" db:"storage_used"`
	ConnectionCount  int               `json:"connection_count" db:"connection_count"`
	Health           DatabaseHealth    `json:"health" db:"health"`
}

// DatabaseStatus represents the status of a tenant database
type DatabaseStatus string

const (
	StatusProvisioning DatabaseStatus = "provisioning"
	StatusActive       DatabaseStatus = "active"
	StatusMaintenance  DatabaseStatus = "maintenance"
	StatusSuspended    DatabaseStatus = "suspended"
	StatusMigrating    DatabaseStatus = "migrating"
	StatusError        DatabaseStatus = "error"
)

// DatabaseHealth represents health metrics for a tenant database
type DatabaseHealth struct {
	Status           string    `json:"status"`
	LastHealthCheck  time.Time `json:"last_health_check"`
	ResponseTime     time.Duration `json:"response_time"`
	ConnectionCount  int       `json:"connection_count"`
	ActiveQueries    int       `json:"active_queries"`
	LockWaitTime     time.Duration `json:"lock_wait_time"`
	CacheHitRatio    float64   `json:"cache_hit_ratio"`
	DiskUsagePercent float64   `json:"disk_usage_percent"`
	CPUUsagePercent  float64   `json:"cpu_usage_percent"`
	MemoryUsagePercent float64 `json:"memory_usage_percent"`
	Errors           []string  `json:"errors"`
}

// Value implements driver.Valuer for database storage
func (ic IsolationConfig) Value() (driver.Value, error) {
	return json.Marshal(ic)
}

// Scan implements sql.Scanner for database retrieval
func (ic *IsolationConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, ic)
}

// Value implements driver.Valuer for database storage
func (dh DatabaseHealth) Value() (driver.Value, error) {
	return json.Marshal(dh)
}

// Scan implements sql.Scanner for database retrieval
func (dh *DatabaseHealth) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, dh)
}

// Migration represents a database migration
type Migration struct {
	ID          string         `json:"id" db:"id"`
	TenantID    string         `json:"tenant_id" db:"tenant_id"`
	Version     string         `json:"version" db:"version"`
	Description string         `json:"description" db:"description"`
	Script      string         `json:"script" db:"script"`
	Status      MigrationStatus `json:"status" db:"status"`
	StartedAt   *time.Time     `json:"started_at" db:"started_at"`
	CompletedAt *time.Time     `json:"completed_at" db:"completed_at"`
	Error       string         `json:"error" db:"error"`
	CreatedAt   time.Time      `json:"created_at" db:"created_at"`
	Checksum    string         `json:"checksum" db:"checksum"`
}

// MigrationStatus represents the status of a migration
type MigrationStatus string

const (
	MigrationPending   MigrationStatus = "pending"
	MigrationRunning   MigrationStatus = "running"
	MigrationCompleted MigrationStatus = "completed"
	MigrationFailed    MigrationStatus = "failed"
	MigrationRolledBack MigrationStatus = "rolled_back"
)

// ConnectionPool represents a database connection pool for a tenant
type ConnectionPool struct {
	TenantID        string            `json:"tenant_id"`
	DatabaseName    string            `json:"database_name"`
	MaxConnections  int               `json:"max_connections"`
	ActiveConnections int             `json:"active_connections"`
	IdleConnections int               `json:"idle_connections"`
	WaitingQueries  int               `json:"waiting_queries"`
	AverageWaitTime time.Duration     `json:"average_wait_time"`
	LastActivity    time.Time         `json:"last_activity"`
	HealthStatus    string            `json:"health_status"`
}

// QueryPerformance tracks query performance metrics
type QueryPerformance struct {
	TenantID       string        `json:"tenant_id" db:"tenant_id"`
	QueryHash      string        `json:"query_hash" db:"query_hash"`
	QueryTemplate  string        `json:"query_template" db:"query_template"`
	ExecutionTime  time.Duration `json:"execution_time" db:"execution_time"`
	RowsExamined   int64         `json:"rows_examined" db:"rows_examined"`
	RowsReturned   int64         `json:"rows_returned" db:"rows_returned"`
	IndexesUsed    []string      `json:"indexes_used" db:"indexes_used"`
	LockTime       time.Duration `json:"lock_time" db:"lock_time"`
	Timestamp      time.Time     `json:"timestamp" db:"timestamp"`
	ExecutionPlan  string        `json:"execution_plan" db:"execution_plan"`
}

// IsolationMetrics tracks isolation performance and health
type IsolationMetrics struct {
	TenantID              string        `json:"tenant_id" db:"tenant_id"`
	Timestamp             time.Time     `json:"timestamp" db:"timestamp"`
	DatabaseConnections   int           `json:"database_connections" db:"database_connections"`
	ActiveQueries         int           `json:"active_queries" db:"active_queries"`
	AverageQueryTime      time.Duration `json:"average_query_time" db:"average_query_time"`
	QueriesPerSecond      float64       `json:"queries_per_second" db:"queries_per_second"`
	CacheHitRatio         float64       `json:"cache_hit_ratio" db:"cache_hit_ratio"`
	StorageUsedBytes      int64         `json:"storage_used_bytes" db:"storage_used_bytes"`
	IndexesUsed           int           `json:"indexes_used" db:"indexes_used"`
	LockWaitTime          time.Duration `json:"lock_wait_time" db:"lock_wait_time"`
	BackgroundTasksCount  int           `json:"background_tasks_count" db:"background_tasks_count"`
	ReplicationLag        time.Duration `json:"replication_lag" db:"replication_lag"`
}

// DataAccessLog tracks data access for auditing
type DataAccessLog struct {
	ID          string    `json:"id" db:"id"`
	TenantID    string    `json:"tenant_id" db:"tenant_id"`
	UserID      string    `json:"user_id" db:"user_id"`
	Action      string    `json:"action" db:"action"` // SELECT, INSERT, UPDATE, DELETE
	Table       string    `json:"table" db:"table"`
	RowsAffected int64    `json:"rows_affected" db:"rows_affected"`
	Query       string    `json:"query" db:"query"`
	IPAddress   string    `json:"ip_address" db:"ip_address"`
	UserAgent   string    `json:"user_agent" db:"user_agent"`
	Timestamp   time.Time `json:"timestamp" db:"timestamp"`
	Duration    time.Duration `json:"duration" db:"duration"`
	Success     bool      `json:"success" db:"success"`
	Error       string    `json:"error" db:"error"`
}

// TenantDataSummary provides a summary of tenant data
type TenantDataSummary struct {
	TenantID         string    `json:"tenant_id"`
	DatabaseName     string    `json:"database_name"`
	SchemaName       string    `json:"schema_name"`
	TableCount       int       `json:"table_count"`
	TotalRows        int64     `json:"total_rows"`
	StorageUsedMB    int64     `json:"storage_used_mb"`
	IndexSizeMB      int64     `json:"index_size_mb"`
	LastBackup       *time.Time `json:"last_backup"`
	BackupSizeMB     int64     `json:"backup_size_mb"`
	CreatedAt        time.Time `json:"created_at"`
	LastModified     time.Time `json:"last_modified"`
	ConnectionsUsed  int       `json:"connections_used"`
	QueriesPerMinute float64   `json:"queries_per_minute"`
}

// GetConnectionString builds a connection string for the tenant database
func (td *TenantDatabase) GetConnectionString() string {
	if td.ConnectionString != "" {
		return td.ConnectionString
	}
	
	// Build connection string based on isolation level
	switch td.IsolationLevel {
	case DatabaseLevel:
		return fmt.Sprintf("dbname=%s", td.DatabaseName)
	case SchemaLevel:
		return fmt.Sprintf("search_path=%s", td.SchemaName)
	default:
		return ""
	}
}

// IsHealthy checks if the tenant database is healthy
func (td *TenantDatabase) IsHealthy() bool {
	return td.Status == StatusActive && 
		   td.Health.Status == "healthy" && 
		   len(td.Health.Errors) == 0
}

// NeedsBackup checks if the tenant database needs a backup
func (td *TenantDatabase) NeedsBackup() bool {
	if !td.Config.BackupConfig.Enabled {
		return false
	}
	
	if td.LastBackup == nil {
		return true
	}
	
	retentionDuration := time.Duration(td.Config.BackupConfig.RetentionDays) * 24 * time.Hour
	return time.Since(*td.LastBackup) > retentionDuration
}

// GetDiskUsagePercent calculates disk usage percentage
func (td *TenantDatabase) GetDiskUsagePercent() float64 {
	return td.Health.DiskUsagePercent
}

// GetStorageUsedMB returns storage used in MB
func (td *TenantDatabase) GetStorageUsedMB() int64 {
	return td.StorageUsed / (1024 * 1024)
}