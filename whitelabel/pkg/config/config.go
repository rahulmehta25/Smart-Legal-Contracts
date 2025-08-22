package config

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
)

// Config holds all configuration for the white-label platform
type Config struct {
	Server     ServerConfig     `mapstructure:"server"`
	Database   DatabaseConfig   `mapstructure:"database"`
	Redis      RedisConfig      `mapstructure:"redis"`
	Security   SecurityConfig   `mapstructure:"security"`
	Monitoring MonitoringConfig `mapstructure:"monitoring"`
	Storage    StorageConfig    `mapstructure:"storage"`
	Email      EmailConfig      `mapstructure:"email"`
	Features   FeatureConfig    `mapstructure:"features"`
}

// ServerConfig contains server configuration
type ServerConfig struct {
	Address         string        `mapstructure:"address"`
	Port            int           `mapstructure:"port"`
	ReadTimeout     time.Duration `mapstructure:"read_timeout"`
	WriteTimeout    time.Duration `mapstructure:"write_timeout"`
	ShutdownTimeout time.Duration `mapstructure:"shutdown_timeout"`
	MaxHeaderBytes  int           `mapstructure:"max_header_bytes"`
	TLS             TLSConfig     `mapstructure:"tls"`
	CORS            CORSConfig    `mapstructure:"cors"`
	RateLimit       RateLimitConfig `mapstructure:"rate_limit"`
}

// TLSConfig contains TLS configuration
type TLSConfig struct {
	Enabled  bool   `mapstructure:"enabled"`
	CertFile string `mapstructure:"cert_file"`
	KeyFile  string `mapstructure:"key_file"`
}

// CORSConfig contains CORS configuration
type CORSConfig struct {
	Enabled        bool     `mapstructure:"enabled"`
	AllowedOrigins []string `mapstructure:"allowed_origins"`
	AllowedMethods []string `mapstructure:"allowed_methods"`
	AllowedHeaders []string `mapstructure:"allowed_headers"`
	ExposedHeaders []string `mapstructure:"exposed_headers"`
	MaxAge         int      `mapstructure:"max_age"`
}

// RateLimitConfig contains rate limiting configuration
type RateLimitConfig struct {
	Enabled       bool          `mapstructure:"enabled"`
	RequestsPerHour int          `mapstructure:"requests_per_hour"`
	BurstLimit    int           `mapstructure:"burst_limit"`
	WindowSize    time.Duration `mapstructure:"window_size"`
}

// DatabaseConfig contains database configuration
type DatabaseConfig struct {
	Host               string        `mapstructure:"host"`
	Port               int           `mapstructure:"port"`
	Name               string        `mapstructure:"name"`
	User               string        `mapstructure:"user"`
	Password           string        `mapstructure:"password"`
	SSLMode            string        `mapstructure:"ssl_mode"`
	MaxOpenConns       int           `mapstructure:"max_open_conns"`
	MaxIdleConns       int           `mapstructure:"max_idle_conns"`
	ConnMaxLifetime    time.Duration `mapstructure:"conn_max_lifetime"`
	ConnMaxIdleTime    time.Duration `mapstructure:"conn_max_idle_time"`
	MigrationsPath     string        `mapstructure:"migrations_path"`
	AutoMigrate        bool          `mapstructure:"auto_migrate"`
}

// RedisConfig contains Redis configuration
type RedisConfig struct {
	Host         string        `mapstructure:"host"`
	Port         int           `mapstructure:"port"`
	Password     string        `mapstructure:"password"`
	Database     int           `mapstructure:"database"`
	MaxRetries   int           `mapstructure:"max_retries"`
	PoolSize     int           `mapstructure:"pool_size"`
	MinIdleConns int           `mapstructure:"min_idle_conns"`
	PoolTimeout  time.Duration `mapstructure:"pool_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
	TLS          bool          `mapstructure:"tls"`
}

// SecurityConfig contains security configuration
type SecurityConfig struct {
	JWTSecret           string        `mapstructure:"jwt_secret"`
	JWTExpiration       time.Duration `mapstructure:"jwt_expiration"`
	RefreshTokenExpiration time.Duration `mapstructure:"refresh_token_expiration"`
	EncryptionKey       string        `mapstructure:"encryption_key"`
	PasswordMinLength   int           `mapstructure:"password_min_length"`
	SessionTimeout      time.Duration `mapstructure:"session_timeout"`
	MaxLoginAttempts    int           `mapstructure:"max_login_attempts"`
	LockoutDuration     time.Duration `mapstructure:"lockout_duration"`
	RequireHTTPS        bool          `mapstructure:"require_https"`
	CSRFProtection      bool          `mapstructure:"csrf_protection"`
	ContentTypeNosniff  bool          `mapstructure:"content_type_nosniff"`
	FrameOptions        string        `mapstructure:"frame_options"`
	HSTSMaxAge          int           `mapstructure:"hsts_max_age"`
}

// MonitoringConfig contains monitoring configuration
type MonitoringConfig struct {
	Enabled           bool          `mapstructure:"enabled"`
	MetricsEnabled    bool          `mapstructure:"metrics_enabled"`
	TracingEnabled    bool          `mapstructure:"tracing_enabled"`
	LogLevel          string        `mapstructure:"log_level"`
	MetricsPath       string        `mapstructure:"metrics_path"`
	HealthPath        string        `mapstructure:"health_path"`
	PrometheusEnabled bool          `mapstructure:"prometheus_enabled"`
	JaegerEnabled     bool          `mapstructure:"jaeger_enabled"`
	JaegerEndpoint    string        `mapstructure:"jaeger_endpoint"`
	LogFormat         string        `mapstructure:"log_format"`
	MetricsRetention  time.Duration `mapstructure:"metrics_retention"`
}

// StorageConfig contains storage configuration
type StorageConfig struct {
	Type               string `mapstructure:"type"` // local, s3, gcs, azure
	LocalPath          string `mapstructure:"local_path"`
	S3Config           S3Config `mapstructure:"s3"`
	MaxFileSize        int64  `mapstructure:"max_file_size"`
	AllowedExtensions  []string `mapstructure:"allowed_extensions"`
	CDNEnabled         bool   `mapstructure:"cdn_enabled"`
	CDNDomain          string `mapstructure:"cdn_domain"`
}

// S3Config contains S3 storage configuration
type S3Config struct {
	Region          string `mapstructure:"region"`
	Bucket          string `mapstructure:"bucket"`
	AccessKeyID     string `mapstructure:"access_key_id"`
	SecretAccessKey string `mapstructure:"secret_access_key"`
	Endpoint        string `mapstructure:"endpoint"`
	UseSSL          bool   `mapstructure:"use_ssl"`
	PathStyle       bool   `mapstructure:"path_style"`
}

// EmailConfig contains email configuration
type EmailConfig struct {
	Enabled    bool       `mapstructure:"enabled"`
	Provider   string     `mapstructure:"provider"` // smtp, sendgrid, ses
	SMTPConfig SMTPConfig `mapstructure:"smtp"`
	From       string     `mapstructure:"from"`
	FromName   string     `mapstructure:"from_name"`
	ReplyTo    string     `mapstructure:"reply_to"`
}

// SMTPConfig contains SMTP configuration
type SMTPConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Username string `mapstructure:"username"`
	Password string `mapstructure:"password"`
	UseTLS   bool   `mapstructure:"use_tls"`
}

// FeatureConfig contains feature flags configuration
type FeatureConfig struct {
	MultiTenancy       bool `mapstructure:"multi_tenancy"`
	Customization      bool `mapstructure:"customization"`
	AdvancedAnalytics  bool `mapstructure:"advanced_analytics"`
	BackupRestore      bool `mapstructure:"backup_restore"`
	APIVersioning      bool `mapstructure:"api_versioning"`
	WebhookSupport     bool `mapstructure:"webhook_support"`
	AuditLogging       bool `mapstructure:"audit_logging"`
	ComplianceMode     bool `mapstructure:"compliance_mode"`
	DataEncryption     bool `mapstructure:"data_encryption"`
	SSO                bool `mapstructure:"sso"`
}

// Load loads configuration from file and environment variables
func Load(configPath string) (*Config, error) {
	viper.SetConfigFile(configPath)
	viper.SetConfigType("yaml")
	
	// Set defaults
	setDefaults()
	
	// Read config file
	if err := viper.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	
	// Override with environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("WHITELABEL")
	
	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}
	
	// Validate configuration
	if err := validate(&config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}
	
	return &config, nil
}

// setDefaults sets default configuration values
func setDefaults() {
	// Server defaults
	viper.SetDefault("server.address", "0.0.0.0")
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.read_timeout", "30s")
	viper.SetDefault("server.write_timeout", "30s")
	viper.SetDefault("server.shutdown_timeout", "10s")
	viper.SetDefault("server.max_header_bytes", 1048576) // 1MB
	
	// TLS defaults
	viper.SetDefault("server.tls.enabled", false)
	
	// CORS defaults
	viper.SetDefault("server.cors.enabled", true)
	viper.SetDefault("server.cors.allowed_origins", []string{"*"})
	viper.SetDefault("server.cors.allowed_methods", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"})
	viper.SetDefault("server.cors.allowed_headers", []string{"*"})
	viper.SetDefault("server.cors.max_age", 86400)
	
	// Rate limit defaults
	viper.SetDefault("server.rate_limit.enabled", true)
	viper.SetDefault("server.rate_limit.requests_per_hour", 1000)
	viper.SetDefault("server.rate_limit.burst_limit", 100)
	viper.SetDefault("server.rate_limit.window_size", "1h")
	
	// Database defaults
	viper.SetDefault("database.host", "localhost")
	viper.SetDefault("database.port", 5432)
	viper.SetDefault("database.name", "whitelabel")
	viper.SetDefault("database.user", "postgres")
	viper.SetDefault("database.ssl_mode", "disable")
	viper.SetDefault("database.max_open_conns", 25)
	viper.SetDefault("database.max_idle_conns", 5)
	viper.SetDefault("database.conn_max_lifetime", "1h")
	viper.SetDefault("database.conn_max_idle_time", "10m")
	viper.SetDefault("database.migrations_path", "migrations")
	viper.SetDefault("database.auto_migrate", false)
	
	// Redis defaults
	viper.SetDefault("redis.host", "localhost")
	viper.SetDefault("redis.port", 6379)
	viper.SetDefault("redis.database", 0)
	viper.SetDefault("redis.max_retries", 3)
	viper.SetDefault("redis.pool_size", 10)
	viper.SetDefault("redis.min_idle_conns", 2)
	viper.SetDefault("redis.pool_timeout", "4s")
	viper.SetDefault("redis.idle_timeout", "5m")
	viper.SetDefault("redis.tls", false)
	
	// Security defaults
	viper.SetDefault("security.jwt_expiration", "24h")
	viper.SetDefault("security.refresh_token_expiration", "168h") // 7 days
	viper.SetDefault("security.password_min_length", 8)
	viper.SetDefault("security.session_timeout", "24h")
	viper.SetDefault("security.max_login_attempts", 5)
	viper.SetDefault("security.lockout_duration", "15m")
	viper.SetDefault("security.require_https", false)
	viper.SetDefault("security.csrf_protection", true)
	viper.SetDefault("security.content_type_nosniff", true)
	viper.SetDefault("security.frame_options", "DENY")
	viper.SetDefault("security.hsts_max_age", 31536000) // 1 year
	
	// Monitoring defaults
	viper.SetDefault("monitoring.enabled", true)
	viper.SetDefault("monitoring.metrics_enabled", true)
	viper.SetDefault("monitoring.tracing_enabled", false)
	viper.SetDefault("monitoring.log_level", "info")
	viper.SetDefault("monitoring.metrics_path", "/metrics")
	viper.SetDefault("monitoring.health_path", "/health")
	viper.SetDefault("monitoring.prometheus_enabled", true)
	viper.SetDefault("monitoring.jaeger_enabled", false)
	viper.SetDefault("monitoring.log_format", "json")
	viper.SetDefault("monitoring.metrics_retention", "15d")
	
	// Storage defaults
	viper.SetDefault("storage.type", "local")
	viper.SetDefault("storage.local_path", "./storage")
	viper.SetDefault("storage.max_file_size", 10485760) // 10MB
	viper.SetDefault("storage.allowed_extensions", []string{".jpg", ".jpeg", ".png", ".gif", ".pdf", ".doc", ".docx"})
	viper.SetDefault("storage.cdn_enabled", false)
	
	// S3 defaults
	viper.SetDefault("storage.s3.use_ssl", true)
	viper.SetDefault("storage.s3.path_style", false)
	
	// Email defaults
	viper.SetDefault("email.enabled", false)
	viper.SetDefault("email.provider", "smtp")
	viper.SetDefault("email.smtp.port", 587)
	viper.SetDefault("email.smtp.use_tls", true)
	
	// Feature defaults
	viper.SetDefault("features.multi_tenancy", true)
	viper.SetDefault("features.customization", true)
	viper.SetDefault("features.advanced_analytics", false)
	viper.SetDefault("features.backup_restore", false)
	viper.SetDefault("features.api_versioning", true)
	viper.SetDefault("features.webhook_support", true)
	viper.SetDefault("features.audit_logging", true)
	viper.SetDefault("features.compliance_mode", false)
	viper.SetDefault("features.data_encryption", true)
	viper.SetDefault("features.sso", false)
}

// validate validates the configuration
func validate(config *Config) error {
	// Validate server configuration
	if config.Server.Port <= 0 || config.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", config.Server.Port)
	}
	
	// Validate database configuration
	if config.Database.Host == "" {
		return fmt.Errorf("database host is required")
	}
	if config.Database.Port <= 0 || config.Database.Port > 65535 {
		return fmt.Errorf("invalid database port: %d", config.Database.Port)
	}
	if config.Database.Name == "" {
		return fmt.Errorf("database name is required")
	}
	if config.Database.User == "" {
		return fmt.Errorf("database user is required")
	}
	
	// Validate Redis configuration
	if config.Redis.Host == "" {
		return fmt.Errorf("redis host is required")
	}
	if config.Redis.Port <= 0 || config.Redis.Port > 65535 {
		return fmt.Errorf("invalid redis port: %d", config.Redis.Port)
	}
	
	// Validate security configuration
	if config.Security.JWTSecret == "" {
		return fmt.Errorf("JWT secret is required")
	}
	if len(config.Security.JWTSecret) < 32 {
		return fmt.Errorf("JWT secret must be at least 32 characters long")
	}
	if config.Security.EncryptionKey != "" && len(config.Security.EncryptionKey) != 32 {
		return fmt.Errorf("encryption key must be exactly 32 characters long")
	}
	
	// Validate TLS configuration
	if config.Server.TLS.Enabled {
		if config.Server.TLS.CertFile == "" {
			return fmt.Errorf("TLS cert file is required when TLS is enabled")
		}
		if config.Server.TLS.KeyFile == "" {
			return fmt.Errorf("TLS key file is required when TLS is enabled")
		}
	}
	
	// Validate storage configuration
	if config.Storage.Type == "" {
		return fmt.Errorf("storage type is required")
	}
	if config.Storage.Type == "s3" {
		if config.Storage.S3Config.Region == "" {
			return fmt.Errorf("S3 region is required when using S3 storage")
		}
		if config.Storage.S3Config.Bucket == "" {
			return fmt.Errorf("S3 bucket is required when using S3 storage")
		}
	}
	
	// Validate email configuration
	if config.Email.Enabled {
		if config.Email.From == "" {
			return fmt.Errorf("email from address is required when email is enabled")
		}
		if config.Email.Provider == "smtp" {
			if config.Email.SMTPConfig.Host == "" {
				return fmt.Errorf("SMTP host is required when using SMTP email provider")
			}
		}
	}
	
	// Validate monitoring configuration
	if config.Monitoring.LogLevel != "" {
		validLevels := []string{"debug", "info", "warn", "error", "fatal"}
		valid := false
		for _, level := range validLevels {
			if config.Monitoring.LogLevel == level {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("invalid log level: %s", config.Monitoring.LogLevel)
		}
	}
	
	return nil
}

// GetDatabaseURL returns the database connection URL
func (c *Config) GetDatabaseURL() string {
	return fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=%s",
		c.Database.User,
		c.Database.Password,
		c.Database.Host,
		c.Database.Port,
		c.Database.Name,
		c.Database.SSLMode,
	)
}

// GetRedisURL returns the Redis connection URL
func (c *Config) GetRedisURL() string {
	if c.Redis.Password != "" {
		return fmt.Sprintf("redis://:%s@%s:%d/%d",
			c.Redis.Password,
			c.Redis.Host,
			c.Redis.Port,
			c.Redis.Database,
		)
	}
	return fmt.Sprintf("redis://%s:%d/%d",
		c.Redis.Host,
		c.Redis.Port,
		c.Redis.Database,
	)
}

// IsProduction returns true if running in production mode
func (c *Config) IsProduction() bool {
	return c.Monitoring.LogLevel == "warn" || c.Monitoring.LogLevel == "error"
}

// IsDevelopment returns true if running in development mode
func (c *Config) IsDevelopment() bool {
	return c.Monitoring.LogLevel == "debug"
}