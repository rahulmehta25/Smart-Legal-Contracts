package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TenantSpec defines the desired state of Tenant
type TenantSpec struct {
	// TenantID is the unique identifier for the tenant
	TenantID string `json:"tenantId"`

	// Name is the display name of the tenant
	Name string `json:"name"`

	// Plan defines the billing plan for the tenant
	// +kubebuilder:validation:Enum=trial;starter;professional;enterprise
	Plan string `json:"plan"`

	// IsolationStrategy defines how tenant data is isolated
	// +kubebuilder:validation:Enum=database_per_tenant;schema_per_tenant;row_level;shared
	IsolationStrategy string `json:"isolationStrategy"`

	// AdminContact contains administrator contact information
	AdminContact AdminContact `json:"adminContact"`

	// ResourceQuota defines resource limits for the tenant
	ResourceQuota ResourceQuota `json:"resourceQuota,omitempty"`

	// DatabaseConfig contains database-specific configuration
	DatabaseConfig DatabaseConfig `json:"databaseConfig,omitempty"`

	// CustomDomains is a list of custom domains for the tenant
	CustomDomains []string `json:"customDomains,omitempty"`

	// FeatureFlags contains enabled features for the tenant
	FeatureFlags []string `json:"featureFlags,omitempty"`

	// BackupConfig defines backup configuration
	BackupConfig BackupConfig `json:"backupConfig,omitempty"`

	// MonitoringConfig defines monitoring configuration
	MonitoringConfig MonitoringConfig `json:"monitoringConfig,omitempty"`

	// SecurityConfig defines security configuration
	SecurityConfig SecurityConfig `json:"securityConfig,omitempty"`

	// AutoscalingConfig defines autoscaling configuration
	AutoscalingConfig AutoscalingConfig `json:"autoscalingConfig,omitempty"`

	// ComplianceConfig defines compliance requirements
	ComplianceConfig ComplianceConfig `json:"complianceConfig,omitempty"`
}

// AdminContact contains administrator contact information
type AdminContact struct {
	Email string `json:"email"`
	Name  string `json:"name"`
	Phone string `json:"phone,omitempty"`
}

// ResourceQuota defines resource limits
type ResourceQuota struct {
	// CPU request limit
	CPURequest string `json:"cpuRequest,omitempty"`
	// CPU limit
	CPULimit string `json:"cpuLimit,omitempty"`
	// Memory request limit
	MemoryRequest string `json:"memoryRequest,omitempty"`
	// Memory limit
	MemoryLimit string `json:"memoryLimit,omitempty"`
	// Storage request limit
	StorageRequest string `json:"storageRequest,omitempty"`
	// Maximum number of persistent volume claims
	PVCCount int32 `json:"pvcCount,omitempty"`
	// Maximum number of services
	ServiceCount int32 `json:"serviceCount,omitempty"`
	// Maximum number of deployments
	DeploymentCount int32 `json:"deploymentCount,omitempty"`
}

// DatabaseConfig contains database configuration
type DatabaseConfig struct {
	// Engine version
	EngineVersion string `json:"engineVersion,omitempty"`
	// Instance class
	InstanceClass string `json:"instanceClass,omitempty"`
	// Allocated storage in GB
	AllocatedStorage int32 `json:"allocatedStorage,omitempty"`
	// Maximum allocated storage in GB
	MaxAllocatedStorage int32 `json:"maxAllocatedStorage,omitempty"`
	// Whether storage is encrypted
	StorageEncrypted bool `json:"storageEncrypted,omitempty"`
	// Backup retention period in days
	BackupRetentionPeriod int32 `json:"backupRetentionPeriod,omitempty"`
	// Whether multi-AZ is enabled
	MultiAZ bool `json:"multiAz,omitempty"`
	// Whether performance insights is enabled
	PerformanceInsights bool `json:"performanceInsights,omitempty"`
}

// BackupConfig defines backup configuration
type BackupConfig struct {
	// Whether automated backups are enabled
	Enabled bool `json:"enabled"`
	// Backup schedule in cron format
	Schedule string `json:"schedule,omitempty"`
	// Retention period in days
	RetentionDays int32 `json:"retentionDays,omitempty"`
	// Whether cross-region backup is enabled
	CrossRegionBackup bool `json:"crossRegionBackup,omitempty"`
	// Whether point-in-time recovery is enabled
	PointInTimeRecovery bool `json:"pointInTimeRecovery,omitempty"`
}

// MonitoringConfig defines monitoring configuration
type MonitoringConfig struct {
	// Whether monitoring is enabled
	Enabled bool `json:"enabled"`
	// Whether Prometheus metrics are enabled
	Prometheus bool `json:"prometheus,omitempty"`
	// Whether Grafana dashboards are enabled
	Grafana bool `json:"grafana,omitempty"`
	// Whether distributed tracing is enabled
	Tracing bool `json:"tracing,omitempty"`
	// Log level
	LogLevel string `json:"logLevel,omitempty"`
	// Metrics retention period
	MetricsRetention string `json:"metricsRetention,omitempty"`
}

// SecurityConfig defines security configuration
type SecurityConfig struct {
	// Whether network policies are enabled
	NetworkPoliciesEnabled bool `json:"networkPoliciesEnabled,omitempty"`
	// Pod security standard
	PodSecurityStandard string `json:"podSecurityStandard,omitempty"`
	// Whether RBAC is enabled
	RBACEnabled bool `json:"rbacEnabled,omitempty"`
	// Whether service mesh is enabled
	ServiceMeshEnabled bool `json:"serviceMeshEnabled,omitempty"`
	// Whether mTLS is enabled
	MTLSEnabled bool `json:"mtlsEnabled,omitempty"`
}

// AutoscalingConfig defines autoscaling configuration
type AutoscalingConfig struct {
	// Whether horizontal pod autoscaler is enabled
	Enabled bool `json:"enabled"`
	// Minimum number of replicas
	MinReplicas int32 `json:"minReplicas,omitempty"`
	// Maximum number of replicas
	MaxReplicas int32 `json:"maxReplicas,omitempty"`
	// Target CPU utilization percentage
	TargetCPUPercent int32 `json:"targetCpuPercent,omitempty"`
	// Target memory utilization percentage
	TargetMemoryPercent int32 `json:"targetMemoryPercent,omitempty"`
}

// ComplianceConfig defines compliance requirements
type ComplianceConfig struct {
	// Whether GDPR compliance is required
	GDPREnabled bool `json:"gdprEnabled,omitempty"`
	// Whether HIPAA compliance is required
	HIPAAEnabled bool `json:"hipaaEnabled,omitempty"`
	// Whether SOX compliance is required
	SOXEnabled bool `json:"soxEnabled,omitempty"`
	// Whether audit logging is enabled
	AuditLogging bool `json:"auditLogging,omitempty"`
	// Data retention period in days
	DataRetentionDays int32 `json:"dataRetentionDays,omitempty"`
}

// TenantStatus defines the observed state of Tenant
type TenantStatus struct {
	// Phase represents the current phase of tenant provisioning
	// +kubebuilder:validation:Enum=Pending;Provisioning;Active;Suspended;Deleting;Failed
	Phase TenantPhase `json:"phase,omitempty"`

	// Message provides additional information about the current status
	Message string `json:"message,omitempty"`

	// Conditions represents the latest available observations of tenant state
	Conditions []TenantCondition `json:"conditions,omitempty"`

	// DatabaseStatus contains database-specific status information
	DatabaseStatus DatabaseStatus `json:"databaseStatus,omitempty"`

	// StorageStatus contains storage-specific status information
	StorageStatus StorageStatus `json:"storageStatus,omitempty"`

	// NetworkingStatus contains networking-specific status information
	NetworkingStatus NetworkingStatus `json:"networkingStatus,omitempty"`

	// LastReconcileTime is the last time the tenant was reconciled
	LastReconcileTime *metav1.Time `json:"lastReconcileTime,omitempty"`

	// ObservedGeneration is the generation observed by the controller
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Resources contains references to created resources
	Resources TenantResources `json:"resources,omitempty"`

	// Usage contains current resource usage information
	Usage ResourceUsage `json:"usage,omitempty"`
}

// TenantPhase represents the current phase of tenant provisioning
type TenantPhase string

const (
	// TenantPhasePending means the tenant is waiting to be processed
	TenantPhasePending TenantPhase = "Pending"
	// TenantPhaseProvisioning means the tenant infrastructure is being created
	TenantPhaseProvisioning TenantPhase = "Provisioning"
	// TenantPhaseActive means the tenant is fully operational
	TenantPhaseActive TenantPhase = "Active"
	// TenantPhaseSuspended means the tenant has been suspended
	TenantPhaseSuspended TenantPhase = "Suspended"
	// TenantPhaseDeleting means the tenant is being deleted
	TenantPhaseDeleting TenantPhase = "Deleting"
	// TenantPhaseFailed means tenant provisioning has failed
	TenantPhaseFailed TenantPhase = "Failed"
)

// TenantCondition represents a condition of a Tenant
type TenantCondition struct {
	// Type of tenant condition
	Type TenantConditionType `json:"type"`
	// Status of the condition
	Status corev1.ConditionStatus `json:"status"`
	// Last time the condition transitioned from one status to another
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`
	// Unique, one-word, CamelCase reason for the condition's last transition
	Reason string `json:"reason,omitempty"`
	// Human-readable message indicating details about last transition
	Message string `json:"message,omitempty"`
}

// TenantConditionType represents the type of tenant condition
type TenantConditionType string

const (
	// TenantConditionReady indicates that the tenant is ready and fully operational
	TenantConditionReady TenantConditionType = "Ready"
	// TenantConditionDatabaseReady indicates that the database is ready
	TenantConditionDatabaseReady TenantConditionType = "DatabaseReady"
	// TenantConditionStorageReady indicates that storage is ready
	TenantConditionStorageReady TenantConditionType = "StorageReady"
	// TenantConditionNetworkingReady indicates that networking is ready
	TenantConditionNetworkingReady TenantConditionType = "NetworkingReady"
	// TenantConditionMonitoringReady indicates that monitoring is ready
	TenantConditionMonitoringReady TenantConditionType = "MonitoringReady"
	// TenantConditionBackupReady indicates that backup is ready
	TenantConditionBackupReady TenantConditionType = "BackupReady"
)

// DatabaseStatus contains database-specific status information
type DatabaseStatus struct {
	// Database endpoint
	Endpoint string `json:"endpoint,omitempty"`
	// Database port
	Port int32 `json:"port,omitempty"`
	// Database name
	DatabaseName string `json:"databaseName,omitempty"`
	// Database status
	Status string `json:"status,omitempty"`
	// Storage used in bytes
	StorageUsed int64 `json:"storageUsed,omitempty"`
	// Connection count
	ConnectionCount int32 `json:"connectionCount,omitempty"`
	// Last backup time
	LastBackupTime *metav1.Time `json:"lastBackupTime,omitempty"`
}

// StorageStatus contains storage-specific status information
type StorageStatus struct {
	// S3 bucket name
	BucketName string `json:"bucketName,omitempty"`
	// Storage used in bytes
	StorageUsed int64 `json:"storageUsed,omitempty"`
	// Object count
	ObjectCount int64 `json:"objectCount,omitempty"`
	// CloudFront distribution ID
	DistributionID string `json:"distributionId,omitempty"`
}

// NetworkingStatus contains networking-specific status information
type NetworkingStatus struct {
	// Load balancer hostname
	LoadBalancerHostname string `json:"loadBalancerHostname,omitempty"`
	// Load balancer IP
	LoadBalancerIP string `json:"loadBalancerIp,omitempty"`
	// Custom domain status
	CustomDomainStatus []CustomDomainStatus `json:"customDomainStatus,omitempty"`
}

// CustomDomainStatus represents the status of a custom domain
type CustomDomainStatus struct {
	// Domain name
	Domain string `json:"domain"`
	// Status of the domain
	Status string `json:"status"`
	// SSL certificate ARN
	CertificateARN string `json:"certificateArn,omitempty"`
	// DNS validation status
	DNSValidationStatus string `json:"dnsValidationStatus,omitempty"`
}

// TenantResources contains references to created resources
type TenantResources struct {
	// Kubernetes namespace
	Namespace string `json:"namespace,omitempty"`
	// Service account name
	ServiceAccount string `json:"serviceAccount,omitempty"`
	// ConfigMap name
	ConfigMap string `json:"configMap,omitempty"`
	// Secret name
	Secret string `json:"secret,omitempty"`
	// Service name
	Service string `json:"service,omitempty"`
	// Ingress name
	Ingress string `json:"ingress,omitempty"`
	// HPA name
	HorizontalPodAutoscaler string `json:"horizontalPodAutoscaler,omitempty"`
	// Network policy name
	NetworkPolicy string `json:"networkPolicy,omitempty"`
}

// ResourceUsage contains current resource usage information
type ResourceUsage struct {
	// CPU usage in millicores
	CPUUsage int64 `json:"cpuUsage,omitempty"`
	// Memory usage in bytes
	MemoryUsage int64 `json:"memoryUsage,omitempty"`
	// Storage usage in bytes
	StorageUsage int64 `json:"storageUsage,omitempty"`
	// Network usage in bytes
	NetworkUsage int64 `json:"networkUsage,omitempty"`
	// API request count
	APIRequestCount int64 `json:"apiRequestCount,omitempty"`
	// Active user count
	ActiveUserCount int64 `json:"activeUserCount,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Plan",type="string",JSONPath=".spec.plan"
// +kubebuilder:printcolumn:name="Isolation",type="string",JSONPath=".spec.isolationStrategy"
// +kubebuilder:printcolumn:name="Database",type="string",JSONPath=".status.databaseStatus.status"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// Tenant is the Schema for the tenants API
type Tenant struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TenantSpec   `json:"spec,omitempty"`
	Status TenantStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TenantList contains a list of Tenant
type TenantList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Tenant `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Tenant{}, &TenantList{})
}