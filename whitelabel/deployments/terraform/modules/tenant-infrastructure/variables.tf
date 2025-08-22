# Tenant Infrastructure Module Variables

variable "tenant_id" {
  description = "Unique identifier for the tenant"
  type        = string
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.tenant_id))
    error_message = "Tenant ID must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "tenant_name" {
  description = "Display name for the tenant"
  type        = string
}

variable "tenant_plan" {
  description = "Pricing plan for the tenant"
  type        = string
  default     = "starter"
  validation {
    condition     = contains(["trial", "starter", "professional", "enterprise"], var.tenant_plan)
    error_message = "Tenant plan must be one of: trial, starter, professional, enterprise."
  }
}

variable "isolation_strategy" {
  description = "Data isolation strategy for the tenant"
  type        = string
  default     = "database_per_tenant"
  validation {
    condition     = contains(["database_per_tenant", "schema_per_tenant", "row_level", "shared"], var.isolation_strategy)
    error_message = "Isolation strategy must be one of: database_per_tenant, schema_per_tenant, row_level, shared."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# Infrastructure Configuration
variable "vpc_id" {
  description = "VPC ID for the tenant infrastructure"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for database resources"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for load balancers"
  type        = list(string)
  default     = []
}

variable "database_security_group_ids" {
  description = "Security group IDs for database access"
  type        = list(string)
}

# Database Configuration
variable "database_config" {
  description = "Database configuration for the tenant"
  type = object({
    engine_version          = string
    instance_class         = string
    allocated_storage      = number
    max_allocated_storage  = number
    storage_encrypted      = bool
    backup_retention_period = number
    backup_window          = string
    maintenance_window     = string
    deletion_protection    = bool
    multi_az              = bool
    performance_insights   = bool
    monitoring_interval    = number
  })
  default = {
    engine_version          = "14.9"
    instance_class         = "db.t3.micro"
    allocated_storage      = 20
    max_allocated_storage  = 100
    storage_encrypted      = true
    backup_retention_period = 7
    backup_window          = "03:00-04:00"
    maintenance_window     = "sun:04:00-sun:05:00"
    deletion_protection    = true
    multi_az              = false
    performance_insights   = false
    monitoring_interval    = 0
  }
}

# Storage Configuration
variable "storage_config" {
  description = "Storage configuration for the tenant"
  type = object({
    bucket_versioning       = bool
    bucket_encryption      = bool
    lifecycle_rules        = bool
    public_access_block    = bool
    cors_enabled          = bool
    access_logging        = bool
    cloudfront_enabled    = bool
  })
  default = {
    bucket_versioning    = true
    bucket_encryption    = true
    lifecycle_rules      = true
    public_access_block  = true
    cors_enabled        = true
    access_logging      = true
    cloudfront_enabled  = false
  }
}

# VPC Configuration (for network isolation)
variable "vpc_config" {
  description = "VPC configuration for network-isolated tenants"
  type = object({
    cidr_block              = string
    enable_dns_hostnames    = bool
    enable_dns_support      = bool
    enable_nat_gateway      = bool
    single_nat_gateway      = bool
    availability_zones      = list(string)
    private_subnet_cidrs    = list(string)
    public_subnet_cidrs     = list(string)
  })
  default = null
}

# Resource Quotas
variable "resource_quotas" {
  description = "Kubernetes resource quotas for the tenant"
  type = object({
    cpu_request      = string
    cpu_limit        = string
    memory_request   = string
    memory_limit     = string
    storage_request  = string
    pvc_count       = string
    service_count   = string
    configmap_count = string
    secret_count    = string
    deployment_count = string
  })
  default = {
    cpu_request      = "500m"
    cpu_limit        = "2000m"
    memory_request   = "1Gi"
    memory_limit     = "4Gi"
    storage_request  = "10Gi"
    pvc_count       = "5"
    service_count   = "5"
    configmap_count = "10"
    secret_count    = "10"
    deployment_count = "5"
  }
}

# Autoscaling Configuration
variable "autoscaling_config" {
  description = "Horizontal Pod Autoscaler configuration"
  type = object({
    enabled             = bool
    min_replicas       = number
    max_replicas       = number
    target_cpu_percent = number
  })
  default = {
    enabled             = true
    min_replicas       = 2
    max_replicas       = 10
    target_cpu_percent = 70
  }
}

# Pod Disruption Budget Configuration
variable "pdb_config" {
  description = "Pod Disruption Budget configuration"
  type = object({
    min_available = string
  })
  default = {
    min_available = "50%"
  }
}

# Monitoring Configuration
variable "monitoring_config" {
  description = "Monitoring and observability configuration"
  type = object({
    enabled           = bool
    prometheus        = bool
    grafana          = bool
    jaeger           = bool
    log_level        = string
    metrics_retention = string
    alerts_enabled   = bool
  })
  default = {
    enabled           = true
    prometheus        = true
    grafana          = true
    jaeger           = false
    log_level        = "info"
    metrics_retention = "15d"
    alerts_enabled   = true
  }
}

# Backup Configuration
variable "backup_config" {
  description = "Backup and disaster recovery configuration"
  type = object({
    enabled                = bool
    schedule              = string
    retention_days        = number
    cross_region_backup   = bool
    point_in_time_recovery = bool
    automated_backups     = bool
  })
  default = {
    enabled                = true
    schedule              = "0 2 * * *"
    retention_days        = 30
    cross_region_backup   = false
    point_in_time_recovery = true
    automated_backups     = true
  }
}

# Custom Domains
variable "custom_domains" {
  description = "List of custom domains for the tenant"
  type        = list(string)
  default     = []
}

# Rate Limiting Configuration
variable "rate_limit_config" {
  description = "Rate limiting configuration for the tenant"
  type = object({
    requests_per_minute = number
    burst_limit        = number
    enabled           = bool
  })
  default = {
    requests_per_minute = 1000
    burst_limit        = 100
    enabled           = true
  }
}

# Feature Flags
variable "feature_flags" {
  description = "List of enabled features for the tenant"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "security_config" {
  description = "Security configuration for the tenant"
  type = object({
    network_policies_enabled = bool
    pod_security_standards   = string
    rbac_enabled            = bool
    service_mesh_enabled    = bool
    mtls_enabled           = bool
  })
  default = {
    network_policies_enabled = true
    pod_security_standards   = "restricted"
    rbac_enabled            = true
    service_mesh_enabled    = false
    mtls_enabled           = false
  }
}

# API Configuration
variable "api_key" {
  description = "API key for the tenant"
  type        = string
  sensitive   = true
}

variable "encryption_key" {
  description = "Encryption key for tenant data"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT secret for authentication"
  type        = string
  sensitive   = true
}

variable "webhook_secret" {
  description = "Webhook secret for external integrations"
  type        = string
  sensitive   = true
}

# Application Configuration
variable "application_config" {
  description = "Application-specific configuration"
  type = object({
    image_repository = string
    image_tag       = string
    replicas        = number
    ports          = list(number)
    health_check_path = string
    readiness_probe   = object({
      initial_delay_seconds = number
      period_seconds       = number
      timeout_seconds      = number
      success_threshold    = number
      failure_threshold    = number
    })
    liveness_probe = object({
      initial_delay_seconds = number
      period_seconds       = number
      timeout_seconds      = number
      success_threshold    = number
      failure_threshold    = number
    })
  })
  default = {
    image_repository = "whitelabel/tenant-app"
    image_tag       = "latest"
    replicas        = 2
    ports          = [8080]
    health_check_path = "/health"
    readiness_probe = {
      initial_delay_seconds = 30
      period_seconds       = 10
      timeout_seconds      = 5
      success_threshold    = 1
      failure_threshold    = 3
    }
    liveness_probe = {
      initial_delay_seconds = 60
      period_seconds       = 20
      timeout_seconds      = 5
      success_threshold    = 1
      failure_threshold    = 3
    }
  }
}

# Compliance Configuration
variable "compliance_config" {
  description = "Compliance and regulatory configuration"
  type = object({
    gdpr_enabled     = bool
    hipaa_enabled    = bool
    sox_enabled      = bool
    audit_logging    = bool
    data_encryption  = bool
    data_retention_days = number
  })
  default = {
    gdpr_enabled     = false
    hipaa_enabled    = false
    sox_enabled      = false
    audit_logging    = true
    data_encryption  = true
    data_retention_days = 365
  }
}

# Integration Configuration
variable "integration_config" {
  description = "External integration configuration"
  type = object({
    webhook_endpoints = list(string)
    api_integrations = list(object({
      name         = string
      endpoint     = string
      auth_type    = string
      enabled      = bool
    }))
    third_party_services = list(object({
      name    = string
      config  = map(string)
      enabled = bool
    }))
  })
  default = {
    webhook_endpoints = []
    api_integrations = []
    third_party_services = []
  }
}

# Cost Optimization Configuration
variable "cost_config" {
  description = "Cost optimization configuration"
  type = object({
    spot_instances_enabled = bool
    auto_shutdown_enabled  = bool
    shutdown_schedule     = string
    startup_schedule      = string
    rightsizing_enabled   = bool
  })
  default = {
    spot_instances_enabled = false
    auto_shutdown_enabled  = false
    shutdown_schedule     = "0 18 * * 1-5"  # 6 PM weekdays
    startup_schedule      = "0 8 * * 1-5"   # 8 AM weekdays
    rightsizing_enabled   = true
  }
}