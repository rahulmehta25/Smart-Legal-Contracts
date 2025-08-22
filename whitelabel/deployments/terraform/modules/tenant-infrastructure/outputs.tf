# Tenant Infrastructure Module Outputs

# Tenant Information
output "tenant_id" {
  description = "The tenant ID"
  value       = var.tenant_id
}

output "tenant_name" {
  description = "The tenant name"
  value       = var.tenant_name
}

output "tenant_plan" {
  description = "The tenant plan"
  value       = var.tenant_plan
}

output "isolation_strategy" {
  description = "The isolation strategy used for this tenant"
  value       = var.isolation_strategy
}

# Database Outputs
output "database_host" {
  description = "Database host endpoint"
  value       = module.database.host
  sensitive   = false
}

output "database_port" {
  description = "Database port"
  value       = module.database.port
}

output "database_name" {
  description = "Database name"
  value       = module.database.database_name
}

output "database_username" {
  description = "Database username"
  value       = module.database.username
  sensitive   = true
}

output "database_password" {
  description = "Database password"
  value       = module.database.password
  sensitive   = true
}

output "database_connection_string" {
  description = "Database connection string"
  value       = module.database.connection_string
  sensitive   = true
}

output "database_arn" {
  description = "Database ARN"
  value       = module.database.arn
}

# Storage Outputs
output "storage_bucket_name" {
  description = "S3 bucket name for tenant storage"
  value       = module.storage.bucket_name
}

output "storage_bucket_arn" {
  description = "S3 bucket ARN"
  value       = module.storage.bucket_arn
}

output "storage_bucket_domain_name" {
  description = "S3 bucket domain name"
  value       = module.storage.bucket_domain_name
}

output "storage_cloudfront_distribution_id" {
  description = "CloudFront distribution ID (if enabled)"
  value       = var.storage_config.cloudfront_enabled ? module.storage.cloudfront_distribution_id : null
}

# Kubernetes Outputs
output "kubernetes_namespace" {
  description = "Kubernetes namespace for the tenant"
  value       = kubernetes_namespace.tenant.metadata[0].name
}

output "kubernetes_service_account" {
  description = "Kubernetes service account for tenant applications"
  value       = kubernetes_service_account.tenant.metadata[0].name
}

output "kubernetes_config_map_name" {
  description = "Kubernetes ConfigMap name containing tenant configuration"
  value       = kubernetes_config_map.tenant_config.metadata[0].name
}

output "kubernetes_secret_name" {
  description = "Kubernetes Secret name containing tenant secrets"
  value       = kubernetes_secret.tenant_secrets.metadata[0].name
}

output "kubernetes_service_name" {
  description = "Kubernetes Service name for tenant application"
  value       = kubernetes_service.tenant_app.metadata[0].name
}

# Load Balancer Outputs
output "load_balancer_hostname" {
  description = "Load balancer hostname"
  value       = kubernetes_service.tenant_app.status[0].load_balancer[0].ingress[0].hostname
}

output "load_balancer_ip" {
  description = "Load balancer IP address"
  value       = kubernetes_service.tenant_app.status[0].load_balancer[0].ingress[0].ip
}

# DNS and SSL Outputs
output "custom_domains" {
  description = "Custom domains configured for the tenant"
  value       = var.custom_domains
}

output "ssl_certificate_arn" {
  description = "SSL certificate ARN (if custom domains are configured)"
  value       = length(var.custom_domains) > 0 ? module.dns[0].certificate_arn : null
}

output "dns_records" {
  description = "DNS records created for custom domains"
  value       = length(var.custom_domains) > 0 ? module.dns[0].dns_records : []
}

# Networking Outputs (for network isolation)
output "vpc_id" {
  description = "VPC ID for network-isolated tenants"
  value       = var.isolation_strategy == "network" ? module.networking[0].vpc_id : var.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = var.isolation_strategy == "network" ? module.networking[0].private_subnet_ids : var.private_subnet_ids
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = var.isolation_strategy == "network" ? module.networking[0].public_subnet_ids : var.public_subnet_ids
}

output "security_group_ids" {
  description = "Security group IDs for the tenant"
  value       = var.isolation_strategy == "network" ? module.networking[0].security_group_ids : var.database_security_group_ids
}

# Resource Quota Outputs
output "resource_quota_name" {
  description = "Kubernetes ResourceQuota name"
  value       = kubernetes_resource_quota.tenant.metadata[0].name
}

output "resource_limits" {
  description = "Resource limits applied to the tenant"
  value       = var.resource_quotas
}

# Monitoring Outputs
output "monitoring_enabled" {
  description = "Whether monitoring is enabled for the tenant"
  value       = var.monitoring_config.enabled
}

output "prometheus_service_monitor" {
  description = "Prometheus ServiceMonitor name (if monitoring is enabled)"
  value       = var.monitoring_config.enabled ? module.monitoring[0].service_monitor_name : null
}

output "grafana_dashboard_url" {
  description = "Grafana dashboard URL (if monitoring is enabled)"
  value       = var.monitoring_config.enabled ? module.monitoring[0].grafana_dashboard_url : null
}

# Backup Outputs
output "backup_enabled" {
  description = "Whether automated backups are enabled"
  value       = var.backup_config.enabled
}

output "backup_schedule" {
  description = "Backup schedule (cron format)"
  value       = var.backup_config.schedule
}

output "backup_retention_days" {
  description = "Backup retention period in days"
  value       = var.backup_config.retention_days
}

output "backup_s3_bucket" {
  description = "S3 bucket used for backups"
  value       = var.backup_config.enabled ? module.backup[0].backup_bucket_name : null
}

# Security Outputs
output "network_policy_name" {
  description = "Kubernetes NetworkPolicy name for tenant isolation"
  value       = kubernetes_network_policy.tenant_isolation.metadata[0].name
}

output "rbac_role_name" {
  description = "RBAC role name for tenant service account"
  value       = kubernetes_role.tenant.metadata[0].name
}

output "security_config" {
  description = "Security configuration applied to the tenant"
  value       = var.security_config
}

# Application Outputs
output "application_image" {
  description = "Docker image used for tenant application"
  value       = "${var.application_config.image_repository}:${var.application_config.image_tag}"
}

output "application_replicas" {
  description = "Number of application replicas"
  value       = var.application_config.replicas
}

output "health_check_endpoints" {
  description = "Health check endpoints for the tenant application"
  value = {
    readiness = var.application_config.health_check_path
    liveness  = var.application_config.health_check_path
  }
}

# Autoscaling Outputs
output "hpa_enabled" {
  description = "Whether Horizontal Pod Autoscaler is enabled"
  value       = var.autoscaling_config.enabled
}

output "hpa_min_replicas" {
  description = "Minimum number of replicas for autoscaling"
  value       = var.autoscaling_config.min_replicas
}

output "hpa_max_replicas" {
  description = "Maximum number of replicas for autoscaling"
  value       = var.autoscaling_config.max_replicas
}

# Cost Optimization Outputs
output "cost_optimization_enabled" {
  description = "Cost optimization features enabled"
  value = {
    spot_instances = var.cost_config.spot_instances_enabled
    auto_shutdown  = var.cost_config.auto_shutdown_enabled
    rightsizing    = var.cost_config.rightsizing_enabled
  }
}

output "cost_schedule" {
  description = "Auto-shutdown and startup schedule"
  value = var.cost_config.auto_shutdown_enabled ? {
    shutdown = var.cost_config.shutdown_schedule
    startup  = var.cost_config.startup_schedule
  } : null
}

# Compliance Outputs
output "compliance_features" {
  description = "Compliance features enabled for the tenant"
  value = {
    gdpr_enabled       = var.compliance_config.gdpr_enabled
    hipaa_enabled      = var.compliance_config.hipaa_enabled
    sox_enabled        = var.compliance_config.sox_enabled
    audit_logging      = var.compliance_config.audit_logging
    data_encryption    = var.compliance_config.data_encryption
    retention_days     = var.compliance_config.data_retention_days
  }
}

# Integration Outputs
output "webhook_endpoints" {
  description = "Configured webhook endpoints"
  value       = var.integration_config.webhook_endpoints
}

output "api_integrations" {
  description = "Configured API integrations"
  value       = var.integration_config.api_integrations
  sensitive   = true
}

# Feature Flags Output
output "enabled_features" {
  description = "List of enabled features for the tenant"
  value       = var.feature_flags
}

# Environment and Tags
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "common_tags" {
  description = "Common tags applied to all resources"
  value = {
    TenantId      = var.tenant_id
    IsolationLevel = var.isolation_strategy
    Environment   = var.environment
    ManagedBy     = "terraform"
  }
}

# Ingress Outputs
output "ingress_name" {
  description = "Kubernetes Ingress name (if custom domains are configured)"
  value       = length(var.custom_domains) > 0 ? kubernetes_ingress_v1.tenant_app[0].metadata[0].name : null
}

output "ingress_class" {
  description = "Ingress class used"
  value       = length(var.custom_domains) > 0 ? "nginx" : null
}

output "ssl_redirect_enabled" {
  description = "Whether SSL redirect is enabled"
  value       = length(var.custom_domains) > 0
}

# Rate Limiting Outputs
output "rate_limit_config" {
  description = "Rate limiting configuration"
  value = var.rate_limit_config.enabled ? {
    requests_per_minute = var.rate_limit_config.requests_per_minute
    burst_limit        = var.rate_limit_config.burst_limit
  } : null
}

# Pod Disruption Budget Output
output "pod_disruption_budget" {
  description = "Pod Disruption Budget configuration"
  value = {
    name         = kubernetes_pod_disruption_budget.tenant_app.metadata[0].name
    min_available = var.pdb_config.min_available
  }
}

# Summary Output for Easy Reference
output "tenant_summary" {
  description = "Complete summary of tenant infrastructure"
  value = {
    tenant = {
      id               = var.tenant_id
      name             = var.tenant_name
      plan             = var.tenant_plan
      isolation_strategy = var.isolation_strategy
      environment      = var.environment
    }
    database = {
      host     = module.database.host
      port     = module.database.port
      name     = module.database.database_name
    }
    storage = {
      bucket_name = module.storage.bucket_name
    }
    kubernetes = {
      namespace     = kubernetes_namespace.tenant.metadata[0].name
      service_name  = kubernetes_service.tenant_app.metadata[0].name
      config_map    = kubernetes_config_map.tenant_config.metadata[0].name
    }
    networking = {
      load_balancer = kubernetes_service.tenant_app.status[0].load_balancer[0].ingress[0].hostname
      custom_domains = var.custom_domains
    }
    features = {
      monitoring_enabled = var.monitoring_config.enabled
      backup_enabled    = var.backup_config.enabled
      autoscaling_enabled = var.autoscaling_config.enabled
      ssl_enabled       = length(var.custom_domains) > 0
    }
  }
}