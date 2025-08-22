# Tenant Infrastructure Module
# This module creates isolated infrastructure for each tenant

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    postgresql = {
      source  = "cyrilgdn/postgresql"
      version = "~> 1.0"
    }
  }
}

# Local variables
locals {
  tenant_name = var.tenant_id
  common_tags = {
    TenantId      = var.tenant_id
    IsolationLevel = var.isolation_strategy
    Environment   = var.environment
    ManagedBy     = "terraform"
    CreatedAt     = timestamp()
  }
}

# Database resources based on isolation strategy
module "database" {
  source = "./modules/database"
  
  tenant_id           = var.tenant_id
  isolation_strategy  = var.isolation_strategy
  database_config     = var.database_config
  environment         = var.environment
  vpc_id             = var.vpc_id
  subnet_ids         = var.private_subnet_ids
  security_group_ids = var.database_security_group_ids
  
  tags = local.common_tags
}

# Storage resources
module "storage" {
  source = "./modules/storage"
  
  tenant_id      = var.tenant_id
  environment    = var.environment
  storage_config = var.storage_config
  
  tags = local.common_tags
}

# Networking resources for tenant isolation
module "networking" {
  source = "./modules/networking"
  count  = var.isolation_strategy == "network" ? 1 : 0
  
  tenant_id   = var.tenant_id
  environment = var.environment
  vpc_config  = var.vpc_config
  
  tags = local.common_tags
}

# Kubernetes namespace and resources
resource "kubernetes_namespace" "tenant" {
  metadata {
    name = "tenant-${var.tenant_id}"
    
    labels = {
      "app.kubernetes.io/name"       = "whitelabel-tenant"
      "app.kubernetes.io/instance"   = var.tenant_id
      "app.kubernetes.io/part-of"    = "whitelabel-platform"
      "app.kubernetes.io/managed-by" = "terraform"
      "tenant.whitelabel.io/id"      = var.tenant_id
      "tenant.whitelabel.io/isolation" = var.isolation_strategy
    }
    
    annotations = {
      "tenant.whitelabel.io/created-by" = "terraform"
      "tenant.whitelabel.io/plan"       = var.tenant_plan
    }
  }
}

# Resource quotas for tenant isolation
resource "kubernetes_resource_quota" "tenant" {
  metadata {
    name      = "tenant-${var.tenant_id}-quota"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  spec {
    hard = {
      "requests.cpu"                = var.resource_quotas.cpu_request
      "requests.memory"             = var.resource_quotas.memory_request
      "limits.cpu"                  = var.resource_quotas.cpu_limit
      "limits.memory"               = var.resource_quotas.memory_limit
      "requests.storage"            = var.resource_quotas.storage_request
      "persistentvolumeclaims"      = var.resource_quotas.pvc_count
      "services"                    = var.resource_quotas.service_count
      "configmaps"                  = var.resource_quotas.configmap_count
      "secrets"                     = var.resource_quotas.secret_count
      "count/deployments.apps"      = var.resource_quotas.deployment_count
    }
  }
}

# Network policies for tenant isolation
resource "kubernetes_network_policy" "tenant_isolation" {
  metadata {
    name      = "tenant-${var.tenant_id}-isolation"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  spec {
    pod_selector {
      match_labels = {
        "tenant.whitelabel.io/id" = var.tenant_id
      }
    }
    
    policy_types = ["Ingress", "Egress"]
    
    ingress {
      from {
        namespace_selector {
          match_labels = {
            "name" = "whitelabel-system"
          }
        }
      }
      from {
        namespace_selector {
          match_labels = {
            "tenant.whitelabel.io/id" = var.tenant_id
          }
        }
      }
    }
    
    egress {
      # Allow egress to same tenant namespace
      to {
        namespace_selector {
          match_labels = {
            "tenant.whitelabel.io/id" = var.tenant_id
          }
        }
      }
      
      # Allow egress to system services
      to {
        namespace_selector {
          match_labels = {
            "name" = "whitelabel-system"
          }
        }
      }
      
      # Allow egress to external services on specific ports
      ports {
        protocol = "TCP"
        port     = "443"
      }
      ports {
        protocol = "TCP"
        port     = "80"
      }
      ports {
        protocol = "TCP"
        port     = "5432"  # PostgreSQL
      }
      ports {
        protocol = "TCP"
        port     = "6379"  # Redis
      }
    }
  }
}

# Service account for tenant applications
resource "kubernetes_service_account" "tenant" {
  metadata {
    name      = "tenant-${var.tenant_id}-app"
    namespace = kubernetes_namespace.tenant.metadata[0].name
    
    labels = {
      "tenant.whitelabel.io/id" = var.tenant_id
    }
  }
  
  automount_service_account_token = true
}

# Role for tenant service account
resource "kubernetes_role" "tenant" {
  metadata {
    name      = "tenant-${var.tenant_id}-role"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  rule {
    api_groups = [""]
    resources  = ["configmaps", "secrets", "services", "pods"]
    verbs      = ["get", "list", "watch", "create", "update", "patch"]
  }
  
  rule {
    api_groups = ["apps"]
    resources  = ["deployments", "replicasets"]
    verbs      = ["get", "list", "watch", "create", "update", "patch"]
  }
  
  rule {
    api_groups = ["networking.k8s.io"]
    resources  = ["networkpolicies"]
    verbs      = ["get", "list", "watch"]
  }
}

# Role binding for tenant service account
resource "kubernetes_role_binding" "tenant" {
  metadata {
    name      = "tenant-${var.tenant_id}-binding"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.tenant.metadata[0].name
  }
  
  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.tenant.metadata[0].name
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
}

# ConfigMap for tenant configuration
resource "kubernetes_config_map" "tenant_config" {
  metadata {
    name      = "tenant-${var.tenant_id}-config"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  data = {
    "tenant.id"               = var.tenant_id
    "tenant.name"             = var.tenant_name
    "tenant.plan"             = var.tenant_plan
    "isolation.strategy"      = var.isolation_strategy
    "database.host"           = module.database.host
    "database.port"           = module.database.port
    "database.name"           = module.database.database_name
    "storage.bucket"          = module.storage.bucket_name
    "monitoring.enabled"      = var.monitoring_config.enabled
    "logging.level"           = var.monitoring_config.log_level
    "features.enabled"        = join(",", var.feature_flags)
  }
}

# Secret for sensitive tenant configuration
resource "kubernetes_secret" "tenant_secrets" {
  metadata {
    name      = "tenant-${var.tenant_id}-secrets"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  type = "Opaque"
  
  data = {
    "database.username"     = base64encode(module.database.username)
    "database.password"     = base64encode(module.database.password)
    "api.key"               = base64encode(var.api_key)
    "encryption.key"        = base64encode(var.encryption_key)
    "jwt.secret"            = base64encode(var.jwt_secret)
    "webhook.secret"        = base64encode(var.webhook_secret)
  }
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler" "tenant_app" {
  count = var.autoscaling_config.enabled ? 1 : 0
  
  metadata {
    name      = "tenant-${var.tenant_id}-hpa"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "tenant-${var.tenant_id}-app"
    }
    
    min_replicas = var.autoscaling_config.min_replicas
    max_replicas = var.autoscaling_config.max_replicas
    
    target_cpu_utilization_percentage = var.autoscaling_config.target_cpu_percent
  }
}

# Pod Disruption Budget
resource "kubernetes_pod_disruption_budget" "tenant_app" {
  metadata {
    name      = "tenant-${var.tenant_id}-pdb"
    namespace = kubernetes_namespace.tenant.metadata[0].name
  }
  
  spec {
    min_available = var.pdb_config.min_available
    
    selector {
      match_labels = {
        "app"                     = "tenant-${var.tenant_id}-app"
        "tenant.whitelabel.io/id" = var.tenant_id
      }
    }
  }
}

# Monitoring resources
module "monitoring" {
  source = "./modules/monitoring"
  count  = var.monitoring_config.enabled ? 1 : 0
  
  tenant_id   = var.tenant_id
  environment = var.environment
  namespace   = kubernetes_namespace.tenant.metadata[0].name
  
  monitoring_config = var.monitoring_config
  
  tags = local.common_tags
}

# Backup configuration
module "backup" {
  source = "./modules/backup"
  count  = var.backup_config.enabled ? 1 : 0
  
  tenant_id      = var.tenant_id
  environment    = var.environment
  backup_config  = var.backup_config
  database_config = module.database
  storage_config  = module.storage
  
  tags = local.common_tags
}

# DNS and SSL certificates for custom domains
module "dns" {
  source = "./modules/dns"
  count  = length(var.custom_domains) > 0 ? 1 : 0
  
  tenant_id      = var.tenant_id
  custom_domains = var.custom_domains
  environment    = var.environment
  
  tags = local.common_tags
}

# Load balancer configuration
resource "kubernetes_service" "tenant_app" {
  metadata {
    name      = "tenant-${var.tenant_id}-service"
    namespace = kubernetes_namespace.tenant.metadata[0].name
    
    labels = {
      "app"                     = "tenant-${var.tenant_id}-app"
      "tenant.whitelabel.io/id" = var.tenant_id
    }
    
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
      "external-dns.alpha.kubernetes.io/hostname"         = join(",", var.custom_domains)
    }
  }
  
  spec {
    selector = {
      "app"                     = "tenant-${var.tenant_id}-app"
      "tenant.whitelabel.io/id" = var.tenant_id
    }
    
    type = "LoadBalancer"
    
    port {
      name        = "http"
      port        = 80
      target_port = 8080
    }
    
    port {
      name        = "https"
      port        = 443
      target_port = 8080
    }
  }
}

# Ingress for custom domains
resource "kubernetes_ingress_v1" "tenant_app" {
  count = length(var.custom_domains) > 0 ? 1 : 0
  
  metadata {
    name      = "tenant-${var.tenant_id}-ingress"
    namespace = kubernetes_namespace.tenant.metadata[0].name
    
    annotations = {
      "kubernetes.io/ingress.class"                = "nginx"
      "cert-manager.io/cluster-issuer"             = "letsencrypt-prod"
      "nginx.ingress.kubernetes.io/ssl-redirect"   = "true"
      "nginx.ingress.kubernetes.io/force-ssl-redirect" = "true"
      "nginx.ingress.kubernetes.io/rate-limit"     = tostring(var.rate_limit_config.requests_per_minute)
    }
  }
  
  spec {
    tls {
      hosts       = var.custom_domains
      secret_name = "tenant-${var.tenant_id}-tls"
    }
    
    dynamic "rule" {
      for_each = var.custom_domains
      content {
        host = rule.value
        http {
          path {
            path      = "/"
            path_type = "Prefix"
            backend {
              service {
                name = kubernetes_service.tenant_app.metadata[0].name
                port {
                  number = 80
                }
              }
            }
          }
        }
      }
    }
  }
}