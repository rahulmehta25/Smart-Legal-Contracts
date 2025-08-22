# Cloudflare Workers Edge Computing Module

resource "cloudflare_worker_script" "edge_router" {
  account_id = var.cloudflare_account_id
  name       = "edge-router-${var.environment}"
  content    = file("${path.module}/scripts/router.js")
  
  kv_namespace_binding {
    name         = "EDGE_CONFIG"
    namespace_id = cloudflare_workers_kv_namespace.edge_config.id
  }
  
  kv_namespace_binding {
    name         = "CACHE"
    namespace_id = cloudflare_workers_kv_namespace.edge_cache.id
  }
  
  secret_text_binding {
    name = "API_KEY"
    text = var.api_key
  }
  
  service_binding {
    name    = "AUTH"
    service = cloudflare_worker_script.edge_auth.name
  }
  
  service_binding {
    name    = "ML_INFERENCE"
    service = cloudflare_worker_script.ml_inference.name
  }
}

resource "cloudflare_worker_script" "edge_auth" {
  account_id = var.cloudflare_account_id
  name       = "edge-auth-${var.environment}"
  content    = file("${path.module}/scripts/auth.js")
  
  kv_namespace_binding {
    name         = "SESSIONS"
    namespace_id = cloudflare_workers_kv_namespace.sessions.id
  }
  
  kv_namespace_binding {
    name         = "USERS"
    namespace_id = cloudflare_workers_kv_namespace.users.id
  }
}

resource "cloudflare_worker_script" "ml_inference" {
  account_id = var.cloudflare_account_id
  name       = "ml-inference-${var.environment}"
  content    = file("${path.module}/scripts/ml-inference.js")
  
  # Wasm module for ML model
  wasm_binding {
    name   = "MODEL"
    module = filebase64("${path.module}/models/quantized-model.wasm")
  }
  
  kv_namespace_binding {
    name         = "MODEL_CACHE"
    namespace_id = cloudflare_workers_kv_namespace.model_cache.id
  }
}

resource "cloudflare_worker_script" "edge_transform" {
  account_id = var.cloudflare_account_id
  name       = "edge-transform-${var.environment}"
  content    = file("${path.module}/scripts/transform.js")
  
  plain_text_binding {
    name = "TRANSFORM_RULES"
    text = jsonencode(var.transform_rules)
  }
}

resource "cloudflare_worker_script" "ab_testing" {
  account_id = var.cloudflare_account_id
  name       = "ab-testing-${var.environment}"
  content    = file("${path.module}/scripts/ab-testing.js")
  
  kv_namespace_binding {
    name         = "EXPERIMENTS"
    namespace_id = cloudflare_workers_kv_namespace.experiments.id
  }
  
  analytics_engine_binding {
    name    = "ANALYTICS"
    dataset = "ab_testing_metrics"
  }
}

resource "cloudflare_worker_script" "ddos_protection" {
  account_id = var.cloudflare_account_id
  name       = "ddos-protection-${var.environment}"
  content    = file("${path.module}/scripts/ddos-protection.js")
  
  kv_namespace_binding {
    name         = "RATE_LIMITS"
    namespace_id = cloudflare_workers_kv_namespace.rate_limits.id
  }
  
  durable_object_namespace_binding {
    name       = "RATE_LIMITER"
    class_name = "RateLimiter"
    script_name = cloudflare_worker_script.rate_limiter_do.name
  }
}

# Durable Objects for stateful edge computing
resource "cloudflare_worker_script" "rate_limiter_do" {
  account_id = var.cloudflare_account_id
  name       = "rate-limiter-do-${var.environment}"
  content    = file("${path.module}/scripts/durable-objects/rate-limiter.js")
  
  module = true
}

# KV Namespaces for edge storage
resource "cloudflare_workers_kv_namespace" "edge_config" {
  account_id = var.cloudflare_account_id
  title      = "edge-config-${var.environment}"
}

resource "cloudflare_workers_kv_namespace" "edge_cache" {
  account_id = var.cloudflare_account_id
  title      = "edge-cache-${var.environment}"
}

resource "cloudflare_workers_kv_namespace" "sessions" {
  account_id = var.cloudflare_account_id
  title      = "sessions-${var.environment}"
}

resource "cloudflare_workers_kv_namespace" "users" {
  account_id = var.cloudflare_account_id
  title      = "users-${var.environment}"
}

resource "cloudflare_workers_kv_namespace" "model_cache" {
  account_id = var.cloudflare_account_id
  title      = "model-cache-${var.environment}"
}

resource "cloudflare_workers_kv_namespace" "experiments" {
  account_id = var.cloudflare_account_id
  title      = "experiments-${var.environment}"
}

resource "cloudflare_workers_kv_namespace" "rate_limits" {
  account_id = var.cloudflare_account_id
  title      = "rate-limits-${var.environment}"
}

# Worker Routes
resource "cloudflare_worker_route" "main" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "${var.domain}/*"
  script_name = cloudflare_worker_script.edge_router.name
}

resource "cloudflare_worker_route" "api" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "${var.domain}/api/*"
  script_name = cloudflare_worker_script.edge_router.name
}

resource "cloudflare_worker_route" "static" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "${var.domain}/static/*"
  script_name = cloudflare_worker_script.edge_transform.name
}

# Worker Cron Triggers
resource "cloudflare_worker_cron_trigger" "cache_cleanup" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.edge_router.name
  schedules   = ["*/15 * * * *"] # Every 15 minutes
}

resource "cloudflare_worker_cron_trigger" "model_update" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.ml_inference.name
  schedules   = ["0 */6 * * *"] # Every 6 hours
}

# Outputs
output "endpoints" {
  value = {
    main   = "https://${var.domain}"
    api    = "https://${var.domain}/api"
    static = "https://${var.domain}/static"
  }
}

output "kv_namespaces" {
  value = {
    config      = cloudflare_workers_kv_namespace.edge_config.id
    cache       = cloudflare_workers_kv_namespace.edge_cache.id
    sessions    = cloudflare_workers_kv_namespace.sessions.id
    model_cache = cloudflare_workers_kv_namespace.model_cache.id
  }
}