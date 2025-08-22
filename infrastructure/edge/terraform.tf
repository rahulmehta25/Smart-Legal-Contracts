# Edge Computing Infrastructure - Main Configuration
# Deploy to 200+ edge locations globally

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "edge-infrastructure-state"
    key            = "global/edge/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "edge-infrastructure-locks"
  }
}

# Provider configurations for multi-region deployment
provider "aws" {
  alias  = "us-east-1"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu-west-1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "ap-southeast-1"
  region = "ap-southeast-1"
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# Global variables
variable "cloudflare_api_token" {
  description = "Cloudflare API token for Workers deployment"
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "edge_regions" {
  description = "List of edge regions for deployment"
  type        = list(string)
  default = [
    "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
    "ap-southeast-1", "ap-northeast-1", "ap-south-1",
    "sa-east-1", "ca-central-1", "me-south-1"
  ]
}

# Edge deployment locations
locals {
  edge_locations = {
    cloudflare = [
      "atlanta", "boston", "chicago", "dallas", "denver", "los-angeles",
      "miami", "new-york", "phoenix", "san-francisco", "seattle",
      "london", "paris", "frankfurt", "amsterdam", "madrid", "milan",
      "stockholm", "warsaw", "zurich", "dublin",
      "tokyo", "osaka", "singapore", "hong-kong", "seoul", "sydney",
      "mumbai", "bangalore", "chennai", "delhi",
      "sao-paulo", "rio-de-janeiro", "buenos-aires",
      "toronto", "montreal", "vancouver",
      "dubai", "tel-aviv", "johannesburg", "cairo"
    ]
    
    aws_cloudfront = [
      "IAD", "DFW", "LAX", "ORD", "JFK", "SEA", "SFO", "ATL",
      "LHR", "FRA", "CDG", "AMS", "MAD", "MXP", "ARN", "DUB",
      "NRT", "HND", "SIN", "HKG", "ICN", "SYD", "MEL", "AKL",
      "BOM", "DEL", "MAA", "BLR",
      "GRU", "GIG", "EZE",
      "YYZ", "YUL", "YVR"
    ]
  }
  
  deployment_tags = {
    Environment = var.environment
    Platform    = "edge-computing"
    ManagedBy   = "terraform"
    CostCenter  = "infrastructure"
  }
}

# Outputs
output "edge_endpoints" {
  description = "Edge computing endpoints by provider"
  value = {
    cloudflare_workers = module.cloudflare_workers.endpoints
    lambda_edge       = module.lambda_edge.endpoints
    fastly_compute    = module.fastly_compute.endpoints
    k3s_clusters      = module.k3s_edge.cluster_endpoints
  }
}

output "edge_metrics_dashboard" {
  description = "Edge monitoring dashboard URLs"
  value = {
    grafana    = module.edge_monitoring.grafana_url
    prometheus = module.edge_monitoring.prometheus_url
    jaeger     = module.edge_monitoring.jaeger_url
  }
}

output "monthly_cost_estimate" {
  description = "Estimated monthly cost breakdown"
  value = module.cost_optimization.monthly_estimate
}