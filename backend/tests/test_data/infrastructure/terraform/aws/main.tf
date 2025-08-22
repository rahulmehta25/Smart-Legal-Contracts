# AWS Infrastructure - Main Configuration
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
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
    bucket         = "terraform-state-prod"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "Terraform"
      Project     = var.project_name
      CostCenter  = var.cost_center
    }
  }
}

provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "../modules/vpc"
  
  vpc_cidr            = var.vpc_cidr
  environment         = var.environment
  availability_zones  = data.aws_availability_zones.available.names
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs
  enable_nat_gateway   = true
  enable_vpn_gateway   = true
  enable_flow_logs     = true
  
  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# EKS Cluster
module "eks" {
  source = "../modules/eks"
  
  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  
  # Node groups configuration
  node_groups = {
    general = {
      desired_capacity = 3
      min_capacity     = 3
      max_capacity     = 10
      instance_types   = ["t3.large"]
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "general"
      }
      
      additional_tags = {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/enabled"             = "true"
      }
    }
    
    spot = {
      desired_capacity = 2
      min_capacity     = 0
      max_capacity     = 20
      instance_types   = ["t3.large", "t3a.large"]
      capacity_type    = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "spot"
      }
      
      taints = [{
        key    = "spot"
        value  = "true"
        effect = "NoSchedule"
      }]
    }
  }
  
  # OIDC provider for IRSA
  enable_irsa = true
  
  # Cluster addons
  cluster_addons = {
    coredns = {
      addon_version = "v1.10.1-eksbuild.1"
    }
    kube-proxy = {
      addon_version = "v1.28.1-eksbuild.1"
    }
    vpc-cni = {
      addon_version = "v1.14.1-eksbuild.1"
    }
    aws-ebs-csi-driver = {
      addon_version = "v1.23.0-eksbuild.1"
    }
  }
}

# RDS Aurora Serverless v2
module "aurora" {
  source = "../modules/aurora"
  
  cluster_identifier      = "${var.project_name}-${var.environment}"
  engine                  = "aurora-postgresql"
  engine_version          = "15.3"
  database_name           = var.database_name
  master_username         = var.database_username
  vpc_id                  = module.vpc.vpc_id
  subnet_ids              = module.vpc.private_subnet_ids
  
  serverlessv2_scaling_configuration = {
    max_capacity = 16
    min_capacity = 0.5
  }
  
  backup_retention_period = 30
  preferred_backup_window = "03:00-04:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  create_db_cluster_parameter_group = true
  db_cluster_parameter_group_parameters = [
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements,pg_hint_plan"
    }
  ]
}

# ElastiCache Redis
module "elasticache" {
  source = "../modules/elasticache"
  
  cluster_id           = "${var.project_name}-${var.environment}"
  node_type            = "cache.r7g.large"
  num_cache_nodes      = 3
  engine_version       = "7.0"
  vpc_id               = module.vpc.vpc_id
  subnet_ids           = module.vpc.private_subnet_ids
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 7
  snapshot_window          = "03:00-05:00"
  
  # Enable Redis AUTH
  auth_token_enabled = true
  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
}

# S3 Buckets
module "s3_buckets" {
  source = "../modules/s3"
  
  buckets = {
    static_assets = {
      bucket_name = "${var.project_name}-${var.environment}-static"
      versioning  = true
      lifecycle_rules = [{
        id      = "expire_old_versions"
        status  = "Enabled"
        noncurrent_version_expiration = {
          days = 90
        }
      }]
    }
    
    backups = {
      bucket_name = "${var.project_name}-${var.environment}-backups"
      versioning  = true
      lifecycle_rules = [{
        id      = "transition_to_glacier"
        status  = "Enabled"
        transition = [{
          days          = 30
          storage_class = "GLACIER"
        }]
      }]
    }
  }
  
  enable_encryption = true
  enable_logging    = true
}

# CloudFront CDN
module "cloudfront" {
  source = "../modules/cloudfront"
  
  origin_domain_name = module.s3_buckets.bucket_regional_domain_names["static_assets"]
  s3_origin_id       = "S3-${module.s3_buckets.bucket_ids["static_assets"]}"
  
  aliases = ["cdn.${var.domain_name}"]
  
  default_cache_behavior = {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${module.s3_buckets.bucket_ids["static_assets"]}"
    
    forwarded_values = {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 86400
    max_ttl                = 31536000
  }
  
  price_class = "PriceClass_100"
  
  geo_restriction = {
    restriction_type = "none"
  }
  
  viewer_certificate = {
    acm_certificate_arn = aws_acm_certificate.cdn.arn
    ssl_support_method  = "sni-only"
  }
}

# WAF for CloudFront
module "waf" {
  source = "../modules/waf"
  
  scope = "CLOUDFRONT"
  
  rules = [
    {
      name     = "RateLimitRule"
      priority = 1
      action   = "block"
      
      rate_based_statement = {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    },
    {
      name     = "AWSManagedRulesCommonRuleSet"
      priority = 2
      
      managed_rule_group_statement = {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }
  ]
}

# Auto Scaling for EC2 instances (if needed outside EKS)
module "auto_scaling" {
  source = "../modules/auto-scaling"
  
  name                = "${var.project_name}-${var.environment}-asg"
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  
  min_size            = 2
  max_size            = 100
  desired_capacity    = 3
  
  instance_type       = "t3.medium"
  
  # Auto-scaling policies
  target_group_arns   = [module.alb.target_group_arns[0]]
  
  scaling_policies = [
    {
      name               = "cpu-utilization"
      policy_type        = "TargetTrackingScaling"
      target_value       = 70.0
      predefined_metric_type = "ASGAverageCPUUtilization"
    },
    {
      name               = "request-count"
      policy_type        = "TargetTrackingScaling"
      target_value       = 1000.0
      predefined_metric_type = "ALBRequestCountPerTarget"
    }
  ]
}

# Application Load Balancer
module "alb" {
  source = "../modules/alb"
  
  name               = "${var.project_name}-${var.environment}-alb"
  vpc_id             = module.vpc.vpc_id
  subnets            = module.vpc.public_subnet_ids
  
  enable_deletion_protection = true
  enable_http2              = true
  enable_cross_zone_load_balancing = true
  
  access_logs = {
    enabled = true
    bucket  = module.s3_buckets.bucket_ids["logs"]
  }
  
  listeners = [
    {
      port     = 443
      protocol = "HTTPS"
      certificate_arn = aws_acm_certificate.main.arn
      
      default_action = {
        type             = "forward"
        target_group_arn = aws_lb_target_group.main.arn
      }
    },
    {
      port     = 80
      protocol = "HTTP"
      
      default_action = {
        type = "redirect"
        redirect = {
          port        = "443"
          protocol    = "HTTPS"
          status_code = "HTTP_301"
        }
      }
    }
  ]
}

# CloudWatch Monitoring
module "cloudwatch" {
  source = "../modules/cloudwatch"
  
  log_groups = {
    "/aws/eks/${var.cluster_name}/cluster" = {
      retention_in_days = 30
    }
    "/aws/rds/cluster/${module.aurora.cluster_identifier}/postgresql" = {
      retention_in_days = 7
    }
  }
  
  alarms = {
    high_cpu = {
      alarm_name          = "${var.project_name}-high-cpu"
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
      metric_name         = "CPUUtilization"
      namespace           = "AWS/EKS"
      period              = 300
      statistic           = "Average"
      threshold           = 80
      alarm_description   = "This metric monitors EKS CPU utilization"
    }
    
    high_memory = {
      alarm_name          = "${var.project_name}-high-memory"
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
      metric_name         = "MemoryUtilization"
      namespace           = "AWS/EKS"
      period              = 300
      statistic           = "Average"
      threshold           = 85
      alarm_description   = "This metric monitors EKS memory utilization"
    }
  }
  
  sns_topic_arn = aws_sns_topic.alerts.arn
}

# SNS for alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-${var.environment}-alerts"
  
  kms_master_key_id = aws_kms_key.sns.id
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# KMS Keys
resource "aws_kms_key" "main" {
  description             = "KMS key for ${var.project_name} ${var.environment}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_key" "sns" {
  description             = "KMS key for SNS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

# ACM Certificates
resource "aws_acm_certificate" "main" {
  domain_name       = var.domain_name
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.${var.domain_name}"
  ]
  
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_acm_certificate" "cdn" {
  provider = aws.us_east_1  # CloudFront requires certs in us-east-1
  
  domain_name       = "cdn.${var.domain_name}"
  validation_method = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
}