# Edge Computing Cost Optimization Module
# Estimated monthly cost: $8,500 - $12,000 for 200+ edge locations

module "cost_optimization" {
  source = "./modules/cost-optimization"
  
  environment = var.environment
  
  # CloudFlare Workers pricing
  cloudflare_workers = {
    requests_per_month = 100000000  # 100M requests
    cpu_time_ms        = 50         # Average CPU time per request
    kv_reads          = 50000000    # 50M KV reads
    kv_writes         = 5000000     # 5M KV writes
    kv_storage_gb     = 100         # 100GB storage
  }
  
  # AWS Lambda@Edge pricing
  lambda_edge = {
    requests_per_month = 50000000   # 50M requests
    duration_ms        = 100        # Average duration
    memory_mb          = 256        # Memory allocation
  }
  
  # CloudFront pricing
  cloudfront = {
    data_transfer_gb = {
      north_america = 10000  # 10TB
      europe        = 5000   # 5TB
      asia_pacific  = 3000   # 3TB
      south_america = 1000   # 1TB
    }
    requests = {
      http  = 100000000  # 100M
      https = 200000000  # 200M
    }
  }
  
  # K3s clusters
  k3s_clusters = {
    node_count        = 50          # Total nodes across all locations
    node_type        = "t3.medium"  # Instance type
    storage_per_node  = 100         # GB
  }
  
  # Monitoring costs
  monitoring = {
    metrics_per_month      = 10000000  # 10M custom metrics
    logs_ingestion_gb      = 1000      # 1TB logs
    traces_per_month       = 50000000  # 50M traces
    dashboards            = 20
    alerts                = 100
  }
}

# Cost allocation tags
resource "aws_ce_cost_allocation_tag" "edge" {
  tag_key = "CostCenter"
  status  = "Active"
}

resource "aws_ce_cost_allocation_tag" "environment" {
  tag_key = "Environment"
  status  = "Active"
}

resource "aws_ce_cost_allocation_tag" "platform" {
  tag_key = "Platform"
  status  = "Active"
}

# Budget alerts
resource "aws_budgets_budget" "edge_monthly" {
  name              = "edge-computing-monthly-${var.environment}"
  budget_type       = "COST"
  limit_amount      = "12000"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2024-01-01_00:00"
  
  cost_filters = {
    TagKeyValue = [
      "user:Platform$edge-computing",
      "user:Environment$${var.environment}"
    ]
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type            = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type            = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}

# Savings Plans
resource "aws_ce_anomaly_monitor" "edge" {
  name              = "edge-computing-anomaly-${var.environment}"
  monitor_type      = "CUSTOM"
  monitor_frequency = "DAILY"
  
  monitor_specification = jsonencode({
    Or = [
      {
        Tags = {
          Key          = "Platform"
          Values       = ["edge-computing"]
          MatchOptions = ["EQUALS"]
        }
      }
    ]
  })
}

resource "aws_ce_anomaly_subscription" "edge" {
  name      = "edge-computing-anomaly-subscription"
  frequency = "DAILY"
  
  monitor_arn_list = [
    aws_ce_anomaly_monitor.edge.arn
  ]
  
  subscriber {
    type    = "EMAIL"
    address = var.budget_alert_email
  }
  
  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = ["100"]
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }
}

# Reserved Instance recommendations
data "aws_ce_recommendation" "compute_savings" {
  filter {
    name   = "SERVICE"
    values = ["Amazon Elastic Compute Cloud - Compute"]
  }
  
  filter {
    name   = "LINKED_ACCOUNT"
    values = [data.aws_caller_identity.current.account_id]
  }
  
  lookback_period_in_days = "SIXTY_DAYS"
  payment_option          = "NO_UPFRONT"
  service_specification {
    ec2_specification {
      offering_class = "STANDARD"
    }
  }
  term_in_years = "ONE_YEAR"
}

# Cost optimization Lambda
resource "aws_lambda_function" "cost_optimizer" {
  filename         = data.archive_file.cost_optimizer.output_path
  function_name    = "edge-cost-optimizer-${var.environment}"
  role            = aws_iam_role.cost_optimizer.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.cost_optimizer.output_base64sha256
  runtime         = "python3.11"
  timeout         = 300
  memory_size     = 512
  
  environment {
    variables = {
      ENVIRONMENT       = var.environment
      SNS_TOPIC_ARN    = aws_sns_topic.cost_alerts.arn
      S3_BUCKET        = aws_s3_bucket.cost_reports.id
      SLACK_WEBHOOK    = var.slack_webhook_url
    }
  }
  
  tags = local.tags
}

# Schedule cost optimizer to run daily
resource "aws_cloudwatch_event_rule" "cost_optimizer" {
  name                = "edge-cost-optimizer-schedule"
  description         = "Trigger cost optimization analysis"
  schedule_expression = "cron(0 6 * * ? *)"  # Daily at 6 AM UTC
}

resource "aws_cloudwatch_event_target" "cost_optimizer" {
  rule      = aws_cloudwatch_event_rule.cost_optimizer.name
  target_id = "CostOptimizerLambda"
  arn       = aws_lambda_function.cost_optimizer.arn
}

resource "aws_lambda_permission" "cost_optimizer" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_optimizer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cost_optimizer.arn
}

# IAM role for cost optimizer
resource "aws_iam_role" "cost_optimizer" {
  name = "cost-optimizer-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.tags
}

resource "aws_iam_role_policy" "cost_optimizer" {
  name = "cost-optimizer-policy"
  role = aws_iam_role.cost_optimizer.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ce:GetCostAndUsage",
          "ce:GetCostForecast",
          "ce:GetReservationUtilization",
          "ce:GetSavingsPlansUtilization",
          "ce:GetRightsizingRecommendation",
          "ce:GetAnomalies"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeReservedInstances",
          "ec2:ModifyInstanceAttribute",
          "ec2:StopInstances",
          "ec2:StartInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:ListFunctions",
          "lambda:GetFunction",
          "lambda:UpdateFunctionConfiguration"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.cost_reports.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.cost_alerts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# SNS topic for cost alerts
resource "aws_sns_topic" "cost_alerts" {
  name = "edge-cost-alerts-${var.environment}"
  
  kms_master_key_id = aws_kms_key.monitoring.id
  
  tags = local.tags
}

resource "aws_sns_topic_subscription" "cost_alerts_email" {
  topic_arn = aws_sns_topic.cost_alerts.arn
  protocol  = "email"
  endpoint  = var.budget_alert_email
}

# S3 bucket for cost reports
resource "aws_s3_bucket" "cost_reports" {
  bucket = "edge-cost-reports-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = local.tags
}

resource "aws_s3_bucket_lifecycle_configuration" "cost_reports" {
  bucket = aws_s3_bucket.cost_reports.id
  
  rule {
    id     = "delete-old-reports"
    status = "Enabled"
    
    expiration {
      days = 90
    }
  }
}

data "archive_file" "cost_optimizer" {
  type        = "zip"
  source_dir  = "${path.module}/functions/cost-optimizer"
  output_path = "${path.module}/dist/cost-optimizer.zip"
}

data "aws_caller_identity" "current" {}

# Cost Breakdown Output
output "monthly_estimate" {
  value = {
    cloudflare_workers = {
      requests     = "$500"   # 100M requests @ $0.50 per million
      cpu_time     = "$250"   # 5B ms @ $0.05 per million ms
      kv_storage   = "$50"    # 100GB @ $0.50 per GB
      kv_operations = "$300"  # 55M operations @ $0.50 per million reads, $5 per million writes
      subtotal     = "$1,100"
    }
    
    aws_lambda_edge = {
      requests     = "$100"   # 50M requests @ $0.20 per million
      compute_time = "$420"   # 5B ms @ $0.00001667 per GB-second
      subtotal     = "$520"
    }
    
    cloudfront = {
      data_transfer = {
        north_america = "$850"   # 10TB @ $0.085 per GB
        europe        = "$425"   # 5TB @ $0.085 per GB
        asia_pacific  = "$360"   # 3TB @ $0.120 per GB
        south_america = "$160"   # 1TB @ $0.160 per GB
      }
      requests      = "$300"    # 300M requests @ $0.01 per 10,000 HTTPS
      subtotal      = "$2,095"
    }
    
    k3s_infrastructure = {
      ec2_instances = "$1,500"  # 50 t3.medium @ ~$30/month
      ebs_storage   = "$500"    # 5TB @ $0.10 per GB
      data_transfer = "$200"    # Inter-AZ transfer
      subtotal      = "$2,200"
    }
    
    monitoring = {
      cloudwatch_metrics = "$300"   # 10M custom metrics
      cloudwatch_logs    = "$500"   # 1TB ingestion + storage
      elasticsearch      = "$800"   # 3 m5.xlarge instances
      grafana_cloud     = "$200"   # Pro tier
      subtotal          = "$1,800"
    }
    
    ml_inference = {
      model_storage     = "$50"     # S3 storage for models
      inference_compute = "$500"    # GPU/CPU time for inference
      data_transfer    = "$100"    # Model updates
      subtotal         = "$650"
    }
    
    security = {
      waf_requests     = "$200"    # WAF rule evaluations
      shield_standard  = "$0"      # Included
      kms_operations   = "$50"     # Encryption operations
      subtotal         = "$250"
    }
    
    networking = {
      nat_gateway      = "$450"    # 10 NAT gateways @ $45/month
      vpc_endpoints    = "$100"    # S3, DynamoDB endpoints
      route53         = "$50"     # Hosted zones and queries
      subtotal        = "$600"
    }
    
    storage = {
      s3_storage      = "$200"    # Logs, artifacts, models
      dynamodb        = "$300"    # Edge cache tables
      ebs_snapshots   = "$50"     # Backup snapshots
      subtotal        = "$550"
    }
    
    total_estimated  = "$9,765"
    with_20_percent_buffer = "$11,718"
    
    savings_opportunities = [
      "Use Cloudflare Workers KV for frequently accessed data (-$200/month)",
      "Implement aggressive caching strategies (-$500/month)",
      "Use Spot instances for non-critical K3s nodes (-$400/month)",
      "Compress data before transfer (-$300/month)",
      "Use S3 Intelligent-Tiering for logs (-$100/month)",
      "Total potential savings: $1,500/month"
    ]
  }
}