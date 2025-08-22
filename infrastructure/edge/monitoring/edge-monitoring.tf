# Edge Monitoring Infrastructure
# Distributed monitoring across 200+ edge locations

# Grafana for visualization
resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = "edge-monitoring"
  version    = "7.0.0"
  
  values = [
    file("${path.module}/values/grafana-values.yaml")
  ]
  
  set {
    name  = "persistence.enabled"
    value = "true"
  }
  
  set {
    name  = "persistence.size"
    value = "10Gi"
  }
  
  set {
    name  = "adminPassword"
    value = var.grafana_admin_password
  }
}

# Prometheus for metrics collection
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "edge-monitoring"
  version    = "51.0.0"
  
  values = [
    file("${path.module}/values/prometheus-values.yaml")
  ]
  
  set {
    name  = "prometheus.prometheusSpec.retention"
    value = "30d"
  }
  
  set {
    name  = "prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage"
    value = "100Gi"
  }
}

# Jaeger for distributed tracing
resource "helm_release" "jaeger" {
  name       = "jaeger"
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger"
  namespace  = "edge-monitoring"
  version    = "0.71.0"
  
  values = [
    file("${path.module}/values/jaeger-values.yaml")
  ]
  
  set {
    name  = "storage.type"
    value = "elasticsearch"
  }
  
  set {
    name  = "elasticsearch.nodeCount"
    value = "3"
  }
}

# Vector for log aggregation
resource "helm_release" "vector" {
  name       = "vector"
  repository = "https://helm.vector.dev"
  chart      = "vector"
  namespace  = "edge-monitoring"
  version    = "0.31.0"
  
  values = [
    templatefile("${path.module}/values/vector-values.yaml", {
      elasticsearch_endpoint = aws_elasticsearch_domain.logs.endpoint
      s3_bucket             = aws_s3_bucket.logs.id
    })
  ]
}

# ElasticSearch for log storage
resource "aws_elasticsearch_domain" "logs" {
  domain_name           = "edge-logs-${var.environment}"
  elasticsearch_version = "8.9"
  
  cluster_config {
    instance_type          = "m5.xlarge.elasticsearch"
    instance_count         = 3
    zone_awareness_enabled = true
    
    zone_awareness_config {
      availability_zone_count = 3
    }
  }
  
  ebs_options {
    ebs_enabled = true
    volume_type = "gp3"
    volume_size = 100
    iops        = 3000
    throughput  = 125
  }
  
  node_to_node_encryption {
    enabled = true
  }
  
  encrypt_at_rest {
    enabled    = true
    kms_key_id = aws_kms_key.monitoring.arn
  }
  
  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }
  
  advanced_security_options {
    enabled                        = true
    internal_user_database_enabled = true
    
    master_user_options {
      master_user_name     = var.elasticsearch_admin_user
      master_user_password = var.elasticsearch_admin_password
    }
  }
  
  tags = local.tags
}

# CloudWatch for AWS metrics
resource "aws_cloudwatch_dashboard" "edge" {
  dashboard_name = "edge-computing-${var.environment}"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/CloudFront", "Requests", { stat = "Sum" }],
            ["AWS/CloudFront", "BytesDownloaded", { stat = "Sum" }],
            ["AWS/CloudFront", "BytesUploaded", { stat = "Sum" }],
            ["AWS/CloudFront", "4xxErrorRate", { stat = "Average" }],
            ["AWS/CloudFront", "5xxErrorRate", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = "us-east-1"
          title  = "CloudFront Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", { stat = "Sum" }],
            ["AWS/Lambda", "Errors", { stat = "Sum" }],
            ["AWS/Lambda", "Duration", { stat = "Average" }],
            ["AWS/Lambda", "ConcurrentExecutions", { stat = "Maximum" }]
          ]
          period = 300
          stat   = "Average"
          region = "us-east-1"
          title  = "Lambda@Edge Metrics"
        }
      }
    ]
  })
}

# SNS topics for alerts
resource "aws_sns_topic" "edge_alerts" {
  name = "edge-alerts-${var.environment}"
  
  kms_master_key_id = aws_kms_key.monitoring.id
  
  tags = local.tags
}

resource "aws_sns_topic_subscription" "edge_alerts_email" {
  topic_arn = aws_sns_topic.edge_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_sns_topic_subscription" "edge_alerts_slack" {
  topic_arn = aws_sns_topic.edge_alerts.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.slack_notifier.arn
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "edge-high-error-rate-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "5xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "1"
  alarm_description   = "This metric monitors 5xx error rate"
  alarm_actions       = [aws_sns_topic.edge_alerts.arn]
  
  dimensions = {
    DistributionId = aws_cloudfront_distribution.edge.id
  }
  
  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "edge-high-latency-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = "60"
  statistic           = "Average"
  threshold           = "1000"
  alarm_description   = "Lambda execution time exceeds 1 second"
  alarm_actions       = [aws_sns_topic.edge_alerts.arn]
  
  tags = local.tags
}

# Application Insights
resource "aws_applicationinsights_application" "edge" {
  resource_group_name = aws_resourcegroups_group.edge.name
  
  tags = local.tags
}

resource "aws_resourcegroups_group" "edge" {
  name = "edge-computing-${var.environment}"
  
  resource_query {
    query = jsonencode({
      ResourceTypeFilters = [
        "AWS::Lambda::Function",
        "AWS::CloudFront::Distribution",
        "AWS::DynamoDB::Table",
        "AWS::S3::Bucket"
      ]
      TagFilters = [
        {
          Key    = "Environment"
          Values = [var.environment]
        },
        {
          Key    = "Platform"
          Values = ["edge-computing"]
        }
      ]
    })
  }
  
  tags = local.tags
}

# X-Ray for tracing
resource "aws_xray_sampling_rule" "edge" {
  rule_name      = "edge-sampling"
  priority       = 1000
  version        = 1
  reservoir_size = 10
  fixed_rate     = 0.1
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "*"
  resource_arn   = "*"
  
  tags = local.tags
}

# Lambda for Slack notifications
resource "aws_lambda_function" "slack_notifier" {
  filename         = data.archive_file.slack_notifier.output_path
  function_name    = "edge-slack-notifier-${var.environment}"
  role            = aws_iam_role.slack_notifier.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.slack_notifier.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 30
  
  environment {
    variables = {
      SLACK_WEBHOOK_URL = var.slack_webhook_url
    }
  }
  
  tags = local.tags
}

resource "aws_iam_role" "slack_notifier" {
  name = "slack-notifier-role-${var.environment}"
  
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

resource "aws_iam_role_policy_attachment" "slack_notifier" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.slack_notifier.name
}

resource "aws_lambda_permission" "sns_invoke" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.slack_notifier.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.edge_alerts.arn
}

data "archive_file" "slack_notifier" {
  type        = "zip"
  source_dir  = "${path.module}/functions/slack-notifier"
  output_path = "${path.module}/dist/slack-notifier.zip"
}

# S3 bucket for logs
resource "aws_s3_bucket" "logs" {
  bucket = "edge-logs-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = local.tags
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  
  rule {
    id     = "delete-old-logs"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}

# KMS key for monitoring encryption
resource "aws_kms_key" "monitoring" {
  description             = "Monitoring encryption key for ${var.environment}"
  deletion_window_in_days = 10
  enable_key_rotation     = true
  
  tags = local.tags
}

# Outputs
output "grafana_url" {
  value = "https://grafana.${var.domain}"
}

output "prometheus_url" {
  value = "https://prometheus.${var.domain}"
}

output "jaeger_url" {
  value = "https://jaeger.${var.domain}"
}

output "elasticsearch_endpoint" {
  value = aws_elasticsearch_domain.logs.endpoint
}

output "cloudwatch_dashboard" {
  value = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.edge.dashboard_name}"
}