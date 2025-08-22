# AWS Lambda@Edge Infrastructure

# Lambda@Edge functions must be in us-east-1
provider "aws" {
  alias  = "lambda_edge"
  region = "us-east-1"
}

# IAM role for Lambda@Edge
resource "aws_iam_role" "lambda_edge" {
  provider = aws.lambda_edge
  name     = "lambda-edge-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "lambda.amazonaws.com",
            "edgelambda.amazonaws.com"
          ]
        }
      }
    ]
  })
  
  tags = local.tags
}

# IAM policy for Lambda@Edge
resource "aws_iam_role_policy" "lambda_edge" {
  provider = aws.lambda_edge
  name     = "lambda-edge-policy"
  role     = aws_iam_role.lambda_edge.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.edge_cache.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.edge_storage.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.edge_encryption.arn
      }
    ]
  })
}

# Viewer Request Function - Authentication & Routing
resource "aws_lambda_function" "viewer_request" {
  provider         = aws.lambda_edge
  filename         = data.archive_file.viewer_request.output_path
  function_name    = "edge-viewer-request-${var.environment}"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.viewer_request.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 128
  publish         = true
  
  environment {
    variables = {
      ENVIRONMENT = var.environment
      JWT_SECRET  = var.jwt_secret
    }
  }
  
  tags = local.tags
}

# Origin Request Function - Cache Control & Optimization
resource "aws_lambda_function" "origin_request" {
  provider         = aws.lambda_edge
  filename         = data.archive_file.origin_request.output_path
  function_name    = "edge-origin-request-${var.environment}"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.origin_request.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 30
  memory_size     = 512
  publish         = true
  
  environment {
    variables = {
      CACHE_TABLE = aws_dynamodb_table.edge_cache.name
      S3_BUCKET   = aws_s3_bucket.edge_storage.id
    }
  }
  
  tags = local.tags
}

# Viewer Response Function - Headers & Security
resource "aws_lambda_function" "viewer_response" {
  provider         = aws.lambda_edge
  filename         = data.archive_file.viewer_response.output_path
  function_name    = "edge-viewer-response-${var.environment}"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.viewer_response.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 128
  publish         = true
  
  tags = local.tags
}

# Origin Response Function - Caching & Compression
resource "aws_lambda_function" "origin_response" {
  provider         = aws.lambda_edge
  filename         = data.archive_file.origin_response.output_path
  function_name    = "edge-origin-response-${var.environment}"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.origin_response.output_base64sha256
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 256
  publish         = true
  
  tags = local.tags
}

# ML Inference Function
resource "aws_lambda_function" "ml_inference" {
  provider         = aws.lambda_edge
  filename         = data.archive_file.ml_inference.output_path
  function_name    = "edge-ml-inference-${var.environment}"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.ml_inference.output_base64sha256
  runtime         = "python3.11"
  timeout         = 30
  memory_size     = 3008  # Maximum for Lambda@Edge
  publish         = true
  
  layers = [
    aws_lambda_layer_version.tensorflow_lite.arn,
    aws_lambda_layer_version.numpy.arn
  ]
  
  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.ml_models.id
      MODEL_PATH   = "models/quantized/"
    }
  }
  
  tags = local.tags
}

# Lambda Layers for ML dependencies
resource "aws_lambda_layer_version" "tensorflow_lite" {
  provider            = aws.lambda_edge
  filename            = "layers/tensorflow-lite.zip"
  layer_name          = "tensorflow-lite"
  compatible_runtimes = ["python3.11"]
  
  description = "TensorFlow Lite for edge inference"
}

resource "aws_lambda_layer_version" "numpy" {
  provider            = aws.lambda_edge
  filename            = "layers/numpy.zip"
  layer_name          = "numpy"
  compatible_runtimes = ["python3.11"]
  
  description = "NumPy for numerical computations"
}

# CloudFront Distribution with Lambda@Edge
resource "aws_cloudfront_distribution" "edge" {
  enabled             = true
  is_ipv6_enabled    = true
  comment            = "Edge Computing Distribution - ${var.environment}"
  default_root_object = "index.html"
  price_class        = "PriceClass_All"  # Use all edge locations
  
  origin {
    domain_name = aws_s3_bucket.origin.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.origin.id}"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.oai.cloudfront_access_identity_path
    }
  }
  
  origin {
    domain_name = aws_alb.api.dns_name
    origin_id   = "ALB-${aws_alb.api.id}"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  # Default cache behavior with Lambda@Edge
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "S3-${aws_s3_bucket.origin.id}"
    
    forwarded_values {
      query_string = true
      headers      = ["Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"]
      
      cookies {
        forward = "whitelist"
        whitelisted_names = ["session", "auth_token"]
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
    
    # Lambda@Edge associations
    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.viewer_request.qualified_arn
      include_body = false
    }
    
    lambda_function_association {
      event_type   = "origin-request"
      lambda_arn   = aws_lambda_function.origin_request.qualified_arn
      include_body = true
    }
    
    lambda_function_association {
      event_type   = "origin-response"
      lambda_arn   = aws_lambda_function.origin_response.qualified_arn
      include_body = false
    }
    
    lambda_function_association {
      event_type   = "viewer-response"
      lambda_arn   = aws_lambda_function.viewer_response.qualified_arn
      include_body = false
    }
  }
  
  # API cache behavior
  ordered_cache_behavior {
    path_pattern     = "/api/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-${aws_alb.api.id}"
    
    forwarded_values {
      query_string = true
      headers      = ["*"]
      
      cookies {
        forward = "all"
      }
    }
    
    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
    compress               = true
    
    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.viewer_request.qualified_arn
      include_body = true
    }
  }
  
  # ML inference cache behavior
  ordered_cache_behavior {
    path_pattern     = "/ml/*"
    allowed_methods  = ["GET", "HEAD", "OPTIONS", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.origin.id}"
    
    forwarded_values {
      query_string = true
      headers      = ["Content-Type", "Accept"]
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 300
    max_ttl                = 3600
    compress               = false
    
    lambda_function_association {
      event_type   = "origin-request"
      lambda_arn   = aws_lambda_function.ml_inference.qualified_arn
      include_body = true
    }
  }
  
  # Geo restrictions
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  # SSL certificate
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.edge.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  # WAF
  web_acl_id = aws_wafv2_web_acl.edge.arn
  
  # Logging
  logging_config {
    include_cookies = false
    bucket          = aws_s3_bucket.cf_logs.bucket_domain_name
    prefix          = "cloudfront/"
  }
  
  tags = local.tags
}

# DynamoDB table for edge caching
resource "aws_dynamodb_table" "edge_cache" {
  name           = "edge-cache-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "cache_key"
  
  attribute {
    name = "cache_key"
    type = "S"
  }
  
  attribute {
    name = "ttl"
    type = "N"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  global_secondary_index {
    name            = "ttl-index"
    hash_key        = "ttl"
    projection_type = "ALL"
  }
  
  stream_enabled   = true
  stream_view_type = "NEW_AND_OLD_IMAGES"
  
  point_in_time_recovery {
    enabled = true
  }
  
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.edge_encryption.arn
  }
  
  tags = local.tags
}

# S3 buckets for edge storage
resource "aws_s3_bucket" "edge_storage" {
  bucket = "edge-storage-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = local.tags
}

resource "aws_s3_bucket" "ml_models" {
  bucket = "ml-models-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = local.tags
}

resource "aws_s3_bucket" "origin" {
  bucket = "edge-origin-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = local.tags
}

resource "aws_s3_bucket" "cf_logs" {
  bucket = "cf-logs-${var.environment}-${random_id.bucket_suffix.hex}"
  
  tags = local.tags
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "edge_storage" {
  bucket = aws_s3_bucket.edge_storage.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.edge_encryption.arn
    }
  }
}

# KMS key for encryption
resource "aws_kms_key" "edge_encryption" {
  description             = "Edge encryption key for ${var.environment}"
  deletion_window_in_days = 10
  enable_key_rotation     = true
  
  tags = local.tags
}

# CloudFront Origin Access Identity
resource "aws_cloudfront_origin_access_identity" "oai" {
  comment = "OAI for ${var.environment} edge distribution"
}

# S3 bucket policy for CloudFront
resource "aws_s3_bucket_policy" "origin" {
  bucket = aws_s3_bucket.origin.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudFrontAccess"
        Effect = "Allow"
        Principal = {
          AWS = aws_cloudfront_origin_access_identity.oai.iam_arn
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.origin.arn}/*"
      }
    ]
  })
}

# Random ID for bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Archive files for Lambda functions
data "archive_file" "viewer_request" {
  type        = "zip"
  source_dir  = "${path.module}/functions/viewer-request"
  output_path = "${path.module}/dist/viewer-request.zip"
}

data "archive_file" "origin_request" {
  type        = "zip"
  source_dir  = "${path.module}/functions/origin-request"
  output_path = "${path.module}/dist/origin-request.zip"
}

data "archive_file" "viewer_response" {
  type        = "zip"
  source_dir  = "${path.module}/functions/viewer-response"
  output_path = "${path.module}/dist/viewer-response.zip"
}

data "archive_file" "origin_response" {
  type        = "zip"
  source_dir  = "${path.module}/functions/origin-response"
  output_path = "${path.module}/dist/origin-response.zip"
}

data "archive_file" "ml_inference" {
  type        = "zip"
  source_dir  = "${path.module}/functions/ml-inference"
  output_path = "${path.module}/dist/ml-inference.zip"
}

# Outputs
output "endpoints" {
  value = {
    cloudfront_domain = aws_cloudfront_distribution.edge.domain_name
    cloudfront_id     = aws_cloudfront_distribution.edge.id
    s3_bucket        = aws_s3_bucket.origin.id
  }
}

output "lambda_functions" {
  value = {
    viewer_request  = aws_lambda_function.viewer_request.arn
    origin_request  = aws_lambda_function.origin_request.arn
    viewer_response = aws_lambda_function.viewer_response.arn
    origin_response = aws_lambda_function.origin_response.arn
    ml_inference    = aws_lambda_function.ml_inference.arn
  }
}