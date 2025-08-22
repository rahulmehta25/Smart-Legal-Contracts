# AWS Infrastructure Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "eks_cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "aurora_cluster_endpoint" {
  description = "Aurora cluster endpoint"
  value       = module.aurora.cluster_endpoint
  sensitive   = true
}

output "aurora_reader_endpoint" {
  description = "Aurora reader endpoint"
  value       = module.aurora.reader_endpoint
  sensitive   = true
}

output "elasticache_cluster_address" {
  description = "ElastiCache cluster address"
  value       = module.elasticache.cache_nodes[0].address
  sensitive   = true
}

output "s3_bucket_arns" {
  description = "ARNs of S3 buckets"
  value       = module.s3_buckets.bucket_arns
}

output "cloudfront_distribution_domain_name" {
  description = "CloudFront distribution domain name"
  value       = module.cloudfront.distribution_domain_name
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.dns_name
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    eks_cluster     = "$216.00"  # EKS control plane
    eks_nodes       = "$372.60"  # 5 t3.large instances (3 on-demand + 2 spot)
    aurora_serverless = "$180.00" # Minimum 0.5 ACU to max 16 ACU
    elasticache     = "$291.60"  # 3 cache.r7g.large nodes
    load_balancer   = "$22.50"   # ALB
    data_transfer   = "$100.00"  # Estimated
    s3_storage      = "$23.00"   # ~1TB storage
    cloudwatch      = "$30.00"   # Logs and metrics
    total_estimated = "$1,235.70"
  }
}

output "cost_optimization_recommendations" {
  description = "Cost optimization recommendations"
  value = [
    "Use Spot instances for non-critical workloads (saving ~70%)",
    "Enable S3 Intelligent-Tiering for automatic cost optimization",
    "Use Aurora Serverless v2 for variable workloads",
    "Implement Reserved Instances for stable workloads (saving ~40%)",
    "Use AWS Savings Plans for compute resources (saving ~30%)",
    "Enable AWS Cost Anomaly Detection",
    "Use AWS Compute Optimizer recommendations",
    "Implement proper tagging for cost allocation"
  ]
}