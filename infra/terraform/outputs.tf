output "s3_bucket_name" {
  description = "Data bucket name"
  value       = module.s3.bucket_name
}

output "dynamodb_table_name" {
  description = "Pipeline manifest table name"
  value       = module.dynamodb.table_name
}

output "lambda_function_arn" {
  description = "Lambda trigger function ARN"
  value       = module.lambda.function_arn
}

output "state_machine_arn" {
  description = "Step Functions state machine ARN"
  value       = module.step_functions.state_machine_arn
}

output "ecr_lambda_repo_url" {
  description = "ECR repository URL for Lambda image"
  value       = module.ecr.lambda_repo_url
}

output "ecr_processing_repo_url" {
  description = "ECR repository URL for Processing image"
  value       = module.ecr.processing_repo_url
}

output "sns_topic_arn" {
  description = "SNS alert topic ARN"
  value       = module.monitoring.sns_topic_arn
}

output "model_package_group" {
  description = "SageMaker Model Package Group name"
  value       = module.sagemaker.model_package_group_name
}

output "ecr_streamlit_repo_url" {
  description = "ECR repository URL for Streamlit image"
  value       = module.ecr.streamlit_repo_url
}

output "streamlit_alb_dns_name" {
  description = "ALB DNS name for the Streamlit dashboard"
  value       = module.ecs.alb_dns_name
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "ECS Streamlit service name"
  value       = module.ecs.service_name
}

output "github_deploy_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC (set as AWS_DEPLOY_ROLE_ARN secret)"
  value       = module.github_oidc.deploy_role_arn
}
