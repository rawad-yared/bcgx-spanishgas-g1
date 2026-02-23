variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "github_repo" {
  description = "GitHub repository in owner/repo format"
  type        = string
}

variable "s3_bucket_arn" {
  description = "S3 data bucket ARN"
  type        = string
}

variable "dynamodb_table_arn" {
  description = "DynamoDB manifest table ARN"
  type        = string
}

variable "ecr_arns" {
  description = "List of ECR repository ARNs"
  type        = list(string)
}

variable "lambda_function_arn" {
  description = "Lambda function ARN"
  type        = string
}

variable "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  type        = string
  default     = "*"
}

variable "ecs_service_arn" {
  description = "ECS service ARN"
  type        = string
  default     = "*"
}
