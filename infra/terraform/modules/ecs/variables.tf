variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "vpc_id" {
  description = "VPC ID for ALB and ECS"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ALB and ECS tasks"
  type        = list(string)
}

variable "alb_security_group_id" {
  description = "Security group ID for the ALB"
  type        = string
}

variable "ecs_security_group_id" {
  description = "Security group ID for ECS tasks"
  type        = string
}

variable "streamlit_image_uri" {
  description = "ECR image URI for the Streamlit container"
  type        = string
}

variable "task_execution_role_arn" {
  description = "IAM role ARN for ECS task execution (ECR pull, CloudWatch logs)"
  type        = string
}

variable "task_role_arn" {
  description = "IAM role ARN for the ECS task (S3, DynamoDB access)"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 data bucket name (passed as env var to container)"
  type        = string
}

variable "dynamodb_table_name" {
  description = "DynamoDB manifest table name (passed as env var to container)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "cpu" {
  description = "Fargate task CPU units"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of Streamlit tasks to run"
  type        = number
  default     = 1
}
