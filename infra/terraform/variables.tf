variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "spanishgas"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "alert_email" {
  description = "Email address for SNS alert subscriptions"
  type        = string
  default     = ""
}

variable "lambda_memory_size" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 512
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
}

variable "processing_instance_type" {
  description = "SageMaker Processing instance type"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "training_instance_type" {
  description = "SageMaker Training instance type"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {}
}
