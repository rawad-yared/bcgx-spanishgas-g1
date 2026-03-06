variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "s3_bucket_arn" {
  type = string
}

variable "dynamodb_table_arn" {
  type = string
}

variable "ecr_lambda_arn" {
  type = string
}

variable "ecr_processing_arn" {
  type = string
}

variable "step_functions_arn" {
  type = string
}

variable "sns_topic_arn" {
  type = string
}

variable "model_package_group" {
  type = string
}
