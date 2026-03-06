variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "lambda_role_arn" {
  type = string
}

variable "ecr_image_uri" {
  type = string
}

variable "memory_size" {
  type    = number
  default = 512
}

variable "timeout" {
  type    = number
  default = 60
}

variable "s3_bucket_name" {
  type = string
}

variable "s3_bucket_arn" {
  type = string
}

variable "dynamodb_table_name" {
  type = string
}

variable "step_functions_arn" {
  type = string
}
