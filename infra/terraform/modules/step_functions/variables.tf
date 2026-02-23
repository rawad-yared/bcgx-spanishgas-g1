variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "sfn_role_arn" {
  type = string
}

variable "s3_bucket_name" {
  type = string
}

variable "processing_image_uri" {
  type = string
}

variable "sagemaker_role_arn" {
  type = string
}

variable "processing_instance" {
  type    = string
  default = "ml.m5.xlarge"
}

variable "training_instance" {
  type    = string
  default = "ml.m5.xlarge"
}

variable "dynamodb_table_name" {
  type = string
}

variable "sns_topic_arn" {
  type = string
}
