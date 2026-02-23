terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(var.tags, {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    })
  }
}

locals {
  prefix = "${var.project_name}-${var.environment}"
}

# --- ECR ---
module "ecr" {
  source       = "./modules/ecr"
  project_name = var.project_name
  environment  = var.environment
}

# --- S3 ---
module "s3" {
  source       = "./modules/s3"
  project_name = var.project_name
  environment  = var.environment
}

# --- DynamoDB ---
module "dynamodb" {
  source       = "./modules/dynamodb"
  project_name = var.project_name
  environment  = var.environment
}

# --- IAM ---
module "iam" {
  source              = "./modules/iam"
  project_name        = var.project_name
  environment         = var.environment
  s3_bucket_arn       = module.s3.bucket_arn
  dynamodb_table_arn  = module.dynamodb.table_arn
  ecr_lambda_arn      = module.ecr.lambda_repo_arn
  ecr_processing_arn  = module.ecr.processing_repo_arn
  step_functions_arn  = module.step_functions.state_machine_arn
  sns_topic_arn       = module.monitoring.sns_topic_arn
  model_package_group = module.sagemaker.model_package_group_name
}

# --- Lambda ---
module "lambda" {
  source              = "./modules/lambda"
  project_name        = var.project_name
  environment         = var.environment
  lambda_role_arn     = module.iam.lambda_role_arn
  ecr_image_uri       = "${module.ecr.lambda_repo_url}:latest"
  memory_size         = var.lambda_memory_size
  timeout             = var.lambda_timeout
  s3_bucket_name      = module.s3.bucket_name
  s3_bucket_arn       = module.s3.bucket_arn
  dynamodb_table_name = module.dynamodb.table_name
  step_functions_arn  = module.step_functions.state_machine_arn
}

# --- Step Functions ---
module "step_functions" {
  source               = "./modules/step_functions"
  project_name         = var.project_name
  environment          = var.environment
  sfn_role_arn         = module.iam.sfn_role_arn
  s3_bucket_name       = module.s3.bucket_name
  processing_image_uri = "${module.ecr.processing_repo_url}:latest"
  sagemaker_role_arn   = module.iam.sagemaker_role_arn
  processing_instance  = var.processing_instance_type
  training_instance    = var.training_instance_type
  dynamodb_table_name  = module.dynamodb.table_name
  sns_topic_arn        = module.monitoring.sns_topic_arn
}

# --- SageMaker ---
module "sagemaker" {
  source       = "./modules/sagemaker"
  project_name = var.project_name
  environment  = var.environment
}

# --- Monitoring ---
module "monitoring" {
  source             = "./modules/monitoring"
  project_name       = var.project_name
  environment        = var.environment
  alert_email        = var.alert_email
  lambda_function_name = module.lambda.function_name
  state_machine_arn  = module.step_functions.state_machine_arn
}
