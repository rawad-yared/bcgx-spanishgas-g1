resource "aws_sagemaker_model_package_group" "churn" {
  model_package_group_name        = "${var.project_name}-${var.environment}-churn"
  model_package_group_description = "SpanishGas churn prediction models"
}
