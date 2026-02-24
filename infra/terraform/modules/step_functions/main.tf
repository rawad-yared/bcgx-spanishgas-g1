resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project_name}-${var.environment}-pipeline"
  role_arn = var.sfn_role_arn

  definition = templatefile("${path.module}/asl/pipeline.asl.json", {
    processing_image_uri = var.processing_image_uri
    sagemaker_role_arn   = var.sagemaker_role_arn
    s3_bucket            = var.s3_bucket_name
    processing_instance       = var.processing_instance
    processing_instance_large = var.processing_instance_large
    training_instance         = var.training_instance
    dynamodb_table       = var.dynamodb_table_name
    sns_topic_arn        = var.sns_topic_arn
  })
}
