resource "aws_lambda_function" "pipeline_trigger" {
  function_name = "${var.project_name}-${var.environment}-pipeline-trigger"
  role          = var.lambda_role_arn
  package_type  = "Image"
  image_uri     = var.ecr_image_uri
  memory_size   = var.memory_size
  timeout       = var.timeout

  environment {
    variables = {
      DYNAMODB_MANIFEST_TABLE = var.dynamodb_table_name
      STEP_FUNCTIONS_ARN      = var.step_functions_arn
      S3_BUCKET               = var.s3_bucket_name
      LOG_LEVEL               = "INFO"
    }
  }
}

resource "aws_lambda_permission" "s3_invoke" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pipeline_trigger.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.s3_bucket_arn
}

resource "aws_s3_bucket_notification" "raw_upload" {
  bucket = var.s3_bucket_name

  lambda_function {
    lambda_function_arn = aws_lambda_function.pipeline_trigger.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
  }

  depends_on = [aws_lambda_permission.s3_invoke]
}
