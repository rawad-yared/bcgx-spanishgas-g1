resource "aws_dynamodb_table" "manifest" {
  name         = "${var.project_name}-${var.environment}-pipeline-manifest"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "file_key"

  attribute {
    name = "file_key"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }
}
