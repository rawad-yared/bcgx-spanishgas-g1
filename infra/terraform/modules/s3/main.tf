resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-bronze"
    status = "Enabled"
    filter {
      prefix = "bronze/"
    }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "expire-scored"
    status = "Enabled"
    filter {
      prefix = "scored/"
    }
    expiration {
      days = 365
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
