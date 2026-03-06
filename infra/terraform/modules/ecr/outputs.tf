output "lambda_repo_url" {
  value = aws_ecr_repository.lambda.repository_url
}

output "lambda_repo_arn" {
  value = aws_ecr_repository.lambda.arn
}

output "processing_repo_url" {
  value = aws_ecr_repository.processing.repository_url
}

output "processing_repo_arn" {
  value = aws_ecr_repository.processing.arn
}

output "streamlit_repo_url" {
  value = aws_ecr_repository.streamlit.repository_url
}

output "streamlit_repo_arn" {
  value = aws_ecr_repository.streamlit.arn
}
