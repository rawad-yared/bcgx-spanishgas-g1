output "lambda_role_arn" {
  value = aws_iam_role.lambda.arn
}

output "sfn_role_arn" {
  value = aws_iam_role.sfn.arn
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}
