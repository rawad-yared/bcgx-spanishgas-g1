output "function_arn" {
  value = aws_lambda_function.pipeline_trigger.arn
}

output "function_name" {
  value = aws_lambda_function.pipeline_trigger.function_name
}
