output "table_name" {
  value = aws_dynamodb_table.manifest.name
}

output "table_arn" {
  value = aws_dynamodb_table.manifest.arn
}
