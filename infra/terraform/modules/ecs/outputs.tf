output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.streamlit.name
}

output "alb_dns_name" {
  description = "ALB DNS name for the Streamlit dashboard"
  value       = aws_lb.streamlit.dns_name
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.streamlit.arn
}

output "target_group_arn" {
  description = "Target group ARN"
  value       = aws_lb_target_group.streamlit.arn
}
