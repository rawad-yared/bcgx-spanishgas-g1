locals {
  prefix = "${var.project_name}-${var.environment}"
}

# --- ECS Cluster ---
resource "aws_ecs_cluster" "main" {
  name = "${local.prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# --- CloudWatch Log Group ---
resource "aws_cloudwatch_log_group" "streamlit" {
  name              = "/ecs/${local.prefix}-streamlit"
  retention_in_days = 30
}

# --- ECS Task Definition ---
resource "aws_ecs_task_definition" "streamlit" {
  family                   = "${local.prefix}-streamlit"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "streamlit"
      image     = var.streamlit_image_uri
      essential = true

      portMappings = [
        {
          containerPort = 8501
          hostPort      = 8501
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "DATA_SOURCE", value = "s3" },
        { name = "S3_BUCKET", value = var.s3_bucket_name },
        { name = "AWS_REGION", value = var.aws_region },
        { name = "DYNAMODB_MANIFEST_TABLE", value = var.dynamodb_table_name },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.streamlit.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "streamlit"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])
}

# --- Application Load Balancer ---
resource "aws_lb" "streamlit" {
  name               = "${local.prefix}-streamlit-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.alb_security_group_id]
  subnets            = var.subnet_ids
}

resource "aws_lb_target_group" "streamlit" {
  name        = "${local.prefix}-streamlit-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/_stcore/health"
    protocol            = "HTTP"
    port                = "traffic-port"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.streamlit.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.streamlit.arn
  }
}

# --- ECS Service ---
resource "aws_ecs_service" "streamlit" {
  name            = "${local.prefix}-streamlit"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.streamlit.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.ecs_security_group_id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.streamlit.arn
    container_name   = "streamlit"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener.http]
}
