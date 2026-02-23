locals {
  prefix = "${var.project_name}-${var.environment}"
}

# --- OIDC Identity Provider ---
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# --- Deploy Role ---
resource "aws_iam_role" "github_deploy" {
  name = "${local.prefix}-github-deploy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_deploy" {
  name = "${local.prefix}-github-deploy-policy"
  role = aws_iam_role.github_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Terraform state access
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject",
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-terraform-state",
          "arn:aws:s3:::${var.project_name}-terraform-state/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem",
        ]
        Resource = "arn:aws:dynamodb:*:*:table/${var.project_name}-terraform-locks"
      },
      # ECR — push images
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = var.ecr_arns
      },
      # Lambda — update function code
      {
        Effect   = "Allow"
        Action   = ["lambda:UpdateFunctionCode", "lambda:GetFunction"]
        Resource = var.lambda_function_arn
      },
      # ECS — force new deployment
      {
        Effect = "Allow"
        Action = [
          "ecs:UpdateService",
          "ecs:DescribeServices",
          "ecs:DescribeClusters",
        ]
        Resource = "*"
      },
      # Step Functions — start execution (for retrain workflow)
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution",
          "states:DescribeExecution",
          "states:ListStateMachines",
        ]
        Resource = "*"
      },
      # Terraform needs broad read + resource management
      # Scoped to project-prefixed resources where possible
      {
        Effect = "Allow"
        Action = [
          "iam:GetRole", "iam:GetRolePolicy", "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies", "iam:GetPolicy", "iam:GetPolicyVersion",
          "iam:CreateRole", "iam:PutRolePolicy", "iam:AttachRolePolicy",
          "iam:PassRole", "iam:TagRole",
          "iam:CreateOpenIDConnectProvider", "iam:GetOpenIDConnectProvider",
          "iam:TagOpenIDConnectProvider",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:CreateBucket", "s3:GetBucketPolicy", "s3:GetBucketAcl",
          "s3:GetBucketVersioning", "s3:GetBucketLogging",
          "s3:GetEncryptionConfiguration", "s3:GetLifecycleConfiguration",
          "s3:GetBucketTagging", "s3:PutBucketPolicy", "s3:PutBucketVersioning",
          "s3:PutEncryptionConfiguration", "s3:PutLifecycleConfiguration",
          "s3:PutBucketTagging", "s3:PutBucketNotification",
          "s3:GetBucketNotification",
        ]
        Resource = [var.s3_bucket_arn, "${var.s3_bucket_arn}/*"]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:CreateTable", "dynamodb:DescribeTable",
          "dynamodb:UpdateTable", "dynamodb:TagResource",
          "dynamodb:DescribeContinuousBackups",
          "dynamodb:DescribeTimeToLive", "dynamodb:ListTagsOfResource",
        ]
        Resource = var.dynamodb_table_arn
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:CreateFunction", "lambda:GetFunction",
          "lambda:UpdateFunctionConfiguration", "lambda:AddPermission",
          "lambda:GetPolicy", "lambda:TagResource",
          "lambda:ListVersionsByFunction", "lambda:GetFunctionCodeSigningConfig",
        ]
        Resource = var.lambda_function_arn
      },
      {
        Effect = "Allow"
        Action = [
          "states:CreateStateMachine", "states:DescribeStateMachine",
          "states:UpdateStateMachine", "states:TagResource",
          "states:ListTagsForResource",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModelPackageGroup", "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListTags",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:CreateTopic", "sns:GetTopicAttributes", "sns:SetTopicAttributes",
          "sns:Subscribe", "sns:TagResource", "sns:ListTagsForResource",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricAlarm", "cloudwatch:DescribeAlarms",
          "cloudwatch:TagResource", "cloudwatch:ListTagsForResource",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:CreateRepository", "ecr:DescribeRepositories",
          "ecr:GetLifecyclePolicy", "ecr:PutLifecyclePolicy",
          "ecr:ListTagsForResource", "ecr:TagResource",
          "ecr:GetRepositoryPolicy",
        ]
        Resource = var.ecr_arns
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeVpcs", "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups", "ec2:CreateSecurityGroup",
          "ec2:AuthorizeSecurityGroupIngress", "ec2:AuthorizeSecurityGroupEgress",
          "ec2:RevokeSecurityGroupIngress", "ec2:RevokeSecurityGroupEgress",
          "ec2:CreateTags", "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeAccountAttributes",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecs:CreateCluster", "ecs:DescribeClusters",
          "ecs:RegisterTaskDefinition", "ecs:DescribeTaskDefinition",
          "ecs:DeregisterTaskDefinition",
          "ecs:CreateService", "ecs:UpdateService", "ecs:DescribeServices",
          "ecs:TagResource", "ecs:ListTagsForResource",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "elasticloadbalancing:CreateLoadBalancer", "elasticloadbalancing:DescribeLoadBalancers",
          "elasticloadbalancing:DescribeLoadBalancerAttributes",
          "elasticloadbalancing:ModifyLoadBalancerAttributes",
          "elasticloadbalancing:CreateTargetGroup", "elasticloadbalancing:DescribeTargetGroups",
          "elasticloadbalancing:DescribeTargetGroupAttributes",
          "elasticloadbalancing:ModifyTargetGroupAttributes",
          "elasticloadbalancing:CreateListener", "elasticloadbalancing:DescribeListeners",
          "elasticloadbalancing:AddTags", "elasticloadbalancing:DescribeTags",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup", "logs:DescribeLogGroups",
          "logs:PutRetentionPolicy", "logs:TagResource",
          "logs:ListTagsForResource",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutRule", "events:DescribeRule",
          "events:PutTargets", "events:ListTagsForResource",
        ]
        Resource = "*"
      },
    ]
  })
}
