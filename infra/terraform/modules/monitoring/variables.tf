variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "alert_email" {
  type    = string
  default = ""
}

variable "lambda_function_name" {
  type = string
}

variable "state_machine_arn" {
  type = string
}
