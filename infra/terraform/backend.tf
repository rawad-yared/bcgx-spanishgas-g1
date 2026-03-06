terraform {
  backend "s3" {
    bucket         = "spanishgas-terraform-state"
    key            = "infra/terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "spanishgas-terraform-locks"
    encrypt        = true
  }
}
