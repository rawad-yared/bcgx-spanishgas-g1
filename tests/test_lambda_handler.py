"""Tests for src.pipelines.lambda_handler — S3 trigger Lambda."""

from __future__ import annotations

import json

import boto3
from moto import mock_aws


def _make_s3_event(bucket: str, key: str) -> dict:
    return {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key},
                }
            }
        ]
    }


class TestLambdaHandler:
    def test_new_file_starts_execution(self, monkeypatch):
        with mock_aws():
            region = "us-east-1"
            # Create DynamoDB table
            ddb = boto3.resource("dynamodb", region_name=region)
            ddb.create_table(
                TableName="test-manifest",
                KeySchema=[{"AttributeName": "file_key", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "file_key", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )

            # Create Step Functions state machine
            sfn = boto3.client("stepfunctions", region_name=region)
            iam = boto3.client("iam", region_name=region)
            role = iam.create_role(
                RoleName="sfn-role",
                AssumeRolePolicyDocument="{}",
                Path="/",
            )
            sm = sfn.create_state_machine(
                name="test-sm",
                definition=json.dumps({"StartAt": "Pass", "States": {"Pass": {"Type": "Succeed"}}}),
                roleArn=role["Role"]["Arn"],
            )

            monkeypatch.setenv("DYNAMODB_MANIFEST_TABLE", "test-manifest")
            monkeypatch.setenv("STEP_FUNCTIONS_ARN", sm["stateMachineArn"])
            monkeypatch.setenv("AWS_REGION", region)
            monkeypatch.setenv("AWS_DEFAULT_REGION", region)

            from src.pipelines.lambda_handler import handler

            event = _make_s3_event("my-bucket", "raw/data.csv")
            result = handler(event, None)

            assert result["statusCode"] == 200
            assert result["results"][0]["status"] == "started"

    def test_duplicate_file_skipped(self, monkeypatch):
        with mock_aws():
            region = "us-east-1"
            ddb = boto3.resource("dynamodb", region_name=region)
            table = ddb.create_table(
                TableName="test-manifest",
                KeySchema=[{"AttributeName": "file_key", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "file_key", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
            # Pre-insert completed item
            table.put_item(Item={"file_key": "raw/data.csv", "status": "completed"})

            monkeypatch.setenv("DYNAMODB_MANIFEST_TABLE", "test-manifest")
            monkeypatch.setenv("STEP_FUNCTIONS_ARN", "")
            monkeypatch.setenv("AWS_REGION", region)
            monkeypatch.setenv("AWS_DEFAULT_REGION", region)

            from src.pipelines.lambda_handler import handler

            event = _make_s3_event("my-bucket", "raw/data.csv")
            result = handler(event, None)

            assert result["results"][0]["status"] == "skipped"

    def test_empty_event(self, monkeypatch):
        with mock_aws():
            region = "us-east-1"
            ddb = boto3.resource("dynamodb", region_name=region)
            ddb.create_table(
                TableName="test-manifest",
                KeySchema=[{"AttributeName": "file_key", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "file_key", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )

            monkeypatch.setenv("DYNAMODB_MANIFEST_TABLE", "test-manifest")
            monkeypatch.setenv("STEP_FUNCTIONS_ARN", "")
            monkeypatch.setenv("AWS_REGION", region)
            monkeypatch.setenv("AWS_DEFAULT_REGION", region)

            from src.pipelines.lambda_handler import handler

            result = handler({"Records": []}, None)
            assert result["statusCode"] == 200
            assert result["results"] == []
