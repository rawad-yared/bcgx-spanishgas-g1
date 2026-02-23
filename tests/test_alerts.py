"""Tests for src.monitoring.alerts — SNS and CloudWatch alert publishing."""

from __future__ import annotations

import boto3
from moto import mock_aws

from src.monitoring.alerts import publish_cloudwatch_metrics, publish_sns_alert

REGION = "us-east-1"


class TestPublishSnsAlert:
    def test_publish_returns_message_id(self):
        with mock_aws():
            sns = boto3.client("sns", region_name=REGION)
            topic = sns.create_topic(Name="test-alerts")
            topic_arn = topic["TopicArn"]

            response = publish_sns_alert(topic_arn, "Test Alert", "Body text", region=REGION)
            assert "MessageId" in response

    def test_long_subject_truncated(self):
        with mock_aws():
            sns = boto3.client("sns", region_name=REGION)
            topic = sns.create_topic(Name="test-alerts")
            topic_arn = topic["TopicArn"]

            long_subject = "A" * 200
            response = publish_sns_alert(topic_arn, long_subject, "Body", region=REGION)
            assert "MessageId" in response


class TestPublishCloudwatchMetrics:
    def test_publish_single_metric(self):
        with mock_aws():
            metrics = [{"MetricName": "DriftDetected", "Value": 1.0, "Unit": "Count"}]
            # Should not raise
            publish_cloudwatch_metrics("SpanishGas/MLOps", metrics, region=REGION)

    def test_publish_multiple_metrics(self):
        with mock_aws():
            metrics = [
                {"MetricName": "PRAUC", "Value": 0.82},
                {"MetricName": "ROCAUC", "Value": 0.91},
                {"MetricName": "Precision", "Value": 0.75},
            ]
            publish_cloudwatch_metrics("SpanishGas/MLOps", metrics, region=REGION)

    def test_publish_with_dimensions(self):
        with mock_aws():
            metrics = [{
                "MetricName": "ModelLatency",
                "Value": 120.5,
                "Unit": "Milliseconds",
                "Dimensions": [{"Name": "ModelName", "Value": "xgboost"}],
            }]
            publish_cloudwatch_metrics("SpanishGas/MLOps", metrics, region=REGION)
