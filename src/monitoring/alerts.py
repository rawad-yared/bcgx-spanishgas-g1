"""Alert publishing to SNS and CloudWatch."""

from __future__ import annotations

import logging

import boto3

logger = logging.getLogger(__name__)


def publish_sns_alert(
    topic_arn: str,
    subject: str,
    message: str,
    region: str = "eu-west-1",
) -> dict:
    """Publish alert message to SNS topic."""
    client = boto3.client("sns", region_name=region)
    response = client.publish(
        TopicArn=topic_arn,
        Subject=subject[:100],  # SNS subject max 100 chars
        Message=message,
    )
    logger.info("Published SNS alert: %s (MessageId=%s)", subject, response.get("MessageId"))
    return response


def publish_cloudwatch_metrics(
    namespace: str,
    metrics: list[dict],
    region: str = "eu-west-1",
) -> None:
    """Publish custom metrics to CloudWatch.

    Each metric dict should have: MetricName, Value, Unit (optional).
    """
    client = boto3.client("cloudwatch", region_name=region)

    metric_data = []
    for m in metrics:
        datum = {
            "MetricName": m["MetricName"],
            "Value": float(m["Value"]),
            "Unit": m.get("Unit", "None"),
        }
        if "Dimensions" in m:
            datum["Dimensions"] = m["Dimensions"]
        metric_data.append(datum)

    # CloudWatch accepts max 1000 metrics per call
    for i in range(0, len(metric_data), 1000):
        batch = metric_data[i : i + 1000]
        client.put_metric_data(Namespace=namespace, MetricData=batch)

    logger.info("Published %d CloudWatch metrics to %s", len(metric_data), namespace)
