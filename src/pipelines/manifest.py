"""DynamoDB manifest store for pipeline idempotency."""

from __future__ import annotations

from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


class ManifestStore:
    """Track processed files in DynamoDB for idempotent pipeline runs."""

    def __init__(self, table_name: str, region: str = "eu-west-1"):
        self.table_name = table_name
        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def check_processed(self, file_key: str) -> bool:
        resp = self.table.get_item(Key={"file_key": file_key})
        item = resp.get("Item")
        return item is not None and item.get("status") == "completed"

    def mark_started(self, file_key: str, run_id: str) -> bool:
        try:
            self.table.put_item(
                Item={
                    "file_key": file_key,
                    "run_id": run_id,
                    "status": "started",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                ConditionExpression="attribute_not_exists(file_key)",
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def mark_completed(self, file_key: str, run_id: str) -> None:
        self.table.update_item(
            Key={"file_key": file_key},
            UpdateExpression="SET #s = :s, completed_at = :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "completed",
                ":t": datetime.now(timezone.utc).isoformat(),
                ":r": run_id,
            },
            ConditionExpression="run_id = :r",
        )
