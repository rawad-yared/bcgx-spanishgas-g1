"""Lambda handler — triggered by S3 PutObject on raw/ prefix."""

from __future__ import annotations

import json
import logging
import os
import uuid

import boto3

from src.pipelines.manifest import ManifestStore

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict, context) -> dict:
    """Handle S3 PutObject event, start Step Functions execution."""
    table_name = os.environ.get("DYNAMODB_MANIFEST_TABLE", "spanishgas-pipeline-manifest")
    sfn_arn = os.environ.get("STEP_FUNCTIONS_ARN", "")
    region = os.environ.get("AWS_REGION", "eu-west-1")

    manifest = ManifestStore(table_name, region)
    sfn_client = boto3.client("stepfunctions", region_name=region)

    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        file_key = record["s3"]["object"]["key"]
        run_id = str(uuid.uuid4())

        logger.info("Processing %s/%s run_id=%s", bucket, file_key, run_id)

        if manifest.check_processed(file_key):
            logger.info("Already processed: %s", file_key)
            results.append({"file_key": file_key, "status": "skipped"})
            continue

        if not manifest.mark_started(file_key, run_id):
            logger.info("Already in progress: %s", file_key)
            results.append({"file_key": file_key, "status": "in_progress"})
            continue

        if sfn_arn:
            sfn_client.start_execution(
                stateMachineArn=sfn_arn,
                name=f"run-{run_id}",
                input=json.dumps({
                    "bucket": bucket,
                    "file_key": file_key,
                    "run_id": run_id,
                }),
            )

        results.append({"file_key": file_key, "status": "started", "run_id": run_id})

    return {"statusCode": 200, "results": results}
