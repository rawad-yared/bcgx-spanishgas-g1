"""SageMaker Model Registry wrapper for champion/challenger management."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manage model versions in SageMaker Model Registry."""

    def __init__(self, model_package_group: str, region: str = "eu-west-1"):
        self.sm_client = boto3.client("sagemaker", region_name=region)
        self.group = model_package_group

    def register_model(
        self,
        model_url: str,
        metrics: dict,
        description: str = "",
    ) -> str:
        """Register a new model version as PendingManualApproval.

        Returns the model package ARN.
        """
        response = self.sm_client.create_model_package(
            ModelPackageGroupName=self.group,
            ModelPackageDescription=description or f"Registered at {datetime.now(timezone.utc).isoformat()}",
            InferenceSpecification={
                "Containers": [{"Image": "placeholder", "ModelDataUrl": model_url}],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            ModelApprovalStatus="PendingManualApproval",
            CustomerMetadataProperties={
                k: str(v) for k, v in metrics.items()
            },
        )
        arn = response["ModelPackageArn"]
        logger.info("Registered model: %s", arn)
        return arn

    def approve_model(self, model_package_arn: str) -> None:
        """Set model status to Approved (promote to champion)."""
        self.sm_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved",
        )
        logger.info("Approved model: %s", model_package_arn)

    def reject_model(self, model_package_arn: str) -> None:
        """Set model status to Rejected."""
        self.sm_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Rejected",
        )
        logger.info("Rejected model: %s", model_package_arn)

    def get_champion_model(self) -> dict | None:
        """Get latest Approved model package ARN + metadata."""
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=self.group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            return None

        arn = packages[0]["ModelPackageArn"]
        detail = self.sm_client.describe_model_package(ModelPackageName=arn)
        return {
            "arn": arn,
            "status": detail.get("ModelApprovalStatus"),
            "created": str(detail.get("CreationTime")),
            "metrics": detail.get("CustomerMetadataProperties", {}),
        }

    def list_models(self, max_results: int = 10) -> list[dict]:
        """List recent model versions."""
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=self.group,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=max_results,
        )
        return [
            {
                "arn": p["ModelPackageArn"],
                "status": p.get("ModelApprovalStatus", "Unknown"),
                "created": str(p.get("CreationTime")),
            }
            for p in response.get("ModelPackageSummaryList", [])
        ]
