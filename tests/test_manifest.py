"""Tests for src.pipelines.manifest — DynamoDB manifest store."""

from __future__ import annotations

import boto3
from moto import mock_aws

from src.pipelines.manifest import ManifestStore


def _create_table():
    """Helper to create DynamoDB manifest table in moto mock."""
    boto3.resource("dynamodb", region_name="us-east-1").create_table(
        TableName="test-manifest",
        KeySchema=[{"AttributeName": "file_key", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "file_key", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


class TestManifestStore:
    def test_check_processed_new_key(self):
        with mock_aws():
            _create_table()
            store = ManifestStore("test-manifest", region="us-east-1")
            assert store.check_processed("raw/file1.csv") is False

    def test_mark_started_first_time(self):
        with mock_aws():
            _create_table()
            store = ManifestStore("test-manifest", region="us-east-1")
            assert store.mark_started("raw/file1.csv", "run-001") is True

    def test_mark_started_duplicate_fails(self):
        with mock_aws():
            _create_table()
            store = ManifestStore("test-manifest", region="us-east-1")
            store.mark_started("raw/file1.csv", "run-001")
            assert store.mark_started("raw/file1.csv", "run-002") is False

    def test_mark_completed_updates_status(self):
        with mock_aws():
            _create_table()
            store = ManifestStore("test-manifest", region="us-east-1")
            store.mark_started("raw/file1.csv", "run-001")
            store.mark_completed("raw/file1.csv", "run-001")
            assert store.check_processed("raw/file1.csv") is True

    def test_check_processed_started_not_completed(self):
        with mock_aws():
            _create_table()
            store = ManifestStore("test-manifest", region="us-east-1")
            store.mark_started("raw/file1.csv", "run-001")
            assert store.check_processed("raw/file1.csv") is False
