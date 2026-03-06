"""Tests for src.pipelines.s3_io — S3 read/write helpers."""

from __future__ import annotations

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.pipelines.s3_io import read_csv, read_json_s3, read_parquet, write_json, write_parquet

BUCKET = "test-bucket"
REGION = "us-east-1"


@pytest.fixture
def s3_bucket():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestParquetRoundTrip:
    def test_write_read_parquet(self):
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            write_parquet(df, BUCKET, "test/data.parquet", region=REGION)
            result = read_parquet(BUCKET, "test/data.parquet", region=REGION)
            pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            df = pd.DataFrame({"a": pd.Series(dtype="int64")})
            write_parquet(df, BUCKET, "test/empty.parquet", region=REGION)
            result = read_parquet(BUCKET, "test/empty.parquet", region=REGION)
            assert len(result) == 0


class TestJsonRoundTrip:
    def test_write_read_json(self):
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            data = {"metric": 0.85, "model": "xgboost"}
            write_json(data, BUCKET, "test/metrics.json", region=REGION)
            result = read_json_s3(BUCKET, "test/metrics.json", region=REGION)
            assert result == data

    def test_nested_json(self):
        with mock_aws():
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
            data = {"metrics": {"pr_auc": 0.75}, "features": ["a", "b"]}
            write_json(data, BUCKET, "test/nested.json", region=REGION)
            result = read_json_s3(BUCKET, "test/nested.json", region=REGION)
            assert result["metrics"]["pr_auc"] == 0.75


class TestCsvRead:
    def test_read_csv(self):
        with mock_aws():
            client = boto3.client("s3", region_name=REGION)
            client.create_bucket(Bucket=BUCKET)
            csv_body = "col1,col2\n1,a\n2,b\n"
            client.put_object(Bucket=BUCKET, Key="test/data.csv", Body=csv_body.encode())
            result = read_csv(BUCKET, "test/data.csv", region=REGION)
            assert list(result.columns) == ["col1", "col2"]
            assert len(result) == 2
