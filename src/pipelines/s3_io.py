"""S3 I/O helpers — read/write parquet, CSV, JSON via boto3 + pyarrow."""

from __future__ import annotations

import io
import json

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_s3_client(region: str = "eu-west-1"):
    return boto3.client("s3", region_name=region)


def read_parquet(bucket: str, key: str, region: str = "eu-west-1") -> pd.DataFrame:
    s3 = get_s3_client(region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def write_parquet(df: pd.DataFrame, bucket: str, key: str, region: str = "eu-west-1") -> None:
    s3 = get_s3_client(region)
    buf = io.BytesIO()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, buf)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def read_csv(bucket: str, key: str, region: str = "eu-west-1") -> pd.DataFrame:
    s3 = get_s3_client(region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def read_json_s3(bucket: str, key: str, region: str = "eu-west-1") -> dict:
    s3 = get_s3_client(region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_json(data: dict, bucket: str, key: str, region: str = "eu-west-1") -> None:
    s3 = get_s3_client(region)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode("utf-8"))
