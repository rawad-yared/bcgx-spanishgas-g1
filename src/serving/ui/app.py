"""Streamlit dashboard for at-risk customers and recommendation details."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Protocol


RUN_DATE_PATTERN = re.compile(r"run_date=(\d{4}-\d{2}-\d{2})")


class S3ClientProtocol(Protocol):
    """Minimal S3 client protocol used by this module."""

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...


def _parse_iso_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not _is_s3_uri(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket_and_key = uri[len("s3://") :]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI, missing bucket: {uri}")
    return bucket, key.lstrip("/")


def _join_location(root: str, relative_path: str) -> str:
    if _is_s3_uri(root):
        return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"
    return str(Path(root) / relative_path)


def _resolve_s3_client(
    paths: Iterable[str], s3_client: S3ClientProtocol | None = None
) -> S3ClientProtocol | None:
    if s3_client is not None:
        return s3_client
    if not any(_is_s3_uri(path) for path in paths):
        return None

    try:
        import boto3

        return boto3.client("s3")
    except ImportError:
        pass

    try:
        from botocore.session import Session

        return Session().create_client("s3")
    except Exception as exc:
        raise RuntimeError(
            "S3 UI data access requested but no S3 SDK is available. "
            "Install boto3 or use local paths."
        ) from exc


def _read_bytes(path: str, s3_client: S3ClientProtocol | None) -> bytes:
    if _is_s3_uri(path):
        if s3_client is None:
            raise RuntimeError("S3 path used without an S3 client.")
        bucket, key = _parse_s3_uri(path)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"Missing input file: {path}") from exc
        return response["Body"].read()

    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return local_path.read_bytes()


def _read_jsonl(path: str, s3_client: S3ClientProtocol | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _read_bytes(path, s3_client).decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def _extract_run_date(value: str) -> str | None:
    match = RUN_DATE_PATTERN.search(value)
    if not match:
        return None
    parsed = match.group(1)
    try:
        return _parse_iso_date(parsed)
    except ValueError:
        return None


def list_available_run_dates(
    gold_root: str,
    s3_client: S3ClientProtocol | None = None,
) -> list[str]:
    """List available run dates based on scoring partitions."""

    if _is_s3_uri(gold_root):
        resolved = _resolve_s3_client(paths=[gold_root], s3_client=s3_client)
        if resolved is None:
            return []
        bucket, key_prefix = _parse_s3_uri(gold_root)
        prefix = f"{key_prefix.rstrip('/')}/scoring/" if key_prefix else "scoring/"
        contents: list[dict[str, Any]] = []
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = resolved.list_objects_v2(**kwargs)
            contents.extend(response.get("Contents") or [])
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break
        run_dates = {
            run_date
            for item in contents
            for run_date in [_extract_run_date(str(item.get("Key", "")))]
            if run_date is not None
        }
        return sorted(run_dates)

    scoring_root = Path(gold_root) / "scoring"
    if not scoring_root.exists():
        return []
    run_dates: set[str] = set()
    for child in scoring_root.iterdir():
        if not child.is_dir():
            continue
        extracted = _extract_run_date(child.name)
        if extracted is not None:
            run_dates.add(extracted)
    return sorted(run_dates)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _risk_tier(churn_score: float) -> str:
    if churn_score >= 0.80:
        return "high"
    if churn_score >= 0.45:
        return "medium"
    return "low"


def _normalize_reason_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def load_dashboard_rows(
    gold_root: str,
    run_date: str | None = None,
    s3_client: S3ClientProtocol | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Load and merge latest scoring output with recommendation details."""

    available = list_available_run_dates(gold_root=gold_root, s3_client=s3_client)
    if not available:
        raise FileNotFoundError(f"No scoring output found under {gold_root}.")

    resolved_run_date = run_date or available[-1]
    _parse_iso_date(resolved_run_date)

    resolved_s3_client = _resolve_s3_client(
        paths=[gold_root],
        s3_client=s3_client,
    )

    scoring_path = _join_location(
        gold_root, f"scoring/run_date={resolved_run_date}/scores.jsonl"
    )
    recommendation_path = _join_location(
        gold_root, f"recommendations/run_date={resolved_run_date}/recommendations.jsonl"
    )
    scoring_rows = _read_jsonl(scoring_path, resolved_s3_client)
    try:
        recommendation_rows = _read_jsonl(recommendation_path, resolved_s3_client)
    except FileNotFoundError:
        recommendation_rows = []

    recommendations_by_customer = {
        str(row.get("customer_id") or "").strip(): row
        for row in recommendation_rows
        if str(row.get("customer_id") or "").strip()
    }

    merged_rows: list[dict[str, Any]] = []
    for row in scoring_rows:
        customer_id = str(row.get("customer_id") or "").strip()
        if not customer_id:
            continue

        recommendation = recommendations_by_customer.get(customer_id, {})
        churn_score = _to_float(row.get("churn_probability"))
        if churn_score is None:
            churn_score = _to_float(row.get("risk_score")) or 0.0

        action = str(recommendation.get("action") or "no_offer")
        timing_window = str(recommendation.get("timing_window") or "n/a")
        expected_margin_impact = (
            _to_float(recommendation.get("expected_margin_impact")) or 0.0
        )
        reason_codes = _normalize_reason_codes(recommendation.get("reason_codes"))
        if not reason_codes:
            reason_codes = _normalize_reason_codes(row.get("reason_codes"))

        merged_rows.append(
            {
                "customer_id": customer_id,
                "run_date": resolved_run_date,
                "asof_date": str(row.get("asof_date") or ""),
                "churn_score": round(churn_score, 6),
                "risk_tier": _risk_tier(churn_score),
                "segment": str(
                    recommendation.get("segment")
                    or row.get("segment")
                    or "unknown"
                ),
                "action": action,
                "timing_window": timing_window,
                "expected_margin_impact": round(expected_margin_impact, 6),
                "reason_codes": reason_codes,
            }
        )

    merged_rows.sort(key=lambda item: item["churn_score"], reverse=True)
    return resolved_run_date, merged_rows


def _render_dashboard() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="SpanishGas Retention Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("SpanishGas Retention Dashboard")
    st.caption(
        "Risk list and recommendation details from latest scoring output."
    )

    default_root = os.environ.get("SPANISHGAS_GOLD_ROOT", "data/gold/")
    gold_root = st.sidebar.text_input("Gold root", value=default_root)

    try:
        available_dates = list_available_run_dates(gold_root=gold_root)
    except Exception as exc:
        st.error(f"Unable to read scoring outputs: {exc}")
        return

    if not available_dates:
        st.warning(f"No scoring output found under `{gold_root}`.")
        return

    selected_run_date = st.sidebar.selectbox(
        "Run date",
        options=available_dates,
        index=len(available_dates) - 1,
    )
    top_n = st.sidebar.slider("Top at-risk customers", min_value=1, max_value=200, value=25)

    try:
        resolved_run_date, rows = load_dashboard_rows(
            gold_root=gold_root,
            run_date=selected_run_date,
        )
    except Exception as exc:
        st.error(f"Unable to load dashboard data: {exc}")
        return

    if not rows:
        st.warning("Scoring file is present but contains no rows.")
        return

    offer_rows = sum(1 for row in rows if row["action"] != "no_offer")
    avg_score = sum(row["churn_score"] for row in rows) / len(rows)
    top_rows = rows[: min(top_n, len(rows))]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run date", resolved_run_date)
    col2.metric("Scored customers", len(rows))
    col3.metric("Offer recommendations", offer_rows)
    col4.metric("Avg churn score", f"{avg_score:.3f}")

    st.subheader("Top At-Risk Customers")
    st.dataframe(
        [
            {
                "customer_id": row["customer_id"],
                "churn_score": row["churn_score"],
                "risk_tier": row["risk_tier"],
                "segment": row["segment"],
                "action": row["action"],
                "timing_window": row["timing_window"],
                "expected_margin_impact": row["expected_margin_impact"],
            }
            for row in top_rows
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Recommendation Details")
    selected_customer = st.selectbox(
        "Customer",
        options=[row["customer_id"] for row in top_rows],
        index=0,
    )
    selected = next(
        row for row in rows if row["customer_id"] == selected_customer
    )
    st.json(selected)


def main() -> None:
    _render_dashboard()


if __name__ == "__main__":
    main()
