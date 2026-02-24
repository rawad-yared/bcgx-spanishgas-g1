"""NLP enrichment for customer interactions — intent classification + sentiment."""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Intent classification (regex, matches notebook exactly) ──────────────────

INTENT_MAP: dict[str, list[str]] = {
    "Cancellation / Switch": [
        r"\bcancel(l?ed|lation)?\b",
        r"\bterminate\b",
        r"\bswitch(ing)?\b",
        r"\bleave\b",
        r"\baccount clos(e|ure)\b",
        r"\bclosing account\b",
    ],
    "Complaint / Escalation": [
        r"\bcomplain(t|ing)?\b",
        r"\bfrustrat(ed|ion)\b",
        r"\bescalat(ed|ion)\b",
        r"\bnot satisfied\b",
        r"\bunsatisfied\b",
        r"\bupset\b",
        r"\bdissatisf(ied|action)\b",
        r"\bunhappy\b",
        r"\bdiscontent\b",
    ],
    "Billing / Payment": [
        r"\bbill(ing)?\b",
        r"\bcharge(s|d)?\b",
        r"\bpayment\b",
        r"\boverdue\b",
        r"\bpast due\b",
    ],
    "Contract Renewal": [
        r"\brenew(al|als|ing)?\b",
        r"\bexpir(e|y|ation)\b",
        r"\bnext renewal\b",
        r"\brenewals discussed\b",
        r"\brenewed\b",
        r"\brenewed contract\b",
    ],
    "Pricing Offers": [
        r"\bprice(s|d)?\b",
        r"\brate(s)?\b",
        r"\bpricing\b",
        r"\bincrease\b",
        r"\bhike\b",
        r"\bdiscount(s)?\b",
        r"\bsavings?\b",
        r"\bcompetitive\b",
        r"\bbetter (deal|rate|offer)\b",
        r"\bcompetitiveness\b",
        r"\bcompetition\b",
        r"\balternatives?\b",
        r"\bseeking alternatives?\b",
        r"\blooking around\b",
    ],
    "Plan / Product Inquiry": [
        r"\bplan(s)?\b",
        r"\bplan options?\b",
        r"\bnew plan\b",
        r"\bfuture plans?\b",
        r"\boptions?\b",
        r"\bexploring options?\b",
    ],
    "Account / Service Inquiry": [
        r"\baccount details?\b",
        r"\baccount questions?\b",
        r"\bservices?\b",
        r"\binquir(y|ed|ies)\b",
        r"\binfo\b",
        r"\bclarified\b",
        r"\bprovided info\b",
        r"\breviewed options\b",
        r"\bdiscussed options\b",
        r"\baccount update(s)?\b",
        r"\baccount setup\b",
        r"\bsetup details?\b",
        r"\baccount issues?\b",
        r"\baccount\b.*\bno issues\b",
        r"\bno issues found\b",
        r"\ball good\b",
    ],
    "General / Operational Contact": [
        r"\bgeneral\b",
        r"\bfollow[- ]?up\b",
        r"\broutine\b",
        r"\binbound call\b",
        r"\bno action required\b",
        r"\bissue resolved\b",
        r"\bresolved\b",
        r"\bnormal process\b",
        r"\bstandard interaction\b",
        r"\bcustomer contacted\b",
    ],
}

# Priority order — first match wins
INTENT_PRIORITY: list[str] = list(INTENT_MAP.keys())

# Pre-compile patterns for speed
INTENT_PATTERNS: dict[str, re.Pattern] = {
    label: re.compile("|".join(patterns), flags=re.IGNORECASE)
    for label, patterns in INTENT_MAP.items()
}


def classify_intent(text: str | None) -> str:
    """Classify interaction text into one of 8 intent categories (priority order).

    Returns "Other / Unclassified" if no pattern matches.
    """
    t = "" if text is None else str(text)
    for label in INTENT_PRIORITY:
        if INTENT_PATTERNS[label].search(t):
            return label
    return "Other / Unclassified"


def enrich_interactions_intent(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``customer_intent`` and ``has_interaction`` columns via regex classification.

    Parameters
    ----------
    df : pd.DataFrame
        Interactions dataframe with ``interaction_summary`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new columns added.
    """
    out = df.copy()

    if "interaction_summary" in out.columns:
        out["customer_intent"] = out["interaction_summary"].map(classify_intent).astype("category")
    else:
        logger.warning("No 'interaction_summary' column — skipping intent classification")

    # has_interaction: date present OR non-empty summary
    has_date = out["date"].notna() if "date" in out.columns else pd.Series(False, index=out.index)
    has_text = (
        out["interaction_summary"].fillna("").astype(str).str.strip().ne("")
        if "interaction_summary" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["has_interaction"] = (has_date | has_text).astype(int)

    return out


# ── Sentiment analysis (HuggingFace, guarded import) ────────────────────────

_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _scores_to_row(score_list: list[dict]) -> dict:
    """Convert HuggingFace top_k output to flat dict."""
    d = {x["label"].lower(): float(x["score"]) for x in score_list}
    neg = d.get("negative", np.nan)
    neu = d.get("neutral", np.nan)
    pos = d.get("positive", np.nan)
    label = max(
        [("negative", neg), ("neutral", neu), ("positive", pos)],
        key=lambda t: t[1],
    )[0]
    return {
        "sentiment_neg": neg,
        "sentiment_neu": neu,
        "sentiment_pos": pos,
        "sentiment_label": label,
    }


def enrich_interactions_sentiment(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Add sentiment columns via HuggingFace ``cardiffnlp/twitter-roberta-base-sentiment-latest``.

    If ``transformers`` is not installed, logs a warning and returns *df* unchanged.
    """
    try:
        from transformers import pipeline as hf_pipeline  # noqa: WPS433
    except ImportError:
        logger.warning("transformers not installed — skipping sentiment enrichment")
        return df

    if "interaction_summary" not in df.columns:
        logger.warning("No 'interaction_summary' column — skipping sentiment enrichment")
        return df

    out = df.copy()

    mask = out["interaction_summary"].notna() & out["interaction_summary"].astype(str).str.strip().ne("")
    texts = out.loc[mask, "interaction_summary"].astype(str).tolist()

    if not texts:
        logger.info("No non-empty interaction texts — skipping sentiment")
        for col in ("sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_label"):
            out[col] = np.nan
        return out

    logger.info("Running sentiment analysis on %d texts (batch_size=%d)", len(texts), batch_size)
    t0 = time.time()

    sent_pipe = hf_pipeline(
        "text-classification",
        model=_SENTIMENT_MODEL,
        tokenizer=_SENTIMENT_MODEL,
        top_k=None,
        truncation=True,
    )
    all_scores = sent_pipe(texts, batch_size=batch_size)

    # Handle single-text edge case
    if len(texts) == 1 and isinstance(all_scores, list) and len(all_scores) == 3 and isinstance(all_scores[0], dict):
        all_scores = [all_scores]

    rows = [_scores_to_row(s) for s in all_scores]
    scores_df = pd.DataFrame(rows, index=out.loc[mask].index)

    # Initialize columns with correct dtypes to avoid Arrow/numpy conflicts
    for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos"]:
        out[col] = np.nan  # float64
    out["sentiment_label"] = pd.Series([None] * len(out), dtype="object")

    # Assign scores back into masked rows
    for col in scores_df.columns:
        out.loc[mask, col] = scores_df[col].values

    elapsed = time.time() - t0
    logger.info("Sentiment analysis complete in %.1fs", elapsed)

    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────


def enrich_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Run intent classification (always) and sentiment analysis (if transformers available).

    Parameters
    ----------
    df : pd.DataFrame
        Raw interactions dataframe.

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with ``customer_intent``, ``has_interaction``,
        and optionally ``sentiment_label``, ``sentiment_neg/neu/pos``.
    """
    t0 = time.time()

    # Intent (regex — always runs)
    out = enrich_interactions_intent(df)

    # Sentiment (HuggingFace — only if transformers installed)
    out = enrich_interactions_sentiment(out)

    elapsed = time.time() - t0
    logger.info("NLP enrichment complete in %.1fs", elapsed)
    return out
