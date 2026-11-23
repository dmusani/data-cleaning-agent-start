"""Pure helper functions for the Streamlit interface (no Streamlit imports)."""

from typing import Optional

import pandas as pd

_BEFORE_AFTER = ["before", "after"]


def _before_after_df(rows: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_BEFORE_AFTER)
    return pd.DataFrame.from_dict(rows, orient="index")[_BEFORE_AFTER]


def missing_chart_df(metrics: dict) -> pd.DataFrame:
    """Per-column missing % before/after (index = column name)."""
    return _before_after_df(metrics.get("missing_pct_by_column", {}))


def outlier_table_df(metrics: dict) -> pd.DataFrame:
    """Per-column outlier counts before/after (index = column name)."""
    return _before_after_df(metrics.get("outliers_by_column", {}))


def can_run(file_present: bool, api_key: Optional[str]) -> bool:
    """The Clean Data button needs an uploaded file and a non-blank API key."""
    return bool(file_present and api_key and api_key.strip())


def read_csv_safely(file) -> pd.DataFrame:
    """Read an uploaded CSV, raising ValueError with a readable message on failure."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty. Please upload a CSV with a header row and data.")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read this file as a CSV: {exc}")
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError("The file has no data rows. Please upload a CSV that contains data.")
    return df


def friendly_error(exc: Exception, key_from_env: bool = False) -> str:
    """Turn an exception from a cleaning run into a short, actionable message.

    key_from_env says the API key came from the environment/.env, where there is
    no sidebar field to re-enter it in.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "authentication" in name or "401" in text or "api key" in text or "api_key" in text:
        if key_from_env:
            return (
                "OpenAI rejected the API key. Correct OPENAI_API_KEY in your "
                "environment or .env file, then restart the app."
            )
        return "OpenAI rejected the API key. Please re-enter it in the sidebar and try again."
    if "ratelimit" in name or "rate limit" in text or "429" in text:
        return "OpenAI rate limit reached. Wait a moment and try again."
    if "connection" in name or "timeout" in name or "connect" in text:
        return "Could not reach OpenAI. Check your network connection and try again."
    return f"Something went wrong while cleaning: {exc}"
