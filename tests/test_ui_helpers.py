import io

import pandas as pd
import pytest

from ui_helpers import (
    can_run,
    friendly_error,
    missing_chart_df,
    outlier_table_df,
    read_csv_safely,
)

METRICS = {
    "missing_pct_by_column": {
        "age": {"before": 5.0, "after": 0.0},
        "city": {"before": 4.0, "after": 0.0},
    },
    "outliers_by_column": {"age": {"before": 3, "after": 0}},
}


def test_missing_chart_df_shape():
    df = missing_chart_df(METRICS)

    assert list(df.columns) == ["before", "after"]
    assert list(df.index) == ["age", "city"]
    assert df.loc["age", "before"] == 5.0


def test_missing_chart_df_empty_when_no_common_columns():
    df = missing_chart_df({"missing_pct_by_column": {}})

    assert df.empty
    assert list(df.columns) == ["before", "after"]


def test_outlier_table_df_values():
    df = outlier_table_df(METRICS)

    assert list(df.columns) == ["before", "after"]
    assert df.loc["age", "before"] == 3


def test_outlier_table_empty_when_no_numeric():
    df = outlier_table_df({"outliers_by_column": {}})

    assert df.empty
    assert list(df.columns) == ["before", "after"]


@pytest.mark.parametrize(
    "file_present, api_key, expected",
    [
        (True, "sk-abc", True),
        (False, "sk-abc", False),
        (True, "", False),
        (True, None, False),
        (True, "   ", False),
    ],
)
def test_can_run_requires_file_and_key(file_present, api_key, expected):
    assert can_run(file_present, api_key) is expected


def test_read_csv_safely_rejects_empty_and_header_only():
    with pytest.raises(ValueError):
        read_csv_safely(io.StringIO(""))
    with pytest.raises(ValueError):
        read_csv_safely(io.StringIO("a,b\n"))


def test_read_csv_safely_returns_frame_for_valid_csv():
    df = read_csv_safely(io.StringIO("a,b\n1,x\n2,y\n"))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_friendly_error_auth_hint():
    msg = friendly_error(Exception("Error code: 401 - Incorrect API key provided"))

    assert "re-enter" in msg.lower()


def test_friendly_error_generic_includes_message():
    assert "kaboom" in friendly_error(RuntimeError("kaboom"))


def test_friendly_error_auth_hint_for_env_key_points_to_env():
    msg = friendly_error(
        Exception("Error code: 401 - Incorrect API key provided"), key_from_env=True
    )

    assert "OPENAI_API_KEY" in msg
    assert "sidebar" not in msg.lower()
