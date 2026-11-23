from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning_agent.utils import compute_quality_metrics

MESSY = Path(__file__).resolve().parent.parent / "data" / "messy_sample.csv"


def test_row_and_missing_counts():
    raw = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": ["x", None, "z", "w"]})
    cleaned = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": ["x", "y", "w"]}, index=[0, 1, 3])

    m = compute_quality_metrics(raw, cleaned)

    assert m["rows"] == {"before": 4, "after": 3}
    assert m["missing_cells"] == {"before": 2, "after": 0}


def test_duplicates_before_after():
    raw = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    m = compute_quality_metrics(raw, raw.drop_duplicates())

    assert m["duplicate_rows"] == {"before": 1, "after": 0}


def test_dropped_and_added_columns():
    raw = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cleaned = raw.rename(columns={"b": "c"})

    m = compute_quality_metrics(raw, cleaned)

    assert m["columns_dropped"] == ["b"]
    assert m["columns_added"] == ["c"]
    assert m["columns"] == {"before": 2, "after": 2}


def test_outliers_use_raw_fences():
    # Q1=2, Q3=4, IQR=2 -> upper fence 7; only 1000 is outside.
    raw = pd.DataFrame({"a": [1, 2, 3, 4, 1000]})
    cleaned = raw.assign(a=raw["a"].clip(upper=7))

    m = compute_quality_metrics(raw, cleaned)

    assert m["outliers_by_column"]["a"] == {"before": 1, "after": 0}


def test_cells_changed_counts_modified_cells():
    raw = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cleaned = pd.DataFrame({"a": [1, 9, 3], "b": ["x", "y", "z"]})

    assert compute_quality_metrics(raw, cleaned)["cells_changed"] == 1


def test_cells_changed_none_when_reindexed():
    # Raw has a non-default index, so a reset_index() on the cleaned frame
    # produces labels (0..2) that do not exist in raw.
    raw = pd.DataFrame({"a": [1, 1, 2, 3]}, index=[10, 11, 12, 13])
    cleaned = raw.drop_duplicates().reset_index(drop=True)

    assert compute_quality_metrics(raw, cleaned)["cells_changed"] is None


def test_zero_row_frame_does_not_raise():
    empty = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)})

    m = compute_quality_metrics(empty, empty)

    assert m["missing_pct_by_column"]["a"] == {"before": 0.0, "after": 0.0}
    assert m["rows"] == {"before": 0, "after": 0}


def test_no_numeric_columns():
    raw = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})

    assert compute_quality_metrics(raw, raw)["outliers_by_column"] == {}


def test_messy_sample_shape():
    raw = pd.read_csv(MESSY)

    m = compute_quality_metrics(raw, raw.drop_duplicates())

    assert m["rows"]["before"] == 100


def test_cells_changed_none_when_labels_were_reset_after_row_removal():
    # Default-index raw; cleaned drops the duplicate then resets labels to 0..n-1.
    # Those labels exist in raw but point at different rows, so alignment is wrong.
    raw = pd.DataFrame({"a": [1, 1, 2, 3, 4]})
    cleaned = raw.drop_duplicates().reset_index(drop=True)

    assert compute_quality_metrics(raw, cleaned)["cells_changed"] is None


def test_cells_changed_zero_when_labels_preserved_after_row_removal():
    raw = pd.DataFrame({"a": [1, 1, 2, 3, 4]})

    assert compute_quality_metrics(raw, raw.drop_duplicates())["cells_changed"] == 0


def test_whole_number_column_rounding_does_not_count_as_outlier():
    # Fences are (-2.5, 7.5). Clipping 100 to 7.5 and rounding to a whole
    # number gives 8, which a whole-number column cannot avoid.
    raw = pd.DataFrame({"a": [0, 1, 2, 3, 4, 100]})
    cleaned = raw.assign(a=raw["a"].clip(-2.5, 7.5).round())

    m = compute_quality_metrics(raw, cleaned)

    assert m["outliers_by_column"]["a"] == {"before": 1, "after": 0}


def test_fractional_column_keeps_exact_fences():
    # Fences are (-0.15, 0.85); 0.9 is genuinely outside for a non-whole column.
    raw = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4, 0.5, 10.0]})
    cleaned = raw.assign(a=[0.1, 0.2, 0.3, 0.4, 0.5, 0.9])

    m = compute_quality_metrics(raw, cleaned)

    assert m["outliers_by_column"]["a"] == {"before": 1, "after": 1}
