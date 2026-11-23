# Utility functions for lightweight data cleaning agent

import math
import re
import logging
from typing import Optional

import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""
    
    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    
    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])
    
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    
    summary = f"""
        Dataset Summary:
        ----------------
        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}"""

    return summary.strip()


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    
    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # Execute the LLM-generated code in an isolated namespace.
    # Note: exec() can be risky - only use with trusted LLM-generated code
    agent_error = None
    result = None
    try:
        local_vars = {}
        global_vars = {}
        exec(agent_code, global_vars, local_vars)

        agent_function = local_vars.get(agent_function_name)
        if not agent_function or not callable(agent_function):
            raise ValueError(f"Function '{agent_function_name}' not found in generated code.")

        result = agent_function(df)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"{agent_function_name}() did not return a DataFrame "
                f"(got {type(result).__name__})."
            )
        if (result.shape[0] == 0 or result.shape[1] == 0) and df.shape[0] > 0 and df.shape[1] > 0:
            raise ValueError(
                f"{agent_function_name}() returned an empty DataFrame "
                f"({result.shape[0]} rows x {result.shape[1]} columns) from non-empty input."
            )
        result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        result = None
        agent_error = f"An error occurred during data cleaning: {str(e)}"

    return {result_key: result, error_key: agent_error}


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }


def _missing_pct(df: pd.DataFrame, col: str) -> float:
    """Percentage of missing values in a column (0.0 for a zero-row frame)."""
    if len(df) == 0:
        return 0.0
    return float(df[col].isna().mean() * 100)


def _count_outside(series: pd.Series, lower: float, upper: float) -> int:
    """Count values strictly outside [lower, upper]; NaN never counts."""
    return int(((series < lower) | (series > upper)).sum())


def compute_quality_metrics(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """
    Compare a raw DataFrame with its cleaned version.

    Outlier counts use the IQR fences (1.5 x IQR) computed on the RAW data for
    both the "before" and "after" figures, so a non-zero "after" means values
    are still outside the original fences.

    Returns
    -------
    dict with keys: rows, columns, missing_cells, duplicate_rows (each
    {"before", "after"}), columns_dropped, columns_added (lists),
    missing_pct_by_column and outliers_by_column ({col: {"before", "after"}},
    for columns present in both frames), and cells_changed (int, or None when
    the cleaned index cannot be aligned with the raw index).
    """
    common = [c for c in raw.columns if c in cleaned.columns]

    missing_pct = {
        col: {"before": _missing_pct(raw, col), "after": _missing_pct(cleaned, col)}
        for col in common
    }

    outliers = {}
    for col in common:
        both_numeric = (
            pd.api.types.is_numeric_dtype(raw[col])
            and pd.api.types.is_numeric_dtype(cleaned[col])
            and not pd.api.types.is_bool_dtype(raw[col])
        )
        if not both_numeric:
            continue
        values = raw[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if pd.isna(lower) or pd.isna(upper):
            continue
        # A whole-number column cannot hold a fractional fence value, so rounding
        # a capped value to 7 when the fence is 7.25 is not an outlier. Widen the
        # fences outward to whole numbers for such columns.
        if (values % 1 == 0).all():
            lower, upper = math.floor(lower), math.ceil(upper)
        outliers[col] = {
            "before": _count_outside(raw[col], lower, upper),
            "after": _count_outside(cleaned[col], lower, upper),
        }

    # After rows were removed, labels 0..n-1 almost always mean the cleaner reset
    # the index, so they no longer identify the same rows as in raw. Treat that
    # as unalignable rather than report a wrong number.
    labels_reset = len(cleaned) != len(raw) and cleaned.index.equals(
        pd.RangeIndex(len(cleaned))
    )

    cells_changed: Optional[int] = None
    if (
        not labels_reset
        and raw.index.is_unique
        and cleaned.index.is_unique
        and cleaned.index.isin(raw.index).all()
    ):
        aligned = raw.loc[cleaned.index, common]
        after = cleaned[common]
        same = (aligned == after) | (aligned.isna() & after.isna())
        cells_changed = int((~same).sum().sum())

    return {
        "rows": {"before": len(raw), "after": len(cleaned)},
        "columns": {"before": raw.shape[1], "after": cleaned.shape[1]},
        "columns_dropped": [c for c in raw.columns if c not in cleaned.columns],
        "columns_added": [c for c in cleaned.columns if c not in raw.columns],
        "missing_cells": {
            "before": int(raw.isna().sum().sum()),
            "after": int(cleaned.isna().sum().sum()),
        },
        "missing_pct_by_column": missing_pct,
        "duplicate_rows": {
            "before": int(raw.duplicated().sum()),
            "after": int(cleaned.duplicated().sum()),
        },
        "outliers_by_column": outliers,
        "cells_changed": cells_changed,
    }
