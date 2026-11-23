import pandas as pd

from data_cleaning_agent.utils import execute_agent_code


def run(code):
    df = pd.DataFrame({"a": [1.0, None]})
    state = {"data_raw": df.to_dict(), "code": code}
    return execute_agent_code(
        state=state,
        data_key="data_raw",
        code_snippet_key="code",
        result_key="out",
        error_key="err",
        agent_function_name="data_cleaner",
    )


def test_syntax_error_is_captured():
    result = run("def data_cleaner(df:")

    assert result["out"] is None
    assert "error" in result["err"].lower()


def test_missing_function_is_captured():
    result = run("def other(df):\n    return df")

    assert result["out"] is None
    assert "data_cleaner" in result["err"]


def test_non_dataframe_return_is_error():
    result = run("def data_cleaner(df):\n    return None")

    assert result["out"] is None
    assert "did not return a DataFrame" in result["err"]


def test_runtime_error_still_captured():
    result = run("def data_cleaner(df):\n    raise ValueError('boom')")

    assert result["out"] is None
    assert "boom" in result["err"]


def test_success_returns_dict_result():
    result = run("def data_cleaner(df):\n    return df.dropna()")

    assert result["err"] is None
    assert result["out"] == {"a": {0: 1.0}}


def test_empty_result_from_non_empty_input_is_error():
    result = run("def data_cleaner(df):\n    return df.iloc[0:0]")

    assert result["out"] is None
    assert "empty" in result["err"].lower()


def test_zero_column_result_is_error():
    result = run("def data_cleaner(df):\n    return df.iloc[:, 0:0]")

    assert result["out"] is None
    assert "empty" in result["err"].lower()
