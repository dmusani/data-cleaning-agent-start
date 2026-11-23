import pandas as pd

from data_cleaning_agent import LightweightDataCleaningAgent
from fakes import BAD_CODE, GOOD_CODE, REPORT, ScriptedModel


def make_agent(responses):
    model = ScriptedModel(responses=responses)
    return LightweightDataCleaningAgent(model=model, log=False), model


def sample_df():
    return pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})


def test_success_populates_metrics_and_summary():
    agent, _ = make_agent([GOOD_CODE, REPORT])

    agent.invoke_agent(data_raw=sample_df())

    assert isinstance(agent.get_data_cleaned(), pd.DataFrame)
    assert agent.get_quality_metrics()["rows"]["before"] == 3
    assert agent.get_change_summary() == REPORT
    assert agent.get_error() is None


def test_failure_after_retries_returns_none():
    agent, _ = make_agent([BAD_CODE, BAD_CODE, BAD_CODE])

    agent.invoke_agent(data_raw=sample_df(), max_retries=1)

    assert agent.get_data_cleaned() is None
    assert agent.get_error()
    assert agent.get_retry_count() == 1
    assert agent.get_change_summary() is None


def test_retry_then_success():
    agent, _ = make_agent([BAD_CODE, GOOD_CODE, REPORT])

    agent.invoke_agent(data_raw=sample_df())

    assert agent.get_retry_count() == 1
    assert isinstance(agent.get_data_cleaned(), pd.DataFrame)
    assert agent.get_error() is None


def test_report_failure_does_not_fail_run():
    agent, _ = make_agent([GOOD_CODE, "RAISE"])

    agent.invoke_agent(data_raw=sample_df())

    assert isinstance(agent.get_data_cleaned(), pd.DataFrame)
    assert agent.get_change_summary() is None
    assert agent.get_quality_metrics() is not None


def test_get_data_cleaned_none_before_invoke():
    agent, _ = make_agent([GOOD_CODE])

    assert agent.get_data_cleaned() is None


def test_prompt_states_instructions_override_defaults():
    agent, model = make_agent([GOOD_CODE, REPORT])

    agent.invoke_agent(data_raw=sample_df(), user_instructions="keep duplicates")

    assert "the User Instructions win" in model.calls[0]


def test_report_prompt_explains_removed_rows_not_only_exact_duplicates():
    agent, model = make_agent([GOOD_CODE, REPORT])

    agent.invoke_agent(data_raw=sample_df())

    report_prompt = model.calls[1]
    assert "near-duplicates" in report_prompt
    assert "never say there were no duplicates" in report_prompt.lower()


def test_report_prompt_says_outliers_are_capped_not_removed():
    agent, model = make_agent([GOOD_CODE, REPORT])

    agent.invoke_agent(data_raw=sample_df())

    report_prompt = model.calls[1].lower()
    assert "capped" in report_prompt
    assert "not as removed" in report_prompt
