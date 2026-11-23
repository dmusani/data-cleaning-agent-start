# Interface Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Streamlit app so a run shows what the agent changed (deterministic metrics plus an LLM-written report), the generated code, and honest error states.

**Architecture:** One shared `compute_quality_metrics` function in `utils.py` feeds a new `summarize_changes` LangGraph node (narrative) and, via agent getters, the UI (numbers and charts). `execute_agent_code` becomes failure-safe so errors enter the existing retry loop. `app.py` is rebuilt around `st.session_state`, a sidebar of inputs and four result tabs.

**Tech Stack:** Python >=3.9, pandas ^2, langchain ~0.3, langgraph ~0.2, streamlit ^1.28, pytest ^7 (already a dev dependency).

**Spec:** `docs/superpowers/specs/2026-11-22-interface-redesign-design.md` (approved 2026-11-22). Executors read both.

## Global Constraints

- Python >=3.9: no `X | Y` type syntax, no `match`.
- No new runtime dependencies (Streamlit built-ins only: tabs, metric, bar_chart, code, expander, download_button).
- Existing public API keeps working: `LightweightDataCleaningAgent`, `invoke_agent`, `get_data_cleaned`, `get_data_raw`, `get_data_cleaner_function`.
- `get_data_cleaned()` returns `None` (not an empty DataFrame) when there is no cleaned data.
- Outlier metrics use the raw data's IQR fences (1.5 x IQR) for both before and after.
- User instructions override the default cleaning steps when they conflict.
- API key: read `OPENAI_API_KEY` from env/.env; only if missing, prompt with a password input held in `st.session_state`; never written to disk or logged.
- Inputs: CSV (required); custom instructions; Advanced expander with max retries (slider 0-5, default 3) and model (selectbox `gpt-4o-mini` default, `gpt-4o`).
- Tests make no network or OpenAI calls.
- Do not commit unless the user asks; "Checkpoint" steps below mean "commit if the user has asked for commits, otherwise skip".

## Review Focus

Failure modes the spec implies; each has a test in the named task.

1. Zero-row or all-NaN frame: `compute_quality_metrics` must not divide by zero (Task 1).
2. No numeric columns (or no text columns): metrics return empty `outliers_by_column` and the UI shows a "nothing to show" message instead of an empty chart (Tasks 1, 4).
3. Generated function renames columns or reindexes rows: `cells_changed` is `None`, nothing crashes (Task 1).
4. Invalid API key or network error during a run: caught and shown with a hint to re-enter the key, never a raw traceback (Task 4).
5. Empty or unparseable CSV upload: friendly error, app stays usable (Task 4).

## File Structure

- Modify `data_cleaning_agent/utils.py`: add `compute_quality_metrics`; harden `execute_agent_code`.
- Modify `data_cleaning_agent/data_cleaning_agent.py`: state fields, `summarize_changes` node, routing, prompt override rule, new getters.
- Create `ui_helpers.py` (project root, beside `app.py`): pure functions for chart data and run-gating, so they are testable without Streamlit.
- Rewrite `app.py`: layout, state, error handling.
- Create `tests/test_metrics.py`, `tests/test_execute.py`, `tests/test_agent_graph.py`, `tests/test_ui_helpers.py`, `tests/test_app.py`.
- Modify `pyproject.toml`: pytest `pythonpath`.

---

### Task 1: Quality metrics function

**Files:**
- Modify: `data_cleaning_agent/utils.py`
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options]` with `pythonpath = ["."]`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `compute_quality_metrics(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict` with keys:
  - `rows`, `columns`, `missing_cells`, `duplicate_rows`: each `{"before": int, "after": int}`
  - `columns_dropped: list[str]`, `columns_added: list[str]`
  - `missing_pct_by_column: dict[str, {"before": float, "after": float}]` (columns in both frames; 0.0 when the frame has zero rows)
  - `outliers_by_column: dict[str, {"before": int, "after": int}]` (numeric columns in both frames; fences from `raw`; empty dict when none)
  - `cells_changed: Optional[int]` (`None` when `cleaned.index` is not a subset of `raw.index`)

- [ ] **Step 1: Add pytest config** `pythonpath = ["."]` to `pyproject.toml`.
- [ ] **Step 2: Write failing tests** in `tests/test_metrics.py`, built on small hand-made frames plus one test on `data/messy_sample.csv`:
  - `test_row_and_missing_counts`: raw 4 rows/2 NaN, cleaned 3 rows/0 NaN gives `rows == {"before": 4, "after": 3}` and `missing_cells == {"before": 2, "after": 0}`.
  - `test_duplicates_before_after`: one exact duplicate in raw, none in cleaned.
  - `test_dropped_and_added_columns`: a renamed column appears in both `columns_dropped` and `columns_added`.
  - `test_outliers_use_raw_fences`: raw `[1,2,3,4,1000]`, cleaned clipped to the raw upper fence; `before == 1`, `after == 0`.
  - `test_cells_changed_none_when_reindexed`: cleaned with `reset_index()` after row removal gives `cells_changed is None`.
  - `test_zero_row_frame_does_not_raise`: empty frames give 0.0 percentages.
  - `test_no_numeric_columns`: `outliers_by_column == {}`.
  - `test_messy_sample_shape`: on `data/messy_sample.csv` cleaned as `drop_duplicates()`, `rows["before"] == 100`.
- [ ] **Step 3: Run** `poetry run pytest tests/test_metrics.py -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 4: Implement** `compute_quality_metrics` in `utils.py` per the Interfaces block. IQR via `Series.quantile(0.25/0.75)`; skip a numeric column if its fences are NaN.
- [ ] **Step 5: Run** the same command. Expected: all PASS.
- [ ] **Step 6: Checkpoint** (`feat: add compute_quality_metrics`).

---

### Task 2: Failure-safe execution

**Files:**
- Modify: `data_cleaning_agent/utils.py` (`execute_agent_code`)
- Test: `tests/test_execute.py`

**Interfaces:**
- Consumes: existing `execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name) -> dict`
- Produces: same signature; now never raises for bad generated code. On failure returns `{result_key: None, error_key: "<message>"}`.

- [ ] **Step 1: Write failing tests** (state built from a small `df.to_dict()`):
  - `test_syntax_error_is_captured`: code `"def data_cleaner(df:"` gives `error_key` containing `"error"` (case-insensitive) and `result_key is None`, no exception.
  - `test_missing_function_is_captured`: code defines `other()`; error mentions `data_cleaner`.
  - `test_non_dataframe_return_is_error`: function returns `None`; error contains `"did not return a DataFrame"`.
  - `test_runtime_error_still_captured` (regression of existing behavior).
  - `test_success_returns_dict_result`.
- [ ] **Step 2: Run** `poetry run pytest tests/test_execute.py -v`. Expected: first three FAIL.
- [ ] **Step 3: Implement**: move `exec(...)` and the callable lookup inside the existing `try`; after the call, if the result is not a `pd.DataFrame`, set the error above.
- [ ] **Step 4: Run** the same command. Expected: all PASS.
- [ ] **Step 5: Checkpoint** (`fix: capture generated-code failures as retryable errors`).

---

### Task 3: Agent graph, report node, getters

**Files:**
- Modify: `data_cleaning_agent/data_cleaning_agent.py`
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: `compute_quality_metrics` (Task 1), failure-safe `execute_agent_code` (Task 2).
- Produces:
  - `GraphState` gains `quality_metrics: dict` and `change_summary: str`.
  - Node `summarize_changes(state) -> dict` returning `{"quality_metrics": dict|None, "change_summary": str|None}`; it never raises.
  - Routing after execute: error and `retry_count < max_retries` -> `fix_data_cleaner_code`; error otherwise -> `END`; no error -> `summarize_changes` -> `END`.
  - `LightweightDataCleaningAgent.get_quality_metrics() -> Optional[dict]`, `get_change_summary() -> Optional[str]`, `get_error() -> Optional[str]`, `get_retry_count() -> Optional[int]`.
  - `get_data_cleaned()` returns `None` when `data_cleaned` is absent or `None`.

Tests use `langchain_core.language_models.fake_chat_models.FakeListChatModel(responses=[...])` (no network); responses are consumed in call order: code, then optional fixes, then the report.

- [ ] **Step 1: Write failing tests**:
  - `test_success_populates_metrics_and_summary`: responses `[good_code_block, "- removed 1 duplicate"]`; assert `get_data_cleaned()` is a DataFrame, `get_quality_metrics()["rows"]["before"]` matches input, `get_change_summary() == "- removed 1 duplicate"`, `get_error() is None`.
  - `test_failure_after_retries_returns_none`: `max_retries=1`, responses are bad code three times; assert `get_data_cleaned() is None`, `get_error()` is truthy, `get_retry_count() == 1`, `get_change_summary() is None`.
  - `test_retry_then_success`: bad code, then good code, then report; `get_retry_count() == 1`, data present.
  - `test_report_failure_does_not_fail_run`: model that raises on the report call (custom subclass or a response list exhausted); cleaned data present, `get_change_summary() is None`, metrics present.
  - `test_get_data_cleaned_none_before_invoke`.
  - `test_prompt_states_instructions_override_defaults`: capture the rendered prompt via a recording fake model and assert it contains the override sentence.
- [ ] **Step 2: Run** `poetry run pytest tests/test_agent_graph.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the state fields, node, three-way `should_retry`, and `add_edge("summarize_changes", END)`. Report prompt (a decision of the plan): give the model only `json.dumps(metrics, default=str)` and the user's instructions; require 3-6 short bullets citing only numbers present; state that a non-zero outliers "after" means "not fully resolved". Add to the code-generation prompt: "If User Instructions conflict with the basic steps, the User Instructions win." In `invoke_agent`, leave signature unchanged.
- [ ] **Step 4: Implement the four getters** and the `get_data_cleaned` change.
- [ ] **Step 5: Run** the same command. Expected: all PASS. Also run `poetry run pytest -v` for regressions.
- [ ] **Step 6: Checkpoint** (`feat: add change-report node and result getters`).

---

### Task 4: UI helpers and app rebuild

**Files:**
- Create: `ui_helpers.py`
- Rewrite: `app.py`
- Test: `tests/test_ui_helpers.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: agent getters from Task 3; metrics schema from Task 1.
- Produces in `ui_helpers.py`:
  - `missing_chart_df(metrics: dict) -> pd.DataFrame` (index = column, columns `["before", "after"]`, percent values; empty frame when no common columns)
  - `outlier_table_df(metrics: dict) -> pd.DataFrame` (index = column, columns `["before", "after"]`; empty when none)
  - `can_run(file_present: bool, api_key: Optional[str]) -> bool`
  - `read_csv_safely(file) -> pd.DataFrame` (wraps `pd.read_csv`; raises `ValueError` with a user-readable message for empty, header-only or unparseable files)
  - `friendly_error(exc: Exception) -> str` (maps auth/connection/other exceptions to a short message with a hint to re-enter the key for auth errors)

- [ ] **Step 1: Write failing tests** `tests/test_ui_helpers.py`:
  - `test_missing_chart_df_shape` from a hand-made metrics dict.
  - `test_outlier_table_empty_when_no_numeric`: returns an empty frame.
  - `test_can_run_requires_file_and_key`: four combinations (`""` and `None` count as no key).
  - `test_read_csv_safely_rejects_empty_and_header_only`: `io.StringIO("")` and `io.StringIO("a,b\n")` both raise `ValueError`; a valid CSV returns a frame.
  - `test_friendly_error_auth_hint`: `openai.AuthenticationError`-like exception text containing `"401"` or `"api key"` yields a message containing `"re-enter"`.
- [ ] **Step 2: Write failing app test** `tests/test_app.py` using `streamlit.testing.v1.AppTest.from_file("app.py")`; monkeypatch `dotenv.load_dotenv` to a no-op and `delenv("OPENAI_API_KEY")`:
  - `test_missing_key_prompts_for_it`: after `run()`, `at.warning` is non-empty and a `text_input` with `type == "password"` exists in the sidebar.
  - `test_key_in_env_hides_prompt`: with the env var set, no password input.
  - `test_no_file_shows_prompt_to_upload`: `at.info` mentions uploading a CSV and there are no tabs.
- [ ] **Step 3: Run** `poetry run pytest tests/test_ui_helpers.py tests/test_app.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement `ui_helpers.py`** per Interfaces.
- [ ] **Step 5: Rebuild `app.py`** per decisions 6-9 of the spec:
  - Sidebar: uploader, instructions text area, Advanced expander (slider 0-5 default 3; selectbox `gpt-4o-mini`, `gpt-4o`), password input only when `OPENAI_API_KEY` is missing.
  - Main: **Clean Data** button disabled unless `can_run(...)`; run inside `st.status`/spinner wrapped in try/except using `friendly_error`; store `{raw_df, cleaned_df, code, metrics, summary, error, retries}` in `st.session_state["result"]`; clear it when the uploaded file name/size changes.
  - Read the CSV with `read_csv_safely`; on `ValueError` show `st.error` with its message and stop.
  - If `get_data_cleaned()` is `None`: `st.error`, plus an expander with the last error and last generated code; render no result tabs.
  - Tabs Overview (metric tiles with deltas from metrics, change report or a "report unavailable, showing metrics only" note), Quality (`missing_chart_df` bar chart, `outlier_table_df`, lists of dropped/added columns; "nothing to show" message when empty), Code (`st.code`, attempts used, `.py` download), Data (raw vs cleaned previews, CSV download).
- [ ] **Step 6: Run** `poetry run pytest -v`. Expected: all PASS.
- [ ] **Step 7: Checkpoint** (`feat: redesign Streamlit interface`).

---

### Task 5: End-to-end check on the messy sample (manual)

**Files:** none (verification only).

- [ ] **Step 1:** Run `poetry run streamlit run app.py` with a valid `OPENAI_API_KEY`; upload `data/messy_sample.csv`; click **Clean Data**.
- [ ] **Step 2: Confirm:**
  - `department` collapses to 4 values; ages and salaries capped, not dropped; `notes` dropped; `employee_id` unchanged.
  - Overview tiles and change report match the Quality tab numbers.
  - Clicking **Download** does not clear the results.
  - Uploading a different file clears the results.
  - Starting without the env key shows the password prompt; an invalid key shows the friendly error.
  - Note any dtype loss from the `to_dict()` round trip as a known limitation.
- [ ] **Step 3:** Run `poetry run pytest -v`; all PASS.
