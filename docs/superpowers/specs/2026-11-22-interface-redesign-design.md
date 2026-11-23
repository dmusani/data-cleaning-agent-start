# Interface redesign: insights, visibility, error handling

> **Status: Approved (2026-11-22).** Approval was provided by the user after reviewing this design, including the calls recorded in decisions 12-17 made after the builder check.

## Intent and why
Assignment: redesign the Streamlit interface to give better insight and a better user experience, covering data quality metrics, visualizations, custom instructions, code visibility and error handling. Today `app.py` is 42 lines: it uploads a CSV, cleans it, and shows `head()` plus a download button. The user cannot tell what changed, cannot see the generated code, and failures are mishandled.

## Goals
- After a run the user can see, at a glance, what the agent changed and how much (deterministic numbers) and why (plain-language report).
- The user can steer the run with custom instructions, max retries and model choice.
- The generated code is visible and downloadable.
- Failures are shown honestly with enough detail to act on.
- Results survive Streamlit reruns (e.g. clicking Download).

## Non-goals
- No always-visible API key field: the key is requested only when `OPENAI_API_KEY` is missing (see decisions 6 and 9).
- No per-capability toggles, column pickers or outlier-sensitivity setting.
- No new dependencies (use Streamlit built-ins: tabs, metric, bar_chart, code, expander).
- No persistence across browser sessions; no authentication.

## Anti-goals (would make it a failure even if it works)
- The UI shows "Done!" or an empty table when cleaning actually failed.
- The narrative change report claims changes that the metrics do not support.
- A failure in the report step breaks or hides a successful cleaning.

## Constraints
- Python >=3.9 (no `X | Y` type syntax); langgraph ~0.2, langchain ~0.3, streamlit ^1.28, pandas ^2.
- Existing public API (`LightweightDataCleaningAgent`, `invoke_agent`, existing getters) keeps working.
- Data crosses the graph as `to_dict()`, so dtype changes may not survive the round trip.

## Approaches considered (how the agent reports what it changed)
- A. New graph node `summarize_changes` after a successful execute (**chosen**): the model narrates metrics computed by code, so it explains rather than invents. Cost: one extra model call per successful run.
- B. Ask for a changelog in the same reply as the code: no extra call, but the changelog can drift from the code and is fragile to parse.
- C. Make the model call from `app.py`: agent untouched, but model logic lives in the UI and is unavailable to scripts and notebooks.

## Decisions (made with the user unless marked "my call")
1. **Metrics function.** `compute_quality_metrics(raw_df, cleaned_df)` in `utils.py` returns a dict: row/column counts before/after, missing cells and per-column missing % before/after, duplicate rows before/after, columns dropped, outlier counts per numeric column before/after (IQR fences, 1.5 x IQR), and cells changed (best effort). One function feeds both the app and the report node.
2. **Graph change.** After `execute_data_cleaner_code`: error with retries left -> fix node (unchanged); error without retries -> END; success -> `summarize_changes` -> END. `summarize_changes` stores `quality_metrics` and `change_summary` in state. If the model call fails it sets `change_summary` to None and does not fail the run (the app falls back to metrics only).
3. **Report prompt (my call).** The model receives only the metrics dict and the user's instructions, and must write 3-6 short bullets citing only numbers present in the metrics.
4. **Agent API additions.** `get_quality_metrics()`, `get_change_summary()`, `get_error()` (last execution error or None), `get_retry_count()`. `get_data_cleaned()` returns None (not an empty DataFrame) when there is no cleaned data.
5. **Execution error handling.** In `execute_agent_code`, move `exec(...)` and the function lookup inside the `try` so syntax errors and "function not found" become `data_cleaner_error` and enter the retry loop instead of crashing the app.
6. **Inputs (user-approved).** Required: CSV. Optional: custom instructions (text area -> `user_instructions`); Advanced expander with max retries (slider 0-5, default 3 -> `max_retries`) and model (selectbox, default `gpt-4o-mini`; options `gpt-4o-mini`, `gpt-4o` - my call). Conditional: if `OPENAI_API_KEY` is not set in the environment/.env, the app prompts the user for it with a password-type input (user-requested). The key is held only in `st.session_state`, passed to `ChatOpenAI(api_key=...)`, never written to disk or logged, and the Clean Data button stays disabled until a key is present.
7. **Layout (my call).** Sidebar: uploader, instructions, Advanced expander. Main: **Clean Data** button, then tabs:
   - Overview: metric tiles (rows, missing cells, duplicates, columns) with before/after deltas, and the change report.
   - Quality: per-column missing % before/after bar chart; outliers-per-column table/chart.
   - Code: generated function (`st.code`), attempts used, download as `.py`.
   - Data: raw vs cleaned preview, download cleaned CSV.
8. **State.** Results stored in `st.session_state`; cleared when a different file is uploaded.
9. **Failure UX.** If cleaning fails after all retries: `st.error` with a plain message, an expander with the last error and last generated code, no result tabs. Missing `OPENAI_API_KEY`: show an `st.warning` explaining the key is needed plus the password input (decision 6) instead of failing; an invalid key surfaces as a caught API error with a hint to re-enter it. Other exceptions (e.g. API/network errors): caught around the run and shown with the message, never a raw traceback.
10. **Tests (my call).** pytest unit tests for `compute_quality_metrics` using `data/messy_sample.csv` and for `execute_agent_code` error paths (syntax error, missing function); no API calls in tests.
11. **Demo data.** `data/messy_sample.csv` (already created) is the demo file.

## Calls made after the builder check (all "my call")
12. **Non-DataFrame result = execution error.** If the generated function returns anything other than a DataFrame, `execute_agent_code` records an error ("function did not return a DataFrame") inside the `try`, so it enters the retry loop like any other failure.
13. **Outlier metrics use the raw data's IQR fences for both before and after.** Labelled "values outside the original IQR fences". "After" is expected to be near 0 when capping worked, and a non-zero "after" is surfaced as "not fully resolved". The report prompt must not describe a non-zero "after" as success.
14. **Column alignment.** Per-column missing %, outlier counts and cells-changed cover only columns present in both frames. Dropped and added columns are listed separately; a renamed column appears as one dropped plus one added.
15. **The state copy of the metrics is authoritative.** The app reads `get_quality_metrics()` and never recomputes from the uploaded frame, so the UI cannot disagree with the report. The "before" side is the dict-round-tripped `data_raw`.
16. **User instructions override the defaults.** The prompt states that where custom instructions conflict with the basic steps, the instructions win (e.g. "do not remove duplicates").
17. **Tests use the existing pytest dev dependency.** "No new dependencies" means no new runtime dependencies.

## Left to the builder
Chart styling, number formatting, exact wording of messages and the report prompt, how "cells changed" aligns rows when rows were removed or reindexed (omit the metric when alignment is not possible), file/folder layout for tests.

## Known risks
- dtype changes lost in the `to_dict()` round trip; verify on the messy sample and note any limitation.
- One extra model call per successful run (cost/latency).
- The model may reindex or reorder rows, which limits the "cells changed" metric.

## Amendments made during implementation and testing
These were made after the design was approved and were not part of the original sign-off. The implementation plan (`docs/superpowers/plans/2026-11-22-interface-redesign.md`) was not changed and describes the original scope.

18. **Whole-number outlier fences (refines 13).** For a column whose raw non-null values are all whole numbers, the fences used for the outlier counts are widened outward to whole numbers (floor of the lower, ceiling of the upper). Reason: the cleaner rounds capped values, so a capped 7.25 becomes 7, which would otherwise count as a remaining outlier and produce a false "not fully resolved" report. Columns with decimals keep exact fences.
19. **`cells_changed` is None when labels were reset (refines 14).** If rows were removed and the cleaned index equals 0..n-1, the labels no longer identify the same rows as in raw (typical after `drop_duplicates().reset_index(drop=True)`), so the metric is None instead of a wrong number. Found in the final review. Cost: also None when only trailing rows were dropped.
20. **Empty result is an error (extends 12).** A cleaned frame with zero rows or zero columns from a non-empty input is an execution error and enters the retry loop, so the UI cannot report success on an empty table.
21. **Key hint depends on the key's source (refines 9).** `friendly_error(exc, key_from_env=...)`: a rejected key from the environment/.env says to correct `OPENAI_API_KEY` and restart; a rejected typed key says to re-enter it in the sidebar.
22. **No delta arrow for zero change.** Metric tiles pass no delta when before equals after.
23. **Report prompt rules (refines 3).** The report prompt now also says: the duplicate figures count exact duplicates only; if rows fell by more than that, say the extra rows were removed (e.g. near-duplicates that matched after text cleanup) and never claim there were no duplicates when rows decreased; describe outliers as capped, not removed, unless user instructions say otherwise.
24. **Cleaning prompt changes** (step order, proper-case text rule, whole-number rounding, pandas/numpy only) are recorded in `2026-11-22-outlier-text-cleaning-design.md`.
25. **Tests.** 50 offline tests. `tests/fakes.py` holds a shared fake chat model. Four run-path app tests were written after the implementation (AppTest cannot upload files, so the uploader and model are patched) and were checked with two deliberate breakages. Real-model behavior was verified manually on `data/messy_sample.csv`.
26. **Delivered:** README "What's New" section; code merged to `main`.
