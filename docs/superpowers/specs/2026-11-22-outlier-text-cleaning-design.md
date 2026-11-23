# Outlier detection and text standardization (short design)

## Intent
Assignment: extend the data cleaning agent with at least two capabilities beyond the baseline (missing values, duplicates, column removal). Chosen pair: outlier detection and text standardization.

## Goals
- The generated `data_cleaner()` handles outliers and text standardization in addition to the baseline steps.
- The LLM is given evidence for both, so it decides from data rather than guessing.
- A messy test CSV shows a clear before/after for each capability.

## Non-goals / anti-goals
- No deterministic helper library in `utils.py` (approach C).
- No row loss from outlier handling: outliers are capped, never dropped.
- Must not mangle identifier columns (e.g. `employee_id`) or person names.

## Approaches considered
- A. Prompt-only: smallest, but the model works blind.
- B. Prompt plus richer `get_dataframe_summary` (**chosen**): model sees outlier bounds and text variants; small, testable change.
- C. Deterministic helpers called from the prompt: most reliable, most code, less flexible.

## Decisions
1. `get_dataframe_summary` (utils.py) gains:
   - Numeric columns: min, max, Q1, Q3, IQR fences (1.5 x IQR), count outside fences.
   - Text columns: unique count, up to 5 sample values, count of values with leading/trailing/double whitespace, number of case-variant groups (e.g. "sales" vs "Sales").
2. Prompt (create_data_cleaner_code) gains two steps, with text standardization placed BEFORE duplicate removal so near-duplicates collapse:
   - Text: strip and collapse whitespace; for low-cardinality categorical columns merge case variants to the most frequent form; leave ID-like and name columns' casing alone.
   - Outliers: cap numeric values at the IQR fences (winsorize); skip ID-like columns.
3. New test file `data/messy_sample.csv` with injected outliers, whitespace and case variants (sample_data.csv is too clean to show either effect).
4. My call: add an optional `user_instructions` text box in `app.py` (currently never passed) so extra rules can be tried from the UI.

## Left to the builder
Exact summary formatting, how to detect ID-like columns, wording of prompt steps.

## Known risk
Data round-trips through `to_dict()`, so dtype changes may not survive; check on the messy sample and note any limitation.
