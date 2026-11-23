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
- A. Prompt-only (**implemented**, see Amendments): smallest; the generated code computes fences itself at run time.
- B. Prompt plus richer `get_dataframe_summary` (originally chosen; **superseded by A**, see Amendments): model sees outlier bounds and text variants; small, testable change.
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

## Amendments made during implementation and testing
These were made after the design above and were not part of the original sign-off.

1. **Approach A (prompt-only) was implemented, not B.** The generated function computes the IQR fences from the data at run time, so `get_dataframe_summary` was not extended. Decision 1 above was not built. Revisit it only if the model keeps picking the wrong columns.
2. **Step order is now:** drop columns, standardize text, cap outliers, impute, remove duplicates. Outliers are capped before imputation so extreme values do not skew the fill mean, and text is standardized first so categorical modes and duplicates are computed on clean values.
3. **Text rule:** categorical columns are converted to proper case (e.g. "sales", "SALES" -> "Sales"). The original "merge to the most frequent form" wording was misread by the model (it lowercased everything and restored case for one value only). Side effect: acronyms become `Hr`, `It`. Name and ID-like columns are left alone.
4. **Column removal rule is explicit:** keep a column only if its missing fraction is 0.40 or less (`data_raw.loc[:, data_raw.isna().mean() <= 0.4]`); `dropna(thresh=...)` is forbidden because it counts non-null values and kept the wrong columns.
5. **Imputation rules:** assign filled columns back into the DataFrame (an unassigned `fillna()` does nothing), and for columns whose non-null values are all whole numbers, round the fill value before filling (check on `col.dropna()`, because NaN breaks the check).
6. **Outlier capping order for whole-number columns:** determine "all whole numbers" BEFORE clipping, then clip, then round if whole. Checking after clipping failed because the fence value is fractional. The prompt now gives this exact snippet.
7. **Only pandas and numpy** may be imported by the generated code (a run imported `sklearn`, which is not a declared dependency).
8. **Prompt lesson:** the model rewrites the whole function each run, so one rule change can alter unrelated steps. If drift continues, move capping into a tested helper in `utils.py` (approach C).
9. **Delivered via the interface redesign:** the `user_instructions` text box (decision 4) and `data/messy_sample.csv` (decision 3). Custom instructions override the default steps.
