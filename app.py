"""Streamlit interface for the Data Cleaning Agent."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_cleaning_agent import LightweightDataCleaningAgent
from ui_helpers import (
    can_run,
    friendly_error,
    missing_chart_df,
    outlier_table_df,
    read_csv_safely,
)

load_dotenv()

st.set_page_config(page_title="Data Cleaning Agent", page_icon="🧹", layout="wide")

MODELS = ["gpt-4o-mini", "gpt-4o"]


def render_failure(result):
    """Cleaning ran but produced no data: show why, never a success banner."""
    st.error(
        f"Cleaning failed after {result.get('retries') or 0} fix attempt(s). "
        "Try again, raise max retries, or add instructions."
    )
    with st.expander("Details: last error and last generated code", expanded=True):
        st.markdown("**Last error**")
        st.code(result.get("error") or "No error message was recorded.")
        st.markdown("**Last generated code**")
        st.code(result.get("code") or "No code was generated.", language="python")


def render_overview(result):
    metrics = result.get("metrics")
    if not metrics:
        st.warning("Quality metrics are unavailable for this run.")
    else:
        cols = st.columns(4)
        tiles = [
            ("Rows", "rows", "off"),
            ("Columns", "columns", "off"),
            ("Missing cells", "missing_cells", "inverse"),
            ("Duplicate rows", "duplicate_rows", "inverse"),
        ]
        for col, (label, key, color) in zip(cols, tiles):
            before, after = metrics[key]["before"], metrics[key]["after"]
            # No delta (and so no arrow) when nothing changed.
            col.metric(label, after, delta=(after - before) or None, delta_color=color)

    st.subheader("What changed")
    if result.get("summary"):
        st.markdown(result["summary"])
    else:
        st.info("The written report is unavailable for this run. Showing metrics only.")


def render_quality(result):
    metrics = result.get("metrics")
    if not metrics:
        st.info("Nothing to show: quality metrics are unavailable.")
        return

    st.subheader("Missing values per column (%)")
    missing = missing_chart_df(metrics)
    if missing.empty:
        st.info("Nothing to show: no columns are present both before and after.")
    else:
        st.bar_chart(missing)

    st.subheader("Outliers per column")
    outliers = outlier_table_df(metrics)
    if outliers.empty:
        st.info("No numeric columns to check for outliers.")
    else:
        st.dataframe(outliers)
        st.caption("Counts of values outside the original IQR fences (1.5 x IQR) of the raw data.")

    st.subheader("Columns")
    st.write("Dropped:", ", ".join(metrics["columns_dropped"]) or "none")
    st.write("Added:", ", ".join(metrics["columns_added"]) or "none")
    if metrics.get("cells_changed") is not None:
        st.write(f"Cells changed: {metrics['cells_changed']}")


def render_code(result):
    code = result.get("code")
    if not code:
        st.info("No generated code is available.")
        return
    st.caption(f"Fix attempts used: {result.get('retries') or 0}")
    st.code(code, language="python")
    st.download_button(
        "Download generated code", data=code, file_name="data_cleaner.py", mime="text/x-python"
    )


def render_data(result):
    left, right = st.columns(2)
    left.subheader("Raw")
    left.dataframe(result["raw_df"].head(50))
    right.subheader("Cleaned")
    right.dataframe(result["cleaned_df"].head(50))
    st.download_button(
        "Download Cleaned Data",
        data=result["cleaned_df"].to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )


def render_result(result):
    if result.get("fatal"):
        st.error(result["fatal"])
        return
    if result.get("cleaned_df") is None:
        render_failure(result)
        return
    overview, quality, code, data = st.tabs(["Overview", "Quality", "Code", "Data"])
    with overview:
        render_overview(result)
    with quality:
        render_quality(result)
    with code:
        render_code(result)
    with data:
        render_data(result)


# ---- Sidebar: inputs ----
with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    instructions = st.text_area(
        "Custom instructions (optional)",
        placeholder="e.g. Treat employee_id as text. Do not remove duplicates.",
        help="Instructions override the default cleaning steps where they conflict.",
    )
    with st.expander("Advanced"):
        max_retries = st.slider("Max retries", 0, 5, 3)
        model_name = st.selectbox("Model", MODELS, index=0)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY was not found in your environment or .env. Enter your key to continue.")
        api_key = st.text_input("OpenAI API key", type="password", key="api_key")

# ---- Main ----
st.title("🧹 Data Cleaning Agent")

if not uploaded_file:
    st.session_state.pop("result", None)
    st.session_state.pop("file_id", None)
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

file_id = (uploaded_file.name, uploaded_file.size)
if st.session_state.get("file_id") != file_id:
    st.session_state["file_id"] = file_id
    st.session_state.pop("result", None)

try:
    df_raw = read_csv_safely(uploaded_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"{uploaded_file.name}: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

ready = can_run(True, api_key)
if st.button("Clean Data", type="primary", disabled=not ready):
    with st.status("Cleaning your data...", expanded=False) as status:
        try:
            llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
            agent = LightweightDataCleaningAgent(model=llm)
            agent.invoke_agent(
                data_raw=df_raw,
                user_instructions=instructions.strip() or None,
                max_retries=max_retries,
            )
            st.session_state["result"] = {
                "raw_df": df_raw,
                "cleaned_df": agent.get_data_cleaned(),
                "code": agent.get_data_cleaner_function(),
                "metrics": agent.get_quality_metrics(),
                "summary": agent.get_change_summary(),
                "error": agent.get_error(),
                "retries": agent.get_retry_count(),
            }
            ok = st.session_state["result"]["cleaned_df"] is not None
            status.update(label="Finished" if ok else "Cleaning failed", state="complete" if ok else "error")
        except Exception as exc:
            st.session_state["result"] = {
                "fatal": friendly_error(exc, key_from_env=bool(os.environ.get("OPENAI_API_KEY")))
            }
            status.update(label="Cleaning failed", state="error")
elif not ready:
    st.info("Enter your OpenAI API key in the sidebar to enable cleaning.")

if st.session_state.get("result"):
    render_result(st.session_state["result"])
