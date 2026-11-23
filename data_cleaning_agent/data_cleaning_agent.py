# Libraries
from typing import Optional, TypedDict
import json
import os
import logging

import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import StateGraph, END

from .utils import (
    PythonOutputParser,
    compute_quality_metrics,
    get_dataframe_summary,
    execute_agent_code,
    fix_agent_code,
)

# Setup
logger = logging.getLogger(__name__)
AGENT_NAME = "lightweight_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class LightweightDataCleaningAgent:
    """
    LLM-powered agent that generates and executes Python code to clean pandas DataFrames.
    
    Uses an LLM to create data cleaning functions based on user instructions. The agent
    automatically retries with error correction if the generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for generating cleaning code (e.g., ChatOpenAI).
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving agent state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_agent() is called.
    """
    
    def __init__(
        self, 
        model, 
        log=True, 
        log_path=None, 
        file_name="data_cleaner.py", 
        function_name="data_cleaner",
        checkpointer: Checkpointer = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.file_name = file_name
        self.function_name = function_name
        self.checkpointer = checkpointer
        self.response = None
        # Build the LangGraph workflow with code generation, execution, and error fixing nodes
        self._compiled_graph = make_lightweight_data_cleaning_agent(
            model=model,
            log=log,
            log_path=log_path,
            file_name=file_name,
            function_name=function_name,
            checkpointer=checkpointer
        )
    
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
        """
        Generate and execute data cleaning code on the provided DataFrame.
        
        Parameters
        ----------
        data_raw : pd.DataFrame
            Raw dataset to clean.
        user_instructions : str, optional
            Custom cleaning instructions. If None, applies default cleaning steps:
            removing columns with >40% missing values, imputing missing values,
            and removing duplicates.
        max_retries : int, default=3
            Maximum number of retry attempts if generated code fails.
        retry_count : int, default=0
            Starting retry count (typically left at 0).
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response  # Store full workflow response for getter methods
        return None
    
    def get_data_cleaned(self):
        """
        Retrieves the cleaned data stored after running invoke_agent.
        """
        if self.response:
            cleaned = self.response.get("data_cleaned")
            if cleaned is not None:
                return pd.DataFrame(cleaned)
        return None

    def get_quality_metrics(self) -> Optional[dict]:
        """
        Retrieves the before/after quality metrics (None if unavailable).
        """
        if self.response:
            return self.response.get("quality_metrics")
        return None

    def get_change_summary(self) -> Optional[str]:
        """
        Retrieves the plain-language change report (None if unavailable).
        """
        if self.response:
            return self.response.get("change_summary")
        return None

    def get_error(self) -> Optional[str]:
        """
        Retrieves the last execution error (None if the run succeeded).
        """
        if self.response:
            return self.response.get("data_cleaner_error")
        return None

    def get_retry_count(self) -> Optional[int]:
        """
        Retrieves how many fix attempts were used.
        """
        if self.response:
            return self.response.get("retry_count")
        return None

    def get_data_raw(self):
        """
        Retrieves the raw data.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_data_cleaner_function(self):
        """
        Retrieves the agent's cleaning function code.
        """
        if self.response:
            return self.response.get("data_cleaner_function")


# Agent Factory Function

def make_lightweight_data_cleaning_agent(
    model,
    log=True,
    log_path=None,
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    checkpointer: Checkpointer = None
):
    """
    Factory function that creates a compiled LangGraph workflow for data cleaning.
    
    Builds a state graph with three nodes: code generation, execution, and error fixing.
    The workflow automatically retries with corrections if generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for generating cleaning code.
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process cleaning requests.
    """
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    

    # Define state schema for the workflow graph
    class GraphState(TypedDict):
        user_instructions: str
        data_raw: dict
        data_cleaned: dict
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        max_retries: int
        retry_count: int
        quality_metrics: dict
        change_summary: str

    
    def create_data_cleaner_code(state: GraphState):
        """
        Generate the data cleaning code based on user instructions.
        """
        logger.info("Creating data cleaner code")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        dataset_summary = get_dataframe_summary(df)
        
        # TODO: Expand this prompt with more detailed cleaning instructions
        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Create a {function_name}() function to clean the data.

            Basic Cleaning Steps to implement (in this order):
            1. Remove columns with more than 40% missing values:
                - keep a column only if its fraction of missing values is 0.40 or less, e.g. data_raw.loc[:, data_raw.isna().mean() <= 0.4]
                - do NOT use dropna(thresh=...), which counts non-null values and keeps the wrong columns
            2. Standardize text in string columns:
                - strip leading/trailing whitespace and collapse repeated spaces
                - for categorical columns (few distinct values, like department or city),
                convert values to proper case (e.g. "sales", "SALES" -> "Sales")
                - do NOT change casing of name columns or ID-like columns (e.g. employee_id)
            3. Handle outliers in numeric columns (before imputation, so outliers do not skew the mean):
                - compute IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR) per column
                - cap values outside the fences (clip), do not drop rows
                - follow this exact order for each numeric column, checking for whole numbers BEFORE clipping:
                    is_whole = data_cleaned[col].dropna().mod(1).eq(0).all()
                    data_cleaned[col] = data_cleaned[col].clip(lower_fence, upper_fence)
                    if is_whole:
                        data_cleaned[col] = data_cleaned[col].round()
                - skip ID-like columns
            4. Impute missing values (mean for numeric, mode for categorical):
                - assign every imputed column back into the DataFrame (e.g. data_cleaned[col] = data_cleaned[col].fillna(value)); a fillna() result that is not assigned has no effect
                - if a numeric column's existing non-null values are all whole numbers (check on col.dropna(), because NaN breaks the check), round the fill value to a whole number before filling
            5. Remove duplicate rows

            If User Instructions conflict with the basic steps, the User Instructions win.

            User Instructions:
            {user_instructions}

            Dataset Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                # Your cleaning code here
                return data_cleaned

            Important: Use only pandas and numpy; do not import any other libraries.
            Important: Ensure fit_transform() outputs are flattened with .ravel() when assigning to DataFrame columns.
            """,
            input_variables=["user_instructions", "all_datasets_summary", "function_name"]
        )

        data_cleaning_agent = data_cleaning_prompt | model | PythonOutputParser()
        
        response = data_cleaning_agent.invoke({
            "user_instructions": state.get("user_instructions") or "Follow the basic cleaning steps.",
            "all_datasets_summary": dataset_summary,
            "function_name": function_name
        })
        
        # Simple logging if enabled
        file_path = None
        if log:
            file_path = os.path.join(log_path, file_name)
            with open(file_path, 'w') as f:
                f.write(response)
            logger.info(f"Code saved to: {file_path}")
   
        return {
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_function_name": function_name,
        }
        
    def execute_data_cleaner_code(state):
        """
        Execute the generated cleaning code on the data.
        """
        return execute_agent_code(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name")
        )
        
    def fix_data_cleaner_code(state: GraphState):
        """
        Fix errors in the generated data cleaning code.
        """
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Fix the broken {function_name}() function.
        
        Return Python code in ```python``` format with the corrected function definition.
        
        Broken code: 
        {code_snippet}

        Error:
        {error}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=model,  
            prompt_template=data_cleaner_prompt,
            function_name=state.get("data_cleaner_function_name"),
        )

    def summarize_changes(state: GraphState):
        """
        Compute before/after quality metrics and ask the model to narrate them.
        Never raises: a failure here must not fail a successful cleaning run.
        """
        metrics = None
        summary = None
        try:
            raw_df = pd.DataFrame.from_dict(state.get("data_raw"))
            cleaned_df = pd.DataFrame.from_dict(state.get("data_cleaned"))
            metrics = compute_quality_metrics(raw_df, cleaned_df)
        except Exception as e:
            logger.warning(f"Could not compute quality metrics: {e}")

        if metrics is not None:
            summary_prompt = PromptTemplate(
                template="""
                You are a Data Cleaning Agent. Summarize what the cleaning changed, for a non-technical reader.

                Write 3-6 short bullet points, each starting with "- ".
                Use ONLY facts and numbers that appear in the metrics below; do not invent anything.
                If an outliers "after" count is greater than 0, say those outliers are not fully resolved.
                Duplicates: the duplicate_rows figures count only exact duplicates in the raw data. If the number of rows
                decreased by more than the exact duplicates explain, say the extra rows were removed (for example
                near-duplicates that matched after text cleanup), and never say there were no duplicates when rows decreased.
                Outliers are capped (clipped) to the IQR fences and their rows are kept; describe them as capped,
                not as removed, unless the User Instructions say otherwise.

                User Instructions:
                {user_instructions}

                Quality metrics (JSON):
                {metrics}
                """,
                input_variables=["user_instructions", "metrics"],
            )
            try:
                text = (summary_prompt | model | StrOutputParser()).invoke({
                    "user_instructions": state.get("user_instructions") or "None (basic cleaning steps only).",
                    "metrics": json.dumps(metrics, default=str),
                })
                summary = text.strip() or None
            except Exception as e:
                logger.warning(f"Could not generate change summary: {e}")

        return {"quality_metrics": metrics, "change_summary": summary}

    # Build the workflow graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    workflow.add_node("summarize_changes", summarize_changes)

    # Set entry point
    workflow.set_entry_point("create_data_cleaner_code")
    
    # Add edges
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("fix_data_cleaner_code", "execute_data_cleaner_code")
    
    # Conditional routing: summarize on success; on error, retry with fixes
    # while retries remain, otherwise stop.
    def should_retry(state):
        if state.get("data_cleaner_error") is None:
            return "summarize"
        can_retry = (
            state.get("retry_count") is not None
            and state.get("max_retries") is not None
            and state["retry_count"] < state["max_retries"]
        )
        return "fix_code" if can_retry else "end"

    workflow.add_conditional_edges(
        "execute_data_cleaner_code",
        should_retry,
        {
            "fix_code": "fix_data_cleaner_code",
            "summarize": "summarize_changes",
            "end": END,
        },
    )
    workflow.add_edge("summarize_changes", END)
    
    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
    
    return app
