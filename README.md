# Data Cleaning Agent

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for common data cleaning tasks like handling missing values, removing duplicates, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

## Setup

### Prerequisites

- **Python 3.9 or higher** (3.9, 3.10, 3.11, 3.12, or 3.13) - **Note**: Python 3.9.7 is not supported due to a Streamlit compatibility issue
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.9, python3.10, python3.12, etc.

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Streamlit Web Interface

The easiest way to use the agent is through the web interface:

```bash
poetry run streamlit run app.py
```

Then:
1. Upload your CSV file
2. Click "Clean Data"
3. Download the cleaned dataset

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(data_raw=df)

# Get the cleaned dataset
cleaned_df = agent.get_data_cleaned()

# Save or use the cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: Provide custom instructions**

```python
# Give specific cleaning instructions to the agent
agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats"
)
```

## What's New

### New cleaning capabilities

The agent now applies these steps, in this order:

1. **Drop columns** with more than 40% missing values
2. **Standardize text**: trim whitespace, collapse repeated spaces, and convert categorical columns (such as department or city) to proper case. Name and ID-like columns are left alone.
3. **Cap outliers**: values outside the IQR fences (Q1 − 1.5×IQR, Q3 + 1.5×IQR) are clipped to the fence. Rows are never dropped, and columns that held whole numbers stay whole.
4. **Impute missing values**: mean for numeric columns, mode for categorical columns. Whole-number columns get a whole-number fill value.
5. **Remove duplicate rows**: this runs after text standardization, so near-duplicates such as `" Sales"` and `"sales"` collapse.

Custom instructions override these defaults wherever they conflict (for example, "do not remove duplicates").

### Redesigned web interface

```bash
poetry run streamlit run app.py
```

- **Inputs (sidebar)**: the CSV, optional custom instructions, and an **Advanced** section with max retries (0–5, default 3) and the model (`gpt-4o-mini` or `gpt-4o`).
- **API key prompt**: if `OPENAI_API_KEY` is not set in your environment or `.env`, the app asks for it in the sidebar. The key is kept only in the browser session and is never written to disk.
- **Results in four tabs**, which stay on screen when you click a download button:
  - **Overview**: before/after tiles for rows, columns, missing cells and duplicate rows, plus a plain-language report of what changed.
  - **Quality**: missing values per column (before vs. after), outlier counts per column, and the columns that were dropped or added.
  - **Code**: the generated cleaning function, how many fix attempts were used, and a download button for the `.py` file.
  - **Data**: raw and cleaned data side by side, with a download button for the cleaned CSV.
- **Error handling**: the app never shows success when cleaning failed. A failed run shows the last error and the last generated code. An empty or unreadable CSV, a rejected API key and network problems each get a clear message instead of a traceback.

### Quality metrics and change report

After a successful run the agent computes before/after metrics (rows, columns, missing cells, duplicate rows, outliers per column, columns dropped or added, cells changed) and asks the model to describe them in a few bullet points. The report uses only numbers from the metrics. It costs one extra model call per run, and if that call fails the metrics are still shown.

Outlier counts always use the IQR fences of the raw data. For whole-number columns the fences are widened to whole numbers, so rounding a capped value does not count as an outlier.

### Python API additions

```python
agent.invoke_agent(data_raw=df, user_instructions="...", max_retries=3)

agent.get_data_cleaned()       # DataFrame, or None if cleaning failed
agent.get_quality_metrics()    # dict of before/after metrics, or None
agent.get_change_summary()     # plain-language report, or None
agent.get_error()              # last execution error, or None on success
agent.get_retry_count()        # number of fix attempts used
```

**Behavior change**: `get_data_cleaned()` now returns `None` when cleaning fails. It used to return an empty DataFrame.

Generated code that raises, has a syntax error, is missing the cleaning function, or returns something other than a non-empty DataFrame is treated as an error and sent through the retry loop.

### Tests and sample data

```bash
poetry run pytest
```

The suite (51 tests) runs offline with a fake chat model, so it makes no API calls and costs nothing. It covers the metrics, the failure paths, the agent graph and the Streamlit app. `data/messy_sample.csv` is a 100-row dataset with injected outliers, whitespace and case variants, missing values, near-duplicates and a mostly empty column. Use it to try the app.

**Note**: because the cleaning code is written by an LLM, its exact form can vary between runs. The metrics and the Quality tab are the way to check what a run actually did.

## Run with Docker

The app can run in a container, so you don't need Python or Poetry installed. You only need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or another Docker engine) running.

### Build and run

```bash
docker build -t data-cleaning-agent .

# Option A: pass your .env at runtime (the key is never baked into the image)
docker run --rm -p 8501:8501 --env-file .env data-cleaning-agent

# Option B: no key; the app asks for it in the sidebar
docker run --rm -p 8501:8501 data-cleaning-agent
```

Then open http://localhost:8501 (not the `0.0.0.0` address printed in the container logs). Press `Ctrl+C` to stop it. If port 8501 is already in use, map another port, for example `-p 8502:8501`, and open that port.

### Make shortcuts

If you have `make`, these wrap the commands above:

| Command | What it does |
|---|---|
| `make docker` | Builds the image, replaces any old container, and starts a new one in the background on port 8501 using your `.env` |
| `make stop` | Stops and removes the container |
| `make run` | Runs the app locally with Poetry (no Docker) |

`make docker` requires a `.env` file with a real `OPENAI_API_KEY`. It also mounts the project's `logs/` folder into the container, so the generated cleaning code (`logs/data_cleaner.py`) is still there after the container is removed. With plain `docker run`, add `-v "$(pwd)/logs:/app/logs"` to get the same effect.

### Notes

- The image is based on `python:3.11-slim`, installs only runtime dependencies (no test tools), and runs as a non-root user.
- `.dockerignore` keeps `.env`, `.git`, `logs/`, `docs/` and `tests/` out of the image.
- The container has a health check. `docker ps` shows `(healthy)` once the app is up, and `curl localhost:8501/_stcore/health` prints `ok`.
- View the app's output with `docker logs -f data-cleaning-agent-app` when started via `make docker`.

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Agent class and LangGraph workflow
│   └── utils.py                # Code execution, metrics and helper functions
├── app.py                      # Streamlit interface
├── ui_helpers.py               # Chart data, CSV reading and error messages for the UI
├── tests/                      # Offline pytest suite
├── data/
│   ├── sample_data.csv
│   └── messy_sample.csv        # Deliberately messy demo dataset
├── docs/superpowers/           # Design specs and implementation plans
├── Dockerfile                  # Container image for the Streamlit app
├── .dockerignore               # Keeps secrets and dev files out of the image
├── Makefile                    # Shortcuts: run, docker, stop
├── pyproject.toml              # Dependencies configuration
├── poetry.lock                 # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
