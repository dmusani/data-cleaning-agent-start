FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install Poetry (same version that generated poetry.lock), then the runtime
# dependencies. These files are copied first so this layer is cached until
# the dependencies change.
RUN pip install "poetry==2.1.3"
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# Application code
COPY app.py ui_helpers.py ./
COPY data_cleaning_agent ./data_cleaning_agent
COPY data ./data

# Run as a non-root user; the app writes generated code to ./logs
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
