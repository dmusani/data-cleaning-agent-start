# Shortcuts for running the Data Cleaning Agent

.PHONY: run docker stop

run:
	@echo "Starting Streamlit app..."
	@poetry run streamlit run app.py

docker:
	@echo "Building Docker image..."
	docker build -t data-cleaning-agent .
	@echo "Stopping and removing existing container..."
	-@docker stop data-cleaning-agent-app && docker rm data-cleaning-agent-app
	@echo "Running Docker container on http://localhost:8501 ..."
	@mkdir -p logs
	@docker run --name data-cleaning-agent-app -d -p 8501:8501 --env-file .env -v "$(CURDIR)/logs:/app/logs" data-cleaning-agent

stop:
	@echo "Stopping Docker container..."
	-@docker stop data-cleaning-agent-app && docker rm data-cleaning-agent-app
