# Makefile for Energy Demand Forecaster

PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe
PYTEST := .venv/Scripts/pytest.exe
RUFF := .venv/Scripts/ruff.exe

.PHONY: setup data train evaluate app test retrain docker-build docker-up lint clean

setup:
	@echo "Creating virtual environment and installing dependencies..."
	py -3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	@echo "Downloading and caching datasets..."
	$(PYTHON) -m src.data.download
	@echo "Preprocessing and feature engineering..."
	$(PYTHON) -m src.data.preprocess

train:
	@echo "Running core neural model training and multi-seed benchmark..."
	$(PYTHON) -m src.training.train

evaluate:
	@echo "Generating final evaluation figures and comparison reports..."
	$(PYTHON) -c "from src.training.train import run_full_training_pipeline; run_full_training_pipeline()"

app:
	@echo "Starting FastAPI and Streamlit dashboard..."
	$(PYTHON) -m streamlit run src/serving/app_streamlit.py

test:
	@echo "Running unit tests for data leakage and models..."
	$(PYTEST) tests/ -v

retrain:
	@echo "Triggering automated retrain on detected concept drift..."
	$(PYTHON) -m src.training.train

lint:
	@echo "Linting code with ruff..."
	$(RUFF) check src/ tests/

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache mlruns.db
