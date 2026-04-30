PROJECT_NAME = conformal-prediction-uncertainty-aware-medical-ai

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:
	black src/ tests/ && isort src/ tests/ --profile black && flake8 src/ tests/ && mypy src/

test:
	pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

security:
	bandit -r src/ -ll -ii

complexity:
	radon cc src/ -nc

docstrings:
	interrogate src/ --fail-under=80

audit:
	pip-audit -r requirements.txt && detect-secrets scan --baseline .secrets.baseline

# CLAUDE.md rule C46 — single source of truth for "is this push-ready?".
# This MUST mirror .github/workflows/ci.yml step-for-step. Drift is a P0 bug.
ci:
	black --check src/ tests/
	isort --check-only src/ tests/ --profile black
	flake8 src/ tests/
	mypy src/
	bandit -r src/ -ll -ii
	radon cc src/ -nc
	interrogate src/ --fail-under=80
	pip-audit -r requirements.txt
	detect-secrets scan --baseline .secrets.baseline
	@test -f data/raw/heart.csv || (mkdir -p data/raw && curl -fsSL -o data/raw/heart.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
	pytest tests/ -v --tb=short --cov=src --cov-fail-under=70
	@echo ""
	@echo "✅ All CI gates green. Safe to git push."

train:
	python -m src.training.train --config config/config.yaml

evaluate:
	python -m src.evaluation.evaluate --config config/config.yaml

serve:
	uvicorn src.api.app:app --reload --port 8000
	# → http://localhost:8000/docs (Swagger UI)
	# → http://localhost:8000/metrics (Prometheus, includes coverage_violations_total)
	# → http://localhost:8000/api/v1/health (model_loaded + uptime + memory)

gradio:
	python -m src.api.gradio_demo

streamlit:
	streamlit run app.py

docker-build:
	docker build -t $(PROJECT_NAME) .

docker-run:
	docker run -p 8000:8000 $(PROJECT_NAME)

dvc-pull:
	dvc pull

dvc-push:
	dvc push
