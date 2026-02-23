.PHONY: install lint test test-cov docker-build-lambda docker-build-processing tf-plan tf-apply streamlit

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests configs
	ruff format --check src tests configs

lint-fix:
	ruff check --fix src tests configs
	ruff format src tests configs

test:
	python -m pytest

test-cov:
	python -m pytest --cov=src --cov=configs --cov-report=term-missing --cov-report=html

docker-build-lambda:
	docker build -f Dockerfile.lambda -t spanishgas-lambda .

docker-build-processing:
	docker build -f Dockerfile.processing -t spanishgas-processing .

tf-plan:
	cd infra/terraform && terraform plan -var-file=environments/dev.tfvars

tf-apply:
	cd infra/terraform && terraform apply -var-file=environments/dev.tfvars

streamlit:
	streamlit run src/serving/ui/app.py --server.headless=true
