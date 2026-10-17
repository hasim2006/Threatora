.PHONY: help install test lint run-unsw run-cicids run-all clean docker-build docker-run

help:
	@echo "Threatora Development & Execution Tasks:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run pytest test suite"
	@echo "  make lint          Lint code with flake8 and black"
	@echo "  make run-all       Run unified pipeline on all benchmark datasets"
	@echo "  make run-unsw      Run standalone UNSW-NB15 benchmark"
	@echo "  make run-cicids    Run standalone CSE-CIC-IDS2018 benchmark"
	@echo "  make clean         Remove build, cache, and temporary files"
	@echo "  make docker-build  Build Docker container"
	@echo "  make docker-run    Run Docker container"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest flake8 black

test:
	pytest tests/ -v

lint:
	flake8 . --count --max-line-length=120 --statistics
	black --check .

run-all:
	python main_unified_pipeline.py --dataset all

run-unsw:
	python pipelines/pipeline_unsw_nb15.py

run-cicids:
	python pipelines/pipeline_cicids2018.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t threatora:latest .

docker-run:
	docker run --rm -v $(PWD)/outputs:/app/outputs threatora:latest
