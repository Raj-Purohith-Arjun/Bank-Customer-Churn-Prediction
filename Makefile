.PHONY: install prepare features train evaluate explain survival monitor test lint clean all

# Install deps
install:
	pip install -r requirements.txt

# Prepare data
prepare:
	python src/data_prep.py

# Build features
features:
	python src/features.py

# Train models
train:
	python src/train.py

# Evaluate models
evaluate:
	python src/evaluate.py

# Explain models
explain:
	python src/explain.py

# Survival analysis
survival:
	python src/survival.py

# Run monitoring
monitor:
	python src/monitoring.py

# Run tests
test:
	pytest tests/ -v

# Lint code
lint:
	flake8 src/ --max-line-length=120

# Clean artifacts
clean:
	rm -rf data/processed/*.pkl
	rm -rf models/*.pkl
	rm -rf figs/*.png

# Full pipeline
all: prepare features train evaluate explain
