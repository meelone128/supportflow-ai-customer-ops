.PHONY: clean help test eval retrieval-baseline quality

SHELL=/bin/bash

## Remove Python cache files
clean:
	find . -name "__pycache__" -type d -exec rm -r {} \+

## Display help information
help:
	@echo "Available commands:"
	@echo "  make clean         - Remove Python cache files"
	@echo "  make help          - Display this help information"
	@echo "  make quality       - Run deterministic SupportFlow quality gates"

test:
	python -m unittest discover -s supportflow/tests -v

eval:
	python -m supportflow.run_evals

retrieval-baseline:
	python -m supportflow.run_retrieval_experiment

quality: test eval retrieval-baseline

# Default target
.DEFAULT_GOAL := help
