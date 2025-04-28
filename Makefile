#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = time_series_model_selection
PYTHON_VERSION = 3.11.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 A Python project for model selection in time series forecasting. This project used purged and embargoed K-fold cross-validation to prevent data leakage.
	isort --check --diff A Python project for model selection in time series forecasting. This project used purged and embargoed K-fold cross-validation to prevent data leakage.
	black --check A Python project for model selection in time series forecasting. This project used purged and embargoed K-fold cross-validation to prevent data leakage.

## Format source code with black
.PHONY: format
format:
	isort A Python project for model selection in time series forecasting. This project used purged and embargoed K-fold cross-validation to prevent data leakage.
	black A Python project for model selection in time series forecasting. This project used purged and embargoed K-fold cross-validation to prevent data leakage.



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) A Python project for model selection in time series forecasting. This project used purged and embargoed K-fold cross-validation to prevent data leakage./dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
