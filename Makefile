.PHONY: help
help:			## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets: "
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: clean
clean:			## Clean unused files.
	@echo "Cleaning up..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@rm -f .coverage
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf Covid19Classification/*.egg-info
	@rm -rf htmlcov
	@rm -rf docs/_build
	@rm -rf docs/_static


.PHONY: install
install:		## Install in development mode.
	pip install -e .[test]


.PHONY: format
format:			## Format code using isort and black
	isort Covid19Classification/
	isort tests/
	black -l 110 Covid19Classification/
	black -l 110 tests/


.PHONY: lint
lint:			## Run linters
	flake8 Covid19Classification/
	black -l 110 --check Covid19Classification/
	black -l 110 --check tests/
	mypy Covid19Classification/


.PHONY: test
test: lint		## Run tests and generate coverage report
	pytest tests/
	coverage html


.PHONY: docs
docs:			## Build documentation
	@echo "Building documentation..."
	pdoc Covid19Classification -o docs
	@echo "Serving API documentation..." 
	pdoc Covid19Classification --host localhost --port 8080
