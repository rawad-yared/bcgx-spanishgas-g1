.PHONY: lint test

lint:
	python -m compileall src tests

test:
	python -m pytest
