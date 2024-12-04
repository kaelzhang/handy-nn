files = handy_nn test *.py
test_files = *

test:
	pytest -s -v test/test_$(test_files).py --doctest-modules --cov handy_nn --cov-config=.coveragerc --cov-report term-missing

lint:
	@echo "Running ruff..."
	@ruff check $(files)
	@echo "Running mypy..."
	@mypy $(files)

fix:
	ruff check --fix $(files)

install:
	pip install -U .[dev]

install-all:
	pip install -U .[dev,doc]

report:
	codecov

build: handy_nn
	rm -rf dist
	python -m build

publish:
	make build
	twine upload --config-file ~/.pypirc -r pypi dist/*

.PHONY: test build
