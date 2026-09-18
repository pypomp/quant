.PHONY: sync sync-gpu lock install_requirements install_pypi install_git install_git_latest list test-interactive test-high test-all render-reports render-reports-slurm lint format

sync:
	uv sync

sync-gpu:
	uv sync --extra gpu

lock:
	uv lock

install_requirements: sync

install_pypi: sync
	uv pip install pypomp

install_git: sync
	uv pip install git+https://github.com/pypomp/pypomp.git

install_git_latest: sync
	uv pip install git+https://github.com/pypomp/pypomp.git --force-reinstall --no-deps

.venv:
	uv sync

list:
	uv run scripts/run_tests.py list tests

test-interactive:
	uv run scripts/run_tests.py run tests --interactive

test-high:
	uv run scripts/run_tests.py run tests --importance high

test-all:
	uv run scripts/run_tests.py run tests

DIR ?= tests

render-reports:
	find $(DIR) -name "*.qmd" -exec quarto render {} \;

render-reports-slurm:
	find $(DIR) -name "*.qmd" -exec sbatch scripts/render_report.sh {} \;

lint:
	uv run ruff check .

format:
	uv run ruff format .



