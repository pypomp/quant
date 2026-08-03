.PHONY: install_requirements install_pypi install_git install_git_latest list test-interactive test-high test-all freeze-r check-r check

install_pypi: install_requirements
	pip install pypomp

install_git: install_requirements
	pip install git+https://github.com/pypomp/pypomp.git

install_git_latest: install_requirements
	pip install git+https://github.com/pypomp/pypomp.git --force-reinstall --no-deps

install_requirements: .venv
	pip install -r requirements.txt

.venv:
	python3.12 -m venv .venv

list:
	.venv/bin/python scripts/run_tests.py list tests

test-interactive:
	.venv/bin/python scripts/run_tests.py run tests --interactive

test-high:
	.venv/bin/python scripts/run_tests.py run tests --importance high

test-all:
	.venv/bin/python scripts/run_tests.py run tests

# Re-extract the frozen R/pomp baselines from the (gitignored) .rds/.rda files.
# Only needed after bumping pomp and re-running the R scripts.
freeze-r:
	.venv/bin/python scripts/freeze_r_results.py freeze

# Verify the committed R_reference/ CSVs still match their manifests. Cheap
# enough for CI and needs neither R nor the original .rds/.rda files.
check-r:
	.venv/bin/python scripts/freeze_r_results.py check

# Evaluate every test's expect.yaml against its most recent results.
check:
	.venv/bin/python scripts/check_expectations.py tests --all

