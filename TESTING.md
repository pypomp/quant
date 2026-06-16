# SLURM Test Runner

Our testing suite uses a centralized Python script, `run_tests.py`, at the root of the repository to submit jobs via SLURM. By configuring your test parameters as inline comments inside the script files themselves, the test runner removes the need for isolated `Makefiles` and relative paths.

## 1. Configuring the SLURM Parameters for a File

A test script (Python or R) is identified by the test runner if it contains a `--- SLURM CONFIG ---` block in its comments at the top of the file.

The block is structured as YAML. Note that for R and Python scripts, every line of the YAML metadata must be prefixed by the comment character `#`. 

### Basic Configuration Example (`test.py` or `.R`)

```python
# --- SLURM CONFIG ---
# sbatch_args:
#   partition: standard
#   time: "00:20:00"
#   cpus-per-task: 4
#   mem: 16GB
#   output: "results/logs/slurm-%j.out"
# --- END SLURM CONFIG ---

import os
# ... rest of your code ...
```

### Multi-Job Configuration Example

If a single script tests both CPU and GPU execution methods, you can group them under a `jobs` key. The test runner will automatically generate an `sbatch` submission for each defined job.

```python
# --- SLURM CONFIG ---
# jobs:
#   gpu:
#     sbatch_args:
#       partition: gpu
#       gpus: "v100:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       time: "00:04:00"
#       output: "gpu_results/logs/slurm-%j.out"
#   cpu:
#     sbatch_args:
#       partition: standard
#       cpus-per-task: 36
#       mem: 80GB
#       time: "00:04:00"
#       output: "cpu_results/logs/slurm-%j.out"
#     env:
#       USE_CPU: "true"
# 
# run_levels:
#   1:
#     sbatch_args: { time: "00:00:30" }
#   2:
#     sbatch_args: { time: "00:04:00" }
# --- END SLURM CONFIG ---
```
In this example, running the file creates two distinct SLURM jobs named "gpu" and "cpu". The "cpu" job injects `USE_CPU=true` into the environment dynamically. The `run_levels` section at the bottom defines overrides—if run level `2` is used, the jobs will run using `time: "00:04:00"`, overriding the base job configurations if necessary.

### Test Metadata (Description, Importance, Tags)

You can enrich the SLURM config block with metadata about the test to make it easier to discover and filter:
- **importance**: The priority/importance level of the test (`low`, `medium`, `high`, or `critical`). Defaults to `low` if not specified.
- **description**: A short, single-line description of the test. (If missing, the script will fall back to using the Python module-level docstring or the Roxygen comments at the top of the file.)
- **tags**: A list of tags to categorize the test (e.g. `[performance, spx, gpu]`).

**Example with Metadata:**
```python
# --- SLURM CONFIG ---
# importance: high
# description: "Benchmarks performance and convergence for S&P 500 model on CPU/GPU"
# tags: [performance, spx, gpu, cpu]
# jobs:
#   gpu:
#     sbatch_args:
#       partition: gpu
#       ...
# --- END SLURM CONFIG ---
```

### Global User Configuration (`test_config.yaml`)

Because personal SLURM arguments (like your email address for job completion notifications) should not be committed to the repository, you can create a `test_config.yaml` file at the exact root of your repository (e.g. `/home/user/research/quant/test_config.yaml`).

Any `sbatch_args` you define in this file will automatically be injected into **every** job you submit using the runner script, preserving your preferences globally!

**Example `test_config.yaml`:**
```yaml
sbatch_args:
  mail-type: ALL
  mail-user: your_email@umich.edu
```

*(Note: `test_config.yaml` is permanently added to `.gitignore` so your personal email is never accidentally tracked.)*

---

## 2. Usage and Execution

The runner takes a command (either `run` or `list`) and a target (which can be a single file or a directory). If a directory is provided, it recursively scans for files containing a `--- SLURM CONFIG ---` block.

```bash
# List all tests anywhere in the repository:
python scripts/run_tests.py list

# Run a single test:
python scripts/run_tests.py run tests/spx/performance/test.py

# Run all tests in a directory:
python scripts/run_tests.py run tests/spx/
```

### Filtering Tests

You can filter tests by their `importance` level or specific `tags` during both `list` and `run` actions:

```bash
# List only tests of high (or critical) importance:
python scripts/run_tests.py list --importance high

# Run only tests tagged with 'performance':
python scripts/run_tests.py run --tag performance
```

### Setting the Run Level

Run levels dynamically modify the execution time (and potentially other args) using the `run_levels` lookup in your YAML config. 

You can set the run level via an environment variable or via an explicit CLI argument:

**Method 1: CLI Argument**
```bash
python scripts/run_tests.py run tests/spx/performance/test.py --run-level 2
```

**Method 2: Environmental Variable**
```bash
RUN_LEVEL=2 python scripts/run_tests.py run tests/spx/performance/test.py
```

### Running a Specific Target Job

If a test file has multiple target setups (for example, comparing `cpu` vs `gpu` under a `jobs:` block), running the script targets ALL of those jobs simultaneously.

If you ONLY want to test one configuration:
```bash
python scripts/run_tests.py run tests/spx/performance/test.py --run-level 2 --job cpu
```

### Testing a Run (`--dry-run`)
If you want to view the `sbatch` script that the python runner dynamically constructs before submitting it to the cluster, use the `--dry-run` flag:
```bash
python scripts/run_tests.py run tests/spx/performance/test.py --run-level 2 --dry-run
```

### Interactive Mode

You can run the tool in interactive mode pointing at the `tests` directory using the `--interactive` (or `-i`) flag:

```bash
python scripts/run_tests.py run tests --interactive
# Or using the Makefile shortcut:
make test-interactive
```

This will display a structured menu of available tests in the `tests` directory, prompt you to input a `RUN_LEVEL` if not specified, and let you select which tests to run by typing numbers or ranges (e.g. `1`, `1,3`, `1-3`, `all`).

---

## 3. Makefile Targets

For convenience, several target shortcuts are defined in the root `makefile` to run and list tests using the virtual environment environment automatically:

- `make list`: Beautifully format and list all discovered tests.
- `make test-interactive` / `make test-i`: Start the interactive test selection runner.
- `make test-high`: Run all high (or critical) importance tests.
- `make test-all`: Run all tests in the repository.
