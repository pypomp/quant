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

## Usage

The runner takes a command (either `run` or `list`) and a target, which can be a single file or an entire directory. If you provide a directory, it recursively finds every `.py` or `.R` script containing a `--- SLURM CONFIG ---` block.

```bash
# List all tests anywhere in the repository:
python scripts/run_tests.py list

# The above command is a bit slow, so maybe limit it to a specific folder, e.g.,:
python scripts/run_tests.py list tests/

# Run a single test:
python scripts/run_tests.py run tests/spx/performance/test.py

# Run all tests in a directory:
python scripts/run_tests.py run tests/spx/
```

### Setting the Run Level

Run levels dynamically modify the execution time (and potentially other args) using the `run_levels` lookup in your YAML config. 

You can set the run level via an environment variable (similar to traditional Makefiles) or via an explicit CLI argument:

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

If you ONLY want to test one configuration without spinning up the other:
```bash
python scripts/run_tests.py run tests/spx/performance/test.py --run-level 2 --job cpu
```
This tells the tool to exclusively run the job named `cpu` and to ignore the `gpu` job definition in that file.

### Running All Tests in a Directory
If you want to run every test script within a directory (that has a `SLURM CONFIG` block), simply point the script to the directory:
```bash
python scripts/run_tests.py run tests/spx/performance/
```

### Testing a Run (`--dry-run`)
If you want to view the `sbatch` script that the python runner dynamically constructs before submitting it to the cluster, use the `--dry-run` flag:
```bash
python scripts/run_tests.py run tests/spx/performance/test.py --run-level 2 --dry-run
```
