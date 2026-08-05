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
python scripts/run_tests.py run tests/spx/estimation/run.py

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
python scripts/run_tests.py run tests/spx/estimation/run.py --run-level 2
```

**Method 2: Environmental Variable**
```bash
RUN_LEVEL=2 python scripts/run_tests.py run tests/spx/estimation/run.py
```

### Running a Specific Target Job

If a test file has multiple target setups (for example, comparing `cpu` vs `gpu` under a `jobs:` block), running the script targets ALL of those jobs simultaneously by default.

If you ONLY want to test one configuration:
```bash
python scripts/run_tests.py run tests/spx/estimation/run.py --run-level 2 --job cpu
```

### Sequential Job Chains & Race Conditions

In some cases, multiple jobs defined in the same test script may depend on each other or share/update the same output/cache files (for example, `tests/samplers/test.qmd` where jobs share `benchmark_results.json` and render to the same `test.html`). Running them concurrently will cause a race condition.

To handle this, you can specify `sequential: true` under the `--- SLURM CONFIG ---` block of the test file:
```yaml
# --- SLURM CONFIG ---
# importance: high
# sequential: true
# jobs:
#   gpu:
#     ...
#   cpu:
#     ...
```

When `sequential: true` is set, `run_tests.py` will automatically chain the SLURM job submissions in the order they are defined using SLURM's `--dependency=afterok:<job_id>` flag, ensuring that each job only executes once the previous job has completed successfully. If running in `--dry-run` mode, the dependency configurations will be printed as part of the output script.

### Testing a Run (`--dry-run`)
If you want to view the `sbatch` script that the python runner dynamically constructs before submitting it to the cluster, use the `--dry-run` flag:
```bash
python scripts/run_tests.py run tests/spx/estimation/run.py --run-level 2 --dry-run
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

- `make list`: Format and list all discovered tests.
- `make test-interactive` / `make test-i`: Start the interactive test selection runner.
- `make test-high`: Run all high (or critical) importance tests.
- `make test-all`: Run all tests in the repository.
- `make freeze-r`: Re-extract the frozen R baselines (see below).
- `make check-r`: Verify the committed R baselines against their manifests.

---

## 4. Test Layout: models and kinds

Tests are organised **model first, then kind**. `tests/spx/` is the migrated
example; the other models still use the older ad-hoc layout.

```
tests/<model>/
    model.py / model.R      # the benchmark's shared setup, defined once
    report.qmd              # one report per model, reading from every kind
    <kind>/
        run.py              # the pypomp entrypoint (always this name)
        run.R               # the R baseline, if there is one
        R_reference/        # frozen R results, committed
        results/<platform>/ # run outputs; CSV/JSON committed, .pkl not
```

There are five kinds. Which one a test belongs to is decided by a single
question: **what varies across the runs inside the test?**

| Kind | What varies | theta | Cost driver |
|---|---|---|---|
| `timing` | nothing (identical work repeated) | fixed | cheap by design |
| `loglik` | replicate seed only | **fixed** | replicate count |
| `estimation` | starting point | **free** | starts x iterations |
| `algorithms` | the algorithm | free | slowest algorithm |
| `scaling` | a size knob (units, J) | fixed | largest size |

### Run outputs

`run.py` calls `save_run()` from `tests/utils.py`, which writes into
`results/<platform>/`:

| File | Committed | What it is |
|---|---|---|
| `fitted.pkl` | no | the whole fitted object -- the fallback |
| `results.csv` | yes | parameter estimates and aggregated logLik |
| `pfilter_logliks.csv` | yes | per-replicate logLiks (`loglik` kind) |
| `traces.csv.gz` | yes | per-iteration traces |
| `timings.csv` | yes | per-phase wall clock |
| `latest.json` | yes | this run's provenance and configuration |

The text files are the main record; the pkl is kept as a fallback, although it is not committed. 
`latest.json` carries various other run info that is not easily recorded in csv files, such as the pypomp and JAX versions, the quant git SHA, the device, the SLURM job id, and algorithmic configuration details.


---

## 5. Frozen R Baselines (`R_reference/`)

Several tests compare `pypomp` against R's `pomp`/`panelPomp`. The R side is by
far the more expensive half — the panel measles parameter comparison is
budgeted **36 hours** of wall clock, against 2 hours for its Python counterpart
— and its results only change when `pomp` changes. So we do not re-run it.

Instead, the small tidy tables each report actually consumes are extracted from
the `.rds`/`.rda` outputs and committed as CSV under `R_reference/`, next to the
test that produced them:

```
tests/measles/R_comparison/parameter_comparison/
    measles.R                      # produces results/mif_coefs.rds (gitignored)
    results/mif_coefs.rds          # scratch output of a fresh run
    R_reference/
        mif_coefs.csv              # committed, human-readable, diffable
        MANIFEST.json              # provenance
    report.qmd                     # reads R_reference/, never results/
```

**`results/` is scratch; `R_reference/` is the record.** Reports read only from
`R_reference/`, so anyone can render a report from a clean checkout without R,
without `pyreadr`, and without the gitignored binaries. That was not previously
true: the SPX report needed a 66 MB `.rda` sitting in a gitignored `_hidden`
directory, so nobody but its author could rebuild it.

Each `MANIFEST.json` records what produced the data, the SHA-256 and mtime of
the source file, the row counts and column names of each frozen CSV, its
SHA-256, and the R/`pomp` versions `renv.lock` pins. That last field is
attribution rather than measurement — for baselines generated before this
tooling existed, the pinned version is the best available evidence for what
produced them, not a reading taken from the job itself.

### Regenerating

Only after bumping `pomp` and re-running the R scripts:

```bash
make freeze-r                                             # everything
python scripts/freeze_r_results.py freeze --only spx      # one entry
python scripts/freeze_r_results.py list                   # what's in the spec
```

`scripts/freeze_r_results.py` holds the spec of what gets frozen. Most entries
are read with `pyreadr` and are bit-exact. The SPX global search is the
exception: its `.rda` holds a 360-element `mif2List` of S4 `pomp` objects that
`pyreadr` cannot read at all, so `scripts/extract_spx_search360.R` pulls out the
likelihoods, IF2 traces, and timings with `pomp` loaded. Those pass through R's
~15-significant-digit CSV output and so agree with the originals to a relative
5e-15 — far below the Monte Carlo error these numbers carry, but worth knowing
before treating them as bit-exact.

`--source-root` lets you read the gitignored source files from another checkout,
which is what makes it possible to freeze into a git worktree.

### Checking

```bash
make check-r
```

verifies every committed CSV still hashes to what its manifest records. It needs
neither R nor the original binaries, so it is cheap enough to run in CI and will
catch a hand-edited baseline or a half-finished re-freeze.
