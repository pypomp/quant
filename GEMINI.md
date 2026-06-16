# GEMINI.md

## Environment Setup
Always activate the virtual environment before running tests or commands:
```bash
source .venv/bin/activate
```

## Pypomp Trace Structure
The `pomp_obj.traces()` dataframe contains:
- `replicate`: Integer ID for the particle filter run.
- `iteration`: The step in the algorithm (MIF/IF2 iterations start at 0; Train/IFAD usually starts after).
- `method`: 
    - `mif`: Iterations from the IF2 algorithm.
    - `train`: Training iterations in IFAD.
    - `pfilter`: Final logLik evaluation steps.
- `logLik`: The log-likelihood estimate for that iteration.
- Parameter columns (e.g., `R0`, `sigma`, `gamma`).

## Troubleshooting
- **JAX/CUDA**: If encountering `RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed`, it usually indicates JAX inability to load CUDA libraries in the current shell environment. This is fine. You are probably working on the computing cluster login node, which does not have a GPU. The code will still run.