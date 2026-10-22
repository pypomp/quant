# Particle filter libraries: pypomp vs pfjax, pydpf, particles, pypfilt

Bootstrap particle filter log-likelihood on the same user-defined models in five Python
libraries, for the Pypomp JMLR article.

| Library | Backend | Devices | Resampling used |
|---|---|---|---|
| pypomp | JAX | GPU, CPU | systematic, every step |
| pfjax | JAX | GPU, CPU | multinomial (the only standard resampler it ships) |
| pydpf | PyTorch | GPU, CPU | systematic |
| particles | NumPy | CPU | systematic (`ESSrmin=1`) |
| pypfilt | NumPy | CPU | `deterministic` (systematic), threshold 1 |

## Models (`common.py`)

- `lg`: 1-D linear Gaussian, T = 100, one process step per observation. The exact
  log-likelihood (Kalman) is -184.111, which every library reproduces.
- `sde`: log-scale logistic SDE, T = 100, 10 Euler-Maruyama substeps per observation,
  Gaussian measurement. It has no closed form; all libraries agree on about -24.3.
- `he10` (pypomp only, `bench_he10.py`): London measles, model 001b, 5,110 daily Euler
  steps. It shows how much of the GPU time per Euler step is fixed overhead.

Every model is written the way a user would write it in each library: per-particle
`rproc` for pypomp (plus its `@vectorized` option), `BaseModel` for pfjax, `pydpf.Module`s,
`StateSpaceModel` for particles, and a scenario TOML for pypfilt. Libraries that observe
the initial state (particles, pfjax, pydpf) get an initial sampler that pushes x0 one
observation interval forward, so all of them target p(y_1, ..., y_T).

## Running

```bash
./setup_envs.sh          # .venv-jax (pypomp, pfjax, particles, pypfilt), .venv-torch (pydpf)
./smoke.sh               # CPU correctness check at J = 2,000
GPU=0 ./run_gpu.sh       # pypomp, pfjax, pydpf on one GPU; then He10
./run_cpu.sh             # particles and pypfilt (CPU-only libraries), pinned to 4 cores
GPU=0 ./run_exp.sh       # pypomp per-step overhead experiments (see below)
```

Each (library, model, J, replicates) runs in its own process. The first call (which
includes JIT compilation for JAX) and the median of 5 more calls go to
`results/<backend>/results.jsonl`, along with the mean and sd of the log-likelihood.
`PFB_REPS` sets how many filters run in parallel: vmapped parameter replicates in pypomp,
`jax.vmap` in pfjax, the batch dimension in pydpf.

## Overhead experiments (`run_exp.sh`)

`pypomp_substep_unroll_toggle.patch` adds a benchmark-only environment switch,
`PYPOMP_SUBSTEP_UNROLL`, that fully unrolls the Euler-substep loop when the number of
substeps is fixed. `run_exp.sh` compares stock pypomp against that switch, against XLA
command buffers for while loops, against both together, and against the `unsafe_rbg` PRNG.

## Caveats

- Hardware: an RTX 3090 (consumer GPU, kevtan-linux), not the GPU used for the article's
  other timings.
- JAX and PyTorch run in float32; particles and pypfilt run in float64.
- pfjax splits one PRNG key per particle with Python unpacking (`key, *subkeys = split(...)`),
  so its compile time grows with J. Runs that exceed the timeout are recorded as failed.
