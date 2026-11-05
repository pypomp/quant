# Particle filter libraries: pypomp vs pfjax, cuthbert, pydpf, particles, pypfilt

Bootstrap particle filter log-likelihood on the same user-defined models in six Python
libraries, for the Pypomp JMLR article.

| Library | Backend | Devices | Resampling used |
|---|---|---|---|
| pypomp | JAX | GPU, CPU | systematic, every step |
| pfjax | JAX | GPU, CPU | multinomial (the only standard resampler it ships) |
| cuthbert | JAX | GPU, CPU | systematic (`cuthbertlib.resampling.systematic`), every step |
| pydpf | PyTorch | GPU, CPU | systematic |
| particles | NumPy | CPU | systematic (`ESSrmin=1`) |
| pypfilt | NumPy | CPU | `deterministic` (systematic), threshold 1 |

cuthbert 0.1.1 (`cuthbert.smc.particle_filter.build_filter`, run by `cuthbert.filter`) lives in
its own env, `.venv-ssm`, with the same JAX (0.12.0.1) as `.venv-jax`. Its model is three
per-particle functions, `init_sample(key)`, `propagate_sample(key, x_prev, model_inputs)` and
`log_potential(x_prev, x, model_inputs)`; they see neither theta nor the time, so each adapter
builds the filter inside a function of theta (as cuthbert's own examples do) and passes the
data, covariates and step sizes of each observation interval as that step's model inputs.
cuthbert draws x0 with no weight and weights only after each propagation, so it targets
p(y_1, ..., y_T) without the push-forward below. Its `init_prepare` and the first filter step
split the same key, so with a random initial state, particle i's initial draw reuses the key of
particle i-1's first propagation. That leaves every marginal and the likelihood estimate's
mean unchanged; on the toy models (4,000 filters at J = 200 and 2,000) the logliks match
pypomp's and an init sampler that splits its key first, and the ms models have deterministic
initial states.

dynamax 1.0.3 was looked at and left out. It has Kalman, extended and unscented Kalman,
conditional-moments Gaussian, and HMM filters, plus a Rao-Blackwellised particle filter that
only applies to switching linear-Gaussian models, but no particle filter for a user-written
transition and measurement density, so it cannot run any of these models as a particle
filter. Its Kalman filter would only reproduce the exact `lg` likelihood.

## Models (`common.py`)

- `lg`: 1-D linear Gaussian, T = 100, one process step per observation. The exact
  log-likelihood (Kalman) is -184.111, which every library reproduces.
- `sde`: log-scale logistic SDE, T = 100, 10 Euler-Maruyama substeps per observation,
  Gaussian measurement. It has no closed form; all libraries agree on about -24.3.
- `he10` (pypomp only, `bench_he10.py`): London measles, model 001b, 5,110 daily Euler
  steps. It shows how much of the GPU time per Euler step is fixed overhead.

Every model is written the way a user would write it in each library: per-particle
`rproc` for pypomp (plus its `@vectorized` option), `BaseModel` for pfjax, the three
per-particle functions for cuthbert, `pydpf.Module`s, `StateSpaceModel` for particles, and a
scenario TOML for pypfilt. Libraries that observe the initial state (particles, pfjax, pydpf)
get an initial sampler that pushes x0 one observation interval forward, so all of them target
p(y_1, ..., y_T).

## Running

```bash
./setup_envs.sh          # .venv-jax (pypomp, pfjax, particles, pypfilt), .venv-torch (pydpf), .venv-ssm (cuthbert)
./smoke.sh               # CPU correctness check at J = 2,000
GPU=0 ./run_gpu.sh       # pypomp, pfjax, cuthbert, pydpf on one GPU; then He10
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
  which traces about two ops per particle (2,300 at J = 1,000; 64,000 at J = 30,000), so its
  compile time grows with J (18 minutes at J = 10^6 on the 3090 box). `pfjax_keysplit_fix.patch`
  adds an env switch, `PFJAX_FIX_KEYSPLIT=1`, that slices the key array instead; pfjax is
  reported both as shipped and with that fix (`multinomial/keyfix`).
