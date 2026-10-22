# Dhaka cholera (dacca) in five particle-filter libraries

This ports the paper's Dhaka cholera benchmark to pfjax, pydpf, particles and pypfilt, so
the pfilter timing table (36 parameter vectors x 36 replicate filters x J = 5,000 particles)
can include them. The benchmark is `tests/dacca`: pypomp's `dacca()` with 20 Euler steps per
monthly observation, T = 600, 12,000 Euler steps. Every adapter runs the same bootstrap
filter on the same inputs.

## Files

| File | What |
|---|---|
| `export_inputs.py` | builds the model as quant does and writes `inputs.npz` (run with `.venv-jax`) |
| `inputs.npz` | ys, times, t0, `nstep`, per-step `dt`, covariates at every Euler step (`covars` = pypomp's `covars_extended`, float32), the 36 `fixed_starts` (`thetas`), pypomp's built-in parameters (`theta_published`), names |
| `dacca_common.py` | CLI, inputs (truncation, per-interval blocks), result rows, CPU process pool; its docstring states the model |
| `dacca_numpy.py` | the model in float64 numpy (particles, pypfilt) |
| `bench_{pypomp,pfjax,pydpf,particles,pypfilt}.py` | one adapter per library |
| `pypfilt_dacca.py` | pypfilt model and observation components (loaded by dotted path from the scenario TOML) |
| `compare.py` | per-theta mean +/- se of each library against pypomp, with z-scores |
| `results/validation_cpu*.jsonl`, `results/cpu/single_filter_J5000.jsonl` | the runs below |

## Model and conventions

The math is the same as `pypomp/models/dacca.py`. The covariates are interpolated at the
start of every Euler step; they are taken from pypomp's struct, not recomputed. `Mn` resets at
the start of each observation interval. `count` flags any compartment that goes negative
before clipping, and it never resets. The measurement density is
`logaddexp(log N(y; Mn, tau Mn + 1e-18), log 1e-18)`, or `log 1e-18` once `count > 0`. `rinit`
is deterministic given theta.

All five target p(y_1..y_T | theta). particles, pfjax and pydpf weight their initial state,
so their initial sampler draws x(t0) and pushes it through the first observation interval
before the first weighting. Every library resamples at every observation. All use systematic
resampling except pfjax, whose only standard resampler is multinomial. particles
(`ESSrmin = 1`) and pypfilt (`threshold = 1`) skip resampling only when all weights are
exactly equal.

| Library | Precision | RNG | Batching |
|---|---|---|---|
| pypomp | float32 | jax.random | `functional.pfilter`. `--layout map` is `Pomp.pfilter`'s layout (vmap thetas, `lax.map` reps); `--layout vmap` vmaps all nthetas x nreps filters; default `auto` = vmap on GPU, map on CPU |
| pfjax | float32 | jax.random | `jit(vmap)` over all filters; `--chunk n` = `lax.map(batch_size=n)` |
| pydpf | float32 | one `torch.Generator` | all filters in one batch, theta per batch element via `series_metadata`; `--chunk n` |
| particles | float64 | numpy's global `np.random` (what particles' own distributions and resamplers use), seeded per filter | one filter per task, `--procs` single-threaded workers |
| pypfilt | float64 | numpy Generators seeded by pypfilt (`prng_seed`) | as particles |

## CLI

All adapters take `--J --nthetas --nreps --device {cpu,gpu} --T --procs --seed --theta-set {starts,published}`.
Each runs the nthetas x nreps workload twice and appends one JSON line to
`results/<device>/results.jsonl` (`PFB_RESULTS` overrides the file). The line holds `cold_s`
(first call, including JIT compilation) and `warm_s` (second call, fresh seeds), the
per-theta mean and sd of the first call's logliks, versions, dtype, resampler, layout or
chunk, and the load average and CPU set at the start and end. `--T` truncates the series.
`--theta-set published` uses pypomp's built-in dacca parameters, one vector near the MLE with
a small Monte Carlo sd, for validation.

```bash
cd tests/library_comparison/dacca
VJ=~/src/pf-bench/.venv-jax/bin/python VT=~/src/pf-bench/.venv-torch/bin/python
$VJ export_inputs.py        # writes inputs.npz next to the scripts
# GPU libraries, one card (not GPU 0 while another benchmark is timing it)
export CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false
$VJ bench_pypomp.py --J 5000 --nthetas 36 --nreps 36 --device gpu --layout vmap
$VJ bench_pypomp.py --J 5000 --nthetas 36 --nreps 36 --device gpu --layout map   # the paper's layout
$VJ bench_pfjax.py  --J 5000 --nthetas 36 --nreps 36 --device gpu    # add --chunk 324 if out of memory
$VT bench_pydpf.py  --J 5000 --nthetas 36 --nreps 36 --device gpu    # likewise
# CPU-only libraries (on the shared box: nice -n 19 taskset -c 8-31, at most 8 workers)
CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 8-31 $VJ bench_particles.py --J 5000 --nthetas 36 --nreps 36 --procs 8
CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 8-31 $VJ bench_pypfilt.py   --J 5000 --nthetas 36 --nreps 36 --procs 8
```

## Validation (CPU, J = 1,000, full T = 600)

Mean loglik +/- standard error over replicate filters, with z = difference from pypomp over
the combined standard error. Nothing here was run on a GPU.

First three `fixed_starts`, 20 filters each (`results/validation_cpu.jsonl`):

| Library | theta 1 | theta 2 | theta 3 |
|---|---|---|---|
| pypomp | -7097.5 +/- 5.6 | -5890.9 +/- 1.1 | -12324.9 +/- 69.3 |
| pfjax | -7102.8 +/- 5.4 (z -0.69) | -5894.0 +/- 2.0 (z -1.37) | -12352.9 +/- 73.7 (z -0.28) |
| pydpf | -7094.5 +/- 7.2 (z +0.32) | -5890.4 +/- 1.9 (z +0.25) | -12221.6 +/- 73.1 (z +1.03) |
| particles | -7095.1 +/- 6.0 (z +0.28) | -5893.1 +/- 1.4 (z -1.21) | -12409.5 +/- 83.5 (z -0.78) |
| pypfilt | -7086.4 +/- 7.2 (z +1.21) | -5891.8 +/- 2.3 (z -0.34) | -12269.6 +/- 78.9 (z +0.53) |

The starting points are far from the MLE, so their Monte Carlo sds are large (5 to 370).
The sharper check uses pypomp's published parameters, 30 filters each:

| pypomp | pfjax | pydpf | particles | pypfilt |
|---|---|---|---|---|
| -3749.59 +/- 0.26 | -3749.98 +/- 0.34 (z -0.91) | -3749.84 +/- 0.39 (z -0.54) | -3750.03 +/- 0.37 (z -0.97) | -3749.88 +/- 0.36 (z -0.66) |

With 100 filters each (`results/validation_cpu_n100.jsonl`), the sds also agree: pypomp 1.76
(vmap) and 1.99 (map), particles 1.93, pydpf 2.01. With the same seeds, pypfilt's
`ministeps` and `euler` modes return identical logliks.

## Single-filter CPU timings (J = 5,000, theta 1, full T, one process)

These ran under `nice -n 19 taskset -c 8-31` while the 32-thread box had a load average of
35 to 42, so they are inflated. The cold and warm calls are two separate filters
(`results/cpu/single_filter_J5000.jsonl`).

| Library | cold s | warm s | Note |
|---|---|---|---|
| particles | 6.96 | 5.65 | one numpy process |
| pypfilt (ministeps, default) | 15.36 | 11.92 | one numpy process |
| pypfilt (euler, window 30) | 31.69 | 20.55 | one numpy process |
| pydpf (CPU, 1 torch thread) | 5.16 | 4.85 | for reference |
| pfjax (CPU) | 7.76 | 1.58 | cold includes about 6 s of compilation |
| pypomp (CPU, layout map) | 3.28 | 1.06 | for reference; XLA uses the 24 allowed cores |

Projection for the full CPU workload (1,296 filters per call; the script makes two calls):
particles takes 7,300 to 9,000 CPU-s per call, which is 15 to 19 min on 8 workers or about
4 to 5 min on 32 with ideal scaling. pypfilt (ministeps) takes 15,400 to 19,900 CPU-s per
call, which is 32 to 41 min on 8 workers or 8 to 10 min on 32.

## Deviations and limitations

- pfjax resamples multinomially, the only standard resampler it ships. Its loglik sds still
  fall within the spread of the systematic libraries (theta 2: 5.0 to 10.2 across all five).
  pfjax's model methods do not receive the time, so each observation carries its interval's
  covariate block and step sizes (`y_meas = {deaths, cov, dt}`), and the model overrides
  `pf_step` and `pf_init`. pfjax's per-particle key unpacking costs about 6 s of compilation
  at J = 5,000 on CPU. The clone's bench-only `PFJAX_FIX_KEYSPLIT` toggle stays unset, and
  rows record it.
- pypfilt reports no marginal likelihood, so the observation model's `log_llhd` accumulates
  it. Its time axis counts observation intervals; dt in years and the covariates come from
  `inputs.npz` by step index. Theta is passed in the scenario's `[dacca]` table. The default
  `--steps ministeps` runs one pypfilt step per observation with the 20 Euler steps as
  `@ministeps`, and keeps the full history (601 rows). `--steps euler` (one pypfilt step per
  Euler step, as the toy adapters do) needs a sliding history window, because the full history
  is about 5 GB at J = 5,000. Each window slide rescans every step of the run, which costs
  O(T^2 / window); with window 1 a filter at J = 1,000 took about 40 s.
- pypomp, pfjax and pydpf run in float32 (the JAX and torch default); particles and pypfilt
  run in float64. The inputs are the float32 values pypomp uses, upcast for numpy.
- The paper's `timing/run.py` times `pfilter` at the 36 mif endpoints (mif overwrites
  `theta`); these adapters use `fixed_starts(36)`, as specified. Bootstrap-filter run time
  barely depends on theta here, but the logliks differ from the paper's.
- The adapters assume the same number of Euler steps in every interval, which holds for dacca.
- The GPU commands have run only on CPU (`--device cpu`). GPU memory for the 1,296 x 5,000
  batch has not been checked; `--chunk` (pfjax, pydpf) and `--layout map` (pypomp) reduce it.
- On CPU, pypomp's `map` layout beat `vmap` (100 filters at J = 1,000: 25 s against 65 s
  warm). On GPU, run both layouts and report the faster.
