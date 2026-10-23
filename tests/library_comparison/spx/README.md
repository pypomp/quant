# SPX stochastic volatility with leverage: pypomp vs pfjax, cuthbert, pydpf, particles, pypfilt

The S&P 500 model of Bretó (2014), `pypomp.models.spx()` (the model behind
`tests/spx/`), ported to five other Python particle-filter libraries so the
paper's SPX pfilter timing (36 parameter vectors × 24 filters × J = 1,000) can
gain rows for them.

## Model

T = 3,523 daily log returns, one process step per observation. The step into
observation n reads the previous return `y_prev = covars_extended[n]` (0 for the
first step; this is pypomp's covariate, linearly interpolated onto the step grid,
which here is exactly y_{n-1}):

```
dWs = (y_prev - mu + V/2) / sqrt(V)
dWv = rho dWs + sqrt(1 - rho^2) dZ,          dZ ~ N(0, 1)
S  <- S + S (mu + sqrt(max(V, 1e-12)) dWs)
V  <- max(V + kappa (theta - V) + xi sqrt(V) dWv, 1e-12)
y_n ~ N(mu - V/2, sqrt(V));        x0 = (V_0, 1105) at t0 = 0, deterministic
```

`S` does not enter the likelihood, but pypomp simulates it, so every port does too.

Files: `export_inputs.py` (writes `inputs.npz`), `spx_common.py` (CLI, inputs,
JSON record), `bench_{pypomp,pfjax,cuthbert,pydpf,particles,pypfilt}.py`, `pypfilt_spx.py`
(pypfilt components), `validate_cpu.sh` + `validate.py` (CPU validation),
`run_full.sh` (full workload).

## Inputs (`export_inputs.py` -> `inputs.npz`)

Written from pypomp's own `PompStruct` so all six libraries read identical arrays:
`ys`, `times`, `t0`, `covars_extended` (T+1), `y_prev` (= `covars_extended[:T]`,
the covariate each step reads), `dt_array_extended`, `nstep_array` (all 1),
state/parameter/covariate names, `S0`, `V_floor`, and `thetas` (36 × 6, natural
scale, order mu, kappa, theta, xi, rho, V_0). Data arrays are float32 as in the
struct; NumPy libraries upcast them exactly to float64.

`thetas` are the 36 starts that `tests/spx/timing/run.py` draws at RUN_LEVEL 4,
reproduced with the same calls (`jax.random.key(MAIN_SEED)`,
`np.random.seed(MAIN_SEED)`, one split, `model.sample_starts(36, subkey)`).
Note: run.py's pfilter phase runs after `mif`, so it filters at the 36 **mif
endpoints**, not at these starts. A bootstrap filter that resamples every step
does the same work at any theta, so the timing workload is the same; the
log-likelihood values are not comparable to the paper's.

## Ports

| Library | Batching | Resampling (every obs) | Float | RNG | Initial state |
|---|---|---|---|---|---|
| pypomp 1.2.0 | `functional.pfilter`, thetas (n,6), keys (n,reps), under `jax.jit` | systematic (`thresh=0`) | 32 | JAX | rinit at t0 |
| pfjax 0.0.3rc1 | `particle_filter` under `jit(vmap)` over all n·reps filters; `--chunk` = `lax.map` | multinomial (only one shipped) | 32 | JAX | `pf_init` pushes x0 one step |
| cuthbert 0.1.1 | `smc.particle_filter` run by `cuthbert.filter`, `jit(vmap)` over all n·reps filters; `--chunk` = `lax.map` | `cuthbertlib` systematic | 32 | JAX | `init_sample` at t0, unweighted |
| pydpf 1.2.3 | batch dim B = n·reps; thetas in `series_metadata`, y_prev in `control` | `SystematicResampler` | 32 | torch.Generator | prior pushes x0 one step |
| particles 0.3 | one filter per task, `--procs` pool | systematic, `ESSrmin=1` | 64 | NumPy global | `PX0` pushes x0 one step |
| pypfilt 0.8.7 | one filter per task, `--procs` pool | `deterministic` (systematic), threshold 1 | 64 | pypfilt model PRNG | model `init` at t0 |

Covariate plumbing: pfjax carries `y_prev` in the measurement pytree
(`y_meas[n] = (y_n, y_prev_n)`, read by an overridden `pf_step`); cuthbert in each step's
model inputs (`{y: y_n, y_prev: y_prev_n}`), with theta closed over; pydpf passes it
as `control`; particles and pypfilt index `y_prev` by the time index. pypfilt's
time axis is the observation index (time n = observation n, one step per unit),
since its steps have fixed length while trading days are 1, 3 or more calendar
days apart; pypomp's rproc ignores dt, so this is exact. pypfilt reports no
marginal likelihood; the observation model accumulates
log Σ w_i p(y_n | x_n^i) with the pre-update weights, as in `../pypfilt_models.py`.

## CLI

```
python bench_<lib>.py --J 1000 --nthetas 36 --nreps 24 --device {cpu,gpu} [--T N] [--procs P] [--seed S]
                      [--theta-set {starts,sun2024}]   (sun2024: validation only, --nthetas 1)
  pypomp:  --layout {nested,flat}   nested = keys (nthetas, nreps) as Pomp.pfilter(reps=) does (default);
                                    flat = every filter in one vmap
  pfjax:   --chunk N                 lax.map over chunks of N filters (default: one vmap)
  cuthbert: --chunk N                as pfjax (run with .venv-ssm)
           --resampler systematic    validation diagnostic only
  pydpf:   --procs P                 torch intra-op threads on CPU (default 1)
```

`--device cpu` sets `CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu` before any
import. CPU-only libraries refuse `--device gpu` and set OMP/MKL/OPENBLAS threads
to 1. Each run appends one JSON line to `results/<device>/results.jsonl`
(`PFB_RESULTS` overrides): lib, model, device, J, nthetas, nreps, T, `cold_s`
(first full workload call, JIT included), `warm_s` (identical second call; same
seeds, results asserted equal), `loglik_by_theta`, `loglik_sd_by_theta`, the raw
`loglik` matrix, versions, variant, dtype, resampler.

## Validation (CPU, J = 1,000, full T, 24 filters per theta)

`./validate_cpu.sh` (results in `results/validation/results.jsonl`, tables from
`validate.py`). Mean ± Monte Carlo SE of the log-likelihood; z = difference from
pypomp in combined SEs. Starts 0–2 are the first three timing starts; `sun2024` is
`SUN2024_THETA` from `tests/spx/model.py` (rho = −0.74, near the MLE), added
because it is the sharp test of the leverage term.

| theta | pypomp | pfjax (multinomial) | cuthbert | pydpf | particles | pypfilt |
|---|---|---|---|---|---|---|
| start 0 | 10881.4 ± 9.6 | 10845.8 ± 13.4 (z −2.2) | 10859.8 ± 10.8 (−1.5) | 10873.5 ± 9.6 (−0.6) | 10880.2 ± 9.1 (−0.1) | 10864.2 ± 10.5 (−1.2)* |
| start 1 | 10923.9 ± 13.6 | 10902.7 ± 23.0 (−0.8) | 10927.8 ± 13.9 (+0.2) | 10919.1 ± 13.8 (−0.3) | 10942.2 ± 11.0 (+1.1) | 10940.7 ± 12.1 (+0.9) |
| start 2 | 11525.94 ± 0.28 | 11525.16 ± 0.41 (−1.6) | 11525.92 ± 0.44 (−0.04) | 11525.39 ± 0.30 (−1.3) | 11525.65 ± 0.37 (−0.6) | 11526.40 ± 0.38 (+1.0) |
| sun2024 | 11849.95 ± 0.31 | 11848.35 ± 0.57 (−2.5) | 11849.57 ± 0.38 (−0.8) | 11849.25 ± 0.39 (−1.4) | 11849.17 ± 0.39 (−1.6) | 11850.09 ± 0.47 (+0.3) |

All agree within 3 combined SEs. pfjax runs lowest because multinomial resampling
has a larger variance and E[log L̂] ≈ log L − σ²/2. Compared on mean + sd²/2, its z
is +1.4, +2.1, +0.4 and +0.9. The diagnostic `--resampler systematic` gives pfjax
z = −2.5, +1.4, −1.5 and −1.2.

\* Starts 0 and 1 have rho ≈ +0.85 (the wrong sign for leverage), and the
log-likelihood is badly behaved there: sd ≈ 50–110 at J = 1,000, and now and then
a filter **collapses**. A large negative return drives every particle's V to the
1e-12 floor, and the next nonzero return costs about −1e8 or more. pypfilt
(float64, its own robust resampler) collapsed once in 24 filters at start 0, and
the pfjax diagnostic twice at start 1. `validate.py` counts collapses (more than
1,000 below the median) and leaves them out. This is model behaviour at those
parameters, not a port bug, but it will show up as huge negative values in
`loglik_by_theta` for the full 36-start workload.

## CPU-only libraries: one filter (J = 1,000, one theta, one process, full T)

`results/cpu_single/results.jsonl`, 3 runs each, on kevtan-linux while its load
average was 37–40 on 32 cores (other jobs), so treat these as upper bounds:

| run (1-min load avg) | particles warm / cold (s) | pypfilt warm / cold (s) |
|---|---|---|
| 1 (36.9) | 1.16 / 1.76 | 12.5 / 10.6 |
| 2 (36.1) | 0.70 / 1.40 | 12.8 / 10.9 |
| 3 (39.9) | 0.80 / 1.30 | 13.7 / 13.6 |

Projected full workload (864 filters) from the median warm time: particles
~690 s on one core, ~19 s with 36 workers; pypfilt ~10,800 s on one core,
~300 s with 36 workers (ideal scaling). These runs were not niced or pinned; on
the shared box, use `PIN="nice -n 19 taskset -c 8-31"` with the scripts below.

For reference, the paper's SPX pfilter rows are R pomp 36.4 s (36 cores), pypomp
6.0 s (36-core CPU) and 3.6 s (RTX PRO 6000).

## Running the full workload

```bash
GPU=0 ./run_full.sh gpu        # pypomp (nested and flat), pfjax, cuthbert, pydpf on one GPU -> results/gpu/
PROCS=36 ./run_full.sh cpu     # particles, pypfilt, 36 workers -> results/cpu/
PROCS=36 ./run_full.sh cpu-all # plus pypomp, pfjax, cuthbert, pydpf on CPU
```

## Deviations and caveats

- Theta set: the starts, not run.py's post-mif estimates (see Inputs).
- pfjax: multinomial resampling only, so its log-likelihood has a larger variance.
  Its stock `particle_filter` splits one key per particle with Python unpacking, so
  cold time grows with J. The pf-bench clone's `PFJAX_FIX_KEYSPLIT` toggle is left unset.
- Precision: pypomp, pfjax, cuthbert and pydpf run in float32; particles and pypfilt in float64.
- pypfilt: every filter parses its TOML and builds a context (natural usage, inside
  the timing). Its resampler loops in Python. `history_window` makes no difference
  (tested at T = 1,000).
- pydpf on CPU: torch's default thread count (one per core) ran 100× slower on the
  loaded box, so `--procs` sets the threads (default 1). GPU runs are unaffected.
- Loglik spread: at start 2 and sun2024, pypomp's sd (1.36, 1.50) is a little
  below the other systematic resamplers (1.5–2.3). That is not significant at 24
  filters, but it is worth a look at larger nreps.
