# He10 measles, London: pypomp vs pfjax, pydpf, particles, pypfilt

The paper's London benchmark (pypomp model 001b, 730 weekly observations
1950-1963, 5,110 daily Euler steps) written in each library, so the timing
table can gain rows for the other four. The 4-town panel lives in
`../panel_measles/` and uses the same scripts with `--panel`.

## Files

| File | What it is |
|---|---|
| `export_inputs.py` | Builds the pypomp model exactly as `tests/measles/timing/run.py` does and writes `inputs.npz` |
| `inputs.npz` | Everything the libraries consume (below) |
| `he10.py` | Shared CLI, input loading, result recording, and the NumPy (float64) model used by particles and pypfilt |
| `bench_{pypomp,pfjax,pydpf,particles,pypfilt}.py` | One adapter per library, same CLI |
| `pypfilt_he10.py` | pypfilt model and observation components (loaded by dotted path from the scenario TOML) |
| `compare.py` | Agreement table (mean ± se and z against pypomp and pfjax) from a `results.jsonl` |

## Inputs (`inputs.npz`)

Built as quant's timing script builds it (`measles("London", fixed_starts(36))`, model
001b, uncleaned data as in `clean=False`). All arrays come from pypomp's own struct
(`Pomp.to_struct()`), so they are the values pypomp uses:
`ys` (units × T), `times`, `t0`, `nstep` per interval (all 7), `dt` per Euler step,
`pop` and `birthrate` at every Euler step (index k is the covariate step k uses; index 0
also feeds rinit), and `theta`: the 36 `fixed_starts`, natural scale, as pypomp's float32
array (stored as float64). Also stored, per Euler step: `t_step`, the time pypomp's Euler
loop hands to rproc (float32, previous observation time plus dt once per substep; checked
equal to pypomp's real loop at all 5,110 steps with a probe model), and the two indicators
rproc derives from it, `term` (term time) and `cohort_step` (the cohort-entry day),
evaluated with model 001b's float32 expressions. Recomputing them in float64 flips 16 term
days and moves one cohort day, because some term boundaries land exactly on a grid day, so
every library reads the indicators instead of recomputing them.

Regenerate: `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= ~/src/pf-bench/.venv-jax/bin/python export_inputs.py`.

## Model, as each library runs it

Same maths everywhere: Euler-multinomial exits for S, E, I (rates foi·dW/dt, σ, γ, each
with death rate μ = 0.02), gamma white noise dW, Poisson births with the cohort pulse,
term-time seasonality, W and C reset to 0 at the start of every observation interval
(pypomp's accumvars), He10's discretised-normal dmeasure with tol 1e-18. Every filter
targets p(y_1..y_730 | θ): libraries that weight the initial state (particles, pfjax,
pydpf) draw it at t0 with pypomp's rinit and push it through the first interval first.

| Library | Written as | Samplers | Precision | Resampling (every obs.) | Batching |
|---|---|---|---|---|---|
| pypomp 1.2.0 | its own UKMeasles model 001b, `pypomp.functional.pfilter` / `panel_pfilter` | `pypomp.random` fast inverse-CDF (approximate) | float32 | systematic | vmap over filters |
| pfjax 0.0.3rc1 | `BaseModel`, per particle; `pf_init`/`pf_step` take the interval's grid and covariates from the data pytree | exact `jax.random` gamma/poisson/binomial; `--samplers pypomp` uses pypomp's | float32 | multinomial (pfjax has no systematic) | `jit(vmap)` over filters, `--chunk` = `lax.map` over chunks |
| pydpf 1.2.3 | `pydpf.Module`s in a `ParticleFilter`; per-batch-element θ and covariates | exact torch `_standard_gamma`/`poisson`/`binomial`, one `Generator` | float32 | `SystematicResampler` | pydpf batch dimension |
| particles 0.3 | `StateSpaceModel` with vectorised custom `ProbDist`s, `Bootstrap`, `SMC` | exact NumPy `Generator` | float64 | systematic, `ESSrmin=1` | one filter per process, `--procs` pool |
| pypfilt 0.8.7 | `Model` + `obs.Univariate`, scenario TOML, time counted in Euler steps | exact NumPy `Generator` | float64 | `deterministic` (systematic), threshold 1 | one filter per process, `--procs` pool |

pypfilt reports no marginal likelihood, so `He10Obs.log_llhd` accumulates it from the
pre-update weights. pypfilt runs with `history_window = 49` (steps): its default of keeping
the whole history would allocate about 1.8 GB per filter at J = 5000.

## CLI

```
python bench_<lib>.py --J 5000 --nthetas 36 --nreps 36 --device {cpu,gpu} [--T T] [--procs P] [--seed S] [--calls 1|2] [--tag x]
  pypomp:  --layout flat|api   (flat: all n·reps filters in one vmap; api: Pomp.pfilter's vmap over θ, lax.map over reps)
  pfjax:   --samplers jax|pypomp   --chunk C
  pydpf:   --chunk B   --dtype float32|float64 (float64 only as a precision check);  --procs = torch CPU threads
```

Each run appends one JSON line to `results/<device>/results.jsonl` (override with
`PFB_RESULTS=<file>`): lib, model, device, J, nthetas, nreps, T, `cold_s` (first full
workload call, including JIT compilation), `warm_s` (a second identical call),
`loglik_by_theta` (mean over reps), `loglik_sd_by_theta`, every filter's loglik (small runs),
library versions, samplers, dtype, plus `cold_cpu_s` / `warm_cpu_s` (CPU seconds of the
process and its worker processes, all threads) and the load average at start and end. On a
shared machine the CPU seconds are the more trustworthy cost for the CPU-only libraries. `--device cpu` sets `CUDA_VISIBLE_DEVICES=` and
`JAX_PLATFORMS=cpu` before anything is imported. CPU-only libraries set one BLAS/OpenMP
thread per worker.

Full workloads (36 θ × 36 reps × J = 5000), from this folder:

```bash
J=~/src/pf-bench/.venv-jax/bin/python; TORCH=~/src/pf-bench/.venv-torch/bin/python
W="--J 5000 --nthetas 36 --nreps 36"
$J bench_pypomp.py $W --device gpu                    # also: --layout api
$J bench_pfjax.py  $W --device gpu                    # long compile; add --chunk 216 if memory runs out
$J bench_pfjax.py  $W --device gpu --samplers pypomp  # try with --T 5 first (see limitations)
$TORCH bench_pydpf.py $W --device gpu                 # add --chunk 324 if memory runs out
nice taskset -c <cores> $J bench_particles.py $W --device cpu --procs <cores>
nice taskset -c <cores> $J bench_pypfilt.py  $W --device cpu --procs <cores>
```

## Validation (CPU, J = 1000, full T = 730, first 3 of the 36 starts)

Mean loglik ± standard error over reps; z is the difference over the combined se.

| Library | reps | θ1 | θ2 | θ3 | z vs pypomp | z vs pfjax (exact) |
|---|---|---|---|---|---|---|
| pypomp | 20 | −4038.96 ± 1.36 | −5516.9 ± 63.9 | −6312.1 ± 227.4 | | 1.32, −0.01, 1.77 |
| pfjax (exact samplers) | 10 | −4043.11 ± 2.84 | −5516.0 ± 95.1 | −7357.8 ± 545.1 | −1.32, 0.01, −1.77 | |
| pydpf | 20 | −4038.71 ± 1.67 | −5365.1 ± 54.7 | −6478.1 ± 270.0 | 0.12, 1.81, −0.47 | 1.33, 1.38, 1.45 |
| particles | 20 | −4041.03 ± 1.57 | −5400.3 ± 59.1 | −6119.8 ± 186.0 | −0.99, 1.34, 0.65 | 0.64, 1.03, 2.15 |
| pypfilt | 20 | −4037.86 ± 1.43 | −5441.8 ± 63.5 | −6279.9 ± 232.4 | 0.56, 0.83, 0.10 | 1.65, 0.65, 1.82 |

Every library agrees with pypomp and with exact-sampler pfjax within 3 combined se. θ1 is
the sharp test (se ≈ 1.5); θ2 and θ3 are poor parameter vectors whose loglik has a heavy
lower tail (sd 300-1700 across filters), so they only rule out gross errors.

## Single-filter CPU timings (J = 5000, one θ, full T, one process)

Run with `nice -n 19 taskset -c 8-31` on a box other users kept at load 25-85 (32
threads), so wall times mostly measure the scheduler. CPU seconds (process user + system)
are the steadier number, though SMT sharing still inflates them at high load.

| Library | run | wall cold / warm (s) | CPU cold / warm (s) | load average (1 min) |
|---|---|---|---|---|
| particles | 1 | 17.1 / 12.5 | n/a | 28-32 |
| particles | 2 | 294.6 / 164.8 | 23.1 / 18.2 | 72-85 |
| pypfilt | 1 | 17.3 / 24.9 | n/a | 25-28 |
| pypfilt | 2 | 248.0 / 305.1 | 31.4 / 31.9 | 81-82 |

Projection for the full London workload (36 θ × 36 reps = 1,296 filters at J = 5000), from
the lightly loaded run 1 and the CPU seconds of run 2: particles 12.5-18 s per filter, about
4.5-6.5 CPU-hours; pypfilt 17-32 s per filter, about 6-11.5 CPU-hours. On 36 free cores that
is roughly 8-11 and 10-20 minutes of wall time.

## Deviations and limitations

- **float32 measurement tail.** pypomp's dmeasure is `Φ(y+½) − Φ(y−½)`; in float32 both
  CDFs round to 1 once an observation is more than about 5.4 predictive sd above a
  particle's mean, so the likelihood floors at the 1e-18 tolerance, where float64 still
  resolves it. At poorly fitting parameters this lowers the float32 loglik by hundreds of
  log units (see the panel README for the measured case). It is a property of pypomp's
  float32 model, not of any adapter: pydpf in float64 matches the float64 libraries, and
  the NumPy model with float32-rounded CDFs (`PFB_DMEAS_F32=1`, diagnostic only) matches
  pypomp. R pomp works in double precision, so the same effect separates pypomp from R at
  such parameters. Timings are unaffected.
- pfjax resamples multinomially (it ships no systematic resampler), so its loglik
  estimates have a little more variance.
- pfjax compile time: the exact-sampler filter takes about 1-1.5 minutes to compile on
  CPU even at J = 8 (vmapped rejection samplers), plus pfjax's per-particle key unpacking
  (`key, *subkeys = split(...)`), which grows with J. The clone has a bench-only
  `PFJAX_FIX_KEYSPLIT=1` switch that removes the unpacking; results record whether it was on.
- `pfjax --samplers pypomp` (pypomp's inverse-CDF samplers inside pfjax's per-particle
  vmap) did not finish compiling on CPU within 30+ minutes even at J = 8, T = 3; the single
  Euler step compiles in 3-6 s. Untested end to end; try it on GPU with `--T 5` first.
- pfjax on CPU is slow per particle step (the London validation, 30 filters at J = 1000,
  took 47 minutes); its numbers only matter on GPU.
- CPU-only libraries run one filter per process; their workload time is the pool's wall
  time and scales with free cores.
