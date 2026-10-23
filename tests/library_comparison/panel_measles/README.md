# He10 measles, 4-town panel: pypomp vs pfjax, cuthbert, pydpf, particles, pypfilt

The paper's panel benchmark: London, Halesworth, Hastings and Cardiff, model 001b in
each town, shared R0, σ, γ, σ_SE, cohort, amplitude and town-specific ι, ρ, ψ, S_0, E_0,
I_0, R_0. Given the parameters the towns are independent, so the panel loglik is the sum
of four independent unit filters; this matches `PanelPomp.pfilter`, which runs one filter
per (replicate, unit) and sums the unit logliks. A "filter" below is one unit filter; one
panel evaluation is four of them.

The code is shared with London (`../measles/`, see its README for the model, the
per-library implementations and the CLI). Here:

| File | What it is |
|---|---|
| `export_inputs.py` | Builds `panel_measles(fixed_starts(36))` as `tests/panel_measles/timing/run.py` does (R's covariate grid substituted by `align_covariates`) and writes `inputs.npz` |
| `inputs.npz` | Same fields as London, per unit; `theta` is (36, 4, 13), expanded per unit the way `PanelPomp.pfilter` does |
| `bench_<lib>.py` | Runs `../measles/bench_<lib>.py --panel` with the same arguments; results go to `results/<device>/results.jsonl` here |

All four towns share the observation times and Euler grid (checked at export). Each
result row adds `loglik_by_theta_unit`. pypomp runs `pypomp.functional.panel_pfilter`
with PanelPomp's layout (θ × reps vmapped, keys per unit) and `--chunk-size` units vmapped
together (default all 4; `PanelPomp.pfilter` defaults to 1, i.e. units in sequence).

Full workload (36 θ × 36 reps × 4 towns × J = 5000), from this folder:

```bash
J=~/src/pf-bench/.venv-jax/bin/python; TORCH=~/src/pf-bench/.venv-torch/bin/python
W="--J 5000 --nthetas 36 --nreps 36"
$J bench_pypomp.py $W --device gpu                    # --chunk-size 1 for PanelPomp's default
$J bench_pfjax.py  $W --device gpu                    # add --chunk 864 if memory runs out
~/src/pf-bench/.venv-ssm/bin/python bench_cuthbert.py $W --device gpu
$TORCH bench_pydpf.py $W --device gpu                 # add --chunk 1296 if memory runs out
nice taskset -c <cores> $J bench_particles.py $W --device cpu --procs <cores>
nice taskset -c <cores> $J bench_pypfilt.py  $W --device cpu --procs <cores>
```

## Validation (CPU, J = 1000, full T = 730, first 3 of the 36 starts, 10 reps)

Panel loglik (sum of the four towns), mean ± se over reps; z against pypomp.

| Library | precision | θ1 | θ2 | θ3 |
|---|---|---|---|---|
| pypomp | float32 | −14944.0 ± 68.7 | −21208.8 ± 355.6 | −11099.2 ± 28.9 |
| cuthbert | float32 | −15180.7 ± 90.0 (z −2.09) | −21433.2 ± 324.9 (−0.47) | −11123.0 ± 73.5 (−0.30) |
| pydpf | float32 | −15019.1 ± 64.2 (z −0.80) | −20526.9 ± 435.5 (1.21) | −11098.2 ± 15.8 (0.03) |
| particles | float64 | −14564.0 ± 121.3 (2.73) | −20965.7 ± 336.4 (0.50) | −11095.0 ± 21.9 (0.12) |
| pypfilt | float64 | −14424.9 ± 125.2 (3.63) | −21010.1 ± 358.6 (0.39) | −11136.1 ± 22.4 (−1.01) |

cuthbert's θ1 is low against pypomp at 10 reps (Cardiff carries it: −5848 against pypomp's
−5613) and its θ3 se is large (one filter at −11750). With 50 reps each on GPU 0 (J = 1000,
`--tag validation_n50` in `results/gpu/results.jsonl`) the two agree:

| Library | reps | θ1 | θ2 | θ3 |
|---|---|---|---|---|
| pypomp | 50 | −14985.0 ± 44.9 | −21150.7 ± 172.8 | −11083.5 ± 12.8 |
| cuthbert | 50 | −14913.2 ± 37.5 (z 1.23) | −21400.6 ± 120.5 (−1.19) | −11071.3 ± 11.4 (0.71) |

θ1 splits by precision, not by library. The two checks below, both at θ1 with 10 reps,
put each side on the other's precision:

| Run | θ1 |
|---|---|
| particles, NumPy dmeas with CDFs rounded to float32 (`PFB_DMEAS_F32=1`) | −14952.9 ± 68.2 (matches pypomp) |
| pydpf in float64 (`--dtype float64`) | −14361.7 ± 123.8 (matches particles, pypfilt) |

Cause: at θ1 the London and Cardiff filters often see counts far above every particle's
predicted mean; in float32 both CDFs in He10's `Φ(y+½) − Φ(y−½)` round to 1 and the
likelihood floors at the 1e-18 tolerance, where float64 still resolves the tail (see the
London README). The adapters implement the same model; the float32 libraries (pypomp,
pydpf, pfjax, cuthbert) and the float64 ones (particles, pypfilt) differ only through this.

pfjax on CPU costs about 17 µs per particle-step, too slow for a full-T panel run on the
shared box, so pfjax's per-unit wiring was checked on the first 26 observations (half a
year) against the others at the same T (J = 1000, 10 reps; pfjax 8 reps):

| Library | θ1 | θ2 | θ3 |
|---|---|---|---|
| pypomp | −895.6 ± 13.5 | −1033.5 ± 15.7 | −475.0 ± 2.1 |
| pydpf | −886.7 ± 13.2 | −1035.0 ± 12.8 | −475.5 ± 1.1 |
| particles | −859.6 ± 15.0 | −1023.0 ± 13.1 | −475.3 ± 1.9 |
| pfjax | −890.8 ± 17.4 (z vs pypomp 0.2) | −1034.6 ± 11.7 (−0.1) | −473.8 ± 1.3 (0.5) |

pfjax's per-town means track the others town by town (e.g. θ3: −231, −36, −53, −154 vs
pypomp's −231, −36, −53, −156). Regenerate the table with
`python ../measles/compare.py results/cpu/results.jsonl --tag validation_T26`.

A smaller smoke run (T = 3, J = 20, 3 reps, `--chunk 5`) gives per-town means within noise
of pypomp's for every town, e.g. θ1: pfjax −23.0, −3.1, −6.3, −9.3 vs pypomp −23.0, −3.3,
−6.4, −9.2 (London, Halesworth, Hastings, Cardiff).

## Single-evaluation CPU timings (J = 5000, one θ, 4 towns, full T, one process)

Same conditions as London (`nice -n 19 taskset -c 8-31`, other users' load 47-95 on 32
threads); CPU seconds are the steadier number.

| Library | run | wall cold / warm (s) | CPU cold / warm (s) | load average (1 min) |
|---|---|---|---|---|
| particles | 1 | 127.5 / 85.7 | n/a | 47 |
| particles | 2 | 438.2 / 577.2 | 65.1 / 60.2 | 75-80 |
| pypfilt | 2 | 3305.6 / 3414.1 | 118.5 / 122.0 | 76-99 |

A panel evaluation costs about four London filters (15-16 CPU-s per town for particles at
J = 5000, the same as London's 12.5-23 s; the J = 1000 validation runs show the same for
pypfilt, about 10 s per town vs 13 s for London). Projection for the full panel workload
(1,296 evaluations = 5,184 unit filters): particles about 18-23 CPU-hours, pypfilt about
4 × the London figure, 25-45 CPU-hours; on 36 free cores roughly 30-40 and 40-75 minutes.
