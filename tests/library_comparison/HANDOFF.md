# pf-bench: pypomp vs other Python particle-filter libraries, and kernel work (2026-10-22, cuthbert 2026-10-23)

## Goal (Kevin's words)
"can you also perf pypomp against pfjax, pypfilt, pydpf, particles?" ... "i want to augment this ms"
(https://github.com/pypomp/tutorials/blob/main/article_jmlr/ms.pdf) ... "bench over the examples in
[the ms], noting that ideally we extend the tables to add the comparisons with other libraries other
than just R pomp". Kernel work: pypomp "has to be pretty general, modular and/or user-friendly ...
take in user-defined rproc". GPU numbers are what matter; CPU ablations are not useful. Use CuTe,
not Triton. On RNG: "i dont care about reproducibility and exact threefry, i care about speed and
correctness". 2026-10-23: "Can you also extend the benchmarks to cover the cuthbert and dynamax
packages?" ... "if dynamax doesn't support PF then let's just skip it."

## State
- **Tables**: `TABLES.md` (from `make_tables.py`): PF time on the four ms workloads for R pomp,
  pypomp (released, parallel reps, CuTe), pfjax (released, key fix), cuthbert, pydpf, and
  particles/pypfilt projected to 36 cores. Complete; pfjax and cuthbert on the panel hit the one-hour
  limit and show as > 3,600 s.
- **Kernel work**: `kernels/README.md`. In pypomp, split into separate PRs to main (combined #54 was
  closed): #55 pyright fixes for jax 0.12, #56 parallel pfilter replicates (branch
  `perf/pfilter-parallel-reps`), #57 the CuTe backend (branch `feature/cute-rproc`, worktree
  `../../../pypomp-wt-cute-pr`), and the measles dmeas fix from its own session (branch
  `measles-dmeas-tails`). test_sir_mif_regression fails on main under jax 0.12; Kevin is raising
  it with Aaron.
- **pfjax**: key-splitting fix submitted as mlysy/pfjax#38 (Kevin's text).
- **cuthbert 0.1.1** (2026-10-23): `bench_cuthbert.py` in the toy folder and in every model
  folder, run from `.venv-ssm` (jax 0.12.0.1, same as `.venv-jax`). Bootstrap filter
  (`cuthbert.smc.particle_filter`, run by `cuthbert.filter`), systematic resampling every
  observation, float32; measles uses exact `jax.random` samplers like pfjax. CPU validation
  against pypomp is in each model README. GPU 0 times (warm, first call): spx 1.7 (2.7),
  dacca 11.6 (12.8) after the adapter fix below (45.7 before), London 1,220 (1,224); the panel was killed at the one-hour limit
  (table shows > 3,600, as for pfjax). See "cuthbert notes" below.
- **Cholera gap** (found 2026-10-23): pfjax and cuthbert were about 4x slower than pypomp on
  dacca only because our adapters counted negative states with `jnp.any(jnp.stack(new) < 0)`.
  XLA builds that 7-row stack inside the per-step kernel, which then runs at about 165 GB/s
  instead of 750 (`kernels/any_stack_repro.py`: 2.5 vs 0.56 ms per step at 1,296 x 5,000), and in
  pfjax it also split the step into two kernels with 150 copies per observation. An OR of the
  seven comparisons (5daf9a5) gives identical log-likelihoods. Reruns: pfjax 11.2 s, cuthbert
  11.6 s, pypomp 1.2 10.6 s. Their step kernel now matches pypomp's (502 vs 475 ms at T = 48);
  the rest is resampling (pfjax's multinomial gather and copies, cuthbert's CUB sort) and drawing
  each interval's noise in a separate kernel.
- **pydpf** runs PyTorch op by op: 81 kernels per Euler step on dacca, each a full pass over the
  particles at close to peak bandwidth (an add takes 93 us, about 840 GB/s), so 5.5 ms per step
  against about 0.5 ms for pypomp's one fused step kernel (`kernels/torch_kernels.py`).
- **Aaron's `cute-real-branches`** (07e627a, on top of `feature/cute-rproc` ac6d0a8), checked on
  the 3090 on 2026-10-23. Warm times, ac6d0a8 -> 07e627a: London interval 17.7 -> 13.7 ms; London PF
  13.9 -> 10.7 s and IF2 4.9 -> 3.8 s; cholera unchanged (PF 3.2 s, IF2 2.2 s); XLA unchanged.
  Sampler outputs on the XLA path are bitwise identical (about 32,000 grid values, batched and
  vmapped). `tests/test_cute`: 31 pass; `test_dpop_train_on_cute` fails on both commits at its first
  line, because Pomp has added `_logw` to `accumvars` since 4e3ee49, and passes with
  `assert model.rproc.accumvars == (1,)`. Rows: `kernels/results/cute_real_branches_3090.jsonl`.
- **Why XLA's He10 step kernel needs 183 registers** (2026-10-26): XLA runs its elementwise kernels
  with 4 particles per thread whenever the element count divides by 4, which it always does with 1,296
  filters. At 1,295 x 4,999 it uses 1 per thread. London's main step kernel then needs 72 registers
  instead of 183 (ptxas) and takes 9.6 ms instead of 43 per interval (7 launches). The London interval drops from 59
  to 24 ms, and the full London XLA PF from 33.7 to 21.0 s. Cholera doesn't change (7.5 s). No kernel
  spills. At 2 per thread (1,295 x 4,998): 108 registers and 10.6 ms per interval. The kernel is straight-line
  code of 47, 92 and 186 KB at 1, 2 and 4 per thread, and the same number of particles is in flight
  per SM at 2 and 4, so the cliff is most likely instruction fetch. That is unconfirmed: the GPU
  counters need admin rights here (ERR_NVGPUCTRPERM; ncu 2022.4 is under ~/anaconda3/nsight-compute).
  Cholera's step kernel is 21 KB at 4 per thread and costs nothing. `xla_gpu_experimental_max_unroll_factor`
  has no effect. Scripts: `kernels/xla_regs.py` (time, registers and particles per thread),
  `kernels/pf_odd.py` (full PF at both sizes), `kernels/ptxas_save.sh` (first on PATH, with
  `--xla_gpu_enable_libnvptxcompiler=false --xla_gpu_enable_libnvjitlink=false`, saves XLA's PTX and
  the ptxas -v report).
  At 1 per thread, XLA's interval is limited by memory bandwidth: its 44 kernels per Euler step move
  2.25 GB per step (15.8 GB per interval, about 690 GB/s on average, the big kernels at 680-860 GB/s),
  because each step's intermediates and the state go through GPU memory between kernels. CuTe reads
  and writes the 6-variable state (155 MB) once per interval and is limited by arithmetic instead
  (`kernels/xla_bytes.py`).
- **dynamax 1.0.3**: skipped, per Kevin. It has Kalman, EKF/UKF, conditional-moments and HMM
  filters, plus a Rao-Blackwellised PF only for switching linear-Gaussian models; nothing that
  runs a user-written transition and measurement density as a particle filter.
- **Correctness**: every library agrees with pypomp within Monte Carlo error on every model
  (per-model READMEs). pypomp's float32 measles dmeas underflowed at poorly fitting parameters;
  fixed on the branch.

## Layout
- `common.py`, `bench_{pypomp,pfjax,cuthbert,particles,pypfilt,pydpf}.py`: toy models (`lg`, `sde`) in each library.
- `{spx,dacca,measles,panel_measles}/`: the ms models ported to each library; `bench_*.py`,
  `export_inputs.py`, `README.md` (validation, single-filter CPU timings, deviations), `results/`.
- `kernels/`: Pallas probe and SDE kernel, CuTe translator prototype, He10 Pallas prototype,
  layout and CuTe paper benches, `results/*.jsonl`.
- `make_tables.py` -> `TABLES.md`.
- Envs on kevtan-linux in `~/src/pf-bench`: `.venv-jax` (stock pypomp, pfjax with the env-gated
  `PFJAX_FIX_KEYSPLIT` switch, particles, pypfilt), `.venv-torch` (pydpf), `.venv-ssm` (cuthbert and
  cuthbertlib 0.1.1, jax[cuda12] 0.12.0.1; `setup_envs.sh` builds it), `.venv-cute`
  (nvidia-cutlass-dsl 4.7.1, pypomp branch via PYTHONPATH), `.venv-pypomp-tests` (CPU test suite).

## How to verify
- Tables: `python make_tables.py` (reads the jsonl files and quant's `tests/*/timing/results/R`).
- cuthbert: `./smoke.sh` (toy models, CPU; `lg` should give about -184.14 at J = 2,000, exact
  -184.111); per model, the validation commands and z tables in each README (`compare.py`,
  `spx/validate.py`).
- CuTe branch: `tests/test_cute` on a GPU with `.venv-cute` and `PYTHONPATH=<worktree>`; the CPU suite
  with `.venv-pypomp-tests`.
- Paper workloads, XLA vs CuTe: `PYTHONPATH=<worktree> python kernels/cute_paper_bench.py dacca measles spx`.

## cuthbert notes
- Written as cuthbert's examples write models: `init_sample(key)`, `propagate_sample(key, x, inputs)`,
  `log_potential(x_prev, x, inputs)`, filter built inside a function of theta (closures), per-step
  data and covariates as `model_inputs`, loglik = `cuthbert.filter(...).log_normalizing_constant[-1]`.
  No push-forward: cuthbert leaves x0 unweighted.
- `cuthbert.filter` returns every step's particles, but under `jit` XLA drops the unused history
  (compiled temp memory is the same at T = 100 and 400, J = 100,000), so memory does not grow with T.
- `init_prepare` and the first `filter_combine` split the same key (`split(k, N)` and
  `split(k, N + 1)` share their first N keys), so with a random x0, particle i's initial draw reuses
  particle i-1's first propagation key. Harmless for the estimate (toy models, 4,000 filters at
  J = 200 and 2,000: logliks match pypomp's and an init that splits its key first); the ms models
  have deterministic x0. Not reported upstream.
- On CPU, cuthbertlib resamples through a numba `pure_callback` (`vmap_method="sequential"`); on
  GPU it uses `searchsorted(method="sort")`.
- Validation z against pypomp: spx -1.49, +0.20, -0.04, sun2024 -0.77; dacca (20 filters)
  +0.76, -2.77, +0.29, published +0.38, and with 100 filters -0.08, -0.25, -0.89; London (20 filters)
  +0.49, +0.46, -2.63, and with 100 filters on GPU -1.62, +2.25, +0.17; panel (10 filters, CPU) -2.09, -0.47, -0.30, and with 50 filters on GPU +1.23, -1.19, +0.71.
- GPU runs: GPU 0, `taskset -c 0-3`, one-hour `timeout` per run, from a scratch runner
  (`~/src/pf-bench/scratch/gpu_cuthbert.sh`, not in the repo; the commands are in each README).
  CPU validations ran on other cores at the same time (load 3-7). The London 100-filter check
  shared GPU 0 with the panel timing, which was going to hit the limit either way.
- `run_gpu.sh` (toy GPU grid) has a cuthbert line, but the toy grid was not run for cuthbert.

## GPU coordination
- GPU 0 was reserved via `/tmp/gpu0-reserved-pypomp-bench` for these runs; released 2026-10-22.
- Other users keep the CPU at load 40-110 on 32 threads; CPU wall times on this box mostly measure
  the scheduler, so CPU-library projections use the best single-filter time.

## Next steps
1. `ms_table_libraries.qmd` is the ms table (reads `table.csv`); it needs this folder merged into
   the quant commit the ms builds from.
2. pypomp PRs #55-#58 await review; #55 unblocks lint on the others.
3. Kevin's call: whether cuthbert goes into `ms_table_libraries.qmd`. `table.csv` already has the
   `cuthbert` column; the qmd is unchanged.
4. Kevin's call (2026-10-23): the ms table keeps the released Pypomp column for now. After #56 and #57
   merge, rerun the Pypomp rows on an idle GPU (first call and warm), then add a "+ speedups" column
   (parallel replicates + CuTe "auto") to `ms_table_libraries.qmd`.
