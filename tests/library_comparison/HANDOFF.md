# pf-bench: pypomp vs other Python particle-filter libraries, and kernel work (2026-10-22)

## Goal (Kevin's words)
"can you also perf pypomp against pfjax, pypfilt, pydpf, particles?" ... "i want to augment this ms"
(https://github.com/pypomp/tutorials/blob/main/article_jmlr/ms.pdf) ... "bench over the examples in
[the ms], noting that ideally we extend the tables to add the comparisons with other libraries other
than just R pomp". Kernel work: pypomp "has to be pretty general, modular and/or user-friendly ...
take in user-defined rproc". GPU numbers are what matter; CPU ablations are not useful. Use CuTe,
not Triton. On RNG: "i dont care about reproducibility and exact threefry, i care about speed and
correctness".

## State
- **Tables**: `TABLES.md` (from `make_tables.py`): PF time on the four ms workloads for R pomp,
  pypomp (released, parallel reps, CuTe), pfjax (released, key fix), pydpf, and particles/pypfilt
  projected to 36 cores. Missing: London measles pfjax and pydpf, all panel GPU rows, panel pypfilt.
  Rerun `python make_tables.py > TABLES.md` when they land.
- **Kernel work**: `kernels/README.md`. The CuTe backend is in pypomp branch `cute-rproc`
  (worktree `../../../pypomp-wt-cute`), which also has the parallel/adaptive replicate batching
  (was `pfilter-vmap-reps`) and the float32 measles dmeas fix (was `measles-dmeas-tails`, made by
  another session). One combined PR to pypomp/pypomp main.
- **pfjax**: key-splitting fix submitted as mlysy/pfjax#38 (Kevin's text).
- **Correctness**: every library agrees with pypomp within Monte Carlo error on every model
  (per-model READMEs). pypomp's float32 measles dmeas underflowed at poorly fitting parameters;
  fixed on the branch.

## Layout
- `common.py`, `bench_{pypomp,pfjax,particles,pypfilt,pydpf}.py`: toy models (`lg`, `sde`) in each library.
- `{spx,dacca,measles,panel_measles}/`: the ms models ported to each library; `bench_*.py`,
  `export_inputs.py`, `README.md` (validation, single-filter CPU timings, deviations), `results/`.
- `kernels/`: Pallas probe and SDE kernel, CuTe translator prototype, He10 Pallas prototype,
  layout and CuTe paper benches, `results/*.jsonl`.
- `make_tables.py` -> `TABLES.md`.
- Envs on kevtan-linux in `~/src/pf-bench`: `.venv-jax` (stock pypomp, pfjax with the env-gated
  `PFJAX_FIX_KEYSPLIT` switch, particles, pypfilt), `.venv-torch` (pydpf), `.venv-cute`
  (nvidia-cutlass-dsl 4.7.1, pypomp branch via PYTHONPATH), `.venv-pypomp-tests` (CPU test suite).

## How to verify
- Tables: `python make_tables.py` (reads the jsonl files and quant's `tests/*/timing/results/R`).
- CuTe branch: `tests/test_cute` on a GPU with `.venv-cute` and `PYTHONPATH=<worktree>`; the CPU suite
  with `.venv-pypomp-tests`.
- Paper workloads, XLA vs CuTe: `PYTHONPATH=<worktree> python kernels/cute_paper_bench.py dacca measles spx`.

## Running jobs (kevtan-linux)
- `/tmp/measles_gpu.sh` -> `/tmp/measles_gpu.log` on GPU 0: London then panel, for pypomp, pfjax
  (released and key fix) and pydpf. Copy its rows into `measles/results/gpu/` and
  `panel_measles/results/gpu/` when done.

## GPU coordination
- GPU 0 reserved via `/tmp/gpu0-reserved-pypomp-bench`. genperms and gsim agreed to keep GPU 0 free
  while it exists. REMOVE IT WHEN THE GPU RUNS FINISH.
- Other users keep the CPU at load 40-110 on 32 threads; CPU wall times on this box mostly measure
  the scheduler, so CPU-library projections use the best single-filter time.

## Next steps
1. When `/tmp/measles_gpu.log` says `MEASLES_GPU_DONE`: copy rows, regenerate `TABLES.md`, remove the
   GPU 0 reservation.
2. Draft the ms table (LaTeX in the ms's `latex_table` style) from `TABLES.md` once the rows are in.
