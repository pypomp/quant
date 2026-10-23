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
  projected to 36 cores. Complete; pfjax on the panel hit the one-hour limit and shows as > 3,600 s.
- **Kernel work**: `kernels/README.md`. In pypomp, split into separate PRs to main (combined #54 was
  closed): #55 pyright fixes for jax 0.12, #56 parallel pfilter replicates (branch
  `perf/pfilter-parallel-reps`), #57 the CuTe backend (branch `feature/cute-rproc`, worktree
  `../../../pypomp-wt-cute-pr`), and the measles dmeas fix from its own session (branch
  `measles-dmeas-tails`). test_sir_mif_regression fails on main under jax 0.12; Kevin is raising
  it with Aaron.
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

## GPU coordination
- GPU 0 was reserved via `/tmp/gpu0-reserved-pypomp-bench` for these runs; released 2026-10-22.
- Other users keep the CPU at load 40-110 on 32 threads; CPU wall times on this box mostly measure
  the scheduler, so CPU-library projections use the best single-filter time.

## Next steps
1. Draft the ms table (LaTeX in the ms's `latex_table` style) from `TABLES.md`.
2. pypomp PRs #55-#58 await review; #55 unblocks lint on the others.
