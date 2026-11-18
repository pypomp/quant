# quant: Quantitative tests of pypomp

These __quantitative tests__, or simply __quant tests__, are designed to assess the accuracy and performance of `pypomp` for problems existing on a scale too large to be run on a laptop within the unit tests in [pypomp:pypomp/tests](https://github.com/pypomp/pypomp/tree/main/tests). 

The quant tests also provide additional examples of pypomp, focused on technical issues that extend the simpler examples in [pypomp:tutorials](https://pypomp.github.io/tutorials).

---

### 🚀 Quick Links
* 💻 **[Core Repository](https://github.com/pypomp/pypomp)** — Source code and issue tracker for the `pypomp` package.
* 📖 **[Read the Documentation](https://pypomp.readthedocs.io/)** — Core library API reference and user guide.
* 🎓 **[Tutorials](https://pypomp.github.io/tutorials)** — Examples and tutorials for learning `pypomp`.

---

## Quantitative Tests Index

### Model tests

Each model is tested the same four ways. Likelihood, timing and scaling each have one report covering every model; estimation has one report per model.

* **[Likelihood](reports/loglik.html)**: particle filter log-likelihoods at fixed parameters, `pypomp` vs R `pomp`, compared as distributions (KS test, densities).
* **Estimation** ([SPX](tests/spx/estimation/report.html), [measles](tests/measles/estimation/report.html), [panel measles](tests/panel_measles/estimation/report.html)): IF2 global search from identical starting points, `pypomp` vs `pomp`: traces, final parameter estimates, final log-likelihoods.
* **[Timing](reports/timing.html)**: IF2 and particle filter runtime on identical work, `pypomp` on GPU and CPU vs `pomp`. For measles, also compares the fast samplers in `pypomp.random` with stock JAX samplers.
* **[Scaling](reports/scaling.html)**: GPU runtime and peak VRAM as particles ($J$) and chains grow; where the GPU saturates.

The table shows the worst result among each model's automated checks in each report, computed when the report was last rendered: ✅ pass, ⚠️ check, ❌ fail. The automated checks only compare summary statistics, so each report also lists what its plots should show. Timing and scaling have no automated checks.

| Model | Likelihood | Estimation |
|---|---|---|
| SPX (S&P 500) | <!-- check reports/loglik#spx --> ✅ | <!-- check tests/spx/estimation/report --> ✅ |
| Dhaka cholera | <!-- check reports/loglik#dacca --> ✅ | — |
| Measles | <!-- check reports/loglik#measles --> ✅¹ | <!-- check tests/measles/estimation/report --> ✅ |
| Panel measles | <!-- check reports/loglik#panel_measles --> ✅ | <!-- check tests/panel_measles/estimation/report --> ✅² |

¹ Also compares `pypomp` in 32- and 64-bit precision, on London and Halesworth.<br>
² Block IF2, plus the `pypomp`-only mix-and-match step.

### Method comparisons

* **[IF2 vs IFAD](tests/dacca/algorithms/report.html)** (Dhaka cholera): how far iterated filtering alone and iterated filtering with gradient training get on the same likelihood for a comparable wall-clock budget.

### Bayesian inference

Both share one SIR problem (`tests/bayesian/model.py`, `model.R`): estimate `beta1` and `rho` under a flat box prior. `tests/bayesian/reference/run.py` computes the reference answers (grid posterior, rejection-sampled ABC posterior) that both reports read.

* **[PMCMC](tests/bayesian/pmcmc/report.html)** <!-- check tests/bayesian/pmcmc/report --> ✅: checks that `pypomp`'s particle marginal Metropolis-Hastings matches the grid posterior and R `pomp`'s `pmcmc`, and does not depend on the particle count.
* **[ABC](tests/bayesian/abc/report.html)** <!-- check tests/bayesian/abc/report --> ✅: checks that `pypomp`'s ABC-MCMC matches the exact ABC posterior and R `pomp`'s `abc` at each tolerance in a ladder, and returns the prior when the tolerance does not bind.

### Random number generators

* **[Fast samplers](tests/samplers/report.html)**: speed and statistical accuracy of `pypomp`'s fast approximate inverse-CDF samplers (`fast_poisson`, `fast_binomial`, `fast_gamma`, `fast_nbinomial`) against `jax.random` and `scipy.stats`.
