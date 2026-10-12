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

Each model is tested the same four ways. The report for test `X` on model `M` is at `tests/M/X/report.html`.

* **Likelihood**: particle filter log-likelihoods at fixed parameters, `pypomp` vs R `pomp`, compared as distributions (KS test, densities).
* **Estimation**: IF2 global search from identical starting points, `pypomp` vs `pomp`: traces, final parameter estimates, final log-likelihoods.
* **Timing**: IF2 and particle filter runtime on identical work, `pypomp` on GPU and CPU vs `pomp`.
* **Scaling**: GPU runtime and peak VRAM as particles ($J$) and chains grow; where the GPU saturates.

| Model | Likelihood | Estimation | Timing | Scaling |
|---|---|---|---|---|
| SPX (S&P 500) | [report](tests/spx/loglik/report.html) | [report](tests/spx/estimation/report.html) | [report](tests/spx/timing/report.html) | [report](tests/spx/scaling/report.html) |
| Dhaka cholera | [report](tests/dacca/loglik/report.html) | — | [report](tests/dacca/timing/report.html) | [report](tests/dacca/scaling/report.html) |
| Measles | [report](tests/measles/loglik/report.html)¹ | [report](tests/measles/estimation/report.html) | [report](tests/measles/timing/report.html)² | [report](tests/measles/scaling/report.html) |
| Panel measles | [report](tests/panel_measles/loglik/report.html) | [report](tests/panel_measles/estimation/report.html)³ | [report](tests/panel_measles/timing/report.html) | [report](tests/panel_measles/scaling/report.html) |

¹ Also compares `pypomp` in 32- and 64-bit precision, on London and Halesworth.<br>
² Also compares the fast samplers in `pypomp.random` with stock JAX samplers.<br>
³ Block IF2, plus the `pypomp`-only mix-and-match step.

### Method comparisons

* **[IF2 vs IFAD](tests/dacca/algorithms/report.html)** (Dhaka cholera): how far iterated filtering alone and iterated filtering with gradient training get on the same likelihood for a comparable wall-clock budget.

### Bayesian inference

Both share one SIR problem (`tests/bayesian/model.py`, `model.R`): estimate `beta1` and `rho` under a flat box prior. `tests/bayesian/reference/run.py` computes the reference answers (grid posterior, rejection-sampled ABC posterior) that both reports read.

* **[PMCMC](tests/bayesian/pmcmc/report.html)**: checks that `pypomp`'s particle marginal Metropolis-Hastings matches the grid posterior and R `pomp`'s `pmcmc`, and does not depend on the particle count.
* **[ABC](tests/bayesian/abc/report.html)**: checks that `pypomp`'s ABC-MCMC matches the exact ABC posterior and R `pomp`'s `abc` at each tolerance in a ladder, and returns the prior when the tolerance does not bind.

### Random number generators

* **[Fast samplers](tests/samplers/test.html)**: speed and statistical accuracy of `pypomp`'s fast approximate inverse-CDF samplers (`fast_poisson`, `fast_binomial`, `fast_gamma`, `fast_nbinomial`) against `jax.random` and `scipy.stats`.
