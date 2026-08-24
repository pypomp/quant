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

Below is a list of quantitative test reports available in this repository:

### 1. SPX (S&P 500) Model
* **[Parameter Estimation & Convergence](tests/spx/estimation/report.html)** (`tests/spx/estimation`): Compares parameter estimation traces and log-likelihood estimates on the SPX index dataset using `pypomp` (CPU/GPU) and R's `pomp`.
* **[Runtime & Performance Benchmark](tests/spx/timing/report.html)** (`tests/spx/timing`): Benchmarks execution speed, speedup factors, and CPU/GPU throughput relative to R `pomp`.
* **[Fixed-Parameter Likelihood Validation](tests/spx/loglik/report.html)** (`tests/spx/loglik`): Evaluates particle filter log-likelihood estimates at fixed parameter values (`rep3600`) via two-sample KS test.
* **[Computational Scaling Benchmark](tests/spx/scaling/report.html)** (`tests/spx/scaling`): Evaluates execution speed and VRAM memory scaling across particle counts ($J$) and parallel search chains on GPU.


### 2. Dhaka Cholera Model
* **[IF2 vs IFAD](tests/dacca/algorithms/report.html)** (`tests/dacca/algorithms`): Compares how far iterated filtering alone and iterated filtering with gradient training get on the same likelihood for a comparable wall-clock budget.
* **[Runtime & Performance Benchmark](tests/dacca/timing/report.html)** (`tests/dacca/timing`): Benchmarks IF2 and particle filter execution speed in `pypomp` (CPU/GPU) against R `pomp`, on identical work.
* **[Fixed-Parameter Likelihood Validation](tests/dacca/loglik/report.html)** (`tests/dacca/loglik`): Checks that the particle filter yields the same distribution of log-likelihoods in `pypomp` and `pomp` at the published MLE.

### 3. Random Number Generators
* **[Random Number Generators Benchmark & Comparison](tests/samplers/test.html)** (`tests/samplers`): Benchmarks the execution speed and validates the statistical accuracy of `pypomp`'s fast approximate inverse CDF samplers (`fast_poisson`, `fast_binomial`, `fast_gamma`, `fast_nbinomial`) against `jax.random` and `scipy.stats`.

### 4. Measles Model
* **[Fixed-Parameter Likelihood Validation](tests/measles/loglik/report.html)** (`tests/measles/loglik`): Compares the distribution of particle filter log-likelihoods at the He et al. (2010) estimates in `pypomp` (32- and 64-bit) against R's `pomp`.
* **[Parameter Estimation](tests/measles/estimation/report.html)** (`tests/measles/estimation`): Compares the distribution of IF2 parameter estimates from a global search in `pypomp` versus `pomp`, from identical starting points.
* **[Runtime & Throughput Benchmark](tests/measles/timing/report.html)** (`tests/measles/timing`): Benchmarks IF2 and particle filter execution speed on the discrete measles model. Contrasts `pypomp` on GPU and CPU with R `pomp`, and the fast samplers in `pypomp.random` with stock JAX samplers.

### 5. Panel Measles Model
* **[Fixed-Parameter Likelihood Validation](tests/panel_measles/loglik/report.html)** (`tests/panel_measles/loglik`): Compares the distribution of panel particle filter log-likelihoods at the He et al. (2010) estimates in `pypomp` against R's `panelPomp`.
* **[Parameter Estimation](tests/panel_measles/estimation/report.html)** (`tests/panel_measles/estimation`): Compares the distribution of block-IF2 parameter estimates from a global search in `pypomp` versus `panelPomp`, from identical starting points, and reports the `pypomp`-only mix-and-match step.
* **[Runtime & Throughput Benchmark](tests/panel_measles/timing/report.html)** (`tests/panel_measles/timing`): Benchmarks block-IF2 and particle filter execution speed on the 4-unit mixed panel model, contrasting `pypomp` on GPU and CPU with R `panelPomp`.

### 6. Bayesian Inference (PMCMC & ABC)
Organized by method rather than by model: all three read one shared SIR benchmark defined in `tests/bayesian/model.py` and `model.R`, estimating `beta1` and `rho` under a flat bounded prior.
* **[Reference Posterior](tests/bayesian/reference/report.html)** (`tests/bayesian/reference`): Builds a grid-quadrature posterior over `(beta1, rho)` from batched particle filter evaluations, giving the samplers an absolute target on a model with no analytic posterior.
* **[PMCMC](tests/bayesian/pmcmc/report.html)** (`tests/bayesian/pmcmc`): Checks that `pypomp`'s particle marginal Metropolis-Hastings targets the reference posterior, agrees with R `pomp`'s `pmcmc`, and is invariant to the particle count `J` as the exact-approximate property requires.
* **[ABC](tests/bayesian/abc/report.html)** (`tests/bayesian/abc`): Compares `pypomp`'s approximate Bayesian computation against R `pomp`'s `abc` at identical probes and tolerance, and checks that the posterior converges as the tolerance shrinks and collapses onto the prior when it does not bind.

