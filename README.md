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

### 2. Dhaka Cholera Model
* **[Dhaka Report](tests/dacca/report.html)** (`tests/dacca`): Analyzes the performance, runtime, and parameter convergence of IF2 versus IFAD for the Dhaka cholera model.
Also checks that the particle filter yields the same distribution of log likelihoods in both `pypomp` and `pomp`.

### 3. Random Number Generators
* **[Random Number Generators Benchmark & Comparison](tests/samplers/test.html)** (`tests/samplers`): Benchmarks the execution speed and validates the statistical accuracy of `pypomp`'s fast approximate inverse CDF samplers (`fast_poisson`, `fast_binomial`, `fast_gamma`, `fast_nbinomial`) against `jax.random` and `scipy.stats`.

### 4. Measles Model
* **[Log-Likelihood and Parameter Comparison: Pypomp vs R](tests/measles/R_comparison/report.html)** (`tests/measles/R_comparison`): Compares distributions of log-likelihood and parameter estimates obtained via `pypomp` versus R's `pomp`.
* **[Runtime Comparison: Pypomp vs R](tests/measles/R_comparison/speed_comparison/report.html)** (`tests/measles/R_comparison/speed_comparison`): Benchmarks runtime of IF2 on a discrete measles model using `pypomp` versus `pomp`. This demonstrates the utility of the fast samplers in `pypomp.random`.

### 5. Panel Measles Model
* **[Panel Measles (mixed parameters)](tests/panel_measles/R_comparison/mixed/report.html)** (`tests/panel_measles/R_comparison/mixed`): Compares parameter and log-likelihood estimates on a panel measles dataset using `pypomp` and R's `pomp`.

