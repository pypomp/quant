# quant: Quantitative tests of pypomp

These __quantitative tests__, or simply __quant tests__, are designed to assess the accuracy and performance of `pypomp` for problems existing on a scale too large to be run on a laptop within the unit tests in [pypomp:pypomp/tests](https://github.com/pypomp/pypomp/tree/main/tests). 

The quant tests also provide additional examples of pypomp, focused on technical issues that extend the simpler examples in [pypomp:tutorials](https://pypomp.github.io/tutorials).

## Existing Quantitative Tests

Below is a list of quantitative test reports available in this repository:

### 1. SPX (S&P 500) Model
* **[SPX Report](tests/spx/report.html)** (`tests/spx`): Compares parameter estimation traces and log-likelihood estimates on the SPX index dataset using `pypomp` (CPU/GPU) and R's `pomp`.

### 2. Dhaka Cholera Model
* **[Dhaka Report](tests/dacca/report.html)** (`tests/dacca`): Analyzes the performance, runtime, and parameter convergence of IF2 versus IFAD for the Dhaka cholera model.
Also checks that the particle filter yields the same distribution of log likelihoods in both `pypomp` and `pomp`.

### 3. Random Number Generators
* **[Random Number Generators Benchmark & Comparison](tests/samplers/test.html)** (`tests/samplers`): Benchmarks the execution speed and validates the statistical accuracy of `pypomp`'s fast approximate inverse CDF samplers (`fast_poisson`, `fast_binomial`, `fast_gamma`, `fast_nbinomial`) against `jax.random` and `scipy.stats`.

### 4. Measles Model
* **[LogLik Comparison: Pypomp vs R](tests/measles/R_comparison/report.html)** (`tests/measles/R_comparison`): Verifies log-likelihood evaluation accuracy by comparing `pypomp` (both exact and approximate implementations) against R's `pomp` package.
* **[Parameter Comparison: Pypomp vs R](tests/measles/R_comparison/parameter_comparison/report.html)** (`tests/measles/R_comparison/parameter_comparison`): Compares the parameter estimates obtained via `pypomp` versus R's `pomp`.
* **[Measles Report](tests/measles/performance/report.html)** (`tests/measles/performance`): Benchmarks parameter estimation via IF2 on a discrete measles model.

### 5. Panel Measles Model
* **[Panel Measles Report](tests/panel_measles/performance/report.html)** (`tests/panel_measles/performance`): Benchmarks panel parameter estimation via IF2 on a panel measles dataset.

### 6. Differentiated Process Off-Parameter Filtering
* **[DPOP London Measles Benchmark](tests/dpop/london_measles/report.html)** (`tests/dpop/london_measles`): Compares DPOP training against an IF2 baseline on the single-unit London measles model, including likelihood distribution, elapsed-time trace, and runtime summaries.

### 7. Bayesian Iterated Filtering
* **[BIF versus PMCMC Benchmarks](tests/bif/sir_dacca_pmcmc/report.html)** (`tests/bif/sir_dacca_pmcmc`): Compares BIF against PMCMC on a four-parameter SIR model and a four-parameter, 100-observation Dacca cholera benchmark, including runtime, posterior marginal, interval, and PMCMC diagnostic summaries.


