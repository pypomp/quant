# quant: Quantitative tests of pypomp

We are concerned with accuracy tests to make sure that the code gives correct answers (up to small Monte Carlo error) in situations where this is knowable.
We are also concerned with performance benchmark tests. This can involve measuring time, memory requirement, or iterations to convergence for a maximization algorithm.
Put together, we call these __quantitative tests__, or simply __quant tests__, to distinguish them from unit tests.
Our unit tests, in [pypomp:pypomp/test](https://github.com/pypomp/pypomp/tree/main/test), are quick tests designed to check that the code is not broken, and to ensure we are told if numerical results change.
The quant tests sit in their own repository, since they are not necessarily run often.
Some of them are of an exploratory nature, such as code profiling investigations.

The quant tests also provide additional examples of pypomp, focused on technical issues that extend the
 simpler examples in [pypomp:tutorials](https://pypomp.github.io/tutorials)

## Existing tests

* [Testing log-likelihood calculations on a linear Gaussian model](pfilter_LG/report.html). This is a test of correctness of pfilter and MOP(alpha) against a Kalman filter.

* Other quant tests are available in [draft form](https://github.com/pypomp/quant).
These test timing, memory, and iterated filtering for the linear Gaussian model.
Also, a Dacca cholera model and a daphnia mesocosm model.

## Proposed structure of the tests

__These are aspirational suggestions rather than requirements. In practice, we post the best available quant tests whether or not they conform to this structure.__

We propose many different short tests, each of which can be run independently.
Each test produces an html report.
An index links these results.
At a later date, some of these tests could be selected for a tutorial or a software announcement publication.

Each test has its own directory, within which we have a standard file structure. We don't have to follow this exactly. For example, it may be helpful to include R code (to compare against R-pomp), or it may make sense to have various py code files, some of which get run on greatlakes and others on a laptop.

* report.qmd. A report presenting and discussing the results pulled in from the results directory. qmd is currently the preferred format, since it plays nicely with git and facilitates cross-testing against [R-pomp](https://kingaa.github.io/pomp/). report.ipynb is also acceptable.

* There may be other qmd or ipynb files, for exploratory or supplementary analysis, but `report.qmd` should be the main one. Supplementary analysis may be less frequently updated for new pypomp versions.

* code.py. The test code, which can be run on greatlakes, or wherever

* code.sbat. A slurm batch file for running code.py on greatlakes

* results. A directory with saved results from code.py

* report.html. The rendering of report.ipynb

* Makefile. To help automate the building of report.html

* README. A brief overview of the test

Each test should report on the following, if this is feasible:

1. The pypomp/Python/Jax versions used

2. The run time for each calculation. Can we split this into compilation vs calculation? We'd need a special facility for this since JIT carries out compilation as needed. It might be useful to get GPU vs CPU run time for all the tests, since we may start to see patterns in efficiency of GPU usage. Comparing with the R-pomp time across these tests would also help to discover where we are quicker or slower, and give ideas for areas that can use more code optimization. Some code profiling may be useful to see where the code bottlenecks are.

3. The memory requirement. I'm currently sure how to obtain this. Perhaps only a specific set of tests could investigat this, but probably once we've figured out how to do it, we can easily do it for all tests. For R-pomp, calculations have generally been CPU-limited, but the AD requires storing a potentially large graph.

4. Numerical results

## Some proposed tests

Much of this is similar to tests that Kevin already has in his ipynb files. 

**LG1**. Compare pfilter on the LG model with a Kalman filter. For this, it would be ideal to have a basic KF in pypomp, similar to the one in pomp. As a stop-gap, one could use the R-pomp kalman function.

**LG2**. pfilter at various values of N and J. Check that the scaling is as expected. 

**LG3**. mop-alpha at the same values of N and J. Check that the scaling is as expected. 

**LG4**. IFAD. 

**Dacca1.** Compare pfilter on the Dacca cholera model with the R-pomp version, with a fixed set of parameters, presumably the same that are used in the dacca R-pomp object.

**Dacca2.** Test IFAD using a pypomp version of the code used for the IFAD arXiv paper.


-------------------


