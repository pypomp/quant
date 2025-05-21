#!/usr/bin/env python
# coding: utf-8

# ---
# title: 'Speed test for pypomp on cpu'
# jupyter: python3
# embed-resources: true
# format: 
#     html:
#         page-layout: full
# execute:
#     daemon: false
# ---

# In[ ]:


#| label: run_level
#| echo: false
import os
import datetime
import shutil
from importlib.metadata import version

run_level = 2

out_dir="results_" + str(run_level)

# clean the cached results automatically
# unlike other quant tests, these are relatively quick and will always replace saved files
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir)


# N = [10, 50, 100][run_level]
J = [10,100,1000][run_level]


# Testing pypomp `{python} version('pypomp')` on `{python} datetime.date.today().strftime("%Y-%m-%d")` at run level `{python} run_level` (0 is for debugging, 2 is full-length).

# In[ ]:


#| label: imports
#| echo: false
import jax
import time
import pypomp as pp
import unittest
import tracemalloc
import jax.numpy as jnp
import numpy as np
import pandas as pd

# for saving partial results
import pickle

# import pykalman
import seaborn as sns
import matplotlib.pyplot as plt
import jax.scipy.special
from jax.scipy.special import logit, expit

from tqdm import tqdm

# jax.config.update("jax_platform_name", "cpu")


# In[ ]:


#| label: oo-test
#| echo: true
d = pp.dacca()
start = time.perf_counter()
loglik1, params1 = d.mif(
    sigmas=0.02, sigmas_init=0.1, J=J, thresh=-1, key=jax.random.key(111), M=1
    )
end = time.perf_counter()
elapsed1 = end - start
start = time.perf_counter()
loglik2, params2 = d.mif(
    sigmas=0.02, sigmas_init=0.1, J=J, thresh=-1, key=jax.random.key(111), M=1
    )
end = time.perf_counter()
elapsed2 = end - start
pickle_file = out_dir + "/mif-test.pkl"
pickle_data = [elapsed1,loglik1,elapsed2,loglik2]
file=open(pickle_file,'wb')
pickle.dump(pickle_data,file)


# Time taken: first call `{python} round(elapsed1,6)`s, second call  `{python} round(elapsed2,6)`s.
# 
# Check that first log-likelihood evaluation, `{python} str(round(loglik1,6))`, matches second evaluation,  `{python} str(round(loglik2,6))`. 
# Note that this is not currently working, for some reason

# In[ ]:


#| label: functional-test
#| echo: true
#| eval: true

import pypomp.pfilter
start = time.perf_counter()
loglik3 = pypomp.pfilter(theta=d.theta,
    ys =d.ys, J = J, rinit = d.rinit, rproc = d.rproc,
    dmeas = d.dmeas, covars = d.covars, thresh = -1,key=jax.random.key(111))
end = time.perf_counter()
elapsed3 = end - start
start = time.perf_counter()
loglik4 = pypomp.pfilter(theta=d.theta,
    ys =d.ys, J = J, rinit = d.rinit, rproc = d.rproc,
    dmeas = d.dmeas, covars = d.covars, thresh = -1,key=jax.random.key(111))
end = time.perf_counter()
elapsed4 = end - start
pickle_file = out_dir + "/pfilter-test.pkl"
pickle_data = [elapsed3,loglik3,elapsed4,loglik4]
file=open(pickle_file,'wb')
pickle.dump(pickle_data,file)


# Time taken: first call `{python} round(elapsed3,6)`s, second call  `{python} round(elapsed4,6)`s.
# 
# Check that first log-likelihood evaluation, `{python} str(round(loglik3,6))`, matches second evaluation,  `{python} str(round(loglik4,6))`. 
# 
