#!/usr/bin/env python
# coding: utf-8

# ---
# title: 'Speed test for pypomp'
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
import platform
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

# we could explicitly ask for a cpu test, but this is usually not necessary
# instead, we record what jax is using, via print(jax.devices())
# jax.config.update("jax_platform_name", "cpu")


# In[ ]:


#| label: mif-test
#| echo: true
d = pp.dacca()
start = time.perf_counter()
d.mif(sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0,n_monitors=0)
print(d.results[-1]["thetas_out"][1,1,])
end = time.perf_counter()
elapsed1 = end - start

start = time.perf_counter()
d.mif(sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0,n_monitors=0)
print(d.results[-1]["thetas_out"][1,1,])
end = time.perf_counter()
elapsed2 = end - start

pickle_file = out_dir + "/mif-test.pkl"
pickle_data = [elapsed1,elapsed2]
file=open(pickle_file,'wb')
pickle.dump(pickle_data,file)

jax.clear_caches()        


# Time taken: first call `{python} round(elapsed1,6)`s, second call  `{python} round(elapsed2,6)`s.
# 

# In[ ]:


#| label: pfilter-test
#| echo: true
#| eval: true

start = time.perf_counter()
d.pfilter(J=J, thresh=0, key=jax.random.key(111))
print(d.results[-1]["logLik"])
end = time.perf_counter()
elapsed3 = end - start
start = time.perf_counter()
d.pfilter(J=J, thresh=0, key=jax.random.key(111))
print(d.results[-1]["logLik"])
end = time.perf_counter()
elapsed4 = end - start
pickle_file = out_dir + "/pfilter-test.pkl"
pickle_data = [elapsed3,elapsed4]
file=open(pickle_file,'wb')
pickle.dump(pickle_data,file)

#pp.mop(J = J, rinit = d.rinit.struct_pf, rprocess = d.rprocess.struct_pf, dmeasure = d.dmeasure, theta = d.theta, ys = d.ys, covars = d.covars, alpha = 0.9)

# We could check that first log-likelihood evaluation, `{python} str(round(loglik3,6))`, matches second evaluation,  `{python} str(round(loglik4,6))`. 


# Time taken: first call `{python} round(elapsed3,6)`s, second call  `{python} round(elapsed4,6)`s.
# 

# In[ ]:


#| label: terminal-output
#| echo: false
#| eval: true
print(
    datetime.date.today().strftime("%Y-%m-%d"), "pypomp speed test using",jax.devices(), 
    "\npypomp", version('pypomp'), "for dacca with J =", J,
    "\nPython", platform.python_version(),
        ", jax", version('jax'), ", jaxlib", version('jaxlib'),
    "\nmif: with jit", round(elapsed1,6), "s, ",
        "pre-jitted", round(elapsed2,6), "s",
    "\npfilter: with jit", round(elapsed3,6), "s, ",
        "pre-jitted", round(elapsed4,6), "s \n"
)

