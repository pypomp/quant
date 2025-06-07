
# versions of ~/git/quant/speed/cpu/report.py
# set up python via the quant env:
# source ~/git/quant/.venv/bin/activate

run_level = 2
J = [10,100,1000][run_level]

import os
import platform
import datetime
import shutil
from importlib.metadata import version
import jax
import time
import pypomp as pp

d = pp.dacca()
start = time.perf_counter()
mif_out1 = d.mif(
    sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0, monitor=False
)
end = time.perf_counter()
elapsed1 = end - start

start = time.perf_counter()
mif_out2 = d.mif(
    sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0, monitor=False
)
end = time.perf_counter()
elapsed2 = end - start

import pypomp.pfilter
start = time.perf_counter()
loglik3 = d.pfilter(J=J, thresh=0, key=jax.random.key(111))
end = time.perf_counter()
elapsed3 = end - start
start = time.perf_counter()
loglik4 = d.pfilter(J=J, thresh=0, key=jax.random.key(111))
end = time.perf_counter()
elapsed4 = end - start


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



