
# versions of ~/git/quant/speed/cpu/report.py
# set up python via the quant env:
# source ~/git/quant/.venv/bin/activate

run_level = 5
J = [10,100,500, 1000,2000,5000,10000,20000,50000][run_level]
K = [5, 5, 10, 10, 10, 10, 10, 10, 10][run_level]

import platform
import datetime
from importlib.metadata import version
import jax
import time
import pypomp as pp
import numpy as np

d = pp.dacca()
start = time.perf_counter()
d.mif(sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0)
# force completion
if d.results[-1]["thetas_out"][0][0,0,0] > -10000:
    print("done mif with jit")
end = time.perf_counter()
elapsed1 = end - start

mif_time = []
for k in range(K):
    start = time.perf_counter()
    d.mif(sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, 
        key=jax.random.key(k), thresh=0)
    # force completion
    if d.results[-1]["thetas_out"][0][0,0,0] > -10000:
        print("done mif", k)
    end = time.perf_counter()
    elapsed2 = end - start
    mif_time.append(elapsed2)

jax.clear_caches()        

start = time.perf_counter()
d.pfilter(J=J, thresh=0, key=jax.random.key(111))
# force completion
if d.results[-1]["logLiks"][0] > -100000:
    print("done pfilter with jit")
end = time.perf_counter()
elapsed3 = end - start

pf_time = []
for k in range(K):
    start = time.perf_counter()
    d.pfilter(J=J, thresh=0, key=jax.random.key(k))
    # force completion
    if d.results[-1]["logLiks"][0] > -100000:
        print("done pfilter", k)
    end = time.perf_counter()
    elapsed4 = end - start
    pf_time.append(elapsed4)

print(
    datetime.date.today().strftime("%Y-%m-%d"), "pypomp speed test using",jax.devices(), 
    "\npypomp", version('pypomp'), "for dacca with J =", J,
    "\nPython", platform.python_version(),
        ", jax", version('jax'), ", jaxlib", version('jaxlib'),
    "\nmif: with jit", round(elapsed1,4), "s, ",
        "pre-jitted", round(np.mean(mif_time),4), "s (sd",
        round(np.std(mif_time),4), ")",
    "\npfilter: with jit", round(elapsed3,4), "s, ",
        "pre-jitted", round(np.mean(pf_time),4), "s (sd",
        round(np.std(pf_time),4), ")\n"
)



