
# versions of ~/git/quant/speed/cpu/report.py
# set up python via the quant env:
# source ~/git/quant/.venv/bin/activate

# a variant of test.py that conducts multiple replications

# monitor memory
#MEM = True
MEM = False
if MEM:
    import tracemalloc
    tracemalloc.start()

# carry out regular garbage collection
GC = True    
if GC:
    import gc

run_level = 7
J = [10,100,1500,2000,5000,10000,20000,50000][run_level]

import os
import platform
import datetime
import shutil
from importlib.metadata import version
import jax
import time
import pypomp as pp

jax.config.update("jax_enable_compilation_cache", False)

print(    
    datetime.date.today().strftime("%Y-%m-%d"), "pypomp speed test using",jax.devices(), 
    "\npypomp", version('pypomp'), "for dacca with J =", J,
    "\nPython", platform.python_version(),
        ", jax", version('jax'), ", jaxlib", version('jaxlib'),
)

if MEM: print("    current memory usage :",tracemalloc.get_traced_memory())

d = pp.dacca()
start = time.perf_counter()
mif_out1 = d.mif(
    sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0, monitor=False
)
end = time.perf_counter()
elapsed1 = end - start

if GC: gc.collect()

start = time.perf_counter()
mif_out2 = d.mif(
    sigmas=0.02, sigmas_init=0.1, M=1, a=0.9, J=J, key=jax.random.key(111), thresh=0, monitor=False
)
end = time.perf_counter()
elapsed2 = end - start

if GC:
    del mif_out1, mif_out2
    gc.collect()

import pypomp.pfilter
start = time.perf_counter()

loglik3 = d.pfilter(J=J, thresh=0, key=jax.random.key(111))

print(
    "mif: with jit", round(elapsed1,6), "s, ",
        "pre-jitted", round(elapsed2,6), "s",
)

if MEM: print("    current memory usage :",tracemalloc.get_traced_memory())

end = time.perf_counter()
elapsed3 = end - start

if GC: gc.collect()

start = time.perf_counter()
loglik4 = d.pfilter(J=J, thresh=0, key=jax.random.key(111))
end = time.perf_counter()
elapsed4 = end - start


print(
    "pfilter: with jit", round(elapsed3,6), "s, ",
        "pre-jitted", round(elapsed4,6), "s, ",
)

if GC:
    del loglik3, loglik4
    gc.collect()

if MEM: print("    current memory usage :",tracemalloc.get_traced_memory())


jax.clear_caches()        
print("reset using jax.clear_caches()")

for k in range(10):
    start = time.perf_counter()
    loglik = d.pfilter(J=J, thresh=0, key=jax.random.key(k))
    end = time.perf_counter()
    elapsed = end - start
    print("pfilter: ", round(elapsed,6), "s")
    if MEM: print("    current memory usage :",tracemalloc.get_traced_memory())
    if GC:
        del loglik
        gc.collect()



jax.clear_caches()        
print("reset using jax.clear_caches()")

for k in range(10):
    start = time.perf_counter()
    loglik = d.pfilter(J=J, thresh=0, key=jax.random.key(1))
    end = time.perf_counter()
    elapsed = end - start
    print("pfilter: ", round(elapsed,6), "s")
    if MEM: print("    current memory usage :",tracemalloc.get_traced_memory())
    if GC:
        del loglik
        gc.collect()


if MEM: tracemalloc.stop()

        
