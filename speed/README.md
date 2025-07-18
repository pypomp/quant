This speed quant test provides a core set of speed benchmarks that can be updated as part of our workflow so we can assess progress.
This complements correctness tests. 


Here's a proposed structure.

(i) use dacca model. The LG model is good for correctness tests; it is poor for speed tests because the linear calculations in rprocess are so quick, potentially leading to rate-limiting steps that differ from practical situations.

(ii) use N=100. This should be enough to assess N scaling, but there's no point using the full N=600.

(iii) use J=1000. This should be less than the number of GPU cores. 

(iv) time one iteration of fit with either mop or mif2, using the class interface. This is an overall test. If it works well (and correctness holds) then, likely, everything is in order. This is the main speed benchmark. We can measure separately the time to jit and the time to evaluate.

(v) diagnostic tests. 
If results are disappointing, or we want to look for ways to further improve speed, we will want to evaluate various different ways; via the class and functional interfaces; with and without jit.

(vi) evaluate on greatlakes cpu and gpu. Avoid laptops for speed tests because it is hard to make sure the CPUs are idle. Jun got erratic behavior on her laptop. Also, the laptop might run out of memory. 

speed/cpu:
    report.qmd, report.py
    Makefile, results/out.pkl
speed/gpu:
    report.qmd, report.py, report.sbat,  
    Makefile, results/out.pkl

speed/report.qmd : collates results from cpu and gpu, and writes report

we can add additional detailed reports that we may run less often, in long.qmd.

## test set-up

First, set up venv in the quant directory

```
python3.12 -m pip install --upgrade pip
python3.12 -m pip install virtualenv

cd ~/git/quant
rm -rf .venv
python3.12 -m venv .venv
source ~/git/quant/.venv/bin/activate

# pip freeze
# This should show nothing, since we have a new environment.
# If things show up, the path has additional python versions
# and these should be removed.
# which python
# which pip
pip install --upgrade pip==25.1.1
pip install ~/git/pypomp
# install jax without cuda support
pip install -U jax
# asking for cuda support on a mac leads to download of an ancient
# jax version
# pip install -U "jax[cuda12]"

# for pypomp
pip install tqdm xarray pandas pytest

# for report
pip install seaborn

# check pypomp installation
# pytest ~/git/pypomp

cd ~/git/quant/speed
python -m test
```
