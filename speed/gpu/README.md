

## Testing on the greatlakes gpu partition

* Using one of the 52 Nvidia Tesla V100 GPUs.

* report.qmd is essentially the same as `../cpu/report.qmd`.

* The code is run on greatlakes on a .py
```

make report.py # extracts Python code from report.qmd
python -m report
```

The test can be run interactively on a greatlakes gpu.

The test is run on greatlakes gpu with 
```
sbatch gpu.sbat
```
which produces files in `results_2`, e.g., `results_2/mif-test.pkl`, that are read by the report generated at gpu.html. These results also get incorporated into combined reports in the upstream directory.

Additionally, notes on speed tests can be put into the speed/notes.txt file.


## Notes to help setup/test/debug the code.


To obtain access to a GPU via an interactive seesion (i.e., a terminal prompt) once can do

salloc --account=ionides1 --partition=gpu --gpus=v100:1 --cpus-per-gpu=1 --mem=8000

I set up the Python environment on the login node

```
module load python
python -m venv ~/opt/py3.12
source ~/opt/py3.12/bin/activate
pip install pypomp pytest tqdm
git clone git@github.com:pypomp/pypomp
```

jax should be built with cuda:

```
pip install -U "jax[cuda12]"
```

Then install everything else needed, with pypomp installed from the local GitHub clone

```
pip install ~/git/pypomp
pip install pytest tqdm
```

Then moving on to the GPU,

salloc --account=ionides1 --partition=gpu --gpus=v100:1 --cpus-per-gpu=1 --mem=8000
nvidia-smi

It is necessary to restart the Python environment for the new machine. A simple JAX test is working. So are the pypomp tests.

```
module load python
source ~/opt/py3.12/bin/activate
python -c "import jax.numpy as np; print(np.ones((3,3)))"
pytest ~/git/pypomp/test
```

Then,
```
cd ~git
git clone git@github.com:pypomp/quant
cd ~/git/quant/speed_pfilter/gpu
python -m report
# sbatch gpu.sbat # batch alternative

```

Note that additional libraries may be needed:
```
pip install pandas seaborn
```