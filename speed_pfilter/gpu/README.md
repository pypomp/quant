

## Testing on the greatlakes gpu partition

* Using one of the 52 Nvidia Tesla V100 GPUs.

* report.qmd is essentially the same as `../cpu/report.qmd`.

* The code is run on greatlakes on a .py
```
quarto convert report.qmd    # makes report.ipynb
jupyter nbconvert --to python report.ipynb # makes report.py
rm report.ipynb # tidying up
python -m report
```

The test is run on greatlakes gpu with 
```
sbatch gpu.sbat
```
which produces files in `results_2`, e.g., `results_2/mif-test.pkl`, that are read by the report generated at gpu.html. These results also get incorporated into combined reports in the upstream directory.

## Notes to help setup/test/debug the code.


To obtain access to a GPU via an interactive seesion (i.e., a terminal prompt) once can do

salloc --account=ionides0 --partition=gpu --gpus=v100:1 --cpus-per-gpu=1 --mem=8000

I set up the Python environment on the login node

```
module load python
python -m venv ~/opt/py3.12
source ~/opt/py3.12/bin/activate
pip install jax[cuda12]
pip install pypomp pytest tqdm
git clone git@github.com:pypomp/pypomp
```

jax shoulb be built with cuda:

```
pip install -U "jax[cuda12]"
```

Then install everything else needed, with pypomp installed from the local GitHub clone

```
pip install ~/git/pypomp
pip install pytest tqdm
```

