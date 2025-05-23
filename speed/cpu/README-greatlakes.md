
## Speed test for pypomp on greatlakes cpus

====

# the first version didn't work
# perhaps because of some virtual environment issue and 
# or because jax insisted on looking for cuda even when
# installed without the [cuda12] option

module load python
# python3 -m venv ~/opt/py3.12
source ~/opt/py3.12/bin/activate
pip install jax
pip install ~/git/pypomp 

srun --nodes=1 --account=stats_dept1 --ntasks-per-node=32 --pty /bin/bash

cd ~/git/quant/speed/cpu
python -m report

======

try again: this time, successfully

module unload python
rm -rf ~/opt

module load python
JAX_PLATFORMS=cpu python -m report
