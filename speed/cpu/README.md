

## Testing on cpus; the greatlakes cpu partition or a Mac

The basic cpu benchmark test is run with
```
make report.html
```
which produces a file `results/cpu.pkl` that is read by the report generated at cpu.html.
These results also get incorporated into combined reports in the upstream directory.
The longer version is run with
```
make long.html
```
and this provides additional diagnostics.

Additionally, we can extract the code into a .py file and run at the command line:
```
make report.py 
python -m report
```

The text output from this can be added, with comments, to `speed/notes.txt`.

## Notes to help setup/test/debug the code.

On greatlakes, set up the Python environment on the login node

```
module load python
python -m venv ~/opt/py3.12
source ~/opt/py3.12/bin/activate
pip install jax[cuda12]
pip install pypomp pytest tqdm
git clone git@github.com:pypomp/pypomp
```

jax can be built without cuda, though presumably it does no harm to include cuda if this causes no error messages.

```
pip install -U "jax[cuda12]"
```

Then install everything else needed, with pypomp installed from the local GitHub clone

```
pip install ~/git/pypomp
pip install pytest tqdm
```

