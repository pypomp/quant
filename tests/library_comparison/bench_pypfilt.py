"""pypfilt: user-defined model + obs model via a scenario TOML; systematic ("deterministic") resampling every obs.

usage: python bench_pypfilt.py MODEL J [J ...]
"""

import io
import os
import sys

import pypfilt

import common as C
import pypfilt_models as M

model_name = sys.argv[1]
Js = [int(j) for j in sys.argv[2:]]
y = C.data(model_name)

obs_file = os.path.join(C.HERE, f"pypfilt-{model_name}-obs.ssv")
with open(obs_file, "w") as f:
    f.write("time value\n")
    for t, v in enumerate(y, start=1):
        f.write(f"{t} {float(v)!r}\n")

if model_name == "lg":
    comp, steps, prior, sdev = "pypfilt_models.LGModel", 1, 'x = { name = "norm", args.loc = 0.0, args.scale = 1.0 }', C.SY
else:
    comp, steps, sdev = "pypfilt_models.SDEModel", C.NSTEP, C.TAU
    prior = f'x = {{ name = "norm", args.loc = {C.Z0M!r}, args.scale = {C.Z0S!r} }}'


def toml(J, seed):
    return f"""
[components]
model = "{comp}"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0.0
until = {float(C.T)}
steps_per_unit = {steps}

[prior]
{prior}

[filter]
particles = {J}
prng_seed = {seed}
history_window = -1
resample.threshold = 1.0
resample.method = "deterministic"
regularisation.enabled = false

[files]
input_directory = "{C.HERE}"
output_directory = "{C.HERE}"
delete_cache_file_before_forecast = true
delete_cache_file_after_forecast = true

[scenario.bench]
name = "bench"

[scenario.bench.observations.y]
model = "pypfilt_models.GaussObs"
file = "{obs_file}"
parameters.sdev = {sdev!r}
"""


for J in Js:
    def run(seed, J=J):
        (inst,) = list(pypfilt.load_instances(io.StringIO(toml(J, seed + 1))))
        ctx = inst.build_context()
        M.LOGLIK[0] = 0.0
        pypfilt.fit(ctx, filename=None)
        return M.LOGLIK[0]

    first, med, lls = C.timeit(run, n_rep=3)
    C.record("pypfilt", "deterministic", model_name, "cpu", J, first, med, lls)
