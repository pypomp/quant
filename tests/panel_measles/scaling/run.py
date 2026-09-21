"""Panel measles: GPU scaling of block-IF2 and pfilter over particles and chains.

Units are filtered one after another inside each iteration, so particles in
flight is J x chains (per unit) and an iteration costs about four single-unit
measles iterations. The grid is therefore smaller than in measles/scaling, and
M and reps are cut to keep the run near 20 min; see ../../scaling.py.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Panel measles: scaling over particles and chains on GPU (runtime & memory)"
# tags: [scaling, panel_measles, gpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "panel measles scaling (gpu)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 16GB
#       output: "results/gpu/logs/slurm-%j.out"
#     env:
#       XLA_PYTHON_CLIENT_PREALLOCATE: "false"
#     run_levels:
#       1:
#         sbatch_args: { time: "00:05:00" }
#       2:
#         sbatch_args: { time: "00:10:00" }
#       3:
#         sbatch_args: { time: "00:20:00" }
#       4:
#         sbatch_args: { time: "00:45:00" }
# --- END SLURM CONFIG ---

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(here, "../.."), os.path.join(here, "..")):
    if p not in sys.path:
        sys.path.append(os.path.abspath(p))

import scaling

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))


def build(chains, key):
    import model
    import pypomp as pp

    starts = pp.PanelPomp.sample_params(
        model.BOX,
        units=model.UNITS,
        n=chains,
        key=key,
        shared_names=model.SHARED_PARAMS,
    )
    return model.panel_measles(starts), starts, model.RW_SD


scaling.main(
    model_name="panel_measles",
    build=build,
    run_level=RUN_LEVEL,
    particle_j_grid=(
        [50, 100],
        [500, 1000],
        [1000, 5000],
        [1000, 2000, 5000, 10000, 15000],
    )[RUN_LEVEL - 1],
    particle_fixed_chains=(2, 10, 60, 100)[RUN_LEVEL - 1],
    chain_grid=(
        [2, 4],
        [10, 20],
        [60, 300],
        [125, 250, 500, 1000, 1500],
    )[RUN_LEVEL - 1],
    chain_fixed_j=(50, 500, 1000, 1000)[RUN_LEVEL - 1],
    M=(2, 3, 3, 3)[RUN_LEVEL - 1],
    reps=(2, 3, 3, 3)[RUN_LEVEL - 1],
    extra_config={"UNITS": ["London", "Halesworth", "Hastings", "Cardiff"]},
)
