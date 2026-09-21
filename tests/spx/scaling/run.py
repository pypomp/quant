"""SPX: GPU scaling of mif and pfilter over particle counts and chain counts.

Grid points span roughly 1e5 to 5e6 concurrent particles (J x chains) to locate
where the GPU saturates. M is kept small because per-iteration cost, not
convergence, is the quantity of interest; see ../../scaling.py.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SPX: scaling over particles and chains on GPU (runtime & memory)"
# tags: [scaling, spx, gpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "spx scaling (gpu)"
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
#         sbatch_args: { time: "00:40:00" }
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

    starts = model.sample_starts(chains, key=key)
    return model.spx(), starts, model.RW_SD


scaling.main(
    model_name="spx",
    build=build,
    run_level=RUN_LEVEL,
    particle_j_grid=(
        [50, 100],
        [500, 1000],
        [1000, 5000, 20000],
        [1000, 2500, 5000, 10000, 20000, 40000],
    )[RUN_LEVEL - 1],
    particle_fixed_chains=(2, 10, 60, 120)[RUN_LEVEL - 1],
    chain_grid=(
        [2, 4],
        [10, 20],
        [60, 300, 1200],
        [125, 250, 500, 1000, 2000, 4000],
    )[RUN_LEVEL - 1],
    chain_fixed_j=(50, 500, 1000, 1000)[RUN_LEVEL - 1],
    M=(2, 3, 5, 10)[RUN_LEVEL - 1],
    reps=(2, 3, 5, 10)[RUN_LEVEL - 1],
)
