"""Shared definition of the Dhaka cholera benchmark.

The Dhaka model is the counterpart to SPX: its rproc is very cheap, but there
are 20 interpolation steps per observation, so it is sensitive to per-step
framework overhead rather than to per-observation overhead. If wall clock ever
climbs here, that overhead is the first place to look.
"""

import os

import pandas as pd
import pypomp as pp

MAIN_SEED = 631409

COOLING_RATE = 0.8

DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 8

IVP_NAMES = ["S_0", "I_0", "Y_0", "R1_0", "R2_0", "R3_0"]

# Not estimated: rho, c, alpha and delta are held at their published values and
# Y_0 starts at zero. BOX gives each of them a degenerate range to match.
FIXED_NAMES = ["rho", "c", "alpha", "delta", "Y_0"]

# The global search box from diffPomp.
BOX = {
    "gamma": (10.0, 40.0),
    "m": (0.03, 0.60),
    "rho": (0.0, 0.0),
    "epsilon": (0.20, 30.0),
    "c": (1.0, 1.0),
    "alpha": (1.0, 1.0),
    "delta": (0.02, 0.02),
    "beta_trend": (-0.01, 0.00),
    "sigma": (1.0, 5.0),
    "tau": (0.10, 0.50),
    "bs1": (-4.0, 4.0),
    "bs2": (0.0, 8.0),
    "bs3": (-4.0, 4.0),
    "bs4": (0.0, 8.0),
    "bs5": (0.0, 8.0),
    "bs6": (0.0, 8.0),
    "omegas1": (-10.0, 0.0),
    "omegas2": (-10.0, 0.0),
    "omegas3": (-10.0, 0.0),
    "omegas4": (-10.0, 0.0),
    "omegas5": (-10.0, 0.0),
    "omegas6": (-10.0, 0.0),
    "S_0": (0.0, 1.0),
    "I_0": (0.0, 1.0),
    "Y_0": (0.0, 0.0),
    "R1_0": (0.0, 1.0),
    "R2_0": (0.0, 1.0),
    "R3_0": (0.0, 1.0),
}

RW_SD = pp.RWSigma(
    sigmas={
        k: (
            0.0
            if k in FIXED_NAMES
            else (DEFAULT_IVP_SD if k in IVP_NAMES else DEFAULT_SD)
        )
        for k in BOX
    },
    init_names=IVP_NAMES,
).geometric_cooling(a=COOLING_RATE)

# The starting points shared by the R and Python halves of `timing/`, so both
# frameworks do exactly the same work. Regenerate with `python model.py`.
STARTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "starting_parameters.csv"
)
N_FIXED_STARTS = 36


def dacca():
    """The Dhaka model, 20 Euler steps per observation.

    `nstep=20` is the same discretisation as pomp's `dacca()`, which uses
    dt = 1/240 on observations spaced 1/12 of a year apart.
    """
    return pp.models.dacca(dt=None, nstep=20)


def sample_starts(n, key):
    """Draw `n` starting parameter vectors from BOX."""
    return pp.Pomp.sample_params(BOX, n, key=key)


def fixed_starts(n=N_FIXED_STARTS, path=STARTS_PATH):
    """The first `n` committed starting points, as a PompParameters."""
    df = pd.read_csv(path).iloc[:n]
    return pp.PompParameters(df.to_dict(orient="records"))


if __name__ == "__main__":
    import jax

    key = jax.random.key(MAIN_SEED)
    key, subkey = jax.random.split(key)
    starts = sample_starts(N_FIXED_STARTS, key=subkey)
    pd.DataFrame(starts.params(as_list=True)).to_csv(STARTS_PATH, index=False)
    print(f"wrote {N_FIXED_STARTS} starting parameter vectors to {STARTS_PATH}")
