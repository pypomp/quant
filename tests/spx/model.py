"""Shared definition of the SPX stochastic volatility benchmark.

Every SPX test kind (`timing/`, `estimation/`, `loglik/`) builds its model,
parameter box and random-walk SDs from here, so a change to the benchmark is a
change in one place rather than three. The model itself comes from
`pypomp.models.spx()`; what this module owns is the *experimental setup* around
it -- the search box, the perturbation sizes, and how starting points are drawn.

The SPX model uses simple random number generation (a normal draw) and one
rproc step per observation, which makes it unusually sensitive to framework
overhead in mif and pfilter. That is what makes it a good early-warning test.
"""

import numpy as np
import pypomp as pp

# Shared across kinds so that runs of different kinds are comparable.
MAIN_SEED = 631409

COOLING_RATE = 0.5

# Random-walk SDs for IF2. V_0 is an initial-value parameter, hence init_names.
RW_SD = pp.RWSigma(
    sigmas={
        "mu": 0.02,
        "kappa": 0.02,
        "theta": 0.02,
        "xi": 0.02,
        "rho": 0.02,
        "V_0": 0.1,
    },
    init_names=["V_0"],
).geometric_cooling(a=COOLING_RATE)

# Global search box.
BOX = {
    "mu": [1e-6, 1e-4],
    "kappa": [1e-8, 0.1],
    "theta": [0.000075, 0.0002],
    "xi": [1e-8, 1e-2],
    "rho": [1e-8, 1],
    "V_0": [1e-10, 1e-4],
}

# The Sun (2024) estimates, used as the fixed parameter vector for `loglik/`.
# These must stay fixed: the whole point of that kind is that theta does not
# move, so the R and Python likelihood distributions are comparable.
SUN2024_THETA = pp.PompParameters(
    {
        "mu": 3.68e-4,
        "kappa": 3.14e-2,
        "theta": 1.12e-4,
        "xi": 2.27e-3,
        "rho": -7.38e-1,
        "V_0": 7.66e-3**2,
    }
)


def spx():
    """The SPX Pomp object."""
    return pp.models.spx()


def sample_starts(n, key):
    """Draw `n` starting parameter vectors from BOX, respecting Feller's condition.

    Feller's condition (2*kappa*theta > xi^2) keeps the variance process from
    hitting zero. Sampling xi independently of kappa and theta would put a
    large fraction of the starts in a region the model cannot support, so xi is
    redrawn conditional on the pair, exactly as the R baseline does.
    """
    starts = pp.Pomp.sample_params(BOX, n, key=key)

    dicts = starts.params(as_list=True)
    for params in dicts:
        params["xi"] = float(
            np.random.uniform(
                low=0,
                high=np.sqrt(params["kappa"] * params["theta"] * 2),
            )
        )
    starts.set_params(dicts)
    return starts
