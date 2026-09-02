"""Shared definition of the UK measles benchmark.

Two model variants live here because they answer different questions:

  001b  the discrete-time model used by the R comparisons (`loglik/`,
        `estimation/`, `timing/`). pomp's Csnippet version of it is in
        model.R, and the two must stay in step for the comparison to mean
        anything.
  003   the continuous-time model, which has no R counterpart and is used by
        `algorithms/` to compare IF2 against IFAD.

Measles is the slow end of the benchmark suite: 365 Euler steps per year over
14 years of weekly data, with a covariate table interpolated at every step.
"""

import os
from typing import Literal

import pandas as pd
import pypomp as pp

MAIN_SEED = 594709947

# London is the largest unit and Halesworth among the smallest, so between them
# they bracket the range of population sizes the model has to cope with.
UNITS = ["London", "Halesworth"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The committed starting points, shared by the R and Python halves of
# `estimation/` and `timing/` so both frameworks do exactly the same work.
# Produced by runif_design over BOX; regenerate with `python model.py`.
STARTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "starting_parameters.csv"
)
N_FIXED_STARTS = 360

COOLING_RATE = 0.5

DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12

IVP_NAMES = ["S_0", "E_0", "I_0", "R_0"]

# The global search box. Must match `specific_bounds` in model.R.
BOX = {
    "R0": (10.0, 60.0),
    "sigma": (25.0, 100.0),
    "gamma": (25.0, 320.0),
    "iota": (0.004, 3.0),
    "rho": (0.1, 0.9),
    "sigmaSE": (0.04, 0.1),
    "psi": (0.05, 3.0),
    "cohort": (0.1, 0.7),
    "amplitude": (0.1, 0.6),
    "S_0": (0.01, 0.07),
    "E_0": (0.000004, 0.0001),
    "I_0": (0.000003, 0.001),
    "R_0": (0.9, 0.99),
}

# The perturbation sizes must match MEASLES_RW_SD in model.R.
RW_SD = pp.RWSigma(
    sigmas={
        "R0": DEFAULT_SD,
        "sigma": DEFAULT_SD,
        "gamma": DEFAULT_SD * 0.5,
        "iota": DEFAULT_SD,
        "rho": DEFAULT_SD * 0.5,
        "sigmaSE": DEFAULT_SD,
        "psi": DEFAULT_SD * 0.25,
        "cohort": DEFAULT_SD * 0.5,
        "amplitude": DEFAULT_SD * 0.5,
        **{k: DEFAULT_IVP_SD for k in IVP_NAMES},
    },
    init_names=IVP_NAMES,
).geometric_cooling(a=COOLING_RATE)

# The continuous model is perturbed less aggressively in R0 and sigma, which
# it is more sensitive to than the discrete one.
CONTINUOUS_RW_SD = pp.RWSigma(
    sigmas={
        "R0": DEFAULT_SD * 0.25,
        "sigma": DEFAULT_SD * 0.25,
        "gamma": DEFAULT_SD * 0.5,
        "iota": DEFAULT_SD,
        "rho": DEFAULT_SD * 0.5,
        "sigmaSE": DEFAULT_SD,
        "psi": DEFAULT_SD * 0.25,
        "cohort": DEFAULT_SD * 0.5,
        "amplitude": DEFAULT_SD * 0.5,
        **{k: DEFAULT_IVP_SD for k in IVP_NAMES},
    },
    init_names=IVP_NAMES,
).geometric_cooling(a=COOLING_RATE)

MLE_NAMES = [
    "R0",
    "sigma",
    "gamma",
    "iota",
    "sigmaSE",
    "psi",
    "rho",
    "cohort",
    "amplitude",
    "S_0",
    "E_0",
    "I_0",
    "R_0",
]


MeaslesModelVariant = Literal["001", "001b", "001d", "002", "002d", "003"]


def measles(unit, theta, model: MeaslesModelVariant = "001b"):
    """The measles model for one unit, at `theta`."""
    return pp.models.UKMeasles.pomp(
        unit=unit,
        theta=theta,
        model=model,
        clean=False,
    )


def measles_jax(theta, unit="London"):
    """Model 001b rebuilt on stock JAX samplers instead of pypomp's fast ones.

    Identical model, different random number generation, so `timing/` can
    price what `pypomp.random` actually buys.
    """
    import model_001b_jax as mj
    from pypomp.core.par_trans import ParTrans
    from pypomp.core.pomp import Pomp

    base = measles(unit, theta=theta)
    return Pomp(
        ys=base.ys,
        theta=theta,
        covars=base.covars,
        t0=base.t0,
        nstep=None,
        dt=1 / 365.25,
        accumvars=mj.accumvars,
        statenames=mj.statenames,
        rinit=mj.rinit,
        rproc=mj.rproc,
        dmeas=mj.dmeas,
        rmeas=mj.rmeas,
        par_trans=ParTrans(to_est=mj.to_est, from_est=mj.from_est),
    )


def measles_continuous(theta, unit="London"):
    """The continuous-time variant, used by `algorithms/`.

    `clean=True` here because model 003 is fitted rather than evaluated at
    published estimates, so the cleaned series is the right target.
    """
    return pp.models.UKMeasles.pomp(
        unit=unit,
        theta=theta,
        model="003",
        clean=True,
    )


def mle_theta(unit):
    """The He et al. (2010) estimates for `unit`, as a PompParameters."""
    mles = pd.read_csv(os.path.join(DATA_DIR, "AK_mles.csv"))
    row = mles[mles["town"] == unit]
    if len(row) == 0:
        raise ValueError(f"no MLE parameters for unit {unit!r} in AK_mles.csv")
    return pp.PompParameters({k: float(row.iloc[0][k]) for k in MLE_NAMES})


def fixed_starts(n, path=STARTS_PATH):
    """The first `n` committed starting points, as a PompParameters."""
    df = pd.read_csv(path).iloc[:n]
    if len(df) < n:
        raise ValueError(f"{path} holds {len(df)} starts, {n} requested")
    return pp.PompParameters(df.to_dict(orient="records"))


def sample_starts(n, key):
    """Draw `n` fresh starting parameter vectors from BOX.

    Used by `algorithms/`, which has no R half to agree with. The comparison
    kinds use `fixed_starts` instead.
    """
    return pp.Pomp.sample_params(BOX, n, key=key)
