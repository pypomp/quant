"""Shared definition of the UK panel measles benchmark.

A four-unit panel on measles model 001b, with a mixed parameter structure:
R0, sigma, gamma, sigmaSE, cohort and amplitude are shared across units, the
rest are estimated per unit. model.R is the pomp/panelPomp counterpart and the
two must stay in step, since a difference in the numbers is supposed to mean a
difference in the frameworks rather than in the models.

The one place they cannot be left to agree by themselves is the covariate
table: python and R smooth the population and birth rate onto the monthly grid
with different spline implementations, and for some units (Cardiff's birth
rate especially) the two disagree enough to move the likelihood. The R grid is
committed as R_covariates.csv and `align_covariates` overwrites pypomp's with
it, so the comparison is of the algorithms and not of scipy against stats.
"""

import os

import numpy as np
import pandas as pd
import pypomp as pp
from pypomp.core.algorithms.helpers import _calc_ys_covars

MAIN_SEED = 594709947

UNITS = ["London", "Halesworth", "Hastings", "Cardiff"]

SHARED_PARAMS = ["R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude"]
SPECIFIC_PARAMS = ["iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0"]
PARAM_NAMES = SHARED_PARAMS + SPECIFIC_PARAMS

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "measles", "data")
STARTS_PATH = os.path.join(_HERE, "starting_parameters.csv")
COVARS_PATH = os.path.join(_HERE, "R_covariates.csv")

N_FIXED_STARTS = 360

COOLING_RATE = 0.5
DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12
IVP_NAMES = ["S_0", "E_0", "I_0", "R_0"]

# Must match specific_bounds in model.R.
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

# Must match PANEL_RW_SD in model.R.
RW_SD = pp.RWSigma(
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


def _dummy_theta():
    """A throwaway parameter vector for constructing the per-unit Pomps.

    The real parameters live on the PanelPomp; these only have to carry the
    right names for the model to build.
    """
    return pp.PompParameters({k: 0.5 * (lo + hi) for k, (lo, hi) in BOX.items()})


def align_covariates(pomp_dict, covar_file=COVARS_PATH):
    if not os.path.exists(covar_file):
        print(f"Warning: {covar_file} not found; using pypomp's own splines.")
        return

    all_covars: pd.DataFrame = pd.read_csv(covar_file)
    for unit, pomp_obj in pomp_dict.items():
        unit_covars = all_covars.loc[all_covars["unit"] == unit].sort_values(by="time")

        if pomp_obj.covars is None:
            raise ValueError(f"{unit} has no covariate table to align")
        pomp_obj.covars = pomp_obj.covars.sort_index()
        if len(pomp_obj.covars) != len(unit_covars):
            raise ValueError(
                f"{unit}: {len(pomp_obj.covars)} covariate rows in pypomp but "
                f"{len(unit_covars)} in {os.path.basename(covar_file)}"
            )

        pomp_obj.covars["pop"] = unit_covars["pop"].values
        pomp_obj.covars["birthrate"] = unit_covars["birthrate"].values

        dt = pomp_obj.rproc.dt
        (
            pomp_obj._covars_extended,
            pomp_obj._dt_array_extended,
            pomp_obj._nstep_array,
            pomp_obj._max_steps_per_interval,
        ) = _calc_ys_covars(
            t0=pomp_obj.t0,
            times=np.array(pomp_obj.ys.index),
            ctimes=np.array(pomp_obj.covars.index),
            covars=np.array(pomp_obj.covars),
            dt=dt,
            nstep=None if dt is not None else pomp_obj.rproc.nstep,
            order="linear",
        )
    print(f"Aligned covariates with {os.path.basename(covar_file)}")


def panel_measles(theta, units=UNITS, align=True):
    pomp_dict = {
        unit: pp.models.UKMeasles.pomp(
            unit=unit,
            theta=_dummy_theta(),
            model="001b",
            clean=False,
        )
        for unit in units
    }
    if align:
        align_covariates(pomp_dict)
    return pp.PanelPomp(pomp_dict=pomp_dict, theta=theta)


def mle_theta(units=UNITS):
    """The He et al. (2010) estimates, shared parameters averaged across units."""
    mles = pd.read_csv(os.path.join(DATA_DIR, "AK_mles.csv")).set_index("town")
    missing = [u for u in units if u not in mles.index]
    if missing:
        raise ValueError(f"no MLE parameters for {missing} in AK_mles.csv")
    mles = mles.loc[units]

    shared = pd.DataFrame(
        {"shared": [float(mles[p].mean()) for p in SHARED_PARAMS]},
        index=pd.Index(SHARED_PARAMS),
    )
    return pp.PanelParameters(
        theta={"shared": shared, "unit_specific": mles[SPECIFIC_PARAMS].T}
    )


def fixed_starts(n, path=STARTS_PATH, units=UNITS):
    """The first `n` committed starting points, as a PanelParameters.

    Both halves of estimation/ and timing/ read this same file, so the two
    frameworks are given exactly the same work.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"starting parameters not found at {path}")
    df: pd.DataFrame = pd.read_csv(path)

    available = df["replicate"].nunique()
    if available < n:
        raise ValueError(f"{path} holds {available} starts, {n} requested")

    reps = []
    for j in sorted(df["replicate"].unique())[:n]:
        rep: pd.DataFrame = df.loc[df["replicate"] == j]
        shared_df: pd.DataFrame = rep.loc[rep["unit"] == "shared"]
        shared = shared_df.set_index("param")["value"]
        specific_df: pd.DataFrame = rep.loc[rep["unit"] != "shared"]
        specific = specific_df.pivot(index="param", columns="unit", values="value")
        reps.append(
            {
                "shared": pd.DataFrame({"shared": shared.loc[SHARED_PARAMS]}),
                "unit_specific": specific.loc[SPECIFIC_PARAMS, units],
            }
        )
    return pp.PanelParameters(theta=reps)


def write_starts(n=N_FIXED_STARTS, path=STARTS_PATH, units=UNITS):
    """Regenerate the committed starting points.

    Run deliberately and commit the result -- rerunning it invalidates every
    estimate already recorded against the old design.
    """
    import jax

    theta = pp.PanelPomp.sample_params(
        BOX,
        units=units,
        n=n,
        key=jax.random.key(MAIN_SEED),
        shared_names=SHARED_PARAMS,
    )

    records = []
    for rep_idx, rep in enumerate(theta._to_list()):
        if rep["shared"] is not None:
            for param in rep["shared"].index:
                records.append(
                    {
                        "replicate": rep_idx,
                        "unit": "shared",
                        "param": param,
                        "value": float(rep["shared"].loc[param, "shared"]),
                    }
                )
        if rep["unit_specific"] is not None:
            spec = rep["unit_specific"]
            for unit in spec.columns:
                for param in spec.index:
                    records.append(
                        {
                            "replicate": rep_idx,
                            "unit": unit,
                            "param": param,
                            "value": float(spec.loc[param, unit]),
                        }
                    )

    pd.DataFrame(records).to_csv(path, index=False)
    print(f"wrote {n} starting parameter vectors to {path}")


if __name__ == "__main__":
    write_starts()
