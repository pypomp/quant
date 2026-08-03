import os
import numpy as np
import pandas as pd
from pypomp.core.algorithms.helpers import _calc_ys_covars


def align_covariates(pomp_dict, covar_file="../R_covariates.csv"):
    """
    Overrides the population and birthrate covariates on the given pomp objects
    with the values computed in R (loaded from R_covariates.csv), and re-calculates
    the JAX covariate grid arrays. The discrepancy between the two languages occurs due
    to their available spline smoothing functions being slightly different, and
    the difference is significant for some cities.
    """
    if os.path.exists(covar_file):
        all_covars = pd.read_csv(covar_file)
        assert isinstance(all_covars, pd.DataFrame)

        for unit, pomp_obj in pomp_dict.items():
            unit_df = all_covars[all_covars["unit"] == unit]
            assert isinstance(unit_df, pd.DataFrame)
            unit_covars = unit_df.sort_values(by="time")

            assert pomp_obj.covars is not None
            pomp_obj.covars = pomp_obj.covars.sort_index()
            assert len(pomp_obj.covars) == len(unit_covars)

            pomp_obj.covars["pop"] = unit_covars["pop"].values
            pomp_obj.covars["birthrate"] = unit_covars["birthrate"].values

            dt = pomp_obj.rproc.dt
            nstep = None if dt is not None else pomp_obj.rproc.nstep

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
                nstep=nstep,
                order="linear",
            )
        print(f"Successfully aligned covariates with {covar_file}")
    else:
        print(
            f"Warning: {covar_file} not found. Running with default Python spline covariates."
        )
