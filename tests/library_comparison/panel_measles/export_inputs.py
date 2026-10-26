"""Write inputs.npz for the 4-town He10 panel benchmark from the pypomp PanelPomp quant uses.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= ~/src/pf-bench/.venv-jax/bin/python export_inputs.py

Builds `panel_measles(fixed_starts(36))` exactly as tests/panel_measles/timing/run.py
does (units London, Halesworth, Hastings, Cardiff; model 001b per unit; R's
covariate grid substituted by `align_covariates`) and stores the same fields
as ../measles/export_inputs.py, per unit, from the PanelPomp's struct. The 36
parameter vectors are expanded per unit the way PanelPomp.pfilter does it
(shared R0, sigma, gamma, sigmaSE, cohort, amplitude; unit-specific iota, rho,
psi, S_0, E_0, I_0, R_0), giving theta of shape (36, 4, 13).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "measles"))
import export_inputs as X  # noqa: E402  (../measles/export_inputs.py)


def main():
    PM = X.load_quant_model("panel_measles")
    starts = PM.fixed_starts(36)
    pm = PM.panel_measles(starts)
    s = pm.to_struct()
    units = list(pm.unit_objects)
    assert units == list(s.unit_names)
    names = list(pm.unit_objects[units[0]].canonical_param_names)
    # Same construction as PanelPomp.pfilter: per unit, the unit Pomp's canonical order.
    thetas = np.stack(
        [
            np.array([[t[name] for name in pm.unit_objects[u].canonical_param_names]
                      for t in pm.get_unit_parameters(u, theta=pm.theta)], dtype=np.float32)
            for u in units
        ],
        axis=1,
    )
    for u in units:
        assert list(pm.unit_objects[u].canonical_param_names) == names
    X.export(
        "he10_panel4",
        dict(pm.unit_objects),
        {u: (np.asarray(s.ys_per_unit)[j, :, 0], np.asarray(s.covars_per_unit)[j]) for j, u in enumerate(units)},
        thetas,
        names,
        os.path.join(HERE, "inputs.npz"),
        extra_meta=dict(
            source="tests/panel_measles/model.py: panel_measles(fixed_starts(36)) (R covariates aligned)",
            shared=list(PM.SHARED_PARAMS),
            specific=list(PM.SPECIFIC_PARAMS),
        ),
    )


if __name__ == "__main__":
    main()
