"""Write inputs.npz: everything the five dacca adapters need, taken from pypomp's own struct.

The model is built exactly as quant's tests/dacca/model.py builds it (pypomp's dacca(),
nstep = 20 Euler steps per monthly observation), so every library consumes the same data,
step sizes and interpolated covariates that pypomp's pfilter uses.

inputs.npz
  ys        (T,)          float64  observed deaths
  times     (T,)          float64  observation times (years); t0 is a scalar
  nstep     (T,)          int64    Euler steps in observation interval i (times0[i] -> times0[i+1])
  dt        (N,)          float32  per-Euler-step dt, N = sum(nstep)   (struct.dt_array_extended)
  covars    (N + 1, C)    float32  covariates at the start of every Euler step, plus the last
                                   observation time (struct.covars_extended); row 0 (t0) feeds rinit
  thetas    (36, P)       float64  natural-scale starting points, model.fixed_starts(36),
                                   columns in pypomp's canonical order (paramnames)
  theta_published (1, P)  float64  pypomp's built-in dacca parameters (King et al. 2008 fit):
                                   loglik near the MLE with small Monte Carlo sd, a sharp check
  statenames, paramnames, covarnames, accumvars, t0

dt and covars are stored at the float32 precision pypomp computes with; the float64
libraries (particles, pypfilt) upcast them, so all five see bit-identical inputs.

usage: .venv-jax/bin/python export_inputs.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DACCA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "dacca"))
sys.path.insert(0, DACCA_DIR)

import model as dacca_model  # noqa: E402  (quant tests/dacca/model.py)

N_THETAS = 36


def main():
    m = dacca_model.dacca()
    s = m.to_struct()
    paramnames = list(s.param_names)

    ys = m.ys.iloc[:, 0].to_numpy(dtype=np.float64)
    assert np.array_equal(ys.astype(np.float32), np.asarray(s.ys)[:, 0])
    times = np.asarray(m.ys.index, dtype=np.float64)
    nstep = np.asarray(s.nstep_array).astype(np.int64)
    dt = np.asarray(s.dt_array_extended)
    covars = np.asarray(s.covars_extended)
    assert dt.shape == (nstep.sum(),) and covars.shape[0] == nstep.sum() + 1

    rows = dacca_model.fixed_starts(N_THETAS).params(as_list=True)
    thetas = np.array([[float(r[k]) for k in paramnames] for r in rows], dtype=np.float64)

    pub = m.theta.params(as_list=True)[0]
    theta_published = np.array([[float(pub[k]) for k in paramnames]], dtype=np.float64)

    out = os.path.join(HERE, "inputs.npz")
    np.savez(
        out,
        ys=ys,
        times=times,
        t0=np.float64(m.t0),
        nstep=nstep,
        dt=dt,
        covars=covars,
        thetas=thetas,
        theta_published=theta_published,
        statenames=np.array(list(m.statenames)),
        paramnames=np.array(paramnames),
        covarnames=np.array(list(m.covar_names)),
        accumvars=np.array([list(m.statenames)[i] for i in s.accumvars]),
    )
    print(f"wrote {out}: T={len(ys)}, Euler steps={dt.size} (nstep {sorted(set(nstep.tolist()))}), "
          f"dt={sorted(set(dt.tolist()))}, covars {covars.shape} {covars.dtype}, thetas {thetas.shape}")
    print("states", list(m.statenames), "accumvars", [list(m.statenames)[i] for i in s.accumvars])
    print("params", paramnames)
    print("covars", list(m.covar_names))


if __name__ == "__main__":
    main()
