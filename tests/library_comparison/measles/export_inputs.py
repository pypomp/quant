"""Write inputs.npz for the London He10 benchmark from the pypomp model quant uses.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= ~/src/pf-bench/.venv-jax/bin/python export_inputs.py

Builds `measles("London", fixed_starts(36), model="001b")` exactly as
tests/measles/timing/run.py does and stores everything the other libraries
need, taken from the model's struct (so values are what pypomp itself uses):
ys, observation times, t0, nstep per interval, dt per Euler step, the
interpolated covariates at every Euler step, and the 36 parameter vectors
(as pypomp's float32 array, stored in float64).

It also stores, per Euler step, the time t that pypomp's Euler loop passes to
rproc (float32: the previous observation time plus dt added once per substep),
checked against pypomp's own loop with a probe model, and the two indicators
rproc derives from it (term time and the cohort-entry day), evaluated with
model 001b's float32 expressions. Some term boundaries fall exactly on a grid
day, where float32 and float64 disagree, so every library uses these
indicators instead of recomputing them from t in its own precision.
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_quant_model(subdir):
    """Import quant's tests/<subdir>/model.py under a unique name."""
    import importlib.util

    path = os.path.join(TESTS, subdir, "model.py")
    sys.path.insert(0, os.path.join(TESTS, subdir))
    spec = importlib.util.spec_from_file_location(f"quant_{subdir}_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pypomp_step_times(t0, times32, nstep, dt32):
    """t at every Euler step as pypomp's loop computes it (float32 accumulation)."""
    K = int(nstep.sum())
    t_step = np.empty(K, dtype=np.float32)
    k = 0
    for i in range(len(nstep)):
        t = np.float32(t0) if i == 0 else times32[i - 1]
        for _ in range(int(nstep[i])):
            t_step[k] = t
            t = np.float32(t + dt32[k])
            k += 1
    return t_step


def flags_001b(t_step, dt32):
    """term-time and cohort indicators with model 001b's own expressions, in float32 JAX."""
    import jax.numpy as jnp

    t = jnp.asarray(t_step, dtype=jnp.float32)
    dt = jnp.asarray(dt32, dtype=jnp.float32)
    t_mod = t - jnp.floor(t)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    t_days = t_mod * 365.25
    in_term_time = (
        ((t_days >= 7) & (t_days <= 100))
        | ((t_days >= 115) & (t_days <= 199))
        | ((t_days >= 252) & (t_days <= 300))
        | ((t_days >= 308) & (t_days <= 356))
    )
    return np.asarray(in_term_time), np.asarray(is_cohort_time)


def probe_step_times(pomp):
    """Run pypomp's real Euler loop with a probe rproc that records t at every substep."""
    import jax
    import jax.numpy as jnp
    import pandas as pd
    import pypomp as pp
    import pypomp.functional as F

    M = int(np.max(pomp._nstep_array))
    names = ["n"] + [f"t{j}" for j in range(M)]

    def rinit(theta_, key, covars, t0):
        return {n: 0.0 for n in names}

    @pp.vectorized
    def rproc(X_, theta_, key, covars, t, dt):
        n = X_["n"]
        out = {f"t{j}": jnp.where(n == j, t, X_[f"t{j}"]) for j in range(M)}
        out["n"] = n + 1.0
        return out

    def dmeas(Y_, X_, theta_, covars, t):
        return 0.0 * theta_["a"]

    probe = pp.Pomp(
        ys=pd.DataFrame({"y": np.zeros(len(pomp.ys))}, index=pomp.ys.index),
        theta=pp.PompParameters({"a": 1.0}),
        statenames=names,
        t0=pomp.t0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        nstep=None,
        dt=pomp.rproc.dt,
        accumvars=names,
    )
    s = probe.to_struct()
    assert np.array_equal(np.asarray(s.dt_array_extended), np.asarray(pomp.to_struct().dt_array_extended))
    out = F.pfilter(s, jnp.ones((1, 1)), 1, jax.random.split(jax.random.key(0), 1)[:, None],
                    filter_mean=True)
    fm = np.asarray(out["filter_mean"])[0, 0]  # (T, 1 + M)
    nstep = np.asarray(pomp._nstep_array)
    return np.concatenate([fm[i, 1:1 + nstep[i]] for i in range(len(nstep))]).astype(np.float32)


def export(model_tag, pomps, unit_structs, thetas, param_names, out_path, extra_meta=None):
    """pomps: {unit: Pomp}; unit_structs: {unit: (ys (T,), covars_extended (K+1, C))};
    thetas: (n, U, P) float32 in `param_names` order."""
    units = list(pomps)
    ref = pomps[units[0]]
    s = ref.to_struct()
    times64 = np.asarray(ref.ys.index, dtype=np.float64)
    times32 = np.asarray(s.times, dtype=np.float32)
    nstep = np.asarray(s.nstep_array, dtype=np.int64)
    dt32 = np.asarray(s.dt_array_extended, dtype=np.float32)
    for u in units[1:]:
        su = pomps[u].to_struct()
        assert np.array_equal(np.asarray(pomps[u].ys.index), np.asarray(ref.ys.index)), u
        assert np.array_equal(np.asarray(su.nstep_array), np.asarray(s.nstep_array)), u
        assert np.array_equal(np.asarray(su.dt_array_extended), np.asarray(s.dt_array_extended)), u
        assert pomps[u].t0 == ref.t0, u

    t_step = pypomp_step_times(ref.t0, times32, nstep, dt32)
    t_probe = probe_step_times(ref)
    n_bad = int(np.sum(t_probe != t_step))
    if n_bad:
        raise AssertionError(f"{n_bad} Euler-step times differ from pypomp's own loop")
    term, cohort = flags_001b(t_step, dt32)
    years = np.floor(t_step).astype(int)
    per_year = np.bincount(years[cohort] - years.min(), minlength=years.max() - years.min() + 1)
    covar_names = list(ref.covars.columns)
    ipop, ibr = covar_names.index("pop"), covar_names.index("birthrate")
    ys = np.stack([np.asarray(unit_structs[u][0], dtype=np.float64) for u in units])
    cov = np.stack([np.asarray(unit_structs[u][1], dtype=np.float64) for u in units])  # (U, K+1, C)
    K = int(nstep.sum())
    assert cov.shape[1] == K + 1 and dt32.shape[0] == K

    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    import importlib.metadata as md

    np.savez(
        out_path,
        model=np.array(model_tag),
        unit_names=np.array(units),
        param_names=np.array(param_names),
        theta=np.asarray(thetas, dtype=np.float32).astype(np.float64),
        ys=ys,
        times=times64,
        times_f32=times32.astype(np.float64),
        t0=np.array(float(ref.t0)),
        nstep=nstep,
        dt=dt32.astype(np.float64),
        pop=cov[:, :, ipop],
        birthrate=cov[:, :, ibr],
        covars=cov,
        covar_names=np.array(covar_names),
        t_step=t_step.astype(np.float64),
        term=term,
        cohort_step=cohort,
        meta=np.array(str(dict(pypomp=md.version("pypomp"), quant_commit=commit, **(extra_meta or {})))),
    )
    print(f"wrote {out_path}")
    print(f"  model {model_tag}, units {units}, T {ys.shape[1]}, K {K} Euler steps, "
          f"nstep {np.unique(nstep).tolist()}, dt {np.unique(dt32).tolist()}")
    print(f"  theta {np.shape(thetas)}, params {list(param_names)}")
    print(f"  t_step matches pypomp's loop at all {K} steps; term-time steps {int(term.sum())}, "
          f"cohort steps per year {per_year.tolist()}")
    print(f"  NaN observations per unit {np.isnan(ys).sum(axis=1).tolist()}")


def main():
    M = load_quant_model("measles")
    starts = M.fixed_starts(36)
    pomp = M.measles("London", theta=starts)
    s = pomp.to_struct()
    thetas = np.asarray(starts.to_jax_array(pomp.canonical_param_names))[:, None, :]
    export(
        "he10_london",
        {"London": pomp},
        {"London": (np.asarray(s.ys)[:, 0], np.asarray(s.covars_extended))},
        thetas,
        list(pomp.canonical_param_names),
        os.path.join(HERE, "inputs.npz"),
        extra_meta=dict(source="tests/measles/model.py: measles('London', fixed_starts(36), model='001b'), clean=False"),
    )


if __name__ == "__main__":
    # CPU only (set before JAX initialises a backend); importers keep their own settings.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
