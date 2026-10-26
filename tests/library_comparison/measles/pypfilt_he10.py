"""pypfilt components for the He10 benchmark (imported by dotted path from the scenario TOML).

Time is counted in Euler steps (pypfilt.Scalar, steps_per_unit = 1): pypfilt step
k runs Euler step k of the shared grid, and observation i sits at the end of its
interval. The scenario's [he10] table names the inputs file, the unit, T and the
13 parameter values; the model reads its covariates and Euler grid from there.
"""

import numpy as np
import pypfilt
import pypfilt.io
import scipy.special
import scipy.stats

import he10 as H

# Accumulated log p(y_1..y_t): pypfilt does not report a marginal likelihood.
LOGLIK = [0.0]
_INPUTS = {}


def _setup(ctx):
    cfg = ctx.settings["he10"]
    key = (cfg["inputs"], cfg["T"])
    if key not in _INPUTS:
        _INPUTS[key] = H.Inputs(path=cfg["inputs"], T=cfg["T"])
    th = dict(zip(H.PARAMS, (float(v) for v in cfg["theta"])))
    return _INPUTS[key], th, int(cfg["unit"])


class He10Model(pypfilt.Model):
    def field_types(self, ctx):
        return [(s, np.dtype(float)) for s in H.STATES]

    def init(self, ctx, vec):
        inp, th, u = _setup(ctx)
        x = H.np_rinit(th, inp.pop[u, 0], vec.shape[0])
        for j, s in enumerate(H.STATES):
            vec[s] = x[j]

    def update(self, ctx, time_step, is_fs, prev, curr):
        inp, th, u = _setup(ctx)
        k = int(round(time_step.start))
        rng = ctx.component["random"]["model"]
        if inp.first_step[k]:  # accumvars: W and C restart at every observation interval
            W = np.zeros(prev.shape[0])
            C = np.zeros(prev.shape[0])
        else:
            W, C = prev["W"], prev["C"]
        S, E, I, R, W, C = H.np_step(rng, prev["S"], prev["E"], prev["I"], W, C, th, k, inp, u)
        curr["S"], curr["E"], curr["I"], curr["R"], curr["W"], curr["C"] = S, E, I, R, W, C


class He10Obs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        # Only used for simulating observations / quantiles, not for filtering.
        _, th, _ = _setup(ctx)
        m = th["rho"] * snapshot.state_vec["C"]
        v = m * (1.0 - th["rho"] + th["psi"] ** 2 * m)
        return scipy.stats.norm(loc=m, scale=np.sqrt(v) + H.TOL)

    def log_llhd(self, ctx, snapshot, obs):
        _, th, _ = _setup(ctx)
        logs = H.np_dmeas(float(obs["value"]), snapshot.state_vec["C"], th["rho"], th["psi"])
        # Weights still hold the previous (normalised) values at this point.
        with np.errstate(divide="ignore"):
            logw = np.log(snapshot.vec["weight"])
        LOGLIK[0] += float(scipy.special.logsumexp(logs + logw))
        return logs

    def from_file(self, filename, time_scale):
        fields = [pypfilt.io.time_field("time"), ("value", np.float64)]
        return pypfilt.io.read_fields(time_scale, filename, fields)
