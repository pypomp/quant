"""pypfilt components for the SPX benchmark (imported by dotted path from the scenario TOML).

Simulation time is the observation index: time n is observation n (1..T) and t0 = 0, with one
model step per unit of time. pypomp's SPX rproc ignores dt and takes one step per observation
interval, so this reproduces it exactly while avoiding pypfilt's fixed-size steps (real
trading days are 1 or 3 or more calendar days apart). Parameters come from the [spx] table.
"""

import numpy as np
import pypfilt
import pypfilt.io
import scipy.special
import scipy.stats

import spx_common as C

_INP = C.load_inputs()
Y_PREV = _INP["y_prev"].astype(np.float64)  # step n-1 -> n reads covars_extended[n-1]
S0 = float(_INP["S0"])
VF = float(_INP["V_floor"])

# Accumulated log p(y_1..y_n); pypfilt does not report a marginal likelihood.
LOGLIK = [0.0]


class SPXModel(pypfilt.Model):
    def field_types(self, ctx):
        return [("V", np.dtype(float)), ("S", np.dtype(float))]

    def init(self, ctx, vec):
        # x0 = (V_0, S0), deterministic, as pypomp's rinit.
        vec["V"] = ctx.settings["spx"]["V_0"]
        vec["S"] = S0

    def update(self, ctx, time_step, is_fs, prev, curr):
        p = ctx.settings["spx"]
        mu, kappa, theta, xi, rho = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"]
        y_prev = Y_PREV[int(round(time_step.start))]
        rnd = ctx.component["random"]["model"]
        V, S = prev["V"], prev["S"]
        dZ = rnd.normal(size=curr.shape)
        dWs = (y_prev - mu + 0.5 * V) / np.sqrt(V)
        dWv = rho * dWs + np.sqrt(1 - rho**2) * dZ
        curr["S"] = S + S * (mu + np.sqrt(np.maximum(V, VF)) * dWs)
        curr["V"] = np.maximum(V + kappa * (theta - V) + xi * np.sqrt(V) * dWv, VF)


class SPXObs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        V = snapshot.state_vec["V"]
        return scipy.stats.norm(loc=ctx.settings["spx"]["mu"] - 0.5 * V, scale=np.sqrt(V))

    def log_llhd(self, ctx, snapshot, obs):
        logs = super().log_llhd(ctx, snapshot, obs)
        # Weights still hold the previous (normalised) values at this point.
        LOGLIK[0] += float(scipy.special.logsumexp(logs + np.log(snapshot.vec["weight"])))
        return logs

    def from_file(self, filename, time_scale):
        fields = [pypfilt.io.time_field("time"), ("value", np.float64)]
        return pypfilt.io.read_fields(time_scale, filename, fields)
