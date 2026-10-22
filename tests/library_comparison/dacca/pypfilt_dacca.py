"""pypfilt components for dacca (imported by dotted path from the scenario TOML in bench_pypfilt).

Time axis: observation index (t0 = 0, y_i observed at t = i). The Euler step size (in years)
and the covariates come from inputs.npz by step index, so the math matches pypomp exactly.
The parameter vector arrives in the scenario's [dacca] table.
"""

import numpy as np
import pypfilt
import pypfilt.io
import scipy.special
import scipy.stats

import dacca_common as DC
from dacca_numpy import Dacca

# Accumulated log p(y_1..y_t) of the current run: pypfilt does not report a marginal likelihood.
LOGLIK = [0.0]
_INPUTS = {}


def _inputs(T):
    if T not in _INPUTS:
        _INPUTS[T] = DC.load_inputs(T)
    return _INPUTS[T]


class DaccaModel(pypfilt.Model):
    """One pypfilt time-step = one Euler step (time.steps_per_unit = nstep)."""

    def field_types(self, ctx):
        return [(s, np.dtype(float)) for s in DC.STATES]

    def init(self, ctx, vec):
        cfg = ctx.settings["dacca"]
        inp = _inputs(int(cfg["T"]))
        self.dacca = Dacca({k: float(v) for k, v in cfg["theta"].items()}, inp)
        self.nstep = inp["nstep_const"]
        self.substeps = int(ctx.settings["time"]["steps_per_unit"]) * int(
            ctx.get_setting(["time", "mini_steps_per_step"], 1))
        if self.substeps != self.nstep:
            raise ValueError("pypfilt steps per observation must equal the Euler steps per interval")
        x0 = self.dacca.rinit(vec.shape[0])
        for s in DC.STATES:
            vec[s] = x0[s]

    def update(self, ctx, time_step, is_fs, prev, curr):
        k = int(round(time_step.start * self.nstep))
        i, kk = divmod(k, self.nstep)
        x = {s: prev[s] for s in DC.STATES}
        if kk == 0:
            x["Mn"] = np.zeros_like(x["Mn"])  # accumulator reset at the start of the interval
        rnd = ctx.component["random"]["model"]
        x = self.dacca.step(x, i, kk, rnd.normal(size=curr.shape))
        for s in DC.STATES:
            curr[s] = x[s]


class DaccaModelMini(DaccaModel):
    """One pypfilt time-step = one observation interval, Euler steps as pypfilt mini-steps."""

    @pypfilt.model.ministeps()
    def update(self, ctx, time_step, is_fs, prev, curr):
        super().update(ctx, time_step, is_fs, prev, curr)


class DeathsObs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        x = snapshot.state_vec
        return scipy.stats.norm(loc=x["Mn"], scale=float(ctx.settings["dacca"]["theta"]["tau"]) * x["Mn"] + DC.TOL)

    def log_llhd(self, ctx, snapshot, obs):
        x = snapshot.state_vec
        logs = ctx.component["model"].dacca.dmeas(obs["value"], x["Mn"], x["count"])
        # Weights still hold the previous (normalised) values at this point.
        LOGLIK[0] += float(scipy.special.logsumexp(logs + np.log(snapshot.vec["weight"])))
        return logs

    def from_file(self, filename, time_scale):
        fields = [pypfilt.io.time_field("time"), ("value", np.float64)]
        return pypfilt.io.read_fields(time_scale, filename, fields)
