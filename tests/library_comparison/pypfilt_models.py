"""pypfilt components for the benchmark (imported by dotted path from the scenario TOML)."""

import numpy as np
import pypfilt
import pypfilt.io
import scipy.special
import scipy.stats

import common as C

# Accumulated log p(y_1..y_t); pypfilt does not report a marginal likelihood.
LOGLIK = [0.0]


class LGModel(pypfilt.Model):
    def field_types(self, ctx):
        return [("x", np.dtype(float))]

    def update(self, ctx, time_step, is_fs, prev, curr):
        rnd = ctx.component["random"]["model"]
        curr["x"] = C.A * prev["x"] + C.SX * rnd.normal(size=curr.shape)


class SDEModel(pypfilt.Model):
    def field_types(self, ctx):
        return [("x", np.dtype(float))]

    def update(self, ctx, time_step, is_fs, prev, curr):
        rnd = ctx.component["random"]["model"]
        dt = time_step.dt
        z = prev["x"]
        curr["x"] = z + (C.R * (1 - np.exp(z)) - C.SIG**2 / 2) * dt \
            + C.SIG * np.sqrt(dt) * rnd.normal(size=curr.shape)


class GaussObs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        return scipy.stats.norm(loc=snapshot.state_vec["x"], scale=self.settings["parameters"]["sdev"])

    def log_llhd(self, ctx, snapshot, obs):
        logs = super().log_llhd(ctx, snapshot, obs)
        # Weights still hold the previous (normalised) values at this point.
        logw = np.log(snapshot.vec["weight"])
        LOGLIK[0] += float(scipy.special.logsumexp(logs + logw))
        return logs

    def from_file(self, filename, time_scale):
        fields = [pypfilt.io.time_field("time"), ("value", np.float64)]
        return pypfilt.io.read_fields(time_scale, filename, fields)
