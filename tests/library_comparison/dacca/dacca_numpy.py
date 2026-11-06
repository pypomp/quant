"""The dacca model in float64 numpy, shared by the particles and pypfilt adapters.

Same math as pypomp's dacca (see dacca_common's docstring). States travel as a dict of 1-D
arrays named as in pypomp. step() is one Euler step; interval() is one observation interval.
"""

import numpy as np

from dacca_common import LTOL, TOL

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


class Dacca:
    def __init__(self, theta, inp):
        """theta: dict of natural-scale parameters; inp: dacca_common.load_inputs() output."""
        self.th = theta
        self.nstep = inp["nstep_const"]
        self.cov0 = inp["cov0"].astype(np.float64)
        self.cov_blocks = inp["cov_blocks"].astype(np.float64)
        self.dt_blocks = inp["dt_blocks"].astype(np.float64)
        bs = np.array([theta[f"bs{i}"] for i in range(1, 7)])
        om = np.array([theta[f"omegas{i}"] for i in range(1, 7)])
        cb = self.cov_blocks
        # Per-step transmission and reservoir rates do not depend on the particle.
        self.beta = np.exp(theta["beta_trend"] * cb[..., 0] + cb[..., 3:9] @ bs)  # (T, nstep)
        self.omega = np.exp(cb[..., 3:9] @ om)

    def rinit(self, n):
        th, pop = self.th, self.cov0[2]
        ivp = np.array([th["S_0"], th["I_0"], th["Y_0"], th["R1_0"], th["R2_0"], th["R3_0"]])
        S, I, Y, R1, R2, R3 = pop * ivp / ivp.sum()
        full = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
        return dict(S=full(S), I=full(I), Y=full(Y), Mn=full(0.0), R1=full(R1), R2=full(R2),
                    R3=full(R3), count=full(0.0))

    def step(self, x, i, k, eps):
        """Euler step k of observation interval i; eps: (n,) standard normals. Returns a new dict.

        The caller resets x["Mn"] to 0 before step 0 of each interval.
        """
        th = self.th
        gamma, m, rho, c = th["gamma"], th["m"], th["rho"], th["c"]
        sigma, alpha, delta = th["sigma"], th["alpha"], th["delta"]
        neps = 3.0 * th["epsilon"]
        S, I, Y, Mn, R1, R2, R3 = x["S"], x["I"], x["Y"], x["Mn"], x["R1"], x["R2"], x["R3"]
        dt = self.dt_blocks[i, k]
        dpopdt, pop = self.cov_blocks[i, k, 1], self.cov_blocks[i, k, 2]
        dw = eps * np.sqrt(dt)
        effI = (I / pop) ** alpha
        inf = (self.omega[i, k] + (self.beta[i, k] + sigma * dw / dt) * effI) * S
        pass0, pass1, pass2, pass3 = gamma * I, neps * R1, neps * R2, neps * R3
        disease = m * I
        S_n = S + (dpopdt + delta * pop - inf - delta * S + pass3 + rho * Y) * dt
        I_n = I + (c * inf - disease - delta * I - pass0) * dt
        Y_n = Y + ((1 - c) * inf - delta * Y - rho * Y) * dt
        R1_n = R1 + (pass0 - pass1 - delta * R1) * dt
        R2_n = R2 + (pass1 - pass2 - delta * R2) * dt
        R3_n = R3 + (pass2 - pass3 - delta * R3) * dt
        Mn_n = Mn + disease * dt
        neg = (S_n < 0) | (I_n < 0) | (Y_n < 0) | (Mn_n < 0) | (R1_n < 0) | (R2_n < 0) | (R3_n < 0)
        return dict(S=np.maximum(S_n, 0), I=np.maximum(I_n, 0), Y=np.maximum(Y_n, 0),
                    Mn=np.maximum(Mn_n, 0), R1=np.maximum(R1_n, 0), R2=np.maximum(R2_n, 0),
                    R3=np.maximum(R3_n, 0), count=x["count"] + neg)

    def interval(self, x, i, normal):
        """Advance x through observation interval i; normal(n) draws n standard normals."""
        x = dict(x, Mn=np.zeros_like(x["Mn"]))  # accumulator reset
        n = x["S"].shape[0]
        for k in range(self.nstep):
            x = self.step(x, i, k, normal(n))
        return x

    def dmeas(self, y, Mn, count):
        """Per-particle log measurement density of y."""
        v = self.th["tau"] * Mn
        sc = v + TOL
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = (y - Mn) / sc
            lp = -0.5 * z * z - np.log(sc) - LOG_SQRT_2PI
            ll = np.logaddexp(lp, LTOL)
        return np.where(np.isfinite(v) & (count <= 0), ll, LTOL)
