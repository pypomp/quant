"""Shared PyPOMP model for canonical R pomp::blowflies1().

JAX's standard Gamma, Poisson and Binomial samplers preserve the discrete
model. The daily transition has an explicit AD guard: a pathwise gradient
would not be its likelihood score. No approximate fast sampler or surrogate
gradient is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
import numpy as np
import pandas as pd
import pypomp as pp
from pypomp.core.par_trans import ParTrans

MAIN_SEED = 828411
DATA_PATH = Path(__file__).with_name("data") / "nicholson_population_I.csv"
UPSTREAM_COMMIT = "7cfb3f9aa84c85de687b82d71b081016b9cc5762"
PARAM_NAMES = ("P", "delta", "N0", "sigma.P", "sigma.d", "sigma.y")
LAG_NAMES = [f"N{i}" for i in range(1, 16)]
STATENAMES = [*LAG_NAMES, "R", "S", "e", "eps"]
PATHWISE_AD_SUPPORTED = False
AD_ERROR = (
    "blowflies1 has discrete Poisson/binomial transitions; ordinary pathwise AD "
    "does not provide their likelihood score. Forward simulation, particle "
    "filtering and derivative-free IF2 are allowed, but transition AD/IFAD "
    "requires a separately specified and validated gradient estimator."
)


@dataclass(frozen=True)
class Parameters:
    P: float = 3.2838
    delta: float = 0.16073
    N0: float = 679.94
    sigma_P: float = 1.3512
    sigma_d: float = 0.74677
    sigma_y: float = 0.026649

    def __post_init__(self):
        values = np.array(list(self.as_r_dict().values()))
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("The log-transformed parameter domain is finite and positive.")

    def as_r_dict(self):
        return dict(zip(PARAM_NAMES, (self.P, self.delta, self.N0,
                                     self.sigma_P, self.sigma_d, self.sigma_y)))


def load_data():
    """The 200 source values include eight observations for initialization."""
    values = np.loadtxt(DATA_PATH, delimiter=";", skiprows=1, dtype=np.int64)
    return values[:, 0].copy(), values[:, 1].copy()


def observations():
    times, values = load_data()
    keep = times > 14
    return times[keep], values[keep]


# R's linear interpolation, reversed into the current-to-lagged state order.
_days, _counts = load_data()
INITIAL_HISTORY = tuple(np.interp(np.arange(15), _days, _counts)[::-1].copy())


def to_est(theta):
    return {name: jnp.log(theta[name]) for name in PARAM_NAMES}


def from_est(theta):
    return {name: jnp.exp(theta[name]) for name in PARAM_NAMES}


def rinit(theta_, key, covars, t0):
    """Use the frozen, actual R linear-interpolation initialization."""
    history = jnp.asarray(INITIAL_HISTORY)
    result = dict(zip(LAG_NAMES, history, strict=True))
    result.update({name: jnp.asarray(0.0, dtype=history.dtype) for name in ("R", "S", "e", "eps")})
    return result


def advance_buffer(history, recruits, survivors):
    """Shift only after draws have used the pre-update current and lag values."""
    return jnp.concatenate((jnp.reshape(recruits + survivors, (1,)), history[:-1]))


@jax.custom_jvp
def daily_transition(history, theta, key):
    """One daily transition; array output holds N1..N15, R, S, e, eps."""
    P, delta, N0, sigma_P, sigma_d, _ = theta
    e_key, eps_key, recruits_key, survivors_key = jax.random.split(key, 4)
    e = jax.random.gamma(e_key, 1.0 / sigma_P**2, dtype=history.dtype) * sigma_P**2
    eps = jax.random.gamma(eps_key, 1.0 / sigma_d**2, dtype=history.dtype) * sigma_d**2
    # No rounding of fractional interpolated lag values, and no clipping.
    rate = P * history[14] * jnp.exp(-history[14] / N0) * e
    recruits = jax.random.poisson(recruits_key, rate).astype(history.dtype)
    survivors = jax.random.binomial(
        survivors_key, history[0], jnp.exp(-delta * eps), dtype=history.dtype
    )
    updated = advance_buffer(history, recruits, survivors)
    return jnp.concatenate((updated, jnp.stack((recruits, survivors, e, eps))))


@daily_transition.defjvp
def _block_transition_ad(primals, tangents):
    # Raising at tracing time also guards reverse-mode grad/value_and_grad.
    raise NotImplementedError(AD_ERROR)


def rproc(X_, theta_, key, covars, t, dt):
    """Daily callback; the factory enforces dt=1 by its fixed grid and nstep=2.

    Non-daily calls return NaN states rather than quietly changing the model.
    The mask is trace-safe and permits PyPOMP's dummy-dt construction probe.
    """
    history = jnp.stack([X_[name] for name in LAG_NAMES])
    theta = jnp.stack([theta_[name] for name in PARAM_NAMES])
    output = daily_transition(history, theta, key)
    output = jnp.where(jnp.asarray(dt) == 1.0, output, jnp.full_like(output, jnp.nan))
    return dict(zip(STATENAMES, output, strict=True))


def dmeas(Y_, X_, theta_, covars=None, t=None):
    """Negative-binomial mass with mean N1 and size sigma.y**(-2)."""
    y, mu = jnp.asarray(Y_["y"]), jnp.asarray(X_["N1"])
    size = 1.0 / theta_["sigma.y"]**2
    loglik = (jspecial.gammaln(y + size) - jspecial.gammaln(size)
              - jspecial.gammaln(y + 1.0) - size * jnp.log1p(mu / size)
              + jspecial.xlogy(y, mu) - y * (jnp.log(size) + jnp.log1p(mu / size)))
    valid_y = jnp.isfinite(y) & (y >= 0) & (y == jnp.floor(y))
    return jnp.where(valid_y, loglik, -jnp.inf)


def rmeas(X_, theta_, key, covars, t):
    """Exact negative-binomial law through its Gamma-Poisson mixture."""
    gamma_key, poisson_key = jax.random.split(key)
    mu = jnp.asarray(X_["N1"])
    size = 1.0 / theta_["sigma.y"]**2
    rate = jax.random.gamma(gamma_key, size, dtype=mu.dtype) * mu / size
    return {"y": jax.random.poisson(poisson_key, rate).astype(mu.dtype)}


def blowflies(*, params=None, n_observations=None):
    """Construct the full canonical dataset or its prefix starting at day 16.

    The prefix restriction ensures every observation interval is exactly two
    days; nstep=2 therefore means daily transitions with the same 15-value lag
    buffer. t0 and the initial state are never reestimated or moved forward.
    JAX x64 must be enabled by the caller for the validated reference precision.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before constructing the validated blowflies1 reference.")
    times, observed = observations()
    if n_observations is not None:
        if (isinstance(n_observations, bool) or not isinstance(n_observations, Integral)
                or n_observations < 1 or n_observations > len(times)):
            raise ValueError("n_observations must be an integer from 1 through 192.")
        times, observed = times[:n_observations], observed[:n_observations]
    params = Parameters() if params is None else params
    try:
        theta = params.as_r_dict()
    except AttributeError as exc:
        raise TypeError("params must supply Parameters.as_r_dict()") from exc
    if set(theta) != set(PARAM_NAMES):
        raise ValueError("Supply exactly the six canonical blowflies1 parameters.")
    values = np.asarray([theta[name] for name in PARAM_NAMES], dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Parameters must be finite positive scalars.")
    # PyPOMP's optional CLL/ESS xarray output names this dimension "time".
    ys = pd.DataFrame({"y": observed}, index=pd.Index(times.astype(float), name="time"))
    obj = pp.Pomp(
        ys=ys, theta=pp.PompParameters(dict(zip(PARAM_NAMES, values, strict=True))),
        statenames=STATENAMES, t0=14.0, nstep=2,
        rinit=rinit, rproc=rproc, dmeas=dmeas, rmeas=rmeas,
        par_trans=ParTrans(to_est=to_est, from_est=from_est),
    )
    obj.blowflies_adapter_metadata = {
        "reference": "R pomp::blowflies1()", "source_commit": UPSTREAM_COMMIT,
        "data_source": "frozen_Nicholson_population_I",
        "observation_prefix_length": len(times), "t0": 14.0,
        "nstep": 2, "daily_dt": 1.0, "delay_days": 14,
        "samplers": "standard jax.random gamma/poisson/binomial",
        "pathwise_ad_supported": False,
        "transition_ad_guard": "custom_jvp raises NotImplementedError",
    }
    return obj
