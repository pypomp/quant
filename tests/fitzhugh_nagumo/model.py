"""Ditlevsen--Samson (2019) FitzHugh--Nagumo model, without measurement noise.

Published equations (22), (30)--(31), (35); see also arXiv:1707.04235v2.
The SDE and observation map are the reference model.
The named Taylor-1.5 simulator is a discretization, NOT an exact SDE sampler.
No author initial state, random seed, or original data is assumed here.
"""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypomp as pp


class FHNParameters(NamedTuple):
    epsilon: float = 0.1
    gamma: float = 1.5
    alpha: float = 0.8  # FHN drift parameter, not the DMOP discount factor.
    sigma: float = 0.3
    s: float = 0.0


REFERENCE_PARAMETERS = FHNParameters()
SIMULATION_DT = 0.002
OBSERVATION_DT = 0.02
N_OBSERVATIONS = 1000


class Trajectory(NamedTuple):
    """Observation-time states; the supplied initial state is not an observation."""

    times: jax.Array
    states: jax.Array
    observations: jax.Array


def validate_parameters(params: FHNParameters) -> None:
    """Host-side checks; mathematical kernels remain compatible with JAX AD."""
    values = np.asarray(params, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Supply five finite FHN parameters.")
    if params.epsilon <= 0 or params.sigma < 0:
        raise ValueError("epsilon must be positive and sigma nonnegative.")


def drift(state, params: FHNParameters = REFERENCE_PARAMETERS):
    """SDE drift from published equation (22), in state order (V, U)."""
    v, u = jnp.asarray(state)
    return jnp.stack(
        (
            (v - v**3 - u + params.s) / params.epsilon,
            params.gamma * v - u + params.alpha,
        )
    )


def diffusion(params: FHNParameters = REFERENCE_PARAMETERS):
    """One Brownian driver; noise acts directly on U only."""
    return jnp.stack(
        (jnp.zeros_like(jnp.asarray(params.sigma)), jnp.asarray(params.sigma))
    )


def drift_jacobian(state, params: FHNParameters = REFERENCE_PARAMETERS):
    v, _ = jnp.asarray(state)
    return jnp.array(
        [[(1 - 3 * v**2) / params.epsilon, -1 / params.epsilon], [params.gamma, -1.0]]
    )


def taylor15_mean(state, dt, params: FHNParameters = REFERENCE_PARAMETERS):
    """Mean of the explicit strong-1.5 step, not the exact SDE mean.

    Both second derivatives with respect to U are zero, so L0 b = (Db)b.
    Derive from equations (22),(30),(31): the printed expansion in equation
    (23) has an inconsistent alpha sign in its first component.
    """
    state = jnp.asarray(state)
    b = drift(state, params)
    return state + dt * b + 0.5 * dt**2 * (drift_jacobian(state, params) @ b)


def taylor15_covariance(dt, params: FHNParameters = REFERENCE_PARAMETERS):
    """Full covariance of the specified step, published equation (35)."""
    eps, sig = params.epsilon, params.sigma
    cross = (-(dt**2) / 2 + dt**3 / 3) / eps
    return sig**2 * jnp.array(
        [[dt**3 / (3 * eps**2), cross], [cross, dt - dt**2 + dt**3 / 3]]
    )


def brownian_integrals(normals, dt):
    """Return eta and xi with Var=(dt,dt^3/3), Cov=dt^2/2."""
    z1, z2 = jnp.asarray(normals)
    eta = jnp.sqrt(dt) * z1
    xi = dt**1.5 * (z1 / 2 + z2 / jnp.sqrt(12.0))
    return eta, xi


def step_from_normals(state, normals, dt, params: FHNParameters = REFERENCE_PARAMETERS):
    """Fixed-innovation strong-1.5 step; no clipping, taming, or extra noise."""
    eta, xi = brownian_integrals(normals, dt)
    noise = jnp.stack((-params.sigma * xi / params.epsilon, params.sigma * (eta - xi)))
    return taylor15_mean(state, dt, params) + noise


def observe(state):
    """The original observation map Y=V, with zero measurement error."""
    return jnp.asarray(state)[..., 0]


def simulate_taylor15(
    key,
    *,
    initial_state,
    params: FHNParameters = REFERENCE_PARAMETERS,
    n_observations: int = N_OBSERVATIONS,
    simulation_dt: float = SIMULATION_DT,
    observation_dt: float = OBSERVATION_DT,
    t0: float = 0.0,
) -> Trajectory:
    """Generate NEW data from a specified numerical approximation of the SDE.

    The caller must provide key and initial_state. No claim is made to reproduce
    the authors' original trajectories. There are n_observations strictly after
    t0; the supplied state is the condition at t0. No burn-in is silently added.
    This convenience wrapper validates inputs on the host; differentiate
    step_from_normals or a caller's lax.scan when taking simulator derivatives.
    """
    validate_parameters(params)
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (2,) or not np.isfinite(initial).all():
        raise ValueError("initial_state must contain finite (V0, U0).")
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 1:
        raise ValueError("n_observations must be a positive integer.")
    if (
        not np.isfinite([simulation_dt, observation_dt, t0]).all()
        or simulation_dt <= 0
        or observation_dt <= 0
    ):
        raise ValueError("Time values must be finite and both intervals positive.")
    ratio = observation_dt / simulation_dt
    nstep = round(ratio)
    if nstep < 1 or not np.isclose(ratio, nstep, rtol=0, atol=1e-10):
        raise ValueError("observation_dt must be an integer multiple of simulation_dt.")
    initial = jnp.asarray(initial)
    normals = jax.random.normal(key, (n_observations, nstep, 2), dtype=initial.dtype)

    def advance_interval(state, interval_normals):
        def advance_substep(x, z):
            return step_from_normals(x, z, simulation_dt, params), None

        next_state, _ = jax.lax.scan(advance_substep, state, interval_normals)
        return next_state, next_state

    _, states = jax.lax.scan(advance_interval, initial, normals)
    if not np.isfinite(np.asarray(states)).all():
        raise FloatingPointError(
            "Taylor-1.5 simulation diverged; no state clipping applied."
        )
    times = t0 + observation_dt * jnp.arange(1, n_observations + 1)
    return Trajectory(times, states, observe(states))


STATENAMES = ["V", "U"]
PARAMETER_NAMES = ("epsilon", "gamma", "alpha", "sigma", "s")


def rinit(theta_, key, covars, t0):
    """Use the explicitly supplied deterministic initial condition."""
    return {"V": theta_["V0"], "U": theta_["U0"]}


def rproc(X_, theta_, key, covars, t, dt):
    """Apply one reference strong-order-1.5 step, without clipping."""
    state = jnp.array([X_["V"], X_["U"]])
    normals = jax.random.normal(key, shape=(2,), dtype=state.dtype)
    params = FHNParameters(*(theta_[name] for name in PARAMETER_NAMES))
    updated = step_from_normals(state, normals, dt, params)
    return {"V": updated[0], "U": updated[1]}


def rmeas(X_, theta_, key, covars, t):
    """Observe V exactly; no additional observation noise is introduced."""
    return {"V": X_["V"]}


def _numeric_array(value, name):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(array, dtype=float)


def make_pomp(
    *,
    initial_state,
    observations: pd.DataFrame | None = None,
    times=None,
    params=None,
    t0: float = 0.0,
    nstep: int = 10,
) -> pp.Pomp:
    """Build a simulation-only Pomp with the reference strong-1.5 transition.

    ``initial_state`` is the caller's ``(V0, U0)`` at ``t0``.  Supply exactly
    one of a finite, single-column ``V`` observation DataFrame or a sequence
    of simulation times.  In the latter case ``ys`` contains NaN placeholders
    only; they are not an observed or reconstructed paper dataset.

    Every time must be strictly after ``t0`` and times must increase.  A fixed
    ``nstep`` is used within each interval: at the reference observation
    spacing 0.02, the default 10 steps have size 0.002.  Different intervals
    remain explicit and use their own interval length divided by ``nstep``.
    No Euler step or measurement-noise approximation is substituted.

    ``dmeas`` remains None even when observations are supplied: the exact
    observation of a continuous state is not an ordinary measurement density
    for PyPOMP's bootstrap particle filter.
    """
    if (observations is None) == (times is None):
        raise ValueError("Supply exactly one of observations or simulation times")
    if isinstance(nstep, bool) or not isinstance(nstep, Integral) or nstep < 1:
        raise ValueError("nstep must be a positive integer")
    if isinstance(t0, bool) or not np.isscalar(t0):
        raise ValueError("t0 must be a finite real scalar")
    t0_array = _numeric_array(t0, "t0")
    if not np.isfinite(t0_array):
        raise ValueError("t0 must be a finite real scalar")
    initial = _numeric_array(initial_state, "initial_state")
    if initial.shape != (2,) or not np.isfinite(initial).all():
        raise ValueError("initial_state must contain finite (V0, U0)")

    params = FHNParameters() if params is None else params
    try:
        values = _numeric_array(
            [getattr(params, name) for name in PARAMETER_NAMES], "params"
        )
    except AttributeError as exc:
        raise TypeError(
            "params must provide epsilon, gamma, alpha, sigma, and s"
        ) from exc
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("All five process parameters must be finite scalars")
    if values[0] <= 0 or values[3] < 0:
        raise ValueError("epsilon must be positive and sigma must be nonnegative")

    if observations is not None:
        if not isinstance(observations, pd.DataFrame):
            raise TypeError("observations must be a pandas DataFrame")
        if list(observations.columns) != ["V"]:
            raise ValueError("observations must contain exactly one column named V")
        observation_values = _numeric_array(
            observations["V"].to_numpy(), "observations"
        )
        if not np.isfinite(observation_values).all():
            raise ValueError(
                "Supplied observations must be finite; NaNs are simulation placeholders only"
            )
        time_values = _numeric_array(observations.index.to_numpy(), "observation times")
        ys = observations.copy(deep=True)
        data_source = "caller_supplied_observations"
    else:
        time_values = _numeric_array(times, "simulation times")
        ys = None
        data_source = "simulation_grid_only_no_paper_dataset"

    if (
        time_values.ndim != 1
        or len(time_values) == 0
        or not np.isfinite(time_values).all()
        or np.any(np.diff(time_values) <= 0)
        or time_values[0] <= float(t0_array)
    ):
        raise ValueError(
            "Times must be a nonempty, finite, strictly increasing grid after t0"
        )
    if ys is None:
        ys = pd.DataFrame({"V": np.full(len(time_values), np.nan)}, index=time_values)
    ys.index = pd.Index(time_values, name="time")
    theta = dict(zip(PARAMETER_NAMES, values.tolist(), strict=True))
    theta.update(V0=float(initial[0]), U0=float(initial[1]))
    obj = pp.Pomp(
        ys=ys,
        theta=pp.PompParameters(theta),
        statenames=STATENAMES,
        t0=float(t0_array),
        rinit=rinit,
        rproc=rproc,
        rmeas=rmeas,
        dmeas=None,
        nstep=int(nstep),
    )
    obj.fhn_adapter_metadata = {
        "simulation_only": True,
        "data_source": data_source,
        "paper_dataset_included": False,
        "observation": "V exactly, without measurement noise",
        "transition": "reference strong-order-1.5 scheme",
        "initial_state_source": "caller",
        "nstep": int(nstep),
    }
    return obj
