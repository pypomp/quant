"""Strang simulator for the s=0 FHN model in Samson et al.'s later code.

The existing Taylor-1.5 implementation is unchanged. This module implements
the same nonlinear/linear/nonlinear splitting as SMCABCFHN, using a stable
matrix exponential for the linear Gaussian moments. It is a numerical SDE
approximation, not an exact transition sampler for the full nonlinear SDE.
"""

from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np

from model import FHNParameters, REFERENCE_PARAMETERS, Trajectory


def validate_parameters(params=REFERENCE_PARAMETERS):
    """Restrict validation to the later author's s=0, oscillatory domain."""
    values = np.asarray(params, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Supply five finite FHN parameters.")
    if params.epsilon <= 0 or params.gamma <= params.epsilon / 4 or params.sigma < 0:
        raise ValueError("Require epsilon>0, gamma>epsilon/4 and sigma>=0.")
    if params.s != 0:
        raise ValueError("This author-reference Strang implementation requires s=0.")


def _valid(params, dt):
    return (jnp.all(jnp.isfinite(jnp.asarray(params))) & jnp.isfinite(dt)
            & (dt > 0) & (params.epsilon > 0) & (params.gamma > params.epsilon / 4)
            & (params.sigma >= 0) & (params.s == 0))


def _unit_linear_moments(dt, params):
    """Van Loan block exponential: C=integral exp(A t) Q exp(A.T t) dt."""
    dtype = jnp.result_type(dt, *params, 1.)
    epsilon = jnp.asarray(params.epsilon, dtype=dtype)
    a = jnp.array([[0., -1. / epsilon], [params.gamma, -1.]], dtype=dtype)
    q = jnp.diag(jnp.array([0., 1.], dtype=dtype))
    block = jnp.zeros((4, 4), dtype=dtype)
    block = block.at[:2, :2].set(a).at[:2, 2:].set(q).at[2:, 2:].set(-a.T)
    exponential = jlinalg.expm(dt * block)
    e = exponential[:2, :2]
    covariance = exponential[:2, 2:] @ e.T
    covariance = (covariance + covariance.T) / 2
    valid = _valid(params, dt)
    return jnp.where(valid, e, jnp.nan), jnp.where(valid, covariance, jnp.nan)


def linear_moments(dt, params=REFERENCE_PARAMETERS):
    """Return E_h and covariance of the linear substep, not of the full step."""
    e, unit_covariance = _unit_linear_moments(dt, params)
    return e, params.sigma**2 * unit_covariance


def linear_noise_factor(dt, params=REFERENCE_PARAMETERS):
    """Lower Cholesky factor; factoring unit covariance also handles sigma=0."""
    _, unit_covariance = _unit_linear_moments(dt, params)
    return params.sigma * jnp.linalg.cholesky(unit_covariance)


def nonlinear_flow(state, duration, params=REFERENCE_PARAMETERS):
    """Exact s=0 nonlinear subflow H_t, including the beta=alpha shift in U."""
    state = jnp.asarray(state)
    v, u = state[..., 0], state[..., 1]
    exponent = -2 * duration / jnp.asarray(params.epsilon)
    # expm1 preserves 1-exp(exponent) for very small steps.
    denominator = jnp.sqrt(jnp.exp(exponent) - v**2 * jnp.expm1(exponent))
    updated = jnp.stack((v / denominator, u + params.alpha * duration), axis=-1)
    return jnp.where(params.s == 0, updated, jnp.nan)


def _step_with_matrix(state, innovation, dt, params, transition):
    first = nonlinear_flow(state, dt / 2, params)
    middle = first @ transition.T + jnp.asarray(innovation)
    return nonlinear_flow(middle, dt / 2, params)


def step_from_innovation(state, innovation, dt, params=REFERENCE_PARAMETERS):
    """Use a supplied additive 2D Gaussian innovation for exact-method replay."""
    transition, _ = _unit_linear_moments(dt, params)
    return _step_with_matrix(state, innovation, dt, params, transition)


def step_from_normals(state, normals, dt, params=REFERENCE_PARAMETERS):
    """Use independent standard normals with this implementation's lower factor."""
    transition, unit_covariance = _unit_linear_moments(dt, params)
    factor = params.sigma * jnp.linalg.cholesky(unit_covariance)
    innovation = jnp.asarray(normals) @ factor.T
    return _step_with_matrix(state, innovation, dt, params, transition)


def path_from_innovations(initial_state, innovations, dt, params=REFERENCE_PARAMETERS):
    """Return N+1 states including the initial state, as in the author code."""
    initial_state, innovations = jnp.asarray(initial_state), jnp.asarray(innovations)
    if initial_state.shape != (2,) or innovations.ndim != 2 or innovations.shape[1] != 2:
        raise ValueError("Require initial_state shape (2,) and innovations shape (N,2).")
    transition, _ = _unit_linear_moments(dt, params)
    dtype = jnp.result_type(initial_state, innovations, transition.dtype, dt, *params, 1.)
    initial_state, innovations = initial_state.astype(dtype), innovations.astype(dtype)
    def advance(state, innovation):
        updated = _step_with_matrix(state, innovation, dt, params, transition)
        return updated, updated
    _, states = jax.lax.scan(advance, initial_state, innovations)
    return jnp.concatenate((initial_state[None, :], states), axis=0)


def simulate_strang(key, *, initial_state, n_observations,
                    params=REFERENCE_PARAMETERS, simulation_dt=.02,
                    observation_dt=.02, t0=0.):
    """New simulated observations strictly after t0; Y=V, with no added noise.

    Default steps are the later paper's ABC simulation setting, not a claim
    about generation of either paper's archived data. The caller provides
    initial state, key and observation count. Intermediate steps are retained
    only internally. No clipping or replacement of divergent states is used.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for this validated reference implementation.")
    validate_parameters(params)
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (2,) or not np.isfinite(initial).all():
        raise ValueError("initial_state must contain finite (V0,U0).")
    if isinstance(n_observations, bool) or not isinstance(n_observations, Integral) or n_observations < 1:
        raise ValueError("n_observations must be a positive integer.")
    if (not np.isfinite([simulation_dt, observation_dt, t0]).all()
            or simulation_dt <= 0 or observation_dt <= 0):
        raise ValueError("Require finite positive step sizes and finite t0.")
    ratio = observation_dt / simulation_dt
    nstep = int(round(ratio))
    if nstep < 1 or not np.isclose(ratio, nstep, rtol=0, atol=1e-10):
        raise ValueError("observation_dt must be an integer multiple of simulation_dt.")
    transition, unit_covariance = _unit_linear_moments(simulation_dt, params)
    factor = params.sigma * jnp.linalg.cholesky(unit_covariance)
    initial = jnp.asarray(initial)
    def advance_interval(carry, unused):
        state, key = carry
        def advance_substep(carry, unused):
            x, k = carry
            k, draw_key = jax.random.split(k)
            xi = factor @ jax.random.normal(draw_key, (2,), dtype=initial.dtype)
            return (_step_with_matrix(x, xi, simulation_dt, params, transition), k), None
        (state, key), _ = jax.lax.scan(advance_substep, (state, key), None, length=nstep)
        return (state, key), state
    _, states = jax.lax.scan(advance_interval, (initial, key), None, length=n_observations)
    if not np.isfinite(np.asarray(states)).all():
        raise FloatingPointError("Nonfinite Strang trajectory; no clipping applied.")
    times = t0 + observation_dt * jnp.arange(1, n_observations + 1)
    return Trajectory(times, states, states[:, 0])
