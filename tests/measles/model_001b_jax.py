"""
He10 model without alpha or mu parameters, using standard JAX samplers instead of fast pypomp samplers.

rproc is @vectorized like pypomp's 001b, so the two differ only in their samplers.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
from pypomp.core.model_mechanics import vectorized
from pypomp.types import (
    CovarDict,
    InitialTimeFloat,
    ObservationDict,
    ParamDict,
    RNGKey,
    StateDict,
    StepSizeFloat,
    TimeFloat,
)

param_names = (
    "R0",  # 0
    "sigma",  # 1
    "gamma",  # 2
    "iota",  # 3
    "rho",  # 4
    "sigmaSE",  # 5
    "psi",  # 6
    "cohort",  # 7
    "amplitude",  # 8
    "S_0",  # 9
    "E_0",  # 10
    "I_0",  # 11
    "R_0",  # 12
)

statenames = ["S", "E", "I", "R", "W", "C"]
accumvars = ["W", "C"]


def euler_exits(key, n, r0, r1, dt):
    """Twin of pypomp's measles `euler_exits`, drawing with jax.random.binomial."""
    k0, k1 = jax.random.split(key)
    r_sum = r0 + r1
    scale = (1.0 - jnp.exp(-r_sum * dt)) / r_sum
    p0, p1 = r0 * scale, r1 * scale
    x0 = jax.random.binomial(k0, n, jnp.clip(p0, 0.0, 1.0))
    p_rem = 1.0 - p0
    p1_cond = p1 / jnp.where(p_rem > 0.0, p_rem, 1.0)
    x1 = jax.random.binomial(k1, n - x0, jnp.clip(p1_cond, 0.0, 1.0))
    return x0, x1


def rinit(theta_: ParamDict, key: RNGKey, covars: CovarDict, t0: InitialTimeFloat):
    S_0 = theta_["S_0"]
    E_0 = theta_["E_0"]
    I_0 = theta_["I_0"]
    R_0 = theta_["R_0"]

    m = covars["pop"] / (S_0 + E_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    E = jnp.round(m * E_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    W = 0
    C = 0
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


@vectorized
def rproc(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
    dt: StepSizeFloat,
):
    S, E, I, W, C = X_["S"], X_["E"], X_["I"], X_["W"], X_["C"]
    J = jnp.asarray(S).shape[0]
    R0 = theta_["R0"]
    sigma = theta_["sigma"]
    gamma = theta_["gamma"]
    iota = theta_["iota"]
    sigmaSE = theta_["sigmaSE"]
    cohort = theta_["cohort"]
    amplitude = theta_["amplitude"]
    pop = covars["pop"]
    birthrate = covars["birthrate"]
    mu = 0.02

    t_mod = t - jnp.floor(t)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    br = jnp.where(
        is_cohort_time,
        cohort * birthrate / dt + (1 - cohort) * birthrate,
        (1 - cohort) * birthrate,
    )

    # term-time seasonality
    t_days = t_mod * 365.25
    in_term_time = (
        ((t_days >= 7) & (t_days <= 100))
        | ((t_days >= 115) & (t_days <= 199))
        | ((t_days >= 252) & (t_days <= 300))
        | ((t_days >= 308) & (t_days <= 356))
    )
    seas = jnp.where(in_term_time, 1.0 + amplitude * 0.2411 / 0.7589, 1 - amplitude)

    # transmission rate
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # expected force of infection
    foi = beta * (I + iota) / pop

    k_dw, k_births, k_S, k_E, k_I = jax.random.split(key, 5)

    # white noise (extrademographic stochasticity)
    dw_shape = jnp.broadcast_to(dt / sigmaSE**2, (J,))
    dw = jax.random.gamma(k_dw, dw_shape) * sigmaSE**2

    # Poisson births
    births = jax.random.poisson(k_births, jnp.broadcast_to(br * dt, (J,)))
    births = births.astype(S.dtype)

    # transitions between classes
    trans_S0, trans_S1 = euler_exits(k_S, S, foi * dw / dt, mu, dt)
    trans_E0, trans_E1 = euler_exits(k_E, E, sigma, mu, dt)
    trans_I0, trans_I1 = euler_exits(k_I, I, gamma, mu, dt)

    S = S + births - trans_S0 - trans_S1
    E = E + trans_S0 - trans_E0 - trans_E1
    I = I + trans_E0 - trans_I0 - trans_I1
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + trans_I0
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


def dmeas(
    Y_: ObservationDict,
    X_: StateDict,
    theta_: ParamDict,
    covars: CovarDict,
    t: TimeFloat,
):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    tol = 1.0e-18

    y = Y_["cases"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    sqrt_v_tol = jnp.sqrt(v) + tol

    upper_cdf = jax.scipy.stats.norm.cdf(y + 0.5, m, sqrt_v_tol)
    lower_cdf = jax.scipy.stats.norm.cdf(y - 0.5, m, sqrt_v_tol)

    lik = (
        jnp.where(
            y > tol,
            upper_cdf - lower_cdf,
            upper_cdf,
        )
        + tol
    )

    lik = jnp.where(C < 0, 0.0, lik)
    lik = jnp.where(jnp.isnan(y), 1.0, lik)
    return jnp.log(lik)


def rmeas(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return {"cases": jnp.where(cases > 0.0, jnp.round(cases), 0.0)}


def to_est(theta: ParamDict) -> ParamDict:
    SEIR_0 = jnp.array([theta["S_0"], theta["E_0"], theta["I_0"], theta["R_0"]])
    S_0, E_0, I_0, R_0 = jnp.log(SEIR_0 / jnp.sum(SEIR_0))
    return {
        "R0": jnp.log(theta["R0"]),
        "sigma": jnp.log(theta["sigma"]),
        "gamma": jnp.log(theta["gamma"]),
        "iota": jnp.log(theta["iota"]),
        "sigmaSE": jnp.log(theta["sigmaSE"]),
        "psi": jnp.log(theta["psi"]),
        "cohort": jspecial.logit(theta["cohort"]),
        "amplitude": jspecial.logit(theta["amplitude"]),
        "rho": jspecial.logit(theta["rho"]),
        "S_0": S_0,
        "E_0": E_0,
        "I_0": I_0,
        "R_0": R_0,
    }


def from_est(theta: ParamDict) -> ParamDict:
    SEIR_0 = jnp.exp(
        jnp.array([theta["S_0"], theta["E_0"], theta["I_0"], theta["R_0"]])
    )
    S_0, E_0, I_0, R_0 = SEIR_0 / jnp.sum(SEIR_0)
    return {
        "R0": jnp.exp(theta["R0"]),
        "sigma": jnp.exp(theta["sigma"]),
        "gamma": jnp.exp(theta["gamma"]),
        "iota": jnp.exp(theta["iota"]),
        "sigmaSE": jnp.exp(theta["sigmaSE"]),
        "psi": jnp.exp(theta["psi"]),
        "cohort": jspecial.expit(theta["cohort"]),
        "amplitude": jspecial.expit(theta["amplitude"]),
        "rho": jspecial.expit(theta["rho"]),
        "S_0": S_0,
        "E_0": E_0,
        "I_0": I_0,
        "R_0": R_0,
    }
