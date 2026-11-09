"""PyPOMP process and measurement models for the nav03 robot trajectories."""

from math import prod

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from pypomp import ParTrans
from pypomp.core.model_mechanics import _time_interp
from pypomp.functional import mop
from pypomp.functional.structs import PompStruct

# The fitted vector stores the PyDPF weights in row-major order.
TENSOR_SHAPES = (
    ("state_encoder.0.weight", (16, 4)),
    ("state_encoder.0.bias", (16,)),
    ("state_encoder.2.weight", (32, 16)),
    ("state_encoder.2.bias", (32,)),
    ("state_encoder.4.weight", (64, 32)),
    ("state_encoder.4.bias", (64,)),
    ("state_encoder.7.weight", (64, 64)),
    ("state_encoder.7.bias", (64,)),
    ("flow_model.flows.0.t1.network.0.weight", (128, 128)),
    ("flow_model.flows.0.t1.network.0.bias", (128,)),
    ("flow_model.flows.0.t1.network.2.weight", (128, 128)),
    ("flow_model.flows.0.t1.network.2.bias", (128,)),
    ("flow_model.flows.0.t1.network.4.weight", (64, 128)),
    ("flow_model.flows.0.t1.network.4.bias", (64,)),
    ("flow_model.flows.0.t2.network.0.weight", (128, 128)),
    ("flow_model.flows.0.t2.network.0.bias", (128,)),
    ("flow_model.flows.0.t2.network.2.weight", (128, 128)),
    ("flow_model.flows.0.t2.network.2.bias", (128,)),
    ("flow_model.flows.0.t2.network.4.weight", (64, 128)),
    ("flow_model.flows.0.t2.network.4.bias", (64,)),
    ("flow_model.flows.1.t1.network.0.weight", (128, 128)),
    ("flow_model.flows.1.t1.network.0.bias", (128,)),
    ("flow_model.flows.1.t1.network.2.weight", (128, 128)),
    ("flow_model.flows.1.t1.network.2.bias", (128,)),
    ("flow_model.flows.1.t1.network.4.weight", (64, 128)),
    ("flow_model.flows.1.t1.network.4.bias", (64,)),
    ("flow_model.flows.1.t2.network.0.weight", (128, 128)),
    ("flow_model.flows.1.t2.network.0.bias", (128,)),
    ("flow_model.flows.1.t2.network.2.weight", (128, 128)),
    ("flow_model.flows.1.t2.network.2.bias", (128,)),
    ("flow_model.flows.1.t2.network.4.weight", (64, 128)),
    ("flow_model.flows.1.t2.network.4.bias", (64,)),
)
PARAMETERS = sum(prod(shape) for _, shape in TENSOR_SHAPES)
PARAM_NAMES = tuple(f"p{i:06}" for i in range(PARAMETERS))
IDENTITY = ParTrans()


def unpack(theta):
    tensors = {}
    offset = 0
    for name, shape in TENSOR_SHAPES:
        end = offset + prod(shape)
        tensors[name] = theta[offset:end].reshape(shape)
        offset = end
    return tensors


def linear(x, p, name):
    return x @ p[name + ".weight"].T + p[name + ".bias"]


def condition(state, p):
    x = jnp.stack(
        (state[..., 0], state[..., 1], jnp.cos(state[..., 2]), jnp.sin(state[..., 2])),
        axis=-1,
    )
    for i in (0, 2, 4):
        x = jax.nn.relu(linear(x, p, f"state_encoder.{i}"))
    return linear(x, p, "state_encoder.7")


def coupling(x, c, p, flow, part):
    x = jnp.concatenate((x, c), axis=-1)
    prefix = f"flow_model.flows.{flow}.{part}.network"
    x = jnp.tanh(linear(x, p, prefix + ".0"))
    x = jnp.tanh(linear(x, p, prefix + ".2"))
    return linear(x, p, prefix + ".4")


def transform(z, state, theta, inverse=False):
    p = unpack(theta)
    c = condition(state, p)
    z = jnp.broadcast_to(z, c.shape[:-1] + (128,))
    lower, upper = z[..., :64], z[..., 64:]
    for flow in (1, 0) if inverse else (0, 1):
        if inverse:
            lower = lower - coupling(upper, c, p, flow, "t2")
            upper = upper - coupling(lower, c, p, flow, "t1")
        else:
            upper = upper + coupling(lower, c, p, flow, "t1")
            lower = lower + coupling(upper, c, p, flow, "t2")
    return jnp.concatenate((lower, upper), axis=-1)


def log_density(z, state, theta):
    f = transform(z, state, theta)
    return -0.5 * jnp.sum(
        f * f + jnp.log(jnp.asarray(2 * jnp.pi, dtype=f.dtype)), axis=-1
    )


def rinit(theta, keys, covars, t0, should_trans):
    del t0, should_trans
    state = jnp.broadcast_to(covars[3:6], (keys.shape[0], 3)).astype(theta.dtype)
    checkify.check(jnp.all(jnp.isfinite(state)), "nonfinite initial state")
    return state


def process_step(state, theta, key, covars, t, dt):
    del t, dt
    x, y, phi = state["x"], state["y"], state["heading"]
    u = covars[:3]
    mean = jnp.stack(
        (
            x + jnp.cos(phi) * u[0] - jnp.sin(phi) * u[1],
            y + jnp.sin(phi) * u[0] + jnp.cos(phi) * u[1],
            phi + u[2],
        ),
        axis=-1,
    )
    noise = jax.random.normal(key, mean.shape, dtype=theta.dtype)
    out = mean + noise * jnp.asarray([0.03, 0.03, 0.1], dtype=theta.dtype)
    checkify.check(jnp.all(jnp.isfinite(out)), "nonfinite propagated state")
    return {"x": out[:, 0], "y": out[:, 1], "heading": out[:, 2]}


# Native time interpolation supplies the pre-increment covariate convention.
RPROC = _time_interp(process_step, 1, ["x", "y", "heading"])


def dmeasure(y, state, theta, covars, t, should_trans):
    del covars, t, should_trans
    out = log_density(y, state, theta)
    checkify.check(jnp.all(jnp.isfinite(out)), "nonfinite measurement log density")
    return out


def unsupported(*args):
    raise NotImplementedError("IF2/perturbation mechanics are outside this adapter")


def make_struct(z, u, x0):
    # z contains scored observations only, u the corresponding incoming actions.
    T = z.shape[0]
    controls = jnp.concatenate((u, jnp.zeros((1, 3), dtype=u.dtype)))
    covars = jnp.concatenate((controls, jnp.broadcast_to(x0, (T + 1, 3))), axis=1)
    return PompStruct(
        ys=z,
        dt_array_extended=jnp.ones(T + 1, dtype=z.dtype),
        nstep_array=jnp.ones(T, dtype=jnp.int32),
        t0=0.0,
        times=jnp.arange(1, T + 1, dtype=z.dtype),
        covars_extended=covars,
        accumvars=None,
        rinit_pf=rinit,
        rproc_pf=RPROC,
        dmeas_pf=dmeasure,
        rinit_per=unsupported,
        rproc_per=unsupported,
        dmeas_per=None,
        rmeas_pf=None,
        dprior_pf=None,
        par_trans=IDENTITY,
        param_names=PARAM_NAMES,
        y_names=tuple(f"z{i}" for i in range(128)),
        statenames=("x", "y", "heading"),
    )


def trajectory_nll(theta, z, u, x0, key, J, alpha):
    return mop(make_struct(z, u, x0), theta[None], J, alpha, key[None])[0]


def batch_nll(theta, z, u, x0, keys, J, alpha):
    values = jax.vmap(
        lambda zz, uu, xx, kk: trajectory_nll(theta, zz, uu, xx, kk, J, alpha)
    )(z, u, x0, keys)
    return jnp.mean(values) / z.shape[1]


def require_finite_inputs(theta, z, u, x0):
    for name, value in [
        ("parameters", theta),
        ("observations", z),
        ("controls", u),
        ("initial states", x0),
    ]:
        checkify.check(jnp.all(jnp.isfinite(value)), "nonfinite " + name)


def batch_value_and_gradient(theta, z, u, x0, keys, J, alpha):
    require_finite_inputs(theta, z, u, x0)

    # Carry checks as auxiliary outputs through autodiff. Checking only after
    # value_and_grad lost model checks in the recorded nested-remat path.
    def objective_with_errors(th):
        error, value = checkify.checkify(
            lambda t: batch_nll(t, z, u, x0, keys, J, alpha)
        )(th)
        return value, error

    (loss, inner_error), gradient = jax.value_and_grad(
        objective_with_errors, has_aux=True
    )(theta)
    checkify.check_error(inner_error)
    checkify.check(jnp.isfinite(loss), "nonfinite objective")
    checkify.check(jnp.all(jnp.isfinite(gradient)), "nonfinite gradient")
    return loss, gradient
