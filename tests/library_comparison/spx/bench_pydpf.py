"""pydpf (PyTorch): SPX as pydpf Modules; plain ParticleFilter, SystematicResampler (float32).

All nthetas*nreps filters run as pydpf's batch dimension B. Per-filter parameters go in
`series_metadata` (B, 6); the leverage covariate goes in `control` (T, B, 1), with
control[n] = covars_extended[n], the previous return pypomp's rproc reads on the step into
observation n. pydpf observes its initial state, so the prior draws x0 = (V_0, S0) and applies
one step, and every library targets p(y_1..y_T). Resampling happens at every observation.

On --device cpu, --procs sets torch's intra-op thread count (default 1).

usage: python bench_pydpf.py --J 1000 --nthetas 36 --nreps 24 --device gpu
"""

import spx_common as C

args = C.parse_args("pydpf")

import math  # noqa: E402

import numpy as np  # noqa: E402
import pydpf  # noqa: E402
import torch  # noqa: E402

inp = C.load_inputs(args.T, args.nthetas, args.theta_set)
T = len(inp["ys"])
S0 = float(inp["S0"])
VF = float(inp["V_floor"])

dev = torch.device("cuda" if args.device == "gpu" else "cpu")
if args.device == "cpu":
    # torch's default (one intra-op thread per core) oversubscribes a shared box badly:
    # 100x slower here under load. --procs sets the thread count instead.
    torch.set_num_threads(args.procs)
gen = torch.Generator(device=dev)
J, nth, nrep = args.J, args.nthetas, args.nreps
B = nth * nrep

f32 = dict(dtype=torch.float32, device=dev)
y = torch.tensor(inp["ys"], **f32).reshape(T, 1, 1).expand(T, B, 1).contiguous()
ctrl = torch.tensor(inp["y_prev"], **f32).reshape(T, 1, 1).expand(T, B, 1).contiguous()
meta = torch.tensor(np.repeat(inp["thetas"], nrep, axis=0), **f32)  # (B, 6)
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def move(x, y_prev, th):
    """x (B, K, 2) = (V, S); y_prev (B, 1); th (B, 6)."""
    V, S = x[..., 0], x[..., 1]
    mu, kappa, theta, xi, rho = (th[:, i : i + 1] for i in range(5))
    dZ = torch.randn(V.shape, generator=gen, **f32)
    dWs = (y_prev - mu + 0.5 * V) / torch.sqrt(V)
    dWv = rho * dWs + torch.sqrt(1 - rho**2) * dZ
    S = S + S * (mu + torch.sqrt(torch.clamp(V, min=VF)) * dWs)
    V = V + kappa * (theta - V) + xi * torch.sqrt(V) * dWv
    return torch.stack([torch.clamp(V, min=VF), S], dim=-1)


class Prior(pydpf.Module):
    def sample(self, batch_size, n_particles, control, series_metadata, **kw):
        V0 = series_metadata[:, 5:6].expand(batch_size, n_particles)
        x0 = torch.stack([V0, torch.full_like(V0, S0)], dim=-1)
        return move(x0, control, series_metadata)  # state at the first observation


class Dyn(pydpf.Module):
    def sample(self, prev_state, control, series_metadata, **kw):
        return move(prev_state, control, series_metadata)


class Obs(pydpf.Module):
    def fitness(self, state, observation, series_metadata, **kw):
        V = state[..., 0]
        sd = torch.sqrt(V)
        d = (observation[:, 0:1] - (series_metadata[:, 0:1] - 0.5 * V)) / sd
        return -0.5 * d * d - torch.log(sd) - HALF_LOG_2PI


pf = pydpf.ParticleFilter(
    resampler=pydpf.SystematicResampler(generator=gen),
    SSM=pydpf.FilteringModel(dynamic_model=Dyn(), observation_model=Obs(), prior_model=Prior()),
).to(dev)
agg = pydpf.LogLikelihoodFactors()


def run():
    gen.manual_seed(args.seed)
    with torch.inference_mode():
        out = pf(n_particles=J, time_extent=T - 1, aggregation_function=agg,
                 observation=y, control=ctrl, series_metadata=meta)
        ll = out.to(torch.float64).sum(0).cpu().numpy()  # .cpu() syncs the device
    return ll.reshape(nth, nrep)


cold, ll = C.timed(run)
warm, ll2 = C.timed(run)
C.record(
    args, "pydpf", T, cold, warm, ll,
    C.versions("pydpf", "torch"),
    variant="ParticleFilter/batch", dtype="float32", resampler="systematic",
    torch_threads=torch.get_num_threads(), warm_reproduced=bool(np.array_equal(ll, ll2, equal_nan=True)),
)
