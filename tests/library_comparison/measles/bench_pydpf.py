"""pydpf (PyTorch): He10 measles as pydpf Modules in a plain ParticleFilter, batched over filters.

Every (theta, rep[, unit]) filter is one element of pydpf's batch dimension; the
modules hold per-batch-element parameters (B, 13) and covariates (B, K+1), and the
dynamic model reads pydpf's time index t to pick the interval's Euler steps. The
prior module draws the initial state at t0 (pypomp's rinit) and pushes it through
the first interval, so the filter targets p(y_1..y_T) like pypomp.
Samplers: torch's exact torch._standard_gamma (what torch.distributions.Gamma uses),
torch.poisson and torch.binomial, all on one seeded torch.Generator. float32.
Resampling: pydpf.SystematicResampler at every observation; loglik from
pydpf.LogLikelihoodFactors summed over time.

    python bench_pydpf.py --J 1000 --nthetas 3 --nreps 10 --device cpu [--panel] [--procs 4]

--procs sets torch's intra-op CPU threads (default 1). --chunk B runs the filters in
sequential batches of B (memory bound on large GPU workloads).
"""

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he10 as H  # noqa: E402


def add_args(p):
    p.add_argument("--chunk", type=int, default=0, help="filters per torch batch (0 = all at once)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="float64 is a precision cross-check, not the benchmark setting")


args = H.parse_args("pydpf", add_args)

import numpy as np  # noqa: E402
import pydpf  # noqa: E402
import torch  # noqa: E402

if args.device == "gpu":
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, args.procs))
FT = torch.float64 if args.dtype == "float64" else torch.float32
I = H.P
MU, TOL = H.MU, H.TOL
inp = H.Inputs(args.panel, args.T)
n, R, U, J = args.nthetas, args.nreps, inp.U, args.J
gen = torch.Generator(device=dev)
DT = inp.dt.tolist()
TERM = inp.term.tolist()
COH = inp.cohort_step.tolist()
STEP0 = inp.step0.tolist()
NSTEP = inp.nstep.tolist()


class Batch:
    """Per-batch-element parameters, covariates and observations for one group of filters."""

    def __init__(self, tasks):
        th = np.stack([inp.thetas(n)[i, u] for i, r, u in tasks])
        us = [u for i, r, u in tasks]
        self.B = len(tasks)
        self.th = {k: torch.tensor(th[:, j], dtype=FT, device=dev)[:, None] for k, j in I.items()}
        self.pop = torch.tensor(inp.pop[us], dtype=FT, device=dev)  # (B, K+1)
        self.br = torch.tensor(inp.birthrate[us], dtype=FT, device=dev)
        self.y = torch.tensor(inp.ys[us].T, dtype=FT, device=dev)[:, :, None].contiguous()  # (T, B, 1)


def euler_interval(b, x, i):
    """Advance x (B, J, 6) through observation interval i; W and C reset to 0 first."""
    th = b.th
    S, E, Ii = x[..., 0], x[..., 1], x[..., 2]
    W = torch.zeros_like(S)
    C = torch.zeros_like(S)
    s2 = th["sigmaSE"] ** 2
    shape = S.shape
    for k in range(STEP0[i], STEP0[i] + NSTEP[i]):
        dt = DT[k]
        pop = b.pop[:, k:k + 1]
        bk = b.br[:, k:k + 1]
        coh = th["cohort"]
        br = coh * bk / dt + (1 - coh) * bk if COH[k] else (1 - coh) * bk
        seas = 1.0 + th["amplitude"] * 0.2411 / 0.7589 if TERM[k] else 1.0 - th["amplitude"]
        beta = th["R0"] * seas * (1.0 - torch.exp(-(th["gamma"] + MU) * dt)) / dt
        foi = beta * (Ii + th["iota"]) / pop
        dw = torch._standard_gamma((dt / s2).expand(shape).contiguous(), generator=gen) * s2
        births = torch.poisson((br * dt).expand(shape).contiguous(), generator=gen)
        r0 = torch.stack([foi * dw / dt, th["sigma"].expand(shape), th["gamma"].expand(shape)])
        rs = r0 + MU
        scale = (1.0 - torch.exp(-rs * dt)) / rs
        p0 = r0 * scale
        p1 = MU * scale
        cnt = torch.stack([S, E, Ii])
        x0 = torch.binomial(cnt, p0.clamp(0.0, 1.0), generator=gen)
        prem = 1.0 - p0
        x1 = torch.binomial(cnt - x0, (p1 / torch.where(prem > 0.0, prem, torch.ones_like(prem))).clamp(0.0, 1.0),
                            generator=gen)
        S = S + births - x0[0] - x1[0]
        E = E + x0[0] - x0[1] - x1[1]
        Ii = Ii + x0[1] - x0[2] - x1[2]
        W = W + (dw - dt) / th["sigmaSE"]
        C = C + x0[2]
    Rr = b.pop[:, STEP0[i] + NSTEP[i] - 1:STEP0[i] + NSTEP[i]] - S - E - Ii
    return torch.stack([S, E, Ii, Rr.expand(shape), W, C], dim=-1)


class Prior(pydpf.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b

    def sample(self, batch_size, n_particles, **kw):
        th = self.b.th
        m = self.b.pop[:, 0:1] / (th["S_0"] + th["E_0"] + th["I_0"] + th["R_0"])
        cols = [torch.round(m * th[k]).expand(batch_size, n_particles) for k in ("S_0", "E_0", "I_0", "R_0")]
        z = torch.zeros((batch_size, n_particles), dtype=FT, device=dev)
        x0 = torch.stack(cols + [z, z], dim=-1)
        return euler_interval(self.b, x0, 0)  # state at the first observation


class Dyn(pydpf.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b

    def sample(self, prev_state, t, **kw):
        return euler_interval(self.b, prev_state, int(t))


class Obs(pydpf.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b

    def fitness(self, state, observation, **kw):
        th = self.b.th
        rho, psi = th["rho"], th["psi"]
        C = state[..., 5]
        y = observation[:, 0:1]
        m = rho * C
        v = m * (1.0 - rho + psi ** 2 * m)
        sd = torch.sqrt(v) + TOL
        upper = torch.special.ndtr((y + 0.5 - m) / sd)
        lower = torch.special.ndtr((y - 0.5 - m) / sd)
        lik = torch.where(y > TOL, upper - lower, upper) + TOL
        lik = torch.where(C < 0, torch.zeros_like(lik), lik)
        lik = torch.where(torch.isnan(y), torch.ones_like(lik), lik)
        return torch.log(lik)


tasks = H.task_list(n, R, U)
N = len(tasks)
CH = args.chunk if 0 < args.chunk < N else N
groups = [tasks[s:s + CH] for s in range(0, N, CH)]
batches = [Batch(g) for g in groups]
agg = pydpf.LogLikelihoodFactors()
pfs = [
    pydpf.ParticleFilter(
        resampler=pydpf.SystematicResampler(generator=gen),
        SSM=pydpf.FilteringModel(dynamic_model=Dyn(b), observation_model=Obs(b), prior_model=Prior(b)),
    ).to(dev)
    for b in batches
]


def run():
    gen.manual_seed(args.seed)
    out = []
    with torch.inference_mode():
        for pf, b in zip(pfs, batches):
            fac = pf(n_particles=J, time_extent=inp.T - 1, aggregation_function=agg, observation=b.y)
            if fac.shape[0] != inp.T:
                raise RuntimeError(f"pydpf stopped early at step {fac.shape[0]} (all weights zero)")
            out.append(fac.sum(0).reshape(-1).double().cpu().numpy())  # .cpu() syncs the device
    return np.concatenate(out).reshape(n, R, U)


ll, tm = H.timed_calls(run, args.calls)
cold, warm = tm.pop("cold_s"), tm.pop("warm_s", None)

H.record(
    args, "pydpf", inp, ll, cold, warm, **tm,
    samplers="torch._standard_gamma / torch.poisson / torch.binomial (exact), one torch.Generator",
    dtype=args.dtype,
    versions=dict(H.versions("pydpf", "torch"), torch_threads=torch.get_num_threads(),
                  cuda=torch.version.cuda if args.device == "gpu" else None),
    resampling="systematic every observation (pydpf.SystematicResampler)",
    chunk=CH,
)
