"""pydpf (PyTorch): dacca as pydpf Modules, plain ParticleFilter, systematic resampling every step.

float32 torch, one torch.Generator for the dynamics and the resampler. All nthetas x nreps
filters share one batch dimension (B = nthetas * nreps, or --chunk filters at a time); each
batch element's parameter vector travels as pydpf's per-series `series_metadata` (B, P). The
dynamic model reads the covariate block of interval t from the `t` pydpf passes it.
pydpf weights the initial state, so the prior draws x(t0) and pushes it through the first
interval: the filter targets p(y_1..y_T), like pypomp.

usage: .venv-torch/bin/python bench_pydpf.py --J 5000 --nthetas 36 --nreps 36 --device gpu
"""

import os
import time

import dacca_common as DC


def _extra(ap):
    ap.add_argument("--chunk", type=int, default=0, help="filters per batch (0 = all at once)")


args = DC.parse_args("pydpf", extra=_extra)
if args.device == "cpu":
    DC.set_cpu_env(threads=args.procs)

import math  # noqa: E402

import numpy as np  # noqa: E402
import pydpf  # noqa: E402
import torch  # noqa: E402

dev = torch.device("cuda" if args.device == "gpu" else "cpu")
if args.device == "cpu":
    torch.set_num_threads(args.procs)
inp = DC.load_inputs(args.T, args.theta_set)
T, K = inp["T"], inp["nstep_const"]
P = {k: i for i, k in enumerate(inp["paramnames"])}
IVP = [P[k] for k in ("S_0", "I_0", "Y_0", "R1_0", "R2_0", "R3_0")]
BS = [P[f"bs{i}"] for i in range(1, 7)]
OM = [P[f"omegas{i}"] for i in range(1, 7)]
TOL, LTOL, LOG_SQRT_2PI = DC.TOL, DC.LTOL, 0.5 * math.log(2 * math.pi)
COV = torch.tensor(inp["cov_blocks"], device=dev)  # (T, K, C) float32
# Per-step scalars as Python floats (the float32 values), so the step loop never syncs the device.
DT = inp["dt_blocks"].astype(np.float64).tolist()
DPOP = inp["cov_blocks"][..., 1].astype(np.float64).tolist()
POP = inp["cov_blocks"][..., 2].astype(np.float64).tolist()
POP0 = float(inp["cov0"][2])
gen = torch.Generator(device=dev)


def interval(x, th, t):
    """Advance x (B, J, 8) through observation interval t; th (B, P)."""
    g = lambda k: th[:, P[k]].unsqueeze(1)  # noqa: E731  (B, 1)
    gamma, m, rho, c = g("gamma"), g("m"), g("rho"), g("c")
    sigma, alpha, delta, neps = g("sigma"), g("alpha"), g("delta"), 3.0 * g("epsilon")
    cov = COV[t]
    beta = torch.exp(g("beta_trend") * cov[:, 0] + th[:, BS] @ cov[:, 3:9].T)  # (B, K)
    omega = torch.exp(th[:, OM] @ cov[:, 3:9].T)
    S, I, Y, _, R1, R2, R3, cnt = x.unbind(-1)
    Mn = torch.zeros_like(S)  # accumulator reset at the start of the interval
    for k in range(K):
        dt, dpopdt, pop = DT[t][k], DPOP[t][k], POP[t][k]
        dw = torch.randn(S.shape, device=dev, generator=gen) * math.sqrt(dt)
        effI = (I / pop) ** alpha
        inf = (omega[:, k : k + 1] + (beta[:, k : k + 1] + sigma * dw / dt) * effI) * S
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
        cnt = cnt + neg
        S, I, Y = S_n.clamp_min(0), I_n.clamp_min(0), Y_n.clamp_min(0)
        R1, R2, R3, Mn = R1_n.clamp_min(0), R2_n.clamp_min(0), R3_n.clamp_min(0), Mn_n.clamp_min(0)
    return torch.stack([S, I, Y, Mn, R1, R2, R3, cnt], dim=-1)


class Prior(pydpf.Module):
    def sample(self, batch_size, n_particles, series_metadata, **kw):
        th = series_metadata
        ivp = th[:, IVP]
        S, I, Y, R1, R2, R3 = (POP0 * ivp / ivp.sum(dim=1, keepdim=True)).unbind(-1)
        z = torch.zeros_like(S)
        x0 = torch.stack([S, I, Y, z, R1, R2, R3, z], dim=-1)  # (B, 8)
        x0 = x0.unsqueeze(1).expand(batch_size, n_particles, 8)
        return interval(x0, th, 0)  # state at the first observation


class Dyn(pydpf.Module):
    def sample(self, prev_state, t, series_metadata, **kw):
        return interval(prev_state, series_metadata, t)


class Obs(pydpf.Module):
    def score(self, state, observation, series_metadata, **kw):
        Mn, cnt = state[..., 3], state[..., 7]  # (B, J)
        v = series_metadata[:, P["tau"]].unsqueeze(1) * Mn
        sc = v + TOL
        z = (observation[:, 0:1] - Mn) / sc
        lp = -0.5 * z * z - torch.log(sc) - LOG_SQRT_2PI
        ll = torch.logaddexp(lp, torch.full_like(lp, LTOL))
        return torch.where(torch.isfinite(v) & (cnt <= 0), ll, torch.full_like(ll, LTOL))


pf = pydpf.ParticleFilter(
    resampler=pydpf.SystematicResampler(generator=gen),
    SSM=pydpf.FilteringModel(dynamic_model=Dyn(), observation_model=Obs(), prior_model=Prior()),
).to(dev)
agg = pydpf.LogLikelihoodFactors()

nth, nrep, J = args.nthetas, args.nreps, args.J
B = nth * nrep
thetas = torch.tensor(np.repeat(inp["thetas"][:nth], nrep, axis=0), dtype=torch.float32, device=dev)
ys = torch.tensor(inp["ys"], dtype=torch.float32, device=dev).reshape(T, 1, 1)
chunk = args.chunk if 0 < args.chunk < B else B


def workload(c):
    gen.manual_seed(int(DC.task_seed(args.seed, c, 0, 0)))
    out = []
    with torch.inference_mode():
        for lo in range(0, B, chunk):
            th = thetas[lo : lo + chunk]
            y = ys.expand(T, th.shape[0], 1)
            f = pf(n_particles=J, time_extent=T - 1, aggregation_function=agg, observation=y,
                   series_metadata=th)
            out.append(f.sum(0)[:, 0])
        ll = torch.cat(out).cpu().numpy()  # .cpu() syncs the device
    return ll.reshape(nth, nrep)


def call(c):
    t0 = time.perf_counter()
    ll = workload(c)
    return time.perf_counter() - t0, ll


cold_s, ll = call(0)
warm_s, _ = call(1)
DC.record("pydpf", args, inp, cold_s, warm_s, ll, DC.versions("pydpf", "torch"),
          variant="ParticleFilter/systematic", dtype="float32", resampler="systematic", chunk=chunk,
          torch_threads=torch.get_num_threads() if args.device == "cpu" else None)
