"""pydpf (PyTorch): user-defined modules, plain ParticleFilter, systematic resampling every step.

usage: python bench_pydpf.py MODEL DEVICE J [J ...]     (DEVICE = cpu | cuda)
"""

import math
import sys

import pydpf
import torch

import common as C

model_name, device = sys.argv[1], sys.argv[2]
Js = [int(j) for j in sys.argv[3:]]
dev = torch.device(device)
# (time, batch=REPS, dim=1)
y = torch.tensor(C.data(model_name), dtype=torch.float32, device=dev).reshape(C.T, 1, 1).expand(C.T, C.REPS, 1).contiguous()
gen = torch.Generator(device=dev)


def lg_step(x):
    return C.A * x + C.SX * torch.randn(x.shape, device=dev, generator=gen)


def sde_step(z):
    for _ in range(C.NSTEP):
        eps = torch.randn(z.shape, device=dev, generator=gen)
        z = z + (C.R * (1 - torch.exp(z)) - C.SIG**2 / 2) * C.DT + C.SIG * math.sqrt(C.DT) * eps
    return z


step = lg_step if model_name == "lg" else sde_step
sy = C.SY if model_name == "lg" else C.TAU


class Prior(pydpf.Module):
    def sample(self, batch_size, n_particles, **kw):
        if model_name == "lg":
            x0 = torch.randn((batch_size, n_particles, 1), device=dev, generator=gen)
        else:
            x0 = C.Z0M + C.Z0S * torch.randn((batch_size, n_particles, 1), device=dev, generator=gen)
        return step(x0)  # state at the first observation


class Dyn(pydpf.Module):
    def sample(self, prev_state, **kw):
        return step(prev_state)


class Obs(pydpf.Module):
    def fitness(self, state, observation, **kw):
        # state (B, K, 1), observation (B, 1) -> (B, K)
        d = (state[..., 0] - observation[:, None, 0]) / sy
        return -0.5 * d * d - math.log(sy) - 0.5 * math.log(2 * math.pi)


pf = pydpf.ParticleFilter(
    resampler=pydpf.SystematicResampler(generator=gen),
    SSM=pydpf.FilteringModel(dynamic_model=Dyn(), observation_model=Obs(), prior_model=Prior()),
).to(dev)
agg = pydpf.LogLikelihoodFactors()

for J in Js:
    def run(seed, J=J):
        gen.manual_seed(seed)
        with torch.inference_mode():
            out = pf(n_particles=J, time_extent=C.T - 1, aggregation_function=agg, observation=y)
            ll = out.sum(0).mean().item()  # .item() syncs the device
        return ll

    first, med, lls = C.timeit(run)
    C.record("pydpf", "systematic", model_name, "gpu" if device == "cuda" else "cpu", J, first, med, lls)
