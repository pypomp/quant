"""Run one of the published fits using the original train_features.npz and starts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = "pypomp-maze-dmop-20261106"


def batch_order(seed, epoch, size):
    fields = {
        "study": STUDY,
        "domain": "batch_order",
        "training_domain": "train_formal",
        "start": seed,
        "epoch": epoch,
    }
    digest = key_digest(fields)
    return np.random.Generator(
        np.random.PCG64(int.from_bytes(digest, "big"))
    ).permutation(size)


def key_digest(fields):
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).digest()


def fit(features, starts, seed, particles, alpha, seconds):
    import jax
    import jax.numpy as jnp
    import optax
    from jax.experimental import checkify

    sys.path.insert(0, str(HERE.parent))
    import model

    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64:
        raise RuntimeError("The published fits used a GPU and float32.")
    jax.config.update("jax_default_matmul_precision", "highest")
    adam = optax.scale_by_adam(b1=0.8, b2=0.99, eps=1e-9)
    theta = jnp.asarray(starts[str(seed)])
    state = adam.init(theta)

    @jax.jit
    @jax.vmap
    def fold_key(words):
        key = jax.random.key(0, impl="threefry2x32")
        for word in words:
            key = jax.random.fold_in(key, word)
        return key

    @jax.jit
    @checkify.checkify
    def step(theta, state, z, controls, x0, keys, discount, flow_lr):
        loss, gradient = model.batch_value_and_gradient(
            theta, z, controls, x0, keys, particles, discount
        )
        updates, state = adam.update(gradient, state, theta)
        rates = jnp.concatenate(
            (jnp.full(6896, 5 * flow_lr), jnp.full(model.PARAMETERS - 6896, flow_lr))
        )
        proposal = optax.apply_updates(theta, -rates * updates)
        checkify.check(jnp.all(jnp.isfinite(proposal)), "nonfinite parameters")
        checkify.check(
            jnp.all(jnp.isfinite(state.mu)) & jnp.all(jnp.isfinite(state.nu)),
            "nonfinite Adam moments",
        )
        return proposal, state, loss, gradient

    def complete(indices, domain, update):
        start = perf_counter()
        z, controls, x0 = (
            jnp.asarray(features[k][indices])
            for k in ("z", "u_observed", "initial_pose")
        )
        words = []
        for trajectory in features["trajectory_id"][indices]:
            fields = {
                "study": STUDY,
                "domain": domain,
                "start": seed,
                "trajectory_id": str(trajectory),
                "global_update": update,
            }
            words.append(
                np.frombuffer(key_digest(fields), dtype=">u4").astype(np.uint32)
            )
        keys = fold_key(jnp.asarray(np.stack(words)))
        error, value = step(
            theta, state, z, controls, x0, keys, jnp.float32(alpha), jnp.float32(0.001)
        )
        jax.block_until_ready(value)
        error.throw()
        np.asarray(jax.random.key_data(keys))
        return value, perf_counter() - start

    size = len(features["trajectory_id"])
    for indices in (np.arange(64), np.arange(896, 900)):
        complete(indices, "preflight", 0)
    elapsed = 0.0
    updates = 0
    epoch = 0
    while elapsed < seconds:
        order = batch_order(seed, epoch, size)
        for first in range(0, size, 64):
            (theta, state, _, _), duration = complete(
                order[first : first + 64], "train_formal", updates
            )
            updates += 1
            elapsed += duration
            if elapsed >= seconds:
                break
        epoch += 1
    return {"theta": np.asarray(theta), "active_seconds": elapsed, "updates": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, required=True, help="Archived train_features.npz"
    )
    parser.add_argument(
        "--starts", type=Path, required=True, help="Archived initial_parameters.npz"
    )
    parser.add_argument("--seed", type=int, choices=(0, 10, 20, 30, 40), required=True)
    parser.add_argument("--particles", type=int, choices=(100, 500), required=True)
    parser.add_argument(
        "--alpha", type=float, choices=(0, 0.75, 0.97, 1), required=True
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New endpoint NPZ file"
    )
    args = parser.parse_args()
    record = json.loads((HERE / "results/gpu/latest.json").read_text())
    for path, digest in (
        (args.features, "train_features_sha256"),
        (args.starts, "initial_parameters_sha256"),
    ):
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != record[digest]:
                raise ValueError(f"Unknown input: {path}")
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    with (
        np.load(args.features, allow_pickle=False) as data,
        np.load(args.starts, allow_pickle=False) as starts,
    ):
        features = {key: data[key] for key in data.files}
        result = fit(
            features,
            starts,
            args.seed,
            args.particles,
            args.alpha,
            record["active_seconds_per_fit"],
        )
    with args.output.open("xb") as f:
        np.savez(f, **result)


if __name__ == "__main__":
    main()
