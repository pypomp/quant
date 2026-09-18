"""FHN: compare JAX Strang paths and samples with the R author reference."""

# --- SLURM CONFIG ---
# importance: medium
# description: "FHN: Strang simulator checks against original author C++/R"
# tags: [reference, fitzhugh_nagumo, cpu]
# sbatch_args:
#   job-name: "fhn reference"
#   partition: standard
#   cpus-per-task: 1
#   mem: 6GB
#   time: "00:05:00"
#   output: "results/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:15:00" }
#   3:
#     sbatch_args: { time: "00:15:00" }
#   4:
#     sbatch_args: { time: "00:15:00" }
# --- END SLURM CONFIG ---

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["JAX_ENABLE_X64"] = "true"
os.environ["JAX_PLATFORMS"] = "cpu"

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
DATA = MODEL / "data"
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(MODEL.parent))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import ks_2samp

import strang
from utils import run_metadata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(data):
    return strang.FHNParameters(
        data["epsilon"], data["gamma"], data["beta"], data["sigma"], 0.0
    )


def verify_reference_files(
    directory=DATA, *, sources=DATA, full=True, expected_record_sha256=None
):
    directory, sources = Path(directory), Path(sources)
    record_path = directory / "reference_results.json"
    if (
        expected_record_sha256 is not None
        and sha(record_path) != expected_record_sha256
    ):
        raise ValueError("R baseline provenance does not match reference_results.json")
    records = json.loads(record_path.read_text())
    for record in records["sources"]:
        if sha(sources / record["path"]) != record["sha256"]:
            raise ValueError(f"Author source hash changed: {record['path']}")
    if sha(directory / "fixtures.json") != records["output_sha256"]["fixtures"]:
        raise ValueError("R fixture hash changed")
    if full:
        if not records["distribution_ran_this_invocation"]:
            raise ValueError(
                "Full validation requires a full R reference, not fixtures-only smoke"
            )
        if (
            sha(directory / "distribution_samples.json")
            != records["output_sha256"]["distribution_samples"]
        ):
            raise ValueError("R distribution sample hash changed")
    if records["metadata"]["author_cpp_modified"] is not False:
        raise ValueError("Reference must use unmodified author C++")
    return records


def coupled_checks(directory=DATA):
    fixtures = json.loads((Path(directory) / "fixtures.json").read_text())
    results = []
    for case in fixtures["scenarios"]:
        p = parameters(case["parameters"])
        actual = np.asarray(
            strang.path_from_innovations(
                case["initial_state"], case["innovations"], case["dt"], p
            )
        )
        expected = np.asarray(case["author_states"])
        voltage = np.asarray(case["author_V"])
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
        np.testing.assert_allclose(actual[:, 0], voltage, rtol=2e-11, atol=2e-12)
        e, c = map(np.asarray, strang.linear_moments(case["dt"], p))
        np.testing.assert_allclose(
            e, case["experiment_transition"], rtol=2e-13, atol=2e-15
        )
        a = np.array([[0.0, -1 / p.epsilon], [p.gamma, -1.0]])

        def integrand(t):
            column = expm(a * t)[:, 1] * p.sigma
            return np.outer(column, column)

        independent, _ = quad_vec(
            integrand, 0.0, case["dt"], epsabs=1e-25, epsrel=3e-13
        )
        np.testing.assert_allclose(c, independent, rtol=1e-11, atol=1e-22)
        results.append(
            {
                "name": case["name"],
                "dt": case["dt"],
                "n_steps": case["n_steps"],
                "parameters": case["parameters"],
                "initial_state": case["initial_state"],
                "max_absolute_state_error": float(np.max(np.abs(actual - expected))),
                "max_absolute_voltage_error_vs_package": float(
                    np.max(np.abs(actual[:, 0] - voltage))
                ),
                "jax_covariance_max_relative_error_vs_quadrature": float(
                    np.max(np.abs((c - independent) / independent))
                ),
                "author_package_covariance_max_absolute_error_vs_quadrature": float(
                    np.max(np.abs(np.asarray(case["author_covariance"]) - independent))
                ),
                "author_package_covariance_max_relative_error_vs_quadrature": float(
                    np.max(
                        np.abs(
                            (np.asarray(case["author_covariance"]) - independent)
                            / independent
                        )
                    )
                ),
                "author_experiment_covariance_max_relative_error_vs_quadrature": float(
                    np.max(
                        np.abs(
                            (np.asarray(case["experiment_covariance"]) - independent)
                            / independent
                        )
                    )
                ),
                "passed": True,
            }
        )
    return results


@partial(jax.jit, static_argnames=("n_steps", "n_paths"))
def sample_endpoints(key, initial_state, dt, p, *, n_steps, n_paths):
    e, c_unit = strang._unit_linear_moments(dt, p)
    factor = p.sigma * jnp.linalg.cholesky(c_unit)
    initial = jnp.broadcast_to(
        jnp.asarray(initial_state, dtype=jnp.float64), (n_paths, 2)
    )

    def advance(i, carry):
        states, key = carry
        key, draw_key = jax.random.split(key)
        xi = jax.random.normal(draw_key, (n_paths, 2), dtype=jnp.float64) @ factor.T
        return strang._step_with_matrix(states, xi, dt, p, e), key

    return jax.lax.fori_loop(0, n_steps, advance, (initial, key))[0]


def distribution_metrics(a, b, number_of_cdf_checks):
    """Descriptive moment z values plus a predeclared conservative CDF bound."""
    n, m = len(a), len(b)
    ca, cb = np.cov(a.T), np.cov(b.T)
    mean_se = np.sqrt(np.diag(ca) / n + np.diag(cb) / m)
    mean_z = (b.mean(0) - a.mean(0)) / mean_se
    da, db = a - a.mean(0), b - b.mean(0)
    influence_a = np.stack((da[:, 0] ** 2, da[:, 0] * da[:, 1], da[:, 1] ** 2), axis=1)
    influence_b = np.stack((db[:, 0] ** 2, db[:, 0] * db[:, 1], db[:, 1] ** 2), axis=1)
    cov_se = np.sqrt(influence_a.var(0, ddof=1) / n + influence_b.var(0, ddof=1) / m)
    cov_diff = np.array([cb[0, 0] - ca[0, 0], cb[0, 1] - ca[0, 1], cb[1, 1] - ca[1, 1]])
    cov_z = cov_diff / cov_se
    # DKW + union bound for the two empirical CDFs in every comparison.
    # This tests sampling compatibility, not exact distributional equality.
    family_alpha = 0.001
    logarithm = np.log(4 * number_of_cdf_checks / family_alpha)
    cdf_limit = np.sqrt(logarithm / (2 * n)) + np.sqrt(logarithm / (2 * m))
    ks = [ks_2samp(a[:, j], b[:, j]) for j in range(2)]
    return {
        "R_mean": a.mean(0).tolist(),
        "JAX_mean": b.mean(0).tolist(),
        "R_covariance": ca.tolist(),
        "JAX_covariance": cb.tolist(),
        "mean_difference_combined_mcse": mean_se.tolist(),
        "mean_difference_z": mean_z.tolist(),
        "covariance_difference_combined_mcse": cov_se.tolist(),
        "covariance_difference_z": cov_z.tolist(),
        "ks_distance": [float(k.statistic) for k in ks],
        "ks_pvalue": [float(k.pvalue) for k in ks],
        "cdf_distance_bound": float(cdf_limit),
        "cdf_family_alpha": family_alpha,
        "moment_z_threshold": 5.0,
        "quantile_probabilities": [0.05, 0.25, 0.5, 0.75, 0.95],
        "R_quantiles": np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0).tolist(),
        "JAX_quantiles": np.quantile(b, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0).tolist(),
        "passed": bool(
            np.max(np.abs(mean_z)) < 5
            and np.max(np.abs(cov_z)) < 5
            and max(k.statistic for k in ks) < cdf_limit
        ),
    }


def validate(directory=DATA, *, sources=DATA, full=True, expected_record_sha256=None):
    """Run the coupled checks and, optionally, independent-path comparisons."""
    directory = Path(directory)
    reference = verify_reference_files(
        directory,
        sources=sources,
        full=full,
        expected_record_sha256=expected_record_sha256,
    )
    coupled = coupled_checks(directory)
    print(
        f"Coupled paths passed: {len(coupled)}, max error {max(x['max_absolute_state_error'] for x in coupled):.3g}",
        flush=True,
    )
    r = (
        json.loads((directory / "distribution_samples.json").read_text())
        if full
        else {"scenarios": []}
    )
    records, samples = [], []
    for index, case in enumerate(r["scenarios"]):
        seed = 672013 + index
        start = time.perf_counter()
        a = np.asarray(case["endpoint_samples"], dtype=float)
        b = np.asarray(
            sample_endpoints(
                jax.random.key(seed),
                np.asarray(case["initial_state"], float),
                case["dt"],
                parameters(case["parameters"]),
                n_steps=case["n_steps"],
                n_paths=case["n_paths"],
            )
        )
        assert a.shape == b.shape == (case["n_paths"], 2)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        metrics = distribution_metrics(a, b, 2 * len(r["scenarios"]))
        records.append(
            {
                "name": case["name"],
                "dt": case["dt"],
                "horizon": case["horizon"],
                "parameters": case["parameters"],
                "initial_state": case["initial_state"],
                "paths_per_implementation": case["n_paths"],
                "jax_seed": seed,
                "elapsed_jax_seconds_including_compile": time.perf_counter() - start,
                **metrics,
            }
        )
        samples.append(
            {"name": case["name"], "jax_seed": seed, "endpoint_samples": b.tolist()}
        )
        print(
            f"Distribution dt={case['dt']}: passed={metrics['passed']}, max mean z={max(abs(x) for x in metrics['mean_difference_z']):.3f}, max covariance z={max(abs(x) for x in metrics['covariance_difference_z']):.3f}",
            flush=True,
        )
    report = {
        "scope": "Strang simulator only; no likelihood, gradient, optimizer, or 2019 experiment reproduction.",
        "coupled_scenarios": coupled,
        "distribution_scenarios": records,
        "covariance_note": "Stable Van Loan linear-substep moments; shared additive noise separates state-update error from author covariance cancellation.",
        "all_passed": all(x["passed"] for x in coupled + records),
    }
    return report, samples, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    level = int(os.environ.get("RUN_LEVEL", "1"))
    if level not in (1, 2, 3, 4):
        parser.error("RUN_LEVEL must be 1, 2, 3, or 4")
    results = args.output_dir.resolve()
    rdir = results / "R"
    if level == 1 and (results / "smoke/R/latest.json").exists():
        rdir = results / "smoke/R"
    r_metadata = json.loads((rdir / "latest.json").read_text())
    reference_dir = rdir if (rdir / "reference_results.json").exists() else DATA

    started = time.perf_counter()
    report, samples, reference = validate(
        reference_dir,
        full=level != 1,
        expected_record_sha256=r_metadata["reference_record_sha256"],
    )
    metadata = run_metadata(
        {
            "kind": "reference",
            "model": "fitzhugh_nagumo",
            "RUN_LEVEL": level,
            "validation_scope": "coupled-only smoke"
            if level == 1
            else "full simulator check",
            "scope": report["scope"],
            "jax_enable_x64": True,
            "elapsed_seconds_including_compile": time.perf_counter() - started,
        }
    )
    metadata.update(
        reference_record_sha256=r_metadata["reference_record_sha256"],
        reference_input_sha256=reference["output_sha256"],
        author_reference=reference["metadata"],
        source_sha256={
            name: sha(MODEL / name)
            for name in ("model.py", "strang.py", "reference/run.py")
        },
    )

    out_dir = results / ("smoke/cpu" if level == 1 else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    coupled = pd.DataFrame(report["coupled_scenarios"])
    coupled[
        [
            "name",
            "dt",
            "n_steps",
            "max_absolute_state_error",
            "max_absolute_voltage_error_vs_package",
            "jax_covariance_max_relative_error_vs_quadrature",
            "passed",
        ]
    ].to_csv(out_dir / "coupled.csv", index=False)
    distributions = report["distribution_scenarios"]
    rows = []
    for case in distributions:
        rows.append(
            {
                "name": case["name"],
                "dt": case["dt"],
                "horizon": case["horizon"],
                "n_paths": case["paths_per_implementation"],
                "jax_seed": case["jax_seed"],
                "max_abs_mean_z": max(map(abs, case["mean_difference_z"])),
                "max_abs_covariance_z": max(map(abs, case["covariance_difference_z"])),
                "max_cdf_distance": max(case["ks_distance"]),
                "cdf_distance_bound": case["cdf_distance_bound"],
                "passed": case["passed"],
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "name",
            "dt",
            "horizon",
            "n_paths",
            "jax_seed",
            "max_abs_mean_z",
            "max_abs_covariance_z",
            "max_cdf_distance",
            "cdf_distance_bound",
            "passed",
        ],
    ).to_csv(out_dir / "distribution.csv", index=False)
    endpoints = [
        {
            "name": case["name"],
            "dt": case["dt"],
            "path": index,
            "V": state[0],
            "U": state[1],
        }
        for case, sample in zip(distributions, samples, strict=True)
        for index, state in enumerate(sample["endpoint_samples"])
    ]
    pd.DataFrame(endpoints, columns=["name", "dt", "path", "V", "U"]).to_csv(
        out_dir / "endpoints.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "stage": case["name"] + " (includes JAX compilation)",
                "seconds": case["elapsed_jax_seconds_including_compile"],
            }
            for case in distributions
        ],
        columns=["stage", "seconds"],
    ).to_csv(out_dir / "timings.csv", index=False)
    (out_dir / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    metadata["output_sha256"] = {
        name: sha(out_dir / name)
        for name in (
            "coupled.csv",
            "distribution.csv",
            "endpoints.csv",
            "timings.csv",
            "validation.json",
        )
    }
    (out_dir / "latest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Wrote {out_dir}: {metadata['run_config']['validation_scope']}; passed={report['all_passed']}"
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
