"""The models in the cross-model reports, and where each kind's results live.

Directories are relative to the repository root. `arms` maps a display label to
a subdirectory of the kind's results directory. In `loglik`, every arm is
compared with `reference`, and arms listed in `info` are reported without a
verdict.
"""

MODELS = {
    "spx": {
        "label": "SPX",
        "panel": False,
        "loglik": {
            "dir": "tests/spx/loglik/results",
            "arms": {"R pomp": "R", "pypomp (GPU)": "gpu"},
            "reference": "R pomp",
        },
        "timing": {
            "dir": "tests/spx/timing/results",
            "arms": {
                "R pomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (CPU, 36 cores)": "cpu",
            },
        },
        "scaling": {"dir": "tests/spx/scaling/results/gpu"},
    },
    "dacca": {
        "label": "Dhaka cholera",
        "panel": False,
        "loglik": {
            "dir": "tests/dacca/loglik/results",
            "arms": {"R pomp": "R", "pypomp (GPU)": "gpu"},
            "reference": "R pomp",
        },
        "timing": {
            "dir": "tests/dacca/timing/results",
            "arms": {
                "R pomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (CPU, 36 cores)": "cpu",
            },
        },
        "scaling": {"dir": "tests/dacca/scaling/results/gpu"},
    },
    "measles": {
        "label": "Measles",
        "panel": False,
        "loglik": {
            "dir": "tests/measles/loglik/results",
            "arms": {
                "R pomp": "R",
                "pypomp (64-bit)": "f64",
                "pypomp (32-bit)": "f32",
            },
            "reference": "R pomp",
            "info": ["pypomp (32-bit)"],
        },
        "timing": {
            "dir": "tests/measles/timing/results",
            "arms": {
                "R pomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, JAX samplers)": "gpu_jax",
                "pypomp (CPU, 36 cores)": "cpu",
            },
        },
        "scaling": {"dir": "tests/measles/scaling/results/gpu"},
    },
    "panel_measles": {
        "label": "Panel measles",
        "panel": True,
        "loglik": {
            "dir": "tests/panel_measles/loglik/results",
            "arms": {"R panelPomp": "R", "pypomp (GPU)": "gpu"},
            "reference": "R panelPomp",
        },
        "timing": {
            "dir": "tests/panel_measles/timing/results",
            "arms": {"R panelPomp (36 cores)": "R", "pypomp (GPU)": "gpu"},
        },
        "scaling": {"dir": "tests/panel_measles/scaling/results/gpu"},
    },
}
