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
            "arms": {
                "R pomp": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
            "reference": "R pomp",
        },
        "timing": {
            "dir": "tests/spx/timing/results",
            "arms": {
                "R pomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
                "pypomp (CPU, 36 cores)": "cpu",
            },
        },
        "scaling": {
            "dir": "tests/spx/scaling/results",
            "arms": {
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
        },
    },
    "dacca": {
        "label": "Dhaka cholera",
        "panel": False,
        "loglik": {
            "dir": "tests/dacca/loglik/results",
            "arms": {
                "R pomp": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
            "reference": "R pomp",
        },
        "timing": {
            "dir": "tests/dacca/timing/results",
            "arms": {
                "R pomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
                "pypomp (CPU, 36 cores)": "cpu",
            },
        },
        "scaling": {
            "dir": "tests/dacca/scaling/results",
            "arms": {
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
        },
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
                "pypomp (32-bit, CuTe philox)": "f32_cute",
            },
            "reference": "R pomp",
            "info": ["pypomp (32-bit)", "pypomp (32-bit, CuTe philox)"],
        },
        "timing": {
            "dir": "tests/measles/timing/results",
            "arms": {
                "R pomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, JAX samplers)": "gpu_jax",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
                "pypomp (CPU, 36 cores)": "cpu",
            },
        },
        "scaling": {
            "dir": "tests/measles/scaling/results",
            "arms": {
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
        },
    },
    "panel_measles": {
        "label": "Panel measles",
        "panel": True,
        "loglik": {
            "dir": "tests/panel_measles/loglik/results",
            "arms": {
                "R panelPomp": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
            "reference": "R panelPomp",
        },
        "timing": {
            "dir": "tests/panel_measles/timing/results",
            "arms": {
                "R panelPomp (36 cores)": "R",
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
        },
        "scaling": {
            "dir": "tests/panel_measles/scaling/results",
            "arms": {
                "pypomp (GPU)": "gpu",
                "pypomp (GPU, CuTe philox)": "gpu_cute",
            },
        },
    },
}
