"""
repro.py — Central reproducibility: seed-fixing + package version logging.
Call `repro.fix_seeds()` from every notebook's first cell.
"""

import os, random, json, datetime

SEED = 42

def fix_seeds():
    """Fix seeds for numpy, random, sklearn, and tensorflow (if available)."""
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass

    print(f"[repro] Seeds fixed to {SEED}.")
    _log_versions()

def _log_versions():
    """Print key library versions to stdout for traceability."""
    pkgs = ["numpy", "pandas", "scipy", "sklearn", "mne", "tensorflow", "pyarrow"]
    versions = {}
    for pkg in pkgs:
        try:
            import importlib
            m = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(m, "__version__", "?")
        except ImportError:
            versions[pkg] = "NOT INSTALLED"

    print("[repro] Package versions:", json.dumps(versions, indent=2))
    return versions
