"""
features.py
Paper mapping: Section IV.C

Constructs three explicit, non-overlapping feature sets:
  Set 1 — Baseline:  geometry/morphological + WSS_mean columns (MinMax-scaled, 14 features)
  Set 2 — ODE-only:  10 biomarkers extracted from ODE simulation
  Set 3 — Hybrid:    concatenation of Set 1 + Set 2 (24 features)

CRITICAL: shape assertions guard against the silent concat-bug (identical sets).
"""

import pandas as pd


def build_feature_sets(
    X_baseline_df: pd.DataFrame,
    biomarkers_dict: dict,
) -> tuple:
    """Return (X_baseline, X_ode, X_hybrid) as three distinct DataFrames.

    Parameters
    ----------
    X_baseline_df   : pd.DataFrame — shape (n, 14)  MinMax-scaled geometry + WSS features
    biomarkers_dict : dict          — 10-key dict, each value is a list of length n

    Returns
    -------
    X_baseline : pd.DataFrame — (n, 14)
    X_ode      : pd.DataFrame — (n, 10)
    X_hybrid   : pd.DataFrame — (n, 24)
    """
    bio_df = pd.DataFrame(biomarkers_dict)

    X_baseline = X_baseline_df.reset_index(drop=True).copy()
    X_ode      = bio_df.reset_index(drop=True).copy()
    X_hybrid   = pd.concat([X_baseline, X_ode], axis=1)

    # ── Shape assertions (guard against concat bug) ────────────────────────────
    assert X_baseline.shape[1] != X_ode.shape[1], (
        f"[BUG] Baseline ({X_baseline.shape[1]} cols) == ODE ({X_ode.shape[1]} cols) "
        "— feature sets are identical!"
    )
    assert X_hybrid.shape[1] == X_baseline.shape[1] + X_ode.shape[1], (
        f"[BUG] Hybrid shape mismatch: expected "
        f"{X_baseline.shape[1] + X_ode.shape[1]} cols, got {X_hybrid.shape[1]}"
    )
    assert X_baseline.shape[0] == X_ode.shape[0] == X_hybrid.shape[0], (
        "[BUG] Row count mismatch across feature sets!"
    )

    print(f"[OK] Feature sets constructed:")
    print(f"     Baseline: {X_baseline.shape}  | "
          f"ODE: {X_ode.shape}  | "
          f"Hybrid: {X_hybrid.shape}")

    # ── Biomarker variance check ───────────────────────────────────────────────
    bio_stds = X_ode.std().sort_values()
    low_var  = bio_stds[bio_stds < 0.01]
    if not low_var.empty:
        print(f"[WARN] Low-variance ODE biomarkers (std < 0.01): "
              f"{list(low_var.index)} - these may be uninformative.")
    print("[INFO] ODE biomarker std values:")
    for col, std_val in bio_stds.items():
        flag = "  ** LOW **" if std_val < 0.01 else ""
        print(f"       {col}: {std_val:.4f}{flag}")

    return X_baseline, X_ode, X_hybrid