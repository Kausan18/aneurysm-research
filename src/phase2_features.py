"""
phase2_features.py
Phase 2 contribution — Section 4.1 of Plan of Action.

Runs RFECV (Recursive Feature Elimination with Cross-Validation) on the HYBRID
feature set to remove redundant ODE biomarkers identified in Phase 1 diagnostics:

  Likely redundant pairs:
    - r0 / r_end    — both derived from S, highly correlated by construction
    - I / I_dur / I_auc — three versions of the same inflammation excess signal
    - AUC_i / i_max — both capture peak inflammation magnitude

Target: reduce ~23 hybrid features to ~15 by removing low-importance ODE columns.
This should directly fix the RF/GBM degradation seen in Phase 1.

After selection, the function returns three ALIGNED feature sets:
  X_baseline_sel — baseline columns that survived selection  (or all 13 if none pruned)
  X_ode_sel      — ODE biomarkers that survived selection
  X_hybrid_sel   — concatenation of the two selected sets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


def select_features_rfecv(
    X_baseline: pd.DataFrame,
    X_ode: pd.DataFrame,
    X_hybrid: pd.DataFrame,
    y: np.ndarray,
    min_features: int = 10,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple:
    """Apply RFECV on the hybrid set; split selected columns back into baseline/ODE.

    Parameters
    ----------
    X_baseline  : pd.DataFrame — shape (n, 13), MinMax-scaled baseline features
    X_ode       : pd.DataFrame — shape (n, 10), ODE biomarkers
    X_hybrid    : pd.DataFrame — shape (n, 23), concat of baseline + ODE
    y           : np.ndarray   — binary rupture labels
    min_features: int — minimum features RFECV may select (floor guard)
    cv_splits   : int — StratifiedKFold splits inside RFECV (5 keeps it fast)
    random_state: int

    Returns
    -------
    X_baseline_sel : pd.DataFrame — baseline columns surviving selection
    X_ode_sel      : pd.DataFrame — ODE biomarker columns surviving selection
    X_hybrid_sel   : pd.DataFrame — concat of the two selected subsets
    selected_cols  : list[str]    — names of all surviving features
    rfecv_support  : np.ndarray   -- boolean mask over X_hybrid.columns
    """
    print(f"\n[Phase 2 Features] Running RFECV on hybrid set "
          f"({X_hybrid.shape[1]} features -> target >= {min_features})...")

    # ── RFECV estimator: RF with balanced class weights ───────────────────────
    rf_estimator = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rfecv = RFECV(
        estimator=rf_estimator,
        step=1,                        # remove one feature per iteration
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=min_features,
        n_jobs=-1,
    )

    rfecv.fit(X_hybrid.values, y)

    # ── Selected feature names ────────────────────────────────────────────────
    support       = rfecv.support_                  # bool mask over X_hybrid cols
    selected_cols = list(X_hybrid.columns[support])
    n_sel         = int(support.sum())

    print(f"[Phase 2 Features] RFECV selected {n_sel} / {X_hybrid.shape[1]} features.")
    print(f"  Optimal n_features (by CV AUC): {rfecv.n_features_}")

    # ── Which selected cols belong to baseline vs ODE? ───────────────────────
    baseline_cols = list(X_baseline.columns)
    ode_cols      = list(X_ode.columns)

    sel_baseline = [c for c in selected_cols if c in baseline_cols]
    sel_ode      = [c for c in selected_cols if c in ode_cols]

    print(f"\n  Baseline features kept ({len(sel_baseline)}/{len(baseline_cols)}): "
          f"{sel_baseline}")
    print(f"  ODE biomarkers kept   ({len(sel_ode)}/{len(ode_cols)}): "
          f"{sel_ode}")

    # Features removed from ODE (the key diagnostic)
    removed_ode = [c for c in ode_cols if c not in sel_ode]
    if removed_ode:
        print(f"\n  [INFO] ODE biomarkers eliminated by RFECV: {removed_ode}")
    else:
        print(f"\n  [INFO] No ODE biomarkers eliminated — all retained.")

    # ── Build aligned selected DataFrames ────────────────────────────────────
    # Preserve original column ordering within each subset
    X_baseline_sel = X_baseline[sel_baseline].reset_index(drop=True)
    X_ode_sel      = X_ode[sel_ode].reset_index(drop=True)
    X_hybrid_sel   = pd.concat(
        [X_baseline_sel, X_ode_sel], axis=1
    ).reset_index(drop=True)

    # Guard: hybrid_sel must equal baseline_sel + ode_sel
    assert X_hybrid_sel.shape[1] == len(sel_baseline) + len(sel_ode), (
        "[BUG] Hybrid-sel shape mismatch after RFECV split."
    )

    print(f"\n[Phase 2 Features] Post-selection shapes:")
    print(f"  Baseline-sel : {X_baseline_sel.shape}")
    print(f"  ODE-sel      : {X_ode_sel.shape}")
    print(f"  Hybrid-sel   : {X_hybrid_sel.shape}")

    return X_baseline_sel, X_ode_sel, X_hybrid_sel, selected_cols, len(selected_cols)


def print_feature_importance_ranking(X_hybrid: pd.DataFrame,
                                     y: np.ndarray,
                                     random_state: int = 42) -> pd.DataFrame:
    """Fit a single RF on the full hybrid set and print permutation importances.

    Useful as a pre-RFECV diagnostic to understand which ODE biomarkers
    are contributing before recursive elimination starts.

    Returns
    -------
    pd.DataFrame with columns [feature, importance, std] sorted descending.
    """
    from sklearn.inspection import permutation_importance

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_hybrid.values, y)

    pi = permutation_importance(
        rf, X_hybrid.values, y,
        n_repeats=10,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature":    X_hybrid.columns,
        "importance": pi.importances_mean,
        "std":        pi.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n[Phase 2 Features] Permutation importances (hybrid set, RF, AUC):")
    print(imp_df.to_string(index=False))
    return imp_df
