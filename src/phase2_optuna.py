"""
phase2_optuna.py
Phase 2 contribution — Section 3 of Plan of Action.

Bayesian search over the 9 ODE parameters (a1–c3) using Optuna TPE sampler.
Objective: maximise mean CV AUC on the HYBRID feature set with Random Forest.

Design decisions:
  - RF is the objective model (most sensitive to biomarker quality).
  - RepeatedStratifiedKFold(5×3) kept consistent with Phase 1 evaluation.
  - n_estimators=50 inside objective (fast proxy); full evaluation uses 100.
  - MedianPruner prunes unpromising trials early.
  - study seed=42 for reproducibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from src.ode_model  import simulate_and_extract
from src.features   import build_feature_sets
from config         import PHASE2_PARAM_BOUNDS, NOMINAL_PARAMS


# Silence Optuna's per-trial INFO logging; only show warnings and above.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_auc_hybrid(params: dict,
                   L: np.ndarray,
                   H: np.ndarray,
                   S: np.ndarray,
                   X_baseline_df: pd.DataFrame,
                   y: np.ndarray,
                   n_splits: int = 5,
                   n_repeats: int = 3,
                   random_state: int = 42) -> float:
    """Compute mean CV AUC on the hybrid feature set for a given param dict.

    Parameters
    ----------
    params         : dict — ODE coefficient dict (keys a1..c3)
    L, H, S        : np.ndarray — geometry surrogates, shape (n_patients,)
    X_baseline_df  : pd.DataFrame — MinMax-scaled baseline features (n, 13)
    y              : np.ndarray — binary rupture labels
    n_splits, n_repeats : RepeatedStratifiedKFold configuration

    Returns
    -------
    float — mean AUC across all folds (higher = better)
    """
    # ── Recompute ODE biomarkers with trial params ────────────────────────────
    biomarkers = simulate_and_extract(L, H, S, params=params)

    # ── Build hybrid feature set ──────────────────────────────────────────────
    _, _, X_hybrid = build_feature_sets(X_baseline_df, biomarkers)
    X_np = X_hybrid.values

    # ── Cross-validated AUC with RF (fast, n_estimators=50) ──────────────────
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=50,           # faster proxy for objective evaluation
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    fold_aucs = []
    for train_idx, test_idx in cv.split(X_np, y):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        rf.fit(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        fold_aucs.append(auc)

    return float(np.mean(fold_aucs))


def run_optuna_study(L: np.ndarray,
                     H: np.ndarray,
                     S: np.ndarray,
                     X_baseline_df: pd.DataFrame,
                     y: np.ndarray,
                     n_trials: int = 50,
                     random_state: int = 42) -> tuple:
    """Run the Bayesian ODE coefficient optimisation study.

    Parameters
    ----------
    L, H, S        : np.ndarray — geometry surrogates from preprocess_features()
    X_baseline_df  : pd.DataFrame — MinMax-scaled baseline features (n, 13)
    y              : np.ndarray — binary rupture labels
    n_trials       : int — number of Optuna trials (50 per plan, 100 if time allows)
    random_state   : int — for TPE sampler seed

    Returns
    -------
    best_params : dict — optimised ODE parameters (a1..c3)
    study       : optuna.Study — full study object for diagnostics/plots
    """
    print(f"\n[Phase 2] Starting Optuna study: {n_trials} trials, TPE sampler, "
          f"MedianPruner, seed={random_state}")
    print(f"[Phase 2] Nominal baseline AUC will be printed for reference.\n")

    # ── Nominal params baseline (so we can compare before/after) ─────────────
    nominal_auc = _cv_auc_hybrid(
        NOMINAL_PARAMS, L, H, S, X_baseline_df, y
    )
    print(f"[Phase 2] Nominal params CV AUC (hybrid, RF): {nominal_auc:.4f}\n")

    # ── Objective closure ─────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            key: trial.suggest_float(key, lo, hi)
            for key, (lo, hi) in PHASE2_PARAM_BOUNDS.items()
        }
        return _cv_auc_hybrid(params, L, H, S, X_baseline_df, y)

    # ── Create and run study ──────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Seed first trial with nominal params so TPE has a warm reference point
    study.enqueue_trial(
        {k: v for k, v in NOMINAL_PARAMS.items()}
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_auc    = study.best_value

    print(f"\n[Phase 2] Optuna complete.")
    print(f"  Nominal  AUC : {nominal_auc:.4f}")
    print(f"  Optimised AUC: {best_auc:.4f}  "
          f"({'↑ improved' if best_auc > nominal_auc else '↓ no improvement'})")
    print(f"\n[Phase 2] Best parameters found:")
    for key in PHASE2_PARAM_BOUNDS:
        nominal_val = NOMINAL_PARAMS[key]
        opt_val     = best_params[key]
        delta       = opt_val - nominal_val
        print(f"  {key}: {nominal_val:.3f}  →  {opt_val:.4f}  "
              f"(Δ {'+' if delta >= 0 else ''}{delta:.4f})")

    return best_params, study


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def check_params_changed(best_params: dict,
                         tol: float = 0.01) -> bool:
    """Return True if optimised params differ meaningfully from nominal."""
    diffs = {k: abs(best_params[k] - NOMINAL_PARAMS[k])
             for k in NOMINAL_PARAMS}
    max_diff = max(diffs.values())
    if max_diff < tol:
        print(f"[WARN] Optimised params near-identical to nominal "
              f"(max_diff={max_diff:.4f} < tol={tol}). "
              "Optuna may not have converged — try more trials.")
        return False
    print(f"[OK] Params changed (max_diff={max_diff:.4f} > tol={tol}).")
    return True


def compare_biomarker_stds(biomarkers_p1: dict,
                           biomarkers_p2: dict) -> pd.DataFrame:
    """Print and return a DataFrame comparing Phase 1 vs Phase 2 biomarker stds."""
    df1 = pd.DataFrame(biomarkers_p1).std().rename("std_phase1")
    df2 = pd.DataFrame(biomarkers_p2).std().rename("std_phase2")
    comparison = pd.concat([df1, df2], axis=1)
    comparison["delta"] = comparison["std_phase2"] - comparison["std_phase1"]
    comparison["improved"] = comparison["delta"] > 0
    print("\n[Phase 2] Biomarker std comparison (Phase 1 → Phase 2):")
    print(comparison.to_string())
    return comparison
