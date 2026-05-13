"""
phase2_optuna.py
Phase 2 contribution -- Section 3 of Plan of Action.

Bayesian search over the 9 ODE parameters (a1-c3) using Optuna TPE sampler.
Objective: maximise mean CV AUC on the HYBRID feature set with Random Forest.

FIX: _cv_auc_hybrid() and run_optuna_study() now accept AR_n and SR_n and
forward them to simulate_and_extract() so that r0 = 0.3*SR_n + 0.7*AR_n
per Paper Eq. 8. Previously these calls were missing AR_n/SR_n, causing
the ODE to use the wrong S_REF population anchor for r0 in every trial.

Design decisions:
  - RF is the objective model (most sensitive to biomarker quality).
  - RepeatedStratifiedKFold(5x3) kept consistent with Phase 1 evaluation.
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_auc_hybrid(params, L, H, S, X_baseline_df, y,
                   AR_n=None, SR_n=None,
                   n_splits=5, n_repeats=3, random_state=42):
    """Compute mean CV AUC on the hybrid feature set for a given param dict.

    Parameters
    ----------
    params        : dict -- ODE coefficient dict (keys a1..c3)
    L, H, S       : np.ndarray -- geometry surrogates, shape (n_patients,)
    X_baseline_df : pd.DataFrame -- MinMax-scaled baseline features (n, 13)
    y             : np.ndarray -- binary rupture labels
    AR_n, SR_n    : np.ndarray -- normalised AR and SR for r0 per Eq. 8
    n_splits, n_repeats : RepeatedStratifiedKFold configuration

    Returns
    -------
    float -- mean AUC across all folds (higher = better)
    """
    # Recompute ODE biomarkers with trial params and correct r0
    biomarkers = simulate_and_extract(
        L, H, S,
        params=params,
        AR_n=AR_n,
        SR_n=SR_n,
    )

    _, _, X_hybrid = build_feature_sets(X_baseline_df, biomarkers)
    X_np = X_hybrid.values

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=50,
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
        fold_aucs.append(roc_auc_score(y_te, y_prob))

    return float(np.mean(fold_aucs))


def run_optuna_study(L, H, S, X_baseline_df, y,
                     n_trials=50, random_state=42,
                     AR_n=None, SR_n=None):
    """Run the Bayesian ODE coefficient optimisation study.

    Parameters
    ----------
    L, H, S        : np.ndarray -- geometry surrogates from preprocess_features()
    X_baseline_df  : pd.DataFrame -- MinMax-scaled baseline features (n, 13)
    y              : np.ndarray -- binary rupture labels
    n_trials       : int -- number of Optuna trials
    random_state   : int -- for TPE sampler seed
    AR_n, SR_n     : np.ndarray -- normalised AR and SR for r0 per Eq. 8

    Returns
    -------
    best_params : dict -- optimised ODE parameters (a1..c3)
    study       : optuna.Study -- full study object
    """
    print(f"\n[Phase 2] Starting Optuna study: {n_trials} trials, TPE sampler, "
          f"MedianPruner, seed={random_state}")

    nominal_auc = _cv_auc_hybrid(
        NOMINAL_PARAMS, L, H, S, X_baseline_df, y,
        AR_n=AR_n, SR_n=SR_n,
    )
    print(f"[Phase 2] Nominal params CV AUC (hybrid, RF): {nominal_auc:.4f}\n")

    def objective(trial):
        params = {
            key: trial.suggest_float(key, lo, hi)
            for key, (lo, hi) in PHASE2_PARAM_BOUNDS.items()
        }
        return _cv_auc_hybrid(
            params, L, H, S, X_baseline_df, y,
            AR_n=AR_n, SR_n=SR_n,
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Seed first trial with nominal params
    study.enqueue_trial({k: v for k, v in NOMINAL_PARAMS.items()})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_auc    = study.best_value

    print(f"\n[Phase 2] Optuna complete.")
    print(f"  Nominal  AUC : {nominal_auc:.4f}")
    print(f"  Optimised AUC: {best_auc:.4f}  "
          f"({'up improved' if best_auc > nominal_auc else 'down no improvement'})")
    print(f"\n[Phase 2] Best parameters found:")
    for key in PHASE2_PARAM_BOUNDS:
        nominal_val = NOMINAL_PARAMS[key]
        opt_val     = best_params[key]
        delta       = opt_val - nominal_val
        print(f"  {key}: {nominal_val:.3f}  ->  {opt_val:.4f}  "
              f"(D {'+' if delta >= 0 else ''}{delta:.4f})")

    return best_params, study


def check_params_changed(best_params, tol=0.01):
    """Return True if optimised params differ meaningfully from nominal."""
    diffs    = {k: abs(best_params[k] - NOMINAL_PARAMS[k]) for k in NOMINAL_PARAMS}
    max_diff = max(diffs.values())
    if max_diff < tol:
        print(f"[WARN] Optimised params near-identical to nominal "
              f"(max_diff={max_diff:.4f} < tol={tol}). "
              "Try more trials.")
        return False
    print(f"[OK] Params changed (max_diff={max_diff:.4f} > tol={tol}).")
    return True


def compare_biomarker_stds(biomarkers_p1, biomarkers_p2):
    """Print and return a DataFrame comparing Phase 1 vs Phase 2 biomarker stds."""
    df1 = pd.DataFrame(biomarkers_p1).std().rename("std_phase1")
    df2 = pd.DataFrame(biomarkers_p2).std().rename("std_phase2")
    comparison = pd.concat([df1, df2], axis=1)
    comparison["delta"]    = comparison["std_phase2"] - comparison["std_phase1"]
    comparison["improved"] = comparison["delta"] > 0
    print("\n[Phase 2] Biomarker std comparison (Phase 1 -> Phase 2):")
    print(comparison.to_string())
    return comparison