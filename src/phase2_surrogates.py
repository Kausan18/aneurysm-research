"""
phase2_surrogates.py
Phase 2 contribution — Section 4.2 of Plan of Action.

Expands the single-term L and H surrogates to weighted multi-term combinations,
then learns the optimal weights via LogisticRegression (same approach used to
derive S weights in Phase 1).

  L_expanded = w1*tortuosity_n + w2*(1-WSS_n) + w3*length_n
               (tortuosity carries flow disturbance; (1-WSS_n) represents low-shear
               damage from Aneurysm_WSS_values_clean.csv; length captures vessel complexity)

  H_expanded = w1*maxCurvature_n + w2*tortuosity_n + w3*meanCurvature_n
               (maxCurvature is the primary WSS×OSI proxy; meanCurvature adds
               the global curvature signal; tortuosity provides flow disturbance)

Weights derived by fitting LogisticRegression(class_weight='balanced') on the
expanded feature set against rupture status — same methodology as S derivation.
Weights are then softmax-normalised to [0,1] before computing the surrogate so
the final L_refined, H_refined remain in [0,1] via MinMax rescaling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def _minmax_1d(arr: np.ndarray) -> np.ndarray:
    """MinMax-scale a 1-D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _fit_lr_weights(X_candidates: np.ndarray,
                    y: np.ndarray,
                    random_state: int = 42) -> np.ndarray:
    """Fit a balanced LogisticRegression and return abs(coef) normalised to sum=1.

    Parameters
    ----------
    X_candidates : np.ndarray — shape (n, k), candidate columns already in [0,1]
    y            : np.ndarray — binary rupture labels

    Returns
    -------
    np.ndarray — shape (k,), non-negative weights summing to 1.0
    """
    lr = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    lr.fit(X_candidates, y)
    raw_coefs = np.abs(lr.coef_[0])          # absolute logistic coefs
    weights   = raw_coefs / (raw_coefs.sum() + 1e-12)  # normalise to sum=1
    return weights


def refine_surrogate_weights(df: pd.DataFrame,
                             y: np.ndarray,
                             random_state: int = 42) -> tuple:
    """Learn refined multi-term weights for L and H surrogates.

    Uses logistic regression on each expanded surrogate candidate set against
    rupture status (same method used for S in Phase 1).

    Parameters
    ----------
    df           : pd.DataFrame — raw dataframe from data_loader.load_data()
    y            : np.ndarray   — binary rupture labels (1=ruptured, 0=unruptured)
    random_state : int

    Returns
    -------
    L_refined : np.ndarray — shape (n,), improved low-shear proxy in [0, 1]
    H_refined : np.ndarray — shape (n,), improved haemodynamic stress proxy in [0,1]
    L_weights : dict — {term: weight} for reporting / paper
    H_weights : dict — {term: weight} for reporting / paper
    """
    # ── Columns used for surrogate expansion ─────────────────────────────────
    # All must exist in Merged_Aneurysm.csv — verified against BASELINE_FEATURES
    needed = ["tortuosity", "WSS_mean", "length", "maxCurvature", "meanCurvature"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"[phase2_surrogates] Missing columns in df: {missing}")

    # ── MinMax-normalise each raw column independently ────────────────────────
    scaler   = MinMaxScaler()
    norm_arr = scaler.fit_transform(df[needed].values)

    tort_n  = norm_arr[:, 0]   # tortuosity      (normalised)
    wss_n   = norm_arr[:, 1]   # WSS_mean        (normalised)
    len_n   = norm_arr[:, 2]   # length          (normalised)
    maxC_n  = norm_arr[:, 3]   # maxCurvature    (normalised)
    meanC_n = norm_arr[:, 4]   # meanCurvature   (normalised)

    # Low-shear damage factor: (1 - WSS_n) means low WSS → high damage risk
    low_wss_n = 1.0 - wss_n

    # ── L expansion: {tortuosity, (1-WSS), length} ──────────────────────────
    # Phase 1 L = minmax(tortuosity / minRadius), a geometric low-shear proxy.
    # Now using real WSS from Aneurysm_WSS_values_clean.csv: (1 - WSS_n) for low-shear damage.
    # Additive weighted form allows data to balance the contributions.
    L_candidates = np.column_stack([tort_n, low_wss_n, len_n])
    L_w = _fit_lr_weights(L_candidates, y, random_state=random_state)

    L_raw     = L_w[0] * tort_n + L_w[1] * low_wss_n + L_w[2] * len_n
    L_refined = _minmax_1d(L_raw)

    L_weights = {
        "tortuosity":   float(L_w[0]),
        "low_shear_WSS": float(L_w[1]),
        "length":       float(L_w[2]),
    }

    # ── H expansion: {maxCurvature, tortuosity, meanCurvature} ───────────────
    # Phase 1 H = minmax(maxCurvature × tortuosity).
    # Additive form separates local curvature peaks from global flow disturbance.
    H_candidates = np.column_stack([maxC_n, tort_n, meanC_n])
    H_w = _fit_lr_weights(H_candidates, y, random_state=random_state)

    H_raw     = H_w[0] * maxC_n + H_w[1] * tort_n + H_w[2] * meanC_n
    H_refined = _minmax_1d(H_raw)

    H_weights = {
        "maxCurvature":  float(H_w[0]),
        "tortuosity":    float(H_w[1]),
        "meanCurvature": float(H_w[2]),
    }

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[Phase 2 Surrogates] Refined L weights (LR-derived):")
    for term, w in L_weights.items():
        print(f"  {term}: {w:.4f}")
    print(f"  L_refined std = {L_refined.std():.4f}  "
          f"{'OK' if L_refined.std() > 0.05 else '<< LOW'}")

    print("\n[Phase 2 Surrogates] Refined H weights (LR-derived):")
    for term, w in H_weights.items():
        print(f"  {term}: {w:.4f}")
    print(f"  H_refined std = {H_refined.std():.4f}  "
          f"{'OK' if H_refined.std() > 0.05 else '<< LOW'}")

    # Surrogate correlation with rupture status (informational)
    for name, arr in [("L_refined", L_refined), ("H_refined", H_refined)]:
        from scipy.stats import pearsonr
        r, p = pearsonr(arr, y)
        print(f"  {name}: r={r:.3f}, p={p:.3f} vs rupture status")

    return L_refined, H_refined, L_weights, H_weights
