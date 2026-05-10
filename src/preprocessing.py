"""
preprocessing.py
Paper mapping: Sections II.C, IV.A

Revised for Merged_Aneurysm.csv:
- Imputes 2 missing ellipsoidMinSemiaxis rows with column median
- Scales baseline feature set with MinMaxScaler
- Computes REVISED surrogates L, H, S from geometry proxies:
    S = 0.025 * AR_n  + 0.975 * SR_n          (data-derived weights)
    L = MinMax(tortuosity_n / (minRadius_n + ε)) (LSA proxy, r=−0.201)
    H = MinMax(maxCurvature_n * tortuosity_n)    (WSS×OSI proxy, r=−0.265)
- Returns X_baseline_df (scaled, shape [103, 13]), L, H, S arrays, y
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from config import SURROGATE_WEIGHTS, BASELINE_FEATURES


# ── Helper ────────────────────────────────────────────────────────────────────
def _minmax_1d(arr: np.ndarray) -> np.ndarray:
    """MinMax-scale a 1-D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def preprocess_features(df: pd.DataFrame):
    """Impute, scale, and compute geometry-derived surrogates.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataframe from data_loader.load_data() (includes 'y').

    Returns
    -------
    X_baseline_df : pd.DataFrame  — shape (103, 13), MinMax-scaled baseline features
    L             : np.ndarray    — Low-shear damage proxy  [0, 1]
    H             : np.ndarray    — Haemodynamic stress proxy [0, 1]
    S             : np.ndarray    — Shape stress surrogate   [0, 1]
    y             : np.ndarray    — Binary rupture labels
    """
    y = df["y"].values if "y" in df.columns else None

    # ── 1. Surrogate input columns ────────────────────────────────────────────
    # These 5 columns drive L, H, S.  All must be present in Merged_Aneurysm.csv.
    surrogate_cols = [
        "aspectRatio_star",   # AR
        "sizeRatio_star",     # SR
        "minRadius",          # L proxy input
        "tortuosity",         # L proxy + H proxy input
        "maxCurvature",       # H proxy input
    ]
    # No missing values in these columns — direct scaling
    X_surr_norm = MinMaxScaler().fit_transform(df[surrogate_cols].values)
    AR_n, SR_n, minR_n, tort_n, curv_n = X_surr_norm.T

    # ── 2. Compute revised surrogates (Section 3.2 of plan) ──────────────────

    # S — Shape Stress (data-derived weights from logistic regression on cohort)
    S = SURROGATE_WEIGHTS["S_ar"] * AR_n + SURROGATE_WEIGHTS["S_sr"] * SR_n

    # L — Low Shear Damage (geometry proxy for LSA; r=−0.201, p=0.042)
    L_raw = tort_n / (minR_n + 1e-8)
    L     = _minmax_1d(L_raw)

    # H — Haemodynamic Stress (geometry proxy for WSS×OSI; r=−0.265, p=0.007)
    H_raw = curv_n * tort_n
    H     = _minmax_1d(H_raw)

    print(f"[OK] Surrogates computed. "
          f"S std={S.std():.4f}, L std={L.std():.4f}, H std={H.std():.4f}")

    # Sanity check — warn if any surrogate has near-zero variance
    for name, arr in [("S", S), ("L", L), ("H", H)]:
        if arr.std() < 0.05:
            print(f"[WARN] Surrogate {name} has std={arr.std():.4f} < 0.05 "
                  "— near-constant, ODE outputs may be uninformative.")

    # ── 3. Baseline feature set (Set 1) ──────────────────────────────────────
    # Only keep columns that exist in the dataframe
    baseline_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    missing_from_plan = set(BASELINE_FEATURES) - set(baseline_cols)
    if missing_from_plan:
        print(f"[WARN] These baseline features are absent in the dataframe: "
              f"{missing_from_plan}")

    X_bl_raw = df[baseline_cols].copy()

    # Impute 2 missing ellipsoidMinSemiaxis rows with median
    imputer  = SimpleImputer(strategy="median")
    X_bl_imp = imputer.fit_transform(X_bl_raw)

    # Scale to [0, 1]
    bl_scaler   = MinMaxScaler()
    X_bl_scaled = bl_scaler.fit_transform(X_bl_imp)
    X_baseline_df = pd.DataFrame(X_bl_scaled, columns=baseline_cols)

    print(f"[OK] Baseline feature set scaled: {X_baseline_df.shape} "
          f"(13 features, MinMaxScaled).")

    return X_baseline_df, L, H, S, y


if __name__ == "__main__":
    from src.data_loader import load_data
    df, y = load_data()
    X_bl, L, H, S, y = preprocess_features(df)
    print(X_bl.shape, L.shape, H.shape, S.shape)