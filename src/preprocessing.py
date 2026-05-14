"""
preprocessing.py
Paper mapping: Sections II.C, IV.A

Revised for Merged_Aneurysm.csv with WSS integration:
- Imputes 2 missing ellipsoidMinSemiaxis rows with column median
- Scales baseline feature set with MinMaxScaler
- Computes surrogates L, H, S:
    S = 0.5 * AR_n  + 0.5 * SR_n              (Phase-1 equal weights, Paper Eq. 4)
    L = MinMax(tortuosity_n / (minRadius_n + e)) (LSA proxy, r=-0.201)
    H = MinMax(WSS_mean)                         (actual WSS data, replaces geometry proxy)

CHANGE: H is now directly MinMax-normalised WSS_mean from Aneurysm_WSS_values_clean.csv
        (approximated WSS for all 103 patients), replacing the previous proxy
        H = MinMax(maxCurvature_n * tortuosity_n).

CHANGE: now also returns AR_n and SR_n (the individually MinMax-normalised
aspectRatio_star and sizeRatio_star arrays) so that ode_model.simulate_and_extract()
can compute r0 = 0.3*SR_n + 0.7*AR_n per Paper Eq. 8.

Return signature: X_baseline_df, L, H, S, y, AR_n, SR_n
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from config import SURROGATE_WEIGHTS, BASELINE_FEATURES


def _minmax_1d(arr: np.ndarray) -> np.ndarray:
    """MinMax-scale a 1-D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def preprocess_features(df: pd.DataFrame):
    """Impute, scale, and compute geometry-derived surrogates and WSS-based H.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataframe from data_loader.load_data() (includes 'y' and 'WSS_mean').

    Returns
    -------
    X_baseline_df : pd.DataFrame  -- shape (103, 14), MinMax-scaled baseline features (including WSS_mean)
    L             : np.ndarray    -- Low-shear damage proxy  [0, 1]
    H             : np.ndarray    -- Haemodynamic stress driver [0, 1] (MinMax-normalised WSS_mean)
    S             : np.ndarray    -- Shape stress surrogate   [0, 1]
    y             : np.ndarray    -- Binary rupture labels
    AR_n          : np.ndarray    -- MinMax-normalised aspectRatio_star (for Eq. 8)
    SR_n          : np.ndarray    -- MinMax-normalised sizeRatio_star   (for Eq. 8)
    """
    y = df["y"].values if "y" in df.columns else None

    # Surrogate input columns (for S; L and H now use WSS_mean directly)
    surrogate_cols = [
        "aspectRatio_star",   # AR
        "sizeRatio_star",     # SR
    ]
    X_surr_norm = MinMaxScaler().fit_transform(df[surrogate_cols].values)
    AR_n, SR_n = X_surr_norm.T

    # S -- Shape Stress (data-derived weights from logistic regression on cohort)
    S = SURROGATE_WEIGHTS["S_ar"] * AR_n + SURROGATE_WEIGHTS["S_sr"] * SR_n

    # L -- Low Shear Damage (now from real WSS: low WSS → high damage risk)
    WSS_n = _minmax_1d(df["WSS_mean"].values)  # Normalize WSS to [0, 1]
    L = _minmax_1d(1.0 - WSS_n)  # Low-shear damage: (1 - WSS_n) from Aneurysm_WSS_values_clean.csv

    # H -- Haemodynamic Stress driver (high-shear damage, from actual WSS_mean, MinMax-normalised)
    H = WSS_n

    print(f"[OK] Surrogates computed. "
          f"S std={S.std():.4f}, L std={L.std():.4f}, H std={H.std():.4f}")
    print(f"[OK] L is now MinMaxNorm(1 - WSS_n) -- low-shear damage from real WSS")
    print(f"[OK] H is MinMaxNorm(WSS_n) -- high-shear damage from Aneurysm_WSS_values_clean.csv")
    print(f"[OK] AR_n std={AR_n.std():.4f}, SR_n std={SR_n.std():.4f}  "
          f"(returned for r0 = 0.3*SR_n + 0.7*AR_n per Paper Eq. 8)")

    for name, arr in [("S", S), ("L", L), ("H", H)]:
        if arr.std() < 0.05:
            print(f"[WARN] Surrogate {name} has std={arr.std():.4f} < 0.05 "
                  "-- near-constant, ODE outputs may be uninformative.")

    # Baseline feature set (Set 1)
    baseline_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    missing_from_plan = set(BASELINE_FEATURES) - set(baseline_cols)
    if missing_from_plan:
        print(f"[WARN] These baseline features are absent: {missing_from_plan}")

    X_bl_raw  = df[baseline_cols].copy()
    imputer   = SimpleImputer(strategy="median")
    X_bl_imp  = imputer.fit_transform(X_bl_raw)
    bl_scaler = MinMaxScaler()
    X_bl_scaled   = bl_scaler.fit_transform(X_bl_imp)
    X_baseline_df = pd.DataFrame(X_bl_scaled, columns=baseline_cols)

    print(f"[OK] Baseline feature set scaled: {X_baseline_df.shape} "
          f"(14 features including WSS_mean, MinMaxScaled).")

    return X_baseline_df, L, H, S, y, AR_n, SR_n


if __name__ == "__main__":
    from src.data_loader import load_data
    df, y = load_data()
    X_bl, L, H, S, y, AR_n, SR_n = preprocess_features(df)
    print(X_bl.shape, L.shape, H.shape, S.shape, AR_n.shape, SR_n.shape)