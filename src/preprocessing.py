import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from config import SURROGATE_WEIGHTS, COLUMN_MAP
 
 
def preprocess_features(df):
    """Return normalized features and computed surrogates L, H, S."""
    clean_col = lambda c: (
        c.strip()
        .replace("\n", "_").replace(" ", "_")
        .replace("(", "").replace(")", "")
        .replace("（", "").replace("）", "")
        .replace("[", "").replace("]", "")
        .replace(",", "").replace(".", "_")
    )
 
    AR_col   = clean_col(COLUMN_MAP["morph"]["AR"])
    SR_col   = clean_col(COLUMN_MAP["morph"]["SR"])
    WSS_col  = clean_col(COLUMN_MAP["hemo"]["WSS"])
    OSI_col  = clean_col(COLUMN_MAP["hemo"]["OSI"])
 
    # ── NEW: columns needed for LSA surrogate ──────────────────────────────
    MIN_WSS_col  = clean_col(COLUMN_MAP["hemo"]["MIN_WSS"])   # add to COLUMN_MAP
    MEAN_WSS_col = clean_col(COLUMN_MAP["hemo"]["MEAN_WSS"])  # add to COLUMN_MAP
    MAX_WSS_col  = clean_col(COLUMN_MAP["hemo"]["MAX_WSS"])   # add to COLUMN_MAP
    # ───────────────────────────────────────────────────────────────────────
 
    feat_cols = [AR_col, SR_col, WSS_col, OSI_col,
                 MIN_WSS_col, MAX_WSS_col]
 
    X_raw = df[feat_cols].copy()
    y     = df[clean_col(COLUMN_MAP["clinical"]["RUPTURE"])].values
 
    # 1. Impute missing with median
    imputer  = SimpleImputer(strategy="median")
    X_imp    = imputer.fit_transform(X_raw)
    X_imp_df = pd.DataFrame(X_imp, columns=feat_cols)
 
    # 2. Pull WSS family before global normalisation (raw scale needed for LSA)
    min_wss_raw  = X_imp_df[MIN_WSS_col].values
    mean_wss_raw = X_imp_df[WSS_col].values  # WSS is already Mean WSS, so use it directly
    max_wss_raw  = X_imp_df[MAX_WSS_col].values
 
    # ── BUG 1 FIX + OPT: LSA surrogate selection ──────────────────────────
    # Candidate A — range-normalised version (optimisation choice)
    lsa_A = (mean_wss_raw - min_wss_raw) / (max_wss_raw - min_wss_raw + 1e-8)
 
    # Candidate B — subtraction version (original fix)
    lsa_B = 1.0 - (min_wss_raw / (mean_wss_raw + 1e-8))
    lsa_B = np.clip(lsa_B, 0, None)          # keep non-negative
 
    # MinMax-scale both to [0,1] so std is comparable
    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)
 
    lsa_A_scaled = _minmax(lsa_A)
    lsa_B_scaled = _minmax(lsa_B)
 
    if lsa_A_scaled.std() >= lsa_B_scaled.std():
        LSA_n = lsa_A_scaled
        print(f"[OK] LSA: range-normalised version selected "
              f"(std={lsa_A_scaled.std():.4f} vs {lsa_B_scaled.std():.4f})")
    else:
        LSA_n = lsa_B_scaled
        print(f"[OK] LSA: subtraction version selected "
              f"(std={lsa_B_scaled.std():.4f} vs {lsa_A_scaled.std():.4f})")
    # ───────────────────────────────────────────────────────────────────────
 
    # 3. Normalise the 4 model features to [0,1]  (Paper Eq. 1)
    model_feat_cols = [AR_col, SR_col, WSS_col, OSI_col]
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_norm  = scaler.fit_transform(X_imp_df[model_feat_cols])
 
    AR_n, SR_n, WSS_n, OSI_n = X_norm.T
 
    # 4. Compute surrogates  (Paper Eq. 2-4)
    L = SURROGATE_WEIGHTS["L_wss"] * (1 - WSS_n) + SURROGATE_WEIGHTS["L_lsa"] * LSA_n
    H = WSS_n * OSI_n
    S = SURROGATE_WEIGHTS["S_ar"] * AR_n + SURROGATE_WEIGHTS["S_sr"] * SR_n
 
    print("[OK] Preprocessing complete. Surrogates L, H, S computed.")
    return X_norm, L, H, S, y, scaler