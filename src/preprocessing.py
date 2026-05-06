"""
preprocessing.py
Paper mapping: Section IV.A, II.C
Imputes missing values, normalizes to [0,1], computes damage/stress surrogates L, H, S.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from config import SURROGATE_WEIGHTS, COLUMN_MAP

def preprocess_features(df):
    """Return normalized features and computed surrogates L, H, S."""
    # Build feature column names from COLUMN_MAP using same cleaning as data_loader
    clean_col = lambda c: c.strip().replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace("[", "").replace("]", "").replace(",", "").replace(".", "_")
    
    AR_col = clean_col(COLUMN_MAP["morph"]["AR"])
    SR_col = clean_col(COLUMN_MAP["morph"]["SR"])
    WSS_col = clean_col(COLUMN_MAP["hemo"]["WSS"])
    OSI_col = clean_col(COLUMN_MAP["hemo"]["OSI"])
    
    feat_cols = [AR_col, SR_col, WSS_col, OSI_col]
    
    X = df[feat_cols].copy()
    y = df[clean_col(COLUMN_MAP["clinical"]["RUPTURE"])].values
    
    # 1. Impute missing with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    # 2. Normalize to [0, 1] (Paper Eq. 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_norm = scaler.fit_transform(X_imputed)
    
    AR_n, SR_n, WSS_n, OSI_n = X_norm.T
    
    # 3. Compute surrogates (Paper Eq. 2-4)
    # Note: LSA not in dataset → set LSA_n = 0 temporarily
    LSA_n = np.zeros_like(WSS_n)
    L = SURROGATE_WEIGHTS["L_wss"] * (1 - WSS_n) + SURROGATE_WEIGHTS["L_lsa"] * LSA_n
    H = WSS_n * OSI_n
    S = SURROGATE_WEIGHTS["S_ar"] * AR_n + SURROGATE_WEIGHTS["S_sr"] * SR_n
    
    print("[OK] Preprocessing complete. Surrogates L, H, S computed.")
    return X_norm, L, H, S, y, scaler