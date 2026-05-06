"""
features.py
Paper mapping: Section IV.C
Constructs three feature sets: baseline-only, ODE-only, hybrid.
"""

import pandas as pd
import numpy as np

def build_feature_sets(X_norm, biomarkers):
    """Return baseline, ODE-only, and hybrid DataFrames."""
    # Baseline: normalized AR, SR, WSS, OSI
    X_baseline = pd.DataFrame(X_norm, columns=["AR_n", "SR_n", "WSS_n", "OSI_n"])
    
    # ODE-only: 8 biomarkers
    X_ode = pd.DataFrame(biomarkers)
    
    # Hybrid: concatenation
    X_hybrid = pd.concat([X_baseline, X_ode], axis=1)
    
    print("[OK] Feature sets constructed: baseline, ODE-only, hybrid.")
    return X_baseline, X_ode, X_hybrid