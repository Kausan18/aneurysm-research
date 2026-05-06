"""
config.py
Paper mapping: Sections II.E, II.F, IV.C, IV.D
Stores nominal parameters, file paths, column mappings, and reproducibility settings.
"""

import os

# 📂 Paths
DATA_DIR = os.path.join("data")
RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 🌱 Reproducibility
RANDOM_SEED = 42

# 📏 Nominal ODE Parameters (Paper Eq. 11-13)
NOMINAL_PARAMS = {
    # Growth tendency dr/dt
    "a1": 0.8, "a2": 0.9, "a3": 0.7,
    # Wall reserve dc/dt
    "b1": 0.6, "b2": 0.8, "b3": 0.2,
    # Inflammation di/dt
    "c1": 1.0, "c2": 0.7, "c3": 0.9
}

# 📐 Damage & Stress Surrogate Weights (Paper Eq. 2-4)
SURROGATE_WEIGHTS = {
    "L_wss": 0.6, "L_lsa": 0.4,  # L = 0.6*(1-WSS_n) + 0.4*LSA_n
    "S_ar": 0.5, "S_sr": 0.5      # S = 0.5*AR_n + 0.5*SR_n
}

# 🏷️ Column Mapping (clean names for internal use)
COLUMN_MAP = {
    "morph": {
        "AR": "The ratio of H to NW(AR)",
        "SR": "The ratio of H to the average of D1, D2, and D3.(SR)"
    },
    "hemo": {
        "WSS": "Mean WSS[Pa]",
        "OSI": "Oscillatory Shear\nIndex（OSI）"
    },
    "clinical": {
        "ID": "number",
        "HAS_ANEURYSM": "Has aneurysm",
        "RUPTURE": "Rupture"
    }
}