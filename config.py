"""
config.py
Paper mapping: Sections II.E, II.F, IV.C, IV.D
Stores nominal parameters, file paths, column mappings, and reproducibility settings.

Updated for Merged_Aneurysm.csv (103 patients, 44R / 59U).
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join("data")
RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Nominal ODE Parameters (Paper Eq. 5-7) ───────────────────────────────────
# Phase 1: fixed nominal values from the paper.
# Phase 2: these will be Bayesian-optimised over param_space.
NOMINAL_PARAMS = {
    # dr/dt = a1*(S - 0.5) + a2*i - a3*c
    "a1": 0.8,  "a2": 0.9,  "a3": 0.7,
    # dc/dt = b1*(1 - S)   - b2*i*c - b3*c
    "b1": 0.6,  "b2": 0.8,  "b3": 0.2,
    # di/dt = c1*L          + c2*H   - c3*i
    "c1": 1.0,  "c2": 0.7,  "c3": 0.9,
}

# ── Surrogate Weights (data-derived from Merged_Aneurysm.csv) ─────────────────
# S = S_ar * AR_n + S_sr * SR_n
#   Derived via logistic regression on this cohort:
#   SR (r=+0.196, p=0.048) dominates; AR (r=+0.046, p=0.642) near-zero.
# L = MinMax(tortuosity_n / (minRadius_n + 1e-8))
#   Proxy for LSA (unavailable); r=−0.201, p=0.042 with rupture.
# H = MinMax(maxCurvature_n * tortuosity_n)
#   Proxy for WSS×OSI; r=−0.265, p=0.007 with rupture.
SURROGATE_WEIGHTS = {
    "S_ar": 0.025,   # logistic-regression weight for AR_n
    "S_sr": 0.975,   # logistic-regression weight for SR_n
    # L and H are computed entirely in preprocessing.py (no scalar weights needed)
}

# ── Column Mapping for Merged_Aneurysm.csv ────────────────────────────────────
COLUMN_MAP = {
    "morph": {
        "AR":                   "aspectRatio_star",
        "SR":                   "sizeRatio_star",
        "ostiumShapeFactor":    "ostiumShapeFactor",
        "sacVolume":            "sacVolume",
        "ellipsoidMinSemiaxis": "ellipsoidMinSemiaxis",
    },
    "vessel": {
        "minRadius":    "minRadius",
        "vesselDiameter": "vesselDiameter",
        "meanRadius":   "meanRadius",
        "maxRadius":    "maxRadius",
        "tortuosity":   "tortuosity",
        "maxCurvature": "maxCurvature",
        "meanCurvature": "meanCurvature",
        "length":       "length",
    },
    "clinical": {
        "RUPTURE":  "ruptureStatus",   # encode: R=1, U=0
        "age":      "age",
        "sex":      "sex",
        "location": "aneurysmLocation",
        "type":     "aneurysmType",
    },
}

# ── Baseline Feature Columns (Set 1) ─────────────────────────────────────────
# These 13 geometry/morphological columns form the baseline feature set.
# ellipsoidMinSemiaxis has 2 missing rows → imputed with median in preprocessing.
BASELINE_FEATURES = [
    "minRadius",
    "vesselDiameter",
    "meanRadius",
    "maxRadius",
    "tortuosity",
    "length",
    "maxCurvature",
    "meanCurvature",
    "sizeRatio_star",
    "aspectRatio_star",
    "ostiumShapeFactor",
    "ellipsoidMinSemiaxis",
    "sacVolume",
]

# ── CFD Columns to Drop (95/103 missing — not usable) ─────────────────────────
CFD_DROP_COLS = [
    "sacMinPressure", "sacMaxPressure", "sacMeanPressure",
    "sacMaxSpeed",    "sacMeanSpeed",
    "sacMinTAWSS",    "sacMaxTAWSS",    "sacMeanTAWSS",
    "sacMinOSI",      "sacMaxOSI",      "sacMeanOSI",
    "minPressure",    "maxPressure",    "meanPressure",
    "maxSpeed",       "meanSpeed",
    "minTAWSS",       "maxTAWSS",       "meanTAWSS",
    "minOSI",         "maxOSI",         "meanOSI",
]

# ── ID / Duplicate Columns to Drop ───────────────────────────────────────────
ID_DROP_COLS = ["case_id", "patient_id", "vesselName"]